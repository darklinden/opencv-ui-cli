use anyhow::{Context, Result};
use clap::Parser;
use opencv::{
    core::{self, Mat, Point, Rect, VecN},
    imgcodecs, imgproc,
    prelude::*,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "opencv-ui-cli", about = "Match UI components against a design image")]
struct Cli {
    /// Path to the design image (e.g. design.png)
    design: PathBuf,

    /// Directory containing component images to match
    components_dir: PathBuf,

    /// Output TOML file path
    #[arg(short = 'o', long, default_value = "match_result.toml")]
    output: PathBuf,

    /// Starting threshold (optimistic). Lowered stepwise until a match is found.
    #[arg(long, default_value = "0.95")]
    start_threshold: f64,

    /// Minimum threshold to try before giving up.
    #[arg(long, default_value = "0.5")]
    min_threshold: f64,

    /// Amount to lower the threshold on each retry.
    #[arg(long, default_value = "0.05")]
    threshold_step: f64,

    /// NMS IoU threshold
    #[arg(long, default_value = "0.3")]
    nms_threshold: f64,

    /// Skip generating {component}-match.png mask images
    #[arg(long)]
    no_mask: bool,
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct Position {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    confidence: f64,
    trust: String,
}

#[derive(Serialize)]
struct MatchEntry {
    component: String,
    count: usize,
    positions: Vec<Position>,
}

#[derive(Serialize)]
struct DesignInfo {
    file: String,
    width: i32,
    height: i32,
}

#[derive(Serialize)]
struct MatchResult {
    design: DesignInfo,
    matches: Vec<MatchEntry>,
}

// ---------------------------------------------------------------------------
// Image loading helpers
// ---------------------------------------------------------------------------

fn load_image(path: &Path) -> Result<Mat> {
    let img = imgcodecs::imread(
        path.to_str().context("invalid path")?,
        imgcodecs::IMREAD_COLOR,
    )?;
    if img.empty() {
        anyhow::bail!("failed to load image: {}", path.display());
    }
    Ok(img)
}

/// Load an image with alpha channel preserved.
/// Returns (bgr_pixels, alpha_mask) where alpha_mask is None for images without alpha.
fn load_with_mask(path: &Path) -> Result<(Mat, Option<Mat>)> {
    let raw = imgcodecs::imread(
        path.to_str().context("invalid path")?,
        imgcodecs::IMREAD_UNCHANGED,
    )?;
    if raw.empty() {
        anyhow::bail!("failed to load image: {}", path.display());
    }

    if raw.channels() == 4 {
        // Convert BGRA -> BGR (drops alpha)
        let mut bgr = Mat::default();
        imgproc::cvt_color(
            &raw,
            &mut bgr,
            imgproc::COLOR_BGRA2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Extract alpha channel (index 3)
        let mut alpha = Mat::default();
        core::extract_channel(&raw, &mut alpha, 3)?;

        Ok((bgr, Some(alpha)))
    } else {
        Ok((raw, None))
    }
}

fn component_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Skip mask output files from previous runs
            if name.ends_with("-match.png") {
                continue;
            }
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                match ext.to_lowercase().as_str() {
                    "png" | "jpg" | "jpeg" | "webp" | "bmp" => files.push(path),
                    _ => {}
                }
            }
        }
    }
    files.sort();
    Ok(files)
}

// ---------------------------------------------------------------------------
// Template matching with NMS
// ---------------------------------------------------------------------------

/// Compute per-channel pixel variance of the template.
/// When all channels have near-zero per-pixel variance, the template is a
/// solid uniform rectangle — TM_CCOEFF_NORMED fails on these (divide by ~0).
fn template_variance(templ: &Mat, mask: &Option<Mat>) -> f64 {
    let (h, w) = (templ.rows(), templ.cols());
    if h < 2 || w < 2 {
        return 0.0;
    }

    let mut ch_sums = [0.0f64; 3];
    let mut ch_sq_sums = [0.0f64; 3];
    let mut count = 0u64;

    for r in 0..h {
        for c in 0..w {
            // Skip transparent pixels when a mask is available
            if let Some(m) = mask.as_ref() {
                if let Ok(a) = m.at_2d::<u8>(r, c) {
                    if *a == 0 {
                        continue;
                    }
                }
            }
            if let Ok(px) = templ.at_2d::<VecN<u8, 3>>(r, c) {
                for ch in 0..3 {
                    let v = px[ch] as f64;
                    ch_sums[ch] += v;
                    ch_sq_sums[ch] += v * v;
                }
                count += 1;
            }
        }
    }

    if count < 2 {
        return 0.0;
    }
    let n = count as f64;

    let mut min_var = f64::MAX;
    for ch in 0..3 {
        let mean = ch_sums[ch] / n;
        let var = (ch_sq_sums[ch] / n) - (mean * mean);
        if var < min_var {
            min_var = var;
        }
    }
    min_var
}

/// Run matchTemplate and return all detections above the threshold after NMS.
/// Returns (confidence, rect) where confidence is normalized to 0..1 (higher = better).
fn match_template_nms(
    design: &Mat,
    templ: &Mat,
    mask: &Option<Mat>,
    threshold: f64,
    nms_threshold: f64,
) -> Result<Vec<(f64, Rect)>> {
    // Template must not be larger than the design
    if templ.rows() > design.rows() || templ.cols() > design.cols() {
        return Ok(Vec::new());
    }

    let tvar = template_variance(templ, mask);

    // For low-variance (uniform) templates, TM_CCOEFF_NORMED fails (divide by ~zero).
    // Use TM_SQDIFF_NORMED instead: perfect match = 0.0, worst match → 1.0.
    let use_sqdiff = tvar < 1.0;

    let method = if use_sqdiff {
        imgproc::TM_SQDIFF_NORMED
    } else {
        imgproc::TM_CCOEFF_NORMED
    };

    let result_rows = design.rows() - templ.rows() + 1;
    let result_cols = design.cols() - templ.cols() + 1;
    let mut result = Mat::default();

    let mask_mat = mask.as_ref().map_or_else(Mat::default, |m| m.clone());
    imgproc::match_template(
        design,
        templ,
        &mut result,
        method,
        &mask_mat,
    )?;

    // Collect all points passing threshold
    let mut candidates: Vec<(f64, Point)> = Vec::new();
    for r in 0..result_rows {
        for c in 0..result_cols {
            let raw = *result.at_2d::<f32>(r, c)? as f64;
            let score = if use_sqdiff {
                // SQDIFF: 0 = perfect, 1 = worst → invert so higher = better
                1.0 - raw
            } else {
                raw
            };
            if score >= threshold {
                candidates.push((score, Point::new(c, r)));
            }
        }
    }

    // Sort by confidence descending
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // NMS
    let tw = templ.cols();
    let th = templ.rows();
    let mut kept: Vec<(f64, Rect)> = Vec::new();

    for (conf, pt) in candidates {
        let rect = Rect::new(pt.x, pt.y, tw, th);
        let mut suppressed = false;
        for (_, k) in &kept {
            if iou(&rect, k) > nms_threshold {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            kept.push((conf, rect));
        }
    }

    Ok(kept)
}

fn iou(a: &Rect, b: &Rect) -> f64 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);

    let inter_w = (x2 - x1).max(0);
    let inter_h = (y2 - y1).max(0);
    let inter_area = inter_w as f64 * inter_h as f64;

    let area_a = a.width as f64 * a.height as f64;
    let area_b = b.width as f64 * b.height as f64;
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

// ---------------------------------------------------------------------------
// Dynamic threshold descent matching
// ---------------------------------------------------------------------------

fn trust_label(confidence: f64) -> &'static str {
    if confidence >= 0.90 {
        "high"
    } else if confidence >= 0.75 {
        "medium"
    } else {
        "low"
    }
}

/// Try thresholds from start down to min, stepping by `step`.
/// Returns as soon as at least one match is found at the current threshold.
/// Trust is assigned per-match based on individual confidence.
fn match_component(
    design: &Mat,
    templ: &Mat,
    mask: &Option<Mat>,
    component_name: &str,
    start_threshold: f64,
    min_threshold: f64,
    step: f64,
    nms_threshold: f64,
) -> Result<(f64, Vec<Position>)> {
    let tw = templ.cols();
    let th = templ.rows();
    let mut threshold = start_threshold;

    while threshold >= min_threshold - 1e-9 {
        let matches = match_template_nms(design, templ, mask, threshold, nms_threshold)?;
        if !matches.is_empty() {
            let positions: Vec<Position> = matches
                .into_iter()
                .map(|(conf, rect)| Position {
                    x: rect.x,
                    y: rect.y,
                    width: tw,
                    height: th,
                    confidence: (conf * 10000.0).round() / 10000.0,
                    trust: trust_label(conf).into(),
                })
                .collect();
            return Ok((threshold, positions));
        }
        threshold = (threshold - step * 100.0).round() / 100.0;
    }

    anyhow::bail!("no matches found for {} (tried down to {})", component_name, min_threshold)
}

// ---------------------------------------------------------------------------
// TOML output
// ---------------------------------------------------------------------------

fn write_toml(result: &MatchResult, path: &Path) -> Result<()> {
    let toml_str = toml::to_string_pretty(result)?;
    std::fs::write(path, toml_str)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// SVG mask → PNG visualization (per component)
// ---------------------------------------------------------------------------

fn generate_mask_image(
    design: &Mat,
    positions: &[Position],
    component_name: &str,
    output_path: &Path,
) -> Result<()> {
    let dw = design.cols();
    let dh = design.rows();

    // Build SVG with semi-transparent rectangles
    let mut svg = format!(
        r#"<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">"#,
        w = dw,
        h = dh
    );

    for pos in positions {
        let (fill, stroke, text_fill) = match pos.trust.as_str() {
            "high" => (
                "rgba(0,200,0,0.25)",
                "rgb(0,180,0)",
                "rgb(0,140,0)",
            ),
            "medium" => (
                "rgba(200,200,0,0.25)",
                "rgb(180,180,0)",
                "rgb(140,140,0)",
            ),
            _ => (
                "rgba(200,0,0,0.25)",
                "rgb(180,0,0)",
                "rgb(140,0,0)",
            ),
        };

        svg.push_str(&format!(
            r#"<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>"#,
            x = pos.x,
            y = pos.y,
            w = pos.width,
            h = pos.height,
            fill = fill,
            stroke = stroke
        ));

        let label = format!(
            "{} conf:{:.2}",
            component_name,
            pos.confidence
        );
        let label_y = if pos.y >= 16 { pos.y - 4 } else { pos.y + pos.height + 14 };
        svg.push_str(&format!(
            r#"<text x="{x}" y="{y}" font-size="13" font-family="sans-serif" fill="{c}">{label}</text>"#,
            x = pos.x,
            y = label_y,
            c = text_fill,
            label = label
        ));
    }

    svg.push_str("</svg>");

    // Render SVG to pixels using resvg
    let usvg_tree =
        usvg::Tree::from_str(&svg, &usvg::Options::default()).context("failed to parse SVG")?;

    let mut pixmap =
        resvg::tiny_skia::Pixmap::new(dw as u32, dh as u32).context("failed to create pixmap")?;

    resvg::render(
        &usvg_tree,
        resvg::tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );

    // Alpha blend pixmap pixels onto a clone of the design
    let mut output = design.clone();

    let mask_data = pixmap.data();
    for r in 0..dh {
        for c in 0..dw {
            let idx = ((r as u32 * dw as u32 + c as u32) * 4) as usize;
            let mr = mask_data[idx] as f64;
            let mg = mask_data[idx + 1] as f64;
            let mb = mask_data[idx + 2] as f64;
            let ma = mask_data[idx + 3] as f64 / 255.0;

            if ma > 0.0 {
                let px = output.at_2d_mut::<VecN<u8, 3>>(r, c)?;
                // pixmap is RGBA, OpenCV is BGR — map R->R, G->G, B->B (but OpenCV stores BGR)
                // Actually pixmap RGBA: idx=R, idx+1=G, idx+2=B
                // OpenCV Vec3b: [0]=B, [1]=G, [2]=R
                let design_b = px[0] as f64;
                let design_g = px[1] as f64;
                let design_r = px[2] as f64;
                px[0] = (mb * ma + design_b * (1.0 - ma)) as u8;
                px[1] = (mg * ma + design_g * (1.0 - ma)) as u8;
                px[2] = (mr * ma + design_r * (1.0 - ma)) as u8;
            }
        }
    }

    let params = opencv::core::Vector::<i32>::new();
    imgcodecs::imwrite(
        output_path.to_str().context("invalid output path")?,
        &output,
        &params,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Validate inputs
    if !cli.design.exists() {
        anyhow::bail!("design file not found: {}", cli.design.display());
    }
    if !cli.components_dir.is_dir() {
        anyhow::bail!(
            "components directory not found: {}",
            cli.components_dir.display()
        );
    }

    // Load design
    let design = load_image(&cli.design)?;
    let design_w = design.cols();
    let design_h = design.rows();
    eprintln!(
        "Loaded design: {} ({}x{})",
        cli.design.display(),
        design_w,
        design_h
    );

    // Load component images
    let component_paths = component_files(&cli.components_dir)?;
    if component_paths.is_empty() {
        anyhow::bail!(
            "no image files found in {}",
            cli.components_dir.display()
        );
    }
    eprintln!("Found {} component(s)", component_paths.len());

    let mut all_matches: Vec<MatchEntry> = Vec::new();

    for comp_path in &component_paths {
        let comp_name = comp_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        eprintln!("  Matching {} ...", comp_name);

        let (templ, mask) = load_with_mask(comp_path)?;

        match match_component(
            &design,
            &templ,
            &mask,
            &comp_name,
            cli.start_threshold,
            cli.min_threshold,
            cli.threshold_step,
            cli.nms_threshold,
        ) {
            Ok((matched_at, positions)) => {
                let count = positions.len();
                let trusts: Vec<&str> = positions.iter().map(|p| p.trust.as_str()).collect();
                eprintln!(
                    "    {} match(es) @threshold={} [{}]",
                    count,
                    matched_at,
                    trusts.join(", ")
                );

                // Generate per-component mask image
                if !cli.no_mask {
                    let stem = comp_path.file_stem().unwrap().to_str().unwrap();
                    let mask_path = comp_path
                        .parent()
                        .unwrap_or(Path::new("."))
                        .join(format!("{}-match.png", stem));
                    if let Err(e) =
                        generate_mask_image(&design, &positions, &comp_name, &mask_path)
                    {
                        eprintln!("    warning: failed to generate mask: {}", e);
                    } else {
                        eprintln!("    mask -> {}", mask_path.display());
                    }
                }

                all_matches.push(MatchEntry {
                    component: comp_name,
                    count,
                    positions,
                });
            }
            Err(e) => {
                eprintln!("    skipped: {}", e);
            }
        }
    }

    // Output TOML
    let result = MatchResult {
        design: DesignInfo {
            file: cli
                .design
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
            width: design_w,
            height: design_h,
        },
        matches: all_matches,
    };

    write_toml(&result, &cli.output)?;
    eprintln!("Result written to {}", cli.output.display());

    Ok(())
}
