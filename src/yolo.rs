use anyhow::{Context, Result};
use opencv::{core::Rect, prelude::*};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::Position;

const YOLO_SCRIPT: &str = include_str!("../scripts/yolo_detect.py");

// ---------------------------------------------------------------------------
// YOLO JSON structures (match Python sidecar output)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
pub(crate) struct YoloDetection {
    pub(crate) class_name: String,
    pub(crate) confidence: f64,
    pub(crate) bbox: [i32; 4], // [x, y, width, height]
}

#[derive(Deserialize, Debug)]
struct YoloResult {
    detections: Vec<YoloDetection>,
    #[allow(dead_code)]
    count: usize,
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub struct YoloConfig {
    pub enabled: bool,
    #[allow(dead_code)]
    pub threshold: f64,
    pub conf: f64,
    pub epochs: u32,
    pub cache: PathBuf,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn run_yolo(
    config: &YoloConfig,
    design_path: &Path,
    components_dir: &Path,
) -> Result<Option<Vec<YoloDetection>>> {
    if !config.enabled {
        return Ok(None);
    }

    let ext_dir = match ext_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("    warning: YOLO ext dir setup failed: {}", e);
            return Ok(None);
        }
    };

    let python = match ensure_venv(&ext_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("    warning: YOLO venv setup failed: {}", e);
            return Ok(None);
        }
    };

    let script_path = ext_dir.join("yolo_detect.py");

    // Train if cache is stale
    let need_train = !config.cache.exists()
        || is_cache_stale(components_dir, &config.cache);

    if need_train {
        eprintln!("    YOLO: training model (this may take a few minutes)...");
        let status = Command::new(&python)
            .arg(&script_path)
            .arg("train")
            .arg("--components")
            .arg(components_dir)
            .arg("--output")
            .arg(&config.cache)
            .arg("--epochs")
            .arg(config.epochs.to_string())
            .status()
            .context("failed to run YOLO train")?;

        if !status.success() {
            eprintln!("    warning: YOLO training failed, keeping template results");
            return Ok(None);
        }
    }

    let output_json = ext_dir.join("yolo_result.json");

    let status = Command::new(&python)
        .arg(&script_path)
        .arg("detect")
        .arg("--model")
        .arg(&config.cache)
        .arg("--source")
        .arg(design_path)
        .arg("--output")
        .arg(&output_json)
        .arg("--conf")
        .arg(config.conf.to_string())
        .status()
        .context("failed to run YOLO detect")?;

    if !status.success() {
        eprintln!("    warning: YOLO detection failed, keeping template results");
        return Ok(None);
    }

    let json_str = std::fs::read_to_string(&output_json)
        .context("failed to read YOLO output")?;
    let result: YoloResult = serde_json::from_str(&json_str)
        .context("failed to parse YOLO output")?;

    if let Some(err) = &result.error {
        if !err.is_empty() {
            eprintln!("    warning: YOLO error: {}", err);
            return Ok(None);
        }
    }

    eprintln!("    YOLO: found {} detection(s)", result.detections.len());
    Ok(Some(result.detections))
}

// ---------------------------------------------------------------------------
// YOLO-guided localized template matching
// ---------------------------------------------------------------------------

/// Use a YOLO detection bbox as a search region for precise template matching.
/// Expands the YOLO region by `margin` ratio, crops the design, and runs
/// template matching on the crop. Returns precise pixel coordinates or `None`
/// if no match found on the crop.
pub fn refine_yolo_region(
    design: &opencv::core::Mat,
    templ: &opencv::core::Mat,
    mask: &Option<opencv::core::Mat>,
    yolo_bbox: &[i32; 4],
    margin: f64,
    start_threshold: f64,
    step: f64,
    nms_threshold: f64,
) -> Option<Position> {
    let (dw, dh) = (design.cols(), design.rows());

    // Expand the YOLO bbox by margin on all sides
    let expand_w = (yolo_bbox[2] as f64 * margin) as i32;
    let expand_h = (yolo_bbox[3] as f64 * margin) as i32;

    let crop_x = (yolo_bbox[0] - expand_w).max(0);
    let crop_y = (yolo_bbox[1] - expand_h).max(0);
    let crop_w = (yolo_bbox[2] + 2 * expand_w).min(dw - crop_x);
    let crop_h = (yolo_bbox[3] + 2 * expand_h).min(dh - crop_y);

    if crop_w < templ.cols() || crop_h < templ.rows() {
        return None;
    }

    let crop_rect = Rect::new(crop_x, crop_y, crop_w, crop_h);
    let crop = match opencv::core::Mat::roi(design, crop_rect) {
        Ok(m) => m,
        Err(_) => return None,
    };
    // Copy ROI to a standalone Mat (BoxedRef doesn't deref to Mat)
    let crop_mat = crop.clone_pointee();

    // Run template matching on the crop with dynamic threshold descent
    let mut threshold = start_threshold;
    while threshold >= -1e-9 {
        if let Ok(matches) =
            crate::match_template_nms(&crop_mat, templ, mask, threshold, nms_threshold)
        {
            if let Some((conf, rect)) = matches.into_iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()) {
                return Some(Position {
                    x: crop_x + rect.x,
                    y: crop_y + rect.y,
                    width: rect.width,
                    height: rect.height,
                    confidence: (conf * 10000.0).round() / 10000.0,
                    trust: crate::trust_label(conf).into(),
                    source: "yolo+template".into(),
                });
            }
        }
        threshold = ((threshold - step) * 100.0).round() / 100.0;
    }

    None
}

// ---------------------------------------------------------------------------
// Ext dir & venv management
// ---------------------------------------------------------------------------

/// Find or create `.opencv-ui-yolo-ext/` next to the executable.
pub(crate) fn ext_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    let exe_dir = exe.parent().unwrap_or(Path::new("."));
    let ext_dir = exe_dir.join(".opencv-ui-yolo-ext");

    std::fs::create_dir_all(&ext_dir)?;

    // Always write the latest embedded script
    let script_path = ext_dir.join("yolo_detect.py");
    std::fs::write(&script_path, YOLO_SCRIPT)?;

    Ok(ext_dir)
}

/// Ensure a Python venv exists at `ext_dir/.venv/` with ultralytics installed.
/// Returns the path to the venv's python binary.
fn ensure_venv(ext_dir: &Path) -> Result<PathBuf> {
    let venv_dir = ext_dir.join(".venv");
    let python = venv_python(&venv_dir);

    if python.exists() {
        // Check if ultralytics is already installed
        let check = Command::new(&python)
            .arg("-c")
            .arg("import ultralytics")
            .output();
        if check.map_or(false, |o| o.status.success()) {
            return Ok(python);
        }
        // ultralytics missing — install
        let pip = venv_pip(&venv_dir);
        eprintln!("    YOLO: installing ultralytics into venv...");
        let status = Command::new(&pip)
            .arg("install")
            .arg("-q")
            .arg("ultralytics")
            .status()
            .context("failed to run pip install")?;
        if !status.success() {
            anyhow::bail!("pip install ultralytics failed");
        }
        return Ok(python);
    }

    // Create venv
    let system_python = find_system_python()?;
    eprintln!("    YOLO: creating venv at {}...", venv_dir.display());
    let status = Command::new(&system_python)
        .arg("-m")
        .arg("venv")
        .arg(&venv_dir)
        .status()
        .context("failed to create Python venv")?;
    if !status.success() {
        anyhow::bail!("failed to create Python venv");
    }

    // Install ultralytics
    let pip = venv_pip(&venv_dir);
    eprintln!("    YOLO: installing ultralytics (this may take a while)...");
    let status = Command::new(&pip)
        .arg("install")
        .arg("-q")
        .arg("ultralytics")
        .status()
        .context("failed to run pip install")?;
    if !status.success() {
        anyhow::bail!("pip install ultralytics failed");
    }

    Ok(python)
}

fn find_system_python() -> Result<String> {
    for cmd in &["python3", "python"] {
        if let Ok(output) = Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                return Ok(cmd.to_string());
            }
        }
    }
    anyhow::bail!("python3/python not found on PATH")
}

fn venv_python(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    }
}

fn venv_pip(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts").join("pip.exe")
    } else {
        venv_dir.join("bin").join("pip")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_cache_stale(dir: &Path, cache: &Path) -> bool {
    let cache_mtime = match std::fs::metadata(cache)
        .ok()
        .and_then(|m| m.modified().ok())
    {
        Some(t) => t,
        None => return true,
    };

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return true,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.contains("-matches-") {
            continue;
        }
        if let Ok(mtime) = entry.metadata().and_then(|m| m.modified()) {
            if mtime > cache_mtime {
                return true;
            }
        }
    }

    false
}
