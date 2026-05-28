#!/usr/bin/env python3
"""YOLO sidecar for opencv-ui-cli: auto-train on components/, detect on design.

Subcommands:
    train   --components DIR --output PATH [--epochs N] [--imgsz SZ]
    detect  --model PATH --source PATH --output PATH [--conf FLOAT]
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Set ultralytics cache directory to the same dir as this script
# (extracted to .opencv-ui-yolo-ext/ next to the executable).
# Must be set before `from ultralytics import YOLO`.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="YOLO sidecar for UI component detection")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Generate training data and fine-tune YOLO")
    p.add_argument("--components", required=True, help="Path to components/ directory")
    p.add_argument("--output", required=True, help="Output model path (.pt)")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs (default 50)")
    p.add_argument("--imgsz", type=int, default=640, help="Image size (default 640)")

    p = sub.add_parser("detect", help="Run YOLO detection on an image")
    p.add_argument("--model", required=True, help="Path to trained model (.pt)")
    p.add_argument("--source", required=True, help="Path to image to run detection on")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _make_guid(name: str) -> str:
    """Derive a meaningless GUID from a filename so YOLO is not misled by semantics."""
    return "c" + hashlib.md5(name.encode()).hexdigest()[:7]


def collect_components(comp_dir: Path) -> tuple[dict[str, Image.Image], dict[str, str]]:
    """Return ({guid: PIL.Image}, {guid: original_stem}) for every image in comp_dir."""
    components = {}
    guid_map = {}  # guid → original stem (for reverse lookup)
    for f in sorted(comp_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        if "-matches-" in f.stem:
            continue
        stem = f.stem
        guid = _make_guid(stem)
        guid_map[guid] = stem
        img = Image.open(f).convert("RGBA")
        components[guid] = img
    return components, guid_map


def random_background(w: int, h: int) -> Image.Image:
    """Generate a random solid or gradient background."""
    choice = random.random()
    if choice < 0.4:
        # Solid random color
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        return Image.new("RGB", (w, h), (r, g, b))
    elif choice < 0.7:
        # Vertical gradient
        r1, g1, b1 = random.randint(180, 255), random.randint(180, 255), random.randint(180, 255)
        r2, g2, b2 = random.randint(0, 80), random.randint(0, 80), random.randint(0, 80)
        bg = Image.new("RGB", (w, h))
        for y in range(h):
            t = y / max(h - 1, 1)
            rr = int(r1 + (r2 - r1) * t)
            gg = int(g1 + (g2 - g1) * t)
            bb = int(b1 + (b2 - b1) * t)
            for x in range(w):
                bg.putpixel((x, y), (rr, gg, bb))
        return bg
    else:
        # Noise / checker
        bg = Image.new("RGB", (w, h))
        c1 = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        c2 = (random.randint(100, 180), random.randint(100, 180), random.randint(100, 180))
        for y in range(h):
            for x in range(w):
                bg.putpixel((x, y), c1 if (x // 20 + y // 20) % 2 == 0 else c2)
        return bg


def generate_training_data(
    components: dict[str, Image.Image],
    output_dir: Path,
    num_samples: int = 50,
    imgsz: int = 640,
):
    """Generate synthetic training images and YOLO annotations."""
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    class_names = sorted(components.keys())
    class_map = {name: i for i, name in enumerate(class_names)}

    sample_idx = 0

    for _ in range(num_samples):
        bg_w = random.randint(imgsz // 2, imgsz)
        bg_h = random.randint(imgsz // 2, imgsz)
        bg = random_background(bg_w, bg_h)

        labels = []

        # Place 2–5 random components on each background
        num_objs = random.randint(2, min(5, len(components)))
        placed_components = random.sample(list(components.items()), num_objs)

        for cname, cimg in placed_components:
            # Random scale ±20%
            scale = random.uniform(0.8, 1.2)
            cw = int(cimg.width * scale)
            ch = int(cimg.height * scale)
            if cw < 10 or ch < 10:
                continue

            resized = cimg.resize((cw, ch), Image.LANCZOS)

            # Random rotation ±5°
            angle = random.uniform(-5, 5)
            if angle != 0:
                resized = resized.rotate(angle, expand=True, resample=Image.BILINEAR)
                cw, ch = resized.size

            # Random position
            if cw >= bg_w or ch >= bg_h:
                continue
            x = random.randint(0, bg_w - cw)
            y = random.randint(0, bg_h - ch)

            # Paste with alpha compositing
            bg.paste(resized, (x, y), resized)

            # YOLO format: class_id cx cy w h (normalized)
            cx = (x + cw / 2) / bg_w
            cy = (y + ch / 2) / bg_h
            nw = cw / bg_w
            nh = ch / bg_h
            labels.append(f"{class_map[cname]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not labels:
            continue

        # Save image
        img_path = img_dir / f"synth_{sample_idx:05d}.jpg"
        bg.save(img_path, quality=90)

        # Save labels
        lbl_path = lbl_dir / f"synth_{sample_idx:05d}.txt"
        lbl_path.write_text("\n".join(labels))

        sample_idx += 1

    return class_names


def train_yolo(
    components_dir: Path,
    output_path: Path,
    epochs: int,
    imgsz: int,
) -> None:
    """Full training pipeline: generate data → train → save model."""
    from ultralytics import YOLO

    components, guid_map = collect_components(components_dir)
    if not components:
        print(json.dumps({"error": "no component images found"}))
        sys.exit(1)

    class_names = sorted(components.keys())  # GUIDs
    stem_names = [guid_map[g] for g in class_names]
    print(f"Found {len(components)} component(s): {stem_names}", file=sys.stderr)

    # Generate training data
    work_dir = Path(tempfile.mkdtemp(prefix="yolo_train_"))
    try:
        datasets_dir = work_dir / "datasets" / "train"
        generate_training_data(components, datasets_dir, num_samples=50, imgsz=imgsz)

        # Write data.yaml (uses GUID class names)
        data_yaml = work_dir / "data.yaml"
        data_yaml.write_text(
            f"path: {work_dir / 'datasets'}\n"
            "train: train/images\n"
            "val: train/images\n"
            f"nc: {len(class_names)}\n"
            f"names: {json.dumps(class_names)}\n"
        )

        print(f"Generated {len(list((datasets_dir / 'images').iterdir()))} training images", file=sys.stderr)

        # Train
        model = YOLO("./yolov8n.pt")
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            device="cpu",
            verbose=False,
            exist_ok=True,
        )

        # Save class map alongside model (GUID → original stem)
        class_map_path = output_path.parent / (output_path.stem + "_map.json")
        class_map_path.write_text(json.dumps(guid_map, indent=2))

        # Export/save trained model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy(best_pt, output_path)
        else:
            model.save(str(output_path))

        print(f"Model saved to {output_path}", file=sys.stderr)
        print(f"Class map saved to {class_map_path}", file=sys.stderr)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------

def run_detect(model_path: Path, source_path: Path, output_path: Path, conf: float) -> None:
    """Run YOLO detection and write JSON results."""
    from ultralytics import YOLO

    if not model_path.exists():
        print(json.dumps({"error": f"model not found: {model_path}", "count": 0}))
        sys.exit(1)

    # Load GUID → original stem mapping
    class_map_path = model_path.parent / (model_path.stem + "_map.json")
    guid_to_stem: dict[str, str] = {}
    if class_map_path.exists():
        guid_to_stem = json.loads(class_map_path.read_text())

    model = YOLO(str(model_path))
    results = model(source_path, conf=conf, device="cpu", verbose=False)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            guid = model.names.get(cls_id, str(cls_id))
            # Reverse-map GUID to original component name
            cls_name = guid_to_stem.get(guid, guid)
            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            bbox = [
                int(xyxy[0]),                    # x
                int(xyxy[1]),                    # y
                int(xyxy[2] - xyxy[0]),          # width
                int(xyxy[3] - xyxy[1]),          # height
            ]
            detections.append({
                "class_name": cls_name,
                "confidence": round(conf_val, 4),
                "bbox": bbox,
            })

    output = {
        "detections": detections,
        "count": len(detections),
        "error": None,
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.command == "train":
        train_yolo(
            components_dir=Path(args.components),
            output_path=Path(args.output),
            epochs=args.epochs,
            imgsz=args.imgsz,
        )
    elif args.command == "detect":
        run_detect(
            model_path=Path(args.model),
            source_path=Path(args.source),
            output_path=Path(args.output),
            conf=args.conf,
        )


if __name__ == "__main__":
    main()
