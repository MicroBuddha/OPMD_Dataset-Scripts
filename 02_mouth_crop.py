"""
02_mouth_crop.py — crop every EXIF-corrected image to the mouth region using
a trained YOLO segmentation model, and re-project lesion polygons into the
cropped image's coordinate space.

Order of operations for the overall dataset build: mouth-crop first, THEN
train/test split, THEN add controls (see 05_build_dataset.py).

Crop-offset safety note: this script stores and uses the actual padded /
clamped box that pixels were cropped with (`used_box`) for the coordinate
transform — NOT the raw YOLO detection bbox. Using the raw detection box
here was a previously-hit bug: labels end up offset from the lesion because
the crop and the annotation transform used two different boxes.

Usage:
    python 02_mouth_crop.py
    python 02_mouth_crop.py --centre BM7
    python 02_mouth_crop.py --device cpu          # default; avoids GPU OOM
"""

import argparse
import json

import cv2
import numpy as np
from ultralytics import YOLO

import config
from utils import (
    clip_polygon_to_box,
    ensure_dir,
    iter_centre_dirs,
    load_geojson_polygons,
    save_geojson_polygons,
    translate_points,
)


def detect_mouth_box(model, img_bgr, imgsz, conf):
    """Run the mouth-crop YOLO-seg model and return the best detection's
    box as (x1, y1, x2, y2) in pixel coords, or None if nothing detected.
    """
    results = model.predict(img_bgr, imgsz=imgsz, conf=conf, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    # pick highest-confidence detection
    best_idx = int(r.boxes.conf.argmax())
    x1, y1, x2, y2 = r.boxes.xyxy[best_idx].tolist()
    return x1, y1, x2, y2


def pad_and_clamp_box(box, img_w, img_h, pad_frac):
    """Expand a detection box by pad_frac on each side and clamp to the
    image bounds. Returns the box actually used for cropping (used_box) —
    this exact box, not the raw detection, must be reused for the polygon
    coordinate transform.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * pad_frac, h * pad_frac
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)
    return x1, y1, x2, y2


def crop_and_reproject(img_path, geojson_path, out_img_path, out_geojson_path, model, on_clip):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    img_h, img_w = img_bgr.shape[:2]

    box = detect_mouth_box(model, img_bgr, config.CROP_IMGSZ, config.CROP_CONF)
    if box is None:
        raise RuntimeError(f"No mouth-region detection for: {img_path}")

    used_box = pad_and_clamp_box(box, img_w, img_h, config.CROP_PAD_FRAC)
    x1, y1, x2, y2 = [int(round(v)) for v in used_box]
    crop = img_bgr[y1:y2, x1:x2]

    ensure_dir(out_img_path.parent)
    cv2.imwrite(str(out_img_path), crop)

    box_w, box_h = x2 - x1, y2 - y1

    if geojson_path.exists():
        polygons = load_geojson_polygons(geojson_path)
        kept = []
        for poly in polygons:
            pts = translate_points(poly["points"], offset_x=x1, offset_y=y1)
            clamped = clip_polygon_to_box(pts, box_w, box_h)

            if clamped is None:
                # polygon fell entirely outside the crop
                continue

            was_clipped = not np.array_equal(clamped, pts)
            if was_clipped and on_clip == "skip":
                continue
            # on_clip == "expand" is handled by never clamping the crop box
            # itself in this simple version — polygons that would clip are
            # instead clamped to the box, which keeps them fully inside.

            poly["points"] = clamped
            kept.append(poly)

        save_geojson_polygons(kept, out_geojson_path)

    # record the box actually used, for auditing / QC overlays
    meta_path = out_img_path.with_suffix(".crop.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "source_image": str(img_path),
                "detection_box_xyxy": list(box),
                "used_box_xyxy": [x1, y1, x2, y2],
                "orig_w": img_w,
                "orig_h": img_h,
            },
            f,
            indent=2,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centre", help="Process a single centre only, e.g. BM7")
    ap.add_argument("--device", default="cpu", help="YOLO device, e.g. 'cpu' or '0'")
    args = ap.parse_args()

    centres = [args.centre] if args.centre else config.CENTRES + [config.CONTROL_DIR_NAME]

    model = YOLO(str(config.YOLO_MOUTH_CROP_WEIGHTS))
    model.to(args.device)

    n_ok, n_fail = 0, 0

    for centre, date_dir in iter_centre_dirs(config.EXIF_CORRECTED_ROOT, centres):
        images_dir = date_dir / "Images"
        geojson_dir = date_dir / "Annotation_GeoJSON"
        if not images_dir.is_dir():
            continue

        rel = date_dir.relative_to(config.EXIF_CORRECTED_ROOT)
        out_images_dir = config.CROPPED_ROOT / rel / "Images"
        out_geojson_dir = config.CROPPED_ROOT / rel / "Annotation_GeoJSON"

        for img_path in sorted(images_dir.glob("*.*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            geojson_path = geojson_dir / f"{img_path.stem}.geojson"
            try:
                crop_and_reproject(
                    img_path,
                    geojson_path,
                    out_images_dir / img_path.name,
                    out_geojson_dir / f"{img_path.stem}.geojson",
                    model,
                    config.ON_CLIP,
                )
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"[ERROR] {img_path}: {e}")

    print(f"\nCropped OK: {n_ok}")
    print(f"Failed:     {n_fail}")
    print(f"Output:     {config.CROPPED_ROOT}")


if __name__ == "__main__":
    main()
