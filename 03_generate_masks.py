"""
03_generate_masks.py — preprocess: turn each cropped image's GeoJSON
annotations into the mask / label formats used downstream, and lay
everything out per-centre in the shape the split step expects:

    PROCESSED_ROOT/
      <CENTRE>/
        images/            <name>.jpg
        labels/            <name>.txt        (YOLO-seg polygons, C1 subclasses kept)
        masks_binary/       <name>.png        (0 = background, 255 = any lesion)
        masks_semantic/     <name>.png        (0 = bg, 1 = C1, 2 = C2)
        masks_instance/      <name>.png + <name>.json   (per-instance colour + class name)

Class handling:
  - H1/H2/H3 histological grading is dropped everywhere.
  - Semantic/binary/instance masks collapse subclass-suffixed labels
    (C2H1, C2H2, ... ) to their parent class C1/C2 (see utils.collapse_class).
  - The YOLO label file is the one place subclasses are NOT collapsed —
    it keeps C1's subclasses as distinct classes.
  - Controls have no annotations; they are still copied into images/ with
    an empty labels/ file and an all-background mask, tagged class C1
    per centre metadata (see 05_build_dataset.py for how controls are
    merged in).

Usage:
    python 03_generate_masks.py
    python 03_generate_masks.py --centre BM7
"""

import argparse
import json

import cv2
import numpy as np

import config
from utils import collapse_class, ensure_dir, iter_centre_dirs, load_geojson_polygons


def golden_ratio_hue(index: int) -> float:
    """Evenly-spaced, visually distinct hues for instance colouring."""
    golden_ratio_conjugate = 0.618033988749895
    return (index * golden_ratio_conjugate) % 1.0


def hsv_to_bgr_u8(h, s=0.85, v=0.95):
    hsv = np.uint8([[[h * 179, s * 255, v * 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


def write_yolo_seg_label(polygons, img_w, img_h, out_path):
    """YOLO-seg format: one line per instance,
    '<class_idx> x1 y1 x2 y2 ... xn yn' with coords normalised to [0, 1].
    Subclasses are NOT collapsed here (kept as-is for C1's subclasses).
    """
    lines = []
    for poly in polygons:
        label = poly["label"].strip()
        base = label[:2] if label[:2] in ("C1", "C2", "C3") else None
        cls_idx = config.YOLO_CLASS_INDEX.get(base if base else label)
        if cls_idx is None:
            continue  # e.g. "unlabelled" — needs a manual class mapping before export
        pts = poly["points"]
        norm = pts / np.array([img_w, img_h])
        coords_str = " ".join(f"{c:.6f}" for c in norm.flatten())
        lines.append(f"{cls_idx} {coords_str}")

    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def build_masks(polygons, img_w, img_h):
    """Returns (binary_mask, semantic_mask, instance_mask, instance_meta)."""
    binary = np.zeros((img_h, img_w), dtype=np.uint8)
    semantic = np.zeros((img_h, img_w), dtype=np.uint8)
    instance = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    instance_meta = []

    inst_idx = 0
    for poly in polygons:
        cls = collapse_class(poly["label"])
        if cls not in config.MASK_CLASSES:
            continue  # e.g. unlabelled / ROI objects are excluded from masks
        sem_val = config.MASK_CLASSES.index(cls) + 1  # 0 is background

        pts_i = poly["points"].round().astype(np.int32)
        cv2.fillPoly(binary, [pts_i], 255)
        cv2.fillPoly(semantic, [pts_i], sem_val)

        colour = hsv_to_bgr_u8(golden_ratio_hue(inst_idx))
        cv2.fillPoly(instance, [pts_i], colour)
        instance_meta.append({"instance_id": inst_idx, "class": cls, "colour_bgr": colour})
        inst_idx += 1

    return binary, semantic, instance, instance_meta


def process_one(img_path, geojson_path, out_dirs):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    img_h, img_w = img.shape[:2]

    polygons = load_geojson_polygons(geojson_path) if geojson_path.exists() else []

    ensure_dir(out_dirs["images"])
    cv2.imwrite(str(out_dirs["images"] / img_path.name), img)

    write_yolo_seg_label(polygons, img_w, img_h, out_dirs["labels"] / f"{img_path.stem}.txt")

    binary, semantic, instance, instance_meta = build_masks(polygons, img_w, img_h)

    ensure_dir(out_dirs["masks_binary"])
    cv2.imwrite(str(out_dirs["masks_binary"] / f"{img_path.stem}.png"), binary)

    ensure_dir(out_dirs["masks_semantic"])
    cv2.imwrite(str(out_dirs["masks_semantic"] / f"{img_path.stem}.png"), semantic)

    ensure_dir(out_dirs["masks_instance"])
    cv2.imwrite(str(out_dirs["masks_instance"] / f"{img_path.stem}.png"), instance)
    with open(out_dirs["masks_instance"] / f"{img_path.stem}.json", "w") as f:
        json.dump(instance_meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centre", help="Process a single centre only, e.g. BM7")
    args = ap.parse_args()

    centres = [args.centre] if args.centre else config.CENTRES + [config.CONTROL_DIR_NAME]

    n_ok, n_fail = 0, 0

    for centre, date_dir in iter_centre_dirs(config.CROPPED_ROOT, centres):
        images_dir = date_dir / "Images"
        geojson_dir = date_dir / "Annotation_GeoJSON"
        if not images_dir.is_dir():
            continue

        centre_out = config.PROCESSED_ROOT / centre
        out_dirs = {
            "images": centre_out / "images",
            "labels": centre_out / "labels",
            "masks_binary": centre_out / "masks_binary",
            "masks_semantic": centre_out / "masks_semantic",
            "masks_instance": centre_out / "masks_instance",
        }

        for img_path in sorted(images_dir.glob("*.*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            geojson_path = geojson_dir / f"{img_path.stem}.geojson"
            try:
                process_one(img_path, geojson_path, out_dirs)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"[ERROR] {img_path}: {e}")

    print(f"\nProcessed OK: {n_ok}")
    print(f"Failed:       {n_fail}")
    print(f"Output:       {config.PROCESSED_ROOT}")


if __name__ == "__main__":
    main()
