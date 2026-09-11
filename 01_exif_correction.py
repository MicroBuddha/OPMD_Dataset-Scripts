"""
01_exif_correction.py — correct EXIF-rotated images and re-project their
GeoJSON lesion polygons into the corrected frame.

Why this exists: some BM folders (e.g. BM7) contain images whose pixel data
is stored in one orientation while the EXIF Orientation tag says to rotate
it on display. Viewers that respect EXIF show the image "upright"; anything
that reads raw pixels (segmentation training, OpenCV, etc.) sees it
sideways/upside-down — and the QuPath-exported polygon coordinates were
drawn against the upright *display* orientation, so they no longer line up
with raw pixel data unless we physically rotate the pixels and transform
the polygons together.

IMPORTANT: rotation must use the RAW on-disk image dimensions (read fresh
from the file), never dimensions cached anywhere else — a cached width/
height may already reflect a previous correction pass and silently produces
wrong offsets.

Usage:
    python 01_exif_correction.py
    python 01_exif_correction.py --centre BM7          # single centre only
    python 01_exif_correction.py --dry-run             # report only, write nothing
"""

import argparse
import shutil

from PIL import Image, ImageOps

import config
from utils import ensure_dir, iter_centre_dirs, load_geojson_polygons, rotate_points, save_geojson_polygons

# PIL EXIF Orientation tag -> degrees of clockwise rotation needed to correct it
_EXIF_ORIENTATION_TO_ROTATION = {
    1: 0,
    3: 180,
    6: 270,   # 6 = rotated 90 CW at capture -> needs 270 CW to correct (i.e. 90 CCW)
    8: 90,    # 8 = rotated 90 CCW at capture -> needs 90 CW to correct
}


def get_exif_rotation(img: Image.Image) -> int:
    """Return the clockwise rotation (0/90/180/270) implied by the image's
    EXIF Orientation tag, or 0 if there is none / it's already upright.
    """
    try:
        exif = img.getexif()
    except Exception:
        return 0
    orientation = exif.get(0x0112, 1)  # 0x0112 = Orientation tag
    return _EXIF_ORIENTATION_TO_ROTATION.get(orientation, 0)


def correct_image(img_path, out_path, dry_run=False):
    """Load img_path, apply EXIF-implied rotation to the pixel data, strip
    the now-redundant Orientation tag, and save to out_path. Returns
    (rotation_deg, orig_w, orig_h) using the image's RAW pre-correction
    dimensions.
    """
    with Image.open(img_path) as img:
        orig_w, orig_h = img.size
        rotation = get_exif_rotation(img)

        # ImageOps.exif_transpose physically rotates pixels to match EXIF
        # intent and drops the Orientation tag — this is the corrected image.
        corrected = ImageOps.exif_transpose(img)

        if not dry_run:
            ensure_dir(out_path.parent)
            corrected.save(out_path)

    return rotation, orig_w, orig_h


def correct_geojson(geojson_path, out_path, rotation, orig_w, orig_h, dry_run=False):
    if rotation == 0:
        if not dry_run:
            ensure_dir(out_path.parent)
            shutil.copy2(geojson_path, out_path)
        return

    polygons = load_geojson_polygons(geojson_path)
    for poly in polygons:
        poly["points"] = rotate_points(poly["points"], rotation, orig_w, orig_h)

    if not dry_run:
        save_geojson_polygons(polygons, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centre", help="Process a single centre only, e.g. BM7")
    ap.add_argument("--dry-run", action="store_true", help="Report what would happen without writing files")
    args = ap.parse_args()

    centres = [args.centre] if args.centre else config.CENTRES + [config.CONTROL_DIR_NAME]

    n_images, n_rotated, n_errors = 0, 0, 0

    for centre, date_dir in iter_centre_dirs(config.RAW_ROOT, centres):
        images_dir = date_dir / "Images"
        geojson_dir = date_dir / "Annotation_GeoJSON"
        if not images_dir.is_dir():
            continue

        rel = date_dir.relative_to(config.RAW_ROOT)
        out_images_dir = config.EXIF_CORRECTED_ROOT / rel / "Images"
        out_geojson_dir = config.EXIF_CORRECTED_ROOT / rel / "Annotation_GeoJSON"

        for img_path in sorted(images_dir.glob("*.*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            n_images += 1
            try:
                rotation, orig_w, orig_h = correct_image(
                    img_path, out_images_dir / img_path.name, dry_run=args.dry_run
                )
            except Exception as e:
                n_errors += 1
                print(f"[ERROR] {img_path}: {e}")
                continue

            if rotation != 0:
                n_rotated += 1

            geojson_path = geojson_dir / f"{img_path.stem}.geojson"
            if geojson_path.exists():
                try:
                    correct_geojson(
                        geojson_path,
                        out_geojson_dir / geojson_path.name,
                        rotation,
                        orig_w,
                        orig_h,
                        dry_run=args.dry_run,
                    )
                except Exception as e:
                    n_errors += 1
                    print(f"[ERROR] {geojson_path}: {e}")

    print(f"\nImages scanned:  {n_images}")
    print(f"Images rotated:  {n_rotated}")
    print(f"Errors:          {n_errors}")
    if args.dry_run:
        print("(dry run — nothing written)")
    else:
        print(f"Corrected data written to: {config.EXIF_CORRECTED_ROOT}")


if __name__ == "__main__":
    main()
