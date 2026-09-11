"""
04_train_test_split.py — build the final train/val/test dataset from the
per-centre processed folders, using a CENTRE-WISE split (whole centres are
assigned to a single split) so that no patient or centre leaks across
train/val/test. The centre -> split assignment is set in config.SPLIT_ASSIGNMENT.

(Centre-wise was chosen after first considering leave-one-centre-out and a
pooled-stratified patient-wise split; centre-wise is what's implemented here.
If you need a different scheme — leave-one-centre-out, pooled patient-wise
stratified by class — write a new SPLIT_ASSIGNMENT-builder function and keep
the rest of this script as-is.)

Output layout:
    SPLIT_ROOT/
      train/images/, train/labels/, train/masks_binary/, ...
      val/...
      test/...

Files are copied by default (config.SPLIT_MODE = "copy"); set it to
"symlink" to save disk space during experimentation.

Usage:
    python 04_train_test_split.py
    python 04_train_test_split.py --dry-run
"""

import argparse
import shutil

import config
from utils import ensure_dir


SUBFOLDERS = ["images", "labels", "masks_binary", "masks_semantic", "masks_instance"]


def place_file(src, dst, mode, dry_run):
    if dry_run:
        return
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # sanity check: every centre appears in exactly one split
    assigned = {}
    for split, centres in config.SPLIT_ASSIGNMENT.items():
        for c in centres:
            if c in assigned:
                raise ValueError(f"Centre {c} assigned to both '{assigned[c]}' and '{split}'")
            assigned[c] = split

    counts = {split: 0 for split in config.SPLIT_ASSIGNMENT}

    for centre, split in assigned.items():
        centre_dir = config.PROCESSED_ROOT / centre
        images_dir = centre_dir / "images"
        if not images_dir.is_dir():
            print(f"[WARN] no processed images for centre {centre}, skipping")
            continue

        for img_path in sorted(images_dir.glob("*.*")):
            stem = img_path.stem
            for sub in SUBFOLDERS:
                src_dir = centre_dir / sub
                if sub == "images":
                    src = src_dir / img_path.name
                elif sub == "labels":
                    src = src_dir / f"{stem}.txt"
                elif sub == "masks_instance":
                    # copy both the PNG and its JSON sidecar
                    for ext in (".png", ".json"):
                        s = src_dir / f"{stem}{ext}"
                        if s.exists():
                            place_file(s, config.SPLIT_ROOT / split / sub / s.name, config.SPLIT_MODE, args.dry_run)
                    continue
                else:
                    src = src_dir / f"{stem}.png"

                if src.exists():
                    place_file(src, config.SPLIT_ROOT / split / sub / src.name, config.SPLIT_MODE, args.dry_run)

            counts[split] += 1

    print("\nSplit sizes (images):")
    for split, n in counts.items():
        centres = config.SPLIT_ASSIGNMENT[split]
        print(f"  {split:5s}: {n:5d}   (centres: {', '.join(centres)})")
    if args.dry_run:
        print("(dry run — nothing written)")
    else:
        print(f"\nOutput: {config.SPLIT_ROOT}")


if __name__ == "__main__":
    main()
