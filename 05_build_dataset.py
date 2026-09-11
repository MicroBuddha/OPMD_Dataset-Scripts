"""
05_build_dataset.py — the "wholesome" script: runs the full pipeline
end-to-end in the correct order, so a reviewer/collaborator can rebuild the
dataset from raw centre folders with one command.

    raw (BM2..BM10, CONTROL)
      -> 01 EXIF correction
      -> 02 mouth crop (YOLO) + annotation re-projection
      -> 03 mask / label generation (per centre)
      -> 04 centre-wise train/val/test split

Each stage is a thin wrapper around the corresponding numbered script's
main(), so you can also just run the numbered scripts individually for
debugging one stage at a time — this file adds nothing stage-specific of
its own.

Usage:
    python 05_build_dataset.py                  # run everything
    python 05_build_dataset.py --from-stage 3    # resume from mask generation
    python 05_build_dataset.py --dry-run         # exif + crop stages support this; see notes below
    python 05_build_dataset.py --skip-crop       # if you already have cropped images and just
                                                  # want to (re)generate masks + split
"""

import argparse
import sys
import time

import config

STAGES = [
    ("EXIF correction", "01_exif_correction"),
    ("Mouth crop", "02_mouth_crop"),
    ("Mask / label generation", "03_generate_masks"),
    ("Train/val/test split", "04_train_test_split"),
]


def run_stage(stage_num, name, module_name, argv):
    print(f"\n{'=' * 60}\nSTAGE {stage_num}: {name}\n{'=' * 60}")
    t0 = time.time()

    module = __import__(module_name)
    old_argv = sys.argv
    sys.argv = [module_name + ".py"] + argv
    try:
        module.main()
    finally:
        sys.argv = old_argv

    print(f"[{name}] done in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4], help="Resume from this stage (1-4)")
    ap.add_argument("--skip-crop", action="store_true", help="Skip stage 2 (mouth crop)")
    ap.add_argument("--dry-run", action="store_true", help="Passed through to stages that support it (01, 04)")
    args = ap.parse_args()

    print("Config check:")
    for name in ["RAW_ROOT", "EXIF_CORRECTED_ROOT", "CROPPED_ROOT", "PROCESSED_ROOT", "SPLIT_ROOT", "YOLO_MOUTH_CROP_WEIGHTS"]:
        print(f"  {name}: {getattr(config, name)}")

    for i, (name, module_name) in enumerate(STAGES, start=1):
        if i < args.from_stage:
            continue
        if i == 2 and args.skip_crop:
            print("\nSTAGE 2: Mouth crop — skipped (--skip-crop)")
            continue

        stage_argv = []
        if module_name in ("01_exif_correction", "04_train_test_split") and args.dry_run:
            stage_argv.append("--dry-run")

        run_stage(i, name, module_name, stage_argv)

    print(f"\n{'=' * 60}\nPipeline complete. Final split at: {config.SPLIT_ROOT}\n{'=' * 60}")


if __name__ == "__main__":
    main()
