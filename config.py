"""
config.py — single place to edit before running any script.

Everything the other scripts need to know about your machine, your dataset
layout, and your model weights lives here. Edit these values, then run the
scripts in order (01 -> 04), or run 05_build_dataset.py to do all of it
in one shot.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Paths — EDIT THESE
# ----------------------------------------------------------------------

# Root of the raw, as-collected dataset. Expected layout:
#   RAW_ROOT/
#     BM2/1 - 5th June 2026/Annotation_GeoJSON/*.geojson
#     BM2/1 - 5th June 2026/Images/*.jpg
#     BM2/1 - 5th June 2026/<Cases Metadata>.xlsx
#     BM2/1 - 5th June 2026/classes.json
#     ...
#     CONTROL/...
RAW_ROOT = Path("/path/to/ICMR_OPMD_2606")

# Where corrected (EXIF-fixed) images + geojson get written by 01_exif_correction.py
EXIF_CORRECTED_ROOT = Path("/path/to/ICMR_OPMD_2606/_exif_corrected")

# Where mouth-cropped images + re-projected annotations get written by 02_mouth_crop.py
CROPPED_ROOT = Path("/path/to/ICMR_OPMD_2606/_cropped")

# Where per-centre processed folders (images/labels/masks_*) get written by 03_generate_masks.py
PROCESSED_ROOT = Path("/path/to/ICMR_OPMD_2606/_processed")

# Where the final train/val/test split (symlinks or copies) gets written by 04_train_test_split.py
SPLIT_ROOT = Path("/path/to/ICMR_OPMD_2606/_split")

# Path to your trained mouth-crop YOLO segmentation weights (.pt)
YOLO_MOUTH_CROP_WEIGHTS = Path("/path/to/weights/mouth_crop_yolo.pt")

# ----------------------------------------------------------------------
# Centres included in the multicentric split
# ----------------------------------------------------------------------
CENTRES = ["BM2", "BM3", "BM4", "BM5", "BM6", "BM7", "BM9"]
CONTROL_DIR_NAME = "CONTROL"

# ----------------------------------------------------------------------
# Filename / patient-ID convention
#   BM{centre}{CS|CN}{patient4}{batch2}{seq2}
#   e.g. BM2CS00132601 -> centre=2, case, patient=0013, batch=26, seq=01
#   patient key = first 9 characters (BM2CS0013)
# ----------------------------------------------------------------------
PATIENT_KEY_LEN = 9

# ----------------------------------------------------------------------
# Class scheme
#   H1/H2/H3 histological grading: dropped, not used downstream
#   Subclass-suffixed labels (C2H1/C2H2/C2H3, C1H1/...) collapse to parent
#   Binary/instance/semantic masks: only C1, C2
#   YOLO export: keeps C1's subclasses (does NOT collapse for YOLO labels)
#   Controls are always labelled C1
# ----------------------------------------------------------------------
MASK_CLASSES = ["C1", "C2"]          # order fixes mask pixel values (1, 2); 0 = background
YOLO_CLASS_INDEX = {"C1": 0, "C2": 1, "C3": 2}   # yolo_idx = sem_idx - 1

# ----------------------------------------------------------------------
# Crop parameters (mouth-region YOLO crop)
# ----------------------------------------------------------------------
CROP_IMGSZ = 1024          # inference size; CPU-safe default (avoids GPU OOM seen previously)
CROP_CONF = 0.25           # detection confidence threshold
CROP_PAD_FRAC = 0.08       # fractional padding added around the detected box on each side
ON_CLIP = "skip"           # "skip" or "expand" — behaviour when a lesion polygon
                            # would be clipped by the crop box. "skip": drop that
                            # annotation for this image. "expand": grow the crop
                            # box to keep the polygon fully inside.

# ----------------------------------------------------------------------
# Train / val / test split (centre-wise, not patient-wise pooled)
#   Whole centres are assigned to a split so no patient/centre leaks across
#   train/val/test.
# ----------------------------------------------------------------------
SPLIT_ASSIGNMENT = {
    # EDIT: assign each centre in CENTRES to exactly one split
    "train": ["BM2", "BM3", "BM4", "BM6", "BM9"],
    "val":   ["BM5"],
    "test":  ["BM7"],
}
SPLIT_MODE = "copy"        # "copy" or "symlink"

# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------
MERGED_METADATA_CSV = Path("/path/to/merged_clean.csv")
