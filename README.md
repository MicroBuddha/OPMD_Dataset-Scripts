# OPMD_26 — data preparation scripts

Standalone reproducibility scripts for the image preprocessing pipeline
used to build the ICMR_OPMD_2606 dataset (oral lesion photographs,
multicentric, BM2–BM10 + CONTROL). Intended for the paper's Code
Availability section / a reproducibility repo — this is deliberately
separate from the OPMD Data Manager web app used for day-to-day
annotation review.

## What's here

| File | Stage | What it does |
|---|---|---|
| `config.py` | — | Every path and parameter you need to edit before running anything |
| `utils.py` | — | Shared helpers (filename parsing, GeoJSON I/O, coordinate transforms) |
| `01_exif_correction.py` | 1 | Physically rotates images to match EXIF orientation; re-projects GeoJSON lesion polygons into the corrected frame |
| `02_mouth_crop.py` | 2 | YOLO-seg mouth-region crop; re-projects annotations into crop coordinates with padding/clamping |
| `03_generate_masks.py` | 3 | Builds YOLO-seg labels, binary masks, semantic masks (C1/C2), and colour-coded instance masks per centre |
| `04_train_test_split.py` | 4 | Centre-wise train/val/test split (whole centres per split, no leakage) |
| `05_build_dataset.py` | 1–4 | Runs the full pipeline end to end in order |

## Setup

```bash
pip install -r requirements.txt
```

Edit **`config.py`**: set `RAW_ROOT`, the intermediate/output roots, the
path to your trained mouth-crop YOLO weights, and `SPLIT_ASSIGNMENT`
(which centres go to train/val/test).

## Running

Full pipeline:

```bash
python 05_build_dataset.py
```

Or run stages individually (useful for debugging one stage, or if you
already have output from an earlier stage):

```bash
python 01_exif_correction.py --dry-run   # check what would be rotated, first
python 01_exif_correction.py
python 02_mouth_crop.py
python 03_generate_masks.py
python 04_train_test_split.py
```

Resume the orchestrator partway through:

```bash
python 05_build_dataset.py --from-stage 3
```

## Conventions this pipeline assumes

- **Filenames**: `BM{centre}{CS|CN}{patient4}{batch2}{seq2}`, e.g.
  `BM2CS00132601`. Centre codes are variable length (`BM1`–`BM9` = 3
  chars, `BM10` = 4 chars) — patient keys are parsed with a regex, not a
  fixed character offset.
- **Classes**: H1/H2/H3 histological grading is dropped everywhere.
  Subclass-suffixed labels (`C2H1`, `C2H2`, …) collapse to their parent
  class (`C1`, `C2`) for masks — **except** the YOLO label file, which
  keeps `C1`'s subclasses distinct. Controls are treated as class `C1`.
- **Rotation math** always uses the raw on-disk image's pre-correction
  width/height, never a cached value from a database or elsewhere.
- **Crop math** always re-uses the exact padded/clamped box that pixels
  were cropped with (not the raw YOLO detection box) when transforming
  annotation coordinates.
- **Split**: centre-wise, not patient-wise-pooled — an entire centre's
  images go to exactly one of train/val/test, so no patient or centre
  appears in more than one split.

## Not included here

Near-duplicate removal (cosine-similarity de-dup), metadata-workbook
merging/cleaning (`merged_clean.csv`), and the annotation-review /
QC-overlay tooling live in the Data Manager app and separate QC
scripts — out of scope for this reproducibility bundle, which covers
image crop, preprocessing, and split only.
