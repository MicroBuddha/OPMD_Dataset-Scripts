"""
utils.py — shared helpers used by every script in this folder.

Nothing in here should need editing; machine-specific values belong in
config.py.
"""

import json
import re
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Patient / filename parsing
#   BM{centre}{CS|CN}{patient4}{batch2}{seq2}
#   Centre codes are variable length: BM1..BM9 = 3 chars, BM10 = 4 chars.
#   Do NOT use a fixed-length prefix to extract the patient key.
# ----------------------------------------------------------------------
_FILENAME_RE = re.compile(
    r"^(?P<centre>BM\d{1,2})(?P<kind>CS|CN)(?P<patient>\d{4})(?P<batch>\d{2})(?P<seq>\d{2})"
)


def parse_filename(stem: str) -> dict:
    """Parse an image/annotation stem like 'BM2CS00132601' into its parts.

    Raises ValueError if the stem doesn't match the expected convention.
    """
    m = _FILENAME_RE.match(stem)
    if not m:
        raise ValueError(f"Filename does not match BM<centre><CS|CN><patient><batch><seq>: {stem}")
    d = m.groupdict()
    d["is_case"] = d["kind"] == "CS"
    return d


def patient_key(stem: str, key_len: int = 9) -> str:
    """Return the patient-level key for a filename stem, e.g. 'BM2CS0013'.

    Uses parse_filename rather than a fixed character offset so BM10 (4-char
    centre code) doesn't collide with BM1-BM9 patients.
    """
    parts = parse_filename(stem)
    key = f"{parts['centre']}{parts['kind']}{parts['patient']}"
    assert len(key) == key_len, f"Unexpected key length for {stem}: {key} ({len(key)} chars)"
    return key


def collapse_class(label: str) -> str:
    """Collapse subclass-suffixed labels (C2H1, C2H2, C1H3, ...) to their
    parent class (C2, C1) for mask generation. H-grading is dropped.
    """
    m = re.match(r"^(C\d)", label.strip())
    if not m:
        return label.strip()
    return m.group(1)


# ----------------------------------------------------------------------
# GeoJSON <-> polygon helpers
# ----------------------------------------------------------------------

def load_geojson_polygons(geojson_path: Path) -> list[dict]:
    """Load a QuPath-exported GeoJSON file into a flat list of
    {"label": str, "points": np.ndarray[N,2]} dicts. Handles Polygon and
    MultiPolygon geometries; only the exterior ring of each polygon is kept.
    """
    with open(geojson_path) as f:
        gj = json.load(f)

    features = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    out = []
    for feat in features:
        props = feat.get("properties", {})
        label = (
            props.get("classification", {}).get("name")
            or props.get("name")
            or props.get("objectType")
            or "unlabelled"
        )
        geom = feat["geometry"]
        gtype = geom["type"]
        coords = geom["coordinates"]

        if gtype == "Polygon":
            rings = [coords[0]]
        elif gtype == "MultiPolygon":
            rings = [poly[0] for poly in coords]
        else:
            continue

        for ring in rings:
            pts = np.array(ring, dtype=np.float64)
            out.append({"label": label, "points": pts})
    return out


def save_geojson_polygons(polygons: list[dict], out_path: Path) -> None:
    """Write a list of {"label", "points"} dicts back out as a GeoJSON
    FeatureCollection, matching the shape load_geojson_polygons reads.
    """
    features = []
    for poly in polygons:
        pts = poly["points"]
        ring = pts.tolist()
        if ring[0] != ring[-1]:
            ring.append(ring[0])  # close the ring
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"classification": {"name": poly["label"]}},
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


# ----------------------------------------------------------------------
# Coordinate transforms
# ----------------------------------------------------------------------

def rotate_points(points: np.ndarray, angle_deg: int, orig_w: int, orig_h: int) -> np.ndarray:
    """Rotate polygon points to match a PIL/EXIF-style rotation of the image
    by angle_deg in {0, 90, 180, 270} (clockwise, matching EXIF Orientation
    correction), given the ORIGINAL (pre-rotation) image width/height.

    Must be called with the raw on-disk image's pre-correction dimensions,
    not any dimensions cached in a database — cached dimensions may already
    reflect the post-correction frame and will silently produce wrong
    coordinates.
    """
    x, y = points[:, 0], points[:, 1]
    if angle_deg == 0:
        new_pts = np.stack([x, y], axis=1)
    elif angle_deg == 90:
        new_pts = np.stack([orig_h - y, x], axis=1)
    elif angle_deg == 180:
        new_pts = np.stack([orig_w - x, orig_h - y], axis=1)
    elif angle_deg == 270:
        new_pts = np.stack([y, orig_w - x], axis=1)
    else:
        raise ValueError(f"Unsupported rotation angle: {angle_deg}")
    return new_pts


def translate_points(points: np.ndarray, offset_x: float, offset_y: float) -> np.ndarray:
    """Shift polygon points into a cropped image's coordinate space."""
    out = points.copy()
    out[:, 0] -= offset_x
    out[:, 1] -= offset_y
    return out


def clip_polygon_to_box(points: np.ndarray, box_w: float, box_h: float) -> np.ndarray | None:
    """Clamp a polygon's points to lie within [0, box_w] x [0, box_h].

    Returns the clamped points, or None if the polygon has zero area after
    clamping (fully outside the box).
    """
    clamped = points.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, box_w)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, box_h)
    x_range = clamped[:, 0].max() - clamped[:, 0].min()
    y_range = clamped[:, 1].max() - clamped[:, 1].min()
    if x_range < 1 or y_range < 1:
        return None
    return clamped


# ----------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_centre_dirs(raw_root: Path, centres: list[str]):
    """Yield (centre_name, date_range_subdir) pairs under raw_root for each
    configured centre, skipping anything that isn't a directory.
    """
    for centre in centres:
        centre_dir = raw_root / centre
        if not centre_dir.is_dir():
            continue
        for sub in sorted(centre_dir.iterdir()):
            if sub.is_dir():
                yield centre, sub
