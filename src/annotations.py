"""Loading the MAGFiLO COCO annotations and rasterising polygons.

The competition json stores `segmentation`, `spine`, `bbox`, `area`, `category_id` and `iscrowd` as JSON *strings*,
so every field is parsed here once. Image records are keyed by `record_id = "<annotator>-<stem>"`; several records
share one physical observation (`stem`), which is the unit for fold assignment (validation.md §2.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

IMAGE_SIZE = 2048
OBSERVATORIES = {"B": "Big Bear", "C": "Cerro Tololo", "L": "Learmonth", "M": "Mauna Loa", "T": "El Teide", "U": "Udaipur"}


def _parse(value):
    return json.loads(value) if isinstance(value, str) else value


def parse_stem(stem: str) -> dict:
    """'20140609195854Bh' -> datetime, observatory code."""
    return {
        "datetime": pd.to_datetime(stem[:14], format="%Y%m%d%H%M%S"),
        "observatory": stem[14],
    }


def load_coco(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (records, annotations).

    records: one row per annotator-image: record_id, annotator, stem, file_name, datetime, observatory
    annotations: one row per filament: ann_id, record_id, stem, polygon (np.ndarray (N,2) float32),
                 spine (np.ndarray (M,2)), bbox [x,y,w,h], area, category_id
    """
    with Path(path).open(encoding="utf-8-sig") as f:
        coco = json.load(f)

    rec_rows = []
    for im in coco["images"]:
        annotator, stem = im["id"].split("-", 1)
        rec_rows.append({"record_id": im["id"], "annotator": annotator, "stem": stem,
                         "file_name": im["file_name"], **parse_stem(stem)})
    records = pd.DataFrame(rec_rows)
    stem_of = dict(zip(records["record_id"], records["stem"]))

    ann_rows = []
    for a in coco["annotations"]:
        seg = _parse(a["segmentation"])
        poly = np.asarray(seg[0] if seg and isinstance(seg[0], list) else seg, dtype=np.float32).reshape(-1, 2)
        spine = np.asarray(_parse(a["spine"]) or [], dtype=np.float32).reshape(-1, 2)
        ann_rows.append({
            "ann_id": a["id"], "record_id": a["image_id"], "stem": stem_of[a["image_id"]],
            "polygon": poly, "spine": spine, "bbox": [float(v) for v in _parse(a["bbox"])],
            "area": float(_parse(a["area"])), "category_id": int(_parse(a["category_id"])),
        })
    annotations = pd.DataFrame(ann_rows)
    return records, annotations


def polygon_to_rle(polygon: np.ndarray, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> str:
    """Rasterise one closed polygon (N,2) [x,y] to a compressed COCO RLE counts string."""
    flat = [float(v) for v in np.asarray(polygon, dtype=np.float64).ravel()]
    rles = mask_util.frPyObjects([flat], height, width)
    rle = mask_util.merge(rles)
    return rle["counts"].decode("ascii")


def rle_to_mask(counts: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> np.ndarray:
    """Compressed counts string -> (H, W) uint8 mask."""
    return mask_util.decode({"size": [height, width], "counts": counts.encode("ascii")})


def mask_to_rle(mask: np.ndarray) -> str:
    """(H, W) binary mask -> compressed counts string (Fortran order, as pycocotools expects)."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def annotations_to_gt_df(annotations: pd.DataFrame) -> pd.DataFrame:
    """Ground truth in the host's evaluation format: filament_id = '<record_id>_<k>', segmentation_rle."""
    rows = []
    for record_id, group in annotations.groupby("record_id", sort=False):
        for k, poly in enumerate(group["polygon"], start=1):
            rows.append({"filament_id": f"{record_id}_{k}", "segmentation_rle": polygon_to_rle(poly)})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
