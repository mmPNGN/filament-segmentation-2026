"""Submission CSV: writing and the pre-submit validator (submission.md §4).

Format: columns `filament_id,segmentation_rle`; filament_id = '<stem>_<k>'; compressed COCO RLE for 2048x2048.
Masks of one image must not overlap (host validator rejects the file otherwise, competition.md §2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.annotations import IMAGE_SIZE, rle_to_mask

COLUMNS = ["filament_id", "segmentation_rle"]


def write_submission(rows: list[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return df


def validate_submission(df: pd.DataFrame, expected_stems: list[str] | None = None) -> dict:
    """Raise ValueError on any format violation; return summary stats otherwise."""
    if list(df.columns) != COLUMNS:
        raise ValueError(f"columns must be {COLUMNS}, got {list(df.columns)}")
    if df.isna().any().any():
        raise ValueError("NaN in submission")
    if df["filament_id"].duplicated().any():
        raise ValueError("duplicate filament_id")
    if not df["filament_id"].str.fullmatch(r"\d{14}[A-Z]h_\d+").all():
        bad = df.loc[~df["filament_id"].str.fullmatch(r"\d{14}[A-Z]h_\d+"), "filament_id"].head(3).tolist()
        raise ValueError(f"bad filament_id format: {bad}")
    stems = df["filament_id"].str.rsplit("_", n=1).str[0]
    if expected_stems is not None:
        unknown = set(stems) - set(expected_stems)
        if unknown:
            raise ValueError(f"predictions for unknown images: {sorted(unknown)[:3]}")
    areas = []
    for stem, g in df.groupby(stems, sort=False):
        acc = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for rle in g["segmentation_rle"]:
            m = rle_to_mask(rle)
            a = int(m.sum())
            if a == 0:
                raise ValueError(f"empty mask in {stem}")
            areas.append(a)
            acc += m
        if acc.max() > 1:
            raise ValueError(f"overlapping masks in image {stem}")
    n_images = stems.nunique()
    return {
        "rows": len(df), "images_with_pred": n_images,
        "images_missing": 0 if expected_stems is None else len(set(expected_stems) - set(stems)),
        "per_image_mean": len(df) / max(n_images, 1),
        "area_median": float(np.median(areas)) if areas else 0.0,
    }
