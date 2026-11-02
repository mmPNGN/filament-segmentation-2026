"""Fold assignment (the measuring instrument; validation.md §1.2). Generates data/folds.csv once.

    uv run python -m src.folds --dry-run     # print the balance table only
    uv run python -m src.folds               # write data/folds.csv (needs user approval; commit it)

Design (research/logs/2026-09-29_eda.md §3): GroupKFold(5) over *time blocks*. Observations sorted by time are
chained into one block while the gap to the previous observation is <= `gap_hours` (72h = ±3 days), so that
every annotator record of one observation, and every observation of a multi-day filament, lands in one fold.
Blocks are dealt to folds greedily (largest first, to the fold with the fewest observations), then checked for
observatory / year balance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.annotations import load_coco  # noqa: E402

N_FOLDS = 5
GAP_HOURS = 72.0
SEED = 42
ANNOTATIONS = ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FOLDS_CSV = ROOT / "data" / "folds.csv"


def time_blocks(stems: pd.DataFrame, gap_hours: float = GAP_HOURS) -> pd.Series:
    """Block id per stem: consecutive observations closer than gap_hours share a block."""
    s = stems.sort_values("datetime")
    gaps = s["datetime"].diff().dt.total_seconds().div(3600).fillna(np.inf)
    block = (gaps > gap_hours).cumsum()
    return pd.Series(block.values, index=s["stem"].values, name="block")


def assign_folds(block_of: pd.Series, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.Series:
    """Greedy balanced assignment of blocks to folds (ties broken by a seeded shuffle)."""
    rng = np.random.default_rng(seed)
    sizes = block_of.value_counts()
    order = sizes.sample(frac=1, random_state=int(rng.integers(0, 2**31))).sort_values(ascending=False, kind="stable")
    load = np.zeros(n_folds, dtype=int)
    fold_of_block: dict[int, int] = {}
    for block, size in order.items():
        f = int(np.argmin(load))
        fold_of_block[block] = f
        load[f] += size
    return block_of.map(fold_of_block).rename("fold")


def make_folds(gap_hours: float = GAP_HOURS, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    records, _ = load_coco(ANNOTATIONS)
    stems = records.drop_duplicates("stem")[["stem", "datetime", "observatory"]].copy()
    block_of = time_blocks(stems, gap_hours)
    fold_of = assign_folds(block_of, n_folds, seed)
    df = stems.set_index("stem").join(block_of).join(fold_of).reset_index()
    df["year"] = df["datetime"].dt.year
    return df.sort_values("stem").reset_index(drop=True)


def balance_report(df: pd.DataFrame, records: pd.DataFrame) -> str:
    lines = [f"blocks={df['block'].nunique()} max_block={df['block'].value_counts().max()}"]
    per = df.groupby("fold").agg(stems=("stem", "size"), blocks=("block", "nunique"))
    per["records"] = records.groupby(records["stem"].map(df.set_index("stem")["fold"])).size()
    lines.append(per.to_string())
    lines.append(pd.crosstab(df["fold"], df["observatory"]).to_string())
    lines.append(pd.crosstab(df["fold"], df["year"]).to_string())
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gap-hours", type=float, default=GAP_HOURS)
    args = ap.parse_args()
    df = make_folds(gap_hours=args.gap_hours)
    records, _ = load_coco(ANNOTATIONS)
    print(balance_report(df, records))
    if args.dry_run:
        return
    df[["stem", "fold", "block"]].to_csv(FOLDS_CSV, index=False, lineterminator="\n")
    print(f"wrote {FOLDS_CSV} ({len(df)} stems)")


if __name__ == "__main__":
    main()
