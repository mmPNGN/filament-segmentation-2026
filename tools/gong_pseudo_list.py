"""Freeze the list of unlabelled GONG frames for pseudo-labelling, with a LABEL-FREE, image-only quality filter
(leak audit 2026-10-03: no model prediction may decide which frames enter, because the list is shared by all folds).

Input: the downloader's manifest.csv (a row is written only after the JPEG is complete) and the JPEGs.
Per frame: analysis/image_stats.image_stats (disk radius / centre, disk mean / std, flat std, sharpness, noise).
keep = 6 competition sites, and every statistic inside the [0.5 %, 99.5 %] range of the 887 competition images'
values in analysis/image_stats.csv (clouds, bad seeing, mis-centred or partial disks fall out).
Writes <out_dir>/pseudo_list_<n>.csv (stem, keep, stats). Usage:
    uv run python tools/gong_pseudo_list.py --dir data/external/gong_ha_2023_2026 [--workers 4]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.image_stats import image_stats  # noqa: E402

STATS = ["disk_r", "disk_cx", "disk_cy", "disk_mean", "disk_std", "flat_std", "lap_var", "hf_noise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/external/gong_ha_2023_2026")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = ROOT / a.dir
    man = pd.read_csv(d / "manifest.csv").drop_duplicates("stem")
    man = man[man["site"].isin(list("BCLMTU"))]
    jobs = [(s, str(d / "jpg" / f"{s}.jpg"), None) for s in man["stem"] if (d / "jpg" / f"{s}.jpg").exists()]
    with ProcessPoolExecutor(a.workers) as pool:
        df = pd.DataFrame(list(pool.map(image_stats, jobs, chunksize=8)))
    ref = pd.read_csv(ROOT / "analysis" / "image_stats.csv")
    keep = pd.Series(True, index=df.index)
    for c in STATS:
        lo, hi = ref[c].quantile([0.005, 0.995])
        ok = df[c].between(lo, hi)
        print(f"{c}: competition [{lo:.4g}, {hi:.4g}]  pass {ok.mean():.3f}")
        keep &= ok
    df["keep"] = keep
    out = d / f"pseudo_list_{len(df)}.csv"
    df.to_csv(out, index=False)
    print(f"{out}: {int(keep.sum())} / {len(df)} kept")


if __name__ == "__main__":
    main()
