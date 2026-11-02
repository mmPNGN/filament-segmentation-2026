"""Shared paths, loaders and plotting style for the technical-report figures (research/report/figs/).

TRAIN / OOF data only: nothing here reads test images, test probability maps or scorer_test.parquet.
Matching follows src/metric.py exactly (IoU > 0.5 per annotator record, micro pooling); IoU matrices come from
pycocotools on the RLE strings (identical to metric.overlap_matrices on decoded masks) and Dice = 2 IoU / (1 + IoU),
which is the exact per-pair identity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.metric import RecordOverlap  # noqa: E402
from src.preprocess import cache_key  # noqa: E402

EXP = "exp118"
MAPS = "ens_dml_rx_b4_cnx43_hr43"
TAU = 0.20
SIZE = 2048
OUT = ROOT / "research" / "report" / "figs"
CACHE = Path(__file__).resolve().parent / "_cache"
GT_PATH = ROOT / "data" / "processed" / "gt_rles.parquet"
SCORER_OOF = ROOT / "outputs" / f"{EXP}-ensemble" / "scorer_oof.parquet"
PROB_DIR = ROOT / "outputs" / f"{MAPS}-full" / "oof"
TRAIN_IMG = ROOT / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"

# Okabe-Ito (colour-blind safe). Fixed roles across all figures.
C_OURS = "#0072B2"     # blue: our model
C_HUMAN = "#E69F00"    # orange: human vs human
C_GT = ["#E69F00", "#56B4E9", "#CC79A7"]  # annotator records 1..3 (outlines)
C_KEPT = "#009E73"     # kept instances (outlines)
C_DROP = "#E8E8E8"     # candidates dropped by the gate
C_REF = "#555555"      # reference lines / text
INK = "#222222"

SINGLE_W = 8.5 / 2.54  # inches
DOUBLE_W = 17.5 / 2.54


def setup_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.edgecolor": "#444444", "axes.labelcolor": INK, "xtick.color": "#444444", "ytick.color": "#444444",
        "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
        "axes.grid": False, "lines.linewidth": 1.2,
    })


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"wrote {OUT / name}.pdf / .png")


def rle_obj(r: str) -> dict:
    return {"size": [SIZE, SIZE], "counts": r.encode("ascii")}


def decode(r: str) -> np.ndarray:
    return mask_util.decode(rle_obj(r))


def iou_matrix(gt_rles: list[str], pred_rles: list[str]) -> np.ndarray:
    """(n_gt, n_pred) IoU, exact pixel counts (pycocotools, iscrowd = 0)."""
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)))
    m = mask_util.iou([rle_obj(r) for r in pred_rles], [rle_obj(r) for r in gt_rles], [0] * len(gt_rles))
    return np.asarray(m, dtype=np.float64).reshape(len(pred_rles), len(gt_rles)).T


def record_overlap(record_id: str, gt_rles: list[str], pred_rles: list[str]) -> RecordOverlap:
    iou = iou_matrix(gt_rles, pred_rles)
    return RecordOverlap(record_id, len(gt_rles), len(pred_rles), iou, 2 * iou / (1 + iou))


def load_gt() -> dict[str, dict[str, list[str]]]:
    from src.scorer import gt_records_by_stem

    return gt_records_by_stem(pd.read_parquet(GT_PATH))


def load_candidates() -> pd.DataFrame:
    return pd.read_parquet(SCORER_OOF, columns=["stem", "comp", "rle", "u", "u_hat", "area", "p_mean", "fold"])


def load_flat(stem: str) -> np.ndarray:
    """Radial-median flattened input (the network's only channel), float32 2048x2048, from the training cache."""
    import yaml

    cfg = yaml.safe_load((ROOT / "outputs" / f"{EXP}-ensemble" / "config_resolved.yaml").read_text(encoding="utf-8"))
    return np.load(ROOT / cfg["cache_dir"] / cache_key(cfg["input"]) / "x" / f"{stem}.npy")[0].astype(np.float32)


def load_raw(stem: str) -> np.ndarray:
    from src.preprocess import read_gray

    return read_gray(TRAIN_IMG / f"{stem}.jpeg")


def load_prob(stem: str) -> np.ndarray:
    return np.load(PROB_DIR / f"{stem}.npy").astype(np.float32) / 255.0


def load_stats() -> dict:
    import pickle

    with open(CACHE / "stats.pkl", "rb") as f:
        return pickle.load(f)


def write_numbers(key: str, value: dict) -> None:
    """Merge `value` into research/report/figs/numbers.json under `key` (quotable numbers for the report)."""
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "numbers.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d[key] = value
    p.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
