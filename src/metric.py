"""Panoptic Quality exactly as the host's self-evaluation notebook computes it (competition.md §2).

Reference: `research/intel/raw/2026-09-27/kernels/azimahmadzadeh__self-evaluation-notebook/` (copied verbatim into
`tests/host_reference.py`; `tests/test_metric.py` asserts that this module reproduces it).

Definition
    * ground truth is per (annotator, image) record: gt filament_id = "<annotator>-<stem>_<k>"
    * predictions are per image:                     pred filament_id = "<stem>_<k>"
    * every record is matched separately against the image's predictions (3 annotators -> counted 3 times)
    * TP = every (gt, pred) pair with IoU > 0.5 (strict); FP = predictions with no TP; FN = gt with no TP
    * PQ = sum(IoU of TP) / (|TP| + 0.5 |FP| + 0.5 |FN|), micro-pooled over all records
    * a record with no gt makes all its predictions FP; a record with no predictions makes all its gt FN

This file is a measuring instrument: changing it needs user approval (autoresearch.md §3).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

IOU_THRESHOLD = 0.5
IMAGE_SIZE = 2048


def rles_to_layers(rles: list[str], height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> np.ndarray:
    """Decode compressed RLE strings to an (n, H, W) uint8 stack (empty (0, H, W) for no masks)."""
    if not rles:
        return np.zeros((0, height, width), dtype=np.uint8)
    masks = mask_util.decode([{"size": [height, width], "counts": r.encode("ascii")} for r in rles])
    return masks.transpose(2, 0, 1)


def overlap_matrices(gt_layers: np.ndarray, pred_layers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise (n_gt, n_pred) IoU and Dice. Exact integer pixel counts, float64 ratios; 0 where union is 0."""
    n_gt, n_pred = gt_layers.shape[0], pred_layers.shape[0]
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred)), np.zeros((n_gt, n_pred))
    gt_flat = gt_layers.reshape(n_gt, -1).astype(np.float32)
    pred_flat = pred_layers.reshape(n_pred, -1).astype(np.float32)
    inter = (gt_flat @ pred_flat.T).astype(np.float64)  # counts < 2**24 are exact in float32
    gt_area = gt_flat.sum(axis=1, dtype=np.float64)[:, None]
    pred_area = pred_flat.sum(axis=1, dtype=np.float64)[None, :]
    union = gt_area + pred_area - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union == 0, 0.0, inter / union)
        dice = np.where(union == 0, 0.0, 2 * inter / (gt_area + pred_area))
    return iou, dice


@dataclass
class RecordOverlap:
    record_id: str
    n_gt: int
    n_pred: int
    iou: np.ndarray  # (n_gt, n_pred)
    dice: np.ndarray


@dataclass
class PQResult:
    pq: float
    sq: float  # mean IoU of TP pairs
    rq: float  # TP / (TP + 0.5 FP + 0.5 FN)
    tp: int
    fp: int
    fn: int
    tp_ious: list[float] = field(default_factory=list)
    one_to_many: Counter = field(default_factory=Counter)  # gt -> #preds with IoU>0 (host's plot_m2n_counts)
    many_to_one: Counter = field(default_factory=Counter)  # pred -> #gts with IoU>0

    def as_metrics(self) -> dict[str, float]:
        return {"pq": self.pq, "sq": self.sq, "rq": self.rq, "tp": self.tp, "fp": self.fp, "fn": self.fn}


def _ids_to_image(filament_ids: pd.Series) -> pd.Series:
    """'<prefix>_<k>' -> '<prefix>' (record id for gt, stem for predictions)."""
    return filament_ids.str.split("_", n=1).str[0]


def compute_overlaps(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> list[RecordOverlap]:
    """One RecordOverlap per gt record, matching the record's gt against the predictions for its image."""
    gt_rec = _ids_to_image(gt_df["filament_id"])
    pred_img = _ids_to_image(pred_df["filament_id"])
    gt_groups = {k: v.tolist() for k, v in gt_df["segmentation_rle"].groupby(gt_rec, sort=False)}
    pred_groups = {k: v.tolist() for k, v in pred_df["segmentation_rle"].groupby(pred_img, sort=False)}
    out = []
    for record_id in gt_rec.unique():
        stem = record_id.split("-", 1)[1]
        gt_rles = gt_groups[record_id]
        pred_rles = pred_groups.get(stem, [])
        iou, dice = overlap_matrices(rles_to_layers(gt_rles), rles_to_layers(pred_rles))
        out.append(RecordOverlap(record_id, len(gt_rles), len(pred_rles), iou, dice))
    return out


def panoptic_quality(overlaps: list[RecordOverlap], iou_threshold: float = IOU_THRESHOLD) -> PQResult:
    tp_ious: list[float] = []
    fp = fn = 0
    one_to_many: Counter = Counter()
    many_to_one: Counter = Counter()
    for r in overlaps:
        if r.n_gt == 0:
            fp += r.n_pred
            many_to_one.update([0] * r.n_pred)
            continue
        if r.n_pred == 0:
            fn += r.n_gt
            one_to_many.update([0] * r.n_gt)
            continue
        hit = r.iou > iou_threshold
        tp_ious.extend(r.iou[hit].tolist())
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())
        touch = r.iou > 0
        one_to_many.update(touch.sum(axis=1).tolist())
        many_to_one.update(touch.sum(axis=0).tolist())
    tp = len(tp_ious)
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum(tp_ious) / denom if denom > 0 else 0.0
    sq = float(np.mean(tp_ious)) if tp else 0.0
    rq = tp / denom if denom > 0 else 0.0
    return PQResult(pq, sq, rq, tp, fp, fn, tp_ious, one_to_many, many_to_one)


def evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> PQResult:
    """Official score for a (gt, prediction) pair of dataframes with columns filament_id, segmentation_rle."""
    return panoptic_quality(compute_overlaps(gt_df, pred_df))
