"""src/metric.py must reproduce the host's self-evaluation notebook (tests/host_reference.py) and hand-computed cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.annotations import mask_to_rle, polygon_to_rle, rle_to_mask
from src.metric import evaluate
from tests import host_reference as host

H = W = 2048


def rect(x0: int, y0: int, x1: int, y1: int) -> str:
    m = np.zeros((H, W), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return mask_to_rle(m)


def df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def test_perfect_match_is_one():
    gt = df([("a1-img_1", rect(0, 0, 100, 100)), ("a1-img_2", rect(200, 200, 300, 250))])
    pred = df([("img_1", rect(0, 0, 100, 100)), ("img_2", rect(200, 200, 300, 250))])
    r = evaluate(gt, pred)
    assert r.pq == pytest.approx(1.0) and r.tp == 2 and r.fp == 0 and r.fn == 0


def test_hand_computed_mixed_case():
    # gt A: 100x100 at origin; pred A': 100x100 shifted by 20 px -> inter 80*100=8000, union 12000, IoU 2/3 (TP)
    # gt B: 50x50; pred B': overlaps half -> inter 25*50=1250, union 3750, IoU 1/3 (< 0.5 -> FN + FP)
    # pred C: no overlap at all -> FP
    gt = df([("a1-img_1", rect(0, 0, 100, 100)), ("a1-img_2", rect(500, 500, 550, 550))])
    pred = df([("img_1", rect(20, 0, 120, 100)), ("img_2", rect(525, 500, 575, 550)), ("img_3", rect(1000, 1000, 1010, 1010))])
    r = evaluate(gt, pred)
    assert (r.tp, r.fp, r.fn) == (1, 2, 1)
    assert r.pq == pytest.approx((2 / 3) / (1 + 0.5 * 2 + 0.5 * 1))
    assert r.sq == pytest.approx(2 / 3) and r.rq == pytest.approx(1 / 2.5)


def test_iou_exactly_half_is_not_a_hit():
    # gt 100x100, pred 100x200 containing it: IoU = 10000/20000 = 0.5 exactly -> strict '>' means no TP
    gt = df([("a1-img_1", rect(0, 0, 100, 100))])
    pred = df([("img_1", rect(0, 0, 100, 200))])
    r = evaluate(gt, pred)
    assert (r.tp, r.fp, r.fn) == (0, 1, 1) and r.pq == 0.0


def test_each_annotator_record_counted_separately():
    m = rect(0, 0, 100, 100)
    gt = df([("a1-img_1", m), ("a2-img_1", m), ("a3-img_1", m), ("a3-img_2", rect(300, 300, 400, 400))])
    pred = df([("img_1", m)])
    r = evaluate(gt, pred)
    assert (r.tp, r.fp, r.fn) == (3, 0, 1)  # the same prediction is matched three times
    assert r.pq == pytest.approx(3 / 3.5)


def test_empty_sides():
    gt = df([("a1-img_1", rect(0, 0, 10, 10)), ("a1-other_1", rect(0, 0, 10, 10))])
    pred = df([("img_1", rect(0, 0, 10, 10)), ("nogt_1", rect(0, 0, 10, 10))])  # 'nogt' has no gt record -> ignored
    r = evaluate(gt, pred)
    assert (r.tp, r.fp, r.fn) == (1, 0, 1)
    assert evaluate(gt, df([])).pq == 0.0


def test_matches_host_reference_on_random_shapes():
    rng = np.random.default_rng(0)
    gt_rows, pred_rows = [], []
    for img in range(6):
        for ann in ["a1", "a2"][: 1 + img % 2]:
            for k in range(rng.integers(0, 5)):
                x, y = rng.integers(0, 1800, 2)
                w, h = rng.integers(20, 200, 2)
                gt_rows.append((f"{ann}-img{img}_{k + 1}", rect(x, y, x + w, y + h)))
        for k in range(rng.integers(0, 6)):
            x, y = rng.integers(0, 1800, 2)
            w, h = rng.integers(20, 200, 2)
            pred_rows.append((f"img{img}_{k + 1}", rect(x, y, x + w, y + h)))
    # plus near-duplicates of some gt so that there are real TPs
    for fid, rle in gt_rows[::3]:
        m = rle_to_mask(rle)
        m = np.roll(m, 5, axis=1)
        pred_rows.append((fid.split("-", 1)[1].rsplit("_", 1)[0] + f"_{100 + len(pred_rows)}", mask_to_rle(m)))
    gt, pred = df(gt_rows), df(pred_rows)
    ours = evaluate(gt, pred)
    theirs = host.get_pq_score(host.get_overlap_df(gt, pred))
    assert ours.pq == pytest.approx(theirs, abs=1e-6)
    assert ours.tp > 0 and ours.fp > 0 and ours.fn > 0


def test_polygon_roundtrip():
    poly = np.array([[10, 10], [110, 10], [110, 60], [10, 60]], dtype=np.float32)
    m = rle_to_mask(polygon_to_rle(poly))
    assert m.sum() == pytest.approx(100 * 50, rel=0.05)
