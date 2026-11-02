"""Component-level expected-IoU scorer (GBDT): replaces the fixed score_min gate in post-processing.

For every candidate instance (connected component after merge, score_min 0) a LightGBM regressor predicts
u = mean over the image's annotator records of (IoU if IoU > 0.5 else 0), i.e. the component's expected contribution
to the PQ numerator per record; a component is kept when u_hat > tau (tau ~ PQ/2, protocol: fixed at 0.20 before the
experiment). Prototype and evaluation: analysis/scorer_gbdt.py, research/logs/2026-09-30_gbdt_scorer.md.

Leak rules: the OOF instances of fold k are scored by a model trained on the components of the other folds only
(the same folds.csv as the segmentation model); the test instances are scored by a model trained on all OOF components.
Features use only what is available at test time (probability map, geometry, the flat input channel); never the
number of annotator records.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from src.annotations import IMAGE_SIZE
from src.data import Cache
from src.postprocess import _ellipse, instances_from_prob, upsample_prob

IOU_T = 0.5
OBS_CODE = {c: i for i, c in enumerate("BCLMTU")}
LGB_PARAMS = dict(objective="regression", n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
                  deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
NON_FEATURES = {"stem", "comp", "u", "n_tp_records", "n_records", "max_iou_any", "rle", "fold", "u_hat", "any_tp", "score_gate"}


def _moments_shape(sub: np.ndarray) -> tuple[float, float, float]:
    """elongation sqrt(l1/l2) and centroid (x, y) inside the crop from central moments."""
    m = cv2.moments(sub, binaryImage=True)
    mu20, mu02, mu11 = m["mu20"], m["mu02"], m["mu11"]
    tr, det = mu20 + mu02, mu20 * mu02 - mu11**2
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
    return float(np.sqrt(l1 / l2)), m["m10"] / max(m["m00"], 1e-6), m["m01"] / max(m["m00"], 1e-6)


def component_features(i: int, mask: np.ndarray, p: np.ndarray, flat: np.ndarray, labels: np.ndarray, rmap: np.ndarray) -> dict:
    """37 test-time features of one component (identical to analysis/scorer_gbdt.py)."""
    x, y, w, h = cv2.boundingRect(mask)
    pad = 48
    x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, IMAGE_SIZE), min(y + h + pad, IMAGE_SIZE)
    sub = mask[y0:y1, x0:x1]
    mb = sub.astype(bool)
    pv, fv = p[y0:y1, x0:x1][mb], flat[y0:y1, x0:x1][mb]
    ring = cv2.dilate(sub, _ellipse(8)).astype(bool) & ~mb
    p_ring, f_ring = p[y0:y1, x0:x1][ring], flat[y0:y1, x0:x1][ring]
    lsub = labels[y0:y1, x0:x1]
    near24 = np.unique(lsub[cv2.dilate(sub, _ellipse(24)).astype(bool)])
    near48 = np.unique(lsub[cv2.dilate(sub, _ellipse(48)).astype(bool)])
    elong, cx, cy = _moments_shape(sub)
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    hull_area = float(cv2.contourArea(cv2.convexHull(np.concatenate(contours)))) if contours else 0.0
    area = int(mb.sum())
    f_q05 = float(np.quantile(fv, 0.05))
    return {
        "area": area, "log_area": float(np.log10(area)),
        "p_mean": float(pv.mean()), "p_max": float(pv.max()), "p_min": float(pv.min()), "p_std": float(pv.std()),
        "p_q10": float(np.quantile(pv, 0.10)), "p_frac_gt09": float((pv > 0.9).mean()),
        "p_frac_mid": float(((pv >= 0.3) & (pv <= 0.7)).mean()), "p_ring_mean": float(p_ring.mean()) if ring.any() else 0.0,
        "rr_centroid": float(rmap[min(int(round(cy)) + y0, IMAGE_SIZE - 1), min(int(round(cx)) + x0, IMAGE_SIZE - 1)]),
        "rr_max": float(rmap[y0:y1, x0:x1][mb].max()),
        "bbox_w": int(w), "bbox_h": int(h), "bbox_aspect": float(max(w, h) / max(min(w, h), 1)),
        "bbox_fill": float(area / max(w * h, 1)), "elongation": elong,
        "perimeter": perimeter, "perim_over_area": float(perimeter / max(area, 1)),
        "perim_over_sqrt_area": float(perimeter / np.sqrt(max(area, 1))),
        "solidity": float(area / hull_area) if hull_area > 0 else 1.0,
        "n_neigh_24": int(len(set(near24.tolist()) - {0, i})), "n_neigh_48": int(len(set(near48.tolist()) - {0, i})),
        "f_mean": float(fv.mean()), "f_min": float(fv.min()), "f_q05": f_q05, "f_std": float(fv.std()),
        "f_ring_mean": float(f_ring.mean()) if ring.any() else 1.0,
        "f_contrast_mean": float(f_ring.mean() - fv.mean()) if ring.any() else 0.0,
        "f_contrast_q05": float(f_ring.mean() - f_q05) if ring.any() else 0.0,
    }


# ------------------------------------------------------------------ per-image extraction (process pool)
_CACHE: Cache | None = None
_PP: dict = {}


def _init(cache_root, input_cfg, pp):
    global _CACHE, _PP
    _CACHE = Cache(cache_root, input_cfg)
    _PP = pp
    cv2.setNumThreads(1)


def _extract(args: tuple[str, np.ndarray, dict | None]) -> list[dict]:
    """Components of one image with features, RLE and (when gt records are given) the target u."""
    stem, prob, gt_records = args
    rmap = _CACHE.disk(stem).radius_map(IMAGE_SIZE)
    disk_mask = (rmap <= _PP["disk_margin"]).astype(np.uint8)
    inst = instances_from_prob(prob, _PP["threshold"], _PP["min_area"], _PP["closing_radius"], disk_mask,
                               merge_radius=_PP.get("merge_radius", 0), score_min=0.0,
                               seed_threshold=_PP.get("seed_threshold", 0.0))
    if not inst:
        return []
    p = upsample_prob(prob)
    flat = cv2.resize(np.load(_CACHE.x_path(stem))[0].astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    labels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
    for i, d in enumerate(inst, start=1):
        labels[d["mask"] > 0] = i
    order = np.argsort([-d["score"] for d in inst], kind="stable")
    rank = np.empty(len(inst), dtype=int)
    rank[order] = np.arange(1, len(inst) + 1)
    pred_rles = [mask_util.encode(np.asfortranarray(d["mask"])) for d in inst]
    u = np.zeros(len(inst))
    if gt_records:
        for gt_rles in gt_records.values():
            gt_objs = [{"size": [IMAGE_SIZE, IMAGE_SIZE], "counts": r.encode("ascii")} for r in gt_rles]
            iou = np.asarray(mask_util.iou(pred_rles, gt_objs, [0] * len(gt_objs)))
            u += np.where(iou > IOU_T, iou, 0.0).sum(axis=1)
        u /= len(gt_records)
    total_area = float(sum(d["area"] for d in inst))
    rows = []
    for i, d in enumerate(inst):
        rows.append({"stem": stem, "comp": i, "score": d["score"], "rank": int(rank[i]), "rank_frac": float(rank[i] / len(inst)),
                     "n_comp": len(inst), "img_total_area": total_area, "area_share": float(d["area"] / total_area),
                     "obs": OBS_CODE.get(stem[14], -1), **component_features(i + 1, d["mask"], p, flat, labels, rmap),
                     "u": float(u[i]), "rle": pred_rles[i]["counts"].decode("ascii")})
    return rows


def extract_components(probs: dict[str, np.ndarray], cache: Cache, pp: dict, gt_by_stem: dict | None,
                       workers: int = 4) -> pd.DataFrame:
    jobs = [(s, p, (gt_by_stem or {}).get(s)) for s, p in probs.items()]
    with ProcessPoolExecutor(max(workers, 1), initializer=_init, initargs=(cache.dir.parent, cache.input_cfg, pp)) as pool:
        rows = [r for chunk in pool.map(_extract, jobs, chunksize=4) for r in chunk]
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def gt_records_by_stem(gt: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """gt rows (filament_id '<annotator>-<stem>_<k>') -> {stem: {record_id: [rle, ...]}}."""
    rec = gt["filament_id"].str.rsplit("_", n=1).str[0]
    out: dict[str, dict[str, list[str]]] = {}
    for record_id, rles in gt["segmentation_rle"].groupby(rec, sort=False):
        out.setdefault(record_id.split("-", 1)[1], {})[record_id] = rles.tolist()
    return out


def fit(df: pd.DataFrame, feats: list[str], seed: int = 42):
    import lightgbm as lgb

    return lgb.LGBMRegressor(**{**LGB_PARAMS, "random_state": seed}).fit(df[feats], df["u"])


def score_oof(df: pd.DataFrame, fold_of: dict[str, int], seeds: tuple[int, ...] = (42,)) -> np.ndarray:
    """Nested: u_hat for the components of fold k from a model fitted on the other folds; mean over LightGBM `seeds`
    (postprocess.scorer_seeds: bagging / feature subsampling make one fit seed-dependent, sd ~0.0004 PQ on exp118)."""
    feats = feature_columns(df)
    folds = df["stem"].map(fold_of).values
    u_hat = np.zeros(len(df))
    for f in sorted(set(folds)):
        va = folds == f
        u_hat[va] = np.mean([fit(df[~va], feats, s).predict(df.loc[va, feats]) for s in seeds], axis=0)
    return u_hat


def score_test(oof_df: pd.DataFrame, test_df: pd.DataFrame, seeds: tuple[int, ...] = (42,)) -> np.ndarray:
    feats = feature_columns(oof_df)
    return np.mean([fit(oof_df, feats, s).predict(test_df[feats]) for s in seeds], axis=0)


def rows_from_components(df: pd.DataFrame, keep: np.ndarray) -> list[dict]:
    """Submission rows ('<stem>_<k>', k by descending u_hat) for the kept components."""
    rows = []
    for stem, g in df[keep].groupby("stem", sort=False):
        g = g.sort_values("u_hat", ascending=False)
        for k, r in enumerate(g.itertuples(), start=1):
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": r.rle, "score": float(r.u_hat), "area": int(r.area)})
    return rows
