"""Matching statistics for the report figures (exp118 5-fold OOF vs human-vs-human), cached for the figure scripts.

ours  : kept instances of exp118 (nested GBDT u_hat > 0.20) matched against every annotator record of all 707 train
        images, exactly as src/metric.panoptic_quality (IoU > 0.5, per record, micro pooled).
human : on the 296 multi-record images, every ordered pair (record A as prediction, record B as ground truth), as in
        research/logs/2026-09-29_eda.md §6 (PQ 0.3399, 1,196 ordered pairs).
gate  : IoU of every candidate component (before the gate) with every record -> pooled S / TP / FP / FN per tau.

Output: analysis/report_figs/_cache/stats.pkl and research/report/figs/numbers.json ("matching").
Usage: uv run python analysis/report_figs/compute_stats.py   (CPU, about 1 min)
"""
from __future__ import annotations

import pickle
from collections import Counter

import numpy as np
import pandas as pd

from common import CACHE, TAU, load_candidates, load_gt, record_overlap, write_numbers
from src.metric import RecordOverlap, panoptic_quality

TAUS = np.round(np.arange(0.0, 0.601, 0.01), 2)


def per_gt_best_dice(ovs: list[RecordOverlap]) -> np.ndarray:
    """Best Dice of each gt filament against any prediction (0 when nothing overlaps)."""
    return np.concatenate([o.dice.max(axis=1) if o.n_pred else np.zeros(o.n_gt) for o in ovs if o.n_gt])


def counter_shares(c: Counter) -> dict[str, float]:
    n = sum(c.values())
    return {"0": c[0] / n, "1": c[1] / n, "2": c[2] / n, "3+": sum(v for k, v in c.items() if k >= 3) / n, "n": n}


def summary(res, best_dice: np.ndarray) -> dict:
    ious = np.asarray(res.tp_ious)
    dices = 2 * ious / (1 + ious)
    return {"pq": res.pq, "sq": res.sq, "rq": res.rq, "tp": res.tp, "fp": res.fp, "fn": res.fn,
            "tp_iou_q10_50_90": np.quantile(ious, [0.1, 0.5, 0.9]).round(4).tolist(),
            "tp_dice_q10_50_90": np.quantile(dices, [0.1, 0.5, 0.9]).round(4).tolist(),
            "gt_best_dice_median": float(np.median(best_dice)),
            "gt_best_dice_share_ge_0667": float((best_dice > 2 / 3).mean()),
            "gt_best_dice_share_zero": float((best_dice == 0).mean()),
            "one_to_many": counter_shares(res.one_to_many), "many_to_one": counter_shares(res.many_to_one)}


def main() -> None:
    gt = load_gt()
    cand = load_candidates()
    kept = cand[cand.u_hat > TAU]
    kept_by = {s: g["rle"].tolist() for s, g in kept.groupby("stem")}
    cand_by = {s: (g["rle"].tolist(), g["u_hat"].to_numpy()) for s, g in cand.groupby("stem")}

    ours, human, gate, rows = [], [], [], []
    for stem, recs in gt.items():
        o_stem = [record_overlap(rid, r, kept_by.get(stem, [])) for rid, r in recs.items()]
        ours += o_stem
        c_rles, c_uh = cand_by.get(stem, ([], np.zeros(0)))
        gate += [(record_overlap(rid, r, c_rles).iou, c_uh) for rid, r in recs.items()]
        row = {"stem": stem, "n_records": len(recs), "n_gt": sum(len(v) for v in recs.values()),
               "n_kept": len(kept_by.get(stem, [])), "pq_ours": panoptic_quality(o_stem).pq, "pq_human": np.nan}
        if len(recs) >= 2:
            pairs = [record_overlap(f"{g}|{p}", recs[g], recs[p]) for g in recs for p in recs if g != p]
            human += pairs
            row["pq_human"] = panoptic_quality(pairs).pq
        rows.append(row)

    multi = {s for s, r in gt.items() if len(r) >= 2}
    ours_multi = [o for o in ours if o.record_id.split("-", 1)[1] in multi]
    res = {k: panoptic_quality(v) for k, v in {"ours": ours, "human": human, "ours_multi": ours_multi}.items()}
    best = {"ours": per_gt_best_dice(ours), "human": per_gt_best_dice(human), "ours_multi": per_gt_best_dice(ours_multi)}

    curve = []
    for t in TAUS:
        s = tp = fp = fn = 0
        for m, uh in gate:
            m = m[:, uh > t]
            hit = m > 0.5
            s += float(m[hit].sum()); tp += int(hit.sum())
            fp += int((hit.sum(axis=0) == 0).sum()); fn += int((hit.sum(axis=1) == 0).sum())
        curve.append({"tau": t, "S": s, "TP": tp, "FP": fp, "FN": fn, "PQ": s / (tp + 0.5 * fp + 0.5 * fn),
                      "n_kept": int((cand.u_hat > t).sum())})
    curve = pd.DataFrame(curve)

    stats = {"res": res, "best_dice": best, "per_image": pd.DataFrame(rows), "gate_curve": curve,
             "calib": cand[["u", "u_hat", "fold"]].copy(), "n_candidates": len(cand), "n_kept": len(kept),
             "n_records": len(ours), "n_pairs": len(human), "n_multi_images": len(multi)}
    CACHE.mkdir(exist_ok=True)
    with open(CACHE / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    nums = {k: summary(res[k], best[k]) for k in res}
    nums["counts"] = {"records": len(ours), "ordered_pairs": len(human), "multi_images": len(multi),
                      "candidates": len(cand), "kept": len(kept)}
    write_numbers("matching", nums)
    for k, r in res.items():
        print(f"{k:10s} PQ {r.pq:.6f} SQ {r.sq:.4f} RQ {r.rq:.4f} TP/FP/FN {r.tp}/{r.fp}/{r.fn} "
              f"TP-IoU med {np.median(r.tp_ious):.4f} 1:N {dict(sorted(r.one_to_many.items()))} N:1 {dict(sorted(r.many_to_one.items()))}")
    print(f"records {len(ours)}, ordered pairs {len(human)}, multi-record images {len(multi)}")
    print(curve.loc[curve.tau.isin([0.0, 0.1, 0.15, 0.2, 0.22, 0.25, 0.3, 0.4, 0.5])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
