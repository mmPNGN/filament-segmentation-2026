"""fig_qualitative: three OOF train examples (good / annotator disagreement / failure).

Each row: flattened input crop | ground truth of every annotator record (coloured outlines) | our candidates
(kept: green outline with u_hat; dropped by the gate: grey dashed). Predictions are the exp118 5-fold OOF, i.e. every
image is predicted by models that did not see it. TRAIN images only.

Selection rule (automatic, from analysis/report_figs/_cache/stats.pkl, per-image PQ = the official metric pooled over
the records of that image):
  eligible   = images with >= MIN_GT_PER_REC ground-truth filaments per record (enough structure to show)
  good       = eligible, >= 2 records, highest PQ(ours)
  disagree   = eligible, >= 2 records, lowest human-vs-human PQ (ordered record pairs of that image)
  failure    = eligible, lowest PQ(ours), not one of the two above
Crop: a CROP x CROP window (stride 32, fully inside the frame) that contains the most centroids of the row's focus
instances; ties broken by the GT pixel area inside the window. Focus instances (match = IoU > 0.5, as in the metric):
  good       all GT filaments of all records + all kept predictions
  disagree   GT filaments that no other record of the image matches (drawn by one annotator only, or drawn with a
             different extent)
  failure    errors: GT filaments matched by no kept prediction (FN) + kept predictions matching no record (FP)

Usage: uv run python analysis/report_figs/fig_qualitative.py
"""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D

from common import (C_DROP, C_GT, C_KEPT, DOUBLE_W, SIZE, TAU, decode, iou_matrix, load_candidates, load_flat, load_gt,
                    load_stats, save, setup_style, write_numbers)

MIN_GT_PER_REC = 6
CROP = 704
STRIDE = 32


def select(per_image) -> dict[str, str]:
    el = per_image[per_image.n_gt / per_image.n_records >= MIN_GT_PER_REC]
    multi = el[el.n_records >= 2]
    good = multi.sort_values("pq_ours", ascending=False).stem.iloc[0]
    disagree = multi[multi.stem != good].sort_values("pq_human").stem.iloc[0]
    fail = el[~el.stem.isin([good, disagree])].sort_values("pq_ours").stem.iloc[0]
    return {"good": good, "disagree": disagree, "failure": fail}


def centroid(m: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


def focus(kind: str, recs: dict[str, list[str]], kept_rles: list[str]) -> tuple[list[tuple[str, int]], list[int]]:
    """(record_id, k) of focus GT filaments and indices of focus kept predictions (see module docstring)."""
    if kind == "good":
        return [(r, k) for r, v in recs.items() for k in range(len(v))], list(range(len(kept_rles)))
    gts, pred_hit = [], np.zeros(len(kept_rles), bool)
    for r, v in recs.items():
        if kind == "disagree":
            other = [iou_matrix(v, recs[o]).max(axis=1, initial=0) for o in recs if o != r]
            gts += [(r, k) for k in np.flatnonzero(np.max(other, axis=0) <= 0.5)]
        else:
            m = iou_matrix(v, kept_rles)
            gts += [(r, k) for k in np.flatnonzero(m.max(axis=1, initial=0) <= 0.5)]
            pred_hit |= (m > 0.5).any(axis=0)
    return gts, ([] if kind == "disagree" else list(np.flatnonzero(~pred_hit)))


def choose_crop(focus_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> tuple[int, int]:
    pts = [centroid(m) for m in focus_masks]
    gt_area = np.zeros((SIZE, SIZE), np.float32)
    for m in gt_masks:
        gt_area += m
    integ = cv2.integral(gt_area)
    best, best_key = (0, 0), (-1, -1.0)
    for y0 in range(0, SIZE - CROP + 1, STRIDE):
        for x0 in range(0, SIZE - CROP + 1, STRIDE):
            n = sum(x0 <= x < x0 + CROP and y0 <= y < y0 + CROP for x, y in pts)
            a = integ[y0 + CROP, x0 + CROP] - integ[y0, x0 + CROP] - integ[y0 + CROP, x0] + integ[y0, x0]
            if (n, a) > best_key:
                best, best_key = (x0, y0), (n, a)
    return best


def outline(ax, m: np.ndarray, color: str, lw: float = 0.7, ls: str = "-") -> None:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        c = c[:, 0, :]
        if len(c) < 3:
            continue
        c = np.vstack([c, c[:1]])
        if ls != "-":  # dashed (dropped) outlines get a dark halo so they stay visible on dark filaments
            ax.plot(c[:, 0], c[:, 1], color="black", lw=lw + 0.7, alpha=0.7, solid_joinstyle="round")
        ax.plot(c[:, 0], c[:, 1], color=color, lw=lw, ls=ls, solid_joinstyle="round")


def show_flat(ax, flat: np.ndarray) -> None:
    lo, hi = np.percentile(flat, [0.5, 99.8])
    ax.imshow(flat, cmap="gray", vmin=lo, vmax=hi, interpolation="antialiased")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color("#888888")


def main() -> None:
    setup_style()
    st = load_stats()
    per = st["per_image"].set_index("stem")
    sel = select(st["per_image"])
    gt = load_gt()
    cand = load_candidates()
    titles = {"good": "good", "disagree": "annotator disagreement", "failure": "failure"}

    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_W, DOUBLE_W * 0.98), gridspec_kw={"wspace": 0.03, "hspace": 0.16})
    info = {}
    for row, (kind, stem) in enumerate(sel.items()):
        recs = gt[stem]
        gt_masks = {rid: [decode(r) for r in rl] for rid, rl in recs.items()}
        c = cand[cand.stem == stem]
        c_masks = [decode(r) for r in c.rle]
        kept = (c.u_hat > TAU).to_numpy()
        best_iou = np.max([iou_matrix(r, c.rle.tolist()).max(axis=0) for r in recs.values()], axis=0)
        kept_masks = [m for m, k in zip(c_masks, kept) if k]
        f_gt, f_pred = focus(kind, recs, c.rle[kept].tolist())
        x0, y0 = choose_crop([gt_masks[r][k] for r, k in f_gt] + [kept_masks[i] for i in f_pred],
                             [m for v in gt_masks.values() for m in v])
        sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        flat = load_flat(stem)[sl]
        for ax in axes[row]:
            show_flat(ax, flat)
        for i, (rid, ms) in enumerate(gt_masks.items()):
            for m in ms:
                outline(axes[row, 1], m[sl], C_GT[i % len(C_GT)], lw=0.7)
        stroke = [pe.withStroke(linewidth=1.0, foreground="black")]
        for m, k, uh, bi in zip(c_masks, kept, c.u_hat, best_iou):
            mc = m[sl]
            if mc.sum() == 0:
                continue
            outline(axes[row, 2], mc, C_KEPT if k else C_DROP, lw=0.7 if k else 0.6, ls="-" if k else (0, (2, 1.5)))
            ys, xs = np.nonzero(mc)
            j = np.argmin(ys)
            axes[row, 2].text(min(max(xs[j], 18), CROP - 18), max(ys[j] - 6, 12), f"{uh:.2f}/{bi:.2f}", fontsize=5,
                              color=C_KEPT if k else "#DDDDDD", ha="center", va="bottom", path_effects=stroke)
        p = per.loc[stem]
        hh = "" if np.isnan(p.pq_human) else f", human vs human {p.pq_human:.2f}"
        axes[row, 0].set_ylabel(f"({'abc'[row]}) {titles[kind]}", fontsize=7.5)
        axes[row, 0].set_title(f"{stem} (flattened input)", fontsize=6, loc="left")
        axes[row, 1].set_title(f"ground truth: {len(recs)} record(s), {int(p.n_gt)} filaments", fontsize=6, loc="left")
        axes[row, 2].set_title(f"ours: {int(p.n_kept)} kept, image PQ {p.pq_ours:.2f}{hh}", fontsize=6, loc="left")
        info[kind] = {"stem": stem, "crop_xy": [int(x0), int(y0)], "n_records": int(p.n_records), "n_gt": int(p.n_gt),
                      "n_kept": int(p.n_kept), "n_focus": len(f_gt) + len(f_pred), "gt_per_record": [len(v) for v in recs.values()], "pq_ours": round(float(p.pq_ours), 4),
                      "pq_human": None if np.isnan(p.pq_human) else round(float(p.pq_human), 4)}
        print(kind, info[kind])

    n_rec = max(v["n_records"] for v in info.values())
    handles = [Line2D([], [], color=C_GT[i], lw=1.2, label=f"annotator record {i + 1}") for i in range(n_rec)]
    handles += [Line2D([], [], color=C_KEPT, lw=1.2, label=f"kept instance ($\\hat u$ > {TAU:.2f})"),
                Line2D([], [], color="#888888", lw=1.0, ls=(0, (2, 1.5)), label="candidate dropped by the gate")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.06), fontsize=6.3,
               handlelength=2.0, columnspacing=1.2)
    fig.text(0.5, 0.05, "number next to each candidate: predicted expected IoU $\\hat u$ / best IoU with any annotator record "
             "(a match needs IoU > 0.5)\nout-of-fold predictions; 704 x 704 px crops of the 2048 x 2048 frame",
             ha="center", va="top", fontsize=6)
    save(fig, "fig_qualitative")
    write_numbers("qualitative", {"rule": f"eligible >= {MIN_GT_PER_REC} GT per record; crop {CROP}px", **info})


if __name__ == "__main__":
    main()
