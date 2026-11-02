"""fig_pipeline: schematic of the exp118 pipeline, illustrated on one TRAIN image with its out-of-fold maps.

Stages: 2048 H-alpha -> radial-median flattening -> 4 DML U-Nets (mean of the probability maps) -> connected
components of p > 0.5 (closing r = 2, min 300 px, no merging) -> LightGBM expected-IoU gate (keep u_hat > 0.20 ~ PQ/2)
-> pixel-disjoint instances -> RLE rows.

Illustration image (automatic): among train images whose per-image PQ (ours) is >= 0.5, the CROP x CROP window
(stride 64) that holds the most candidate components, counting dropped candidates (area >= 600 px) twice and requiring >= 2 such dropped
and >= 3 kept (so that the gate panel shows both outcomes). Candidate positions = pycocotools bbox centres.

Usage: uv run python analysis/report_figs/fig_pipeline.py   (needs compute_stats.py first)
"""
from __future__ import annotations

import cv2
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pycocotools import mask as mask_util

from common import (C_DROP, C_KEPT, C_OURS, DOUBLE_W, SIZE, TAU, decode, load_candidates, load_flat, load_prob, load_raw,
                    load_stats, rle_obj, save, setup_style, write_numbers)
from fig_qualitative import outline

CROP = 512
STRIDE = 64
MIN_DROP_AREA = 600
COMP_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def pick(cand, per_image) -> tuple[str, int, int]:
    ok = set(per_image.loc[per_image.pq_ours >= 0.5, "stem"])
    best, best_score = None, -1
    for stem, g in cand[cand.stem.isin(ok)].groupby("stem"):
        bb = mask_util.toBbox([rle_obj(r) for r in g.rle])
        cx, cy = bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2
        kept = (g.u_hat > TAU).to_numpy()
        big = (g.area >= MIN_DROP_AREA).to_numpy()
        for y0 in range(0, SIZE - CROP + 1, STRIDE):
            for x0 in range(0, SIZE - CROP + 1, STRIDE):
                inside = (cx >= x0) & (cx < x0 + CROP) & (cy >= y0) & (cy < y0 + CROP)
                nk, nd = int((inside & kept).sum()), int((inside & ~kept & big).sum())
                if nd >= 2 and nk >= 3 and nk + 2 * nd > best_score:
                    best, best_score = (stem, x0, y0), nk + 2 * nd
    return best


def img_ax(fig, rect, title: str):
    ax = fig.add_axes(rect)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color("#888888")
    ax.set_title(title, fontsize=6.3, pad=2.5, fontweight="bold")
    return ax


def arrow(fig, x0, x1, y) -> None:
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                   lw=0.8, color="#444444"))


def caption(fig, x, y, text, w) -> None:
    fig.text(x + w / 2, y, text, ha="center", va="top", fontsize=5.4, color="#333333", linespacing=1.25)


def main() -> None:
    setup_style()
    st = load_stats()
    cand = load_candidates()
    stem, x0, y0 = pick(cand, st["per_image"])
    sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    raw, flat, prob = load_raw(stem), load_flat(stem), load_prob(stem)
    g = cand[cand.stem == stem].reset_index(drop=True)
    masks = [decode(r)[sl] for r in g.rle]

    fig = plt.figure(figsize=(DOUBLE_W, 1.9))
    n, w, gap, top, h = 6, 0.128, 0.0384, 0.88, 0.128 * DOUBLE_W / 1.9
    xs = [0.005 + i * (w + gap) for i in range(n)]
    yb = top - h

    a = img_ax(fig, [xs[0], yb, w, h], "2048$^2$ H$\\alpha$ image")
    small = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA)
    a.imshow(small, cmap="gray", vmin=0, vmax=255)
    a.add_patch(Rectangle((x0 / 4, y0 / 4), CROP / 4, CROP / 4, fill=False, lw=0.7, ec="#E69F00"))
    caption(fig, xs[0], yb - 0.03, "GONG full disk, 8-bit\nlimb darkening present\n(box = crop shown right)", w)

    a = img_ax(fig, [xs[1], yb, w, h], "flattening")
    f = flat[sl]
    lo, hi = np.percentile(f, [0.5, 99.8])
    a.imshow(f, cmap="gray", vmin=lo, vmax=hi)
    caption(fig, xs[1], yb - 0.03, "image / radial median\nprofile of the disk\n(one input channel)", w)

    # model box (no image)
    a = fig.add_axes([xs[2], yb, w, h]); a.axis("off")
    a.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.0,rounding_size=0.06", fc="#EAF2F8",
                               ec=C_OURS, lw=0.8, transform=a.transAxes))
    a.set_title("4 U-Nets (DML)", fontsize=6.3, pad=2.5, fontweight="bold")
    a.text(0.5, 0.80, "encoders", ha="center", va="center", fontsize=5.6, color="#333333", transform=a.transAxes)
    for i, e in enumerate(["ResNeXt50", "EfficientNet-B4", "ConvNeXt-S", "HRNet-W32"]):
        a.text(0.5, 0.64 - i * 0.13, e, ha="center", va="center", fontsize=5.8, color=C_OURS, transform=a.transAxes,
               fontweight="bold")
    a.text(0.5, 0.08, "4-flip TTA each", ha="center", va="center", fontsize=5.4, color="#333333", transform=a.transAxes)
    caption(fig, xs[2], yb - 0.03, "trained on 1024$^2$ tiles of 2048$^2$\nBCE + Dice semimetric loss\n"
            "target = annotator-mean mask", w)

    a = img_ax(fig, [xs[3], yb, w, h], "mean probability")
    a.imshow(prob[sl], cmap="magma", vmin=0, vmax=1)
    caption(fig, xs[3], yb - 0.03, "equal-weight mean\nof the 4 maps\n(full 2048$^2$ frame)", w)

    a = img_ax(fig, [xs[4], yb, w, h], "components")
    a.imshow(f, cmap="gray", vmin=lo, vmax=hi, alpha=0.55)
    over = np.zeros((CROP, CROP, 4))
    for i, m in enumerate(masks):
        col = np.array(plt.matplotlib.colors.to_rgb(COMP_COLORS[i % len(COMP_COLORS)]))
        over[m > 0, :3] = col
        over[m > 0, 3] = 0.9
    a.imshow(over)
    caption(fig, xs[4], yb - 0.03, "connected comp. of p > 0.5\nclosing r = 2, area $\\geq$ 300 px\nno fragment merging", w)

    a = img_ax(fig, [xs[5], yb, w, h], "expected-IoU gate")
    a.imshow(f, cmap="gray", vmin=lo, vmax=hi)
    stroke = [pe.withStroke(linewidth=0.9, foreground="black")]
    n_k = n_d = 0
    for m, uh in zip(masks, g.u_hat):
        if m.sum() == 0:
            continue
        k = uh > TAU
        n_k += k; n_d += (not k)
        outline(a, m, C_KEPT if k else C_DROP, lw=0.7 if k else 0.6, ls="-" if k else (0, (2, 1.5)))
        ys, xs_ = np.nonzero(m)
        j = np.argmin(ys)
        a.text(min(max(xs_[j], 25), CROP - 25), max(ys[j] - 4, 34), f"{max(uh, 0):.2f}", fontsize=4.6, ha="center",
               va="bottom", color=C_KEPT if k else "#DDDDDD", path_effects=stroke)
    a.set_xlim(0, CROP); a.set_ylim(CROP, 0)
    caption(fig, xs[5], yb - 0.03, "LightGBM predicts each\ncomponent's IoU contribution $\\hat u$\n"
            f"keep $\\hat u$ > {TAU:.2f} ($\\approx$ PQ/2)\n(solid: kept, dashed: dropped)", w)

    for i in range(n - 1):
        arrow(fig, xs[i] + w + 0.004, xs[i + 1] - 0.004, yb + h / 2)
    fig.text(0.5, 0.035, "kept components are pixel-disjoint by construction $\\rightarrow$ one row per instance: "
             "filament_id = <stem>_<k>, segmentation_rle = COCO compressed RLE (2048 x 2048)",
             ha="center", va="bottom", fontsize=5.8, color="#222222")
    save(fig, "fig_pipeline")
    write_numbers("pipeline", {"stem": stem, "crop_xy": [int(x0), int(y0)], "crop": CROP, "kept_in_crop": int(n_k),
                               "dropped_in_crop": int(n_d)})
    print(stem, x0, y0, n_k, n_d)


if __name__ == "__main__":
    main()
