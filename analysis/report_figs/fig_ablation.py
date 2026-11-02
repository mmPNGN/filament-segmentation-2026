"""fig_ablation: cumulative ablation (full 5-fold CV, pooled-per-fold PQ as recorded in research/results.tsv).

Rows = research/report/outline.md §4 plus exp118 (four DML families). Filled markers = the kept pipeline at each step
(cumulative chain); open markers = single-model runs without the gate that isolate one change (their delta is against
a single-model reference, marked *). Human-vs-human PQ 0.340 (research/logs/2026-09-29_eda.md §6) as reference.
The CV values are copied from results.tsv (verified 2026-10-22); the script reads nothing else.

Usage: uv run python analysis/report_figs/fig_ablation.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import C_HUMAN, C_OURS, SINGLE_W, save, setup_style, write_numbers

HUMAN_PQ = 0.3399
# (label, exp, CV, delta text, in cumulative chain)
ROWS = [
    ("ResNet34 U-Net, 1024 px, connected comp.", "exp000", 0.3920, "", True),
    ("+ fragment merging (r = 12 px)", "exp002", 0.3994, "+0.0074", True),
    ("+ 3-seed average", "exp006", 0.4048, "+0.0053", True),
    ("+ component score threshold", "exp007", 0.4105, "+0.0058", True),
    ("+ GBDT expected-IoU gate", "exp012", 0.4155, "+0.0050", True),
    ("+ ResNeXt50 encoder", "exp015", 0.4201, "+0.0046", True),
    ("+ 4-way flip TTA", "exp022", 0.4277, "+0.0076", True),
    ("native 2048, 1024 tiles (1 model, no gate)", "exp038", 0.4307, "+0.0109*", False),
    ("   + Dice semimetric loss (1 model, no gate)", "exp055", 0.4361, "+0.0055*", False),
    ("native 2048 (1 model) + GBDT gate", "exp039", 0.4407, "+0.0100*", True),
    ("2 DML families (ResNeXt50 + EffNet-B4)", "exp076", 0.4485, "+0.0078", True),
    ("+ no fragment merging", "exp079", 0.4521, "+0.0036", True),
    ("4 DML families (+ ConvNeXt-S, HRNet-W32)", "exp118", 0.4559, "+0.0038", True),
]


def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.75))
    n = len(ROWS)
    y = np.arange(n)[::-1]
    cv = np.array([r[2] for r in ROWS])
    chain = np.array([r[4] for r in ROWS])
    ax.axvline(HUMAN_PQ, color=C_HUMAN, lw=1.0)
    ax.text(HUMAN_PQ + 0.0012, y[-1] - 0.35, "human vs human\nPQ 0.340", fontsize=5.6, color="#9A6A00", va="bottom")
    for yi in y:
        ax.axhline(yi, color="#EEEEEE", lw=0.5, zorder=0)
    ax.plot(cv[chain], y[chain], color=C_OURS, lw=0.8, zorder=1, drawstyle="steps-pre", alpha=0.6)
    ax.scatter(cv[chain], y[chain], s=14, color=C_OURS, zorder=3, edgecolor="white", linewidth=0.5)
    ax.scatter(cv[~chain], y[~chain], s=14, facecolor="white", edgecolor=C_OURS, linewidth=0.9, zorder=3)
    for (lab, exp, v, d, _), yi in zip(ROWS, y):
        ax.text(v + 0.0016, yi, f"{v:.4f}", va="center", fontsize=5.4, color="#333333")
        ax.text(1.16, yi, d, va="center", ha="right", fontsize=5.4, color="#333333", transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [r[0] for r in ROWS], fontsize=5.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.335, 0.47)
    ax.set_ylim(-0.7, n - 0.4)
    ax.set_xticks([0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46])
    ax.set_xlabel("PQ, 5-fold CV (out-of-fold)")
    ax.text(1.16, n - 0.45, "$\\Delta$", ha="right", va="bottom", fontsize=6, transform=ax.get_yaxis_transform())
    fig.text(0.5, -0.01, "filled: kept pipeline (cumulative); open: single model without gate. * $\\Delta$ against a\n"
             "single-model reference: native 2048 vs the same model at 1024 px (0.4198);\n"
             "Dice semimetric loss and GBDT gate vs native 2048 without them (0.4307)",
             fontsize=5.2, color="#555555", va="top", ha="center")
    save(fig, "fig_ablation")
    write_numbers("ablation", {r[1]: {"label": r[0], "cv": r[2], "delta": r[3], "chain": r[4]} for r in ROWS}
                  | {"human_vs_human": HUMAN_PQ, "total_gain_exp000_to_exp118": round(ROWS[-1][2] - ROWS[0][2], 4)})


if __name__ == "__main__":
    main()
