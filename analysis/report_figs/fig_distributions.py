"""fig_distributions: the rubric's quantitative distributions, exp118 OOF ("ours") vs human vs human.

(a) IoU of matched pairs (TP, IoU > 0.5)           ours: all 1,154 records of 707 images; human: 1,196 ordered record
(b) best Dice of every GT filament (0 = untouched)     pairs of the 296 multi-record images (one record as prediction,
(c) one-to-many: #predictions touching a GT (IoU > 0)  another as ground truth; research/logs/2026-09-29_eda.md §6)
(d) many-to-one: #GT of a record touched by a prediction
Counts follow src/metric.panoptic_quality (one_to_many / many_to_one are the host's plot_m2n_counts definitions). For
human vs human the two directions are the same population (every ordered pair appears both ways), so (c) = (d).

Usage: uv run python analysis/report_figs/fig_distributions.py   (needs compute_stats.py first)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import C_HUMAN, C_OURS, C_REF, DOUBLE_W, load_stats, save, setup_style


def hist_pair(ax, ours: np.ndarray, human: np.ndarray, bins: np.ndarray) -> None:
    for x, c, lab, fill in ((human, C_HUMAN, "human vs human", False), (ours, C_OURS, "ours (exp118 OOF)", True)):
        w = np.full(len(x), 100.0 / len(x))
        if fill:
            ax.hist(x, bins=bins, weights=w, color=c, alpha=0.22, lw=0)
        ax.hist(x, bins=bins, weights=w, histtype="step", color=c, lw=1.1, label=lab)
        ax.axvline(np.median(x), color=c, lw=0.8, ls=(0, (3, 2)))


def m2n_bars(ax, c_ours, c_human, title: str) -> None:
    cats = ["0", "1", "2", "3+"]

    def shares(c):
        n = sum(c.values())
        return np.array([c[0], c[1], c[2], sum(v for k, v in c.items() if k >= 3)]) / n * 100

    x = np.arange(4)
    w = 0.38
    for off, c, col, lab in ((-w / 2, c_ours, C_OURS, "ours"), (w / 2, c_human, C_HUMAN, "human vs human")):
        s = shares(c)
        ax.bar(x + off, s, w * 0.92, color=col, label=lab)
        for xi, v in zip((x + off)[:2], s[:2]):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.3, color="#333333")
    s_o, s_h = shares(c_ours), shares(c_human)
    ax.text(3.55, 22, f"2 or more:\nours {s_o[2:].sum():.1f} %\nhuman {s_h[2:].sum():.1f} %", ha="right", va="bottom",
            fontsize=5.4, color="#333333")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 95)
    ax.set_ylabel("share (%)")
    ax.set_title(title, loc="left")


def main() -> None:
    setup_style()
    st = load_stats()
    res, best = st["res"], st["best_dice"]
    io, ih = np.asarray(res["ours"].tp_ious), np.asarray(res["human"].tp_ious)
    fig, ax = plt.subplots(1, 4, figsize=(DOUBLE_W, 1.85), gridspec_kw={"wspace": 0.42, "width_ratios": [1.15, 1.15, 1, 1]})

    hist_pair(ax[0], io, ih, np.arange(0.5, 1.0001, 0.025))
    ax[0].set_xlim(0.5, 1.0)
    ax[0].set_ylim(0, 14)
    ax[0].set_xlabel("IoU of matched pair")
    ax[0].set_ylabel("share of matches (%)")
    ax[0].set_title("(a) IoU of matches (TP)", loc="left")
    ax[0].text(0.99, 0.99, f"median {np.median(io):.3f}\nn = {len(io):,}", transform=ax[0].transAxes, ha="right",
               va="top", fontsize=5.6, color=C_OURS)
    ax[0].text(0.99, 0.78, f"median {np.median(ih):.3f}\nn = {len(ih):,}", transform=ax[0].transAxes, ha="right",
               va="top", fontsize=5.6, color="#9A6A00")

    hist_pair(ax[1], best["ours"], best["human"], np.arange(0.0, 1.0001, 0.05))
    ax[1].axvline(2 / 3, color=C_REF, lw=0.7)
    ax[1].text(2 / 3 - 0.02, ax[1].get_ylim()[1] * 0.97, "match\n(Dice > 2/3)", ha="right", va="top", fontsize=5.4, color=C_REF)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("best Dice of each GT filament")
    ax[1].set_ylabel("share of GT filaments (%)")
    ax[1].set_title("(b) Dice per GT filament", loc="left")

    m2n_bars(ax[2], res["ours"].one_to_many, res["human"].one_to_many, "(c) one-to-many")
    ax[2].set_xlabel("predictions touching a GT")
    m2n_bars(ax[3], res["ours"].many_to_one, res["human"].many_to_one, "(d) many-to-one")
    ax[3].set_xlabel("GT touched by a prediction")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h[::-1], ["ours (exp118, out-of-fold, 1,154 records)", "human vs human (1,196 ordered record pairs)"],
               loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=6.3, handlelength=1.6)
    save(fig, "fig_distributions")
    print(f"TP IoU median ours {np.median(io):.4f} human {np.median(ih):.4f}; "
          f"best Dice median ours {np.median(best['ours']):.4f} human {np.median(best['human']):.4f}")


if __name__ == "__main__":
    main()
