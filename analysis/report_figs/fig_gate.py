"""fig_gate: the expected-IoU gate on the exp118 OOF.

(a) pooled PQ when keeping candidates with u_hat > tau (all 5,516 connected components of the 4-family mean map,
    nested GBDT u_hat; exact RLE IoU per annotator record, micro pooled; same numbers as analysis/sparse_label_tau.py)
(b) TP / FP / FN against tau
(c) calibration of the GBDT: mean measured u (record-averaged IoU contribution, IoU > 0.5 else 0) per u_hat bin, with
    the decision rule "add a prediction only if u > PQ/2" (competition.md §2)

Usage: uv run python analysis/report_figs/fig_gate.py   (needs compute_stats.py first)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import C_HUMAN, C_OURS, C_REF, DOUBLE_W, TAU, load_stats, save, setup_style, write_numbers

C_FP, C_FN, C_TP = "#D55E00", "#CC79A7", "#0072B2"


def main() -> None:
    setup_style()
    st = load_stats()
    cur = st["gate_curve"]
    pq_used = float(cur.loc[np.isclose(cur.tau, TAU), "PQ"].iloc[0])
    half = pq_used / 2
    best = cur.loc[cur.PQ.idxmax()]
    no_gate = cur.iloc[0]

    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE_W, 1.95), gridspec_kw={"wspace": 0.38})
    a = ax[0]
    a.plot(cur.tau, cur.PQ, color=C_OURS, lw=1.3)
    a.axvline(TAU, color=C_REF, lw=0.7, ls=(0, (3, 2)))
    a.plot([TAU], [pq_used], "o", ms=4, color=C_OURS, mec="white", mew=0.6)
    a.axvline(half, color=C_HUMAN, lw=0.9)
    a.text(half + 0.012, 0.375, f"u = PQ/2\n= {half:.3f}", fontsize=5.6, color="#9A6A00", ha="left", va="bottom")
    a.text(0.01, 0.405, f"used $\\tau$={TAU:.2f}\nPQ {pq_used:.4f}\n\nbest $\\tau$={best.tau:.2f}\nPQ {best.PQ:.4f}",
           fontsize=5.3, color="#333333", va="top")
    a.text(0.01, no_gate.PQ - 0.005, f"keep all\n{no_gate.PQ:.4f}", fontsize=5.6, color=C_REF, va="top")
    a.set_xlim(0, 0.6)
    a.set_ylim(0.37, 0.462)
    a.set_xlabel("gate threshold $\\tau$ (keep $\\hat u > \\tau$)")
    a.set_ylabel("PQ (OOF, pooled)")
    a.set_title("(a) PQ vs gate threshold", loc="left")

    b = ax[1]
    for col, c, lab in (("TP", C_TP, "TP"), ("FN", C_FN, "FN"), ("FP", C_FP, "FP")):
        b.plot(cur.tau, cur[col] / 1000, color=c, lw=1.2)
        y = cur[col].iloc[-1] / 1000
        b.text(0.605, y, lab, fontsize=6, va="center", color="#333333")
    b.axvline(TAU, color=C_REF, lw=0.7, ls=(0, (3, 2)))
    b.set_xlim(0, 0.6)
    b.set_xlabel("gate threshold $\\tau$")
    b.set_ylabel("count (thousands)")
    b.set_title("(b) matches and errors", loc="left")

    c = ax[2]
    cal = st["calib"]
    edges = np.arange(-0.05, 0.851, 0.05)
    idx = np.digitize(cal.u_hat, edges) - 1
    rows = []
    for i in range(len(edges) - 1):
        u = cal.u[idx == i].to_numpy()
        if len(u) >= 20:
            rows.append((cal.u_hat[idx == i].mean(), u.mean(), 1.96 * u.std(ddof=1) / np.sqrt(len(u)), len(u)))
    rows = np.array(rows)
    c.plot([0, 0.8], [0, 0.8], color="#BBBBBB", lw=0.7)
    c.axhline(half, color=C_HUMAN, lw=0.9)
    c.text(0.79, half + 0.012, f"PQ/2 = {half:.3f}", ha="right", va="bottom", fontsize=5.8, color="#9A6A00")
    c.axvline(TAU, color=C_REF, lw=0.7, ls=(0, (3, 2)))
    c.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], fmt="o", ms=3, color=C_OURS, mec="white", mew=0.4,
               elinewidth=0.8, capsize=0)
    c.set_xlim(-0.06, 0.8)
    c.set_ylim(-0.06, 0.8)
    c.set_aspect("equal")
    c.set_xlabel("predicted $\\hat u$ (nested GBDT)")
    c.set_ylabel("measured u (mean, 95% CI)")
    c.set_title("(c) calibration of $\\hat u$", loc="left")
    save(fig, "fig_gate")

    bins = {f"{r[0]:.3f}": {"u_mean": round(float(r[1]), 4), "n": int(r[3])} for r in rows}
    nums = {"tau_used": TAU, "pq_at_tau": round(pq_used, 6), "pq_over_2": round(half, 4),
            "best_tau": float(best.tau), "best_pq": round(float(best.PQ), 6), "pq_keep_all": round(float(no_gate.PQ), 6),
            "tp_fp_fn_keep_all": [int(no_gate.TP), int(no_gate.FP), int(no_gate.FN)],
            "tp_fp_fn_at_tau": [int(x) for x in cur.loc[np.isclose(cur.tau, TAU), ["TP", "FP", "FN"]].iloc[0]],
            "pq_tau_0.30": round(float(cur.loc[np.isclose(cur.tau, 0.30), "PQ"].iloc[0]), 6),
            "pq_tau_0.10": round(float(cur.loc[np.isclose(cur.tau, 0.10), "PQ"].iloc[0]), 6),
            "candidates": st["n_candidates"], "kept": st["n_kept"], "calibration_bins": bins}
    write_numbers("gate", nums)
    print({k: v for k, v in nums.items() if k != "calibration_bins"})
    print(rows.round(3))


if __name__ == "__main__":
    main()
