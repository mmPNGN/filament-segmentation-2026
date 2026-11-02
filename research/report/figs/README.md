# Report figures

All figures are vector PDF (for Overleaf) plus a 300 dpi PNG preview, English labels, Okabe-Ito (colour-blind safe)
colours with fixed roles: blue = ours, orange = human vs human / annotator record 1, sky blue = record 2,
green = kept instance, light grey dashed = candidate dropped by the gate.
Everything is computed on TRAIN images with the exp118 5-fold out-of-fold (OOF) predictions; no test image or test
prediction is read. Scripts live in `analysis/report_figs/` (CPU only, about 1 min in total). All quotable numbers are
also written to `numbers.json` in this folder.

Re-generate (from the repository root):
```
uv run python analysis/report_figs/compute_stats.py      # matching statistics -> analysis/report_figs/_cache/stats.pkl
uv run python analysis/report_figs/fig_pipeline.py
uv run python analysis/report_figs/fig_qualitative.py
uv run python analysis/report_figs/fig_distributions.py
uv run python analysis/report_figs/fig_ablation.py
uv run python analysis/report_figs/fig_gate.py
```

Matching follows `src/metric.py` exactly (IoU > 0.5 per annotator record, micro pooled). The recomputed OOF PQ is
0.455422 with TP/FP/FN 5195/2254/3004 (= `outputs/exp118-ensemble/oof_summary.json`), and the human-vs-human PQ is
0.339918 with TP/FP/FN 4292/3709/3709 (= research/logs/2026-09-29_eda.md §6), so both references are reproduced.

| figure | width | script | what it shows |
|---|---|---|---|
| `fig_pipeline` | double column | `fig_pipeline.py` | schematic on train image 20120217181154Bh (512 px crop at x=960, y=1280; selection rule in the script): 2048² H-alpha -> radial-median flattening -> 4 DML U-Nets (ResNeXt50, EfficientNet-B4, ConvNeXt-S, HRNet-W32; 1024² tiles, BCE + Dice semimetric loss on the annotator-mean target, 4-flip TTA) -> mean probability -> connected components p > 0.5 (closing r = 2, >= 300 px, no merging) -> LightGBM gate (keep û > 0.20 ≈ PQ/2) -> pixel-disjoint instances -> RLE |
| `fig_qualitative` | double column | `fig_qualitative.py` | 3 rows x (flattened input, GT outlines per record, our candidates with û / best IoU). Selection: images with >= 6 GT per record; good = highest image PQ among multi-record images; disagreement = lowest human-vs-human PQ; failure = lowest image PQ. 704 px crop = window with the most "focus" instances (all / single-annotator GT / errors) |
| `fig_distributions` | double column | `fig_distributions.py` | (a) TP IoU histogram, (b) best Dice per GT filament, (c) one-to-many, (d) many-to-one; ours vs human vs human |
| `fig_ablation` | single column | `fig_ablation.py` | cumulative ablation (outline §4 + exp118) as a dot/step chart with the human-vs-human PQ 0.340 line |
| `fig_gate` | double column | `fig_gate.py` | (a) PQ vs gate threshold τ, with u = PQ/2 marked, (b) TP/FP/FN vs τ, (c) calibration of û (binned measured u, 95 % CI) |

## Key numbers (for the text)

**Qualitative examples** (per-image PQ, official matching)
- good: 20120223063154Uh, 2 records (8 / 7 GT), ours 0.695, human vs human 0.658
- annotator disagreement: 20131128192214Mh, 2 records (9 / 10 GT), human vs human 0.054, ours 0.232
- failure: 20161201160954Bh, 1 record (7 GT), ours 0.101 (in the crop the long filament is fragmented into 2 kept pieces with IoU 0.17 / 0.30, and the other kept instances reach at most IoU ≈ 0.50, below the strict > 0.5 match)

**Distributions** (ours = exp118 OOF, 1,154 records / 707 images; human = 1,196 ordered record pairs / 296 multi-record images)

| | ours | ours, multi-record images only | human vs human |
|---|---|---|---|
| PQ / SQ / RQ | 0.4554 / 0.686 / 0.664 | 0.4558 / 0.683 / 0.668 | 0.3399 / 0.634 / 0.536 |
| TP IoU quantiles 10 / 50 / 90 % | 0.561 / **0.694** / 0.796 | 0.557 / 0.691 / 0.791 | 0.534 / **0.629** / 0.739 |
| TP Dice quantiles 10 / 50 / 90 % | 0.719 / 0.819 / 0.887 | 0.716 / 0.817 / 0.883 | 0.696 / 0.773 / 0.850 |
| best Dice per GT filament, median | 0.757 | 0.763 | 0.687 |
| GT matched (best Dice > 2/3, i.e. IoU > 0.5) | 63.4 % | 65.5 % | 53.6 % |
| GT touched by no prediction (1:N = 0) | 21.3 % | 18.2 % | 17.8 % |
| 1:N = 1 / 2 / 3+ | 75.5 / 2.9 / 0.4 % | 78.2 / 3.1 / 0.5 % | 79.1 / 2.8 / 0.4 % |
| prediction touching no GT (N:1 = 0) | 11.5 % | 12.7 % | 17.8 % |
| N:1 = 1 / 2 / 3+ | 86.3 / 2.1 / 0.1 % | 85.3 / 1.9 / 0.1 % | 79.1 / 2.8 / 0.4 % |

Raw counts: ours 1:N {0: 1742, 1: 6188, 2: 237, 3: 31, 5: 1}, N:1 {0: 856, 1: 6429, 2: 158, 3: 5};
human 1:N = N:1 {0: 1420, 1: 6326, 2: 225, 3: 26, 4: 2} (every ordered pair appears in both directions, so the two
human distributions are identical).

**Ablation**: 0.3920 (ResNet34, 1024 px) -> 0.4559 (4 DML families), +0.064 in total; values from research/results.tsv.

**Gate** (5,516 candidate components, 4,560 kept)
- keep all candidates: PQ 0.4365 (TP/FP/FN 5433/3384/2766); τ = 0.20: 0.4554 (5195/2254/3004) = +0.019 from the gate
- best τ on the OOF grid (0.01 steps): 0.21, PQ 0.4561 (+0.0007, inside the flat 0.20-0.24 plateau); τ = 0.10: 0.4471; τ = 0.30: 0.4518
- PQ/2 = 0.228; calibration bins (mean û -> measured u): 0.175 -> 0.153, 0.225 -> 0.265, 0.276 -> 0.282; the measured u
  crosses PQ/2 between û 0.18 and 0.23, so τ = 0.20 sits at the rule's crossing point. û is well calibrated over
  0-0.7 (top bin 0.77 -> 0.73 slightly over-predicted)
