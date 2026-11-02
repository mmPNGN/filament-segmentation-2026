#!/bin/bash
# Reproduce the final submission (exp135: CV 0.4559, Public LB 0.39) from scratch on one GPU.
# Every step goes through tools/run_exp.py (GPU lock, logs, metrics.json in outputs/expNNN-<stage>/).
# Total ~36 h on an RTX 4070 12 GB: ResNeXt50 ~7 h, EfficientNet-B4 ~11 h, ConvNeXt-S ~8 h, HRNet-W32 ~9 h, post-processing 15 min.
# Usage: bash tools/reproduce_best.sh            (the run directories are found by config name)
set -e
cd "$(dirname "$0")/.."
R() { uv run python tools/run_exp.py "$@"; }
run_dir() {  # newest outputs/expNNN-full whose command used configs/$1
  ls -d outputs/exp*-full | while read d; do grep -q "configs/$1" "$d/meta.json" 2>/dev/null && echo "$(basename "$d")"; done | tail -1
}

# 1. four U-Net encoder families, BCE + Dice semimetric loss on the annotator-mean target, 5 folds each (data/folds.csv),
#    4-way flip TTA at prediction; every config pins the values that differ from configs/base.yaml
for m in final_rx50_dml final_effb4_dml final_convnext_s_dml_s43 final_hrnet_w32_dml_s43; do
  R --stage full --desc "final member $m" -- uv run python main.py --config configs/$m.yaml
done

# 2. equal-weight mean of the OOF and test probability maps -> outputs/ens_final-full/
uv run python tools/average_probs.py --name ens_final --exps $(run_dir final_rx50_dml.yaml) $(run_dir final_effb4_dml.yaml) \
  $(run_dir final_convnext_s_dml_s43.yaml) $(run_dir final_hrnet_w32_dml_s43.yaml)

# 3. connected components (p > 0.5, min 300 px, no merging) + LightGBM expected-IoU gate averaged over 5 seeds
#    (keep u_hat > 0.20) -> submission.csv (= exp135)
R --stage ensemble --desc "final: 4 DML families + merge 0 + GBDT" -- uv run python main.py --config configs/final_ensemble_gbdt5.yaml
