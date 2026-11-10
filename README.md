# Solar Filament Segmentation 2026 — instance segmentation of H-alpha filaments

Solution for the Kaggle / IEEE BigData Cup 2026 challenge
[Solar Filament Segmentation 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) (NSF NSO).
The task: from a 2048 × 2048 GONG H-alpha full-disk image, output one pixel-disjoint mask per filament.
The metric is Panoptic Quality (IoU > 0.5), matched separately against every annotator's record and micro-pooled.

| | CV (5-fold, 707 train images) | Public LB |
|---|---|---|
| one annotator scored against another (human agreement) | 0.340 | — |
| baseline: ResNet34 U-Net 1024 px + connected components | 0.3920 | 0.34 |
| **this pipeline** (4 encoder families + expected-IoU gate) | **0.4559** (95 % CI 0.446–0.465) | 0.39 |

The CV–LB offset is a constant ratio (LB ≈ 0.86 × CV over 40+ submissions, Spearman 0.83). It was tested with a
deliberately mis-gated submission (gate τ = 0.5: predicted 0.32 under a constant ratio, observed 0.32), so models are
selected on CV only.

## Pipeline

1. **Pre-processing** (`src/preprocess.py`): solar disk fit and limb-darkening flattening with a radial median, giving one
   "flat" channel where the quiet Sun is 1.0 at every radius. The cache is kept at native 2048 px.
2. **Target** (`src/data.py`): the mean of the annotators' masks (soft consensus), never a single record.
3. **Networks** (`src/models.py`, `src/train.py`): U-Net with four encoder families (ResNeXt50, EfficientNet-B4,
   ConvNeXt-S, HRNet-W32). Each is trained on random 1024 px tiles of the 2048 frame, half of them centred on filaments.
   The loss is BCE + the Dice semimetric loss (Wang & Blaschko 2023), which is unbiased on soft targets.
   Training uses 30 epochs, AdamW and a cosine schedule, and predicts on full 2048 frames with 4-way flip TTA.
4. **Ensemble** (`tools/average_probs.py`): an equal-weight mean of the probability maps of the 4 families (5 fold models each on test).
5. **Instances** (`src/postprocess.py`): connected components of p > 0.5 after a closing of radius 2, with a minimum of 300 px and no fragment merging.
6. **Gate** (`src/scorer.py`): LightGBM regresses each component's expected PQ contribution u (mean over records of IoU
   if IoU > 0.5) from 37 shape / probability / contrast features, with nested folds on OOF. A component is kept when
   û > 0.20 ≈ PQ/2, because adding a prediction raises the PQ denominator by 0.5 and its numerator by u. The final gate
   averages 5 LightGBM seeds.
7. **Submission** (`src/submission.py`): masks are pixel-disjoint in score order, encoded as COCO RLE, and validated.

The metric (`src/metric.py`) reproduces the host's self-evaluation notebook exactly (`tests/test_metric.py`).

## Reproduce the submission from the shipped weights (no training)

The trained models are in the repository: `weights/<member>/fold<k>.part<j>.pt` (20 fold models of the 4 members,
fp16, split into files below 90 MB) and `models/gate/` (the LightGBM gate, 5 seeds). `notebooks/demo.ipynb`
loads them and writes `submission.csv`; the same from the command line:

```bash
uv sync                                   # Python 3.11, CUDA 12.8 wheels (or: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128)
# put the competition data in data/raw/MAGFiLO_1.0_Kaggle_2026/ (test/test_images/ is enough for inference)
uv run python main.py --config configs/final_release.yaml --infer-only     # -> outputs/manual/submission.csv
```

About 45 min on one RTX 4070 (20 models x 180 images x 4 flips at 2048²); it also runs on CPU, much more slowly.

## Retrain from scratch

```bash
uv run python main.py --config configs/base.yaml --debug     # smoke test (a few steps, 1 fold)
bash tools/reproduce_best.sh              # 4 members x 5 folds (~36 h on one RTX 4070) + ensemble + gate -> submission.csv
```

`main.py` is the single entry point. Every experiment is a small YAML overlay on `configs/base.yaml`; the final members
are pinned in `configs/final_*.yaml`. `tools/run_exp.py` runs each experiment with a GPU lock and saves the git commit,
the patch, the logs and `metrics.json` in `outputs/expNNN-<stage>/`. The fold assignment is fixed in `data/folds.csv`:
GroupKFold(5) over 72-hour time blocks, because filaments persist for days.

## Repository layout

```
main.py            entry point (train / predict / post-process one experiment)
src/               config, preprocess, data, models, train, postprocess, scorer (GBDT gate), metric, submission, folds
configs/           base.yaml + experiment overlays; final_*.yaml = the submitted pipeline
tools/             run_exp.py (runner), results.py (ledger), average_probs.py (ensembles), reproduce_best.sh,
                   export_release.py (writes weights/ and models/gate/)
tests/             metric equivalence with the host notebook
analysis/          report_figs/: scripts for the report figures (out-of-fold predictions)
research/          results.tsv (every run with what changed and why), report/figs/
notebooks/         demo.ipynb (pre-processing -> optional training -> evaluation -> submission, from the shipped weights), pipeline_walkthrough.ipynb (every stage on one image)
weights/, models/  trained fold models (fp16 shards) and the LightGBM gate
```

## Validation and decisions

- σ_seed = 0.0021 (3 seeds); a change is kept only if ΔCV ≥ 0.005 on ≥ 3/5 folds, or if it is positive in ≥ 3
  independent paired tests (rules fixed before the experiments).
- Every run, including the negative ones, is in `research/results.tsv` with what was changed, the prediction and the result.
  Examples: pseudo-labels on unlabelled 2023–26 GONG frames, extra ridge / z-score channels, a two-stage crop refiner,
  separation heads, geometric splitting, and stronger single encoders.

## Data and rules

Only the competition's training annotations are used as ground truth. The public MAGFiLO release (which contains the
test labels) is not downloaded or used. Inference reads only the provided test images.

## License

MIT (see `LICENSE`).
