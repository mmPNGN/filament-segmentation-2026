"""Build notebooks/demo.ipynb, the host's demo notebook (discussions #753665, #755907): pre-processing -> training (optional:
retrain from scratch, or load the shipped weights) -> out-of-fold evaluation -> test inference and post-processing ->
submission.csv. Usage: uv run python notebooks/build_demo.py notebooks/demo.ipynb, then execute it with
notebooks/run_walkthrough.py notebooks/demo.ipynb 0 (no cell time limit; ~75 min on an RTX 4070 with the defaults)."""
import sys

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Demo: pre-processing → training → post-processing → evaluation → `submission.csv`

The whole solution, end to end, calling the modules in `src/`. By default it **loads the shipped trained models**
(`weights/`: 4 U-Net encoder families × 5 fold models, fp16; `models/gate/`: the LightGBM gate) and does not train;
set `TRAIN_FROM_SCRATCH = True` to retrain everything first (~36 h on one RTX 4070).

Requirements: `uv sync`; the competition data in `data/raw/MAGFiLO_1.0_Kaggle_2026/` (`train/` for training and
evaluation, `test/test_images/` for the submission). A GPU is recommended (defaults: ~45 min evaluation + ~30 min test
inference on an RTX 4070). A stage-by-stage explanation on one image is in `pipeline_walkthrough.ipynb`.
""")
code("""
TRAIN_FROM_SCRATCH = False   # True: retrain the 4 members x 5 folds and the gate (~36 h), then use those weights
EVALUATE_OOF = True          # out-of-fold PQ of the models on the 707 training images (~45 min on GPU)
PREDICT_TEST = True          # test inference + post-processing -> submission.csv (~30 min on GPU)

import sys, time, subprocess
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from src.config import load_config

cfg = load_config(ROOT / "configs" / "final_release.yaml", mode="infer")
data_dir = ROOT / cfg.data_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}; "
      f"train images: {len(list((data_dir / 'train' / 'train_images').glob('*.jp*g')))}, "
      f"test images: {len(list((data_dir / 'test' / 'test_images').glob('*.jpeg')))}")
""")
md("""
## 1. Pre-processing
Every image (train and test alike) is reduced to one channel: the solar disk is fitted on the limb, and the image is
divided by its radial median profile, so the quiet Sun is 1.0 at every radius (limb darkening and observatory
brightness differences removed). `src.data.Cache` stores this "flat" channel at the native 2048² resolution.
""")
code("""
from src.preprocess import read_gray, detect_disk, flatten
example = sorted((data_dir / "train" / "train_images").glob("*.jp*g"))[0]
gray = read_gray(example)
disk = detect_disk(gray)
flat = flatten(gray, disk)
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
ax[0].imshow(gray, cmap="gray"); ax[0].set_title(f"{example.stem}: raw H-alpha")
ax[1].imshow(flat, cmap="gray", vmin=0.7, vmax=1.2); ax[1].set_title("flattened (radial median)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
print(f"disk centre ({disk.cx:.0f}, {disk.cy:.0f}), radius {disk.r:.0f} px")
""")
md("""
## 2. Training (optional)
Each member is a U-Net trained for 30 epochs on random 1024² tiles of the 2048² frame (half centred on filaments), with
BCE + Dice semimetric loss on the mean of the annotators' masks, 5 folds over 72-hour time blocks (`data/folds.csv`).
`tools/reproduce_best.sh` trains the 4 members (`configs/final_*.yaml`), averages their maps and fits the gate on the
out-of-fold components; `tools/export_release.py --latest` then writes the new `weights/` and `models/gate/`.
""")
code("""
if TRAIN_FROM_SCRATCH:
    subprocess.run(["bash", "tools/reproduce_best.sh"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "tools/export_release.py", "--latest"], cwd=ROOT, check=True)
rcfg = cfg.release
for m in rcfg.members:
    mcfg = load_config(ROOT / "configs" / f"{m}.yaml", mode="infer")
    files = sorted((ROOT / rcfg.weights_dir / m).glob("fold*.pt"))
    print(f"{m:28s} {mcfg.model.arch} + {mcfg.model.encoder:20s} {len(files):2d} files, "
          f"{sum(f.stat().st_size for f in files) / 1e6:5.0f} MB")
print(f"gate: {[g.name for g in sorted((ROOT / rcfg.gate_dir).glob('lgb_seed*.txt'))]}, tau = {cfg.postprocess.tau}")
""")
md("""
## 3. Evaluation (out-of-fold)
Each training image is predicted only by the fold model that did not see it; the members' maps are averaged; the
candidates (connected components of p > 0.5) are scored by gate boosters fitted on the other folds; PQ is computed with
the official matching (IoU > 0.5, per annotator record, micro-pooled; `src/metric.py`). This is our cross-validation
score (0.4559 for the submitted models).
""")
code("""
if EVALUATE_OOF:
    from src.release import evaluate_oof
    t0 = time.time()
    oof = evaluate_oof(cfg, ROOT, device, lambda *a: None)
    print(f"out-of-fold PQ: mean over folds {oof['cv']:.4f} +- {oof['cv_std']:.4f}; "
          f"per fold {[round(v, 4) for v in oof['folds'].values()]}")
    p = oof["pooled"]
    print(f"pooled: PQ {p['pq']:.4f}, SQ {p['sq']:.4f}, RQ {p['rq']:.4f}, TP/FP/FN {p['tp']}/{p['fp']}/{p['fn']} "
          f"({(time.time() - t0) / 60:.0f} min)")
""")
md("""
## 4. Test inference and post-processing → `submission.csv`
Per member, the mean over its 5 fold models of the 4-way flip-TTA probability; equal-weight mean over the members;
connected components of p > 0.5 (closing radius 2, ≥ 300 px, no merging); the gate keeps a candidate when its predicted
expected IoU contribution û > 0.20 ≈ PQ/2; masks are pixel-disjoint, ranked by û, encoded as COCO RLE.
""")
code("""
if PREDICT_TEST:
    from src.release import run_release
    out_dir = ROOT / "outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sub = run_release(cfg, out_dir, ROOT, device, lambda *a: None)
    print(f"{len(sub)} instances written to {out_dir / 'submission.csv'} in {(time.time() - t0) / 60:.0f} min")
""")
md("## 5. Check the submission")
code("""
if PREDICT_TEST:
    from src.submission import validate_submission
    stems = sorted(p.stem for p in (data_dir / "test" / "test_images").glob("*.jpeg"))
    print(validate_submission(sub, expected_stems=stems))
    ref_path = ROOT / "submission_reference.csv"   # optional: the file submitted to Kaggle, for an instance-by-instance check
    if ref_path.exists():
        from pycocotools import mask as mask_util
        ref = pd.read_csv(ref_path)
        obj = lambda r: {"size": [2048, 2048], "counts": r.encode()}
        key = lambda d: d.filament_id.str.rsplit("_", n=1).str[0]
        ious = []
        for stem, g in sub.groupby(key(sub)):
            r = ref[key(ref) == stem]
            if len(r):
                m = np.asarray(mask_util.iou([obj(x) for x in g.segmentation_rle], [obj(x) for x in r.segmentation_rle], [0] * len(r)))
                ious += list(m.max(axis=1))
        print(f"vs reference: rows {len(sub)} / {len(ref)}, instances with IoU > 0.99: {np.mean(np.array(ious) > 0.99):.3f}")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])
print(f"cells: {len(cells)}")
