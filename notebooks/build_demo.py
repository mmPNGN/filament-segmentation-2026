"""Build notebooks/demo_inference.ipynb: the final model loaded from the shipped weights -> submission.csv (no training).
Usage: uv run python notebooks/build_demo.py notebooks/demo_inference.ipynb, then execute it with
notebooks/run_walkthrough.py <nb> 0 (no cell time limit; GPU recommended: ~45 min on an RTX 4070)."""
import sys

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Demo: final model from the shipped weights → `submission.csv`

This notebook reproduces our final submission **without any training**. It loads the 20 trained fold models
(4 U-Net encoder families × 5 folds, `weights/`) and the LightGBM expected-IoU gate (`models/gate/`), predicts every
test image, and writes `submission.csv`. Requirements: `uv sync`, and the competition's test images in
`data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images/`. A GPU is recommended (~45 min on an RTX 4070; CPU works, slowly).
The stage-by-stage explanation on one training image is in `pipeline_walkthrough.ipynb`.
""")
code("""
import sys, time
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
from src.config import load_config

cfg = load_config(ROOT / "configs" / "final_release.yaml", mode="infer")
test_dir = ROOT / cfg.data_dir / "test" / "test_images"
n_test = len(list(test_dir.glob("*.jpeg")))
assert n_test > 0, f"put the competition test images in {test_dir}"
print(f"device: {'cuda: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}; test images: {n_test}")
""")
md("""
## 1. The shipped model
Four members, each a U-Net trained with BCE + Dice semimetric loss on the annotator-mean mask (5 fold models per member,
fp16 shards below 90 MB), plus the gate: LightGBM boosters (5 seeds) that predict each candidate's expected IoU
contribution *û*; a candidate is kept when *û* > τ = 0.20 ≈ PQ/2.
""")
code("""
rcfg = cfg.release
for m in rcfg.members:
    mcfg = load_config(ROOT / "configs" / f"{m}.yaml", mode="infer")
    files = sorted((ROOT / rcfg.weights_dir / m).glob("fold*.pt"))
    print(f"{m:28s} {mcfg.model.arch} + {mcfg.model.encoder:20s} {len(files):2d} files, {sum(f.stat().st_size for f in files) / 1e6:6.0f} MB")
gate = sorted((ROOT / rcfg.gate_dir).glob("lgb_seed*.txt"))
print(f"gate: {[g.name for g in gate]}, tau = {cfg.postprocess.tau}")
""")
md("""
## 2. Predict the test set and write the submission
`src.release.run_release`: per member, the mean over its 5 fold models of the 4-way flip-TTA probability; equal-weight
mean over the members; connected components of p > 0.5 (closing radius 2, ≥ 300 px); gate; pixel-disjoint masks in
*û* order, COCO RLE. This is the same chain as training-time inference (`src/pipeline.py`).
""")
code("""
from src.release import run_release
out_dir = ROOT / "outputs" / "demo"
out_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()
sub = run_release(cfg, out_dir, ROOT, device, print)
print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir / 'submission.csv'}")
""")
md("## 3. Check the submission")
code("""
from src.submission import validate_submission
stems = sorted(p.stem for p in test_dir.glob("*.jpeg"))
print(validate_submission(sub, expected_stems=stems))
per_image = sub["filament_id"].str.rsplit("_", n=1).str[0].value_counts()
print(f"{len(sub)} instances in {len(per_image)} of {len(stems)} images; per image: median {per_image.median():.0f}, max {per_image.max()}")
sub.head(3).assign(segmentation_rle=lambda d: d.segmentation_rle.str[:40] + "...")
""")
md("""
Optional: if a reference submission is available (e.g. the file submitted to Kaggle), compare the two instance by
instance (IoU of the masks matched per image).
""")
code("""
ref_path = ROOT / "submission_reference.csv"
if ref_path.exists():
    from pycocotools import mask as mask_util
    ref = pd.read_csv(ref_path)
    obj = lambda r: {"size": [2048, 2048], "counts": r.encode()}
    ious = []
    for stem, g in sub.groupby(sub.filament_id.str.rsplit("_", n=1).str[0]):
        r = ref[ref.filament_id.str.rsplit("_", n=1).str[0] == stem]
        if len(r):
            m = np.asarray(mask_util.iou([obj(x) for x in g.segmentation_rle], [obj(x) for x in r.segmentation_rle], [0] * len(r)))
            ious += list(m.max(axis=1))
    print(f"rows {len(sub)} vs reference {len(ref)}; mean best IoU {np.mean(ious):.4f}, IoU > 0.99: {np.mean(np.array(ious) > 0.99):.3f}")
else:
    print("no submission_reference.csv: skipped")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])
print(f"cells: {len(cells)}")
