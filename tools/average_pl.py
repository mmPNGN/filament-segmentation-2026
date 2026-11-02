"""Average the pseudo-labels of several teacher runs per fold (ensemble teacher): outputs/<name>/pl/fold<k>/<stem>.npy
= mean over the given runs' pl/fold<k>/<stem>.npy (uint8 x255). Each member's fold-k labels come from its fold-k model,
which never saw fold k, so the fold-k average stays fold-k clean. Usage:
    uv run python tools/average_pl.py --name pl_ens5 --runs exp064-infer exp065-infer ...
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "outputs"
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--runs", nargs="+", required=True)
a = ap.parse_args()
for k in range(5):
    dirs = [ROOT / r / "pl" / f"fold{k}" for r in a.runs]
    stems = sorted(set.intersection(*[{p.stem for p in d.glob("*.npy")} for d in dirs]))
    out = ROOT / a.name / "pl" / f"fold{k}"
    out.mkdir(parents=True, exist_ok=True)
    for s in stems:
        acc = sum(np.load(d / f"{s}.npy").astype(np.float32) for d in dirs) / len(dirs)
        np.save(out / f"{s}.npy", np.round(acc).astype(np.uint8))
    print(f"fold {k}: {len(stems)} frames averaged over {len(dirs)} teachers")
