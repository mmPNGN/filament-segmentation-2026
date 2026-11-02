"""Average the saved OOF / test probability maps of several experiments into a synthetic source directory that the
post-processing-only pipeline (`postprocess.source_exp`) can consume. Used for seed / model ensembles without
retraining (experiments.md: OOF and test predictions of every experiment are ensemble material).

    uv run python tools/average_probs.py --name ens_seed42_43_44 --exps exp000-full exp003-calib exp005-calib

Writes outputs/<name>-full/{oof,test_prob}/<stem>.npy (uint8 x255, weighted mean) and members.json.
All members must have identical stem sets (same folds.csv); the script refuses otherwise.
--weights: one non-negative weight per member (default: equal). Members of different resolution (e.g. 1024 and 2048
maps) are bilinearly resized to the largest one before averaging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def average(dirs: list[Path], out: Path, weights: list[float]) -> int:
    stems = [sorted(p.stem for p in d.glob("*.npy")) for d in dirs]
    if any(s != stems[0] for s in stems[1:]):
        raise SystemExit(f"stem sets differ between members of {out}")
    out.mkdir(parents=True, exist_ok=True)
    w = np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
    for stem in stems[0]:
        maps = [np.load(d / f"{stem}.npy").astype(np.float32) for d in dirs]
        size = max(m.shape[0] for m in maps)
        acc = np.zeros((size, size), dtype=np.float32)
        for m, wi in zip(maps, w):
            if m.shape[0] != size:
                m = cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)
            acc += wi * m
        np.save(out / f"{stem}.npy", np.round(acc).astype(np.uint8))
    return len(stems[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="synthetic experiment name, written to outputs/<name>-full/")
    ap.add_argument("--exps", nargs="+", required=True, help="member directories under outputs/, e.g. exp000-full exp003-calib")
    ap.add_argument("--weights", nargs="+", type=float, help="one weight per member (default: equal)")
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.exps)
    if len(weights) != len(args.exps) or min(weights) < 0 or sum(weights) <= 0:
        raise SystemExit("--weights needs one non-negative weight per member")
    members = [ROOT / "outputs" / e for e in args.exps]
    for m in members:
        if not (m / "oof").is_dir() or not (m / "test_prob").is_dir():
            raise SystemExit(f"{m} has no oof/ and test_prob/")
    out = ROOT / "outputs" / f"{args.name}-full"
    if out.exists():
        raise SystemExit(f"{out} already exists; pick another --name")
    n_oof = average([m / "oof" for m in members], out / "oof", weights)
    n_test = average([m / "test_prob" for m in members], out / "test_prob", weights)
    (out / "members.json").write_text(json.dumps({"members": args.exps, "weights": weights, "n_oof": n_oof, "n_test": n_test}, indent=1), encoding="utf-8")
    print(f"{out}: averaged {len(members)} members, oof {n_oof} test {n_test}")


if __name__ == "__main__":
    main()
