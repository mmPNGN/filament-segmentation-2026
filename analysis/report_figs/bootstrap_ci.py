"""Uncertainty of the OOF PQ for the report (template asks for error bars): 95 % bootstrap interval over images
(resampling the 707 train images with their records, micro-pooled PQ per resample) and the paired bootstrap of the
difference between two experiments' kept sets on the same resamples. Usage:
uv run python analysis/report_figs/bootstrap_ci.py exp118 exp079"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.scorer as scorer  # noqa: E402
from analysis.sparse_label_tau import IOU_T, rle_obj  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

TAU, N_BOOT = 0.20, 2000


def per_image(exp: str, gt: dict) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "outputs" / f"{exp}-ensemble" / "scorer_oof.parquet", columns=["stem", "rle", "u_hat"])
    pred = {s: [rle_obj(r) for r in g["rle"]] for s, g in df[df.u_hat > TAU].groupby("stem")}
    rows = []
    for stem, recs in gt.items():
        p = pred.get(stem, [])
        s = tp = fp = fn = 0
        for gt_rles in recs.values():
            g = [rle_obj(r) for r in gt_rles]
            if not p:
                fn += len(g); continue
            iou = np.asarray(mask_util.iou(p, g, [0] * len(g))).T.reshape(len(g), len(p))
            hit = iou > IOU_T
            s += float(iou[hit].sum()); tp += int(hit.sum())
            fp += int((hit.sum(axis=0) == 0).sum()); fn += int((hit.sum(axis=1) == 0).sum())
        rows.append((stem, s, tp, fp, fn))
    return pd.DataFrame(rows, columns=["stem", "s", "tp", "fp", "fn"]).set_index("stem")


def pq(a: np.ndarray) -> np.ndarray:
    return a[..., 0] / (a[..., 1] + 0.5 * a[..., 2] + 0.5 * a[..., 3])


def main(exps: list[str]) -> None:
    gt = scorer.gt_records_by_stem(pd.read_parquet(ROOT / "data/processed/gt_rles.parquet"))
    tabs = {e: per_image(e, gt).sort_index() for e in exps}
    stems = tabs[exps[0]].index
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(stems), size=(N_BOOT, len(stems)))
    boots = {}
    for e, t in tabs.items():
        arr = t.loc[stems].to_numpy(dtype=float)
        boots[e] = pq(arr[idx].sum(axis=1))
        lo, hi = np.percentile(boots[e], [2.5, 97.5])
        print(f"{e}: PQ {pq(arr.sum(axis=0)):.4f}, 95% bootstrap CI [{lo:.4f}, {hi:.4f}], sd {boots[e].std():.4f}")
    for e in exps[1:]:
        d = boots[exps[0]] - boots[e]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{exps[0]} - {e}: {d.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], P(>0) {np.mean(d > 0):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
