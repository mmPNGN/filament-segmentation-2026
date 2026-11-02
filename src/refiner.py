"""Stage 2 of a two-stage pipeline (pilot, 2026-10-08): a per-instance crop refiner on top of the stage-1 components.

Stage 1 is an existing post-processing-only experiment (refiner.stage1, e.g. exp079-ensemble): its components with the
nested GBDT score u_hat (scorer_oof.parquet / scorer_test.parquet) and the probability maps it was built from
(refiner.maps, e.g. ens_dml_effb4). Stage 2 sees, for one component at a time, a window around it at native 2048
resolution (refiner.crop px, resized down when the component is larger) with three input channels:
    flattened H-alpha image, stage-1 probability, seed = this component's mask
and predicts the INSTANCE-level soft target: for every annotator record of the image, the union of the filaments that
belong to the seed (the seed covers >= 30 % of the filament, or lies >= 50 % inside it), averaged over the records
(a record without such a filament contributes zeros). This is the expected annotator mask of this one instance, so the
refiner can grow a too-short seed, trim it, or ignore neighbours that the seed touches.

The set of kept components (u_hat > postprocess.tau) is NOT changed; only their masks are replaced by
refined probability > refiner.threshold (kept in descending u_hat order without pixel overlap). Leakage: the refiner of
fold k is trained on crops of images outside fold k only; the stage-1 maps of those images are OOF predictions and
u_hat is nested by fold (same stacking as the GBDT gate). Test crops use the mean of the 5 fold refiners.
Crops are cached once in data/processed/refiner/<key>/ (uint8 memmap, N x 4 x crop x crop: flat, prob, seed, target).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu
from torch.utils.data import Dataset

from src.annotations import IMAGE_SIZE, mask_to_rle
from src.models import build_model
from src.train import _flip_tta, train_fold

FLAT_LO, FLAT_SPAN = 0.4, 1.0   # flat values are quantised to uint8 over [0.4, 1.4] (quiet Sun 1.0, filaments ~0.87)
CROP_VERSION = 1                 # bump when the crop / target rules below change (part of the crop cache key)
S = IMAGE_SIZE


def _obj(counts: str) -> dict:
    return {"size": [S, S], "counts": counts.encode("ascii")}


def crop_window(rle: str, crop: int, pad: int) -> tuple[int, int, int]:
    """Square window (x0, y0, side) centred on the component's bounding box, at least `crop` px wide."""
    x, y, w, h = mu.toBbox(_obj(rle))
    side = int(max(crop, max(w, h) + 2 * pad))
    return int(round(x + w / 2 - side / 2)), int(round(y + h / 2 - side / 2)), side


def _window(a: np.ndarray, x0: int, y0: int, side: int, fill: float) -> np.ndarray:
    out = np.full((side, side), fill, dtype=a.dtype)
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + side, a.shape[1]), min(y0 + side, a.shape[0])
    if xe > xs and ye > ys:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
    return out


def _fit(a: np.ndarray, crop: int, nearest: bool = False) -> np.ndarray:
    if a.shape[0] == crop:
        return a
    return cv2.resize(a, (crop, crop), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def instance_target(seed_rle: str, records: dict[str, list[str]]) -> np.ndarray:
    """Mean over records of the union of the record's filaments that belong to the seed (full 2048 frame, float32)."""
    seed = _obj(seed_rle)
    sa = float(mu.area(seed))
    acc = np.zeros((S, S), np.float32)
    for rles in records.values():
        objs = [_obj(r) for r in rles]
        if not objs:
            continue
        inter = np.array([mu.area(mu.merge([seed, g], intersect=True)) for g in objs], dtype=np.float64)
        ga = mu.area(objs).astype(np.float64)
        own = [g for g, i, a in zip(objs, inter, ga) if i >= 0.3 * a or i >= 0.5 * sa]
        if own:
            acc += mu.decode(mu.merge(own)).astype(np.float32)
    return acc / max(len(records), 1)


def _image_crops(args) -> list[tuple[dict, np.ndarray]]:
    stem, x_path, map_path, comps, records, crop, pad = args
    flat = np.load(x_path, mmap_mode="r")[0]
    prob = np.load(map_path, mmap_mode="r")
    out = []
    for c in comps:
        x0, y0, side = crop_window(c["rle"], crop, pad)
        f = _window(flat, x0, y0, side, 1.0).astype(np.float32)
        f = np.clip((f - FLAT_LO) / FLAT_SPAN * 255, 0, 255).astype(np.uint8)
        p = _window(prob, x0, y0, side, 0)
        s = _window(mu.decode(_obj(c["rle"])) * 255, x0, y0, side, 0).astype(np.uint8)
        t = (_window(instance_target(c["rle"], records), x0, y0, side, 0.0) * 255).astype(np.uint8) if records else np.zeros_like(s)
        arr = np.stack([_fit(f, crop), _fit(p, crop), _fit(s, crop, nearest=True), _fit(t, crop)])
        out.append(({**{k: c[k] for k in ("stem", "comp", "fold", "u_hat", "rle", "area")}, "x0": x0, "y0": y0, "side": side}, arr))
    return out


def build_crops(stage1: Path, maps: Path, cache, gt_by_stem: dict, rcfg: dict, out_dir: Path, workers: int, log,
                stems_limit: set[str] | None = None) -> pd.DataFrame:
    """Write crops.npy (uint8 memmap N x 4 x crop x crop) + meta.parquet once; return meta (row i <-> crops[i])."""
    meta_path, arr_path = out_dir / "meta.parquet", out_dir / "crops.npy"
    if meta_path.exists() and arr_path.exists() and stems_limit is None:
        return pd.read_parquet(meta_path)
    crop, pad = int(rcfg["crop"]), int(rcfg["pad"])
    frames = []
    for split in ("oof", "test"):
        df = pd.read_parquet(stage1 / f"scorer_{split}.parquet")
        df = df.assign(split=split, fold=df["fold"] if "fold" in df else -1)
        frames.append(df[["stem", "comp", "fold", "u_hat", "rle", "area", "split"]])
    comps = pd.concat(frames, ignore_index=True)
    if stems_limit is not None:
        comps = comps[comps["stem"].isin(stems_limit)]
    jobs = []
    for (split, stem), g in comps.groupby(["split", "stem"], sort=True):
        mp = maps / ("oof" if split == "oof" else "test_prob") / f"{stem}.npy"
        jobs.append((stem, str(cache.x_path(stem)), str(mp), g.to_dict("records"),
                     gt_by_stem.get(stem) if split == "oof" else None, crop, pad))
    out_dir.mkdir(parents=True, exist_ok=True)
    n = sum(len(j[3]) for j in jobs)
    mm = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8, shape=(n, 4, crop, crop))
    rows, i = [], 0
    with ProcessPoolExecutor(max(workers, 1)) as pool:
        for k, res in enumerate(pool.map(_image_crops, jobs, chunksize=2)):
            for m, arr in res:
                mm[i] = arr
                rows.append({**m, "split": "oof" if m["fold"] >= 0 else "test", "idx": i})
                i += 1
            if (k + 1) % 100 == 0:
                log(f"refiner crops: {k + 1}/{len(jobs)} images")
    mm.flush()
    meta = pd.DataFrame(rows)
    meta.to_parquet(meta_path, index=False)
    log(f"refiner crops: {len(meta)} ({(meta.split == 'oof').sum()} oof, {(meta.split == 'test').sum()} test) -> {out_dir}")
    return meta


class RefinerCrops(Dataset):
    """(x: 3 x C x C float32, y: 1 x C x C soft target, key) from the crop memmap; flips + rot90 when training."""

    def __init__(self, arr_path: Path, idx: list[int], train: bool):
        self.arr_path, self.idx, self.train = arr_path, list(idx), train
        self.mm = None

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        if self.mm is None:  # opened lazily so each DataLoader worker maps the file itself
            self.mm = np.load(self.arr_path, mmap_mode="r")
        a = np.asarray(self.mm[self.idx[i]], dtype=np.float32) / 255.0
        if self.train:
            if np.random.rand() < 0.5:
                a = a[:, :, ::-1]
            if np.random.rand() < 0.5:
                a = a[:, ::-1, :]
            a = np.rot90(a, np.random.randint(4), axes=(1, 2))
        a = np.ascontiguousarray(a)
        flat = (a[0] * FLAT_SPAN + FLAT_LO - 1.0) / 0.12
        x = np.stack([flat, a[1] * 2 - 1, a[2] * 2 - 1]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(a[3:4].copy()), str(self.idx[i])


@torch.no_grad()
def predict_crops(model: torch.nn.Module, ds: RefinerCrops, batch_size: int, device: torch.device) -> np.ndarray:
    model.to(device).eval()
    out = []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for x, _, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p = _flip_tta(model, x.to(device), False)
        out.append(p.float().cpu().numpy()[:, 0])
    return np.concatenate(out) if out else np.zeros((0, 1, 1), np.float32)


def assemble(meta: pd.DataFrame, refined: dict[int, np.ndarray], tau: float, thr: float, keep_frac: float,
             blend: float = 1.0, arr_path: Path | None = None) -> list[dict]:
    """Rows for the kept components (u_hat > tau) with refined masks, pixel-disjoint in descending u_hat order.
    blend < 1: mask = blend * refined + (1 - blend) * stage-1 probability. A refined mask smaller than keep_frac x the
    stage-1 component falls back to the stage-1 mask."""
    mm = np.load(arr_path, mmap_mode="r") if blend < 1.0 else None
    rows = []
    for stem, g in meta[meta.u_hat > tau].groupby("stem", sort=False):
        taken = np.zeros((S, S), bool)
        k = 0
        for r in g.sort_values("u_hat", ascending=False).itertuples():
            p = refined[r.idx]
            if mm is not None:
                p = blend * p + (1 - blend) * np.asarray(mm[r.idx, 1], dtype=np.float32) / 255.0
            win = cv2.resize(p, (r.side, r.side), interpolation=cv2.INTER_LINEAR) if r.side != p.shape[0] else p
            m = np.zeros((S, S), bool)
            m_win = win > thr
            ys, xs = max(r.y0, 0), max(r.x0, 0)
            ye, xe = min(r.y0 + r.side, S), min(r.x0 + r.side, S)
            m[ys:ye, xs:xe] = m_win[ys - r.y0:ye - r.y0, xs - r.x0:xe - r.x0]
            if m.sum() < keep_frac * r.area:
                m = mu.decode(_obj(r.rle)).astype(bool)
            m &= ~taken
            if m.sum() < 20:
                continue
            taken |= m
            k += 1
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": mask_to_rle(m), "score": float(r.u_hat),
                         "area": int(m.sum())})
    return rows


def run_refiner(cfg, exp_dir: Path, mode: str, report_metric, root: Path, cache, gt: pd.DataFrame, fold_of: dict,
                test_stems: list[str], device: torch.device, log) -> None:
    from src.pipeline import evaluate_rows
    from src.scorer import gt_records_by_stem, rows_from_components
    from src.submission import validate_submission, write_submission

    rcfg = dict(cfg.refiner)
    stage1, maps = root / "outputs" / rcfg["stage1"], root / "outputs" / f"{rcfg['maps']}-full"
    tau, thr, keep_frac = float(cfg.postprocess.tau), float(rcfg["threshold"]), float(rcfg["keep_frac"])
    key = f"v{CROP_VERSION}_{rcfg['stage1']}_{rcfg['maps']}_{cache.dir.name}_c{rcfg['crop']}_p{rcfg['pad']}"
    limit = None
    if mode == "debug":
        oof_stems = sorted(pd.read_parquet(stage1 / "scorer_oof.parquet")["stem"].unique())
        limit = set(oof_stems[:: max(len(oof_stems) // 12, 1)][:12]) | set(test_stems[:2])
        key += "_debug"
    out_dir = root / "data" / "processed" / "refiner" / key
    meta = build_crops(stage1, maps, cache, gt_records_by_stem(gt), rcfg, out_dir, int(cfg.train.num_workers), log, limit)
    arr_path = out_dir / "crops.npy"
    oof, test = meta[meta.split == "oof"], meta[meta.split == "test"]
    tcfg = {**dict(cfg.train), **dict(rcfg["train"])}
    if mode == "debug":
        tcfg.update(epochs=1, max_steps=3, num_workers=0)
    folds = [int(f) for f in sorted(oof["fold"].unique())][: 1 if mode == "debug" else None]
    w_dir = exp_dir / "weights"
    w_dir.mkdir(parents=True, exist_ok=True)
    refined: dict[int, np.ndarray] = {}
    test_acc: dict[int, np.ndarray] = {}
    for fold in folds:
        tr, va = oof[oof["fold"] != fold], oof[oof["fold"] == fold]
        log(f"refiner fold {fold}: train {len(tr)} crops, val {len(va)} crops")
        torch.manual_seed(cfg.seed + fold)
        np.random.seed(cfg.seed + fold)
        model = build_model({"arch": "Unet", "encoder": rcfg["encoder"], "encoder_weights": "imagenet"}, in_channels=3)
        train_fold(model, RefinerCrops(arr_path, tr["idx"].tolist(), train=True), tcfg, cfg.seed + fold, device, log)
        torch.save(model.state_dict(), w_dir / f"fold{fold}.pt")
        for i, p in zip(va["idx"], predict_crops(model, RefinerCrops(arr_path, va["idx"].tolist(), False), 32, device)):
            refined[int(i)] = p
        for i, p in zip(test["idx"], predict_crops(model, RefinerCrops(arr_path, test["idx"].tolist(), False), 32, device)):
            test_acc[int(i)] = test_acc.get(int(i), 0) + p / len(folds)
        del model
        torch.cuda.empty_cache()

    scores, base_scores, blend_scores, oof_rows = {}, {}, {}, []
    for fold in folds:
        va = oof[oof["fold"] == fold]
        stems = sorted(va["stem"].unique()) if mode == "debug" else sorted(s for s, f in fold_of.items() if f == fold)
        rows = assemble(va, refined, tau, thr, keep_frac)
        base = rows_from_components(va, (va.u_hat > tau).values)
        blend = assemble(va, refined, tau, thr, keep_frac, blend=0.5, arr_path=arr_path)
        scores[fold] = evaluate_rows(gt, rows, stems).pq
        base_scores[fold] = evaluate_rows(gt, base, stems).pq
        blend_scores[fold] = evaluate_rows(gt, blend, stems).pq
        oof_rows += [{**r, "fold": fold} for r in rows]
        log(f"refiner fold {fold}: PQ {scores[fold]:.4f} (stage 1 {base_scores[fold]:.4f}, blend 0.5 {blend_scores[fold]:.4f})")
        report_metric(f"fold{fold}", scores[fold])
    s = np.array(list(scores.values()))
    report_metric("cv", float(s.mean()))
    report_metric("cv_std", float(s.std(ddof=0)) if len(s) > 1 else 0.0)
    report_metric("stage1_cv", float(np.mean(list(base_scores.values()))))
    report_metric("blend_cv", float(np.mean(list(blend_scores.values()))))
    pooled = evaluate_rows(gt, oof_rows, sorted(oof[oof["fold"].isin(folds)]["stem"].unique()) if mode == "debug"
                           else sorted(s for s, f in fold_of.items() if f in folds))
    for k, v in pooled.as_metrics().items():
        report_metric("oof_cv" if k == "pq" else k, v)
    pd.DataFrame(oof_rows).to_parquet(exp_dir / "oof.parquet", index=False)
    (exp_dir / "refiner_summary.json").write_text(json.dumps(
        {"refined": scores, "stage1": base_scores, "blend": blend_scores, "crops_dir": str(out_dir)}, indent=1, default=float),
        encoding="utf-8")
    test_rows = assemble(test, test_acc, tau, thr, keep_frac)
    sub = write_submission([{k: r[k] for k in ("filament_id", "segmentation_rle")} for r in test_rows], exp_dir / "submission.csv")
    summary = validate_submission(sub, expected_stems=test_stems if mode != "debug" else None)
    log(f"submission: {summary}")
    report_metric("test_preds_per_image", summary["per_image_mean"])
