"""End-to-end pipeline called by main.py: cache -> per-fold training -> OOF probabilities -> post-processing ->
official PQ per fold and pooled -> test prediction (fold average) -> submission.csv.

Output contract (experiments.md §3): METRIC lines via `report_metric`, and in EXP_DIR:
    oof/<stem>.npy        uint8 probability x255 at input.size (every experiment keeps these: ensemble material)
    test_prob/<stem>.npy  uint8 fold-averaged test probability
    oof.parquet           OOF instance rows: filament_id, segmentation_rle, fold, score
    submission.csv        test instances in the competition format (validated)
    weights/fold<k>.pt    model weights
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.annotations import IMAGE_SIZE, annotations_to_gt_df, load_coco
from src.data import Cache, FilamentDataset
from src.metric import PQResult, evaluate
from src.models import build_model
from src.postprocess import instances_from_prob, instances_to_rows
from src.submission import validate_submission, write_submission
from src.train import predict, train_fold


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_gt(cache_root: Path, annotations: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth RLE rows for every train record, cached once (independent of the input config)."""
    path = cache_root / "gt_rles.parquet"
    if path.exists():
        return pd.read_parquet(path)
    log("rasterising ground truth ...")
    gt = annotations_to_gt_df(annotations)
    path.parent.mkdir(parents=True, exist_ok=True)
    gt.to_parquet(path, index=False)
    return gt


def _subset(stems: list[str], limit: int | None) -> list[str]:
    return stems if limit is None else stems[:limit]


def postprocess_probs(probs: dict[str, np.ndarray], cache: Cache, pp_cfg: dict, fold: int | None = None) -> list[dict]:
    rows = []
    for stem, prob in probs.items():
        disk_mask = (cache.disk(stem).radius_map(IMAGE_SIZE) <= pp_cfg["disk_margin"]).astype(np.uint8)
        inst = instances_from_prob(prob.astype(np.float32), pp_cfg["threshold"], pp_cfg["min_area"], pp_cfg["closing_radius"],
                                   disk_mask, merge_radius=pp_cfg.get("merge_radius", 0), score_min=pp_cfg.get("score_min", 0.0),
                                   seed_threshold=pp_cfg.get("seed_threshold", 0.0))
        for r in instances_to_rows(stem, inst):
            rows.append({**r, "fold": fold})
    return rows


def score_components(oof_probs: dict, test_probs: dict, cache: Cache, pp_cfg: dict, gt: pd.DataFrame, fold_of: dict,
                     workers: int, exp_dir: Path, log) -> tuple[dict[int, list[dict]], list[dict]]:
    """GBDT expected-IoU gate (src/scorer.py): OOF components of fold k are scored by a model fitted on the other folds,
    test components by a model fitted on all OOF components; keep u_hat > pp_cfg['tau']. Returns rows per fold and test rows."""
    from src.scorer import extract_components, gt_records_by_stem, rows_from_components, score_oof, score_test

    tau = float(pp_cfg["tau"])
    t0 = time.time()
    oof_df = extract_components(oof_probs, cache, pp_cfg, gt_records_by_stem(gt), workers)
    test_df = extract_components(test_probs, cache, pp_cfg, None, workers)
    log(f"scorer: {len(oof_df)} oof / {len(test_df)} test components in {(time.time() - t0) / 60:.1f} min")
    oof_df["fold"] = oof_df["stem"].map(fold_of)
    seeds = tuple(int(s) for s in pp_cfg.get("scorer_seeds") or [42])
    oof_df["u_hat"] = score_oof(oof_df, fold_of, seeds)
    test_df["u_hat"] = score_test(oof_df, test_df, seeds)
    oof_df.to_parquet(exp_dir / "scorer_oof.parquet", index=False)
    test_df.to_parquet(exp_dir / "scorer_test.parquet", index=False)
    keep_oof, keep_test = oof_df["u_hat"].values > tau, test_df["u_hat"].values > tau
    log(f"scorer: tau {tau}: keep {keep_oof.mean():.3f} of oof, {keep_test.mean():.3f} of test components")
    per_fold = {f: rows_from_components(oof_df[oof_df["fold"] == f], keep_oof[(oof_df["fold"] == f).values])
                for f in sorted(oof_df["fold"].unique())}
    return per_fold, rows_from_components(test_df, keep_test)


def load_saved_probs(src_dir: Path) -> dict[str, np.ndarray]:
    """uint8 x255 probability maps saved by an earlier experiment -> stem -> float32 in [0, 1]."""
    return {p.stem: np.load(p).astype(np.float32) / 255.0 for p in sorted(src_dir.glob("*.npy"))}


def evaluate_rows(gt: pd.DataFrame, rows: list[dict], stems: list[str]) -> PQResult:
    gt_sub = gt[gt["filament_id"].str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0].isin(stems)]
    pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]) if rows else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    return evaluate(gt_sub, pred)


def pseudo_stems(root: Path, pl_cfg: dict, forbidden: set[str]) -> dict[str, Path]:
    """train.pseudo.list: a FROZEN csv (stem, keep) of unlabelled 2023-2026 GONG H-alpha frames in
    train.pseudo.images_dir (tools/gong_download.py; training only, host ruling #744047), written by
    tools/gong_pseudo_list.py with a label-free, image-only quality filter. Returns stem -> path for keep rows, at most
    n_images (fixed random subset, train.pseudo.seed). Guards: every stem is dated >= 2023-01-01 and none is a
    train/test stem (`forbidden`), so the y_paths override can never replace a real image's target."""
    d = root / pl_cfg["images_dir"]
    lst = pd.read_csv(root / pl_cfg["list"])
    stems = sorted(lst.loc[lst["keep"].astype(bool), "stem"])
    bad = [s for s in stems if s[:8] < "20230101" or s in forbidden or not (d / f"{s}.jpg").exists()]
    if bad:
        raise ValueError(f"pseudo frames outside 2023+, colliding with train/test, or missing: {bad[:5]}")
    n = int(pl_cfg.get("n_images", 0) or 0)
    if 0 < n < len(stems):
        stems = sorted(np.random.default_rng(int(pl_cfg.get("seed", 0))).choice(stems, n, replace=False))
    return {s: d / f"{s}.jpg" for s in stems}


def run(cfg, exp_dir: Path, mode: str, report_metric) -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / cfg.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"mode={mode} device={device} exp_dir={exp_dir}")
    if cfg.get("release", {}).get("enabled", False):
        # inference from the shipped weights only (src/release.py; configs/final_release.yaml): no training data needed
        from src.release import run_release
        run_release(cfg, exp_dir, root, device, log)
        return

    records, annotations = load_coco(data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    folds = pd.read_csv(root / cfg.folds_csv)
    fold_of = dict(zip(folds["stem"], folds["fold"]))
    train_images = {r.stem: data_dir / "train" / "train_images" / r.file_name for r in records.drop_duplicates("stem").itertuples()}
    test_images = {p.stem: p for p in sorted((data_dir / "test" / "test_images").glob("*.jpeg"))}
    test_stems = _subset(list(test_images), cfg.limits.test)

    cache = Cache(root / cfg.cache_dir, dict(cfg.input), dict(cfg.get("target", {})))
    gt = load_gt(root / cfg.cache_dir, annotations)

    oof_dir, test_dir, w_dir = exp_dir / "oof", exp_dir / "test_prob", exp_dir / "weights"
    for d in (oof_dir, test_dir, w_dir):
        d.mkdir(parents=True, exist_ok=True)
    if cfg.get("refiner", {}).get("enabled", False):
        # two-stage pilot: per-instance crop refiner on the components of an earlier experiment (src/refiner.py)
        from src.refiner import run_refiner
        run_refiner(cfg, exp_dir, mode, report_metric, root, cache, gt, fold_of, test_stems, device, log)
        return

    oof_probs: dict[str, np.ndarray] = {}
    oof_rows: list[dict] = []
    fold_scores: dict[int, float] = {}
    test_acc: dict[str, np.ndarray] = {}
    train_minutes = 0.0
    source_exp = cfg.postprocess.get("source_exp")
    scorer_rows: dict[int, list[dict]] = {}
    scorer_test_rows: list[dict] = []
    if source_exp:
        # post-processing-only experiment: reuse the saved OOF / test probabilities of `source_exp` (no training)
        src = root / "outputs" / f"{source_exp}-full"
        log(f"postprocess-only run on {src}")
        oof_probs = load_saved_probs(src / "oof")
        test_acc = {s: p for s, p in load_saved_probs(src / "test_prob").items() if s in set(test_stems)}  # limits.test in --debug
        cache.build({s: train_images[s] for s in oof_probs}, annotations, log)
        cache.build({s: test_images[s] for s in test_acc}, None, log)
        if cfg.postprocess.get("scorer") == "gbdt":
            scorer_rows, scorer_test_rows = score_components(oof_probs, test_acc, cache, dict(cfg.postprocess), gt, fold_of,
                                                             cfg.train.num_workers, exp_dir, log)
        for fold in cfg.train.folds:
            va = [s for s in oof_probs if fold_of[s] == fold]
            if scorer_rows:
                rows = [{**r, "fold": fold} for r in scorer_rows.get(fold, [])]
            else:
                rows = postprocess_probs({s: oof_probs[s] for s in va}, cache, dict(cfg.postprocess), fold)
            res = evaluate_rows(gt, rows, va)
            fold_scores[fold] = res.pq
            oof_rows.extend(rows)
            log(f"fold {fold}: PQ {res.pq:.4f} SQ {res.sq:.4f} RQ {res.rq:.4f} TP/FP/FN {res.tp}/{res.fp}/{res.fn} preds/img {len(rows) / max(len(va), 1):.1f}")
            report_metric(f"fold{fold}", res.pq)
        for s, p in oof_probs.items():
            np.save(oof_dir / f"{s}.npy", np.round(p * 255).astype(np.uint8))
    for fold in ([] if source_exp else cfg.train.folds):
        tr = _subset([s for s in train_images if fold_of[s] != fold], cfg.limits.train)
        va = _subset([s for s in train_images if fold_of[s] == fold], cfg.limits.val)
        cache.build({s: train_images[s] for s in tr + va}, annotations, log)
        cache.build({s: test_images[s] for s in test_stems}, None, log)
        log(f"fold {fold}: train {len(tr)} val {len(va)}")

        torch.manual_seed(cfg.seed + fold)
        np.random.seed(cfg.seed + fold)
        tcfg = cfg.get("target", {})
        spine_head = int(tcfg.get("spine_width", 0)) > 0 and float(cfg.train.get("skel_recall_weight", 0.0)) <= 0
        classes = 1 + int(spine_head or int(tcfg.get("sep_width", 0)) > 0)
        model = build_model(dict(cfg.model), in_channels=len(cfg.input.channels), classes=classes)
        init_from = cfg.train.get("init_from")
        if init_from:
            # fine-tuning: start fold k from the fold-k weights of an earlier run (same folds.csv, so fold k's model never
            # saw fold k's images), then train train.epochs at train.lr as configured
            i_path = root / "outputs" / init_from / "weights" / f"fold{fold}.pt"
            model.load_state_dict(torch.load(i_path, map_location="cpu"))
            log(f"fold {fold}: init from {i_path} md5 {hashlib.md5(i_path.read_bytes()).hexdigest()}")
        weights_from = cfg.infer.get("weights_from")
        resume_from = cfg.train.get("resume_from")
        w_path = None
        if weights_from:
            # inference-only run (e.g. TTA on saved weights): reuse outputs/<weights_from>/weights/fold<k>.pt, no training
            w_path = root / "outputs" / weights_from / "weights" / f"fold{fold}.pt"
        elif resume_from and (root / "outputs" / resume_from / "weights" / f"fold{fold}.pt").exists():
            # resume an interrupted run of the SAME config: folds whose weights were already saved are loaded, the
            # others are trained as usual (fold seeds are seed + fold, so they do not depend on the earlier folds)
            w_path = root / "outputs" / resume_from / "weights" / f"fold{fold}.pt"
        if w_path is not None:
            model.load_state_dict(torch.load(w_path, map_location="cpu"))
            log(f"fold {fold}: loaded {w_path} md5 {hashlib.md5(w_path.read_bytes()).hexdigest()}")  # provenance (leak audit)
            if not weights_from:
                torch.save(model.state_dict(), w_dir / f"fold{fold}.pt")  # keep the resumed run self-contained
        else:
            pl_cfg = dict(cfg.train.get("pseudo") or {})
            y_paths: dict[str, Path] = {}
            if pl_cfg.get("labels_from"):
                # student: pseudo-labels of fold k come from the fold-k TEACHER (never trained on fold k's images)
                pl_dir = root / "outputs" / pl_cfg["labels_from"] / "pl" / f"fold{fold}"
                pl_images = pseudo_stems(root, pl_cfg, set(train_images) | set(test_images))
                missing = [s for s in pl_images if not (pl_dir / f"{s}.npy").exists()]
                if missing:
                    raise FileNotFoundError(f"{len(missing)} of {len(pl_images)} pseudo-labels missing in {pl_dir} (e.g. {missing[:3]})")
                cache.build(pl_images, None, log)
                y_paths = {s: pl_dir / f"{s}.npy" for s in pl_images}
                log(f"fold {fold}: + {len(y_paths)} pseudo-labelled frames from {pl_dir}")
            ratio = float(pl_cfg.get("ratio", 0.0) or 0.0)
            # train.record_weight: sample images in proportion to their annotator-record count (the metric pools over
            # records, so a 3-record image weighs 3x in PQ but only 1x in a plain per-image epoch)
            stem_w = None
            if cfg.train.get("record_weight", False):
                n_rec = records.groupby("stem").size()
                stem_w = {s: float(n_rec.get(s, 1)) for s in tr}
                log(f"fold {fold}: record-weighted sampling, mean weight {np.mean(list(stem_w.values())):.2f}")
            # ratio 0: pseudo frames are appended (longer epochs); ratio > 0: fixed epoch length, frames mixed in at `ratio`
            train_ds = FilamentDataset(tr + ([] if ratio > 0 else sorted(y_paths)), cache, dict(cfg.input.norm), train=True,
                                       aug=dict(cfg.train.aug), tile_size=cfg.train.get("tile_size", 0),
                                       tiles_per_image=cfg.train.get("tiles_per_image", 1),
                                       positive_frac=cfg.train.get("tile_positive_frac", 0.5), y_paths=y_paths,
                                       pseudo_stems=sorted(y_paths) if ratio > 0 else None, pseudo_ratio=ratio,
                                       stem_weights=stem_w)
            stats = train_fold(model, train_ds, dict(cfg.train), cfg.seed + fold, device, log)
            train_minutes += stats["train_min"]
            torch.save(model.state_dict(), w_dir / f"fold{fold}.pt")

        val_ds = FilamentDataset(va, cache, dict(cfg.input.norm), train=False)
        pred_cfg = {**dict(cfg.train), "batch_size": cfg.infer.get("batch_size") or cfg.train.batch_size,  # full frames
                    "sep_suppress": bool(cfg.infer.get("sep_suppress", False)),
                    "tta_scales": list(cfg.infer.get("tta_scales") or [])}
        probs = predict(model, val_ds, pred_cfg, device, tta=cfg.infer.tta, tta_rot90=cfg.infer.get("tta_rot90", False))
        for s, p in probs.items():
            np.save(oof_dir / f"{s}.npy", np.round(p.astype(np.float32) * 255).astype(np.uint8))
        rows = postprocess_probs(probs, cache, dict(cfg.postprocess), fold)
        res = evaluate_rows(gt, rows, va)
        fold_scores[fold] = res.pq
        oof_probs.update(probs)
        oof_rows.extend(rows)
        log(f"fold {fold}: PQ {res.pq:.4f} SQ {res.sq:.4f} RQ {res.rq:.4f} TP/FP/FN {res.tp}/{res.fp}/{res.fn} "
            f"preds/img {len(rows) / max(len(va), 1):.1f}")
        report_metric(f"fold{fold}", res.pq)

        pl_cfg = dict(cfg.train.get("pseudo") or {})
        if pl_cfg.get("make"):
            # teacher: label the unlabelled frames with this fold's model (used only by the fold-k student)
            pl_images = pseudo_stems(root, pl_cfg, set(train_images) | set(test_images))
            cache.build(pl_images, None, log)
            pl_out = exp_dir / "pl" / f"fold{fold}"
            pl_out.mkdir(parents=True, exist_ok=True)
            pl_ds = FilamentDataset(sorted(pl_images), cache, dict(cfg.input.norm), train=False, with_target=False)
            for s, p in predict(model, pl_ds, pred_cfg, device, tta=cfg.infer.tta).items():
                np.save(pl_out / f"{s}.npy", np.round(p.astype(np.float32) * 255).astype(np.uint8))
            log(f"fold {fold}: pseudo-labelled {len(pl_images)} frames -> {pl_out}")

        test_ds = FilamentDataset(test_stems, cache, dict(cfg.input.norm), train=False, with_target=False)
        for s, p in predict(model, test_ds, pred_cfg, device, tta=cfg.infer.tta, tta_rot90=cfg.infer.get("tta_rot90", False)).items():
            test_acc[s] = test_acc.get(s, 0) + p.astype(np.float32) / len(cfg.train.folds)
        del model
        torch.cuda.empty_cache()

    scores = np.array(list(fold_scores.values()))
    report_metric("cv", float(scores.mean()))
    report_metric("cv_std", float(scores.std(ddof=0)) if len(scores) > 1 else 0.0)
    pooled = evaluate_rows(gt, oof_rows, list(oof_probs))
    for k, v in pooled.as_metrics().items():
        report_metric("oof_cv" if k == "pq" else k, v)
    report_metric("train_min", train_minutes)
    pd.DataFrame(oof_rows).to_parquet(exp_dir / "oof.parquet", index=False)
    (exp_dir / "oof_summary.json").write_text(json.dumps({
        "folds": fold_scores, "pooled": pooled.as_metrics(),
        "one_to_many": dict(pooled.one_to_many), "many_to_one": dict(pooled.many_to_one),
        "tp_iou_quantiles": np.quantile(pooled.tp_ious, [0.1, 0.5, 0.9]).tolist() if pooled.tp_ious else [],
    }, indent=1, default=int), encoding="utf-8")

    for s, p in test_acc.items():
        np.save(test_dir / f"{s}.npy", np.round(p * 255).astype(np.uint8))
    test_rows = scorer_test_rows if scorer_rows else postprocess_probs(test_acc, cache, dict(cfg.postprocess))
    sub_rows = [{k: r[k] for k in ("filament_id", "segmentation_rle")} for r in test_rows]
    sub = write_submission(sub_rows, exp_dir / "submission.csv")
    summary = validate_submission(sub, expected_stems=test_stems)
    log(f"submission: {summary}")
    report_metric("test_preds_per_image", summary["per_image_mean"])
