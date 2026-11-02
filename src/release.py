"""Release inference: rebuild the final submission from the shipped weights, without any training (host requirement,
Final Submission page / discussion #753665: the weights live in the repository and the demo notebook loads them).

Layout (repository root):
    weights/<member>/fold<k>.part<j>.pt   fp16 state-dict shards (<= 90 MB each, GitHub's file limit is 100 MB)
    models/gate/lgb_seed<s>.txt           LightGBM expected-IoU gate boosters, fitted on all out-of-fold components
    models/gate/features.json             their feature order

Chain (identical to the training pipeline, src/pipeline.py + tools/average_probs.py):
    member m: mean over its 5 fold models of the 4-way flip-TTA probability -> uint8 round(p * 255)
    ensemble: equal-weight mean of the member uint8 maps -> uint8 round
    connected components (p > 0.5, closing 2, >= 300 px, no merging) -> features -> u_hat = mean over the boosters
    keep u_hat > tau, ranked by u_hat, pixel-disjoint -> submission.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SHARD_BYTES = 90_000_000


def save_sharded(state_dict: dict, out_dir: Path, name: str) -> list[Path]:
    """fp16 copy of the floating tensors (integers kept), split into shards of at most SHARD_BYTES."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shards, cur, size = [], {}, 0
    for k, v in state_dict.items():
        v = v.detach().cpu()
        v = v.half() if v.is_floating_point() else v
        n = v.numel() * v.element_size()
        if cur and size + n > SHARD_BYTES:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = v
        size += n
    shards.append(cur)
    paths = []
    for j, shard in enumerate(shards):
        p = out_dir / f"{name}.part{j}.pt"
        torch.save(shard, p)
        paths.append(p)
    return paths


def load_sharded(member_dir: Path, name: str) -> dict:
    """Merge the shards of one fold model; floating tensors back to float32."""
    parts = sorted(member_dir.glob(f"{name}.part*.pt"), key=lambda p: int(p.stem.rsplit("part", 1)[1]))
    if not parts:
        raise FileNotFoundError(f"no weights {member_dir / name}.part*.pt")
    sd = {}
    for p in parts:
        sd.update(torch.load(p, map_location="cpu"))
    return {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}


def export_gate(scorer_oof: pd.DataFrame, out_dir: Path, seeds: list[int]) -> None:
    """Fit the gate on all out-of-fold components (the model src.scorer.score_test fits for the test set) and save it."""
    from src.scorer import feature_columns, fit

    out_dir.mkdir(parents=True, exist_ok=True)
    feats = feature_columns(scorer_oof)
    for s in seeds:
        fit(scorer_oof, feats, s).booster_.save_model(str(out_dir / f"lgb_seed{s}.txt"))
    (out_dir / "features.json").write_text(json.dumps({"features": feats, "seeds": seeds}, indent=1), encoding="utf-8")


def gate_predict(df: pd.DataFrame, gate_dir: Path) -> np.ndarray:
    import lightgbm as lgb

    meta = json.loads((gate_dir / "features.json").read_text(encoding="utf-8"))
    x = df[meta["features"]]
    # read as text with normalised line endings: a CRLF checkout (Windows, core.autocrlf) breaks LightGBM's model parser
    boosters = [lgb.Booster(model_str=(gate_dir / f"lgb_seed{s}.txt").read_bytes().decode("utf-8").replace("\r\n", "\n"))
                for s in meta["seeds"]]
    return np.mean([b.predict(x) for b in boosters], axis=0)


def predict_member(member_cfg, weights_dir: Path, folds: list[int], stems: list[str], cache, device, log) -> dict[str, np.ndarray]:
    """uint8 fold-averaged probability of one member (same quantisation as the training pipeline's test_prob)."""
    from src.data import FilamentDataset
    from src.models import build_model
    from src.train import predict

    ds = FilamentDataset(stems, cache, dict(member_cfg.input.norm), train=False, with_target=False)
    pred_cfg = {**dict(member_cfg.train), "batch_size": member_cfg.infer.get("batch_size") or 1,
                "sep_suppress": False, "tta_scales": []}
    acc: dict[str, np.ndarray] = {}
    for fold in folds:
        model = build_model({**dict(member_cfg.model), "encoder_weights": None, "grad_checkpointing": False},
                            in_channels=len(member_cfg.input.channels))
        model.load_state_dict(load_sharded(weights_dir, f"fold{fold}"))
        for s, p in predict(model, ds, pred_cfg, device, tta=True).items():
            acc[s] = acc.get(s, 0) + p.astype(np.float32) / len(folds)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(f"release: {weights_dir.name} fold {fold} done")
    return {s: np.round(p * 255).astype(np.uint8) for s, p in acc.items()}


def run_release(cfg, exp_dir: Path, root: Path, device, log) -> pd.DataFrame:
    """Predict the test images with the shipped weights and gate; write submission.csv (+ per-image maps)."""
    from src.config import load_config
    from src.data import Cache
    from src.scorer import extract_components, rows_from_components
    from src.submission import validate_submission, write_submission

    rcfg = cfg.release
    test_images = {p.stem: p for p in sorted((root / cfg.data_dir / "test" / "test_images").glob("*.jpeg"))}
    stems = list(test_images)[: cfg.limits.test] if cfg.limits.get("test") else list(test_images)
    cache = Cache(root / cfg.cache_dir, dict(cfg.input))
    cache.build({s: test_images[s] for s in stems}, None, log)
    members = list(rcfg.members)
    ens: dict[str, np.ndarray] = {s: np.zeros((cfg.input.size, cfg.input.size), np.float32) for s in stems}
    for m in members:
        mcfg = load_config(root / "configs" / f"{m}.yaml", "full")
        maps = predict_member(mcfg, root / rcfg.weights_dir / m, list(rcfg.folds), stems, cache, device, log)
        for s in stems:
            ens[s] += maps[s].astype(np.float32) / len(members)
    probs = {s: np.round(p).astype(np.uint8).astype(np.float32) / 255.0 for s, p in ens.items()}
    (exp_dir / "test_prob").mkdir(parents=True, exist_ok=True)
    for s, p in probs.items():
        np.save(exp_dir / "test_prob" / f"{s}.npy", np.round(p * 255).astype(np.uint8))
    pp = dict(cfg.postprocess)
    test_df = extract_components(probs, cache, pp, None, int(cfg.train.num_workers))
    test_df["u_hat"] = gate_predict(test_df, root / rcfg.gate_dir)
    keep = test_df["u_hat"].values > float(pp["tau"])
    log(f"release: {len(test_df)} components, keep {keep.mean():.3f}")
    rows = rows_from_components(test_df, keep)
    sub = write_submission([{k: r[k] for k in ("filament_id", "segmentation_rle")} for r in rows], exp_dir / "submission.csv")
    log(f"release: submission {validate_submission(sub, expected_stems=stems)}")
    return sub
