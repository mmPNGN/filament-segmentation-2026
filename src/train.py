"""Training loop and prediction for one fold. Fixed epoch count (no early stopping on the validation fold,
validation.md §2.1), AMP, cosine schedule with warmup, BCE + soft Dice on the soft consensus target."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data import worker_init_fn


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0,
                  alpha: float = 0.5, beta: float = 0.5, region: str = "tversky") -> torch.Tensor:
    """BCE + dice_weight * region loss. region 'tversky': alpha weights false positives, beta false negatives;
    alpha = beta = 0.5 is soft Dice (the baseline). region 'dml' (train.region_loss): Dice semimetric loss
    (Wang & Blaschko 2023) 1 - (|p| + |t| - |p - t|) / (|p| + |t|) = 1 - 2 sum min(p, t) / (|p| + |t|): equal to soft Dice
    on hard targets, but minimised at p = t for SOFT targets (soft Dice pushes p to 0/1 on the annotator-mean target)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    if region == "dml":
        s = p.sum(dim=dims) + target.sum(dim=dims)
        dml = 1 - (s - (p - target).abs().sum(dim=dims) + 1.0) / (s + 1.0)
        return bce + dice_weight * dml.mean()
    inter = (p * target).sum(dim=dims)
    fp = (p * (1 - target)).sum(dim=dims)
    fn = ((1 - p) * target).sum(dim=dims)
    tversky = 1 - (inter + 0.5) / (inter + alpha * fp + beta * fn + 0.5)  # == 1 - (2I+1)/(P+T+1) at 0.5/0.5
    return bce + dice_weight * tversky.mean()


def skeleton_recall_loss(logits: torch.Tensor, skel: torch.Tensor) -> torch.Tensor:
    """Skeleton Recall Loss (Kirchhoff et al., ECCV 2024): 1 - soft recall of the tubed skeleton, sum(p * s) / sum(s),
    per sample; samples without skeleton pixels are skipped. `skel` is the (soft, annotator-mean) host spine drawn
    target.spine_width px wide, so the mask itself is pushed to cover each filament's centre line without breaks."""
    p = torch.sigmoid(logits.float())
    s = skel.float()
    total = s.sum(dim=(1, 2, 3))
    has = total > 0
    if not has.any():
        return logits.sum() * 0.0
    recall = (p * s).sum(dim=(1, 2, 3))[has] / total[has]
    return (1 - recall).mean()


def multi_head_loss(logits: torch.Tensor, target: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Mask loss on channel 0; when the target has a second channel (spine target, target.spine_width > 0) either add
    train.skel_recall_weight x the skeleton recall of the MASK output on it (> 0; single-head model), or add
    train.spine_weight x the same loss on a spine head (channel 1), which only shapes the shared features."""
    args = (cfg.get("dice_weight", 1.0), cfg.get("tversky_alpha", 0.5), cfg.get("tversky_beta", 0.5),
            cfg.get("region_loss", "tversky"))
    loss = bce_dice_loss(logits[:, :1], target[:, :1], *args)
    w_skel = float(cfg.get("skel_recall_weight", 0.0))
    if w_skel > 0 and target.shape[1] > 1:
        return loss + w_skel * skeleton_recall_loss(logits[:, :1], target[:, 1:2])
    if target.shape[1] > 1:
        loss = loss + float(cfg.get("spine_weight", 0.5)) * bce_dice_loss(logits[:, 1:2], target[:, 1:2], *args)
    return loss


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
        drop_last=shuffle, persistent_workers=num_workers > 0, worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )


def _amp_dtype(name: str | None):
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(name or "", None)


class _Ema:
    """Exponential moving average of the weights (train.ema_decay > 0; 0 = off). Float32 shadow of every floating
    parameter and buffer; decay warms up as min(decay, (1 + step) / (10 + step)). copy_to() loads it into the model."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().float().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))


def train_fold(model: torch.nn.Module, train_ds: Dataset, cfg: dict, seed: int, device: torch.device, log=print) -> dict:
    """Train in place; return {'train_min': minutes, 'last_loss': float}. With train.ema_decay > 0 the model ends with
    the EMA weights."""
    loader = make_loader(train_ds, cfg["batch_size"], True, cfg["num_workers"], seed)
    epochs, max_steps = cfg["epochs"], cfg.get("max_steps")
    steps_per_epoch = len(loader)
    total = epochs * steps_per_epoch if max_steps is None else min(max_steps, epochs * steps_per_epoch)
    warmup = int(cfg.get("warmup_epochs", 0) * steps_per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warmup, 1) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))
    amp_dtype = _amp_dtype(cfg.get("amp"))
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    cl = bool(cfg.get("channels_last", False))  # speed only (~20 % per step on the RTX 4070); same math
    model.to(device).train()
    if cl:
        model.to(memory_format=torch.channels_last)
    ema = _Ema(model, float(cfg["ema_decay"])) if float(cfg.get("ema_decay", 0.0)) > 0 else None
    t0, step, loss_val = time.time(), 0, float("nan")
    for epoch in range(epochs):
        losses = []
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if cl:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = multi_head_loss(model(x), y, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model, step)
            losses.append(loss.item())
            step += 1
            if step >= total:
                break
        loss_val = float(np.mean(losses)) if losses else loss_val
        log(f"epoch {epoch + 1}/{epochs} loss {loss_val:.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f} min")
        if step >= total:
            break
    if ema is not None:
        ema.copy_to(model)
        log(f"ema: loaded the EMA weights (decay {ema.decay})")
    return {"train_min": (time.time() - t0) / 60, "last_loss": loss_val}


def _dihedral_tta(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Mean probability over the 8 symmetries of the square (4 rotations x optional horizontal flip); the same group
    as the flip + rot90 training augmentation (infer.tta_rot90)."""
    acc = torch.zeros_like(x[:, :1], dtype=torch.float32)
    for k in range(4):
        xr = torch.rot90(x, k, dims=(-2, -1))
        for flip in (False, True):
            xi = xr.flip(-1) if flip else xr
            p = torch.sigmoid(model(xi)[:, :1].float())
            p = p.flip(-1) if flip else p
            acc += torch.rot90(p, -k, dims=(-2, -1))
    return acc / 8


def _prob(model: torch.nn.Module, x: torch.Tensor, sep_suppress: bool) -> torch.Tensor:
    """Mask probability (channel 0). With infer.sep_suppress and a separation head (target.sep_width > 0) the predicted
    gap probability is removed: p = sigmoid(mask) * (1 - sigmoid(sep)), so neighbouring filaments split into
    separate connected components while every downstream step keeps reading a single map."""
    out = model(x).float()
    p = torch.sigmoid(out[:, :1])
    if sep_suppress and out.shape[1] > 1:
        p = p * (1 - torch.sigmoid(out[:, 1:2]))
    return p


def _flip_tta(model: torch.nn.Module, x: torch.Tensor, sup: bool) -> torch.Tensor:
    prob = _prob(model, x, sup)
    prob = prob + _prob(model, x.flip(-1), sup).flip(-1)
    prob = prob + _prob(model, x.flip(-2), sup).flip(-2)
    prob = prob + _prob(model, x.flip(-1, -2), sup).flip(-1, -2)
    return prob / 4


def _scaled(model: torch.nn.Module, x: torch.Tensor, scale: float, sup: bool) -> torch.Tensor:
    """Flip-TTA probability of x resized by `scale` (infer.tta_scales), resized back to x's size. Sizes are rounded to a
    multiple of 32 (encoder stride)."""
    h, w = x.shape[-2:]
    hs, ws = int(round(h * scale / 32)) * 32, int(round(w * scale / 32)) * 32
    xs = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
    return F.interpolate(_flip_tta(model, xs, sup), size=(h, w), mode="bilinear", align_corners=False)


@torch.no_grad()
def predict(model: torch.nn.Module, ds: Dataset, cfg: dict, device: torch.device, tta: bool = False,
            tta_rot90: bool = False) -> dict[str, np.ndarray]:
    """stem -> (S, S) float16 probability. tta: 4-way flip average (exp022); tta_rot90: 8-way dihedral average."""
    loader = make_loader(ds, cfg["batch_size"], False, cfg["num_workers"], 0)
    amp_dtype = _amp_dtype(cfg.get("amp"))
    model.to(device).eval()
    cl = bool(cfg.get("channels_last", False))
    if cl:
        model.to(memory_format=torch.channels_last)
    sup = bool(cfg.get("sep_suppress", False))
    out = {}
    for x, _, stems in loader:
        x = x.to(device, non_blocking=True)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            if tta_rot90:
                prob = _dihedral_tta(model, x)
            else:
                prob = _prob(model, x, sup)  # channel 0 = mask (a spine head is ignored; a sep head optionally subtracts)
            if tta and not tta_rot90:
                prob = prob + _prob(model, x.flip(-1), sup).flip(-1)
                prob = prob + _prob(model, x.flip(-2), sup).flip(-2)
                prob = prob + _prob(model, x.flip(-1, -2), sup).flip(-1, -2)
                prob = prob / 4
                extra = [float(s) for s in cfg.get("tta_scales") or [] if float(s) != 1.0]
                if extra:  # multi-scale TTA: equal-weight mean over scale 1 and the extra scales
                    prob = (prob + sum(_scaled(model, x, s, sup) for s in extra)) / (1 + len(extra))
        for s, p in zip(stems, prob[:, 0].cpu().numpy()):
            out[s] = p.astype(np.float16)
    return out
