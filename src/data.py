"""Cache building and the torch Dataset.

Cache layout (data/processed/<key>/):
    x/<stem>.npy   (C, S, S) float16 input channels (src/preprocess.build_channels)
    y/<stem>.npy   (S, S) uint8 soft target x255  (train only; mean over the annotator records of the stem)
    y_<hash>/      targets for a non-default `target` config (src/data.target_key)
    disks.json     {stem: [cx, cy, r]} in 2048-frame coordinates
The key hashes the `input` config, so changing channels or size never reuses a stale cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

from src.annotations import IMAGE_SIZE
from src.preprocess import Disk, build_channels, cache_key, detect_disk, read_gray


def _poly_rle(poly: np.ndarray) -> dict:
    return mask_util.merge(mask_util.frPyObjects([poly.astype(np.float64).ravel().tolist()], IMAGE_SIZE, IMAGE_SIZE))


def _agreement(rles_per_record: list[list[dict]], iou_threshold: float = 0.5) -> list[list[bool]]:
    """For every filament: does any filament of *another* record of the same image overlap it with IoU > threshold?
    Single-record images have no other record, so all their filaments count as agreed (nothing to compare with)."""
    if len(rles_per_record) < 2:
        return [[True] * len(r) for r in rles_per_record]
    agreed = [[False] * len(r) for r in rles_per_record]
    for i, ri in enumerate(rles_per_record):
        if not ri:
            continue
        others = [r for j, rj in enumerate(rles_per_record) if j != i for r in rj]
        if not others:
            continue
        iou = mask_util.iou(ri, others, [0] * len(others))  # (len(ri), len(others))
        for k in range(len(ri)):
            agreed[i][k] = bool(np.max(iou[k]) > iou_threshold)
    return agreed


def soft_target(polys_per_record: list[list[np.ndarray]], size: int, solo_weight: float = 1.0) -> np.ndarray:
    """Mean over annotator records of the record's union mask, downsampled by area averaging, uint8 x255.

    solo_weight < 1 (target.solo_weight): a filament that no other annotator of the same image matched with IoU > 0.5
    ("only me", 65 % missed by the model and a guaranteed FP in the other records when predicted; error analysis
    2026-09-30) is rasterised with value solo_weight instead of 1 before averaging. 1.0 reproduces the plain mean."""
    acc = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    rles_per_record = [[_poly_rle(p) for p in polys] for polys in polys_per_record]
    agreed = _agreement(rles_per_record) if solo_weight < 1.0 else [[True] * len(r) for r in rles_per_record]
    for rles, ok in zip(rles_per_record, agreed):
        rec = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for rle, a in zip(rles, ok):
            m = mask_util.decode(rle).astype(np.float32) * (1.0 if a else solo_weight)
            rec = np.maximum(rec, m)
        acc += rec
    acc /= max(len(polys_per_record), 1)
    if size != IMAGE_SIZE:
        acc = cv2.resize(acc, (size, size), interpolation=cv2.INTER_AREA)
    return np.round(acc * 255).astype(np.uint8)


def spine_target(spines_per_record: list[list[np.ndarray]], size: int, width: int) -> np.ndarray:
    """Auxiliary target (target.spine_width > 0): every annotated spine drawn as a polyline of `width` px (2048 frame),
    max over the filaments of a record, mean over records, area-downsampled, uint8 x255 (same convention as the mask).
    Width 5 keeps 89 % of the spine pixels inside the polygon (analysis 2026-09-30); it becomes 2.5 px at 1024."""
    acc = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for spines in spines_per_record:
        rec = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        pts = [np.round(s).astype(np.int32).reshape(-1, 1, 2) for s in spines if len(s) >= 2]
        if pts:
            cv2.polylines(rec, pts, False, 1, int(width))
        acc += rec
    acc /= max(len(spines_per_record), 1)
    if size != IMAGE_SIZE:
        acc = cv2.resize(acc, (size, size), interpolation=cv2.INTER_AREA)
    return np.round(acc * 255).astype(np.uint8)


def sep_target(polys_per_record: list[list[np.ndarray]], size: int, gap: int) -> np.ndarray:
    """Auxiliary 'separation' target (target.sep_width > 0; error analysis 2026-10-03: single predicted blobs that cover
    two filaments of the same annotator are the largest instance-structure loss, oracle +0.019). Per record: pixels
    outside every filament that lie within `gap` px (2048 frame) of at least two different filaments, i.e. the gaps
    between neighbouring filaments; mean over records, area-downsampled, uint8 x255 (same convention as the mask)."""
    acc = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    for polys in polys_per_record:
        count = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        union = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for poly in polys:
            m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
            union |= m
            count += cv2.dilate(m, kernel)
        acc += ((count >= 2) & (union == 0)).astype(np.float32)
    acc /= max(len(polys_per_record), 1)
    if size != IMAGE_SIZE:
        acc = cv2.resize(acc, (size, size), interpolation=cv2.INTER_AREA)
    return np.round(acc * 255).astype(np.uint8)


def target_key(target_cfg: dict | None) -> str:
    """Sub-directory for cached targets: 'y' for the baseline target, 'y_<hash>' for any other target config."""
    cfg = dict(target_cfg or {})
    if (cfg.get("mode", "soft_mean") == "soft_mean" and float(cfg.get("solo_weight", 1.0)) >= 1.0
            and int(cfg.get("spine_width", 0)) <= 0 and int(cfg.get("sep_width", 0)) <= 0):
        return "y"
    payload = json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True)
    return "y_" + hashlib.md5(payload.encode()).hexdigest()[:8]


class Cache:
    def __init__(self, root: Path, input_cfg: dict, target_cfg: dict | None = None):
        self.dir = Path(root) / cache_key(input_cfg)
        self.input_cfg = input_cfg
        self.target_cfg = dict(target_cfg or {})
        self.y_dir = self.dir / target_key(self.target_cfg)
        (self.dir / "x").mkdir(parents=True, exist_ok=True)
        self.y_dir.mkdir(parents=True, exist_ok=True)
        self.disks_path = self.dir / "disks.json"
        self.disks: dict[str, list[float]] = json.loads(self.disks_path.read_text(encoding="utf-8")) if self.disks_path.exists() else {}

    def x_path(self, stem: str) -> Path:
        return self.dir / "x" / f"{stem}.npy"

    def y_path(self, stem: str) -> Path:
        return self.y_dir / f"{stem}.npy"

    def disk(self, stem: str) -> Disk:
        return Disk(*self.disks[stem])

    def build(self, images: dict[str, Path], annotations: pd.DataFrame | None, log=print) -> None:
        """images: stem -> jpeg path. annotations (train only): rows with stem, record_id, polygon."""
        todo = [s for s in images if not self.x_path(s).exists() or (annotations is not None and not self.y_path(s).exists())]
        if not todo:
            return
        log(f"cache: building {len(todo)} images into {self.dir}")
        size = self.input_cfg["size"]
        for i, stem in enumerate(todo):
            gray = read_gray(images[stem])
            if stem not in self.disks:
                d = detect_disk(gray)
                self.disks[stem] = [d.cx, d.cy, d.r]
            disk = self.disk(stem)
            if not self.x_path(stem).exists():
                np.save(self.x_path(stem), build_channels(gray, disk, self.input_cfg["channels"], size))
            if annotations is not None and not self.y_path(stem).exists():
                groups = [g for _, g in annotations[annotations["stem"] == stem].groupby("record_id")]
                y = soft_target([g["polygon"].tolist() for g in groups], size, float(self.target_cfg.get("solo_weight", 1.0)))
                spine_width = int(self.target_cfg.get("spine_width", 0))
                sep_width = int(self.target_cfg.get("sep_width", 0))
                if spine_width > 0:  # (2, S, S): mask + spine auxiliary target
                    y = np.stack([y, spine_target([g["spine"].tolist() for g in groups], size, spine_width)])
                elif sep_width > 0:  # (2, S, S): mask + separation (gap between neighbouring filaments) target
                    y = np.stack([y, sep_target([g["polygon"].tolist() for g in groups], size, sep_width)])
                np.save(self.y_path(stem), y)
            if (i + 1) % 50 == 0:
                log(f"cache: {i + 1}/{len(todo)}")
                self.disks_path.write_text(json.dumps(self.disks), encoding="utf-8")
        self.disks_path.write_text(json.dumps(self.disks), encoding="utf-8")


class FilamentDataset(Dataset):
    """Returns (x: (C,S,S) float32 normalised, y: (1,S,S) float32 in [0,1] or zeros for test, stem)."""

    def __init__(self, stems: list[str], cache: Cache, norm: dict, train: bool, aug: dict | None = None, with_target: bool = True,
                 tile_size: int = 0, tiles_per_image: int = 1, positive_frac: float = 0.5,
                 y_paths: dict[str, Path] | None = None, pseudo_stems: list[str] | None = None, pseudo_ratio: float = 0.0,
                 stem_weights: dict[str, float] | None = None):
        """tile_size > 0 (train only, train.tile_size): every item is a random tile_size x tile_size crop of the cached
        image (cache at input.size, e.g. 2048); the epoch holds tiles_per_image crops per image, and a fraction
        positive_frac of the crops is centred on a random filament pixel so thin objects are not under-sampled.
        y_paths (train.pseudo): stem -> uint8 x255 soft target saved elsewhere (pseudo-labels of unlabelled frames),
        used instead of the cache's annotation target for those stems.
        pseudo_stems + pseudo_ratio > 0 (train.pseudo.ratio): the epoch length stays len(stems) x tiles_per_image (equal
        optimisation steps to the run without pseudo-labels) and each item is, with probability pseudo_ratio, a random
        pseudo-labelled frame instead of the indexed labelled one (a different frame mix every epoch).
        stem_weights (train.record_weight): sampling weight per stem; items draw the stem at random with probability
        proportional to the weight (epoch length unchanged) instead of walking the stem list in order."""
        self.stems = list(stems)
        self._p = None
        if stem_weights:
            w = np.asarray([float(stem_weights.get(s, 1.0)) for s in self.stems], dtype=np.float64)
            self._p = w / w.sum()
        self.cache = cache
        self.channels = cache.input_cfg["channels"]
        self.norm = norm
        self.train = train
        self.aug = aug or {}
        self.with_target = with_target
        self.tile = int(tile_size) if train else 0
        self.tiles_per_image = max(int(tiles_per_image), 1) if self.tile else 1
        self.positive_frac = float(positive_frac)
        self.y_paths = dict(y_paths or {})
        self.pseudo_stems = list(pseudo_stems or [])
        self.pseudo_ratio = float(pseudo_ratio) if self.pseudo_stems else 0.0

    def __len__(self) -> int:
        return len(self.stems) * self.tiles_per_image

    def _crop(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s, t = x.shape[1], self.tile
        if t >= s:
            return x, y
        if np.random.rand() < self.positive_frac:
            ys, xs = np.nonzero(y[0] > 0.5)
            if len(ys):
                k = np.random.randint(len(ys))
                cy, cx = int(ys[k]) + np.random.randint(-t // 4, t // 4 + 1), int(xs[k]) + np.random.randint(-t // 4, t // 4 + 1)
                y0, x0 = int(np.clip(cy - t // 2, 0, s - t)), int(np.clip(cx - t // 2, 0, s - t))
                return x[:, y0:y0 + t, x0:x0 + t], y[:, y0:y0 + t, x0:x0 + t]
        y0, x0 = np.random.randint(0, s - t + 1), np.random.randint(0, s - t + 1)
        return x[:, y0:y0 + t, x0:x0 + t], y[:, y0:y0 + t, x0:x0 + t]

    def __getitem__(self, i: int):
        stem = self.stems[np.random.choice(len(self.stems), p=self._p)] if self._p is not None else self.stems[i % len(self.stems)]
        if self.pseudo_ratio > 0 and np.random.rand() < self.pseudo_ratio:
            stem = self.pseudo_stems[np.random.randint(len(self.pseudo_stems))]
        x = np.load(self.cache.x_path(stem)).astype(np.float32)
        for c, name in enumerate(self.channels):
            mean, std = self.norm[name]
            x[c] = (x[c] - mean) / std
        if self.with_target:
            y = np.load(self.y_paths.get(stem, self.cache.y_path(stem))).astype(np.float32) / 255.0
            y = y[None] if y.ndim == 2 else y  # (1, S, S) mask, or (2, S, S) mask + spine (target.spine_width)
            if y.shape[-2:] != x.shape[-2:]:
                raise ValueError(f"target {y.shape} does not match input {x.shape} for {stem}")
        else:
            y = np.zeros((1, x.shape[1], x.shape[2]), dtype=np.float32)
        if self.tile:
            x, y = self._crop(x, y)
        if self.train:
            x, y = self._augment(x, y)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)), stem

    def _augment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.aug.get("flip", False):
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[:, ::-1, :]
        if self.aug.get("rot90", False):
            k = int(np.random.randint(4))
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
        gain, noise = float(self.aug.get("gain", 0.0)), float(self.aug.get("noise", 0.0))
        if gain > 0 or noise > 0:
            # photometric augmentation in the un-normalised units of each channel: multiplicative gain U(1-g, 1+g)
            # (flat ~ 1, so this is also a contrast change) and additive Gaussian noise of std `noise`
            # (train hf_noise median 0.023 in flat units, image_stats 2026-09-30); x is already (v - mean) / std
            x = np.ascontiguousarray(x)
            for c, name in enumerate(self.channels):
                mean, std = self.norm[name]
                if gain > 0:
                    g = float(np.random.uniform(1.0 - gain, 1.0 + gain))
                    x[c] = x[c] * g + (g - 1.0) * mean / std
                if noise > 0:
                    x[c] += np.random.normal(0.0, noise / std, size=x[c].shape).astype(np.float32)
        return x, y


def worker_init_fn(worker_id: int) -> None:
    """Distinct numpy seed per DataLoader worker, derived from the torch seed (coding-style.md §1)."""
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)
