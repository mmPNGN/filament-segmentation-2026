"""Solar-disk detection, radial flattening and the per-image cache.

Images are 2048x2048 uint8 H-alpha full-disk frames with (a) a bright limb, (b) a grey processing halo outside the
limb and (c) residual limb darkening (EDA 2026-09-29 §5). All of that is handled here, once, and cached under
data/processed/<key>/ so the training loop only loads float16 arrays.

Channels (config `input.channels`):
    raw   : gray / 255
    flat  : gray / radial_median_profile(r)  (≈1 on the quiet disk, <1 in filaments), 1 outside the disk
Everything outside the disk (r > radius * margin) is set to the "quiet" value so the network sees a neutral border.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.annotations import IMAGE_SIZE


@dataclass
class Disk:
    cx: float
    cy: float
    r: float

    def radius_map(self, size: int = IMAGE_SIZE) -> np.ndarray:
        """r / R for every pixel of a size x size grid (scaled from the 2048 frame)."""
        s = size / IMAGE_SIZE
        yy, xx = np.mgrid[:size, :size]
        return np.sqrt((xx - self.cx * s) ** 2 + (yy - self.cy * s) ** 2) / (self.r * s)


def _fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit. points: (N, 2) [x, y]."""
    x, y = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)
    a = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(cx), float(cy), float(np.sqrt(c + cx**2 + cy**2))


def _ray_samples(gray: np.ndarray, cx: float, cy: float, radii: np.ndarray, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear samples of gray along n_angles rays from (cx, cy): (n_angles, len(radii)), plus the angles."""
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    xs = (cx + np.outer(np.cos(angles), radii)).astype(np.float32)
    ys = (cy + np.outer(np.sin(angles), radii)).astype(np.float32)
    return cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32), angles


def _limb_points(gray: np.ndarray, cx: float, cy: float, r_lo: float, r_hi: float, n_angles: int = 720,
                 half_width: int = 6, min_prominence: float = 6.0) -> np.ndarray:
    """The limb is a thin bright ring (limb brightening left by the host's sharpening; EDA 2026-09-29). Along every
    ray, take the radius where the intensity most exceeds the mean of its two neighbours at +-half_width; keep rays
    whose peak prominence is clear. Returns (N, 2) [x, y] points in the 2048 frame."""
    radii = np.arange(r_lo, r_hi, 1.0)
    prof, angles = _ray_samples(gray, cx, cy, radii, n_angles)
    prof = cv2.GaussianBlur(prof, (1, 3), 0)
    w = half_width
    prominence = prof[:, w:-w] - 0.5 * (prof[:, : -2 * w] + prof[:, 2 * w :])
    idx = prominence.argmax(axis=1)
    keep = prominence[np.arange(n_angles), idx] >= min_prominence
    r_edge = radii[idx + w][keep]
    a = angles[keep]
    return np.column_stack([cx + r_edge * np.cos(a), cy + r_edge * np.sin(a)])


def _robust_circle(points: np.ndarray, n_iter: int = 3, k: float = 2.5) -> tuple[float, float, float]:
    cx, cy, r = _fit_circle(points)
    for _ in range(n_iter):
        res = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        mad = np.median(res) + 1e-6
        inl = res <= k * max(mad, 1.0)
        if inl.sum() < 10:
            break
        cx, cy, r = _fit_circle(points[inl])
    return cx, cy, r


def detect_disk(gray: np.ndarray, search: tuple[float, float] = (0.85, 1.12)) -> Disk:
    """Initial centre/radius from the Otsu foreground, then two rounds of limb-ring detection along rays and a robust
    least-squares circle fit. Robust to the grey halo outside the limb, whose brightness varies between images."""
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    _, otsu = cv2.threshold(cv2.GaussianBlur(small, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(otsu)
    scale = gray.shape[0] / 512
    cx, cy, r = xs.mean() * scale, ys.mean() * scale, np.sqrt(len(xs) / np.pi) * scale
    for _ in range(2):
        pts = _limb_points(gray, cx, cy, search[0] * r, search[1] * r)
        if len(pts) < 30:
            break
        # the halo's outer edge is a second (weaker) step along some rays: keep only rays whose edge radius
        # agrees with the median before fitting (the centre estimate is within a few px, the ring is a circle)
        rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        agree = np.abs(rad - np.median(rad)) <= 0.03 * np.median(rad)
        cx, cy, r = _robust_circle(pts[agree])
    return Disk(cx, cy, r)


def radial_profile(gray: np.ndarray, disk: Disk, n_bins: int = 128) -> np.ndarray:
    """Median intensity per r/R bin in [0, 1]; extrapolated flat beyond the last populated bin."""
    rr = disk.radius_map(gray.shape[0])
    idx = np.clip((rr * n_bins).astype(int), 0, n_bins - 1)
    inside = rr < 1.0
    prof = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = inside & (idx == b)
        if sel.any():
            prof[b] = np.median(gray[sel])
    valid = ~np.isnan(prof)
    prof = np.interp(np.arange(n_bins), np.flatnonzero(valid), prof[valid])
    return prof


def flatten(gray: np.ndarray, disk: Disk, margin: float = 0.99, n_bins: int = 128) -> np.ndarray:
    """gray / radial median, float32 ≈ 1 on the quiet disk; exactly 1 outside r > margin * R."""
    prof = radial_profile(gray, disk, n_bins)
    rr = disk.radius_map(gray.shape[0])
    denom = np.interp(np.clip(rr, 0, 1) * (n_bins - 1), np.arange(n_bins), prof)
    flat = gray.astype(np.float32) / np.maximum(denom, 1.0).astype(np.float32)
    flat[rr > margin] = 1.0
    return flat


def local_zscore(flat: np.ndarray, rr: np.ndarray, sigma: float = 32.0, margin: float = 0.99) -> np.ndarray:
    """(flat - local mean) / local std with a Gaussian window (sigma px at 2048); 0 outside the disk.
    Normalises the contrast of low-contrast frames (error analysis: 2016-17, Mauna Loa / Udaipur score lowest)."""
    inside = (rr <= margin).astype(np.float32)
    x = np.where(inside > 0, flat, 1.0).astype(np.float32)
    mu = cv2.GaussianBlur(x, (0, 0), sigma)
    var = cv2.GaussianBlur(x * x, (0, 0), sigma) - mu * mu
    z = (x - mu) / np.sqrt(np.maximum(var, 1e-6))
    return np.clip(z, -5.0, 5.0) * inside


def dark_ridge(flat_small: np.ndarray, sigmas=(1.5, 2.5, 4.0), scale: float = 0.05) -> np.ndarray:
    """Sato (Hessian) dark-ridge response of the flattened image at the working resolution, max over `sigmas`
    (px; filament width median 5.5 px at 1024), divided by `scale` (typical response inside an annotated filament,
    analysis/ridge_channel.py) and clipped to [0, 1]. Pixel AUC filament vs background 0.74-0.82 on the EDA images."""
    from skimage.filters import sato

    r = sato(flat_small.astype(np.float32), sigmas=sigmas, black_ridges=True, mode="reflect")
    return np.clip(r / scale, 0.0, 1.0).astype(np.float32)


def build_channels(gray: np.ndarray, disk: Disk, channels: list[str], size: int) -> np.ndarray:
    """(C, size, size) float16 input tensor for the configured channel names.
    raw | flat | zscore (local contrast of flat, sigma 32 px) | rr (r / R, clipped to 1.05) | ridge (dark_ridge of flat
    at `size`, 0 outside the disk)."""
    out = []
    rr = disk.radius_map(gray.shape[0])
    flat = None
    for name in channels:
        if name == "ridge":
            flat = flatten(gray, disk) if flat is None else flat
            small = cv2.resize(flat, (size, size), interpolation=cv2.INTER_AREA) if size != gray.shape[0] else flat
            ch = dark_ridge(small) * (disk.radius_map(size) <= 0.99)
            out.append(ch)
            continue
        if name == "raw":
            ch = gray.astype(np.float32) / 255.0
        elif name == "flat":
            flat = flatten(gray, disk) if flat is None else flat
            ch = flat
        elif name == "zscore":
            flat = flatten(gray, disk) if flat is None else flat
            ch = local_zscore(flat, rr)
        elif name == "rr":
            ch = np.clip(rr, 0, 1.05).astype(np.float32)
        else:
            raise ValueError(f"unknown channel {name!r}")
        if size != gray.shape[0]:
            ch = cv2.resize(ch, (size, size), interpolation=cv2.INTER_AREA)
        out.append(ch)
    return np.stack(out).astype(np.float16)


def cache_key(input_cfg: dict) -> str:
    """Directory name for data/processed/: a hash of the settings that change the cached arrays."""
    payload = json.dumps({k: input_cfg[k] for k in sorted(input_cfg)}, sort_keys=True)
    return f"{input_cfg['size']}_{hashlib.md5(payload.encode()).hexdigest()[:8]}"


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img if img.ndim == 2 else img[..., 0]
