"""GONG H-alpha FITS (.fits.fz) -> 8-bit JPEG exactly as the host's MLEco FITS2JPEG convertor
(https://bitbucket.org/dataresearchlab/mleco-fits2jpegconvertor, src/runThis.sh), ported to numpy because the original
needs cfitsio + bash + ImageMagick (not available on this Windows machine):

  1. funpack                         -> astropy reads the tile-compressed HDU (BSCALE/BZERO applied, float64)
  2. wavelet (src/wavelet.cc)        -> B3-spline kernel at scale 1 (Work) and at scale 2 (New, dilated by 2) of the
                                        input; New += w1 (Old - Work) + w2 (Work - New); 0 < New <= skythr -> skyfac sqrt(New);
                                        a 4 px border stays 0. Defaults w1 2.5, w2 5.0, skythr 1333, skyfac 129.
  3. fits2ppmColorscale grey 640 5335 -> clip to [640, 5335], rint(255 t); the PPM is written bottom row first (FITS y
                                        axis is flipped)
  4. ImageMagick convert ppm -> jpg  -> PIL JPEG, grayscale, quality 92 (ImageMagick's default for a PPM input)

Used ONLY to prepare unlabelled 2023-2026 frames for TRAINING (host ruling #744047). Validated against the distributed
train JPEGs by `--check` (FITS of train observations only; never fetch test observations).
Usage: uv run python tools/gong_fits2jpeg.py IN_DIR OUT_DIR   |   uv run python tools/gong_fits2jpeg.py --check FITS JPEG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

C11, C12, C13, C22, C23, C33 = 0.00391, 0.01563, 0.02344, 0.06250, 0.09375, 0.14063


def _kernel(img: np.ndarray, s: int) -> np.ndarray:
    """The wavelet.cc 5x5 kernel with tap spacing s, evaluated on the interior (4 px border = 0), as in the C loop."""
    h, w = img.shape
    out = np.zeros_like(img)
    sl = lambda dy, dx: img[4 + dy:h - 4 + dy, 4 + dx:w - 4 + dx]  # noqa: E731
    a, b = s, 2 * s
    out[4:h - 4, 4:w - 4] = (
        C11 * (sl(-b, -b) + sl(b, -b) + sl(-b, b) + sl(b, b))
        + C12 * (sl(-b, -a) + sl(b, -a) + sl(-b, a) + sl(b, a) + sl(-a, -b) + sl(a, -b) + sl(-a, b) + sl(a, b))
        + C13 * (sl(-b, 0) + sl(b, 0) + sl(0, -b) + sl(0, b))
        + C22 * (sl(-a, -a) + sl(a, -a) + sl(-a, a) + sl(a, a))
        + C23 * (sl(-a, 0) + sl(a, 0) + sl(0, -a) + sl(0, a))
        + C33 * sl(0, 0))
    return out


def wavelet(old: np.ndarray, w1: float = 2.5, w2: float = 5.0, skythr: float = 1333.0, skyfac: float = 129.0) -> np.ndarray:
    work = _kernel(old, 1)
    new = _kernel(old, 2)
    new = new + w1 * (old - work) + w2 * (work - new)
    h, w = old.shape
    inner = np.zeros_like(new, dtype=bool)
    inner[4:h - 4, 4:w - 4] = True
    new = np.where(inner, new, 0.0)
    sky = inner & (new <= skythr) & (new > 0.0)
    new[sky] = skyfac * np.sqrt(new[sky])
    return new


def to_uint8(data: np.ndarray, vmin: float = 640.0, vmax: float = 5335.0) -> np.ndarray:
    t = (np.clip(data, vmin, vmax) - vmin) / (vmax - vmin)
    return np.rint(255.0 * t).astype(np.uint8)[::-1]  # PPM rows run from FITS y = height-1 down to 0


def read_fits(path: Path) -> np.ndarray:
    from astropy.io import fits

    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data.astype(np.float64)
    raise ValueError(f"no 2-D image in {path}")


def convert(fits_path: Path, jpeg_path: Path) -> None:
    img = to_uint8(wavelet(read_fits(fits_path)))
    Image.fromarray(img, mode="L").save(jpeg_path, quality=92)


def check(fits_path: Path, jpeg_path: Path) -> None:
    ours = to_uint8(wavelet(read_fits(fits_path))).astype(np.int16)
    ref = np.asarray(Image.open(jpeg_path).convert("L")).astype(np.int16)
    print(f"shape ours {ours.shape} ref {ref.shape}")
    d = np.abs(ours - ref)
    print(f"mean |diff| {d.mean():.3f}  p99 {np.quantile(d, 0.99):.1f}  max {d.max()}  corr {np.corrcoef(ours.ravel(), ref.ravel())[0, 1]:.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs=2)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check(Path(a.paths[0]), Path(a.paths[1]))
        sys.exit(0)
    src, dst = Path(a.paths[0]), Path(a.paths[1])
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.fits.fz")):
        out = dst / (f.name[:-8] + ".jpg")
        if not out.exists():
            convert(f, out)
