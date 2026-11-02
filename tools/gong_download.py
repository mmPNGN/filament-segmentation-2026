"""Download unlabelled GONG H-alpha frames (2023-2026, outside the 2011-2022 train/test period) from the public NSO
archive and convert them with tools/gong_fits2jpeg.py. TRAINING ONLY (host ruling #744047: unlabelled 2023-2026 GONG
H-alpha may be used for training, never at inference). No annotations of any kind.

Sampling: for every `--every`-th day, list the day directory, group the frames by observatory letter, pick up to
`--per-day` observatories at random and one random frame of each. Only file names matching
YYYYMMDDHHMMSS<site>h.fits.fz are followed (the listing also holds a crawler-trap link, never visited).
Sequential, polite (pause between requests). FITS files are deleted after conversion.
Usage: uv run python tools/gong_download.py --start 2023-01-01 --end 2026-09-30 --out data/external/gong_ha_2023_2026
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from gong_fits2jpeg import convert

BASE = "https://gong2.nso.edu/HA/haf"
NAME = re.compile(r'href="(\d{14}[A-Z]h\.fits\.fz)"')


def fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "kaggle-filament-research (training data, host ruling #744047)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--per-day", type=int, default=2)
    ap.add_argument("--out", default="data/external/gong_ha_2023_2026")
    ap.add_argument("--pause", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "jpg").mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    manifest = out / "manifest.csv"
    done = set(pd.read_csv(manifest)["stem"]) if manifest.exists() else set()
    rng = random.Random(a.seed)
    day = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end)
    n_ok = n_err = 0
    while day <= end:
        d = day.strftime("%Y%m%d")
        day += dt.timedelta(days=a.every)
        try:
            names = NAME.findall(fetch(f"{BASE}/{d[:6]}/{d}/").decode("utf-8", "replace"))
        except Exception as e:  # missing day directory etc.
            print(f"{d}: listing failed ({e})", flush=True)
            continue
        time.sleep(a.pause)
        by_site: dict[str, list[str]] = {}
        for n in sorted(set(names)):
            by_site.setdefault(n[14], []).append(n)
        sites = sorted(s for s in by_site if s in "BCLMTU")  # the 6 GONG sites of the competition data (2023+ also has Z)
        rng.shuffle(sites)
        rows = []
        for site in sites[: a.per_day]:
            name = rng.choice(by_site[site])
            stem = name[:-8]
            if stem in done:
                continue
            f = tmp / name
            try:
                f.write_bytes(fetch(f"{BASE}/{d[:6]}/{d}/{name}"))
                convert(f, out / "jpg" / f"{stem}.jpg")
                rows.append({"stem": stem, "date": d, "site": site, "n_frames_site_day": len(by_site[site])})
                n_ok += 1
            except Exception as e:
                print(f"{name}: failed ({e})", flush=True)
                n_err += 1
            finally:
                f.unlink(missing_ok=True)
            time.sleep(a.pause)
        if rows:
            pd.DataFrame(rows).to_csv(manifest, mode="a", header=not manifest.exists(), index=False)
            done.update(r["stem"] for r in rows)
        if n_ok and n_ok % 50 == 0:
            print(f"{d}: {n_ok} converted, {n_err} errors", flush=True)
    print(f"done: {n_ok} converted, {n_err} errors", flush=True)


if __name__ == "__main__":
    main()
