"""Small shared helpers: reproducibility and the runner's output contract."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Fix every random source (coding-style.md §1). Fold seeds should be derived as seed + fold."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def report_metric(key: str, value: float | int | str) -> None:
    """Print one `METRIC key=value` line; tools/run_exp.py parses exactly these lines into metrics.json.
    `key` must not contain spaces (use unit_<id> with `_` for per-sample scores)."""
    key = key.replace(" ", "_")
    text = f"{value:.6g}" if isinstance(value, float) else str(value)
    print(f"METRIC {key}={text}", flush=True)
