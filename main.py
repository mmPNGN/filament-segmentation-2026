"""Single entry point for training and inference (see README.md).

    uv run python main.py --config configs/base.yaml            # full run (via tools/run_exp.py!)
    uv run python main.py --config configs/exp.yaml --screen    # short run, setting from research/protocol.md
    uv run python main.py --config configs/base.yaml --debug    # smoke test, may be run directly
    uv run python main.py --config configs/base.yaml --infer-only

New methods are switched on with config flags, never with new run_*.py scripts. The pipeline itself lives in
src/ (created during /kickoff); this file only wires arguments, config, seed and the output contract together.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.utils import report_metric, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/base.yaml", help="experiment yaml, overlaid on configs/base.yaml")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--screen", action="store_true", help="short run (research/protocol.md §2)")
    mode.add_argument("--debug", action="store_true", help="smoke test: tiny data, few steps, every stage")
    mode.add_argument("--infer-only", action="store_true", help="predict from saved weights, no training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mode = "screen" if args.screen else "debug" if args.debug else "infer" if args.infer_only else "full"
    exp_dir = Path(os.environ.get("EXP_DIR") or ROOT / "outputs" / "manual")  # run_exp.py sets EXP_DIR
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT / args.config, mode=mode)
    cfg.save(exp_dir / "config_resolved.yaml")
    seed_everything(cfg.seed)

    # Output contract (experiments.md §3): src.pipeline.run must write oof / test_pred / submission into exp_dir
    # and call report_metric("cv", ...), report_metric("cv_std", ...), report_metric("fold0", ...), ...
    from src.pipeline import run  # noqa: E402  -- created in /kickoff

    run(cfg, exp_dir=exp_dir, mode=mode, report_metric=report_metric)


if __name__ == "__main__":
    main()
