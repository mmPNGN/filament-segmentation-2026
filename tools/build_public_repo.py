"""Assemble the public repository (code + configs + shipped weights + gate + notebooks + report figures) from this
working repository into a separate folder. Internal notes (agent instructions, task lists, Japanese research logs,
competition data, experiment outputs) are not copied. Run tools/export_release.py --out <dest> first for weights/ and
models/gate/.

    uv run python tools/build_public_repo.py --dest ../fs26-release-stage
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["main.py", "README.md", "LICENSE", "requirements.txt", "pyproject.toml", "uv.lock", "data/folds.csv",
         "research/results.tsv"]
DIRS = ["src", "configs", "tests", "notebooks", "research/report/figs", "analysis/report_figs"]
TOOLS = ["run_exp.py", "results.py", "average_probs.py", "average_pl.py", "export_release.py", "build_public_repo.py",
         "reproduce_best.sh", "gong_download.py", "gong_fits2jpeg.py", "gong_pseudo_list.py"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints", "_cache", "kaggle_bench",
                              "verify_release.yaml")
GITIGNORE = """# competition data and caches: put the Kaggle data in data/raw/MAGFiLO_1.0_Kaggle_2026/
data/*
!data/folds.csv
# run artefacts
outputs/
submission.csv
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
analysis/report_figs/_cache/
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    dest = Path(ap.parse_args().dest)
    for f in FILES:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest / f)
    for d in DIRS:
        shutil.copytree(ROOT / d, dest / d, ignore=SKIP, dirs_exist_ok=True)
    (dest / "tools").mkdir(exist_ok=True)
    for t in TOOLS:
        shutil.copy2(ROOT / "tools" / t, dest / "tools" / t)
    (dest / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    missing = [p for p in ("weights", "models/gate/features.json") if not (dest / p).exists()]
    print(f"{dest}: copied; {'MISSING ' + str(missing) + ' (run tools/export_release.py --out)' if missing else 'weights and gate present'}")


if __name__ == "__main__":
    main()
