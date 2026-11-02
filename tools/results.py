"""Experiment ledger (research/results.tsv). The only sanctioned way to write it.

Usage:
    uv run python tools/results.py add --exp exp017 --stage full --status keep --commit a1b2c3d --desc "..."
    uv run python tools/results.py add --exp exp018 --stage screen --status discard   # desc taken from meta.json
    uv run python tools/results.py show [--last 20] [--stage full] [--since 2026-10-01]
    uv run python tools/results.py best [--stage full]
    uv run python tools/results.py set-lb --exp exp017 --lb 0.8123

`add` pulls cv / cv_std / fold scores / gpu_min from outputs/<exp>-<stage>/metrics.json and
computes delta + folds_improved against the parent (default: current best of that stage).
Improvement is always positive, whatever the metric direction in kaggle_project.json.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / "research" / "results.tsv"
RUNS = ROOT / "research" / "runs"
OUTPUTS = ROOT / "outputs"
COLUMNS = [
    "exp_id", "date", "stage", "commit", "parent", "cv", "cv_std", "delta",
    "folds_improved", "lb_public", "gpu_min", "status", "description",
]
STATUSES = ("baseline", "pass", "keep", "discard", "crash", "calib")
BEST_STATUSES = ("baseline", "keep")
FOLD_RE = re.compile(r"^fold(\d+)$")


def direction_sign() -> int:
    cfg = json.loads((ROOT / "kaggle_project.json").read_text(encoding="utf-8-sig"))
    direction = cfg.get("direction", "maximize")
    if direction not in ("maximize", "minimize"):
        sys.exit(f"kaggle_project.json: direction must be maximize|minimize, got {direction!r}")
    return 1 if direction == "maximize" else -1


def read_rows() -> list[dict]:
    if not TSV.exists():
        return []
    with TSV.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_rows(rows: list[dict]) -> None:
    TSV.parent.mkdir(parents=True, exist_ok=True)
    with TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def to_float(x: str | None) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def load_metrics(exp: str, stage: str) -> dict:
    path = OUTPUTS / f"{exp}-{stage}" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def best_row(rows: list[dict], stages: tuple[str, ...], sign: int) -> dict | None:
    cands = [r for r in rows if r["stage"] in stages and r["status"] in BEST_STATUSES and to_float(r["cv"]) is not None]
    return max(cands, key=lambda r: sign * to_float(r["cv"])) if cands else None  # type: ignore[operator]


def reference_row(rows: list[dict], stage: str, sign: int) -> dict | None:
    """Parent for comparison.
    full / calib -> best full model; ensemble -> best of full and ensemble;
    screen -> the screen run of the current best full model (same short setting)."""
    if stage in ("full", "calib"):
        return best_row(rows, ("full",), sign)
    if stage == "ensemble":
        return best_row(rows, ("full", "ensemble"), sign)
    best_full = best_row(rows, ("full",), sign)
    screens = [r for r in rows if r["stage"] == "screen" and to_float(r["cv"]) is not None]
    if best_full:
        same = [r for r in screens if r["exp_id"] == best_full["exp_id"]]
        if same:
            return same[-1]
    base = [r for r in screens if r["status"] == "baseline"]
    return base[-1] if base else None


def unit_scores(metrics: dict) -> dict[str, float]:
    """Per-unit scores for the consistency check: `unit_<id>` (e.g. per sample) if present, else `fold<k>`."""
    m = metrics.get("metrics", {})
    units = {k: float(v) for k, v in m.items() if k.startswith("unit_") and isinstance(v, (int, float))}
    if units:
        return units
    return {k: float(v) for k, v in m.items() if FOLD_RE.match(k) and isinstance(v, (int, float))}


def cmd_add(args: argparse.Namespace) -> None:
    sign = direction_sign()
    rows = read_rows()
    metrics = load_metrics(args.exp, args.stage)
    if not metrics and args.status != "crash" and args.cv is None:
        sys.exit(f"no outputs/{args.exp}-{args.stage}/metrics.json; pass --cv explicitly or use --status crash")
    m = metrics.get("metrics", {})
    cv = args.cv if args.cv is not None else m.get("cv")
    cv_std = m.get("cv_std")

    parent = None
    if args.parent:
        parent = next((r for r in reversed(rows) if r["exp_id"] == args.parent and r["stage"] == args.stage), None)
        if parent is None:
            sys.exit(f"parent {args.parent} ({args.stage}) not found in results.tsv")
    elif args.status != "baseline":
        parent = reference_row(rows, args.stage, sign)

    delta = folds_improved = "-"
    if parent and isinstance(cv, (int, float)) and to_float(parent["cv"]) is not None:
        delta = f"{sign * (cv - to_float(parent['cv'])):+.6g}"  # type: ignore[operator]
        mine, theirs = unit_scores(metrics), unit_scores(load_metrics(parent["exp_id"], args.stage))
        common = sorted(set(mine) & set(theirs))
        if common:
            better = sum(sign * (mine[k] - theirs[k]) > 0 for k in common)
            folds_improved = f"{better}/{len(common)}"

    desc = args.desc
    if not desc:  # fall back to the hypothesis given to run_exp.py
        meta_path = OUTPUTS / f"{args.exp}-{args.stage}" / "meta.json"
        desc = json.loads(meta_path.read_text(encoding="utf-8-sig")).get("desc", "") if meta_path.exists() else ""
    if not desc:
        sys.exit("no --desc given and no outputs/<exp>-<stage>/meta.json to take it from")
    desc = " ".join(desc.split())  # no tabs / newlines inside the TSV cell
    row = {
        "exp_id": args.exp, "date": dt.date.today().isoformat(), "stage": args.stage,
        "commit": args.commit or "-------", "parent": parent["exp_id"] if parent else "-",
        "cv": f"{cv:.6g}" if isinstance(cv, (int, float)) else "-",
        "cv_std": f"{cv_std:.4g}" if isinstance(cv_std, (int, float)) else "-",
        "delta": delta, "folds_improved": folds_improved,
        "lb_public": f"{args.lb:.6g}" if args.lb is not None else "-",
        "gpu_min": str(metrics.get("gpu_min", "-")), "status": args.status, "description": desc,
    }
    rows.append(row)
    write_rows(rows)

    if args.status in ("baseline", "keep", "calib") and metrics:
        dst = RUNS / f"{args.exp}-{args.stage}"
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "meta.json"):
            src = OUTPUTS / f"{args.exp}-{args.stage}" / name
            if src.exists():
                shutil.copy2(src, dst / name)
    print("\t".join(row[c] for c in COLUMNS[:-1]))


def cmd_show(args: argparse.Namespace) -> None:
    rows = read_rows()
    if args.stage:
        rows = [r for r in rows if r["stage"] == args.stage]
    if args.since:
        rows = [r for r in rows if r["date"] >= args.since[:10]]
    rows = rows[-args.last:] if args.last else rows
    if not rows:
        print("(no rows)")
        return
    cols = COLUMNS[:-1]
    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols) + "  description")
    for r in rows:
        desc = r["description"] if len(r["description"]) <= 90 else r["description"][:87] + "..."
        print("  ".join(r[c].ljust(widths[c]) for c in cols) + "  " + desc)
    sign = direction_sign()
    all_rows = read_rows()
    for stage in ("screen", "full"):
        ref = reference_row(all_rows, stage, sign)
        if ref:
            print(f"current reference [{stage}]: {ref['exp_id']} cv={ref['cv']}")
    print(f"total runs logged: {len(all_rows)}")


def cmd_best(args: argparse.Namespace) -> None:
    ref = best_row(read_rows(), (args.stage,), direction_sign())
    print(f"{ref['exp_id']}\t{ref['cv']}" if ref else "(none)")


def cmd_set_lb(args: argparse.Namespace) -> None:
    rows = read_rows()
    for r in reversed(rows):
        if r["exp_id"] == args.exp and (args.stage is None or r["stage"] == args.stage):
            r["lb_public"] = f"{args.lb:.6g}"
            write_rows(rows)
            print(f"{args.exp} [{r['stage']}] lb_public={r['lb_public']}")
            return
    sys.exit(f"{args.exp} not found")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add")
    a.add_argument("--exp", required=True)
    a.add_argument("--stage", required=True, choices=("screen", "full", "calib", "ensemble"))
    a.add_argument("--status", required=True, choices=STATUSES)
    a.add_argument("--desc", help="what/why/result; defaults to the --desc given to run_exp.py")
    a.add_argument("--commit")
    a.add_argument("--parent")
    a.add_argument("--cv", type=float, help="override (e.g. ensemble computed outside the runner)")
    a.add_argument("--lb", type=float)
    a.set_defaults(func=cmd_add)

    s = sub.add_parser("show")
    s.add_argument("--last", type=int, default=0)
    s.add_argument("--stage")
    s.add_argument("--since")
    s.set_defaults(func=cmd_show)

    b = sub.add_parser("best")
    b.add_argument("--stage", default="full")
    b.set_defaults(func=cmd_best)

    lb = sub.add_parser("set-lb")
    lb.add_argument("--exp", required=True)
    lb.add_argument("--lb", type=float, required=True)
    lb.add_argument("--stage")
    lb.set_defaults(func=cmd_set_lb)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
