"""Experiment runner: the only sanctioned way to launch a training run.

Usage:
    uv run python tools/run_exp.py --stage screen --desc "hypothesis" -- uv run python main.py --config configs/x.yaml --screen
    uv run python tools/run_exp.py --stage full --id exp017 --desc "hypothesis" -- uv run python main.py --config configs/x.yaml

What it guarantees (see README.md):
- outputs/<exp_id>-<stage>/ with meta.json, patch.diff, untracked/, train.log, metrics.json
- a GPU lock so that only one local training run exists at a time
- METRIC <key>=<value> lines parsed from stdout into metrics.json
- a one-line summary on stdout (the full log never floods the agent's context)

Standard library only, so it runs before `uv sync` has installed anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
LOCK_PATH = OUTPUTS / ".gpu.lock"
RESULTS_TSV = ROOT / "research" / "results.tsv"
STAGES = ("screen", "full", "calib", "ensemble", "infer", "debug")
METRIC_RE = re.compile(r"^METRIC\s+([A-Za-z0-9_.\-]+)=(\S+)\s*$")
EXP_DIR_RE = re.compile(r"^(exp\d{3,})-")
UNTRACKED_MAX_BYTES = 1_000_000
CODE_DIRS = ("src/", "configs/", "tools/", "notebooks/")
CODE_SUFFIXES = (".py", ".yaml", ".yml", ".toml", ".json")


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=True,
        )
        return out.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def next_exp_id() -> str:
    """One past the highest id seen in outputs/ OR research/results.tsv (rows added with --cv, e.g. ensembles
    computed outside the runner, have no outputs/ folder but must still reserve their number)."""
    numbers = [
        int(m.group(1)[3:])
        for p in OUTPUTS.iterdir() if p.is_dir() and (m := EXP_DIR_RE.match(p.name))
    ] if OUTPUTS.exists() else []
    if RESULTS_TSV.exists():
        for line in RESULTS_TSV.read_text(encoding="utf-8-sig").splitlines()[1:]:
            m = re.match(r"exp(\d+)\t", line)
            if m:
                numbers.append(int(m.group(1)))
    return f"exp{(max(numbers) + 1 if numbers else 0):03d}"


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        # Never use os.kill(pid, 0) on Windows: it calls TerminateProcess.
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = wintypes.DWORD()
        ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
        k32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the command AND its children. `uv run python main.py` is a process tree; killing only `uv`
    would leave the training process holding the GPU after the lock is released."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()
    proc.wait()


def acquire_lock(exp_dir_name: str) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "exp": exp_dir_name, "started": now_iso()})
    for _ in range(2):
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = json.loads(LOCK_PATH.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                holder = {}
            if holder.get("pid") and pid_alive(int(holder["pid"])):
                sys.exit(
                    f"[run_exp] REFUSED: GPU lock held by {holder.get('exp')} "
                    f"(pid {holder['pid']}, since {holder.get('started')}). "
                    "Local training must run one at a time."
                )
            LOCK_PATH.unlink(missing_ok=True)  # stale lock from a dead process
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        return
    sys.exit("[run_exp] could not acquire GPU lock")


def release_lock() -> None:
    try:
        holder = json.loads(LOCK_PATH.read_text(encoding="utf-8-sig"))
        if holder.get("pid") == os.getpid():
            LOCK_PATH.unlink()
    except (OSError, json.JSONDecodeError):
        pass


def is_code(rel: str) -> bool:
    """Whitelist, so a missing .gitignore can never make us copy outputs/ or data/ into itself."""
    if rel.startswith(CODE_DIRS):
        return True
    return "/" not in rel and rel.endswith(CODE_SUFFIXES)


def snapshot_code(exp_dir: Path) -> dict:
    """Save everything needed to reproduce the run even if the change is later discarded."""
    (exp_dir / "patch.diff").write_text(git("diff", "HEAD", "--binary"), encoding="utf-8")
    untracked = [p for p in git("ls-files", "--others", "--exclude-standard").splitlines() if is_code(p)]
    copied = []
    for rel in untracked:
        src = ROOT / rel
        if src.is_file() and src.stat().st_size <= UNTRACKED_MAX_BYTES:
            dst = exp_dir / "untracked" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    return {
        "commit": git("rev-parse", "--short", "HEAD").strip() or "nogit",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(git("status", "--porcelain").strip()),
        "untracked_copied": copied,
    }


def parse_metrics(log_path: Path) -> dict:
    metrics: dict[str, float | str] = {}
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = METRIC_RE.match(line.strip())
            if m:
                key, raw = m.groups()
                try:
                    metrics[key] = float(raw)
                except ValueError:
                    metrics[key] = raw
    return metrics


def tail(path: Path, n: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--desc", required=True, help="hypothesis / what changed")
    parser.add_argument("--id", help="reuse an exp id (e.g. promote exp017 from screen to full)")
    parser.add_argument("--timeout-min", type=float, default=None)
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command to run>")
    args = parser.parse_args()

    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error("missing command after --")
    resolved = shutil.which(cmd[0])
    if resolved:
        cmd = [resolved, *cmd[1:]]

    exp_id = args.id or next_exp_id()
    exp_dir = OUTPUTS / f"{exp_id}-{args.stage}"
    if exp_dir.exists():
        sys.exit(f"[run_exp] {exp_dir.name} already exists; pick another --id or stage")

    acquire_lock(exp_dir.name)
    try:
        exp_dir.mkdir(parents=True)
        meta = {
            "exp_id": exp_id, "stage": args.stage, "desc": args.desc,
            "cmd": cmd, "started": now_iso(), **snapshot_code(exp_dir),
        }
        (exp_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        env = {
            **os.environ, "EXP_DIR": str(exp_dir), "EXP_ID": exp_id, "EXP_STAGE": args.stage,
            "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
        }
        log_path = exp_dir / "train.log"
        t0 = time.time()
        timed_out = False
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = proc.wait(timeout=args.timeout_min * 60 if args.timeout_min else None)
            except subprocess.TimeoutExpired:
                kill_tree(proc)
                returncode = proc.returncode
                timed_out = True
            except KeyboardInterrupt:
                kill_tree(proc)
                raise
        minutes = (time.time() - t0) / 60

        metrics = parse_metrics(log_path)
        ok = returncode == 0 and isinstance(metrics.get("cv"), float) and not timed_out
        result = {
            "exp_id": exp_id, "stage": args.stage, "status": "ok" if ok else "crash",
            "returncode": returncode, "timed_out": timed_out, "gpu_min": round(minutes, 1),
            "finished": now_iso(), "metrics": metrics,
        }
        (exp_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        meta["finished"] = result["finished"]
        (exp_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        release_lock()

    cv = metrics.get("cv")
    cv_std = metrics.get("cv_std")
    cv_txt = f"cv={cv:.6g}" + (f" ±{cv_std:.3g}" if isinstance(cv_std, float) else "") if isinstance(cv, float) else "cv=-"
    print(f"[run_exp] {exp_dir.name} status={result['status']} {cv_txt} gpu_min={result['gpu_min']} dir={exp_dir}")
    if not ok:
        reason = "timeout" if timed_out else f"returncode={returncode}" if returncode else "no 'METRIC cv=' line"
        print(f"[run_exp] CRASH ({reason}). Last 30 log lines:\n{tail(log_path, 30)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
