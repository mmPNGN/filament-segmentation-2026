"""Export the final pipeline's trained models for the public repository (src/release.py):
fp16 sharded fold weights of the 4 members -> weights/<member>/, and the LightGBM gate fitted on all out-of-fold
components with the 5 seeds -> models/gate/. Checks that the saved gate reproduces the run's test u_hat exactly.

    uv run python tools/export_release.py            # from outputs/ of this machine into the repository root
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.release import export_gate, gate_predict, save_sharded  # noqa: E402

MEMBERS = {  # final member config -> training run that produced its 5 fold models
    "final_rx50_dml": "exp055-full",
    "final_effb4_dml": "exp075-full",
    "final_convnext_s_dml_s43": "exp108-calib",
    "final_hrnet_w32_dml_s43": "exp116-calib",
}
GATE_RUN, GATE_SEEDS = "exp135-ensemble", [42, 43, 44, 45, 46]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT), help="repository root to write weights/ and models/gate/ into")
    args = ap.parse_args()
    out = Path(args.out)
    for member, run in MEMBERS.items():
        for k in range(5):
            sd = torch.load(ROOT / "outputs" / run / "weights" / f"fold{k}.pt", map_location="cpu")
            paths = save_sharded(sd, out / "weights" / member, f"fold{k}")
            print(f"{member} fold{k}: {len(paths)} shard(s), {sum(p.stat().st_size for p in paths) / 1e6:.0f} MB")
    oof = pd.read_parquet(ROOT / "outputs" / GATE_RUN / "scorer_oof.parquet")
    export_gate(oof.drop(columns=["u_hat", "fold"], errors="ignore"), out / "models" / "gate", GATE_SEEDS)
    test = pd.read_parquet(ROOT / "outputs" / GATE_RUN / "scorer_test.parquet")
    diff = np.abs(gate_predict(test, out / "models" / "gate") - test["u_hat"].to_numpy()).max()
    print(f"gate: saved boosters reproduce {GATE_RUN} test u_hat, max |diff| = {diff:.2e}")
    if diff > 1e-9:
        raise SystemExit("gate export does not reproduce the run")


if __name__ == "__main__":
    main()
