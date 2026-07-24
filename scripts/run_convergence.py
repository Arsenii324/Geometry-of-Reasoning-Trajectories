"""Experiment — objective convergence metrics over the saved trajectories.

Reads trajectories/manifest.csv + the *.npy point clouds, scores each with the
model-free convergence descriptors, and writes results/convergence.csv. No GPU
needed. OWNER: Extraction+Winding.
STATUS: implemented, verified-live 2026-07-18 (claims_ledger.md B10). Depends
    on the `trajectories/*.npy` point clouds already being saved -- no current
    script writes new ones (see architecture_state.md's orphaned-data note);
    re-running this against fresh data would need that wired up first.

Run: uv run python scripts/run_convergence.py
"""

import os

import numpy as np
import pandas as pd

from traj_geom.metrics.convergence import (
    consecutive_step_cosine,
    convergence_rate,
    drift_to_loop_ratio,
)
from traj_geom.metrics.dynamics import step_norms

TRAJ = "trajectories"
man = pd.read_csv(f"{TRAJ}/manifest.csv")
rows = []
for _, r in man.iterrows():
    tr = np.load(f"{TRAJ}/{r['file']}")
    s = step_norms(tr)
    rows.append(
        {
            **r.to_dict(),
            "conv_rate": round(convergence_rate(tr), 4),
            "cos": round(consecutive_step_cosine(tr), 3),
            "dlr": round(drift_to_loop_ratio(tr), 2),
            "shrink": round(float(s[-1] / (s[:3].mean() + 1e-9)), 4),
        }
    )
df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/convergence.csv", index=False)

print(df[["file", "num_steps", "conv_rate", "cos", "dlr", "shrink"]].to_string(index=False))
print("\n--- by num_steps ---")
print(df.groupby("num_steps")[["conv_rate", "cos", "dlr", "shrink"]].mean().round(3))
print("\nconv_rate<0:", int((df.conv_rate < 0).sum()), "/", len(df))
print("settle (ns=64) shrink mean:", round(df[df.num_steps == 64].shrink.mean(), 3))
print("tight  (ns=16) shrink mean:", round(df[df.num_steps == 16].shrink.mean(), 3))
