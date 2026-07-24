"""Experiment — persistent-homology H1 over the saved trajectories.

Reads trajectories/manifest.csv + the *.npy point clouds, scores each with the
diameter-normalised max H1 persistence, and writes results/homology.csv.
Requires the `tda` extra (ripser). OWNER: Extraction+Winding.
STATUS: implemented, but H1 on a SINGLE curve is ~0 by mathematical
    construction (a single 1-D path has no independent 1-cycles to detect,
    regardless of whether the underlying dynamics loop) -- this is not a
    finding about Huginn, see project_plan.md §10 item 3 and
    `theoretical_framework.md`, which names population-level/delay-embedding
    homology (not implemented here) as the methodologically sound version.

Run: uv run python scripts/run_homology.py
"""

import os

import numpy as np
import pandas as pd

from traj_geom.metrics.homology import h1_persistence

TRAJ = "trajectories"
man = pd.read_csv(f"{TRAJ}/manifest.csv")
rows = []
for _, r in man.iterrows():
    pts = np.load(f"{TRAJ}/{r['file']}")
    dgm, mp = h1_persistence(pts)
    rows.append({**r.to_dict(), "n_h1": len(dgm), "max_h1_persist_norm": round(mp, 3)})
df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/homology.csv", index=False)
print(df[["file", "num_steps", "n_ops", "n_h1", "max_h1_persist_norm"]].to_string(index=False))
print("\nsettle(ns=64) mean:", round(df[df.num_steps == 64].max_h1_persist_norm.mean(), 3))
print("tight (ns=16) mean:", round(df[df.num_steps == 16].max_h1_persist_norm.mean(), 3))
