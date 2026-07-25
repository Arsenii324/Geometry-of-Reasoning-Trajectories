"""Experiment — adjudicate the winding number against a matched-random-walk null.

OWNER: Data+Analysis
STATUS: implemented 2026-07-25. THE missing foundational check: every winding
    number this project has ever reported (H1 shape counts, every
    winding~difficulty correlation, D11/D15/D16) has been un-adjudicated
    against chance. `winding_null_test` was built long ago but never run on
    real Huginn data -- the project's own docs repeatedly state it is blocked
    because "raw per-step trajectories aren't saved."
    **That blocker was false**: 140 real Huginn trajectories with full raw
    [T, 5280] paths are banked in results/trajectories/ (count_ones +
    projection, 8 n_ops levels, 5 init seeds, num_steps 64 and 128). This
    script runs the null test on all of them. 0-GPU.
TASK: for each banked trajectory, compare the observed |winding| against 200
    matched random walks -- surrogates with the SAME per-step displacement
    sizes but directions randomized uniformly on the hypersphere, pushed
    through the SAME per-trajectory 2-D PCA. This isolates the question:
    does the path rotate more than its own step-size profile alone implies?

WHY THIS IS THE RIGHT NULL. `winding_of` fits PCA on each trajectory's own
    points, and PCA always finds *some* plane of maximal spread -- so a path
    that merely drifts with high-dimensional noise can display apparent
    curvature in its top-2 plane. The matched random walk preserves step
    magnitudes (hence the drift/noise profile) and destroys only the
    directional correlation between consecutive steps. If observed |winding|
    does not exceed the surrogate distribution, the measured winding carries
    no rotational information beyond step-size structure.

INTERPRETATION CONTRACT (fixed before looking at the full result, so it
    cannot be rationalized afterwards):
      p_value = fraction of surrogates with |winding| >= observed.
      p < 0.05  -> that trajectory winds more than chance.
      p ~ 0.5   -> the observed winding sits mid-null: no rotational signal.
      A high project-wide rate of non-significant p, with observed ~= null
      mean, means the winding METRIC (as implemented) does not measure
      rotation on this data -- which would retroactively qualify every
      winding-based claim in the project.

I/O: reads results/trajectories/*.npy (raw paths, untracked/local) ->
    results/winding_null.csv (task, n_ops, num_steps, init_seed, observed,
    null_mean, null_std, p_value, z_score).

Run: uv run python -m scripts.run_winding_null
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd
from tqdm import tqdm

from scripts._common import RESULTS_DIR, cached
from traj_geom.metrics.winding import winding_null_test

TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")
N_SURROGATES = 200
_FNAME_RE = re.compile(
    r"^(?P<task>[a-z_]+)_n(?P<n_ops>\d+)_ns(?P<num_steps>\d+)_init(?P<seed>\d+)\.npy$"
)


def compute() -> pd.DataFrame:
    """Run the matched-random-walk null test on every banked trajectory."""
    import numpy as np

    paths = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.npy")))
    if not paths:
        raise FileNotFoundError(
            f"no trajectories in {TRAJ_DIR} -- this experiment needs the raw "
            "[T, hidden] .npy paths (they are untracked/local, not in git)."
        )

    rows = []
    for path in tqdm(paths, desc="null test"):
        name = os.path.basename(path)
        m = _FNAME_RE.match(name)
        if m is None:
            print(f"run_winding_null: skipping unparseable filename {name}")
            continue
        traj = np.load(path)
        # seed per-file so each trajectory's surrogates are reproducible but
        # not identical across files.
        res = winding_null_test(traj, n_surrogates=N_SURROGATES, seed=abs(hash(name)) % (2**31))
        z = (
            (res["observed"] - res["null_mean"]) / res["null_std"]
            if res["null_std"] > 0
            else float("nan")
        )
        rows.append(
            {
                "task": m.group("task"),
                "n_ops": int(m.group("n_ops")),
                "num_steps": int(m.group("num_steps")),
                "init_seed": int(m.group("seed")),
                "observed": res["observed"],
                "null_mean": res["null_mean"],
                "null_std": res["null_std"],
                "p_value": res["p_value"],
                "z_score": z,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Report whether observed winding beats the matched-random-walk null."""
    df = cached("winding_null.csv", compute)

    n = len(df)
    sig = int((df["p_value"] < 0.05).sum())
    print(f"\n--- Matched-random-walk null test on {n} real Huginn trajectories ---")
    print(f"Surrogates per trajectory: {N_SURROGATES}")
    print(f"Trajectories whose |winding| beats the null at p<0.05: {sig}/{n} ({sig / n:.1%})")
    print(f"  (chance expectation if the metric carried no signal: ~5% = {0.05 * n:.1f})")

    print("\n--- Observed vs null magnitude ---")
    print(f"  mean observed |winding| : {df['observed'].mean():.4f}")
    print(f"  mean null-mean |winding|: {df['null_mean'].mean():.4f}")
    print(f"  mean difference         : {(df['observed'] - df['null_mean']).mean():+.4f}")
    print(f"  mean z-score            : {df['z_score'].mean():+.3f}")
    print(f"  median p-value          : {df['p_value'].median():.3f}")

    print("\n--- By task / compute budget ---")
    print(
        df.groupby(["task", "num_steps"]).agg(
            n=("p_value", "size"),
            frac_sig=("p_value", lambda s: (s < 0.05).mean()),
            mean_z=("z_score", "mean"),
            mean_obs=("observed", "mean"),
            mean_null=("null_mean", "mean"),
        )
    )

    print(
        "\nHOW TO READ THIS (contract fixed before the run, see module docstring): "
        "if the significant fraction is at or near the 5% chance rate and the mean "
        "z-score is ~0, then observed winding is indistinguishable from a random "
        "walk with the same step sizes -- i.e. the winding metric as implemented "
        "carries no rotational information on this data, and every winding-based "
        "claim in this project inherits that caveat."
    )


if __name__ == "__main__":
    main()
