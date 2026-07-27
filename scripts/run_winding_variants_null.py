"""Experiment — which rotation metric, if any, actually beats chance?

OWNER: Data+Analysis
STATUS: implemented 2026-07-25. Follow-up to claims_ledger.md D22, where the
    project's baseline winding metric failed a matched-random-walk null test.
    That raised the obvious question: is the FAILURE a property of Huginn's
    trajectories (they really do not rotate), or of that particular
    ESTIMATOR (angle about the centroid of a collapsing path)? This script
    settles it by putting six estimators (src/traj_geom/metrics/
    winding_variants.py) through the identical null.
TASK: for each banked trajectory, compute all variants on the real path and
    on N matched random walks (same per-step displacement magnitudes,
    directions randomised on the hypersphere), then score each variant by how
    far the real value sits from its own surrogate distribution.

THE DECISION RULE, FIXED BEFORE THE RUN (so no metric shopping):
  A variant is better than the baseline only if BOTH hold --
    (1) DISCRIMINATION: its observed value exceeds its own null more often /
        by more z than W1 does. Raw magnitude is meaningless across variants
        (they have different scales), so only z and p are compared.
    (2) INFORMATIVENESS: it actually varies across conditions. A metric that
        returns a constant cannot correlate with difficulty no matter how it
        scores against a null -- this is precisely how W1 failed (std 0.005
        on the best-powered sweep).
  W5 is included as a CONTROL expected to fail (1): random directions turn
  ~90 deg/step, so total unsigned turning is necessarily far below a
  random-walk null. If W5 "loses" by a mile, the harness is behaving.

I/O: reads results/trajectories/*.npy -> results/winding_variants_null.csv
    (one row per trajectory x variant: observed, null_mean, null_std,
    z_score, p_value, plus task/n_ops/num_steps/init_seed).

Run: uv run python -m scripts.run_winding_variants_null
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts._common import RESULTS_DIR, cached
from traj_geom.metrics.winding import matched_random_walk
from traj_geom.metrics.winding_variants import VARIANT_NAMES, winding_variants

TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")
N_SURROGATES = 100
_FNAME_RE = re.compile(
    r"^(?P<task>[a-z_]+)_n(?P<n_ops>\d+)_ns(?P<num_steps>\d+)_init(?P<seed>\d+)\.npy$"
)


def compute() -> pd.DataFrame:
    """Null-test every variant on every banked trajectory."""
    paths = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.npy")))
    if not paths:
        raise FileNotFoundError(
            f"no trajectories in {TRAJ_DIR} -- needs the raw [T, hidden] .npy "
            "paths (untracked/local, not in git)."
        )

    rows = []
    for path in tqdm(paths, desc="variant null"):
        name = os.path.basename(path)
        m = _FNAME_RE.match(name)
        if m is None:
            print(f"run_winding_variants_null: skipping unparseable {name}")
            continue
        traj = np.load(path)
        obs = winding_variants(traj)

        rng = np.random.default_rng(abs(hash(name)) % (2**31))
        null = {k: [] for k in VARIANT_NAMES}
        for _ in range(N_SURROGATES):
            sur = winding_variants(matched_random_walk(traj, rng))
            for k, v in sur.items():
                null[k].append(v)

        for k in VARIANT_NAMES:
            arr = np.asarray(null[k], dtype=float)
            sd = float(arr.std())
            rows.append(
                {
                    "task": m.group("task"),
                    "n_ops": int(m.group("n_ops")),
                    "num_steps": int(m.group("num_steps")),
                    "init_seed": int(m.group("seed")),
                    "variant": k,
                    "observed": obs[k],
                    "null_mean": float(arr.mean()),
                    "null_std": sd,
                    "z_score": (obs[k] - arr.mean()) / sd if sd > 0 else np.nan,
                    "p_value": float(np.mean(arr >= obs[k])),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Rank the variants by how well they separate from their own null."""
    df = cached("winding_variants_null.csv", compute)
    n_traj = df.groupby("variant").size().iloc[0]

    print(f"\n--- {len(VARIANT_NAMES)} rotation metrics vs matched-random-walk null ---")
    print(f"{n_traj} real Huginn trajectories x {N_SURROGATES} surrogates each\n")

    agg = (
        df.groupby("variant")
        .agg(
            mean_z=("z_score", "mean"),
            median_z=("z_score", "median"),
            frac_beats_null=("p_value", lambda s: (s < 0.05).mean()),
            frac_below_null=("z_score", lambda s: (s < 0).mean()),
            obs_mean=("observed", "mean"),
            obs_std=("observed", "std"),
        )
        .sort_values("mean_z", ascending=False)
    )
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n--- criterion 2: does the metric vary across conditions at all? ---")
    for v in agg.index:
        s = df[df["variant"] == v]
        cv = s["observed"].std() / max(abs(s["observed"].mean()), 1e-12)
        print(f"  {v:22s} obs std={s['observed'].std():8.3f}  coeff-of-variation={cv:.3f}")

    print("\n--- criterion 3: does it track difficulty (n_ops), per task? ---")
    from scipy.stats import spearmanr

    for v in agg.index:
        s = df[df["variant"] == v]
        line = []
        for task in sorted(s["task"].unique()):
            t = s[s["task"] == task]
            g = t.groupby("n_ops")["observed"].mean()
            rho = spearmanr(g.index, g.values)[0] if len(g) > 2 else np.nan
            line.append(f"{task}:{rho:+.2f}")
        print(f"  {v:22s} per-level rho -> " + "  ".join(line))

    best = agg.index[0]
    print(
        f"\nRanked by mean z (higher = further ABOVE its own chance baseline). "
        f"Top: {best}. Read with the pre-registered rule in this module's "
        "docstring: a variant only wins if it both beats its null AND varies "
        "across conditions; W5 is the deliberate failing control."
    )


if __name__ == "__main__":
    main()
