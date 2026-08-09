"""Experiment — winding against an on-manifold null, at full power.

OWNER: Data+Analysis
STATUS: implemented 2026-07-25 (second audit pass). Supersedes the ad-hoc
    subsample runs whose numbers appear in an earlier draft of
    docs/rigor_audit.md section 17 -- see WHY THIS EXISTS.
TASK: for every banked trajectory, compare |winding| against a surrogate that
    preserves the state manifold, the radial convergence profile and every
    consecutive angular step, randomising only rotational freedom.

WHY THIS EXISTS (a correction to my own earlier reporting)
  Section 17 of the audit first reported "the winding null reverses": 66% of
  trajectories beating the on-manifold null against 8.5% for the off-manifold
  one. That came from 47 trajectories x 40 surrogates, with a 20-trajectory
  calibration giving a mean excess of +0.0031.

  A subsequent 8-trajectory check restricted to num_steps=128 gave a mean
  excess of **-0.0087** -- the opposite sign. Both samples are too small and
  they were not stratified by num_steps, so neither supports a conclusion. The
  honest position until this script runs at full power is that the sign of the
  effect is UNKNOWN, not that it reversed.

  This script therefore runs every trajectory against many surrogates and
  reports results stratified by num_steps and task, with effect sizes
  alongside z-scores (a tightly-constrained null inflates z, so z alone is
  misleading).

WHAT IT ALSO RECORDS, AND WHY
  * ``pr_real`` / ``pr_null``: participation ratio of the azimuth directions.
    Winding depends on effective dimensionality, and the real paths are more
    concentrated than the isotropic surrogate, so any excess must be checked
    against that gap rather than read directly.
  * ``obs_minus_null``: the effect size. Report this, not just z.
  * A CALIBRATION arm: a surrogate is fed back in as though it were data. An
    unbiased procedure must beat its own null at about the nominal rate; if
    the calibration arm shows an excess, the construction is biased and the
    real-data arm means nothing.

I/O: -> results/manifold_null.csv (one row per trajectory per arm) with a
    provenance sidecar.

Run: uv run python -m scripts.run_manifold_null
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from traj_geom.metrics.surrogate import (  # noqa: E402
    manifold_matched_surrogate,
    spherical_decomposition,
)
from traj_geom.metrics.winding import winding_of  # noqa: E402
from traj_geom.provenance import save_table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ_DIR = os.path.join(ROOT, "results", "trajectories")
N_SURROGATES = 100
_FNAME_RE = re.compile(
    r"^(?P<task>[a-z_]+)_n(?P<n_ops>\d+)_ns(?P<num_steps>\d+)_init(?P<seed>\d+)\.npy$"
)


def azimuth_participation_ratio(traj: np.ndarray) -> float:
    """Effective dimensionality of a trajectory's AZIMUTH directions.

    RENAMED 2026-08-09. This was called `participation_ratio`, colliding with
    `traj_geom.metrics.dimension.participation_ratio` while computing a DIFFERENT
    quantity: that one takes the PR of the rows it is handed (used on unit step
    directions), this one first takes the spherical decomposition and uses the
    mean-centred azimuth component. Same functional form, different input, same
    name, same repo -- so importing the wrong one silently returns a different
    number. The name now says which quantity it is.

    The shared math is delegated to the canonical implementation rather than
    re-derived here; only the input preparation is local, which is the part that
    actually differs. Used because winding is measured after a 2-D projection, so
    effective dimensionality is a confound for it.
    """
    from traj_geom.metrics.dimension import participation_ratio as _pr
    _, _, q, _ = spherical_decomposition(np.asarray(traj, dtype=np.float64))
    qc = q - q.mean(0)
    if not np.any(qc):
        return float("nan")
    return _pr(qc)


def _one_arm(observed_traj: np.ndarray, seed0: int) -> dict[str, float]:
    """Score one trajectory against N_SURROGATES of its own surrogates."""
    obs = abs(winding_of(observed_traj))
    vals = np.empty(N_SURROGATES)
    prs = np.empty(N_SURROGATES)
    for j in range(N_SURROGATES):
        s = manifold_matched_surrogate(observed_traj, np.random.default_rng(seed0 + j))
        vals[j] = abs(winding_of(s))
        prs[j] = azimuth_participation_ratio(s)
    sd = float(vals.std())
    return {
        "observed": float(obs),
        "null_mean": float(vals.mean()),
        "null_std": sd,
        "obs_minus_null": float(obs - vals.mean()),
        "z": float((obs - vals.mean()) / sd) if sd > 0 else float("nan"),
        "p_value": float(np.mean(vals >= obs)),
        "pr_real": azimuth_participation_ratio(observed_traj),
        "pr_null": float(prs.mean()),
    }


def compute() -> pd.DataFrame:
    """Real-data arm plus a calibration arm, for every banked trajectory."""
    paths = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"no trajectories in {TRAJ_DIR}")

    rows = []
    for i, path in enumerate(paths):
        m = _FNAME_RE.match(os.path.basename(path))
        if m is None:
            continue
        a = np.load(path).astype(np.float64)
        meta = {
            "file": os.path.basename(path),
            "task": m.group("task"),
            "n_ops": int(m.group("n_ops")),
            "num_steps": int(m.group("num_steps")),
            "task_seed": int(m.group("seed")),
        }
        rows.append({**meta, "arm": "real", **_one_arm(a, 10_000 + i * 1_000)})
        # calibration: a surrogate treated as data, scored against ITS surrogates
        fake = manifold_matched_surrogate(a, np.random.default_rng(500_000 + i))
        rows.append({**meta, "arm": "calibration", **_one_arm(fake, 900_000 + i * 1_000)})
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(paths)} trajectories", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    """Run both arms and report stratified, with effect sizes beside z."""
    df = compute()
    out = os.path.join(ROOT, "results", "manifold_null.csv")
    save_table(
        out, df, kind="table", experiment="manifold_null", n_surrogates=N_SURROGATES,
        null="on-manifold: sphere + radial profile + consecutive angular steps preserved",
        note="supersedes the subsample numbers in an earlier draft of rigor_audit section 17",
    )

    for arm in ("calibration", "real"):
        s = df[df.arm == arm]
        print(f"\n=== {arm.upper()} arm ({len(s)} trajectories x {N_SURROGATES} surrogates) ===")
        print(f"  beats null (p<.05): {(s.p_value < 0.05).mean():.1%}"
              f"   below null: {(s.z < 0).mean():.1%}")
        print(f"  mean effect (obs - null): {s.obs_minus_null.mean():+.5f}"
              f"  (sd {s.obs_minus_null.std():.5f})")
        print(f"  mean z: {s.z.mean():+.2f}   median z: {s.z.median():+.2f}")
        print(f"  PR real {s.pr_real.mean():.1f} vs null {s.pr_null.mean():.1f}")

    print("\n=== real arm, stratified by num_steps ===")
    r = df[df.arm == "real"]
    print(r.groupby("num_steps")[["obs_minus_null", "z", "pr_real", "pr_null"]]
          .agg(["mean", "count"]).to_string())
    print("\n=== real arm, stratified by task ===")
    print(r.groupby("task")[["obs_minus_null", "z"]].mean().to_string())

    cal_rate = (df[df.arm == "calibration"].p_value < 0.05).mean()
    print(f"\nREAD THIS FIRST: calibration arm beats its own null at {cal_rate:.1%}. "
          f"If that is far from ~5%, the construction is biased and the real arm "
          f"cannot be interpreted.")
    print(f"\nwrote {out} (+ provenance sidecar)")


if __name__ == "__main__":
    main()
