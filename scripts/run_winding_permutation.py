"""Experiment — H2 tested with no surrogate model at all.

OWNER: Data+Analysis
STATUS: implemented and run 2026-07-26.
TASK: ask whether |winding| tracks difficulty using ONLY real trajectories,
    by permuting the difficulty labels.

WHY THIS IS THE STRONGEST AVAILABLE TEST OF H2
  Every other winding test in this project compares real trajectories against
  synthetic surrogates, and docs/rigor_audit.md shows how much rides on the
  surrogate being right: the original `matched_random_walk` leaves the state
  manifold entirely (§8), and even the corrected on-manifold null needs its own
  calibration and an anisotropy control (§17).

  A permutation test needs none of that. The 140 banked trajectories are real,
  so they already lie on the manifold with every structure intact -- the
  convergence profile, the noise floor, the anisotropy, the bf16 quantisation.
  Permuting the `n_ops` labels destroys ONLY the association between winding
  and difficulty, which is exactly H2's claim. No property of the trajectories
  is modelled, approximated or synthesised.

  The trade-off, stated so it is not over-read: this can test "does winding
  VARY with difficulty" but not "is there rotation AT ALL", because every
  trajectory shares whatever baseline rotation exists and permutation cannot
  see a constant. For the latter question a surrogate is unavoidable --
  see scripts/run_manifold_null.py.

STRATIFICATION, AND WHY
  Results are computed per (task, num_steps). Pooling is not safe: §12 of the
  audit shows ns=16/64/128 differ in whether the path has converged, and an
  earlier pooled analysis produced a spurious cross-task agreement that
  vanished on stratification. Multiplicity across strata is then corrected
  explicitly -- with 4 strata, P(at least one nominal hit under the global
  null) is 0.185, so an uncorrected single hit means nothing.

INTERPRETATION CEILING
  Even a surviving positive result would NOT establish that difficulty drives
  winding: `docs/rigor_audit.md` §21 shows rank-corr(n_ops, seq_len) = exactly
  1.000 for every synthetic generator in the project, so difficulty and prompt
  length are indistinguishable by construction. This test can therefore falsify
  H2 but cannot confirm it. That asymmetry is the point of running it.

I/O: -> results/winding_permutation.csv, with a provenance sidecar.

Run: uv run python -m scripts.run_winding_permutation
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from traj_geom.metrics.winding import winding_of  # noqa: E402
from traj_geom.provenance import save_table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ_DIR = os.path.join(ROOT, "results", "trajectories")
N_PERM = 20_000
MIN_LEVELS = 4
_FNAME_RE = re.compile(
    r"^(?P<task>[a-z_]+)_n(?P<n_ops>\d+)_ns(?P<num_steps>\d+)_init(?P<seed>\d+)\.npy$"
)


def _load() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.npy"))):
        m = _FNAME_RE.match(os.path.basename(path))
        if m is None:
            continue
        rows.append(
            {
                "task": m.group("task"),
                "n_ops": int(m.group("n_ops")),
                "num_steps": int(m.group("num_steps")),
                "task_seed": int(m.group("seed")),
                "winding": abs(winding_of(np.load(path))),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no trajectories in {TRAJ_DIR}")
    return pd.DataFrame(rows)


def compute() -> pd.DataFrame:
    """Per-stratum observed rho and its permutation p-value."""
    df = _load()
    rng = np.random.default_rng(0)
    out = []
    for (task, ns), g in df.groupby(["task", "num_steps"]):
        levels = g.groupby("n_ops")["winding"].mean()
        if len(levels) < MIN_LEVELS:
            continue
        obs = spearmanr(levels.index, levels.values)[0]
        null = np.array(
            [spearmanr(levels.index, rng.permutation(levels.values))[0] for _ in range(N_PERM)]
        )
        out.append(
            {
                "task": task,
                "num_steps": ns,
                "n_levels": len(levels),
                "n_trajectories": len(g),
                "rho": float(obs),
                "p_perm": float(np.mean(np.abs(null) >= abs(obs))),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    """Report per stratum, then correct for multiplicity across strata."""
    df = compute().sort_values("p_perm").reset_index(drop=True)
    m = len(df)
    df["bh_crit"] = 0.05 * (df.index + 1) / m
    df["bh_pass"] = df.p_perm <= df.bh_crit
    df["bonferroni_p"] = (df.p_perm * m).clip(upper=1.0)

    out = os.path.join(ROOT, "results", "winding_permutation.csv")
    save_table(
        out, df, kind="table", experiment="winding_permutation", n_permutations=N_PERM,
        design="real trajectories only; n_ops labels permuted within (task, num_steps)",
        ceiling="rank-corr(n_ops, seq_len)=1.000 for every generator, so a positive "
                "result could not be attributed to difficulty rather than length",
    )

    print(f"\n=== |winding| ~ n_ops, permutation test ({N_PERM} permutations/stratum) ===\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  strata: {m}   any surviving BH: {bool(df.bh_pass.any())}")
    print(f"  P(>=1 nominal hit under the global null) = {1 - 0.95**m:.3f}")
    if not df.bh_pass.any():
        print("\n  VERDICT: no stratum survives multiplicity correction. H2, as "
              "|winding| vs n_ops on answer-token trajectories, is NULL under the "
              "assumption-free test.")
    print(f"\nwrote {out} (+ provenance sidecar)")


if __name__ == "__main__":
    main()
