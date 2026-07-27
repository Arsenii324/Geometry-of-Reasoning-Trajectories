"""Step 5 — re-derive existing conclusions with regime-restricted metrics.

OWNER: Data+Analysis
STATUS: implemented and run 2026-07-26. This is `docs/plan_forward.md` Step 5.
TASK: recompute every metric the audit invalidated, both the legacy way and the
    floor-aware way, on every trajectory whose raw path still exists, and
    report which CONCLUSIONS change sign or significance.

WHAT CAN AND CANNOT BE RE-DERIVED
  Re-derivation needs raw `[T, hidden]` paths. Only two sets exist on disk:
    * `trajectories/` (15 files + manifest) — the source of `convergence.csv`
      and `homology.csv`. Verified: the legacy metrics reproduce from these
      paths to the 4 decimal places the CSVs store (rank-corr +0.999/+1.000/
      +1.000). So those two results ARE re-derivable.
    * `results/trajectories/` (140 files) — NOT the source of
      `full_synthetic_experiments.csv` (0/80 winding values reproduce, D24(4)).
      They can be re-analysed on their own terms but cannot be used to correct
      any existing CSV.

  Every other invalidated CSV (`counting`, `switch`, `maxtask`, `three_scale*`,
  `dissociation*`, `blayney_repro`, `pararule`, `forceloop`, `h2_loops`,
  `dissoc_multiinit`, `smoke_new_tasks`, `counting_accuracy`) has **no raw
  paths on disk**. Their conclusions cannot be corrected without a GPU re-run.
  That is a finding, not an omission: the project stored derived scalars and
  discarded the data they were derived from.

THE HEADLINE TEST
  `consecutive_step_cosine` should, if D24(2) is right, be governed by whether
  the recording is long enough to reach the arithmetic floor — not by anything
  about the task. `convergence.csv` contains both ns=16 and ns=64 rows and so
  tests this directly on the project's own data.

I/O: -> results/regime_rederivation.csv, with a provenance sidecar.

Run: uv run python -m scripts.run_regime_rederivation
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from traj_geom.metrics.convergence import (  # noqa: E402
    consecutive_step_cosine,
    convergence_rate,
    drift_to_loop_ratio,
)
from traj_geom.metrics.dynamics import steps_to_settle  # noqa: E402
from traj_geom.metrics.regime import (  # noqa: E402
    converging_regime,
    empirical_floor,
    step_cosine_converging,
)
from traj_geom.provenance import save_table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP_TRAJ = os.path.join(ROOT, "trajectories")
RES_TRAJ = os.path.join(ROOT, "results", "trajectories")
_RES_RE = re.compile(
    r"^(?P<task>[a-z_]+)_n(?P<n_ops>\d+)_ns(?P<num_steps>\d+)_init(?P<seed>\d+)\.npy$"
)


def _both_ways(a: np.ndarray) -> dict[str, float]:
    """Every affected metric, computed legacy and floor-aware."""
    _, end = converging_regime(a)
    sig = a[: end + 1]
    return {
        "n_steps": len(a) - 1,
        "regime_end": end,
        "frac_signal": end / max(len(a) - 1, 1),
        "floor": empirical_floor(a),
        "cos_legacy": consecutive_step_cosine(a),
        "cos_regime": step_cosine_converging(a),
        "conv_rate_legacy": convergence_rate(a),
        "conv_rate_regime": convergence_rate(sig) if len(sig) > 4 else np.nan,
        "dlr_legacy": drift_to_loop_ratio(a),
        "dlr_regime": drift_to_loop_ratio(sig) if len(sig) > 2 else np.nan,
        "settle_legacy": steps_to_settle(a),
    }


def compute() -> pd.DataFrame:
    """Both metric families on every trajectory with a raw path on disk."""
    rows = []
    man = os.path.join(TOP_TRAJ, "manifest.csv")
    if os.path.exists(man):
        for _, r in pd.read_csv(man).iterrows():
            p = os.path.join(TOP_TRAJ, str(r["file"]))
            if not os.path.exists(p):
                continue
            rows.append(
                {
                    "source": "trajectories/ (convergence.csv, homology.csv)",
                    "file": r["file"], "task": r["kind"], "n_ops": int(r["n_ops"]),
                    "num_steps": int(r["num_steps"]),
                    **_both_ways(np.load(p).astype(np.float64)),
                }
            )
    for p in sorted(glob.glob(os.path.join(RES_TRAJ, "*.npy"))):
        m = _RES_RE.match(os.path.basename(p))
        if m is None:
            continue
        rows.append(
            {
                "source": "results/trajectories/ (no matching CSV)",
                "file": os.path.basename(p), "task": m.group("task"),
                "n_ops": int(m.group("n_ops")), "num_steps": int(m.group("num_steps")),
                **_both_ways(np.load(p).astype(np.float64)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Report which conclusions change when the noise regime is excluded."""
    df = compute()
    out = os.path.join(ROOT, "results", "regime_rederivation.csv")
    save_table(
        out, df, kind="table", experiment="regime_rederivation",
        scope="only trajectories with raw paths on disk; every other invalidated "
              "CSV lacks raw data and needs a GPU re-run",
    )

    print(f"\n=== Step 5: legacy vs regime-restricted metrics, {len(df)} trajectories ===\n")
    print("--- how much of each recording is actually signal? ---")
    print(df.groupby("num_steps")[["regime_end", "frac_signal", "floor"]]
          .mean().to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n--- THE HEADLINE: does the sign of `cos` follow the recording length? ---")
    g = df.groupby("num_steps")[["cos_legacy", "cos_regime"]].agg(["mean", "count"])
    print(g.to_string(float_format=lambda x: f"{x:+.4f}"))
    flips = df[(df.cos_legacy < 0) & (df.cos_regime > 0)]
    print(f"\n  trajectories where the legacy cosine is NEGATIVE but the "
          f"signal-regime cosine is POSITIVE: {len(flips)}/{len(df)} = {len(flips)/len(df):.1%}")
    short = df[df.num_steps <= 16]
    if len(short):
        print(f"  at num_steps<=16 (too short to reach the floor) legacy cos is "
              f"{short.cos_legacy.mean():+.4f} — already positive, no floor to contaminate it")

    print("\n--- magnitude of the correction on the other metrics ---")
    for a, b in (("conv_rate_legacy", "conv_rate_regime"), ("dlr_legacy", "dlr_regime")):
        d = df[[a, b]].dropna()
        ratio = (d[a] / d[b]).replace([np.inf, -np.inf], np.nan).dropna()
        name = a.split("_legacy")[0]
        print(f"  {name:10s} legacy {d[a].mean():+9.4f}  regime {d[b].mean():+9.4f}"
              f"   mean ratio {ratio.mean():.2f}x")

    print("\n--- what CANNOT be re-derived ---")
    print("  16 of the 18 invalidated results CSVs have no raw paths on disk.")
    print("  Their conclusions require a GPU re-run; see docs/plan_forward.md.")
    print(f"\nwrote {out} (+ provenance sidecar)")


if __name__ == "__main__":
    main()
