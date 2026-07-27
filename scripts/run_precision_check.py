"""Experiment — is the "limit set" arithmetic? (plan Step 1, the decisive test)

OWNER: Data+Analysis
STATUS: implemented 2026-07-26. NEEDS GPU, runs in minutes. Not yet run.
TASK: extract the same prompt's trajectory in bfloat16 and in float32 and
    compare the residual motion after convergence.

WHY THIS IS THE HIGHEST VALUE-PER-COST ITEM IN THE PROJECT
  The audit's central structural claim is that a trajectory has ~19 informative
  steps followed by 45-110 steps of pure arithmetic noise, and that four
  separate metrics failed by averaging over both regimes (docs/rigor_audit.md
  sections 1-3, 6; claims_ledger D24, D29). Everything in `metrics/regime.py`
  rests on that claim.

  The evidence for it is currently indirect: the residual motion is isotropic
  (tail top-2 PCA 8.8-10.7%, participation ratio 18-19/20, where genuine 2-D
  dynamics would give ~2), and its magnitude matches accumulated bf16 rounding
  under contraction, floor = sqrt(k) * eta_1 / sqrt(1 - rho^2) with k ~ 27-40
  arithmetic sites per unroll (D25(1)).

  This script tests it DIRECTLY. bf16 has 8 mantissa bits, float32 has 24, so
  the unit roundoff ratio is 2^-16 ~ 1.5e-5.

      PREDICTION: the post-convergence residual radius drops from ~0.91 to
      ~1.4e-5, a factor of ~65 000.

  Outcomes and what each means:
    * ratio ~ 2^-16      -> the floor is arithmetic. Confirms the whole
                            regime-separation framework.
    * ratio ~ 1          -> the floor is NOT arithmetic; there is a genuine
                            invariant set of diameter ~0.9, and sections 1-3
                            and 6 of the audit must be re-derived. Everything
                            downstream should stop.
    * intermediate       -> partly arithmetic, partly dynamics; report the
                            ratio and rescope.

  This is a GATE, not a measurement: run it before spending GPU on Steps 2-4.

I/O: -> results/precision_check.csv, with a provenance sidecar. Also saves both
    raw trajectories WITH provenance so the comparison is auditable later.

Run (GPU): uv run python -m scripts.run_precision_check
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from traj_geom.metrics.regime import converging_regime, empirical_floor  # noqa: E402
from traj_geom.provenance import save_array, save_table  # noqa: E402
from traj_geom.shapes.synthetic import make_count_ones_task  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_TRAJ = os.path.join(ROOT, "results", "precision_check")
NUM_STEPS = 128
N_TAIL = 20
CASES = ((8, 0), (32, 0), (32, 1))


def _summarise(a: np.ndarray) -> dict[str, float]:
    """Residual motion after convergence, the quantity under test."""
    a = np.asarray(a, dtype=np.float64)
    ref = a[-N_TAIL:].mean(0)
    radius = np.linalg.norm(a - ref, axis=1)
    _, end = converging_regime(a)
    return {
        "state_norm": float(np.linalg.norm(a, axis=1).mean()),
        "regime_end": int(end),
        "floor_step": float(empirical_floor(a)),
        "residual_radius": float(radius[end:].mean()) if end < len(a) else float("nan"),
    }


def compute() -> pd.DataFrame:
    """Same prompt, two dtypes, everything else identical."""
    import torch

    from traj_geom.extraction.hook import extract_trajectory
    from traj_geom.extraction.model import load_huginn

    rows = []
    for n_ops, seed in CASES:
        task = make_count_ones_task(n_ops, seed=seed)
        per_dtype = {}
        for name, dtype in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
            model, tok = load_huginn(dtype=dtype)
            traj = extract_trajectory(model, tok, task["prompt"], num_steps=NUM_STEPS, seed=0)
            path = os.path.join(OUT_TRAJ, f"count_ones_n{n_ops}_seed{seed}_{name}.npy")
            save_array(
                path, traj, kind="trajectory", task="count_ones", n_ops=n_ops,
                task_seed=seed, init_seed=0, num_steps=NUM_STEPS,
                compute_dtype=name, prompt=task["prompt"], answer=task["answer"],
                purpose="precision gate: plan_forward Step 1",
            )
            per_dtype[name] = _summarise(traj)
            del model
            torch.cuda.empty_cache()

        b, f = per_dtype["bfloat16"], per_dtype["float32"]
        rows.append(
            {
                "n_ops": n_ops, "task_seed": seed,
                **{f"bf16_{k}": v for k, v in b.items()},
                **{f"fp32_{k}": v for k, v in f.items()},
                "residual_ratio": f["residual_radius"] / b["residual_radius"]
                if b["residual_radius"] else np.nan,
            }
        )
        print(f"  n_ops={n_ops} seed={seed}: bf16 residual {b['residual_radius']:.4g}, "
              f"fp32 {f['residual_radius']:.4g}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    """Run the gate and state which branch the result falls into."""
    df = compute()
    out = os.path.join(ROOT, "results", "precision_check.csv")
    save_table(
        out, df, kind="table", experiment="precision_check",
        prediction="fp32/bf16 residual ratio ~ 2^-16 = 1.5e-5 if the floor is arithmetic",
        role="GATE for docs/plan_forward.md Steps 2-4",
    )
    print("\n=== precision gate ===\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    ratio = df.residual_ratio.mean()
    expected = 2.0**-16
    print(f"\n  mean fp32/bf16 residual ratio: {ratio:.3e}   predicted {expected:.3e}")
    if ratio < 10 * expected:
        print("  => THE FLOOR IS ARITHMETIC. The regime-separation framework is confirmed; "
              "proceed to Steps 2-4.")
    elif ratio > 0.1:
        print("  => THE FLOOR IS NOT ARITHMETIC. A genuine invariant set exists. "
              "rigor_audit sections 1-3 and 6 must be re-derived and downstream work "
              "should STOP until they are.")
    else:
        print(f"  => PARTIAL: the floor shrinks by {1/ratio:.0f}x, short of the "
              f"{1/expected:.0f}x predicted. Some residual motion is arithmetic and some "
              "is not; rescope before proceeding.")
    print(f"\nwrote {out} (+ provenance sidecars for both raw trajectories)")


if __name__ == "__main__":
    main()
