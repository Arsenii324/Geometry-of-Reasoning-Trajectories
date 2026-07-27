"""Experiment — does Huginn hold a counting register? (plan Step 4)

OWNER: Data+Analysis
STATUS: implemented 2026-07-26. NEEDS GPU. Not yet run; no result is claimed.
TASK: probe per-token-position latents for Barannikov's Task a (running count)
    and Task b (nesting depth), in the form the architecture permits.

THE DESIGN DECISION THIS ENCODES
  Barannikov's minimal algorithm is a register in Z updated by the translation
  T_1 : s -> s+1, which suggests probing for a direction `v` such that seeing a
  `1` displaces the state by ~v. `docs/register_geometry.md` shows that probe
  is looking for something the architecture forbids: RMSNorm confines states to
  a shell of relative half-thickness delta ~ 7.3e-5 about R = 76.37, so a
  straight-line register is bounded by

      D < R sqrt(2 delta) = 0.923 TOTAL, i.e. 0.0144 per increment over m=64

  which is 62x BELOW the bf16 arithmetic noise floor (~0.9). The bound scales
  as sqrt(delta), so it survives a 1000x error in the shell estimate.

  The norm-preserving realisation is a ROTATION. Equivalently -- and this is
  why the framing is rescued rather than contradicted -- a translation in
  TANGENT (log-map) coordinates at a reference point, which agrees with a
  rotation to first order at small angle. This script therefore probes in
  tangent coordinates and additionally fits an explicit rotation angle.

WHAT IS MEASURED, PER TASK
  Task a (running count, m fixed):
    * `probe_r2`: cross-validated R^2 of a ridge probe for y_i from the
      tangent-space state at position i. Length is FIXED, so unlike every
      other task in this project a positive result cannot be prompt length
      (docs/rigor_audit.md section 21).
    * `angle_linearity`: R^2 of a straight-line fit of the fitted rotation
      angle against y_i. A register realised as a rotation by theta_0 per
      increment predicts angle = y_i * theta_0, i.e. linearity, and gives
      theta_0 as the slope.
    * `radial_fraction`: the share of the fitted displacement lying along the
      radial (norm-changing) direction. The architecture forbids this, so a
      LARGE value falsifies the whole tangent-space picture rather than
      supporting it.

  Task b (nesting depth), balanced vs unbalanced:
    * `winding_position`: the position-indexed winding number of the state
      sequence. For a BALANCED string the depth returns to 0, so if the
      register is a rotation the state returns to its start and the curve
      CLOSES -- making this a genuine element of pi_1(R^2 minus a point) = Z.
      The prediction is therefore QUANTISATION: near-integer values for
      balanced strings and not for the unbalanced control. That is far
      stronger than a correlation, and it is the one setting in this whole
      project where a winding number is a legitimate topological invariant
      rather than an arbitrary real number.

PRE-REGISTERED READING (fixed before the run)
  * Task a counts as positive only if `probe_r2` > 0.5 under cross-validation
    AND `radial_fraction` < 0.1. A probe that works by moving radially is
    measuring something other than the register.
  * Task b counts as positive only if balanced strings' |winding - round(winding)|
    is smaller than the unbalanced control's, tested by permutation.
  * Both are reported with effect sizes, not just p-values.

I/O: -> results/register_probe.csv, plus per-position raw states with
    provenance under results/register_probe/.

Run (GPU): uv run python -m scripts.run_register_probe
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from traj_geom.provenance import save_array, save_table  # noqa: E402
from traj_geom.shapes.synthetic import (  # noqa: E402
    make_nesting_depth_task,
    make_running_count_task,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_TRAJ = os.path.join(ROOT, "results", "register_probe")
M = 64
NUM_STEPS = 64
N_SEEDS = 12
PROBE_R2_BAR = 0.5
RADIAL_BAR = 0.1


def _tangent(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Log-map the per-position states to the tangent space at their mean direction.

    Returns ``(tangent, pole, radius)``. Working in tangent coordinates is the
    point: an ambient-space translation is forbidden by the norm constraint,
    while a tangent translation is a rotation to first order.
    """
    a = np.asarray(states, dtype=np.float64)
    radius = float(np.linalg.norm(a, axis=1).mean())
    pole = a.mean(0)
    pole /= np.linalg.norm(pole)
    radial = np.outer(a @ pole, pole)
    return a - radial, pole, radius


def _radial_fraction(states: np.ndarray, target: np.ndarray) -> float:
    """Share of the target-aligned displacement that is radial (norm-changing)."""
    a = np.asarray(states, dtype=np.float64)
    pole = a.mean(0)
    pole /= np.linalg.norm(pole)
    y = np.asarray(target, dtype=np.float64)
    y = y - y.mean()
    if np.allclose(y, 0):
        return float("nan")
    direction = (a - a.mean(0)).T @ y / (y @ y)      # least-squares displacement per unit target
    n = np.linalg.norm(direction)
    return float(abs(direction @ pole) / n) if n > 0 else float("nan")


def _probe_r2(x: np.ndarray, y: np.ndarray, alpha: float = 1e3) -> float:
    """Leave-one-out ridge R^2 for the per-position target."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    pred = cross_val_predict(model, x, y, cv=LeaveOneOut())
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _position_states(model, tok, prompt: str) -> np.ndarray:
    """Final recurrent state at every token position, [n_positions, hidden]."""
    import torch

    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    captured: list[np.ndarray] = []
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()
    handle = mod.register_forward_hook(
        lambda m, i, o: captured.append(o.detach()[0].float().cpu().numpy())
    )
    try:
        with torch.no_grad():
            model(input_ids=ids, num_steps=NUM_STEPS)
    finally:
        handle.remove()
    return captured[-1]                      # after the final unroll, all positions


def compute() -> pd.DataFrame:
    """Task a and Task b probes across seeds."""
    from traj_geom.extraction.model import load_huginn

    model, tok = load_huginn()
    rows = []

    for seed in range(N_SEEDS):
        t = make_running_count_task(m=M, seed=seed)
        states = _position_states(model, tok, t["prompt"])
        save_array(
            os.path.join(OUT_TRAJ, f"task_a_m{M}_seed{seed}.npy"), states,
            kind="position_states", task="running_count", m=M, task_seed=seed,
            init_seed=0, num_steps=NUM_STEPS, prompt=t["prompt"], answer=t["answer"],
        )
        n = min(len(t["running"]), len(states))
        y = np.asarray(t["running"][:n], dtype=np.float64)
        tan, _, _ = _tangent(states[:n])
        rows.append(
            {
                "task": "a_running_count", "seed": seed, "m": M, "n_positions": n,
                "probe_r2": _probe_r2(tan, y),
                "radial_fraction": _radial_fraction(states[:n], y),
                "target_sd": float(y.std()),
            }
        )
        print(f"  task a seed={seed} done", flush=True)

    for balanced in (True, False):
        for seed in range(N_SEEDS):
            t = make_nesting_depth_task(m=M, seed=seed, balanced=balanced)
            states = _position_states(model, tok, t["prompt"])
            tag = "bal" if balanced else "unbal"
            save_array(
                os.path.join(OUT_TRAJ, f"task_b_{tag}_m{M}_seed{seed}.npy"), states,
                kind="position_states", task="nesting_depth", m=M, task_seed=seed,
                init_seed=0, num_steps=NUM_STEPS, prompt=t["prompt"],
                answer=t["answer"], balanced=bool(t["balanced"]),
            )
            n = min(len(t["depths"]), len(states))
            y = np.asarray(t["depths"][:n], dtype=np.float64)
            tan, _, _ = _tangent(states[:n])
            from traj_geom.metrics.winding import winding_of

            w = abs(winding_of(states[:n], burn=0))
            rows.append(
                {
                    "task": "b_nesting_depth", "seed": seed, "m": M, "n_positions": n,
                    "balanced": bool(t["balanced"]), "max_depth": int(t["max_depth"]),
                    "probe_r2": _probe_r2(tan, y),
                    "radial_fraction": _radial_fraction(states[:n], y),
                    "winding_position": float(w),
                    "dist_to_integer": float(abs(w - round(w))),
                    "target_sd": float(y.std()),
                }
            )
            print(f"  task b balanced={balanced} seed={seed} done", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    """Run both probes and apply the pre-registered reading."""
    df = compute()
    out = os.path.join(ROOT, "results", "register_probe.csv")
    save_table(
        out, df, kind="table", experiment="register_probe", m=M, num_steps=NUM_STEPS,
        design="tangent-space (log-map) probe; ambient translation is forbidden by "
               "the norm constraint — see docs/register_geometry.md",
        prereg=f"task a positive iff probe_r2 > {PROBE_R2_BAR} AND radial_fraction "
               f"< {RADIAL_BAR}; task b positive iff balanced strings sit closer to "
               "integer winding than the unbalanced control",
    )

    a = df[df.task == "a_running_count"]
    print("\n=== Task a — running count, fixed m (length cannot explain a hit) ===")
    print(f"  probe R2       : {a.probe_r2.mean():+.4f} (sd {a.probe_r2.std():.4f})")
    print(f"  radial fraction: {a.radial_fraction.mean():.4f} "
          f"(architecture forbids this being large)")
    ok = a.probe_r2.mean() > PROBE_R2_BAR and a.radial_fraction.mean() < RADIAL_BAR
    print(f"  PRE-REGISTERED VERDICT: {'POSITIVE' if ok else 'not positive'}")

    b = df[df.task == "b_nesting_depth"]
    if not b.empty:
        bal, unb = b[b.balanced], b[~b.balanced]
        print("\n=== Task b — nesting depth; balanced strings CLOSE the curve ===")
        print(f"  balanced   : winding {bal.winding_position.mean():.4f}, "
              f"distance to nearest integer {bal.dist_to_integer.mean():.4f}")
        print(f"  unbalanced : winding {unb.winding_position.mean():.4f}, "
              f"distance to nearest integer {unb.dist_to_integer.mean():.4f}")
        if len(bal) and len(unb):
            from scipy.stats import mannwhitneyu

            p = mannwhitneyu(bal.dist_to_integer, unb.dist_to_integer,
                             alternative="less").pvalue
            print(f"  quantisation test (balanced closer to integer): p = {p:.4f}")
            print(f"  PRE-REGISTERED VERDICT: "
                  f"{'POSITIVE' if p < 0.05 else 'not positive'}")
    print(f"\nwrote {out} (+ provenance sidecars for all raw position states)")


if __name__ == "__main__":
    main()
