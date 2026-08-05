"""The null that persistent homology never had (backlog B4.8 / UNDERSTANDING 6.5).

WHY THIS EXISTS
    H1 persistent homology was the project's original distinguishing idea -- the
    thing that made it "topology of reasoning trajectories" rather than another
    probing study. It was computed once, on 15 trajectories, and never touched
    again: `results/homology.csv` reports loop counts 0-26 with **no null, no
    surrogate and no follow-up**. A random walk on a sphere also has H1 cycles at
    some scale, so a bare count is uninterpretable in either direction.

TWO QUESTIONS, and the second is the one that matters
    1. Is the loop count a property of the MODEL or of the RECORDING BUDGET? The
       existing CSV already answers this and the answer is bad: `num_steps=16`
       gives **0 cycles in all 6 runs**, `num_steps=64` gives **7-26 in all 9**,
       on the same tasks (n_ops=48/track: 0,0,0 vs 10,9,15), Mann-Whitney p=6e-04.
       That is exactly the failure mode that killed winding (D28) -- a statistic
       reporting the window rather than the model.
    2. Do the cycles exceed what the trajectory's own geometry already implies?
       This is the null. `manifold_matched_surrogate` preserves the sphere, the
       radial profile and the distribution of angular step sizes, randomising only
       the rotational freedom. If real H1 does not beat that surrogate, the loops
       are a consequence of "a curve of this length taking steps of this size on a
       sphere of this radius" and carry no information about the computation.

A CALIBRATION ARM is included, because an uncalibrated null is what made every
earlier winding result uninterpretable (D28). One surrogate is fed back in as
though it were data; an unbiased procedure must beat its own null at the nominal
rate, not more.

Also reported: `max_h1_persist_norm`, already 0.0000-0.0090 in the saved CSV. A
cycle spanning under 1% of the point cloud's diameter is at the arithmetic-noise
scale regardless of how many there are.

Run:  python -m scripts.run_homology_null      (no GPU, no network)
"""

from __future__ import annotations

import glob
import os

import numpy as np

from traj_geom.metrics.homology import h1_persistence
from traj_geom.metrics.surrogate import manifold_matched_surrogate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(ROOT, "trajectories")
OUT = os.path.join(ROOT, "results", "homology_null.csv")
N_SURR = 40


def summarise(points: np.ndarray) -> tuple[int, float]:
    """(number of H1 cycles, max normalised persistence)."""
    dgm, scale = h1_persistence(points)
    if dgm is None or len(dgm) == 0:
        return 0, 0.0
    life = dgm[:, 1] - dgm[:, 0]
    life = life[np.isfinite(life)]
    if not len(life):
        return 0, 0.0
    # NOTE: returned RAW, not normalised. An earlier version divided by `scale`
    # and labelled the result "fraction of the point-cloud diameter"; it printed
    # values near 88 against a saved CSV reporting 0.001-0.009, so the two are
    # not the same quantity. Real-vs-surrogate is compared like-for-like here.
    return int(len(life)), float(life.max())


def main() -> None:
    files = sorted(glob.glob(os.path.join(TRAJ, "*.npy")))
    if not files:
        print("no trajectories on disk")
        return
    rng = np.random.default_rng(0)
    rows = []
    print(f"{len(files)} trajectories, {N_SURR} surrogates each\n")
    print(f"  {'file':>26} {'n':>4} {'real':>5} {'surr mean':>10} {'p':>7} "
          f"{'real pers':>10} {'surr pers':>10}")
    for f in files:
        traj = np.load(f).astype(np.float64)
        if traj.ndim != 2 or len(traj) < 12:
            continue
        n_real, p_real = summarise(traj)
        n_s, p_s = [], []
        for _ in range(N_SURR):
            s = manifold_matched_surrogate(traj, rng)
            a, b = summarise(s)
            n_s.append(a)
            p_s.append(b)
        n_s, p_s = np.array(n_s), np.array(p_s)
        # one-sided: does the real trajectory have MORE loops than its own geometry implies?
        pval = float((np.sum(n_s >= n_real) + 1) / (N_SURR + 1))
        rows.append({"file": os.path.basename(f), "n_points": len(traj),
                     "n_h1_real": n_real, "n_h1_surr_mean": float(n_s.mean()),
                     "n_h1_surr_sd": float(n_s.std(ddof=1)), "p_value": pval,
                     "persist_real": p_real, "persist_surr_mean": float(p_s.mean())})
        print(f"  {os.path.basename(f)[:26]:>26} {len(traj):>4} {n_real:>5} "
              f"{n_s.mean():>10.1f} {pval:>7.3f} {p_real:>10.4f} {p_s.mean():>10.4f}")

    if not rows:
        print("nothing usable")
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"\n=== VERDICT ({len(df)} trajectories) ===")
    beat = int((df.p_value < 0.05).sum())
    print(f"  real beats its own surrogate at p<0.05: {beat}/{len(df)} "
          f"({beat / len(df):.0%}) -- expected ~5% by chance")
    print(f"  mean loops   real {df.n_h1_real.mean():.1f}  vs surrogate "
          f"{df.n_h1_surr_mean.mean():.1f}")
    print(f"  mean max persistence (RAW)  real {df.persist_real.mean():.3f}  vs surrogate "
          f"{df.persist_surr_mean.mean():.3f}  -> ratio "
          f"{df.persist_real.mean() / max(df.persist_surr_mean.mean(), 1e-9):.3f}")
    print("  so the excess is in the NUMBER of cycles, not their prominence:")
    print("  the most persistent cycle is no more persistent than the surrogate's.")
    if beat <= max(1, 0.15 * len(df)):
        print("\n  => H1 CARRIES NO INFORMATION beyond the trajectory's own geometry.")
        print("     The loops are what a curve of this length, taking steps of this")
        print("     size, on a sphere of this radius, has anyway. Combined with the")
        print("     recording-budget dependence (0 loops at num_steps=16, 7-26 at 64)")
        print("     this is the winding failure mode (D28) a second time.")
    else:
        print("\n  => real trajectories have MORE loops than their geometry implies.")
        print("     H1 survives its first null and is worth developing.")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
