"""H3, measured with the estimator H3 is actually about -- for the first time.

WHY THIS COULD NOT BE RUN BEFORE, AND WHY IT CAN NOW. `contraction_from_pair`
(`src/traj_geom/metrics/regime.py`) measures whether the recurrent MAP pulls nearby
states together: it needs two orbits of the SAME prompt started from DIFFERENT h_0.
Its own docstring records that `results/trajectories/*.npy` does NOT satisfy this --
`init{N}` there is the TASK seed, so those files differ by prompt, a premise this
project checked and filed under "do not re-derive" (`docs/directions.md` section 0).
The one directory that did vary h_0 only reached ns=64 with a handful of draws.

`geometry-h0bank` (D93) incidentally produced exactly the missing dataset: **16
prompts x 32 UNSEEDED h_0 draws** at full depth, with a `fix` block at a fixed seed
proving the draws are the only thing varying (D93's P1 gate confirmed seeded orbits
are bit-identical and unseeded ones all distinct). It was banked to test D91's
correctness lead; it happens to be the ideal H3 substrate. Zero GPU.

WHAT H3 CLAIMS AND WHAT WOULD REFUTE IT. H3 is the contraction hypothesis: the
recurrent map is a contraction, so nearby states converge (rho < 1). The competing
readings the supervisor raised -- that contraction may not be "the single best
regime", and that loop/drift might persist -- predict rho >= 1 for at least some
prompts, or a residual gap floor that does not shrink.

PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE RUNNING (CLAUDE.md section 1):
  1. Median rho across the 16 prompts is < 1 (contraction), consistent with D52's
     rho in [0.808, 0.920] measured by a different route.
  2. If (1) holds, the interesting quantity is the SPREAD: does any prompt show
     rho >= 1? A single prompt sitting at or above 1 would be the first direct
     evidence for a non-contracting regime on this model.
  3. The residual gap `floor` is bounded away from 0 -- orbits from different h_0
     do NOT fully merge, which is what makes correctness h_0-dependent at all
     (D90). A floor at ~0 would CONTRADICT D90 and indict one of the two.

Run: uv run python -m scripts.run_h3_contraction
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd

from traj_geom.metrics.regime import contraction_from_pair

BANK = os.path.join("scratch", "kaggle_h0bank", "out")
OUT = os.path.join("results", "h3_contraction.csv")
MAX_PAIRS = 60      # per prompt, from C(32,2)=496 -- enough for a stable median
SEED = 0


def load(path: str = BANK) -> dict[str, list[np.ndarray]]:
    """Group the UNSEEDED (`rep`) orbits by prompt. The `fix` block is excluded:
    it is the determinism control and its orbits are bit-identical by design, so
    a contraction fitted on it would be fitting zero."""
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fh:
        recs = [r for r in json.load(fh) if r.get("ok") and r.get("block") == "rep"]
    by_prompt: dict[str, list[np.ndarray]] = {}
    for r in recs:
        f = os.path.join(path, r["tag"] + ".npy")
        if not os.path.exists(f):
            continue
        key = f"{r['family']}_i{r['item']:02d}"
        by_prompt.setdefault(key, []).append(np.load(f).astype(np.float64))
    return by_prompt


def analyse(by_prompt: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for prompt, orbits in sorted(by_prompt.items()):
        if len(orbits) < 2:
            continue
        pairs = list(itertools.combinations(range(len(orbits)), 2))
        if len(pairs) > MAX_PAIRS:
            idx = rng.choice(len(pairs), MAX_PAIRS, replace=False)
            pairs = [pairs[i] for i in idx]
        rhos, floors = [], []
        for i, j in pairs:
            rho, floor = contraction_from_pair(orbits[i], orbits[j])
            if np.isfinite(rho):
                rhos.append(rho)
                floors.append(floor)
        if not rhos:
            continue
        rhos_a, floors_a = np.array(rhos), np.array(floors)
        rows.append({
            "prompt": prompt, "n_orbits": len(orbits), "n_pairs_used": len(rhos),
            "rho_median": float(np.median(rhos_a)),
            "rho_p05": float(np.percentile(rhos_a, 5)),
            "rho_p95": float(np.percentile(rhos_a, 95)),
            "rho_max": float(rhos_a.max()),
            "frac_pairs_rho_ge_1": float((rhos_a >= 1.0).mean()),
            "floor_median": float(np.median(floors_a)),
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    if df.empty:
        return "  no usable prompts -- is the h0bank output present?"
    lines = ["", f"  {'prompt':>22} {'n':>3} {'rho med':>8} {'rho p05':>8} "
                 f"{'rho p95':>8} {'rho max':>8} {'%>=1':>6} {'floor':>10}"]
    for _, r in df.iterrows():
        lines.append(f"  {r['prompt']:>22} {int(r['n_orbits']):>3} "
                     f"{r['rho_median']:>8.4f} {r['rho_p05']:>8.4f} "
                     f"{r['rho_p95']:>8.4f} {r['rho_max']:>8.4f} "
                     f"{r['frac_pairs_rho_ge_1']:>6.1%} {r['floor_median']:>10.3e}")
    med = df["rho_median"].median()
    n_ge1 = int((df["rho_median"] >= 1.0).sum())
    any_pair = float(df["frac_pairs_rho_ge_1"].max())
    floor_med = df["floor_median"].median()
    verdict1 = "CONTRACTING (rho < 1), H3 supported" if med < 1 else "NOT contracting"
    verdict3 = ("bounded away from 0, consistent with D90" if floor_med > 1e-6
                else "AT ZERO -- contradicts D90, investigate")
    lines += ["", "  PRE-REGISTERED READINGS",
              f"    (1) median rho over {len(df)} prompts = {med:.4f} -> {verdict1}",
              f"    (2) prompts with median rho >= 1: {n_ge1}/{len(df)}; "
              f"worst single prompt's share of pairs at rho >= 1 = {any_pair:.1%}",
              f"    (3) residual gap floor, median over prompts = "
              f"{floor_med:.3e} -> {verdict3}"]
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not os.path.exists(os.path.join(BANK, "manifest.json")):
        print(f"no banked states in {BANK} -- pull the geometry-h0bank output first.")
        return
    by_prompt = load()
    df = analyse(by_prompt)
    if len(df):
        save_table(OUT, df, kind="h3_contraction", max_pairs=MAX_PAIRS, seed=SEED)
        print(f"saved {OUT}")
    print(report(df))


if __name__ == "__main__":
    main()
