"""Experiment B4.15 — H2 tested on the Jacobian eigenvalue argument, length-matched.

Reads `scratch/kaggle_h2rot/out/h2rot.json` (produced by the `geometry-h2-rotation`
kernel) and writes `results/h2_rotation.csv`. No GPU needed. OWNER: Data+Analysis.
STATUS: implemented 2026-08-09.

WHY THE ANALYSIS LIVES HERE AND NOT IN THE KERNEL. Three kernels in this project
(D62, `geometry-discourse`, D70) printed a confident verdict that was wrong, in
every case because the verdict was computed from the same variables as the run and
so agreed with the run's mistakes. Keeping the inference local makes it unit-tested
against planted and absent effects, and re-runnable when a question arrives that
the original design did not anticipate -- which in this project has been about one
question per experiment (B4.14).

WHAT IS BEING TESTED. H2 says rotation grows with the number of REASONING STEPS a
problem requires. `make_variants` supplies a length-matched pair: byte-identical
body, `track` needing accumulation over all n_ops steps, `local` needing only the
final step, with `local` a constant +3 tokens at every n_ops. So the H2 statistic
is the PAIRED DIFFERENCE arg(track) - arg(local) against n_ops: if rotation tracks
reasoning it must rise, and if it merely tracks prompt length the difference is
flat while both arms rise together. Winding could not make this distinction --
D28 showed its sign flips with the recording budget alone.

THE NULL IS EXACT, NOT SAMPLED. With 6 levels x 2 seeds there are 12 matched
track/local pairs, so the paired sign-flip null has 2^12 = 4096 arrangements and
can be enumerated rather than sampled. The floor is then 1/4096 = 2.4e-4, well
under any alpha used here -- which matters because `geometry-correctness` was
drafted with n_perm=200 against alpha=0.00417, making rejection arithmetically
impossible and guaranteeing a null that reads like a result (D72, `attainable`).

Run: uv run python -m scripts.run_h2_rotation
"""

from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np
import pandas as pd

from traj_geom.analysis.correlate import spearman_by_level

RAW = os.path.join("scratch", "kaggle_h2rot", "out", "h2rot.json")
OUT = os.path.join("results", "h2_rotation.csv")

# Unrolls to shrink the gap to EPS of its initial size: t* = ln(EPS)/ln(rho).
# Used only for the derived total-phase quantity; H2 is about total turns, not rate.
EPS = 0.01


def load_records(path: str = RAW) -> pd.DataFrame:
    """Flatten the kernel's JSON into one row per measured prompt."""
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    rows = [r for r in blob["records"] if r.get("ok")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["arg"] = df["arg_osc"].astype(float)
    df["turns"] = total_turns(df["arg"], df["rho"])
    return df


def total_turns(arg: np.ndarray, rho: np.ndarray, eps: float = EPS) -> np.ndarray:
    """H2's own quantity: full turns accumulated before the state settles.

    Rotation per unroll is `arg`; the unrolls needed to shrink the gap to `eps` is
    t* = ln(eps)/ln(rho) (rigor_audit section 3, which also showed steps-to-settle
    is predicted by rho alone and carries no task information). Total phase is the
    product, and turns is that over 2*pi. Both factors come from the same spectrum,
    so this needs no extra measurement.
    """
    arg = np.asarray(arg, float)
    rho = np.asarray(rho, float)
    t_star = np.log(eps) / np.log(np.clip(rho, 1e-6, 1 - 1e-9))
    return arg * t_star / (2.0 * math.pi)


def paired_diff(df: pd.DataFrame, value: str = "arg") -> pd.DataFrame:
    """One row per (n_ops, seed) with the track-minus-local difference.

    Pairs are the unit of analysis because the two arms share a byte-identical
    body; anything that varies with the text cancels in the difference.
    """
    wide = df.pivot_table(index=["n_ops", "seed"], columns="kind", values=value)
    wide = wide.dropna(subset=["track", "local"])
    wide["diff"] = wide["track"] - wide["local"]
    return wide.reset_index()


def sign_flip_p(pairs: pd.DataFrame, value: str = "diff") -> tuple[float, float, int]:
    """Exact two-sided paired sign-flip test of `spearman(diff, n_ops) == 0`.

    Under the null that `kind` is arbitrary, swapping `track` and `local` within a
    pair flips the sign of that pair's difference. Enumerating all 2^n sign
    patterns gives an exact p; only if n is large do we fall back to sampling.

    Returns ``(rho_observed, p, n_arrangements)``.
    """
    d = pairs[value].to_numpy(float)
    lv = pairs["n_ops"].to_numpy(float)
    n = len(d)

    obs = spearman_by_level(pd.DataFrame({"n_ops": lv, "v": d}), "n_ops", "v")[0]
    if n <= 16:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    else:  # pragma: no cover - the grid this project runs is 12 pairs
        rng = np.random.default_rng(0)
        signs = rng.choice([1.0, -1.0], size=(20000, n))
    null = _rho_by_level_batch(lv, d[None, :] * signs)
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    return float(obs), p, len(signs)


def _rho_by_level_batch(levels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """`spearman_by_level`'s rho for a whole batch of value-vectors at once.

    Identical by construction to calling `spearman_by_level` per row -- group to
    one mean per level, then Spearman the L levels -- but vectorised, because the
    exact null enumerates 2**12 = 4096 arrangements and a pandas groupby per
    arrangement made the test suite take eight minutes. `tests/test_h2_rotation.py
    ::test_batch_rho_matches_the_canonical_statistic` pins the two together on
    random inputs, so the fast path cannot drift from the statistic the rest of
    the project reports.

    Args:
        levels: [n] the level (x) of each pair.
        values: [S, n] one row per arrangement.

    Returns:
        [S] Spearman rho of level-mean against level, per row.
    """
    from scipy.stats import rankdata

    uniq = np.unique(levels)
    weights = np.stack([(levels == u) / max(int((levels == u).sum()), 1) for u in uniq])
    means = values @ weights.T                      # [S, L] one mean per level
    # `uniq` is already sorted ascending, so its ranks are 1..L.
    x = np.arange(1, len(uniq) + 1, dtype=float)
    y = rankdata(means, axis=1)
    xc = x - x.mean()
    yc = y - y.mean(axis=1, keepdims=True)
    den = np.sqrt((xc**2).sum() * (yc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (yc @ xc) / den, np.nan)


def analyse(df: pd.DataFrame, value: str = "arg") -> dict:
    """All statistics, per arm. Pure: takes a frame, returns numbers.

    P1 is a gate on the INSTRUMENT, not the hypothesis: arg(lambda) is undefined
    for a real eigenvalue, so if the spectrum is mostly real this measures nothing
    and the rest may not be read.
    """
    out: dict = {}
    for arm, sub in df.groupby("arm"):
        complex_frac = float(sub["arg"].notna().mean())
        sub = sub.dropna(subset=["arg"])
        res = {
            "n": int(len(sub)),
            "complex_frac": complex_frac,
            "p1_gate_ok": bool(complex_frac >= 0.8),
        }
        for kind in ("track", "local"):
            k = sub[sub["kind"] == kind]
            if len(k):
                rho, nlev, crit = spearman_by_level(k, "n_ops", value)
                res[kind] = {"rho": rho, "n_levels": nlev, "crit_p05": crit,
                             "sig": bool(crit is not None and abs(rho) >= crit),
                             "mean": float(k[value].mean())}
        pairs = paired_diff(sub, value)
        if len(pairs) >= 4:
            obs, p, n_arr = sign_flip_p(pairs)
            res["paired"] = {"n_pairs": int(len(pairs)), "rho_diff_vs_n_ops": obs,
                             "p_exact": p, "arrangements": int(n_arr),
                             "p_floor": 1.0 / n_arr,
                             "mean_diff": float(pairs["diff"].mean())}
        out[arm] = res
    return out


def _fmt(res: dict) -> str:
    lines = []
    for arm, r in res.items():
        lines.append(f"\n=== {arm} (n={r['n']}) ===")
        verdict = ("PASS" if r["p1_gate_ok"]
                   else "FAIL -- arg is undefined; do not read below")
        lines.append(f"  P1 gate: {r['complex_frac']:.0%} of prompts have a complex "
                     f"oscillatory mode -- {verdict}")
        for kind in ("track", "local"):
            if kind in r:
                k = r[kind]
                crit = "n/a" if k["crit_p05"] is None else f"{k['crit_p05']:.3f}"
                lines.append(f"  {kind:>6}: |arg| vs n_ops rho={k['rho']:+.3f} "
                             f"(N={k['n_levels']}, crit={crit}, "
                             f"{'sig' if k['sig'] else 'n.s.'}), mean |arg|={k['mean']:.3f}")
        if "paired" in r:
            p = r["paired"]
            lines.append(f"  PAIRED (H2's statistic): rho(track-local vs n_ops) = "
                         f"{p['rho_diff_vs_n_ops']:+.3f}, exact p={p['p_exact']:.4f} "
                         f"over {p['arrangements']} arrangements "
                         f"(floor {p['p_floor']:.2e}), mean diff={p['mean_diff']:+.4f}")
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not os.path.exists(RAW):
        print(f"{RAW} not present -- run the geometry-h2-rotation kernel and "
              f"`kaggle kernels output` it into that directory first.")
        return
    df = load_records()
    if df.empty:
        print("no successful records in the raw JSON; nothing to analyse.")
        return
    keep = ["arm", "kind", "n_ops", "seed", "rho", "arg", "turns", "n_osc",
            "lead_is_real", "n_tokens", "matvecs", "secs"]
    save_table(OUT, df[[c for c in keep if c in df.columns]], kind="h2_rotation",
               source=RAW, eps=EPS)
    print(f"saved {OUT}  ({len(df)} measured prompts)")
    print(_fmt(analyse(df)))
    print("\n(|arg| is rotation per unroll; `turns` is the derived total before "
          "settling, arg * ln(eps)/ln(rho) / 2pi.)")


if __name__ == "__main__":
    main()
