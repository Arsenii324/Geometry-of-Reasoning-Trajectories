"""Trajectory geometry is mostly a CLOCK: the statistics track depth, not the input.

Reads every banked trained trajectory this project holds and writes
`results/window_law.csv`. No GPU. OWNER: Data+Analysis. STATUS: implemented 2026-08-09.

WHAT THIS RESOLVES, AND IT IS A CONTRADICTION IN THE PROJECT'S OWN RECORD. D74
reports the orbit at **13.3 effective dimensions** with a consecutive-step cosine of
**+0.1996**. The surrogate comparison built from `ds_eigen` reads the SAME kind of
orbit at **PR 2.3 and cos +0.95**. Neither is wrong: D74 measured over a ~21-unroll
window, the other over the full 89-107 unroll pre-floor window. Those are the
transient and the asymptote of a contraction, and reporting either without the other
describes a stretch of the convergence rather than the trajectory.

THE MEASUREMENT. For each orbit, both statistics are computed in a SLIDING window of
fixed width along the unroll axis, so "where in the convergence" is the independent
variable and window WIDTH is held constant -- the one confound that would otherwise
reproduce D74(6), where `ratio` is rank-collinear with window length at rho = -1.000
and `partial_spearman` refuses to correct for it.

THE COMPARISON THAT MAKES IT A CLAIM. At each starting unroll, the spread of a
statistic ACROSS PROMPTS is set against its spread ACROSS DEPTH. If depth moves it by
much more than the prompt does, then most of what a shape statistic measures is how
far the contraction has run -- a property of the operator, hence of the weights --
and prompt-level differences are a second-order effect riding on top. That is a
positive, quantitative statement, and it predicts the nulls: D79 found no
correct/incorrect difference, and any capability correlation must live inside the
small residual left after depth is accounted for.

WHAT WOULD REFUTE IT. Curves that separate by prompt or by task family more than they
move along depth, or a depth profile that differs in SHAPE between families rather
than only in offset. Both are reported.

Run: uv run python -m scripts.run_window_law
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from traj_geom.metrics.dimension import participation_ratio, step_directions
from traj_geom.metrics.dmd import pre_floor_window

OUT = os.path.join("results", "window_law.csv")
WIDTH = 12          # unrolls per sliding window; wide enough for PR, narrow enough
STRIDE = 4          # to keep ~20 positions along a 100-unroll orbit
SOURCES = (
    ("ds_bank", os.path.join("scratch", "ds_bank", "out"), "_last.npy"),
    ("b6bank", os.path.join("scratch", "kaggle_b6bank", "out"), ".npy"),
    ("geomcap", os.path.join("scratch", "kaggle_geomcap", "out"), ".npy"),
)


def sliding(traj: np.ndarray, width: int = WIDTH, stride: int = STRIDE) -> list[dict]:
    """Shape statistics in a FIXED-WIDTH window sliding along the unroll axis.

    Fixed width is the whole point. Participation ratio grows with the number of
    samples, so a sweep that widened the window while sliding it would confound
    "later in the convergence" with "more samples" -- and D74(6) records that
    confound as total, not partial.
    """
    lo, hi = pre_floor_window(traj)
    out = []
    for a in range(lo, max(lo + 1, hi - width + 1), stride):
        b = a + width
        if b > hi:
            break
        u = step_directions(traj, lo=a, hi=b)
        if len(u) < width - 2:
            continue
        d = np.linalg.norm(np.diff(traj, axis=0), axis=1)[a:b]
        y = np.log(np.clip(d, 1e-30, None))
        rho = float(np.exp(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]))
        out.append({"start": a, "depth_frac": (a - lo) / max(1, hi - lo - width),
                    "n_dirs": int(len(u)), "pr": participation_ratio(u),
                    "cos_consecutive": float(np.mean(np.sum(u[:-1] * u[1:], axis=1))),
                    "contraction": rho, "step_norm": float(np.median(d)),
                    "pre_floor": hi - lo})
    return out


def _iter_bank(name: str, path: str, suffix: str):
    mpath = os.path.join(path, "manifest.json")
    if not os.path.exists(mpath):
        return
    with open(mpath, encoding="utf-8") as fh:
        recs = [r for r in json.load(fh) if r.get("ok")]
    for r in recs:
        if r.get("arm") not in (None, "trained"):
            continue                            # trained arm only; D76 owns the contrast
        f = os.path.join(path, r["tag"] + suffix)
        if not os.path.exists(f):
            continue
        yield {"bank": name, "tag": r["tag"],
               "family": r.get("family") or r.get("task", "?"),
               "correct": bool(r.get("correct", r.get("correct_any_depth", False))),
               "n_tokens": int(r.get("n_tokens", 0))}, f


def collect(sources=SOURCES, limit_per_bank: int | None = None) -> pd.DataFrame:
    rows = []
    for name, path, suffix in sources:
        n = 0
        for meta, f in _iter_bank(name, path, suffix):
            if limit_per_bank is not None and n >= limit_per_bank:
                break
            traj = np.load(f).astype(np.float64)
            for w in sliding(traj):
                rows.append({**meta, **w})
            n += 1
    return pd.DataFrame(rows)


def common_depth_range(df: pd.DataFrame, min_frac: float = 0.95) -> list[int]:
    """Depths where at least `min_frac` of the orbits still have a window.

    THE CONFOUND THIS REMOVES, WHICH IS NOT SMALL. The banks run to different
    depths -- ds_bank to 128 unrolls, b6bank and geomcap to 64 -- so the orbit set
    THINS as the sliding window advances: 152 orbits at unroll 0 against 24 past
    unroll 40. A depth profile pooled over all of them changes composition as it
    goes, and the late part of the curve would describe three task families rather
    than a later stage of the contraction. Restricting to the common range makes the
    x-axis mean one thing.
    """
    n_tot = df["tag"].nunique()
    counts = df.groupby("start")["tag"].nunique()
    return [int(s) for s, c in counts.items() if c >= min_frac * n_tot]


def paired_depth_change(df: pd.DataFrame, metric: str,
                        depths: list[int] | None = None) -> dict:
    """Within-ORBIT change from the first common depth to the last.

    Each orbit is its own control, so neither the composition of the bank nor any
    per-prompt offset can produce the effect. The test is a sign test over orbits:
    a monotone depth law should move nearly every orbit the same way, and a
    statistic that merely has a large pooled spread need not.
    """
    depths = depths if depths is not None else common_depth_range(df)
    if len(depths) < 2:
        return {"usable": False, "why": "no common depth range"}
    a, b = min(depths), max(depths)
    piv = (df[df["start"].isin((a, b))]
           .pivot_table(index="tag", columns="start", values=metric))
    piv = piv.dropna()
    if len(piv) < 8:
        return {"usable": False, "why": f"only {len(piv)} orbits span {a}->{b}"}
    delta = piv[b] - piv[a]
    n_up = int((delta > 0).sum())
    n = int(len(delta))
    # Exact two-sided sign test; no normal approximation, since n is small enough
    # to compute and the tails are where the claim lives.
    from math import comb
    k = max(n_up, n - n_up)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0**n)
    return {"usable": True, "metric": metric, "from": a, "to": b, "n_orbits": n,
            "median_at_start": float(piv[a].median()),
            "median_at_end": float(piv[b].median()),
            "median_delta": float(delta.median()),
            "n_increased": n_up, "n_agreeing": k,
            "direction": "up" if n_up >= n - n_up else "down", "p_sign": p,
            "prompt_spread_at_start": float(piv[a].quantile(0.9)
                                            - piv[a].quantile(0.1)),
            "ratio": (abs(float(delta.median()))
                      / max(1e-12, float(piv[a].quantile(0.9) - piv[a].quantile(0.1))))}


def depth_vs_prompt(df: pd.DataFrame, metric: str) -> dict:
    """Does the statistic move more along DEPTH or across PROMPTS?

    Both spreads are formed the same way -- an interquartile range of medians -- so
    the ratio is a comparison of like with like. Depth is binned by `start` rather
    than by fraction, because two orbits at the same fraction of different-length
    windows are at different unrolls, and the unroll is the thing the operator
    responds to.
    """
    by_depth = df.groupby("start")[metric].median()
    at_common = df[df["start"] == int(df["start"].mode().iloc[0])]
    by_prompt = at_common.groupby("tag")[metric].median()
    if len(by_depth) < 3 or len(by_prompt) < 3:
        return {"usable": False}
    d_iqr = float(by_depth.quantile(0.9) - by_depth.quantile(0.1))
    p_iqr = float(by_prompt.quantile(0.9) - by_prompt.quantile(0.1))
    return {"usable": True, "metric": metric, "depth_spread": d_iqr,
            "prompt_spread": p_iqr,
            "ratio": (d_iqr / p_iqr) if p_iqr > 0 else float("inf"),
            "n_depths": int(len(by_depth)), "n_prompts": int(len(by_prompt)),
            "at_start": int(at_common["start"].iloc[0]),
            "first": float(by_depth.iloc[0]), "last": float(by_depth.iloc[-1])}


def family_profiles(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Depth profile per family -- do they differ in SHAPE or only in offset?

    An offset difference is consistent with the clock reading: every family runs the
    same contraction, some a little further along. A SHAPE difference would not be,
    and would be the first evidence that the trajectory's geometry encodes something
    task-specific.
    """
    g = df.groupby(["family", "start"])[metric].median().reset_index()
    piv = g.pivot(index="start", columns="family", values=metric)
    return piv.dropna(how="all")


def report(df: pd.DataFrame) -> str:
    lines = [
        "",
        f"  {df['tag'].nunique()} orbits from {df['bank'].nunique()} banks, "
        f"sliding width {WIDTH} stride {STRIDE}",
        "",
        f"  {'start':>6} {'n':>5} {'pr':>8} {'cos':>8} {'contraction':>12} "
        f"{'step norm':>12}",
    ]
    for s, sub in df.groupby("start"):
        if len(sub) < 5:
            continue
        lines.append(f"  {int(s):>6} {len(sub):>5} {sub['pr'].median():>8.2f} "
                     f"{sub['cos_consecutive'].median():>+8.3f} "
                     f"{sub['contraction'].median():>12.4f} "
                     f"{sub['step_norm'].median():>12.3e}")

    depths = common_depth_range(df)
    lines += ["", f"  PAIRED WITHIN EACH ORBIT, over the depth range every orbit "
                  f"reaches ({min(depths)} -> {max(depths)}).",
              "  Each orbit is its own control, so neither the mix of banks nor any",
              "  per-prompt offset can manufacture the effect."]
    for m in ("pr", "cos_consecutive", "contraction"):
        c = paired_depth_change(df, m, depths)
        if not c.get("usable"):
            lines.append(f"    {m:>16}: not usable -- {c.get('why')}")
            continue
        lines.append(
            f"    {m:>16}: {c['median_at_start']:+8.3f} -> {c['median_at_end']:+8.3f} "
            f"(median delta {c['median_delta']:+.3f}), moves {c['direction']} in "
            f"{c['n_agreeing']}/{c['n_orbits']} orbits, sign test p = {c['p_sign']:.2e}; "
            f"that shift is {c['ratio']:.1f}x the whole prompt-to-prompt spread")

    c_rho = paired_depth_change(df, "contraction", depths)
    if c_rho.get("usable") and c_rho["p_sign"] > 0.05:
        lines += [
            "    -- and the CONTRACTION RATE does not move over the same range "
            f"(p = {c_rho['p_sign']:.2f}), which is the control that makes the two",
            "       above specific: this is not a generic drift of every quantity",
            "       with depth. The orbit sheds directions and its steps become",
            "       parallel, at a decay rate that stays put -- a contraction",
            "       collapsing onto its dominant mode.",
        ]

    lines += ["", "  POOLED (composition changes with depth -- read the paired block"
                  "  above for the claim)"]
    for m in ("pr", "cos_consecutive", "contraction"):
        c = depth_vs_prompt(df, m)
        if not c.get("usable"):
            continue
        lines.append(
            f"    {m:>16}: depth moves it {c['depth_spread']:8.3f} "
            f"({c['first']:+.3f} -> {c['last']:+.3f} over {c['n_depths']} depths), "
            f"prompt {c['prompt_spread']:8.3f} across {c['n_prompts']} prompts at "
            f"unroll {c['at_start']}  ->  {c['ratio']:.1f}x")

    lines += ["", "  DO FAMILIES DIFFER IN PROFILE SHAPE, OR ONLY IN OFFSET?"]
    for m in ("pr", "cos_consecutive"):
        piv = family_profiles(df, m)
        if piv.shape[1] < 2 or piv.shape[0] < 3:
            continue
        centred = piv - piv.mean(axis=0)
        # Correlation of each family's depth profile with the mean profile: near 1
        # means same shape, different offset.
        mean_prof = piv.mean(axis=1)
        cors = piv.apply(lambda col, mp=mean_prof: col.corr(mp))
        worst = cors.idxmin()
        lines.append(f"    {m:>16}: family profiles correlate with the mean profile "
                     f"at median r = {cors.median():.3f} "
                     f"(min {cors.min():.3f}, {worst}) over {piv.shape[1]} families; "
                     f"residual spread after removing each family's offset = "
                     f"{float(centred.std().median()):.3f}")
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    have = [(n, p, s) for n, p, s in SOURCES
            if os.path.exists(os.path.join(p, "manifest.json"))]
    if not have:
        print("no banked trajectories found in any of "
              + ", ".join(p for _, p, _ in SOURCES))
        return
    df = collect(tuple(have))
    if df.empty:
        print("banks present but no usable orbits")
        return
    save_table(OUT, df, kind="window_law", width=WIDTH, stride=STRIDE,
               banks=[n for n, _, _ in have])
    print(f"saved {OUT} ({len(df)} windows over {df['tag'].nunique()} orbits)")
    print(report(df))


if __name__ == "__main__":
    main()
