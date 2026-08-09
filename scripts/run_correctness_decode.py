"""Is correctness decodable from the orbit's SHAPE — and what could this design see?

Reads every banked trained orbit carrying a correctness label and writes
`results/correctness_decode.csv`. No GPU. OWNER: Data+Analysis. STATUS: implemented
2026-08-09.

THE GAP THIS CLOSES. D84 found correctness undecodable from the shape at 55.3%
balanced accuracy, p = 0.065, on 128 orbits from one bank. That is inconclusive
twice over: it is one bank, and nothing bounds it. Two things fix that here.

FIRST, THE FEATURES ARE CENTRED WITHIN FAMILY, and pooling then costs nothing. A
classifier told to predict `correct` from raw shape features can win by learning
WHICH FAMILY an orbit came from, since families differ enormously in accuracy —
0% to 100% across the 21 (D85) — and D84 showed family is decodable at 100%. That
would be family identity, and hence prompt length, wearing a correctness label.
Subtracting each family's own mean removes it exactly, and what remains is the
contrast D79 built its stratified permutation for: correct against incorrect, at
matched task. The null permutes labels WITHIN family for the same reason.

SECOND, A NULL WITHOUT A DETECTION FLOOR IS NOT A RESULT. `reliability.py` bounds a
CORRELATION by its measurement noise and `stratified.combinatorial_floor` bounds a
permutation test by its design; neither applies to a classifier. So the floor is
measured directly: plant a synthetic direction of known size into the correct
class's features and record how often the test fires. "Correctness is not decodable,
and this design detects a 0.5-sd shift in N% of trials" is a bounded null. "p = 0.065"
alone is not — that is D70's mistake in a different costume, a null read off an
instrument whose reach was never measured.

Run: uv run python -m scripts.run_correctness_decode
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_shape_decode import SOURCES, collect

OUT = "results/correctness_decode.csv"
MIN_PER_CLASS = 6
N_PERM = 400
EFFECTS = (0.0, 0.25, 0.5, 1.0, 2.0)
N_POWER = 60


def _clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.1))


def centre_within_family(df: pd.DataFrame, feature: str) -> np.ndarray:
    """Subtract each family's own mean feature vector.

    This is the whole design. Without it a classifier can score well by recognising
    the family and betting on its base rate, which D84 showed is available at 100%
    and D85 showed is collinear with prompt length. After it, only within-family
    variation survives — the same contrast D79's stratified permutation isolates.
    """
    x = np.stack(df[feature].to_numpy()).astype(np.float64)
    fams = df["family"].to_numpy()
    out = x.copy()
    for f in np.unique(fams):
        m = fams == f
        out[m] -= out[m].mean(axis=0)
    return out


def usable_families(df: pd.DataFrame, min_per_class: int = MIN_PER_CLASS) -> list[str]:
    """Families carrying at least `min_per_class` of BOTH classes.

    A family that is 100% correct or 0% correct contributes no within-family
    contrast at all, and including it would inflate n while adding no information —
    the same accounting `stratified_diff` applies when it drops single-class strata.
    """
    out = []
    for f, s in df.groupby("family"):
        n_pos = int(s["correct"].sum())
        if min(n_pos, len(s) - n_pos) >= min_per_class:
            out.append(str(f))
    return sorted(out)


def _perm_within_family(y: np.ndarray, fams: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    out = y.copy()
    for f in np.unique(fams):
        m = fams == f
        out[m] = rng.permutation(y[m])
    return out


def decode_correctness(df: pd.DataFrame, feature: str = "shape", n_splits: int = 5,
                       n_perm: int = N_PERM, seed: int = 0) -> dict:
    fams_ok = usable_families(df)
    sub = df[df["family"].isin(fams_ok)]
    if len(sub) < 40:
        return {"usable": False, "why": f"only {len(sub)} orbits in mixed families"}
    x = centre_within_family(sub, feature)
    y = sub["correct"].to_numpy().astype(int)
    fams = sub["family"].to_numpy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    obs = float(np.mean(cross_val_score(_clf(), x, y, cv=cv,
                                        scoring="balanced_accuracy")))
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean(cross_val_score(
        _clf(), x, _perm_within_family(y, fams, rng), cv=cv,
        scoring="balanced_accuracy"))) for _ in range(n_perm)])
    return {"usable": True, "feature": feature, "n": int(len(y)),
            "n_families": len(fams_ok), "families": ",".join(fams_ok),
            "n_correct": int(y.sum()), "balanced_accuracy": obs,
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
            "p_floor": 1.0 / (n_perm + 1)}


def detection_power(df: pd.DataFrame, feature: str = "shape",
                    effects: tuple[float, ...] = EFFECTS, n_trials: int = N_POWER,
                    alpha: float = 0.05, n_perm: int = 60, seed: int = 0) -> pd.DataFrame:
    """How large a shape difference would this design have caught?

    A synthetic direction of `d` within-family standard deviations is added to the
    correct class's features and the whole test re-run. The direction is RANDOM per
    trial rather than fixed, so the answer is not about one lucky axis; the effect
    is expressed in sd so it is comparable to D79's power curve, which reported 1.00
    at 1.5 within-stratum sd and 0.12-0.22 at 1.0.
    """
    fams_ok = usable_families(df)
    sub = df[df["family"].isin(fams_ok)]
    x0 = centre_within_family(sub, feature)
    y = sub["correct"].to_numpy().astype(int)
    fams = sub["family"].to_numpy()
    sd = x0.std(axis=0)
    rows = []
    for d in effects:
        rng = np.random.default_rng(seed + int(d * 1000))
        hits = 0
        for _ in range(n_trials):
            v = rng.normal(size=x0.shape[1])
            v /= np.linalg.norm(v)
            x = x0 + np.outer(y, v * sd * d)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            obs = float(np.mean(cross_val_score(_clf(), x, y, cv=cv,
                                                scoring="balanced_accuracy")))
            null = np.array([float(np.mean(cross_val_score(
                _clf(), x, _perm_within_family(y, fams, rng), cv=cv,
                scoring="balanced_accuracy"))) for _ in range(n_perm)])
            hits += ((np.sum(null >= obs) + 1) / (n_perm + 1)) < alpha
        rows.append({"effect_sd": d, "n_trials": n_trials,
                     "detected": hits / n_trials})
    return pd.DataFrame(rows)


def report(res: dict, power: pd.DataFrame, pos: dict | None = None) -> str:
    if not res.get("usable"):
        return f"\n  NOT USABLE: {res['why']}"
    lines = [
        "",
        f"  {res['n']} orbits across {res['n_families']} families carrying at least "
        f"{MIN_PER_CLASS} of both classes",
        f"  ({res['families']})",
        "",
        "  Features are centred WITHIN family, so the classifier cannot win by",
        "  recognising the family and betting on its base rate -- which D84 showed",
        "  is available at 100% and D85 showed is collinear with prompt length.",
        "",
        f"    shape -> correct: balanced accuracy {res['balanced_accuracy']:.1%} "
        f"against a within-family permutation null of {res['null_mean']:.1%} "
        f"(p95 {res['null_p95']:.1%}), p = {res['p']:.4f} "
        f"(floor {res['p_floor']:.4f})",
    ]
    if pos and pos.get("usable"):
        lines.append(f" position -> correct: {pos['balanced_accuracy']:.1%}, "
                     f"p = {pos['p']:.4f}")
    lines += ["", "  WHAT THIS DESIGN COULD HAVE SEEN (planted shifts, random direction)",
              f"    {'effect (sd)':>12} {'detected':>10}"]
    for _, r in power.iterrows():
        lines.append(f"    {r['effect_sd']:>12.2f} {r['detected']:>10.0%}")
    detectable = power[power["detected"] >= 0.8]["effect_sd"]
    floor = float(detectable.min()) if len(detectable) else float("nan")
    lines.append("")
    if np.isfinite(floor):
        lines += [
            f"  So the null rules out a within-family shape difference of "
            f"{floor:.2f} sd or larger,",
            "  and says nothing about anything smaller. That is the bound; the "
            "p-value alone is not.",
        ]
    else:
        lines += [
            "  NO planted effect up to the largest tested reached 80% detection, so",
            "  this design cannot support a null about correctness at all -- the same",
            "  situation `combinatorial_floor` catches for a permutation test (D79(2)),",
            "  where `attainable` passed while simulated power was 0.00.",
        ]
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    df = collect(SOURCES)
    if df.empty:
        print("no banked trained orbits found")
        return
    df = df[df["correct"].notna()]
    res = decode_correctness(df, "shape")
    pos = decode_correctness(df, "position")
    if not res.get("usable"):
        print(report(res, pd.DataFrame()))
        return
    power = detection_power(df, "shape")
    save_table(OUT, power, kind="correctness_decode", shape=res, position=pos,
               min_per_class=MIN_PER_CLASS)
    print(f"saved {OUT}")
    print(report(res, power, pos))


if __name__ == "__main__":
    main()
