"""Is correctness decodable from the orbit's SHAPE — and what could this design see?

Reads every banked trained orbit carrying a correctness label and writes
`results/correctness_decode.csv`. No GPU. OWNER: Data+Analysis. STATUS: implemented
2026-08-09.

THE GAP THIS CLOSES. D84 found correctness undecodable from the shape at 55.3%
balanced accuracy, p = 0.065, on 128 orbits from one bank. That is inconclusive
twice over: it is one bank, and nothing bounds it. Two things fix that here.

STRATIFY BY (FAMILY, GOLD), NOT BY FAMILY, and the reason is written in this
project's own history. A first version centred within FAMILY only, and reported 59.6%
balanced accuracy at p = 0.0025 -- carried by `sub1` at 86.2%. That is D72 exactly:
in `sub1` only 4 of 24 items sit in a gold value carrying both classes, so correctness
is very nearly a deterministic function of the ANSWER, the classifier decodes the
PROMPT (which D84 shows it can do at 100%), and reads correctness off it. Family
centring removes the family; it does not remove the item. Conditioning on the gold
value does, and it is precisely the design D79 built its stratified permutation
around.

FIRST, THE FEATURES ARE CENTRED WITHIN STRATUM, and pooling then costs nothing. A
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_shape_decode import SOURCES, collect

OUT = "results/correctness_decode.csv"
MIN_PER_CLASS = 6
N_PERM = 400
EFFECTS = (0.0, 0.25, 0.5, 1.0, 2.0)
# 20 trials x 40 permutations per point. A power CURVE needs its shape, not three
# decimal places: the first draft used 60 x 60, which is ~90k cross-validated fits
# on 272 x 190 features and ran for over half an hour to sharpen numbers that are
# read as "about 20%" or "about 100%".
N_POWER = 20
N_POWER_PERM = 40


class FamilyCentre(BaseEstimator, TransformerMixin):
    """Subtract each family's mean, LEARNED ON THE TRAINING FOLD ONLY.

    Centring on the full dataset before cross-validation leaks: a test orbit's
    centred value then depends on statistics computed from the test fold itself.
    The effect is mild for a mean, but it is exactly the kind of thing that turns a
    borderline p into a significant one, and this analysis is borderline. Families
    unseen in a training fold fall back to the global mean rather than raising --
    with 6-24 orbits per family a fold can legitimately miss one.

    `family` is passed as the LAST column of X and stripped here, because
    scikit-learn's CV splits X and y and has nowhere else to put a group label.
    """

    def fit(self, x, y=None):
        feats, fams = x[:, :-1].astype(np.float64), x[:, -1]
        self.global_ = feats.mean(axis=0)
        self.means_ = {f: feats[fams == f].mean(axis=0) for f in np.unique(fams)}
        return self

    def transform(self, x):
        feats, fams = x[:, :-1].astype(np.float64), x[:, -1]
        out = feats.copy()
        for i, f in enumerate(fams):
            out[i] -= self.means_.get(f, self.global_)
        return out


def _clf():
    return make_pipeline(FamilyCentre(), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.1))


def _xy(df: pd.DataFrame, feature: str):
    """Features with the family code appended as a final column for `FamilyCentre`."""
    d = with_strata(df)
    x = np.stack(d[feature].to_numpy()).astype(np.float64)
    fams = d["stratum"].to_numpy()
    codes = {f: i for i, f in enumerate(sorted(set(fams)))}
    return (np.hstack([x, np.array([codes[f] for f in fams], dtype=float)[:, None]]),
            d["correct"].to_numpy().astype(int), fams)


def centre_within_family(df: pd.DataFrame, feature: str) -> np.ndarray:
    """Subtract each family's own mean feature vector.

    This is the whole design. Without it a classifier can score well by recognising
    the family and betting on its base rate, which D84 showed is available at 100%
    and D85 showed is collinear with prompt length. After it, only within-family
    variation survives — the same contrast D79's stratified permutation isolates.
    """
    d = with_strata(df)
    x = np.stack(d[feature].to_numpy()).astype(np.float64)
    fams = d["stratum"].to_numpy()
    out = x.copy()
    for f in np.unique(fams):
        m = fams == f
        out[m] -= out[m].mean(axis=0)
    return out


def with_strata(df: pd.DataFrame, by_gold: bool = True) -> pd.DataFrame:
    """Add the stratum key: (family, gold) when `by_gold`, else family alone."""
    out = df.copy()
    if by_gold and "gold" in out.columns:
        out["stratum"] = out["family"].astype(str) + "|" + out["gold"].astype(str)
    else:
        out["stratum"] = out["family"].astype(str)
    return out


def usable_families(df: pd.DataFrame, min_per_class: int = MIN_PER_CLASS,
                    by_gold: bool = True) -> list[str]:
    """Strata carrying at least `min_per_class` of BOTH classes.

    A stratum that is 100% correct or 0% correct contributes no contrast at all, and
    including it would inflate n while adding no information — the same accounting
    `stratified_diff` applies when it drops single-class strata. With `by_gold` the
    stratum is (family, gold), which is what stops correctness from being read off
    the answer value (D72).
    """
    d = with_strata(df, by_gold)
    out = []
    for f, s in d.groupby("stratum"):
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
    sub = with_strata(df)
    sub = sub[sub["stratum"].isin(fams_ok)]
    if len(sub) < 40:
        return {"usable": False, "why": f"only {len(sub)} orbits in mixed strata"}
    x, y, fams = _xy(sub, feature)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    obs = float(np.mean(cross_val_score(_clf(), x, y, cv=cv,
                                        scoring="balanced_accuracy")))
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean(cross_val_score(
        _clf(), x, _perm_within_family(y, fams, rng), cv=cv,
        scoring="balanced_accuracy"))) for _ in range(n_perm)])
    return {"usable": True, "feature": feature, "n": int(len(y)),
            "n_families": len(fams_ok), "families": ",".join(fams_ok[:12]),
            "n_correct": int(y.sum()), "balanced_accuracy": obs,
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
            "p_floor": 1.0 / (n_perm + 1)}


def detection_power(df: pd.DataFrame, feature: str = "shape",
                    effects: tuple[float, ...] = EFFECTS, n_trials: int = N_POWER,
                    alpha: float = 0.05, n_perm: int = N_POWER_PERM,
                    seed: int = 0) -> pd.DataFrame:
    """How large a shape difference would this design have caught?

    A synthetic direction of `d` within-family standard deviations is added to the
    correct class's features and the whole test re-run. The direction is RANDOM per
    trial rather than fixed, so the answer is not about one lucky axis; the effect
    is expressed in sd so it is comparable to D79's power curve, which reported 1.00
    at 1.5 within-stratum sd and 0.12-0.22 at 1.0.
    """
    fams_ok = usable_families(df)
    sub = with_strata(df)
    sub = sub[sub["stratum"].isin(fams_ok)]
    x0, y, fams = _xy(sub, feature)
    feats = x0[:, :-1]
    sd = feats.std(axis=0)
    rows = []
    for d in effects:
        rng = np.random.default_rng(seed + int(d * 1000))
        hits = 0
        for _ in range(n_trials):
            v = rng.normal(size=feats.shape[1])
            v /= np.linalg.norm(v)
            # PLANT ON NULL DATA, not on the real data. The first version added the
            # effect to the observed features and reported 100% detection at an
            # effect of ZERO -- correctly, because the real data already carries a
            # signal, so the "power curve" was measuring that signal at every point
            # instead of the design's reach. Permuting the labels within family
            # first destroys any real correct/incorrect structure while preserving
            # the family structure and the marginal distributions, so what is
            # detected afterwards is only what was planted.
            y_null = _perm_within_family(y, fams, rng)
            x = x0.copy()
            x[:, :-1] = feats + np.outer(y_null, v * sd * d)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            obs = float(np.mean(cross_val_score(_clf(), x, y_null, cv=cv,
                                                scoring="balanced_accuracy")))
            null = np.array([float(np.mean(cross_val_score(
                _clf(), x, _perm_within_family(y_null, fams, rng), cv=cv,
                scoring="balanced_accuracy"))) for _ in range(n_perm)])
            hits += ((np.sum(null >= obs) + 1) / (n_perm + 1)) < alpha
        rows.append({"effect_sd": d, "n_trials": n_trials,
                     "detected": hits / n_trials})
    return pd.DataFrame(rows)


def per_family(df: pd.DataFrame, feature: str = "shape", n_perm: int = 200,
               seed: int = 0) -> pd.DataFrame:
    """The same test one family at a time.

    A pooled hit carried by a single family is a fact about that family, and with
    8 families of 24 the pooled n is large enough to hide that. Reported whatever
    the pooled result says.
    """
    rows = []
    d = with_strata(df)
    for fam in sorted({s.split("|")[0] for s in usable_families(df)}):
        keep = set(usable_families(df))
        sub = d[(d["family"] == fam) & (d["stratum"].isin(keep))]
        y = sub["correct"].to_numpy().astype(int)
        if min(int(y.sum()), len(y) - int(y.sum())) < 5:
            continue
        x = np.stack(sub[feature].to_numpy()).astype(np.float64)
        x = x - x.mean(axis=0)
        code = np.zeros((len(x), 1))
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        obs = float(np.mean(cross_val_score(_clf(), np.hstack([x, code]), y, cv=cv,
                                            scoring="balanced_accuracy")))
        rng = np.random.default_rng(seed)
        null = np.array([float(np.mean(cross_val_score(
            _clf(), np.hstack([x, code]), rng.permutation(y), cv=cv,
            scoring="balanced_accuracy"))) for _ in range(n_perm)])
        rows.append({"family": fam, "n": len(y), "n_correct": int(y.sum()),
                     "balanced_accuracy": obs, "null_mean": float(null.mean()),
                     "p": float((np.sum(null >= obs) + 1) / (n_perm + 1))})
    return pd.DataFrame(rows)


def report(res: dict, power: pd.DataFrame, pos: dict | None = None,
           byfam: pd.DataFrame | None = None) -> str:
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
    if byfam is not None and len(byfam):
        lines += ["", "  ONE FAMILY AT A TIME (a pooled hit carried by one family is",
                  "  a fact about that family, and n = 320 can hide that)",
                  f"    {'family':>14} {'n':>4} {'+ve':>4} {'bal acc':>8} "
                  f"{'null':>7} {'p':>8}"]
        for _, r in byfam.iterrows():
            lines.append(f"    {r['family']:>14} {int(r['n']):>4} "
                         f"{int(r['n_correct']):>4} {r['balanced_accuracy']:>8.1%} "
                         f"{r['null_mean']:>7.1%} {r['p']:>8.4f}")
        above = int((byfam["balanced_accuracy"] > byfam["null_mean"]).sum())
        lines.append(f"    {above}/{len(byfam)} families above their own null; "
                     f"{int((byfam['p'] < 0.05).sum())} individually significant")

    lines += ["", "  WHAT THIS DESIGN COULD HAVE SEEN",
              "  (effect planted on WITHIN-FAMILY-PERMUTED labels, so the curve",
              "   measures the design's reach and not the signal already present)",
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
    byfam = per_family(df, "shape")
    save_table(OUT, power, kind="correctness_decode", shape=res, position=pos,
               per_family=byfam.to_dict("records"), min_per_class=MIN_PER_CLASS)
    print(f"saved {OUT}")
    print(report(res, power, pos, byfam))


if __name__ == "__main__":
    main()
