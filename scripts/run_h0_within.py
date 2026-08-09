"""PRE-REGISTERED. Does the shape predict the answer within a prompt, and where?

Reads `scratch/kaggle_h0bank/out/` and writes `results/h0_within.csv`. No GPU.
OWNER: Data+Analysis. STATUS: **written 2026-08-09 BEFORE the data existed.**

WHY THE TIMING MATTERS AND IS STATED FIRST. D91 found correctness decodable from the
shape at unrolls 6-18 (68.4%, p = 0.010) and 12-24 (65.9%, p = 0.020) and nowhere
else across eight windows -- a lead that does not survive correction at n = 60, in
exactly the place arXiv:2607.20594 predicts ("the algorithm, if there is one, lives
in the head of the trajectory"; past tau* these statistics provably saturate). A
post-hoc window search cannot settle that. So the grid, the statistic, the null and
the decision rule are fixed here, in a file committed before `geometry-h0bank`
returned, and the run is 512 orbits from 16 boundary prompts x 32 unseeded h_0 draws
-- a within-ITEM contrast, 8.5x the usable n of D91.

THE DESIGN IS THE ONE THE CONTRADICTING LITERATURE USES. Published results reporting
correctness decodable from trajectory geometry obtain within-prompt variance from
stochastic ROLLOUTS. Huginn's decode is deterministic, but D90 showed h_0 supplies the
same handle: on a fixed prompt, gold rank varies across h_0 in 6 of 8 prompts and
correctness splits in 3 of 8, while seeding h_0 gives bit-identical states. So the
contrast here is: same prompt, same weights, same depth, same read position, and the
only thing that differs is the initial latent.

PRE-REGISTERED DECISIONS -- none of these may change after seeing the data.

P1  GATE. At least 8 of the 16 prompts must split on correctness (both classes among
    their 32 draws). Below that the binary arm is not analysed and the row says so.
P2  GRID. Windows start at unrolls 0, 6, 12, 18, 24, 30, 36, 44 with width 12 --
    IDENTICAL to D91's grid. No window may be added or dropped after the fact.
P3  STATISTIC. Balanced accuracy of a logistic classifier on the Gram shape code,
    features centred WITHIN PROMPT, 5-fold stratified CV.
P4  NULL. Labels permuted WITHIN PROMPT, 400 draws. Reported p is (hits+1)/(n+1).
P5  THRESHOLD. Bonferroni over the 8 windows: alpha = 0.05/8 = 0.00625. A window is
    called only if it clears that. **The lead is confirmed only if a window in
    6-24 clears alpha AND the far-tail windows (>= 30) do not.**
P6  TIMING CONTROL, the alternative D91(5) names. `best_depth` varies 7-25 within one
    prompt (D90), so correct and incorrect orbits may differ in WHEN they peak rather
    than in shape. The same decode is run with `best_depth` added as a covariate and
    on `best_depth` alone; if `best_depth` alone decodes correctness as well as the
    shape does, the shape result is timing and is reported as such.
P7  COMPARISON ARM. Position features at the same count, as in D84.
P8  DETECTION FLOOR. Planted effects on within-prompt-permuted labels, so a null
    carries the size it could have seen (D70's lesson).

Run: uv run python -m scripts.run_h0_within
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from traj_geom.metrics.shape_code import gram_code, position_code

BANK = os.path.join("scratch", "kaggle_h0bank", "out")
OUT = os.path.join("results", "h0_within.csv")
STARTS = (0, 6, 12, 18, 24, 30, 36, 44)     # P2, identical to D91's grid
WIDTH = 12
N_PERM = 400                                 # P4
ALPHA = 0.05 / len(STARTS)                   # P5
MIN_SPLIT = 8                                # P1
EFFECTS = (0.0, 0.25, 0.5, 1.0)              # P8


def collect(path: str = BANK) -> pd.DataFrame:
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fh:
        recs = [r for r in json.load(fh) if r.get("ok")]
    rows = []
    for r in recs:
        f = os.path.join(path, r["tag"] + ".npy")
        if not os.path.exists(f):
            continue
        traj = np.load(f).astype(np.float64)
        # The prompt key, NOT the family: several prompts come from one family and
        # the contrast is within ITEM.
        prompt = f"{r['family']}_i{r['item']:02d}"
        row = {"tag": r["tag"], "block": r["block"], "prompt": prompt,
               "correct": bool(r["correct"]), "best_rank": r["best_rank"],
               "best_depth": r["best_depth"], "state_sha": r.get("state_sha")}
        ok = True
        for s in STARTS:
            g = gram_code(traj, lo=s, m=WIDTH)
            if g.size == 0:
                ok = False
                break
            row[f"shape_{s}"] = g
            row[f"position_{s}"] = position_code(traj, lo=s, m=WIDTH, n_proj=len(g))
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)


def gate(df: pd.DataFrame) -> dict:
    """P1, plus the determinism control that makes the contrast attributable."""
    rep = df[df["block"] == "rep"]
    fix = df[df["block"] == "fix"]
    per = rep.groupby("prompt")["correct"].agg(["sum", "size"])
    split = per[(per["sum"] > 0) & (per["sum"] < per["size"])]
    fix_sha = fix.groupby("prompt")["state_sha"].nunique()
    rep_sha = rep.groupby("prompt")["state_sha"].nunique()
    rep_n = rep.groupby("prompt").size()
    return {"n_prompts": int(len(per)), "n_split": int(len(split)),
            "passes": bool(len(split) >= MIN_SPLIT),
            "split_prompts": list(split.index),
            "n_orbits_in_split": int(rep[rep["prompt"].isin(split.index)].shape[0]),
            "fix_all_identical": bool((fix_sha == 1).all()) if len(fix_sha) else False,
            "rep_all_distinct": bool((rep_sha == rep_n).all()) if len(rep_sha) else False}


def _centre(x: np.ndarray, keys: np.ndarray) -> np.ndarray:
    out = x.copy()
    for k in np.unique(keys):
        m = keys == k
        out[m] -= out[m].mean(axis=0)
    return out


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))


def _perm(y: np.ndarray, keys: np.ndarray, rng) -> np.ndarray:
    out = y.copy()
    for k in np.unique(keys):
        m = keys == k
        out[m] = rng.permutation(y[m])
    return out


def decode(df: pd.DataFrame, col: str, extra: np.ndarray | None = None,
           n_perm: int = N_PERM, seed: int = 0) -> dict:
    """P3/P4: within-prompt centred features, within-prompt permutation null."""
    y = df["correct"].to_numpy().astype(int)
    keys = df["prompt"].to_numpy()
    if len(np.unique(y)) < 2 or len(y) < 20:
        return {"usable": False, "why": f"{len(y)} orbits, {len(np.unique(y))} classes"}
    x = _centre(np.stack(df[col].to_numpy()).astype(np.float64), keys)
    if extra is not None:
        x = np.hstack([x, _centre(extra.astype(np.float64), keys)])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    obs = float(np.mean(cross_val_score(_clf(), x, y, cv=cv,
                                        scoring="balanced_accuracy")))
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean(cross_val_score(
        _clf(), x, _perm(y, keys, rng), cv=cv, scoring="balanced_accuracy")))
        for _ in range(n_perm)])
    return {"usable": True, "n": int(len(y)), "balanced_accuracy": obs,
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
            "sig": bool((np.sum(null >= obs) + 1) / (n_perm + 1) < ALPHA)}


def detection_floor(df: pd.DataFrame, col: str, effects=EFFECTS, n_trials: int = 20,
                    n_perm: int = 40, seed: int = 0) -> pd.DataFrame:
    """P8. Planted on WITHIN-PROMPT-PERMUTED labels, so the curve is the design's
    reach and not the signal already present -- the correction D85's first draft
    needed."""
    keys = df["prompt"].to_numpy()
    x0 = _centre(np.stack(df[col].to_numpy()).astype(np.float64), keys)
    y = df["correct"].to_numpy().astype(int)
    sd = x0.std(axis=0)
    rows = []
    for d in effects:
        rng = np.random.default_rng(seed + int(d * 1000))
        hits = 0
        for _ in range(n_trials):
            v = rng.normal(size=x0.shape[1])
            v /= np.linalg.norm(v)
            y_null = _perm(y, keys, rng)
            x = x0 + np.outer(y_null, v * sd * d)
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            obs = float(np.mean(cross_val_score(_clf(), x, y_null, cv=cv,
                                                scoring="balanced_accuracy")))
            null = np.array([float(np.mean(cross_val_score(
                _clf(), x, _perm(y_null, keys, rng), cv=cv,
                scoring="balanced_accuracy"))) for _ in range(n_perm)])
            hits += ((np.sum(null >= obs) + 1) / (n_perm + 1)) < ALPHA
        rows.append({"effect_sd": d, "detected": hits / n_trials})
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame, g: dict) -> pd.DataFrame:
    rep = df[(df["block"] == "rep") & df["prompt"].isin(g["split_prompts"])]
    depth = rep[["best_depth"]].to_numpy()
    rows = []
    for s in STARTS:
        for feat in ("shape", "position"):
            r = decode(rep, f"{feat}_{s}")
            rows.append({"start": s, "feature": feat, "arm": "plain", **r})
        # P6: the timing control.
        r = decode(rep, f"shape_{s}", extra=depth)
        rows.append({"start": s, "feature": "shape", "arm": "+best_depth", **r})
    return pd.DataFrame(rows)


def timing_only(df: pd.DataFrame, g: dict, n_perm: int = N_PERM) -> dict:
    """P6: does `best_depth` ALONE decode correctness? If so the shape result is
    timing."""
    rep = df[(df["block"] == "rep") & df["prompt"].isin(g["split_prompts"])].copy()
    rep["only"] = list(rep[["best_depth"]].to_numpy().astype(float))
    return decode(rep, "only", n_perm=n_perm)


def report(g: dict, res: pd.DataFrame, timing: dict, floor: pd.DataFrame) -> str:
    lines = ["", "  P1 GATE", f"    {g['n_split']}/{g['n_prompts']} prompts split on "
             f"correctness (wants >= {MIN_SPLIT}) -> "
             f"{'PASS' if g['passes'] else 'FAIL, binary arm not analysed'}",
             f"    {g['n_orbits_in_split']} orbits in the splitting prompts",
             f"    determinism: seeded identical {g['fix_all_identical']}, "
             f"unseeded all distinct {g['rep_all_distinct']}"]
    if not g["passes"]:
        return "\n".join(lines)
    lines += ["", f"  Bonferroni alpha over {len(STARTS)} pre-registered windows "
                  f"= {ALPHA:.5f}", "",
              f"  {'window':>14} {'arm':>12} {'feature':>9} {'bal acc':>8} "
              f"{'null':>7} {'p':>8} {'sig':>5}"]
    for _, r in res.iterrows():
        if not r.get("usable", True):
            continue
        lines.append(f"  unrolls {int(r['start']):>2}-{int(r['start'])+WIDTH:<3} "
                     f"{r['arm']:>12} {r['feature']:>9} {r['balanced_accuracy']:>8.1%} "
                     f"{r['null_mean']:>7.1%} {r['p']:>8.4f} "
                     f"{'YES' if r['sig'] else '':>5}")
    ok = res[res.get("usable", True).astype(bool) & (res["arm"] == "plain")
             & (res["feature"] == "shape")]
    head = ok[(ok["start"] >= 6) & (ok["start"] <= 12)]
    tail = ok[ok["start"] >= 30]
    lines += ["", "  P5 DECISION RULE (fixed before the data existed)"]
    if len(head) and head["sig"].any() and len(tail) and not tail["sig"].any():
        lines.append("    CONFIRMED: a window in the 6-24 transient clears alpha and "
                     "no far-tail window does.")
        lines.append("    D91's lead replicates, and the correctness nulls (D79, D84, "
                     "D85) narrow to the CONVERGED region.")
    elif len(ok) and ok["sig"].any():
        lines.append("    PARTIAL: some window clears alpha but the head/tail pattern "
                     "is not the pre-registered one. Reported, not interpreted.")
    else:
        lines.append("    NOT CONFIRMED: no window clears alpha. D91 was a lead that "
                     "failed, and D84/D85 stand unqualified.")
    if timing.get("usable"):
        lines += ["", "  P6 TIMING CONTROL",
                  f"    best_depth ALONE decodes correctness at "
                  f"{timing['balanced_accuracy']:.1%} (p = {timing['p']:.4f})."]
        best = ok["balanced_accuracy"].max() if len(ok) else float("nan")
        if timing["balanced_accuracy"] >= best - 0.02:
            lines.append("    That matches the best shape window, so the shape result "
                         "is TIMING and must be reported as such.")
    if len(floor):
        lines += ["", "  P8 DETECTION FLOOR (planted on permuted labels)"]
        for _, r in floor.iterrows():
            lines.append(f"    {r['effect_sd']:>5.2f} sd -> detected "
                         f"{r['detected']:.0%}")
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not os.path.exists(os.path.join(BANK, "manifest.json")):
        print(f"no banked states in {BANK} -- pull the geometry-h0bank output first.")
        return
    df = collect()
    g = gate(df)
    res = analyse(df, g) if g["passes"] else pd.DataFrame()
    timing = timing_only(df, g) if g["passes"] else {}
    rep = df[(df["block"] == "rep") & df["prompt"].isin(g["split_prompts"])]
    # n_perm must be large enough that ALPHA is reachable at all: the minimum
    # p-value with k permutations is 1/(k+1), and 1/41 = 0.024 > ALPHA = 0.00625,
    # so the floor's own default (n_perm=40) can NEVER register a detection
    # regardless of effect size -- caught 2026-08-09 when 1.0 sd read 0% detected.
    # Match N_PERM so the floor's null is on the same footing as the main test's.
    floor = detection_floor(rep, "shape_12", n_perm=N_PERM) if g["passes"] else pd.DataFrame()
    if len(res):
        save_table(OUT, res, kind="h0_within", gate=g, timing=timing,
                   floor=floor.to_dict("records"), starts=list(STARTS), width=WIDTH)
        print(f"saved {OUT}")
    print(report(g, res, timing, floor))


if __name__ == "__main__":
    main()
