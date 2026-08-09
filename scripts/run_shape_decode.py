"""Can ANY function of the orbit's SHAPE decode the task, or only its position?

Reads the banked trained orbits and writes `results/shape_decode.csv`. No GPU.
OWNER: Data+Analysis. STATUS: implemented 2026-08-09.

THE OBJECTION THIS ANSWERS. D79 and D80 test four hand-picked statistics and find
none tracks the computation; D79(5) states the limit itself -- "a geometric statistic
nobody has thought of could still differ". Handing a classifier the shape and
cross-validating replaces the hand-picking with a search over every function the
classifier can express, which is the strongest form of the question available
without a new GPU run.

THE FEATURE IS ROTATION- AND TRANSLATION-INVARIANT BY CONSTRUCTION, and that is the
whole design. Huginn re-injects the prompt embeddings at every unroll (D70(4)), so
the raw states carry the token bag by architecture -- D73 measured RANDOM weights
decoding a count from them at R2 = 0.99999. A classifier fed raw states decodes the
family trivially and says nothing about geometry. The Gram matrix of unit step
directions removes exactly the position information while keeping the path's shape
in full.

AND THE RAW STATES ARE RUN ANYWAY, AS THE POSITIVE CONTROL. A null on shape means
something only if the same classifier, at the same feature count, on the same
orbits, succeeds on position. Without that arm a null is a statement about the
classifier. This is D70's lesson -- a gap of ~0 read off a probe that could not have
moved -- in the form the present question needs.

THE NULL IS A LABEL PERMUTATION, not an assumed chance level. With 4-21 classes and
few orbits per class, the accuracy of a trivial classifier is not 1/n_classes, and
`geometry-correctness` was drafted once against a floor it could not reach (D72).

Run: uv run python -m scripts.run_shape_decode
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

from traj_geom.metrics.dmd import pre_floor_window
from traj_geom.metrics.shape_code import gram_code, is_rotation_invariant, position_code

OUT = os.path.join("results", "shape_decode.csv")
M = 20                 # steps per code; D80 puts this inside the live transient
N_PERM = 200
SOURCES = (
    ("b6bank", os.path.join("scratch", "kaggle_b6bank", "out"), ".npy"),
    ("geomcap", os.path.join("scratch", "kaggle_geomcap", "out"), ".npy"),
    ("ds_bank", os.path.join("scratch", "ds_bank", "out"), "_last.npy"),
    # The only bank where task identity is NOT collinear with prompt length: both
    # forms of every item are byte-identical apart from a trailing marker token,
    # gate-verified per item to tokenise to the same count. This is the arm that
    # decides whether D84's family decoding was about the task or about the length.
    ("lenmatch", os.path.join("scratch", "kaggle_lenmatch", "out"), ".npy"),
)


def collect(sources=SOURCES, m: int = M) -> pd.DataFrame:
    rows, skipped = [], []
    for bank, path, suf in sources:
        mp = os.path.join(path, "manifest.json")
        if not os.path.exists(mp):
            continue
        with open(mp, encoding="utf-8") as fh:
            recs = [r for r in json.load(fh) if r.get("ok")]
        for r in recs:
            if r.get("arm") not in (None, "trained"):
                continue
            if r.get("block") not in (None, "main"):
                continue          # geomcap's replicate blocks are not a task sample
            f = os.path.join(path, r["tag"] + suf)
            if not os.path.exists(f):
                continue
            try:
                traj = np.load(f).astype(np.float64)
            except (ValueError, OSError) as exc:
                # A truncated download is not a reason to abandon 800 orbits. One
                # 18-byte file survived a `kaggle kernels output` pull and took the
                # whole analysis down with a pickle error; skipped and counted here,
                # and the count is printed so a silent shortfall cannot pass as data.
                skipped.append((f, type(exc).__name__))
                continue
            lo, _ = pre_floor_window(traj)
            g = gram_code(traj, lo=lo, m=m)
            p = position_code(traj, lo=lo, m=m, n_proj=len(g))
            if g.size == 0 or p.size == 0:
                continue
            rows.append({"bank": bank, "tag": r["tag"], "_group": bank,
                         "family": r.get("family") or r.get("task", "?"),
                         # The ANSWER VALUE, so correctness can be conditioned on it.
                         # D72 voided a whole run because correctness was a function
                         # of the gold, and any analysis that pools across golds
                         # inherits that confound.
                         "gold": str(r.get("gold", "")),
                         "correct": bool(r.get("correct",
                                               r.get("correct_any_depth", False))),
                         "n_tokens": int(r.get("n_tokens", 0)),
                         "shape": g, "position": p})
    if skipped:
        print(f"  skipped {len(skipped)} unreadable .npy file(s): "
              + ", ".join(os.path.basename(f) for f, _ in skipped[:4]))
    df = pd.DataFrame(rows)
    if len(df):
        # Prompt length, as a THREE-LEVEL target. It is the confound that most
        # plausibly explains a "shape decodes the family" result: families differ
        # systematically in how many tokens they occupy, and sequence length changes
        # the orbit by itself. Binned within bank, because the banks' length
        # distributions differ and a pooled binning would re-encode the bank.
        df["len_bin"] = (df.groupby("bank")["n_tokens"]
                         .transform(lambda v: pd.qcut(v, 3, labels=False,
                                                      duplicates="drop"))
                         .astype("Int64").astype(str))
    return df


def _clf():
    # Strong L2 and a scaler: with ~190 features and ~130 orbits the fit is
    # overdetermined only by regularisation, and an unpenalised logistic model
    # separates any such design perfectly IN SAMPLE. Everything reported is
    # cross-validated.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.1))


def decode(df: pd.DataFrame, feature: str, target: str, n_splits: int = 5,
           n_perm: int = N_PERM, seed: int = 0, scoring: str = "balanced_accuracy",
           group: str | None = None) -> dict:
    """Cross-validated score with a label-permutation null.

    BALANCED accuracy by default, and the reason is a defect this analysis shipped
    with for one run. On `correct` the majority class is 75.0%; plain accuracy gave
    76.9% against a permutation null of 69.3% and p = 0.005, which reads as a real
    effect. It is not: the classifier was scoring at the majority baseline, and the
    permutation null sat BELOW that baseline because permuting labels degrades even
    a majority predictor under stratified CV. A null the trivial predictor beats is
    not a null. Balanced accuracy puts the trivial predictor at 0.5 for any class
    balance, so the comparison means what it looks like.

    `group` restricts to one value of a column -- used to ask the family question
    WITHIN a bank, since the banks differ in depth, run and task set, and "which
    bank" is decodable from the depth profile alone (D80).
    """
    if group is not None:
        df = df[df["_group"] == group]
    if len(df) < 4 * n_splits:
        return {"usable": False, "why": f"only {len(df)} orbits"}
    x = np.stack(df[feature].to_numpy())
    y = df[target].to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < n_splits:
        return {"usable": False,
                "why": f"{len(classes)} classes, smallest has {counts.min()}"}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    obs = float(np.mean(cross_val_score(_clf(), x, y, cv=cv, scoring=scoring)))
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean(cross_val_score(
        _clf(), x, rng.permutation(y), cv=cv, scoring=scoring)))
        for _ in range(n_perm)])
    return {"usable": True, "feature": feature, "target": target,
            "group": group or "all", "scoring": scoring,
            "n": int(len(y)), "n_classes": int(len(classes)),
            "majority": float(counts.max() / counts.sum()),
            "trivial": 1.0 / len(classes),
            "accuracy": obs, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
            "p_floor": 1.0 / (n_perm + 1)}


def length_collinearity(df: pd.DataFrame) -> dict:
    """Do any two families share a prompt length? If not, the confound is TOTAL.

    Not a caveat to mention -- a property that decides whether a "shape decodes the
    task" result can be attributed to the task at all. D74(6) records the same shape
    of problem, where `partial_spearman` REFUSED to correct for window length at
    rho = -1.000 because the confounder was rank-collinear with the predictor. If
    family and length are perfectly collinear here, no statistical control can
    separate them and the answer needs a different experiment, not a better model.
    """
    out = {}
    for bank, s in df.groupby("bank"):
        lens = {f: set(v) for f, v in s.groupby("family")["n_tokens"]}
        fams = sorted(lens)
        shared = [(a, b, sorted(lens[a] & lens[b]))
                  for i, a in enumerate(fams) for b in fams[i + 1:]
                  if lens[a] & lens[b]]
        out[bank] = {"n_families": len(fams), "n_overlapping_pairs": len(shared),
                     "n_pairs": len(fams) * (len(fams) - 1) // 2,
                     "examples": shared[:3],
                     "lengths": {f: sorted(v)[:4] for f, v in lens.items()}}
    return out


def report(df: pd.DataFrame, res: pd.DataFrame, rot: float) -> str:
    lines = [
        "",
        f"  {len(df)} trained orbits, {df['family'].nunique()} families, "
        f"{len(df.iloc[0]['shape'])} features per code, window {M} steps",
        f"  rotation invariance of the shape code: max|delta| = {rot:.3e} "
        + ("(exact)" if rot < 1e-9 else "<- NOT invariant; the code encodes the basis"),
        "",
        "  Balanced accuracy, so the trivial predictor scores 0.5 whatever the",
        "  class balance. Plain accuracy against a permutation null is NOT enough:",
        "  on `correct` (75% majority) it gave 76.9% against a null of 69.3% and",
        "  p = 0.005, which is a majority predictor beating a degraded one.",
        "",
        f"  {'feature':>9} {'group':>9} {'target':>9} {'n':>4} {'cls':>4} "
        f"{'bal acc':>8} {'null':>7} {'null p95':>9} {'p':>8}",
    ]
    for _, r in res.iterrows():
        if not r.get("usable", True):
            lines.append(f"  {r['feature']:>9} {str(r.get('group')):>9} "
                         f"{r['target']:>9}   REFUSED: {r['why']}")
            continue
        lines.append(f"  {r['feature']:>9} {r['group']:>9} {r['target']:>9} "
                     f"{int(r['n']):>4} {int(r['n_classes']):>4} "
                     f"{r['accuracy']:>8.1%} {r['null_mean']:>7.1%} "
                     f"{r['null_p95']:>9.1%} {r['p']:>8.4f}")

    live = res[res["usable"].astype(bool)] if "usable" in res else res

    def pick(feature, target, group):
        m = live[(live["feature"] == feature) & (live["target"] == target)
                 & (live["group"] == group)]
        return m.iloc[0] if len(m) else None

    coll = length_collinearity(df)
    lines += ["", "  IS FAMILY SEPARABLE FROM PROMPT LENGTH AT ALL?"]
    total = True
    for bank, c in coll.items():
        lines.append(f"    {bank}: {c['n_overlapping_pairs']}/{c['n_pairs']} family "
                     f"pairs share any prompt length; per-family lengths "
                     + ", ".join(f"{f}={v[0] if len(v) == 1 else v}"
                                 for f, v in sorted(c["lengths"].items())))
        total &= c["n_overlapping_pairs"] == 0
    if total:
        lines += [
            "    NO family pair shares a token count anywhere in the banked data, so",
            "    task identity and prompt length are PERFECTLY COLLINEAR and no",
            "    statistical control can separate them -- the same shape of problem",
            "    D74(6) records, where partial_spearman refused at rho = -1.000.",
            "    `track`/`local` do not rescue it either: byte-identical bodies, but",
            "    the appended question costs 3 tokens and their counts are disjoint",
            "    (22/34/58/106 against 25/37/61/109), so length separates them",
            "    perfectly on its own. Answering this needs a task pair built to",
            "    tokenise to the SAME length, which is a new run and not a new model.",
        ]

    lines += ["", "  READING IT."]
    pos = pick("position", "family", "all")
    if pos is None or pos["p"] > 0.05:
        lines += [
            "  THE POSITIVE CONTROL FAILED. The raw states carry the prompt by",
            "  architecture (D70(4)), so a classifier that cannot decode the family",
            "  from them has a problem the shape arm cannot be read against.",
        ]
        return "\n".join(lines)

    banks = sorted(g for g in live["group"].unique() if g != "all")
    for g in banks:
        fam = pick("shape", "family", g)
        length = pick("shape", "len_bin", g)
        corr = pick("shape", "correct", g)
        if fam is None:
            continue
        verdict = ("DECODABLE" if fam["p"] < 0.05 else "not decodable")
        lines.append(f"    within {g}: family {verdict} from shape at "
                     f"{fam['accuracy']:.1%} balanced (p = {fam['p']:.4f}, "
                     f"{int(fam['n_classes'])} classes, n = {int(fam['n'])})")
        if length is not None:
            lines.append(f"      prompt-length bin from shape: "
                         f"{length['accuracy']:.1%} (p = {length['p']:.4f})"
                         + ("   <- the confound: families differ in length, and "
                            "length alone changes the orbit"
                            if length["p"] < 0.05 else ""))
        if corr is not None:
            lines.append(f"      correctness from shape: {corr['accuracy']:.1%} "
                         f"balanced (p = {corr['p']:.4f})")

    within = [pick("shape", "family", g) for g in banks]
    within = [w for w in within if w is not None]
    lens = [pick("shape", "len_bin", g) for g in banks]
    lens = [x for x in lens if x is not None]
    lines.append("")
    if within and all(w["p"] < 0.05 for w in within):
        lines += [
            "  The task family IS decodable from the orbit's SHAPE within every bank,",
            "  not just pooled -- so this is not bank identity wearing a family label.",
            "  D79/D80's four hand-picked statistics were the wrong four, and the",
            "  geometry carries task information they cannot see.",
        ]
        if total:
            lines += [
                "  BUT it cannot be attributed to the TASK. Family and prompt length are",
                "  perfectly collinear above, so 'the shape encodes what the model is",
                "  doing' and 'the shape encodes how long the input is' make identical",
                "  predictions on every orbit this project has banked.",
            ]
    elif within:
        bad = [w["group"] for w in within if w["p"] >= 0.05]
        lines += [
            f"  Family is NOT decodable from shape within {', '.join(bad)}, so the",
            "  pooled hit is carried by differences BETWEEN banks rather than between",
            "  tasks. D79/D80's null survives the hand-picking-free test.",
        ]
    corrs = [pick("shape", "correct", g) for g in banks]
    corrs = [c for c in corrs if c is not None]
    if corrs and all(c["p"] >= 0.05 for c in corrs):
        lines += [
            "  CORRECTNESS is not decodable from shape in any bank, which is D79's",
            "  bounded null without the hand-picked statistics.",
        ]
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    df = collect()
    if df.empty:
        print("no banked trained orbits found")
        return
    rot = float(np.nanmedian([
        is_rotation_invariant(np.load(p).astype(np.float64), m=M)
        for p in _sample_paths()] or [np.nan]))
    rows = []
    for f in ("shape", "position"):
        for t in ("family", "correct", "len_bin"):
            rows.append(decode(df, f, t))
        # WITHIN each bank as well as pooled. Pooled "family" is confounded with
        # WHICH BANK, and D80 shows the depth profile alone separates banks running
        # to different num_steps -- so a pooled hit could be bank identity wearing a
        # family label. Only a within-bank hit is about the task.
        for bank in sorted(df["_group"].unique()):
            for t in ("family", "correct", "len_bin"):
                rows.append(decode(df, f, t, group=bank))
    res = pd.DataFrame(rows)
    save_table(OUT, res, kind="shape_decode", m=M, n_perm=N_PERM,
               n_orbits=int(len(df)), rotation_invariance=rot)
    print(f"saved {OUT}")
    print(report(df, res, rot))


def _sample_paths(k: int = 5) -> list[str]:
    for _, path, suf in SOURCES:
        mp = os.path.join(path, "manifest.json")
        if not os.path.exists(mp):
            continue
        with open(mp, encoding="utf-8") as fh:
            recs = [r for r in json.load(fh) if r.get("ok")]
        out = [os.path.join(path, r["tag"] + suf) for r in recs[:k]]
        out = [p for p in out if os.path.exists(p)]
        if out:
            return out
    return []


if __name__ == "__main__":
    main()
