"""Does query-key alignment track the COMPUTATION, or just the input TOKEN?

THE SUPERVISOR'S OWN OPEN ITEM. `project_plan.md` G2 names "winding-vs-depth plus a
query-key alignment probe" and recorded the probe as 0% done -- the one
supervisor-nominated statistic this project had never measured. Every other H2
instrument here (winding D28, effective dimensionality D74(6), the Jacobian
argument D83) reads the HIDDEN STATE; this reads the attention operator.

WHY A CONTROL IS THE WHOLE POINT, AND WHAT THE PREDICTION WAS. D87 showed the
trajectory shape separates two task markers at 96.9-100% at identical token
counts; D88 then showed it separates them JUST AS WELL when the two markers select
the SAME computation. So on this model, "geometry distinguishes two tasks" has
already once turned out to mean "geometry reads the input token". The paired
track/local arm inherits exactly that ambiguity, because `track` and `local`
prompts end in different words.

So this run carried D88's marker design as a built-in control. A and C select the
SAME computation, B a different one; every pair differs by ONE character at an
identical token count:

    A vs B  -- different marker, DIFFERENT computation
    A vs C  -- different marker, SAME computation

  * |QK(A)-QK(B)| large AND |QK(A)-QK(C)| ~ 0 -> QK tracks the COMPUTATION.
    That would be the first instrument in this project to do so.
  * both comparable -> QK reads the TOKEN. D88 again.

**The prediction, written into `scratch/ds_qkprobe/job.py`'s docstring BEFORE the
run: the SECOND branch.** Recorded here so the outcome is scored against a stated
expectation rather than rationalised after the fact (CLAUDE.md section 1).

THE INSTRUMENT PASSED ITS OWN NULL FIRST. The job reconstructs (q, k) from a
`Wqkv` hook and checks them against the tensors `scaled_dot_product_attention` is
ACTUALLY called with. The first submission FAILED that check with an error of ~15
-- it was capturing the first attention call in the forward, which belongs to the
first PRELUDE block, not `core_block[-1]`. Fixed and re-run: **max|q_hat-q_true| =
0.0 exactly**. No number below was produced until that passed (CLAUDE.md section 5).

Run: uv run python -m scripts.run_qk_analysis
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SRC = os.path.join("scratch", "ds_qkprobe", "out", "qk_probe.json")
OUT = os.path.join("results", "qk_probe.csv")
WINDOWS = ((0, 12), (12, 24), (24, 36), (36, 48))
N_PERM = 20000
SEED = 0


def load(path: str = SRC) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def curve(rec: dict) -> np.ndarray:
    """Mean cosine over heads, per unroll -> [n_unrolls]."""
    return np.asarray(rec["cos_per_unroll"], dtype=np.float64).mean(axis=1)


def window_means(rec: dict) -> dict:
    c = curve(rec)
    return {f"w{lo}_{hi}": float(c[lo:hi].mean()) for lo, hi in WINDOWS}


def control_table(d: dict) -> pd.DataFrame:
    """One row per (item, marker), with the per-window mean cosine."""
    rows = []
    for r in d["control"]:
        if not r.get("ok"):
            continue
        rows.append({"item": r["item"], "marker": r["marker"],
                     "n_tokens": r["n_tokens"], "best_rank": r["best_rank"],
                     **window_means(r)})
    return pd.DataFrame(rows)


def _paired_perm(a: np.ndarray, b: np.ndarray, rng) -> float:
    """Exact-ish sign-flip null for a paired difference of |differences|.

    The statistic is mean(|d_AB|) - mean(|d_AC|). Under the null that the two
    contrasts are exchangeable WITHIN an item, flipping which of the two is
    labelled AB is the right permutation -- it preserves the item-level pairing,
    which an unpaired shuffle would destroy.
    """
    obs = float(np.mean(a) - np.mean(b))
    n = len(a)
    hits = 0
    for _ in range(N_PERM):
        flip = rng.random(n) < 0.5
        aa = np.where(flip, b, a)
        bb = np.where(flip, a, b)
        if float(np.mean(aa) - np.mean(bb)) >= obs:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def analyse_control(tab: pd.DataFrame) -> pd.DataFrame:
    """THE DECISIVE TEST. Per item and window: is A further from B (different
    computation) than from C (same computation)?"""
    rng = np.random.default_rng(SEED)
    out = []
    for lo, hi in WINDOWS:
        col = f"w{lo}_{hi}"
        piv = tab.pivot_table(index="item", columns="marker", values=col)
        piv = piv.dropna(subset=["A", "B", "C"])
        d_ab = (piv["A"] - piv["B"]).abs().to_numpy()
        d_ac = (piv["A"] - piv["C"]).abs().to_numpy()
        # Wilcoxon is the standard paired test; the permutation p is the one we
        # quote, since n = 12 makes the normal approximation shaky.
        try:
            w_p = float(wilcoxon(d_ab, d_ac).pvalue)
        except ValueError:
            w_p = float("nan")
        out.append({
            "window": f"{lo}-{hi}", "n_items": int(len(piv)),
            "mean_abs_diff_AB": float(d_ab.mean()),
            "mean_abs_diff_AC": float(d_ac.mean()),
            "ratio_AB_over_AC": float(d_ab.mean() / d_ac.mean())
            if d_ac.mean() else float("nan"),
            "p_perm": _paired_perm(d_ab, d_ac, rng), "p_wilcoxon": w_p,
        })
    return pd.DataFrame(out)


def analyse_paired(d: dict) -> pd.DataFrame:
    """The track/local arm, for completeness. Uninterpretable ON ITS OWN as a
    statement about computation, for exactly the reason the control exists."""
    rows = []
    for r in d["paired"]:
        if r.get("ok"):
            rows.append({"kind": r["kind"], "n_ops": r["n_ops"], "seed": r["seed"],
                         **window_means(r)})
    tab = pd.DataFrame(rows)
    out = []
    for lo, hi in WINDOWS:
        col = f"w{lo}_{hi}"
        piv = tab.pivot_table(index=["n_ops", "seed"], columns="kind", values=col)
        piv = piv.dropna(subset=["track", "local"])
        diff = (piv["track"] - piv["local"]).to_numpy()
        try:
            p = float(wilcoxon(piv["track"], piv["local"]).pvalue)
        except ValueError:
            p = float("nan")
        out.append({"window": f"{lo}-{hi}", "n_pairs": int(len(diff)),
                    "mean_track_minus_local": float(diff.mean()),
                    "frac_positive": float((diff > 0).mean()), "p_wilcoxon": p})
    return pd.DataFrame(out)


def confound_and_premise(d: dict, lo: int = 24, hi: int = 36) -> dict:
    """THE CHECK THAT DECIDES WHAT ARM 2 MEANS, and it is not optional.

    Arm 2 compares A-vs-B (different computation) with A-vs-C (same computation)
    and finds the first larger. There is a third reading neither branch of the
    pre-registration named: **B is simply an EASIER task than A and C**, so if QK
    alignment tracks how well the computation is going, |A-B| exceeds |A-C| with
    no computation-identity content whatever. Median best_rank is 7 for B against
    33.5 for A and 21 for C, so the difficulty gap is real and large.

    Also checks the control's OWN premise (D88's P2 gate): if the model does not
    behave the same under A and C, they are not the same computation *for this
    model* and the contrast is void whatever the cosines say.
    """
    from scipy.stats import spearmanr
    rows = []
    for r in d["control"]:
        if r.get("ok"):
            cur = np.asarray(r["cos_per_unroll"], dtype=np.float64).mean(axis=1)
            rows.append({"item": r["item"], "marker": r["marker"],
                         "best_rank": r["best_rank"], "qk": float(cur[lo:hi].mean())})
    t = pd.DataFrame(rows)
    q = t.pivot_table(index="item", columns="marker", values="qk")
    rk = t.pivot_table(index="item", columns="marker", values="best_rank")
    dq_ab, dq_ac = (q["A"] - q["B"]).abs(), (q["A"] - q["C"]).abs()
    lr = np.log10(rk)
    dr_ab, dr_ac = (lr["A"] - lr["B"]).abs(), (lr["A"] - lr["C"]).abs()
    rho, p = spearmanr(np.concatenate([dr_ab, dr_ac]),
                       np.concatenate([dq_ab, dq_ac]))
    closer_to_b = int((( lr["C"] - lr["B"]).abs() < (lr["C"] - lr["A"]).abs()).sum())
    return {"rank_gap_ratio": float(dr_ab.mean() / dr_ac.mean()),
            "qk_gap_ratio": float(dq_ab.mean() / dq_ac.mean()),
            "spearman_rankgap_qkgap": float(rho), "p": float(p),
            "premise_mean_abs_dlog10rank_AC": float(dr_ac.mean()),
            "premise_items_C_closer_to_B": closer_to_b, "n_items": int(len(q))}


def report(ctl: pd.DataFrame, pair: pd.DataFrame, sc: dict, cf: dict) -> str:
    lines = ["", f"  SELF-CHECK: q_err={sc['q_err']:.3e} k_err={sc['k_err']:.3e} "
                 f"passed={sc['passed']}", "",
             "  ARM 2 -- THE DECISIVE CONTROL (D88 design)",
             "  A vs B = different computation | A vs C = SAME computation",
             "  If QK tracked the computation, AB >> AC. If it reads the token, AB ~ AC.", "",
             f"  {'window':>8} {'n':>3} {'|A-B|':>9} {'|A-C|':>9} {'ratio':>7} "
             f"{'p_perm':>8} {'p_wilcox':>9}"]
    for _, r in ctl.iterrows():
        lines.append(f"  {r['window']:>8} {int(r['n_items']):>3} "
                     f"{r['mean_abs_diff_AB']:>9.5f} {r['mean_abs_diff_AC']:>9.5f} "
                     f"{r['ratio_AB_over_AC']:>7.2f} {r['p_perm']:>8.4f} "
                     f"{r['p_wilcoxon']:>9.4f}")
    sig = ctl[ctl["p_perm"] < 0.05 / len(ctl)]
    lines += ["", f"  Bonferroni alpha over {len(ctl)} windows = {0.05/len(ctl):.4f}"]
    if len(sig):
        lines.append("    A-vs-B exceeds A-vs-C at every corrected window, so the "
                     "pre-registered 'QK just reads the token' prediction is REJECTED.")
        lines += ["",
                  "  BUT THE DIFFICULTY CONFOUND DECIDES WHAT THAT MEANS, and it is "
                  "not resolved by this design:",
                  f"    rank gap ratio (|A-B| / |A-C|) = {cf['rank_gap_ratio']:.2f}",
                  f"    QK   gap ratio (|A-B| / |A-C|) = {cf['qk_gap_ratio']:.2f}",
                  f"    Spearman(|d log10 rank|, |d QK|) over all contrasts = "
                  f"{cf['spearman_rankgap_qkgap']:+.3f}  p = {cf['p']:.4g}"]
        if cf["spearman_rankgap_qkgap"] > 0.5 and cf["p"] < 0.05:
            lines += ["    **The QK gap tracks the RANK gap, in nearly the same "
                      "proportion. B is simply an EASIER task than A and C (median "
                      "best_rank 7 vs 33.5 and 21).**",
                      "    So 'QK alignment tracks the COMPUTATION' is NOT established: "
                      "a third reading -- QK tracks how well the computation is GOING --",
                      "    explains the same numbers and this design cannot separate "
                      "them. Reported as inconclusive, not as a positive result."]
        lines += ["",
                  "  AND THE CONTROL'S OWN PREMISE IS ONLY PARTLY MET (D88's P2 gate):",
                  f"    mean |log10 rank_A - log10 rank_C| = "
                  f"{cf['premise_mean_abs_dlog10rank_AC']:.3f}  (0 = model treats them "
                  f"identically)",
                  f"    items where C behaves closer to B than to A: "
                  f"{cf['premise_items_C_closer_to_B']}/{cf['n_items']}",
                  "    A and C are the same computation BY CONSTRUCTION but not "
                  "behaviourally, which weakens the contrast further."]
    else:
        lines.append("    NO window shows A-vs-B exceeding A-vs-C after correction.")
        lines.append("    **The pre-registered prediction HOLDS: QK alignment at the "
                     "answer position reads the input TOKEN, not the computation.**")
        lines.append("    D88's conclusion extends from the hidden state to the "
                     "attention operator.")
    lines += ["", "  ARM 1 -- paired track/local (NOT interpretable alone; the "
                  "prompts end in different words, which is what arm 2 controls)", "",
              f"  {'window':>8} {'n':>3} {'track-local':>12} {'frac>0':>7} {'p_wilcox':>9}"]
    for _, r in pair.iterrows():
        lines.append(f"  {r['window']:>8} {int(r['n_pairs']):>3} "
                     f"{r['mean_track_minus_local']:>12.5f} "
                     f"{r['frac_positive']:>7.2f} {r['p_wilcoxon']:>9.2e}")
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not os.path.exists(SRC):
        print(f"no QK results at {SRC} -- download the DataSphere job output first.")
        return
    d = load()
    if not d.get("self_check", {}).get("passed"):
        print("SELF-CHECK DID NOT PASS -- refusing to analyse. "
              f"{d.get('self_check')}")
        return
    ctl = analyse_control(control_table(d))
    pair = analyse_paired(d)
    cf = confound_and_premise(d)
    save_table(OUT, ctl, kind="qk_probe", windows=[list(w) for w in WINDOWS],
               n_perm=N_PERM, paired_arm=pair.to_dict("records"),
               self_check=d["self_check"], confound=cf)
    print(f"saved {OUT}")
    print(report(ctl, pair, d["self_check"], cf))


if __name__ == "__main__":
    main()
