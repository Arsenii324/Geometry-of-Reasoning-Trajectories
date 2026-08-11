"""A37 (zero GPU): at WHICH unroll does the rotation regime become linearly decodable?

THE QUESTION D164 LEAVES ON THE TABLE. D164 shows the rotating region is chord-convex in
`e`-space and not in `wte`-space on a matched pair, and closes by noting that a set with that
property "will look linearly separable to a probe on `e` and unseparable to a probe on
embeddings -- which is exactly the pattern of D135/D136's nulls against D140/D161's causal
successes." That sentence is an implication, not a measurement. This measures it.

D155 is the reason the obvious version does not work. D135/D136 fitted classifiers to 49
banked noun embeddings and found nothing, and D155 showed that design cannot resolve a linear
separation below Cohen's d ~ 10 -- so their null is uninformative and cannot be quoted as
evidence for anything. **A matched-design contrast sidesteps that entirely**: run the SAME
probe, at the SAME n, with the SAME regularisation, on features taken at different points of
the computation. If one arm separates and another does not, the design demonstrably CAN detect
a separation at this sample size, and the failing arm is then informative about the features
rather than about the power.

WHAT IS BANKED, AND WHY IT ANSWERS THIS. `scratch/ds_embsep/out/embsep.tgz` holds 204 arrays
of shape (64 unrolls, 4 core blocks, 5280 dims) -- 51 nouns x 2 markers x 2 sequences, from
A23. That is the full per-block trajectory, so the probe can be run at any unroll.

THE ANTI-CIRCULARITY ARGUMENT, WHICH IS THE POINT OF THE DESIGN. The label is
`rotation_power(traj) > 0.6677` (D141), computed over the trajectory's TAIL -- the last 24 of
64 unrolls, at period 6. A probe on the tail state would be reading the same numbers that
define the label, and would prove nothing. **A probe on unroll 1 cannot be**: one state carries
no periodicity at all, and six unrolls are needed before a period-6 component is even defined.
So a high leave-one-noun-out accuracy at unroll 1 says the regime is already determined by the
state the prelude hands to the first core block -- which is a statement about `e`, obtained
without ever fitting `e`.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, VOID WITHOUT IT. n = 51 nouns against p = 5280 dimensions is a regime
      where a linear probe can memorise anything. With the noun labels PERMUTED, leave-one-
      noun-out accuracy must fall to the majority baseline. If the permuted arm scores above
      chance, the probe is fitting the fold structure and no arm may be read.

  P2  GROUPING, WHICH IS WHERE THIS DESIGN WOULD LEAK. All four arrays of a noun (2 markers x
      2 sequences) share its label, so a random-split CV would train and test on the same
      noun and score near 1.0 for trivial reasons. Held out BY NOUN throughout.

  P3  PRIMARY -- THE ONSET CURVE. Leave-one-noun-out accuracy as a function of unroll. Three
      outcomes, distinguished in advance: **(a) high at unroll 1 and flat** -- the regime is
      set by the input and the recurrence only expresses it, which is D161's causal result
      seen from the decoding side; **(b) at chance early and rising** -- the regime is
      constructed by the recurrence, and "the embedding determines it" needs rewriting;
      **(c) never above chance** -- the property is not linear in the state at any depth, and
      the D135/D136 nulls stop looking like a power problem.

  P4  THE LATE ARM IS A POSITIVE CONTROL, NOT A RESULT. Accuracy at the final unroll should be
      high by construction, because the label is a function of the tail. If it is NOT high,
      the features or the labels are mismatched and the whole run is void. It is reported for
      exactly that reason and must not be quoted as a finding.

CORRECTION TO P4, MADE AFTER THE FIRST RUN AND BEFORE READING ANYTHING ELSE. The first pass
ran single states and P4 FAILED -- the final unroll scored 0.657 against a 0.588 baseline.
**The preregistered reasoning for P4 was wrong, not the model.** `rotation_power` consumes the
tail as a SEQUENCE; a period is a property of successive states and no single state can carry
one, so "high by construction" never followed. Under the rule that an instrument failing its
own null decides nothing, the single-state arms are inconclusive rather than negative: they
cannot separate "the regime is not linear in the state" from "this design cannot see it."

So the features become WINDOWS, which is what the label actually reads:

    window(u, W) = states u .. u+W-1 at the last core block, flattened

with a per-fold PCA to 40 components (fit on the training nouns only) ahead of the logistic
probe, since n = 204 arrays against 5280*W raw dimensions is otherwise pure memorisation.

  P4' THE REPAIRED POSITIVE CONTROL. The tail window (W = 24, the exact span `rotation_power`
      integrates) must decode well above baseline. It is circular by design -- that is what a
      positive control is -- and it is what licenses reading any other arm.

  P6  THE SCIENTIFIC ARM IS NOW SHARP. A window of W = 5 CANNOT express a period-6 component:
      five samples do not span one cycle. So an early W = 5 window scoring high would mean the
      regime is legible before the periodicity it names is even defined -- the strongest form
      of "already determined at input" this data can support.

Run:  uv run python scripts/regime_onset.py <dir-with-embsep-npy>
"""

from __future__ import annotations

import collections
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from traj_geom.metrics.dynamics import rotation_power  # noqa: E402

THRESHOLD = 0.6677          # D141, the one threshold
LAST_BLOCK = 3              # every instrument in this project reads core_block[-1]
UNROLLS = (0, 1, 2, 3, 5, 8, 12, 16, 24, 32, 47, 63)
SEED = 20260811


def load(dirpath: str):
    """-> per-file (noun, unroll-indexed last-block states), plus the label per noun."""
    files = sorted(glob.glob(os.path.join(dirpath, "*.npy")))
    if not files:
        raise SystemExit(f"no .npy under {dirpath}")
    per_noun_R = collections.defaultdict(list)
    rows = []
    for f in files:
        noun = os.path.basename(f).rsplit("_", 2)[0]
        arr = np.load(f)                      # (64, 4, 5280) float16
        traj = arr[:, LAST_BLOCK, :].astype(np.float32)
        per_noun_R[noun].append(float(rotation_power(traj)))
        rows.append((noun, traj))
    return rows, per_noun_R


def loo_by_noun(X, y, groups, *, C=1.0, seed=SEED, permute=False, n_pc=40):
    """Leave-one-NOUN-out accuracy. Every array of a noun is held out together (P2).

    PCA is fit on the TRAINING nouns only inside each fold. Fitting it on all the data first
    would leak the held-out noun into the basis -- the same class of error as splitting by
    array instead of by noun, one level down.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    groups = np.asarray(groups)
    if permute:
        rng = np.random.default_rng(seed)
        uniq = sorted(set(groups))
        lab = {g: y[groups == g][0] for g in uniq}
        perm = rng.permutation([lab[g] for g in uniq])
        mapping = dict(zip(uniq, perm))
        y = np.array([mapping[g] for g in groups])

    correct = total = 0
    for g in sorted(set(groups)):
        te = groups == g
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        k = min(n_pc, Xtr.shape[0] - 1, Xtr.shape[1])
        pca = PCA(n_components=k, random_state=seed).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = LogisticRegression(C=C, max_iter=5000)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        correct += int((pred == y[te]).sum())
        total += int(te.sum())
    return correct / total if total else float("nan")


def main() -> int:
    dirpath = sys.argv[1] if len(sys.argv) > 1 else "embsep/out"
    rows, per_noun_R = load(dirpath)

    labels, mixed = {}, []
    for noun, Rs in per_noun_R.items():
        calls = [R > THRESHOLD for R in Rs]
        labels[noun] = int(calls[0])
        if len(set(calls)) > 1:
            mixed.append((noun, [round(R, 3) for R in Rs]))

    n_rot = sum(labels.values())
    print(f"{len(rows)} arrays over {len(labels)} nouns; "
          f"{n_rot} rotating / {len(labels) - n_rot} settling")
    base = max(n_rot, len(labels) - n_rot) / len(labels)
    print(f"majority baseline (by noun) = {base:.3f}")
    if mixed:
        # D134 records that rotation is deterministic given the noun. Any noun whose four
        # arrays disagree contradicts that and is reported rather than silently averaged.
        print(f"WARNING -- {len(mixed)} nouns are NOT deterministic across their arrays:")
        for noun, Rs in mixed[:10]:
            print(f"    {noun:12s} R = {Rs}")
    else:
        print("all nouns deterministic across marker and sequence (consistent with D134)")

    # `set` is the one noun D141 records as genuinely intermediate, and it is the one noun
    # whose four arrays disagree here. It is dropped from the probe rather than forced to a
    # side, and the drop is stated.
    keep = [(n, t) for n, t in rows if n not in {m for m, _ in mixed}]
    print(f"dropping {len(rows) - len(keep)} arrays from ambiguous nouns "
          f"{sorted({m for m, _ in mixed})}; {len(keep)} remain")
    groups = np.array([n for n, _ in keep])
    y = np.array([labels[n] for n, _ in keep])
    n_rot = int((y == 1).sum())
    base = max(n_rot, len(y) - n_rot) / len(y)
    print(f"baseline on the probed set = {base:.3f}")

    ARMS = [("state", u, 1) for u in UNROLLS] + \
           [("win5", u, 5) for u in (0, 1, 2, 4, 8, 16, 32)] + \
           [("win12", 0, 12), ("win24", 0, 24), ("win24-tail", 40, 24)]

    print(f"\n{'arm':>11s} {'start':>6s} {'W':>3s} {'LOO acc':>8s} {'permuted':>9s}   "
          f"(n={len(keep)} arrays, PCA-40 per fold, held out by noun)")
    out = []
    for name, u, W in ARMS:
        if u + W > 64:
            continue
        X = np.stack([t[u:u + W].reshape(-1) for _, t in keep]).astype(np.float64)
        acc = loo_by_noun(X, y, groups)
        nul = loo_by_noun(X, y, groups, permute=True)
        tag = ""
        if name == "win24-tail":
            tag = "  <- P4' repaired positive control (circular by design)"
        elif name == "win5" and u <= 1:
            tag = "  <- P6: 5 samples cannot span a period-6 cycle"
        print(f"{name:>11s} {u:6d} {W:3d} {acc:8.3f} {nul:9.3f}{tag}")
        out.append((name, u, W, acc, nul))

    # P7: THE LABEL-RECOVERABILITY CHECK. `win24-tail` is unrolls 40..63 -- exactly the span
    # `rotation_power(period=6, tail=24)` integrates -- so a perfect classifier from those
    # features to the label EXISTS with accuracy 1.000. If the probe cannot approach it, the
    # deficit is the probe's, not the labels'. Feeding R itself as a single feature separates
    # those two: it must score ~1.000, since the label is a threshold on it.
    Rvals = np.array([rotation_power(t) for _, t in keep]).reshape(-1, 1)
    r1d = loo_by_noun(Rvals, y, groups, n_pc=1)
    print(f"\n{'R (1-D)':>11s} {'-':>6s} {'-':>3s} {r1d:8.3f} "
          f"{loo_by_noun(Rvals, y, groups, permute=True, n_pc=1):9.3f}"
          f"  <- P7: the label IS a threshold on this")
    Xtail = np.stack([t[40:64].reshape(-1) for _, t in keep]).astype(np.float64)
    for k in (100, 150):
        acc = loo_by_noun(Xtail, y, groups, n_pc=k)
        print(f"{'win24-tail':>11s} {40:6d} {24:3d} {acc:8.3f} {'':>9s}"
              f"  <- same window, PCA-{k}: is 40 components the bottleneck?")
        out.append((f"win24-tail-pc{k}", 40, 24, acc, 0.0))

    ctrl = next((a for n, u, W, a, _ in out if n == "win24-tail"), float("nan"))
    early5 = [a for n, u, W, a, _ in out if n == "win5" and u <= 1]
    states = [a for n, u, W, a, _ in out if n == "state"]
    perm_max = max(p for *_, p in out)

    print("\n=== READING ===")
    print(f"  P1  worst permuted arm     {perm_max:.3f}   (must sit near {base:.3f})")
    print(f"  P4' tail-window control    {ctrl:.3f}   (must clear baseline to license "
          f"anything else)")
    print(f"  P6  earliest 5-windows     {[round(a, 3) for a in early5]}")
    print(f"      best single state      {max(states):.3f}")
    if perm_max > base + 0.15:
        print("  P1 FAILS -- permuted labels decode above baseline; nothing here is readable.")
    elif not (ctrl > base + 0.15):
        print("  P4' FAILS -- the design cannot recover the label even from the window the "
              "label is computed on. Every other arm is INCONCLUSIVE, not negative.")
    elif early5 and min(early5) > base + 0.15:
        print("  -> the regime is legible BEFORE a period-6 component is definable: it is "
              "carried into the recurrence, not built by it.")
    else:
        print("  -> with a working positive control, the early windows do NOT carry the "
              "regime: the recurrence builds the separation rather than receiving it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
