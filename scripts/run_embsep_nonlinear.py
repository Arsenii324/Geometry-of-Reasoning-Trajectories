"""D136: is the rotation regime a NONLINEAR function of the token embedding?

WHY THIS EXISTS. D135 found that the regime is not readable from the token embedding
by a linear rule: leave-one-out 0.551 against a permuted null of 0.530, below the
0.592 majority-class baseline. That refuted D134's untested "by elimination it lives
in the embedding" inference -- but only its LINEAR form. The prelude is a nonlinear
map from embedding to `e`, so "not linearly readable" and "not in the embedding" are
different claims, and D135 is only licensed to make the first.

This separates them on the same banked embeddings, with no GPU.

PREREGISTERED PREDICTIONS (written before running):

  P1  If the regime is a nonlinear function of the embedding that 49 examples can
      see, at least one of the six classifiers below beats its own permuted null at
      p < 0.05 AFTER the multiplicity correction in P3.

  P2  The 1-D controls (embedding norm, mean, frequency-proxy) are the cheapest
      possible mechanism. If ANY of them separates, the answer is one scalar and the
      claim becomes concrete rather than "somewhere in 5280 dimensions".

  P3  MULTIPLICITY IS CORRECTED BY CONSTRUCTION, NOT BY A FOOTNOTE. Six classifiers
      on 49 points will produce a best-of-six that looks significant against a
      single-classifier null. The reported test is therefore
          max_over_classifiers(observed)  vs  the null distribution of
          max_over_classifiers(permuted),
      recomputing every classifier on every permuted draw. This is the whole test;
      per-classifier p-values are printed for diagnosis only and license nothing.

  P4  NULL-CAN-MOVE. The permutation must be able to move the statistic: the label
      vector is shuffled, so a classifier that ignores X scores ~majority-class on
      every draw and the null has near-zero spread. `require_null_can_move` fires if
      fewer than 5 distinct values appear, which would mean the test cannot resolve
      anything and no conclusion may be drawn either way.

WHAT A NULL RESULT WOULD MEAN. Not "the property is not in the token" -- 49 points
cannot rule out a nonlinear rule of any complexity. It would mean the property is not
recoverable from the embedding by any simple rule at this sample size, which is the
honest ceiling of what the banked data supports, and it would move the next test to
`e` (the prelude's output) rather than to more nouns.

Run:  uv run python scripts/run_embsep_nonlinear.py
"""

from __future__ import annotations

import collections
import glob
import json
import pathlib
import random

import numpy as np

from traj_geom.rigor import require_null_can_move

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch/ds_embsep/out/out"
PERIOD = 6              # the damped rotation D132 found, in unrolls
TAIL = 24               # unrolls of the orbit tail the period is read from
N_PERM = 2000
SEED = 0


def _period(seq: np.ndarray) -> int:
    """Dominant period of an orbit tail, by minimum median return distance."""
    n, lo = len(seq), len(seq) - TAIL
    d = {p: float(np.median([np.linalg.norm(seq[t] - seq[t + p]) for t in range(lo, n - p)]))
         for p in range(1, PERIOD * 2 + 1)}
    return min(d.items(), key=lambda kv: kv[1])[0]


def load() -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Return (nouns, embeddings, labels, mixed) for the DETERMINISTIC nouns only.

    D135's P3 gate: 2 of 51 nouns are mixed across their 4 orbits. They are excluded
    and reported, not silently folded in -- a mixed noun has no label to predict.
    """
    man = {m["tag"]: m for m in json.loads((OUT / "manifest.json").read_text())}
    emb = json.loads((OUT / "noun_embeddings.json").read_text())
    orbits = collections.defaultdict(list)
    for f in sorted(glob.glob(str(OUT / "*.npy"))):
        m = man.get(pathlib.Path(f).stem)
        if m:
            orbits[m["pair"]].append(_period(np.load(f).astype(np.float64)[:, -1, :]) == PERIOD)
    mixed = sorted(w for w, v in orbits.items() if len(set(v)) > 1)
    words = sorted(w for w, v in orbits.items() if len(set(v)) == 1)
    X = np.array([emb[w]["emb"] for w in words], dtype=np.float64)
    y = np.array([int(orbits[w][0]) for w in words])
    return words, X, y, mixed


# --- classifiers, each (X, y, i) -> predicted label for held-out row i -------------

def _knn(k: int):
    def f(X, y, i):
        m = np.ones(len(y), bool); m[i] = False
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        s = Xn[m] @ Xn[i]
        nn = y[m][np.argsort(-s)[:k]]
        return int(nn.mean() > 0.5)
    return f


def _ridge(lam: float):
    def f(X, y, i):
        m = np.ones(len(y), bool); m[i] = False
        Xt, yt = X[m], 2.0 * y[m] - 1.0
        mu = Xt.mean(0)
        A = Xt - mu
        w = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(len(yt)), yt)
        return int((X[i] - mu) @ w > 0)
    return f


def _scalar(fn):
    """Threshold a 1-D summary of the embedding at the training median."""
    def f(X, y, i):
        m = np.ones(len(y), bool); m[i] = False
        s = fn(X)
        thr = float(np.median(s[m]))
        hi = y[m][s[m] > thr]
        sign = 1 if (len(hi) and hi.mean() > 0.5) else 0
        return sign if s[i] > thr else 1 - sign
    return f


CLASSIFIERS = {
    "cosine-kNN k=1": _knn(1),
    "cosine-kNN k=3": _knn(3),
    "cosine-kNN k=5": _knn(5),
    "ridge lam=1": _ridge(1.0),
    "ridge lam=100": _ridge(100.0),
    "1-D: ||emb||": _scalar(lambda X: np.linalg.norm(X, axis=1)),
}


def loo(f, X, y) -> float:
    return float(np.mean([f(X, y, i) == y[i] for i in range(len(y))]))


def main() -> int:
    words, X, y, mixed = load()
    base = float(max(y.mean(), 1 - y.mean()))
    print(f"{len(words)} deterministic nouns, dim {X.shape[1]}, "
          f"{int(y.sum())} rotating / {len(y) - int(y.sum())} not")
    print(f"excluded as MIXED (D135 P3): {mixed}")
    print(f"majority-class baseline {base:.3f}\n")

    obs = {n: loo(f, X, y) for n, f in CLASSIFIERS.items()}
    rng = random.Random(SEED)
    perm = []
    for _ in range(N_PERM):
        yp = y.copy(); rng.shuffle(yp)
        perm.append({n: loo(f, X, yp) for n, f in CLASSIFIERS.items()})

    print("per-classifier (DIAGNOSTIC ONLY -- licenses nothing, see P3):")
    for n in CLASSIFIERS:
        d = [p[n] for p in perm]
        pv = (sum(1 for v in d if v >= obs[n]) + 1) / (N_PERM + 1)
        print(f"  {n:16s} loo {obs[n]:.3f}   null {np.mean(d):.3f} "
              f"(sd {np.std(d):.3f})   p {pv:.3f}")

    om, dm = max(obs.values()), [max(p.values()) for p in perm]
    require_null_can_move(om, dm, what="max-over-classifiers permutation null")
    pv = (sum(1 for v in dm if v >= om) + 1) / (N_PERM + 1)
    best = max(obs, key=obs.get)
    print(f"\nP3 PRIMARY -- best-of-six against the null of the best-of-six:")
    print(f"  observed max {om:.3f} ({best})")
    print(f"  null of max  {np.mean(dm):.3f} (sd {np.std(dm):.3f}), "
          f"95th pct {np.percentile(dm, 95):.3f}")
    print(f"  p = {pv:.4f}   ->  {'SEPARATES' if pv < 0.05 else 'NULL'}")
    print(f"\n  (a single-classifier null would have put the same {om:.3f} at "
          f"p = {(sum(1 for v in [p[best] for p in perm] if v >= om) + 1) / (N_PERM + 1):.4f}"
          f" -- which is the multiplicity P3 exists to stop.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
