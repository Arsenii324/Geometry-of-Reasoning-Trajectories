"""A3: does the learned block cycle carry TASK identity, or only the input?

THE THREAD (OPEN_THREADS A3, `open, data ready`, zero GPU)
    D98 established that hooking all four core blocks shows each converging to its
    OWN fixed point, the four separated by 22-58 where states have norm 70-76 -- a
    cycle whose perimeter is 1.6x the whole distance the state travels. D104 called
    it learned. A3 asks the question that was never run: does that cycle carry
    information about WHICH TASK is being performed, or only about the input?

    It matters because D125 found the fixed point encodes the INPUT, not the answer
    (cross-marker same-answer similarity +0.00016, z=+0.40, against a shared-input
    effect 4.3x larger). If the cycle is the same story, the dynamics carry the
    prompt and nothing else. If the cycle separates tasks on an IDENTICAL input,
    the dynamics carry the instruction.

WHY THIS BANK MAKES IT CLEANLY ASKABLE
    `ds_blockbank` is a PAIRED design and that is what rescues it. Every sequence
    appears twice -- 65 sequences x 2 markers = 130 orbits -- with byte-identical
    rules and digits, differing only in the final `Task: A` / `Task: B` token. So
    marker can be decoded with the input held EXACTLY fixed: same length (53 or 55
    tokens), same digits, same rules text.

THE CONFOUND I MUST NOT WALK INTO
    Decoding the PAIR (count_vs_last / first_vs_last / max_vs_min) would be
    worthless here. The three pairs differ in their rules text, and worse, D132/D134
    showed a single noun sets the rotation regime -- `symbol` rotates 36/36 while
    `element` and `item` rotate 0/36 at identical length. `count_vs_last` says "last
    SYMBOL"; the others say "element". So a pair-decode would be reading D132's
    regime switch, not task identity. **Only the within-sequence MARKER decode is
    reported as a test of A3.** The pair-decode is computed anyway and reported as
    the positive control it is -- if it does NOT separate, the features are dead.

    D88 (the shape reads the marker token) is why the marker decode is run on the
    RESIDUAL rather than the state: the residual is what is left after each block
    has converged, so it is not the token embedding sitting in the stream.

PRE-REGISTERED, before running
    P1 CONTROL. Pair must be decodable from the residual well above chance (0.333).
       If not, the features carry nothing and nothing below is interpretable.
    P2 TEST. Marker decoded with folds GROUPED BY SEQUENCE, so a sequence never
       appears in train and test. Above the permutation null => the cycle carries
       task identity on a fixed input. At the null => it carries the input only,
       matching D125's finding for the fixed point.
    P3 The permutation null must be centred near chance. With d=21120 and n=130 a
       positive null would mean the classifier is memorising, and P2 unreadable.

Run:  python -m scripts.run_cycle_residual_decode      (no GPU, no network)
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "scratch", "ds_blockbank", "out", "out")
OUT = os.path.join(ROOT, "results", "cycle_residual_decode.csv")
N_NULL = 40


def load():
    meta = json.load(open(os.path.join(BANK, "manifest.json"), encoding="utf-8"))
    resid, states, keep = [], [], []
    for m in meta:
        f = os.path.join(BANK, m["tag"] + ".npy")
        if not os.path.exists(f):
            continue
        a = np.load(f).astype(np.float64)          # [64, 4, 5280]
        # per-block LAST-STEP residual: what is left once each block has converged.
        # D146 uses the same quantity (2.9152 rotating vs 0.0185 settling).
        resid.append((a[-1] - a[-2]).ravel())
        states.append(a[-1].ravel())               # for the D88 contrast
        keep.append(m)
    return np.array(resid), np.array(states), keep


def decode(x, y, groups, n_null=N_NULL, seed=0):
    """Grouped-CV accuracy of a linear classifier, plus a permutation null."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # PCA INSIDE the pipeline, so it is refitted per fold and cannot leak. With
    # n=130 and d=21120 a direct fit is both slow (~615 fits for the nulls) and
    # rank-deficient; 40 components is well above the 3-way and 2-way structure
    # being decoded and keeps every fit closed-form-fast.
    mdl = make_pipeline(StandardScaler(),
                        PCA(n_components=40, random_state=0),
                        LogisticRegression(C=0.1, max_iter=5000))
    n_splits = min(5, len(set(groups)))

    def acc(target):
        p = cross_val_predict(mdl, x, target, cv=GroupKFold(n_splits=n_splits),
                              groups=groups)
        return float((p == target).mean())

    a = acc(y)
    rng = np.random.default_rng(seed)
    null = [acc(rng.permutation(y)) for _ in range(n_null)]
    return a, float(np.mean(null)), float(max(null))


def main() -> None:
    if not os.path.exists(os.path.join(BANK, "manifest.json")):
        print("bank absent")
        return
    resid, states, meta = load()
    pair = np.array([m["pair"] for m in meta])
    marker = np.array([m["marker"] for m in meta])
    seq = np.array([m["seq"] for m in meta])
    ntok = np.array([int(m["n_tokens"]) for m in meta])
    print(f"{len(meta)} orbits, residual dim {resid.shape[1]}, "
          f"{len(set(seq))} distinct sequences, token lengths {sorted(set(ntok))}")
    n_paired = sum(1 for s in set(seq) if len(set(marker[seq == s])) == 2)
    print(f"sequences with BOTH markers: {n_paired}/{len(set(seq))}\n")

    rows = []
    print("=== P1 CONTROL: is PAIR decodable? (confounded by D132's noun; a control only) ===")
    a, nm, nx = decode(resid, pair, seq)
    rows.append({"target": "pair", "features": "residual", "acc": a,
                 "null_mean": nm, "null_max": nx, "chance": 1 / 3})
    print(f"  pair from residual : acc {a:.3f}  null {nm:.3f} (max {nx:.3f})  "
          f"chance 0.333  -> {'features carry signal' if a > nx else 'DEAD FEATURES'}")

    print("\n=== P2 TEST: is MARKER decodable, folds grouped by SEQUENCE? ===")
    print("  (same rules, same digits, same length -- one token differs)")
    for name, feat in (("residual", resid), ("final state", states)):
        a, nm, nx = decode(feat, marker, seq)
        rows.append({"target": "marker", "features": name, "acc": a,
                     "null_mean": nm, "null_max": nx, "chance": 0.5})
        print(f"  marker from {name:>12} : acc {a:.3f}  null {nm:.3f} "
              f"(max {nx:.3f})  chance 0.500")

    import pandas as pd
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

    ctrl = rows[0]
    test = [r for r in rows if r["target"] == "marker" and r["features"] == "residual"][0]
    st = [r for r in rows if r["target"] == "marker" and r["features"] == "final state"][0]
    print("\n=== VERDICT ===")
    if ctrl["acc"] <= ctrl["null_max"]:
        print("  P1 FAILED: pair is not decodable, so the residual features carry")
        print("  nothing and P2 is uninterpretable. Nothing may be concluded.")
    elif test["acc"] > test["null_max"]:
        print(f"  THE CYCLE CARRIES TASK IDENTITY. Marker decodes at {test['acc']:.3f}")
        print(f"  from the residual against a null topping out at {test['null_max']:.3f},")
        print("  on inputs identical but for one token. That is a property of the")
        print("  DYNAMICS, not of the prompt -- and it contrasts with D125, where the")
        print("  fixed point encoded the input and not the answer.")
    else:
        print(f"  NULL. Marker decodes at {test['acc']:.3f} against a null reaching")
        print(f"  {test['null_max']:.3f}. The cycle residual does NOT separate two")
        print("  tasks on an identical input, which matches D125's fixed-point")
        print("  result: the dynamics carry the INPUT, not the instruction.")
    print(f"\n  contrast, final state (D88 reads the marker token here): "
          f"{st['acc']:.3f} vs null {st['null_max']:.3f}")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
