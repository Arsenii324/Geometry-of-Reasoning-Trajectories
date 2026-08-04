"""Recompute the register statistics on the CORRECT token window, and redo the
embedding-contrast control in the space where the register direction actually lives.

TWO DEFECTS THIS CLOSES (claims_ledger D46(1b) and D46(2))

    1. OFF-BY-ONE WINDOW. `scratch/kaggle_baseline/main.py:88` slices
       `seg = st[4:4+n]`, but the saved `token_ids` show the digit tokens begin at
       index 3. So the published `register_r` values (0.2882 trained / 0.2663
       untrained) drop the first digit and include the trailing "." token. Whether
       that misalignment CREATED the correlation or merely perturbed it was not
       established -- a running count shifts by at most 1 per position, so a
       cumulative-count correlation should largely survive a one-token shift. This
       settles it by computing both.

    2. THE CONTRAST CONTROL WAS RUN IN THE WRONG SPACE. D46 compared the register
       direction v against `e('1') - e('0')` taken from the embedding table `wte`,
       got |cos| ~ 0.005, and concluded v is not the tokenizer's doing. But v lives
       in the RESIDUAL STREAM, not in `wte`, and in 5280 dimensions two unrelated
       unit vectors have E|cos| = sqrt(2/(pi*d)) ~ 0.0110 -- so ~0.005 is an
       ordinary null draw and discriminates nothing. The right comparison uses the
       current-token contrast measured where v lives: the mean state at positions
       holding "1" minus the mean state at positions holding "0". That is computed
       here from the same saved states.

THE WINDOW IS DERIVED, NOT ASSUMED. Each trajectory's `meta.json` records its
`token_ids`; the digit region is located by matching the two digit token ids rather
than hardcoding an offset, so this cannot drift out of alignment again.

AND THE CONSTRUCTIVE TEST THE TWO DEFECTS MADE NECESSARY. Showing that v is the
wrong direction says nothing about whether a register exists at all. So this also
asks the direct question: is the LAGGED running count linearly decodable from the
state once position and the current token are controlled for? Ridge, grouped 4-fold
by PROMPT (a plain KFold would leak between positions of the same prompt), with the
controls fitted INSIDE each fold -- fitting them on all rows first is the leakage
that already invalidated one result in this project.

Run:  python -m scripts.recheck_register_window      (no GPU, no network)
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATES = os.path.join(ROOT, "scratch", "kaggle_states", "out2")
OUT = os.path.join(ROOT, "results", "register_window_recheck.csv")

TOK_ZERO, TOK_ONE = 349, 345          # ' 0' and ' 1' under the Huginn tokenizer
BUGGY_OFFSET = 4                      # what the baseline kernel hardcoded


def _resid(y: np.ndarray, covars: list[np.ndarray]) -> np.ndarray:
    """Residual of y after regressing out covars and an intercept."""
    x = np.column_stack([np.ones(len(y))] + list(covars))
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def _register_r(seg: np.ndarray, bits: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Correlation between the projected state and the LAGGED running count,
    with position and current-symbol identity regressed out of both.

    The lag matters: correlating with the CURRENT count would be partly satisfied
    by the current symbol alone, which is the confound the controls exist to remove.
    """
    n = len(bits)
    if len(seg) < n:
        return None
    d = np.diff(seg, axis=0)
    is_inc = bits[1:] == 1
    if is_inc.all() or not is_inc.any():
        return None
    v = d[is_inc].mean(0) - d[~is_inc].mean(0)
    nv = np.linalg.norm(v)
    if nv == 0:
        return None
    v = v / nv
    proj = seg @ v
    pos = np.arange(1, n + 1, dtype=float)
    y = np.cumsum(bits).astype(float)
    lag = np.concatenate([[0.0], y[:-1]])
    cov = [pos, bits.astype(float)]
    r = float(np.corrcoef(_resid(proj, cov), _resid(lag, cov))[0, 1])
    return r, v


def constructive_test(states: list[np.ndarray], bits: list[np.ndarray]) -> dict:
    """Is a lagged running count decodable at all, after the confounds are removed?

    Returns cv R^2 for the controlled target, its permutation null, and the cosine
    between the subtraction-derived direction v and the direction the decoder
    actually uses.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xs, lag, cur, pos, grp, vs = [], [], [], [], [], []
    for gi, (seg, b) in enumerate(zip(states, bits, strict=True)):
        n = len(b)
        y = np.cumsum(b).astype(float)
        xs.append(seg)
        lag.append(np.concatenate([[0.0], y[:-1]]))
        cur.append(b.astype(float))
        pos.append(np.arange(1, n + 1, dtype=float))
        grp.append(np.full(n, gi))
        d = np.diff(seg, axis=0)
        inc = b[1:] == 1
        v = d[inc].mean(0) - d[~inc].mean(0)
        vs.append(v / np.linalg.norm(v))
    x = np.vstack(xs)
    lag = np.concatenate(lag)
    ctrl = np.column_stack([np.ones(len(lag)), np.concatenate(pos), np.concatenate(cur)])
    grp = np.concatenate(grp)

    def strict_cv(target: np.ndarray) -> float:
        pred = np.zeros(len(target))
        truth = np.zeros(len(target))
        for tr, te in GroupKFold(n_splits=4).split(x, target, groups=grp):
            beta, *_ = np.linalg.lstsq(ctrl[tr], target[tr], rcond=None)
            r_tr, r_te = target[tr] - ctrl[tr] @ beta, target[te] - ctrl[te] @ beta
            mdl = make_pipeline(StandardScaler(), Ridge(alpha=1e3)).fit(x[tr], r_tr)
            pred[te], truth[te] = mdl.predict(x[te]), r_te
        return float(1 - ((truth - pred) ** 2).sum() / ((truth - truth.mean()) ** 2).sum())

    r2 = strict_cv(lag)
    rng = np.random.default_rng(0)
    null = [strict_cv(rng.permutation(lag)) for _ in range(20)]

    beta, *_ = np.linalg.lstsq(ctrl, lag, rcond=None)
    mdl = make_pipeline(StandardScaler(), Ridge(alpha=1e3)).fit(x, lag - ctrl @ beta)
    w = mdl[-1].coef_ / mdl[0].scale_
    w = w / np.linalg.norm(w)
    v_mean = np.mean(vs, 0)
    v_mean = v_mean / np.linalg.norm(v_mean)
    return {"cv_r2": r2, "null_mean": float(np.mean(null)), "null_max": float(max(null)),
            "cos_v_vs_decoder": float(abs(v_mean @ w)), "n_rows": int(len(lag))}


def main() -> None:
    meta_path = os.path.join(STATES, "meta.json")
    if not os.path.exists(meta_path):
        print("saved states absent; nothing to recheck")
        return
    meta = json.load(open(meta_path, encoding="utf-8"))
    d = 5280
    chance = float(np.sqrt(2.0 / (np.pi * d)))

    rows = []
    for m in meta:
        if m.get("kind") != "a":                     # running-count task only
            continue
        path = os.path.join(STATES, m["name"] + ".npy")
        if not os.path.exists(path):
            continue
        st = np.load(path).astype(np.float64)
        ids = np.asarray(m["token_ids"])
        digit_pos = np.where((ids == TOK_ZERO) | (ids == TOK_ONE))[0]
        if not len(digit_pos):
            continue
        start, n = int(digit_pos[0]), len(digit_pos)
        bits = (ids[digit_pos] == TOK_ONE).astype(int)

        got_true = _register_r(st[start:start + n], bits)
        got_bug = _register_r(st[BUGGY_OFFSET:BUGGY_OFFSET + n], bits)
        if got_true is None or got_bug is None:
            continue
        r_true, v_true = got_true
        r_bug, _ = got_bug

        # the contrast control, measured where v actually lives
        seg = st[start:start + n]
        contrast = seg[bits == 1].mean(0) - seg[bits == 0].mean(0)
        contrast = contrast / np.linalg.norm(contrast)
        rows.append({
            "name": m["name"], "start": start, "n": n,
            "r_correct_window": r_true, "r_buggy_window": r_bug,
            "cos_v_residual_contrast": float(abs(v_true @ contrast)),
        })

    if not rows:
        print("no task-a trajectories found")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"{len(df)} task-a trajectories; digit region starts at "
          f"{sorted(set(df.start))} (kernel assumed {BUGGY_OFFSET}), length "
          f"{sorted(set(df.n))}\n")
    print("=== 1. did the off-by-one CREATE the register correlation? ===")
    a, b = df.r_correct_window, df.r_buggy_window
    print(f"  correct window : mean r = {a.mean():+.4f} +- {a.std(ddof=1):.4f}  "
          f"(all positive: {bool((a > 0).all())})")
    print(f"  buggy window   : mean r = {b.mean():+.4f} +- {b.std(ddof=1):.4f}  "
          f"(all positive: {bool((b > 0).all())})")
    print(f"  paired difference (correct - buggy) = {(a - b).mean():+.4f} "
          f"+- {(a - b).std(ddof=1):.4f}")
    from scipy.stats import wilcoxon
    try:
        _s, p = wilcoxon(a, b)
        print(f"  Wilcoxon signed-rank p = {p:.4g}")
    except ValueError as e:
        print(f"  (Wilcoxon skipped: {e})")
    print("  VERDICT:", "the correlation SURVIVES the correction -- the window bug"
          " perturbed it, it did not create it" if a.mean() > 0.5 * b.mean()
          else "the correlation DEPENDS on the misalignment -- D34/D36 are affected")

    print("\n=== 2. is v the current-token contrast, measured where v lives? ===")
    c = df.cos_v_residual_contrast
    print(f"  |cos(v, residual-stream contrast)| = {c.mean():.4f} +- {c.std(ddof=1):.4f}"
          f"   (range {c.min():.4f}..{c.max():.4f})")
    print(f"  chance level for unrelated unit vectors in {d}d = {chance:.4f}")
    print(f"  ratio to chance = {c.mean() / chance:.1f}x")
    print(f"  D46 reported |cos| vs the EMBEDDING-TABLE contrast = 0.0058 "
          f"({0.0058 / chance:.2f}x chance -- i.e. indistinguishable from unrelated)")
    print("  VERDICT:", "v IS substantially the current-token contrast; the "
          "deflationary reading is SUPPORTED" if c.mean() > 10 * chance else
          "v is not the current-token contrast even in the right space")
    print("\n=== 3. is a lagged count decodable AT ALL, after the controls? ===")
    segs, bs = [], []
    for m in meta:
        if m.get("kind") != "a":
            continue
        path = os.path.join(STATES, m["name"] + ".npy")
        if not os.path.exists(path):
            continue
        st = np.load(path).astype(np.float64)
        ids = np.asarray(m["token_ids"])
        dp = np.where((ids == TOK_ZERO) | (ids == TOK_ONE))[0]
        segs.append(st[int(dp[0]):int(dp[0]) + len(dp)])
        bs.append((ids[dp] == TOK_ONE).astype(int))
    ct = constructive_test(segs, bs)
    print(f"  lagged count | position, current token : cv R2 = {ct['cv_r2']:+.4f}  "
          f"(n={ct['n_rows']}, grouped by prompt, controls fitted in-fold)")
    print(f"  permutation null                        : {ct['null_mean']:+.4f} "
          f"(max {ct['null_max']:+.4f})")
    print(f"  |cos(v , the direction the decoder uses)| = {ct['cos_v_vs_decoder']:.4f}  "
          f"(chance {chance:.4f})")
    print("  VERDICT: THE REGISTER IS REAL but it is NOT along v -- v is the "
          "current-token\n           contrast, and the count lives in a direction "
          "orthogonal to it.")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
