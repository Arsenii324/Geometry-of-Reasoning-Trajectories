"""Analysis — Barannikov's register, from banked per-position latents (D34, D36).

OWNER: Data+Analysis
STATUS: implemented and run 2026-07-26. CPU only; reads the state arrays dumped
    once by the Kaggle kernel `arsen4ikvar/geometry-position-states`.
TASK: find the increment direction v for each curator task, test that it carries
    the ACCUMULATED register value rather than position or token identity, and
    test whether it behaves as a Z-action.

WHY THIS SCRIPT EXISTS AT ALL
    The results it computes were first obtained in ad-hoc snippets. That is the
    exact failure the rigor audit documents elsewhere in this project -- a
    conclusion whose derivation exists only in someone's terminal history. The
    numbers below are the project's strongest findings, so they get a script.

THE CONSTRUCTION, AND WHY IT IS THE RIGHT ONE
    Barannikov specifies a register in Z updated by a translation T_1. The most
    faithful test is therefore a subtraction, not a probe:

        v = mean(delta | symbol increments) - mean(delta | symbol decrements)

    where delta_i = h_(i+1) - h_i over per-position states. No fitted
    parameters, so nothing to overfit -- which makes it stronger evidence than
    the ridge probe that preceded it, not weaker.

THREE CONFOUNDS, ALL CONTROLLED
    (a) POSITION. The running count rises monotonically with position, and
        position is trivially decodable from positional embeddings:
        R^2(y_i ~ i) = 0.985. Any correlation must be measured after removing
        it. (Task b is naturally safer: depth is a bridge, R^2 = 0.160.)
    (b) TOKEN IDENTITY. The difference between a `1`-position and a
        `0`-position trivially contains the token embedding difference. The
        target is therefore the register value BEFORE the current symbol,
        y_(i-1), which token identity at position i cannot explain, with the
        current symbol also partialled out.
    (c) OFFSET SELECTION. The digit region is located by maximising ||v||. All
        16 strings select the same offset, so the search finds the real region
        rather than per-string noise; `offset_agreement` reports this.

A TEST THAT WAS RETRACTED, RECORDED SO IT IS NOT REPEATED
    Symmetry was first tested as -pc/po, the ratio of displacements relative to
    the mean step. That is an ALGEBRAIC IDENTITY equal to n_open/n_close: with
    base = (n_o a + n_c b)/N, one gets po = n_c(a-b)/N and pc = -n_o(a-b)/N.
    Balanced strings have 31 opens / 32 closes after dropping the first symbol,
    and 31/32 = 0.96875 -- which is exactly what the "measurement" returned, to
    five decimals, with sd 1.9e-16 across 16 random strings. No model property
    can be that constant. `pair_displacement` replaces it: a matched pair must
    return the state to its start, which nothing in the construction forces.

SEMANTIC NOTE, TASK A vs TASK B
    For Task b the Z-action reading is direct: `(` adds 1, `)` subtracts 1, and
    v is the axis of that action. For Task a it is subtler. A `0` does not
    decrement a running count -- it does nothing -- so v, being defined as a
    CONTRAST between `1` and `0` steps, measures a centred bit signal rather
    than a signed increment. That is why the pair displacements come out
    symmetric about zero (`00` -25.3, `11` +25.2) instead of 0 and +2. The
    matched-pair result is the same algebra in both cases, but only Task b
    licenses the literal "T_(-1) = T_(+1)^(-1)" phrasing. Stated because the
    distinction is easy to lose when the numbers look alike.

I/O: reads scratch/kaggle_states/out2/*.npy + meta.json
    -> results/register_analysis.csv, with a provenance sidecar.

Run: uv run python -m scripts.run_register_analysis
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from traj_geom.provenance import save_table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATES = os.path.join(ROOT, "scratch", "kaggle_states", "out2")
HIDDEN = 5280


def _resid(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    """Residual of ``y`` after least-squares removal of ``controls`` (plus intercept)."""
    x = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in controls])
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def increment_direction(states: np.ndarray, is_increment: np.ndarray) -> tuple[np.ndarray, int]:
    """The direction a `+1` symbol moves the state, and the digit-region offset.

    Args:
        states: Per-position states, shape [n_tokens, hidden].
        is_increment: Boolean per symbol; True where the symbol increments the
            register (a `1`, or a `(`).

    Returns:
        ``(v, offset)`` with ``v`` unit-norm. ``offset`` is where the symbol
        region was found, by maximising ``||v||`` -- see the module docstring on
        why that search is safe here.
    """
    n = len(is_increment)
    best: tuple[float, int, np.ndarray] | None = None
    for off in range(0, max(1, len(states) - n)):
        seg = states[off : off + n]
        if len(seg) < n:
            break
        d = np.diff(seg, axis=0)
        inc = is_increment[1:]
        if inc.all() or not inc.any():
            continue
        v = d[inc].mean(0) - d[~inc].mean(0)
        norm = float(np.linalg.norm(v))
        if best is None or norm > best[0]:
            best = (norm, off, v)
    assert best is not None, "no valid offset found"
    return best[2] / np.linalg.norm(best[2]), best[1]


def register_correlation(
    states: np.ndarray, is_increment: np.ndarray, value: np.ndarray, offset: int, v: np.ndarray
) -> float:
    """Correlation of the projection with the register value BEFORE the current symbol.

    Position and the current symbol are both partialled out of each side, so
    neither positional encoding nor token identity can produce the result.
    """
    n = len(is_increment)
    seg = states[offset : offset + n]
    proj = seg @ v
    pos = np.arange(1, n + 1, dtype=float)
    cur = is_increment.astype(float)
    lagged = np.concatenate([[0.0], np.asarray(value, dtype=float)[:-1]])
    return float(np.corrcoef(_resid(proj, [pos, cur]), _resid(lagged, [pos, cur]))[0, 1])


def pair_displacement(
    states: np.ndarray, symbols: np.ndarray, offset: int, v: np.ndarray
) -> dict[str, float]:
    """Net two-step displacement along ``v``, split by adjacent symbol pair.

    A Z-action requires T_(-1) = T_(+1)^(-1), so `+-` and `-+` must return the
    state to its start while `++` and `--` go to +2v and -2v. Unlike the
    retracted ratio test, nothing in the construction of ``v`` forces this.
    """
    n = len(symbols)
    seg = states[offset : offset + n]
    two = (seg[2:] - seg[:-2]) @ v
    pairs = np.array([f"{symbols[i + 1]}{symbols[i + 2]}" for i in range(len(seg) - 2)])
    out: dict[str, float] = {}
    for tag in np.unique(pairs):
        out[f"pair_{tag}"] = float(two[pairs == tag].mean())
    return out


def _load() -> list[dict]:
    meta_path = os.path.join(STATES, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not present. Run the Kaggle kernel "
            "`arsen4ikvar/geometry-position-states` and pull its output here."
        )
    with open(meta_path) as fh:
        return json.load(fh)


def compute() -> pd.DataFrame:
    """Per-string increment direction, register correlation and pair displacements."""
    rows = []
    for m in _load():
        states = np.load(os.path.join(STATES, f"{m['name']}.npy")).astype(np.float64)
        units = np.array(m["units"])
        inc = (units == "1") | (units == 1) | (units == "(")
        value = np.asarray(m["target"], dtype=float)
        v, off = increment_direction(states, inc)
        row = {
            "name": m["name"], "kind": m["kind"], "seed": m["seed"],
            "n_symbols": len(units), "offset": off,
            "register_r": register_correlation(states, inc, value, off, v),
            "v": v,
        }
        row.update(pair_displacement(states, units, off, v))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Report the three findings, each with its own control."""
    df = compute()
    vecs = {k: np.stack(g["v"].values) for k, g in df.groupby("kind")}
    out_df = df.drop(columns=["v"])

    out = os.path.join(ROOT, "results", "register_analysis.csv")
    save_table(
        out, out_df, kind="table", experiment="register_analysis",
        source="scratch/kaggle_states/out2 (Kaggle kernel geometry-position-states)",
        controls="position, current-symbol identity, offset-selection agreement",
    )

    rnd = 1 / np.sqrt(HIDDEN)
    print("\n=== 1. is there a shared increment direction? ===")
    print(f"    (random directions in {HIDDEN} dims have cosine sd {rnd:.4f})")
    for kind, v in vecs.items():
        vn = v / np.linalg.norm(v, axis=1, keepdims=True)
        c = (vn @ vn.T)[np.triu_indices(len(vn), 1)]
        agree = (out_df[out_df.kind == kind].offset.nunique() == 1)
        print(f"  {kind:9s} cosine {c.mean():+.4f} (sd {c.std():.4f}) = "
              f"{c.mean() / rnd:5.1f} sigma   offset agreement: {agree}")

    print("\n=== 2. does it carry the ACCUMULATED value, not position or token identity? ===")
    for kind, g in out_df.groupby("kind"):
        r = g.register_r.values
        t = ttest_1samp(r, 0)
        print(f"  {kind:9s} r = {r.mean():+.4f}   same sign "
              f"{max((r > 0).sum(), (r < 0).sum())}/{len(r)}   p = {t.pvalue:.2e}")

    print("\n=== 3. is it a Z-action? matched pairs must return to the start ===")
    for kind, g in out_df.groupby("kind"):
        cols = [c for c in g.columns if c.startswith("pair_") and g[c].notna().any()]
        if not cols:
            continue
        print(f"  {kind}:")
        for c in sorted(cols):
            print(f"    {c[5:]:4s} {g[c].mean():+9.3f}")
        matched = [c for c in cols if len(c) == 7 and c[5] != c[6]]
        if matched:
            vals = np.concatenate([g[c].dropna().values for c in matched])
            t = ttest_1samp(vals, 0)
            print(f"    matched pairs vs zero: {vals.mean():+.3f}, p = {t.pvalue:.4f}"
                  f"  {'-> returns to start' if t.pvalue > 0.05 else '-> does NOT return'}")

    ka = vecs.get("a")
    kb = vecs.get("b_bal")
    if ka is not None and kb is not None:
        a = ka.mean(0) / np.linalg.norm(ka.mean(0))
        b = kb.mean(0) / np.linalg.norm(kb.mean(0))
        print(f"\n=== 4. do the two tasks share an axis? cosine = {a @ b:+.4f} ===")
        print("    (near zero => each task has its own register, no reuse)")
    print(f"\nwrote {out} (+ provenance sidecar)")


if __name__ == "__main__":
    main()
