"""Experiment — is the answer linearly decodable from the converged state?

OWNER: Data+Analysis
STATUS: implemented and run 2026-07-26. Result is SUGGESTIVE, NOT ESTABLISHED
    (see VERDICT below). Do not cite it as a finding.
TASK: probe the converged answer-token state for the task's answer, with the
    prompt-length confound removed by construction.

WHY THIS IS THE RIGHT FORM OF THE QUESTION
  `claims_ledger.md` D26(2) reports a null: converged states do not cluster by
  answer under a nearest-neighbour geometric test. But that test had near-zero
  power (1-3 same-answer pairs per stratum) and only detects information that
  happens to be visible as proximity. A linear probe is far more sensitive: it
  can find a direction carrying the answer even when nearest-neighbour
  structure is dominated by other variance. The two are not in conflict.

THE CONFOUND, AND HOW IT IS REMOVED
  A naive probe on the raw converged states gives leave-one-out R2 = 0.94
  (count_ones) and 0.81 (projection), both p < 0.005. That number is NOT
  evidence about computation: the answer grows with `n_ops`, and
  rank-corr(n_ops, seq_len) = exactly 1.000 for every generator
  (docs/rigor_audit.md section 21), so the probe may simply be reading prompt
  length off the state.

  The fix is to ask whether the state predicts the answer's variation WITHIN a
  difficulty level, where `seq_len` is constant. Both the target and the
  features are residualised on the level.

  !! THE LEAKAGE TRAP THIS AVOIDS. Residualising with level means computed over
  the WHOLE dataset destroys the test: within a level of size k the residuals
  sum to zero, so a held-out residual is exactly determined by the other k-1,
  and any model that can identify the level reconstructs it perfectly. That
  gives LOO R2 ~ 0.95 for REAL AND PERMUTED labels alike -- which is how the
  bug was caught here (a permutation null must not score 0.946). Level means
  are therefore fitted inside each fold, on training data only.

  The permutation null shuffles answers WITHIN level, so it preserves the
  n_ops-answer relationship and destroys only the within-level association.

VERDICT (2026-07-26)
  count_ones: within-level LOO R2 = +0.219, permutation p = 0.020, and
  Bonferroni across the two tasks gives 0.040 -- nominally significant.
  Robust to the ridge penalty across three orders of magnitude (p < 0.05 for
  alpha in 1e1..1e4). BUT fragile to leave-one-level-out: dropping n_ops=24
  alone takes R2 from +0.219 to -0.009. projection is null (R2 = -0.645).

  With N=38 and a modest effect on a small, fragile sample, this is the
  project's most promising positive signal and warrants a properly powered
  replication -- not a claim.

I/O: -> results/answer_probe.csv, with a provenance sidecar.

Run: uv run python -m scripts.run_answer_probe
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from traj_geom.provenance import save_table  # noqa: E402
from traj_geom.shapes.synthetic import (  # noqa: E402
    make_count_ones_task,
    make_projection_task,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ_DIR = os.path.join(ROOT, "results", "trajectories")
GENERATORS = {"count_ones": make_count_ones_task, "projection": make_projection_task}
ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)
N_PERM = 200
N_TAIL = 20
_FNAME_RE = re.compile(
    r"^(?P<task>[a-z_]+)_n(?P<n_ops>\d+)_ns(?P<num_steps>\d+)_init(?P<seed>\d+)\.npy$"
)


def _loo_r2(x: np.ndarray, y: np.ndarray, level: np.ndarray, alpha: float) -> float:
    """Leave-one-out R2 with level means fitted on TRAINING data only.

    Fitting them on all data leaks — see the module docstring.
    """
    n = len(y)
    pred = np.empty(n)
    targ = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        lv_tr, y_tr, x_tr = level[tr], y[tr], x[tr]
        ymean = {level_: y_tr[lv_tr == level_].mean() for level_ in np.unique(lv_tr)}
        xmean = {level_: x_tr[lv_tr == level_].mean(0) for level_ in np.unique(lv_tr)}
        if level[i] not in ymean:
            pred[i] = targ[i] = 0.0
            continue
        y_res = y_tr - np.array([ymean[level_] for level_ in lv_tr])
        x_res = x_tr - np.stack([xmean[level_] for level_ in lv_tr])
        targ[i] = y[i] - ymean[level[i]]
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x_res, y_res)
        pred[i] = model.predict((x[i] - xmean[level[i]]).reshape(1, -1))[0]
    denom = ((targ - targ.mean()) ** 2).sum()
    return float(1 - ((targ - pred) ** 2).sum() / denom) if denom > 0 else float("nan")


def _load(task: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(TRAJ_DIR, f"{task}*ns128*.npy"))):
        m = _FNAME_RE.match(os.path.basename(path))
        if m is None:
            continue
        t = GENERATORS[task](int(m.group("n_ops")), seed=int(m.group("seed")))
        rows.append(
            {
                "n_ops": int(m.group("n_ops")),
                "answer": float(t["answer"]),
                "prompt": t["prompt"],
                "h": np.load(path).astype(np.float64)[-N_TAIL:].mean(0),
            }
        )
    # drop prompt collisions: identical prompts give bit-identical states and
    # would enter the probe as duplicated rows
    return pd.DataFrame(rows).drop_duplicates("prompt")


def compute() -> pd.DataFrame:
    """Within-level probe per task, swept over the ridge penalty."""
    rng = np.random.default_rng(0)
    out = []
    for task in GENERATORS:
        g = _load(task)
        if len(g) < 10:
            continue
        x = np.stack(g.h.values)
        y = g.answer.values
        level = g.n_ops.values
        for alpha in ALPHAS:
            r2 = _loo_r2(x, y, level, alpha)
            null = np.empty(N_PERM)
            for k in range(N_PERM):
                yy = y.copy()
                for lv in np.unique(level):
                    idx = np.where(level == lv)[0]
                    yy[idx] = rng.permutation(y[idx])
                null[k] = _loo_r2(x, yy, level, alpha)
            out.append(
                {
                    "task": task, "n": len(g), "alpha": alpha, "r2": r2,
                    "null_mean": float(null.mean()),
                    "null_p95": float(np.percentile(null, 95)),
                    "p_perm": float(np.mean(null >= r2)),
                    "within_level_answer_sd": float(
                        np.std(y - pd.Series(y).groupby(level).transform("mean").values)
                    ),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    """Report the sweep and state the verdict, caveats included."""
    df = compute()
    out = os.path.join(ROOT, "results", "answer_probe.csv")
    save_table(
        out, df, kind="table", experiment="answer_probe", n_permutations=N_PERM,
        design="within-level (n_ops fixed => seq_len fixed); level means fitted "
               "inside each LOO fold to avoid the sum-to-zero leakage; "
               "permutation shuffles answers within level",
        status="SUGGESTIVE, NOT ESTABLISHED — fragile to leave-one-level-out",
    )
    print("\n=== answer decodable from the converged state, within difficulty level ===\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n  A permutation null must score WELL BELOW zero here. If null_mean is "
          "near +0.9, the level-mean leakage has returned.")
    for task, g in df.groupby("task"):
        best = g.loc[g.p_perm.idxmin()]
        print(f"  {task:11s} best alpha={best.alpha:.0e}  R2={best.r2:+.4f}  p={best.p_perm:.3f}"
              f"  Bonferroni(2 tasks)={min(1.0, best.p_perm * 2):.3f}")
    print("\n  VERDICT: suggestive only. See this module's docstring for the "
          "leave-one-level-out fragility.")
    print(f"\nwrote {out} (+ provenance sidecar)")


if __name__ == "__main__":
    main()
