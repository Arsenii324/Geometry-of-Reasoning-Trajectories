"""Experiment — does an instance become solvable at a difficulty-dependent depth?

OWNER: Data+Analysis
STATUS: implemented 2026-07-25. NEEDS GPU. Not yet run; no result is claimed
    anywhere from this script until it has been.
TASK: measure, per instance, the recurrent depth r* at which the model starts
    preferring the correct answer, and test whether r* scales with difficulty.

THE HYPOTHESIS, STATED SO IT CAN FAIL
  H2-depth: an instance requiring more operations needs more recurrent unrolls
  before the model can answer it, so r*(n_ops) increases with n_ops.
  Null: r* is independent of n_ops (flat), or undefined because the model
  never prefers the correct answer at any depth.

  This is the same underlying claim as the project's original H2 ("winding
  tracks reasoning depth") but operationalised behaviourally instead of
  geometrically. docs/rigor_audit.md showed the geometric route was measuring
  the contraction rate and arithmetic noise; this route touches neither.

WHY THE EXISTING PROBE IS NOT ENOUGH
  `run_v6_correctness_probe.py` asks a related question, but its 24-row
  output cannot answer this one:
    * it scores the ARGMAX of the FIRST answer token, so only its 13
      single-token rows are fully valid (its own docstring says so);
    * among those 13, correctness is perfectly separated by whether the answer
      happens to be 0 -- 4/5 correct when target == 0 versus 0/8 otherwise --
      and all four "successes" occur at unroll 1. That is a prior on emitting
      zero, not a computation finishing;
    * binary correctness has no resolution for locating a threshold.

  This script fixes all three: a continuous teacher-forced log-probability, a
  distractor control that cancels answer-string priors, and instances with
  answer 0 excluded outright.

DESIGN CHOICES, AND THE REASONING
  * EXCLUDE answer == 0. Established above as the degenerate case. Recorded in
    the CSV as `excluded_zero` so the exclusion is visible, not silent.
  * DISTRACTOR = answer +/- 1, whichever is non-zero and has the same token
    length as the gold answer where possible. Same length keeps the two
    log-probs term-comparable; +/-1 keeps the distractor plausible, so the
    margin measures discrimination rather than the gap to an absurd string.
    `distractor_same_len` records whether length matching succeeded.
  * MARGIN, not raw log P. A model growing uniformly more confident with depth
    would lift log P(gold) with no computation; it lifts the distractor
    equally, so the margin is flat. See tests/test_eval_depth.py.
  * FIXED unroll sweep, no adaptive exit. Huginn's `generate_with_adaptive_
    compute` stops early on a KL criterion, which would confound "the model
    needed depth r" with "the exit rule fired at r". rigor_audit section 10
    documents that the project's existing accuracy numbers and geometry
    numbers differ on exactly this axis.

I/O: -> results/depth_threshold.csv, one row per (task, n_ops, task_seed),
    plus a provenance sidecar and manifest entry via traj_geom.provenance.

Run: uv run python -m scripts.run_depth_threshold
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from traj_geom.eval_depth import answer_token_ids, depth_curve, threshold_depth  # noqa: E402
from traj_geom.provenance import save_table  # noqa: E402
from traj_geom.shapes.synthetic import (  # noqa: E402
    make_count_ones_task,
    make_projection_task,
)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TASKS = {"count_ones": make_count_ones_task, "projection": make_projection_task}
N_OPS = (2, 4, 8, 16, 24, 32, 48, 64)
N_SEEDS = 10
MAX_R = 64


def _pick_distractor(tok, answer: int) -> tuple[int, bool]:
    """A plausible wrong answer, token-length-matched to the gold one if possible.

    Returns:
        ``(distractor, same_length)``.
    """
    gold_len = len(answer_token_ids(tok, answer))
    for cand in (answer + 1, answer - 1, answer + 2, answer - 2):
        if cand == 0 or cand == answer:
            continue
        if len(answer_token_ids(tok, cand)) == gold_len:
            return cand, True
    fallback = answer + 1 if answer + 1 != 0 else answer + 2
    return fallback, False


def compute() -> pd.DataFrame:
    """Sweep depth for every instance and locate its threshold."""
    from traj_geom.extraction.model import load_huginn

    model, tok = load_huginn()
    rows = []
    for task, gen in TASKS.items():
        for n_ops in N_OPS:
            for seed in range(N_SEEDS):
                t = gen(n_ops, seed=seed)
                answer = t["answer"]
                if answer == 0:
                    rows.append(
                        {
                            "task": task, "n_ops": n_ops, "task_seed": seed,
                            "answer": answer, "excluded_zero": True,
                            "r_star": np.nan, "final_margin": np.nan,
                            "distractor": np.nan, "distractor_same_len": np.nan,
                            "n_answer_tokens": np.nan,
                        }
                    )
                    continue

                distractor, same_len = _pick_distractor(tok, answer)
                cur = depth_curve(model, tok, t["prompt"], answer, distractor, max_r=MAX_R)
                margin = cur["margin"]
                rows.append(
                    {
                        "task": task, "n_ops": n_ops, "task_seed": seed,
                        "answer": answer, "excluded_zero": False,
                        "r_star": threshold_depth(margin),
                        "final_margin": float(np.nanmedian(margin[-MAX_R // 4:])),
                        "distractor": distractor,
                        "distractor_same_len": same_len,
                        "n_answer_tokens": len(answer_token_ids(tok, answer)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the sweep and report r* against difficulty, per task."""
    from scipy.stats import spearmanr

    df = compute()
    out = os.path.join(RESULTS, "depth_threshold.csv")
    save_table(
        out, df, kind="table",
        experiment="depth_threshold",
        max_r=MAX_R, n_seeds=N_SEEDS, n_ops=list(N_OPS),
        metric="teacher-forced log P(gold) - log P(distractor), threshold at 90% of tail median",
        excludes="instances with answer == 0 (degenerate zero-prior, see docstring)",
        adaptive_compute=False,
    )

    solved = df[(~df.excluded_zero) & df.r_star.notna()]
    print(f"\ninstances: {len(df)}  excluded (answer==0): {int(df.excluded_zero.sum())}  "
          f"with a threshold: {len(solved)}")
    if solved.empty:
        print("No instance ever preferred the gold answer at any depth -> "
              "H2-depth is untestable on this task set, not refuted.")
        return

    print("\nr* by difficulty:")
    print(solved.groupby(["task", "n_ops"])["r_star"].agg(["mean", "std", "count"]).to_string())
    print("\nTHE TEST -- does required depth scale with difficulty?")
    for task in sorted(solved.task.unique()):
        s = solved[solved.task == task]
        g = s.groupby("n_ops")["r_star"].mean()
        if len(g) > 2:
            rho, p = spearmanr(g.index, g.values)
            print(f"  {task:12s} rho(r*, n_ops) = {rho:+.3f}  p={p:.4f}  (N={len(g)} levels)")
        else:
            print(f"  {task:12s} too few levels with a threshold")
    print(f"\nwrote {out} (+ provenance sidecar)")


if __name__ == "__main__":
    main()
