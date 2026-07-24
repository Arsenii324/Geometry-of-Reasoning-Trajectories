"""Experiment V6 — winding vs three independent length scales.

OWNER: Data+Analysis
STATUS: implemented (2026-07-19) — the one synthetic task that genuinely
    decouples length from difficulty; see `make_three_scale_task`'s docstring.
TASK: sweep (active_len, neutral_len, irrelevant_len, seed) and score each
    trajectory's geometry, so winding can be partialled against each length
    scale independently instead of the single confounded `n_ops`/`seq_len`
    every other synthetic task has.
I/O: -> results/three_scale.csv (active_len, neutral_len, irrelevant_len,
    seq_len, winding, steps_settle, answer_target).

Run: uv run python -m scripts.run_three_scale
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model, save_partial
from traj_geom.analysis.correlate import fmt_by_level, multivariate_rank_control, spearman
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.synthetic import make_three_scale_task

# Active lengths must stay <= 9 so the answer is a single digit token.
ACTIVE_LENS = (1, 3, 5, 7, 9)
NEUTRAL_LENS = (0, 3, 6, 9)
IRRELEVANT_LENS = (0, 5, 10)
N_SEEDS = 3


def compute() -> pd.DataFrame:
    """Extract a trajectory per 3-scale task config and score its geometry."""
    # Standard extraction hook (return_logits=False) — the V6 logit-extraction
    # path is reserved for targeted accuracy checks (run_v6_correctness_probe.py).
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []

    total_iters = len(ACTIVE_LENS) * len(NEUTRAL_LENS) * len(IRRELEVANT_LENS) * N_SEEDS

    n_failed = 0
    with tqdm(total=total_iters, desc="3-scale sweeps") as pbar:
        for act in ACTIVE_LENS:
            for neu in NEUTRAL_LENS:
                for irr in IRRELEVANT_LENS:
                    for s in range(N_SEEDS):
                        try:
                            task = make_three_scale_task(
                                irrelevant_len=irr, neutral_len=neu, active_len=act, seed=s
                            )
                            # 64 steps to match the main branch default.
                            tr = extract_trajectory(
                                model, tok, task["prompt"], num_steps=64, seed=0
                            )
                            seq_len = int(
                                tok(task["prompt"], return_tensors="pt").input_ids.shape[1]
                            )
                            rows.append(
                                {
                                    "active_len": act,
                                    "neutral_len": neu,
                                    "irrelevant_len": irr,
                                    "seq_len": seq_len,
                                    "winding": abs(winding_of(tr, burn=4)),
                                    "steps_settle": steps_to_settle(tr),
                                    "answer_target": task["answer"],
                                }
                            )
                        except Exception as e:  # noqa: BLE001 -- a single bad config
                            # (OOM, tokenizer edge case) must not lose the other ~180
                            # already-completed extractions on a multi-hour GPU sweep.
                            n_failed += 1
                            print(
                                f"\nrun_three_scale: skipping act={act} neu={neu} irr={irr} "
                                f"seed={s} after error: {e!r}"
                            )
                        else:
                            save_partial(rows, "three_scale.csv")
                        pbar.update(1)

    if n_failed:
        print(f"run_three_scale: {n_failed}/{total_iters} configs failed and were skipped.")
    return pd.DataFrame(rows)


def main() -> None:
    """Report the dissociation of winding from raw length.

    Three statistics, in order of how much they're trusted (see
    claims_ledger.md D11):
    1. Per-row Spearman -- kept for continuity with the original design, but
       inflates N (repeated seeds per level) and is not the canonical stat.
    2. spearman_by_level -- the project's canonical per-level statistic
       (collapses to one mean per level first), for each length scale alone.
    3. multivariate_rank_control -- all three length scales controlled for
       *simultaneously*. This is the one that actually answers "does winding
       track a length scale after accounting for the other two" -- a
       single-confounder partial correlation on the composite seq_len instead
       reports active_len flipping to rho=+0.318, an artifact of partialling
       out the composite rather than the three components together.
    """
    cdf = cached("three_scale.csv", compute)

    print("\n--- Per-row Spearman rho (inflated N, not canonical) ---")
    rho_active, p_active = spearman(cdf["active_len"], cdf["winding"])
    print(f"|winding| ~ active_len      : {rho_active:>6.3f} (p={p_active:.2g})")
    rho_neutral, p_neutral = spearman(cdf["neutral_len"], cdf["winding"])
    print(f"|winding| ~ neutral_len     : {rho_neutral:>6.3f} (p={p_neutral:.2g})")
    rho_irr, p_irr = spearman(cdf["irrelevant_len"], cdf["winding"])
    print(f"|winding| ~ irrelevant_len  : {rho_irr:>6.3f} (p={p_irr:.2g})")
    rho_seq, p_seq = spearman(cdf["seq_len"], cdf["winding"])
    print(f"|winding| ~ seq_len (total) : {rho_seq:>6.3f} (p={p_seq:.2g})")

    print("\n--- Canonical per-level Spearman (spearman_by_level) ---")
    print("active_len     :", fmt_by_level(cdf, "active_len", "winding"))
    print("neutral_len    :", fmt_by_level(cdf, "neutral_len", "winding"))
    print("irrelevant_len :", fmt_by_level(cdf, "irrelevant_len", "winding"))
    print("seq_len        :", fmt_by_level(cdf, "seq_len", "winding"))

    print("\n--- Multivariate rank control (all three scales simultaneously) ---")
    mv = multivariate_rank_control(
        cdf["winding"].to_numpy(),
        {
            "active_len": cdf["active_len"].to_numpy(),
            "neutral_len": cdf["neutral_len"].to_numpy(),
            "irrelevant_len": cdf["irrelevant_len"].to_numpy(),
        },
    )
    for name, (beta, p) in mv.items():
        print(f"beta[{name:14s}] = {beta:+7.3f}  (p={p:.2g})")

    print(
        "\nNOTE (see claims_ledger.md D11, docs/project_plan.md §9): "
        "irrelevant_len is a PREFIX block -- it lengthens total context AND "
        "pushes the answer token to a later absolute position. The three "
        "scales are rank-orthogonal to each other, but each is monotone in "
        "total seq_len / answer-token position. So a negative irrelevant_len "
        "coefficient here is consistent with winding tracking total length "
        "or absolute answer position, not reasoning content. This is NOT a "
        "clean H2 test until the task is redesigned (constant total token "
        "count, constant answer position, vary only the active/neutral ratio)."
    )


if __name__ == "__main__":
    main()
