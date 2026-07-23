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
from traj_geom.analysis.correlate import spearman
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
    """Report the dissociation of winding from raw length."""
    cdf = cached("three_scale.csv", compute)

    print("\n--- Correlation Analysis (Spearman rho) ---")

    rho_active = spearman(cdf["active_len"], cdf["winding"])
    print(f"|winding| ~ active_len      : {rho_active:>6.3f} (expect high +)")

    rho_neutral = spearman(cdf["neutral_len"], cdf["winding"])
    print(f"|winding| ~ neutral_len     : {rho_neutral:>6.3f} (expect near 0)")

    rho_irr = spearman(cdf["irrelevant_len"], cdf["winding"])
    print(f"|winding| ~ irrelevant_len  : {rho_irr:>6.3f} (expect near 0)")

    print("\n--- Sanity check on raw sequence length ---")
    rho_seq = spearman(cdf["seq_len"], cdf["winding"])
    print(f"|winding| ~ seq_len (total) : {rho_seq:>6.3f} (the confounded metric)")


if __name__ == "__main__":
    main()
