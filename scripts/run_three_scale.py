"""Experiment V6 — Winding vs Three Independent Length Scales.

Resolves the collinearity problem of counting.csv. Independently scales:
- irrelevant_len (padding text)
- neutral_len (zeros, processable but state-neutral)
- active_len (ones, forces state change)

Run: python -m scripts.run_three_scale
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model
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
    # Using the standard extraction hook to avoid the massive V6 logit-extraction overhead,
    # reserving V6 hook for targeted accuracy checks.
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []
    
    total_iters = len(ACTIVE_LENS) * len(NEUTRAL_LENS) * len(IRRELEVANT_LENS) * N_SEEDS
    
    with tqdm(total=total_iters, desc="3-scale sweeps") as pbar:
        for act in ACTIVE_LENS:
            for neu in NEUTRAL_LENS:
                for irr in IRRELEVANT_LENS:
                    for s in range(N_SEEDS):
                        task = make_three_scale_task(
                            irrelevant_len=irr, 
                            neutral_len=neu, 
                            active_len=act, 
                            seed=s
                        )
                        # We use 64 steps to match the main branch default
                        tr = extract_trajectory(model, tok, task["prompt"], num_steps=64, seed=0)
                        
                        rows.append(
                            {
                                "active_len": act,
                                "neutral_len": neu,
                                "irrelevant_len": irr,
                                "seq_len": int(tok(task["prompt"], return_tensors="pt").input_ids.shape[1]),
                                "winding": abs(winding_of(tr, burn=4)),
                                "steps_settle": steps_to_settle(tr),
                                "answer_target": task["answer"]
                            }
                        )
                        pbar.update(1)
                        
    return pd.DataFrame(rows)


def main() -> None:
    """Report the dissociation of winding from raw length."""
    cdf = cached("three_scale.csv", compute)
    
    print("\n--- Correlation Analysis (Spearman rho) ---")
    
    # 1. Active length (True reasoning depth)
    rho_active = spearman(cdf["active_len"], cdf["winding"])
    print(f"|winding| ~ active_len      : {rho_active:>6.3f} (Expect high +)")
    
    # 2. Neutral length (Processing without state change)
    rho_neutral = spearman(cdf["neutral_len"], cdf["winding"])
    print(f"|winding| ~ neutral_len     : {rho_neutral:>6.3f} (Expect near 0)")
    
    # 3. Irrelevant length (Raw context padding)
    rho_irr = spearman(cdf["irrelevant_len"], cdf["winding"])
    print(f"|winding| ~ irrelevant_len  : {rho_irr:>6.3f} (Expect near 0)")

    print("\n--- Sanity Check on Raw Sequence Length ---")
    rho_seq = spearman(cdf["seq_len"], cdf["winding"])
    print(f"|winding| ~ seq_len (total) : {rho_seq:>6.3f} (The confounded metric)")


if __name__ == "__main__":
    main()
