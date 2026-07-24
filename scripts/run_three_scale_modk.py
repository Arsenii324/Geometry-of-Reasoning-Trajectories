"""Experiment — winding vs length-decoupled MODULUS counting.

OWNER: Data+Analysis
STATUS: implemented 2026-07-24. `make_three_scale_modk_task` fixes D11's
    prefix confound by construction (constant total_len, no distinguished
    prefix region) rather than approximately, and generalises
    `make_switch_task`'s mod-2 parity to modulus 2..10.
TASK: sweep (active_len, irrelevant_len, modulus, seed) at FIXED total_len
    and score each trajectory's geometry, so winding can be checked
    against active_len with total length and answer position held
    genuinely constant -- the actual clean H2 test three_scale was
    supposed to be (project_plan.md §9).
I/O: -> results/three_scale_modk.csv (active_len, irrelevant_len,
    neutral_len, total_len, modulus, seq_len, winding, steps_settle,
    answer_target).

Run: uv run python -m scripts.run_three_scale_modk
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model, save_partial
from traj_geom.analysis.correlate import fmt_by_level, spearman
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.synthetic import make_three_scale_modk_task

# Fixed total length across the whole sweep -- the actual property this
# task exists to guarantee. active_len + irrelevant_len must stay
# <= TOTAL_LEN for every combo below (18+6=24=TOTAL_LEN at the extreme).
TOTAL_LEN = 24
ACTIVE_LENS = (0, 3, 6, 9, 12, 15, 18)
IRRELEVANT_LENS = (0, 3, 6)
MODULI = (2, 5)
N_SEEDS = 3


def compute() -> pd.DataFrame:
    """Extract a trajectory per modk-task config and score its geometry."""
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []

    total_iters = len(ACTIVE_LENS) * len(IRRELEVANT_LENS) * len(MODULI) * N_SEEDS

    n_failed = 0
    with tqdm(total=total_iters, desc="modk sweeps") as pbar:
        for act in ACTIVE_LENS:
            for irr in IRRELEVANT_LENS:
                for k in MODULI:
                    for s in range(N_SEEDS):
                        try:
                            task = make_three_scale_modk_task(
                                active_len=act,
                                irrelevant_len=irr,
                                total_len=TOTAL_LEN,
                                modulus=k,
                                seed=s,
                            )
                            tr = extract_trajectory(
                                model, tok, task["prompt"], num_steps=64, seed=0
                            )
                            seq_len = int(
                                tok(task["prompt"], return_tensors="pt").input_ids.shape[1]
                            )
                            rows.append(
                                {
                                    "active_len": act,
                                    "irrelevant_len": irr,
                                    "neutral_len": task["neutral_len"],
                                    "total_len": TOTAL_LEN,
                                    "modulus": k,
                                    "seq_len": seq_len,
                                    "winding": abs(winding_of(tr, burn=4)),
                                    "steps_settle": steps_to_settle(tr),
                                    "answer_target": task["answer"],
                                }
                            )
                        except Exception as e:  # noqa: BLE001 -- a single bad config
                            # must not lose the rest of this sweep's already-completed
                            # extractions.
                            n_failed += 1
                            print(
                                f"\nrun_three_scale_modk: skipping act={act} irr={irr} "
                                f"k={k} seed={s} after error: {e!r}"
                            )
                        else:
                            save_partial(rows, "three_scale_modk.csv")
                        pbar.update(1)

    if n_failed:
        print(f"run_three_scale_modk: {n_failed}/{total_iters} configs failed and skipped.")
    return pd.DataFrame(rows)


def main() -> None:
    """Report winding vs active_len at constant total_len -- the first
    genuinely clean H2 test this project will have run (project_plan.md
    §9's own prescribed fix, not an approximation of it).
    """
    cdf = cached("three_scale_modk.csv", compute)

    print("\n--- Sanity: seq_len must be ~constant across the whole sweep ---")
    print(cdf["seq_len"].describe())

    print("\n--- Per-row Spearman rho, pooled across modulus ---")
    rho, p = spearman(cdf["active_len"], cdf["winding"])
    print(f"|winding| ~ active_len (all modk pooled): {rho:>6.3f} (p={p:.2g})")

    print("\n--- Canonical per-level Spearman, split by modulus ---")
    for k in MODULI:
        sub = cdf[cdf["modulus"] == k]
        print(f"modulus={k}: winding~active_len", fmt_by_level(sub, "active_len", "winding"))
        print(
            f"modulus={k}: steps~active_len   ",
            fmt_by_level(sub, "active_len", "steps_settle"),
        )

    print(
        "\nNOTE: total_len is held constant by construction across this whole "
        "sweep (neutral_len absorbs the remainder), so unlike three_scale.csv "
        "(claims_ledger.md D11), any winding~active_len signal here is NOT "
        "confounded with total sequence length or answer-token position."
    )


if __name__ == "__main__":
    main()
