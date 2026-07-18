"""Backfill `seq_len` into results/switch.csv and results/maxtask.csv.

OWNER: Data+Analysis
STATUS: new 2026-07-18. `run_counting.py` already records prompt length
    (`seq_len`) alongside every trajectory, which is what makes its
    length-partial correlation (`partial_spearman`) possible. `run_switch.py`
    and `run_maxtask.py` never did the same, so nobody has ever been able to
    tell a real depth-tracking effect on those two tasks apart from the same
    raw-length confound documented for counting (open question, narrative.md
    section 11).
TASK: reconstruct `seq_len` for every row of the two already-committed CSVs
    without re-extracting a single trajectory. This only needs the
    tokenizer, not the model: `make_switch_task`/`make_max_task` are pure
    functions of (n_ops, seed) with no hidden state, and both scripts'
    compute() loops are a deterministic `for n_ops in N_OPS: for s in
    range(N_SEEDS)` — the exact same order the existing CSVs were written
    in. Regenerating that loop and tokenizing each prompt reproduces the
    length of every existing row exactly, with no GPU.
I/O: run as `python -m scripts.backfill_seq_len`. Adds a `seq_len` column to
    results/switch.csv and results/maxtask.csv in place; every other column
    (winding, steps_settle) is untouched. Checks the regenerated n_ops
    sequence against the cached CSV's own n_ops column position-by-position
    (not just a row count) before assigning anything, so a reordering —
    not just a length mismatch — would also be caught.

Run: uv run python -m scripts.backfill_seq_len
"""

from __future__ import annotations

import pandas as pd
from transformers import AutoTokenizer

from scripts._common import RESULTS_DIR
from traj_geom.constants import MODEL_ID, MODEL_REVISION
from traj_geom.shapes.synthetic import make_max_task, make_switch_task

TASKS = {
    "switch.csv": (make_switch_task, (4, 8, 16, 24, 32, 48), 10),
    "maxtask.csv": (make_max_task, (4, 8, 16, 24, 32, 48), 8),
}


def backfill(tok, csv_name: str, make_task, n_ops_levels: tuple[int, ...], n_seeds: int) -> None:
    """Regenerate the exact (n_ops, seed) prompt sequence and add seq_len."""
    path = RESULTS_DIR / csv_name
    df = pd.read_csv(path)

    expected_n_ops = [n_ops for n_ops in n_ops_levels for _ in range(n_seeds)]
    seq_lens = []
    for n_ops in n_ops_levels:
        for s in range(n_seeds):
            prompt = make_task(n_ops, seed=s)["prompt"]
            seq_lens.append(int(tok(prompt, return_tensors="pt").input_ids.shape[1]))

    if expected_n_ops != df["n_ops"].tolist():
        raise ValueError(
            f"{csv_name}: regenerated n_ops sequence doesn't match the cached CSV's "
            f"own n_ops column, position by position — loop order/params have "
            f"drifted out of sync, refusing to silently mislabel data."
        )
    # Insert right after n_ops, matching compute()'s row-dict column order.
    df.insert(df.columns.get_loc("n_ops") + 1, "seq_len", seq_lens)
    df.to_csv(path, index=False)
    print(f"backfilled seq_len into {path} ({len(df)} rows)")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    for csv_name, (make_task, n_ops_levels, n_seeds) in TASKS.items():
        backfill(tok, csv_name, make_task, n_ops_levels, n_seeds)


if __name__ == "__main__":
    main()
