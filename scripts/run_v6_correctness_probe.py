"""Probe how many recurrent unrolls it takes for the correct answer to become
the argmax prediction, per depth.

OWNER: Data+Analysis
STATUS: implemented — depends on `extract_trajectory(..., return_logits=True)`
    in `extraction/hook.py`, whose logit reconstruction is self-validated
    against a real forward() call on every use (see that module's GOTCHAS).
TASK: for each (depth, seed), extract the per-unroll logit lens over the
    counting task and find the first unroll whose argmax matches the correct
    answer's *first* token.
I/O: -> results/v6_correctness_probe.csv (depth, seed, correct_at_step, target).

CAVEAT: this only checks the FIRST token of the answer. A single recurrent
unroll's logit lens predicts one next-token distribution, not a full
multi-token continuation — checking "would greedy decoding from here get the
whole answer right" would need a further generation step per unroll checked,
which this script deliberately does not do (cost). For multi-digit or
negative answers (e.g. "-2" -> ["-", "2"]), this checks whether the model
predicts the leading token ("-") correctly, not the full string. Do not read
`correct_at_step` as "the model's full answer is right from here on."

Run: uv run python -m scripts.run_v6_correctness_probe
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model
from traj_geom.extraction.hook import extract_trajectory
from traj_geom.shapes.synthetic import make_counting_task

DEPTHS = (2, 4, 6, 8, 10, 12, 14, 16)
N_SEEDS = 3
NUM_STEPS = 64


def compute() -> pd.DataFrame:
    """Run the probe across depths/seeds; skip (not crash on) a bad extraction."""
    model, tok = load_model()
    rows = []

    for d in tqdm(DEPTHS, desc="depth"):
        for seed in range(N_SEEDS):
            task = make_counting_task(n_ops=d, seed=seed)
            ans = str(task["answer"])
            # First token of the answer, not the last — the model predicts
            # the leading token first (e.g. "-" before "2" for "-2"); using
            # the last token silently ignores the sign.
            target_token_id = tok.encode(ans, add_special_tokens=False)[0]

            try:
                out = extract_trajectory(
                    model, tok, task["prompt"], num_steps=NUM_STEPS, seed=0, return_logits=True
                )
            except RuntimeError as e:
                print(f"depth={d} seed={seed}: extraction failed, skipping ({e})")
                continue

            logits = out.get("logits")
            if logits is None:
                print(f"depth={d} seed={seed}: no logits returned, skipping")
                continue

            correct_at_step = -1
            for step in range(len(logits)):
                if logits[step].argmax() == target_token_id:
                    correct_at_step = step
                    break

            rows.append(
                {"depth": d, "seed": seed, "correct_at_step": correct_at_step, "target": ans}
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Report correctness-timing per depth."""
    df = cached("v6_correctness_probe.csv", compute)
    print(df.groupby("depth")["correct_at_step"].agg(["mean", lambda s: (s >= 0).mean()]))


if __name__ == "__main__":
    main()
