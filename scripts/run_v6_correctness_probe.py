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

BUG FOUND AND FIXED 2026-07-23, via scripts/diag_v6_token_gap.py: the prompt
ends "...A:" with no trailing space, so the model's real next-token
continuation is a SPACE-PREFIXED token (" 2", not "2") -- confirmed
directly: for a depth=2 answer of "2", the bare token '2' sat at rank 27
(prob 0.0023) while the space-prefixed ' 2' was in the top-5 (prob ~0.065),
an order of magnitude more likely. Every prior run of this script computed
`target_token_id` from the BARE answer string and found correct_at_step=-1
for literally every depth/seed as a direct consequence -- not a finding
about the model's ability, an artifact of checking the wrong token ID. Fixed
by tokenizing `" " + ans` instead of `ans`. Any v6_correctness_probe.csv
predating this fix is invalid for the same reason the coda-skip bug
invalidated earlier copies -- check `architecture_state.md`'s verification
practice / D12 before trusting a cached copy of this file.

Run: uv run python -m scripts.run_v6_correctness_probe
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model, save_partial
from traj_geom.extraction.hook import extract_trajectory
from traj_geom.shapes.synthetic import make_counting_task

DEPTHS = (2, 4, 6, 8, 10, 12, 14, 16)
N_SEEDS = 3
NUM_STEPS = 64
_RESULTS_NAME = "v6_correctness_probe.csv"


def compute() -> pd.DataFrame:
    """Run the probe across depths/seeds; skip (not crash on) a bad extraction."""
    model, tok = load_model()
    rows = []
    n_failed = 0

    for d in tqdm(DEPTHS, desc="depth"):
        for seed in range(N_SEEDS):
            try:
                task = make_counting_task(n_ops=d, seed=seed)
                ans = str(task["answer"])
                # Space-prefixed, not bare: the prompt ends "...A:" with no
                # trailing space, so the model's real continuation tokenizes
                # as " 2", not "2" -- see module CAVEAT, found via
                # diag_v6_token_gap.py. First token of THAT tokenization, not
                # the last — the model predicts the leading token first (e.g.
                # " -" before "2" for "-2"); using the last token would
                # silently ignore the sign.
                ans_ids = tok.encode(" " + ans, add_special_tokens=False)
                if not ans_ids:
                    raise ValueError(f"answer {ans!r} tokenized to zero tokens")
                target_token_id = ans_ids[0]

                out = extract_trajectory(
                    model, tok, task["prompt"], num_steps=NUM_STEPS, seed=0, return_logits=True
                )
                logits = out.get("logits")
                if logits is None:
                    raise RuntimeError("no logits returned from extract_trajectory")

                correct_at_step = -1
                for step in range(len(logits)):
                    if logits[step].argmax() == target_token_id:
                        correct_at_step = step
                        break

                rows.append(
                    {"depth": d, "seed": seed, "correct_at_step": correct_at_step, "target": ans}
                )
            except Exception as e:  # noqa: BLE001 -- a single bad (depth, seed) must
                # not lose the rest of this multi-hour GPU sweep's already-completed rows.
                n_failed += 1
                print(f"depth={d} seed={seed}: skipping after error: {e!r}")
            else:
                save_partial(rows, _RESULTS_NAME)

    if n_failed:
        print(f"run_v6_correctness_probe: {n_failed}/{len(DEPTHS) * N_SEEDS} configs failed.")
    return pd.DataFrame(rows)


def main() -> None:
    """Report correctness-timing per depth."""
    df = cached(_RESULTS_NAME, compute)
    if df.empty:
        # Every (depth, seed) failed -- compute() already printed why per
        # config. Report that plainly instead of crashing on an empty
        # DataFrame with no "depth" column to group by (this is exactly
        # what happened on real hardware 2026-07-23, before the coda-skip
        # reconstruction's missing pre-coda ln_f was found and fixed).
        print(f"{_RESULTS_NAME}: 0 usable rows -- every config failed, nothing to report.")
        return
    print(df.groupby("depth")["correct_at_step"].agg(["mean", lambda s: (s >= 0).mean()]))


if __name__ == "__main__":
    main()
