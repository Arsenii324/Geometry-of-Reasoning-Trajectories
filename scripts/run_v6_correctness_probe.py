"""Probe how many recurrent unrolls it takes for the correct answer to become
the argmax prediction, per depth.

OWNER: Data+Analysis
STATUS: implemented — depends on `extract_trajectory(..., return_logits=True)`
    in `extraction/hook.py`, whose logit reconstruction is self-validated
    against a real forward() call on every use (see that module's GOTCHAS).
TASK: for each (depth, seed), extract the per-unroll logit lens over the
    counting task and find the first unroll whose argmax matches the correct
    answer's *first* token.
I/O: -> results/v6_correctness_probe.csv (depth, seed, correct_at_step,
    topk_correct_at_step, target, n_answer_tokens, is_single_token_answer).

CAVEAT: this only checks the FIRST token of the answer. A single recurrent
unroll's logit lens predicts one next-token distribution, not a full
multi-token continuation — checking "would greedy decoding from here get the
whole answer right" would need a further generation step per unroll checked,
which this script deliberately does not do (cost). Do not read
`correct_at_step` as "the model's full answer is right from here on."

FIX 2026-07-24 (project_plan.md §0.5, D12): the first-token-only check was
silently MISLEADING for negative targets, not just incomplete. Checking only
the leading token of a negative answer means checking only the "-" sign
token — every negative value from -1 to -99 shares that same leading token,
so a "correct" leading-token match confirms nothing about the actual value.
Confirmed live: 11/24 of D12's own rows have negative targets. Two changes:
(1) `is_single_token_answer` records whether the FULL answer (not just its
leading token) is one token — only these rows (0-9) get a first-token check
that actually verifies the whole answer; multi-token rows (negative or >=10)
are marked, not silently scored as if fully checked. (2) `topk_correct_at_step`
(top-5 membership, not strict argmax) is recorded alongside `correct_at_step`
as a softer, additional diagnostic — the model can be "close" (target in
top-5, not top-1) without registering under the strict criterion.

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
TOP_K = 5
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
                is_single_token_answer = len(ans_ids) == 1

                out = extract_trajectory(
                    model, tok, task["prompt"], num_steps=NUM_STEPS, seed=0, return_logits=True
                )
                logits = out.get("logits")
                if logits is None:
                    raise RuntimeError("no logits returned from extract_trajectory")

                correct_at_step = -1
                topk_correct_at_step = -1
                for step in range(len(logits)):
                    step_logits = logits[step]
                    if correct_at_step == -1 and step_logits.argmax() == target_token_id:
                        correct_at_step = step
                    if topk_correct_at_step == -1:
                        topk_ids = step_logits.topk(TOP_K).indices.tolist()
                        if target_token_id in topk_ids:
                            topk_correct_at_step = step
                    if correct_at_step != -1 and topk_correct_at_step != -1:
                        break

                rows.append(
                    {
                        "depth": d,
                        "seed": seed,
                        "correct_at_step": correct_at_step,
                        "topk_correct_at_step": topk_correct_at_step,
                        "target": ans,
                        "n_answer_tokens": len(ans_ids),
                        "is_single_token_answer": is_single_token_answer,
                    }
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
    """Report correctness-timing per depth -- strict argmax AND top-K, with
    the single-token-answer distinction made explicit rather than silently
    conflating "fully verified" with "leading token matched."
    """
    df = cached(_RESULTS_NAME, compute)
    if df.empty:
        # Every (depth, seed) failed -- compute() already printed why per
        # config. Report that plainly instead of crashing on an empty
        # DataFrame with no "depth" column to group by (this is exactly
        # what happened on real hardware 2026-07-23, before the coda-skip
        # reconstruction's missing pre-coda ln_f was found and fixed).
        print(f"{_RESULTS_NAME}: 0 usable rows -- every config failed, nothing to report.")
        return

    if "is_single_token_answer" not in df.columns:
        print(
            f"{_RESULTS_NAME}: cached copy predates the top-k/single-token-answer "
            "fix (project_plan.md §0.5) -- delete and re-run for the corrected columns."
        )
        print(df.groupby("depth")["correct_at_step"].agg(["mean", lambda s: (s >= 0).mean()]))
        return

    single = df[df["is_single_token_answer"]]
    multi = df[~df["is_single_token_answer"]]
    print(
        f"{len(single)}/{len(df)} rows have a single-token answer (0-9) -- only "
        "these are fully verified by a leading-token check. "
        f"{len(multi)}/{len(df)} rows have a multi-token answer (negative or >=10) "
        "-- correct_at_step there only confirms the SIGN or leading digit, not "
        "the full value; do not read it as full-answer correctness."
    )

    print("\n--- Single-token-answer rows only (the honest subset) ---")
    if len(single):
        print(
            single.groupby("depth")[["correct_at_step", "topk_correct_at_step"]].agg(
                lambda s: (s >= 0).mean()
            )
        )
    else:
        print("(none -- every row in this run had a multi-token answer)")

    print(f"\n--- All rows, strict argmax vs top-{TOP_K} membership (leading token only) ---")
    print(
        df.groupby("depth")[["correct_at_step", "topk_correct_at_step"]].agg(
            lambda s: (s >= 0).mean()
        )
    )


if __name__ == "__main__":
    main()
