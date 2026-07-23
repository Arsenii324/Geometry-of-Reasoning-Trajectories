"""Diagnostic: why does run_v6_correctness_probe.py find correct_at_step=-1
everywhere?

OWNER: Data+Analysis
STATUS: implemented 2026-07-23 — one-off diagnostic, not a scored experiment.
    Not wired into cached()/results/ by design: this is meant to be read by
    a human/AI, not aggregated into a CSV.
TASK: for a small, fixed set of counting-task prompts, print the top-5
    predicted tokens (with rank/probability) at the final few recurrent
    unrolls, alongside the actual target token. Answers a specific
    question the aggregate CSV can't: is the target token *close* (low
    rank, model "almost" right, just not argmax) or nowhere near (the
    model isn't tracking this at all)? Also prints what the model's
    argmax token actually decodes to, in case it's predicting something
    plausible-but-differently-tokenized (a leading space variant, etc.)
    that a strict token-ID match would silently miss.
I/O: no args -> printed diagnostic only, nothing saved.

Run: uv run python -m scripts.diag_v6_token_gap
"""

from __future__ import annotations

import torch

from scripts._common import load_model
from traj_geom.extraction.hook import extract_trajectory
from traj_geom.shapes.synthetic import make_counting_task

# Small, fixed set spanning the depth range already probed -- not a sweep,
# a handful of prompts to actually look at closely.
CASES = [(2, 0), (2, 1), (6, 1), (16, 0)]
NUM_STEPS = 64
LAST_N_STEPS = 5
TOP_K = 5


def main() -> None:
    """Dump top-K token predictions near the end of the unroll for a few prompts."""
    model, tok = load_model()

    for depth, seed in CASES:
        task = make_counting_task(n_ops=depth, seed=seed)
        ans = str(task["answer"])
        # Space-prefixed: the prompt ends "...A:" with no trailing space, so
        # the real continuation tokenizes as " 2", not "2" -- this is exactly
        # the bug this diagnostic found on its first (bare-token) run.
        ans_ids = tok.encode(" " + ans, add_special_tokens=False)
        target_id = ans_ids[0]
        target_str = tok.decode([target_id])

        print(
            f"\n=== depth={depth} seed={seed} target={ans!r} "
            f"(first token {target_id}={target_str!r}) ==="
        )
        print(f"prompt: {task['prompt']!r}")

        out = extract_trajectory(
            model, tok, task["prompt"], num_steps=NUM_STEPS, seed=0, return_logits=True
        )
        logits = out.get("logits")
        if logits is None:
            print("  no logits reconstructed (validate_logits failed) -- see error above")
            continue
        logits = torch.as_tensor(logits)

        for step in range(len(logits) - LAST_N_STEPS, len(logits)):
            step_logits = logits[step]
            probs = torch.softmax(step_logits, dim=-1)
            topk = torch.topk(probs, TOP_K)
            target_rank = int((probs > probs[target_id]).sum().item()) + 1
            target_prob = probs[target_id].item()

            top_str = ", ".join(
                f"{tok.decode([tid.item()])!r}={p.item():.3f}"
                for p, tid in zip(topk.values, topk.indices, strict=True)
            )
            print(
                f"  step {step:2d}: top-{TOP_K} = [{top_str}]  "
                f"| target {target_str!r} rank={target_rank} prob={target_prob:.4f}"
            )


if __name__ == "__main__":
    main()
