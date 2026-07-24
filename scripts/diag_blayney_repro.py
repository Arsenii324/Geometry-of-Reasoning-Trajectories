"""Diagnostic: does this project's extraction pipeline reproduce a genuine
winding loop under Blayney et al.'s own, independently-measured,
loop-inducing condition?

OWNER: Data+Analysis
STATUS: implemented 2026-07-24 (project_plan.md Phase 1.2). A positive
    control, not a scored experiment: this project has never observed a
    real (non-starved-budget) loop or drift anywhere (claims_ledger.md B9,
    D13 -- 1,198/1,198 full-compute-budget extractions settle). Rather
    than only hoping this project's own task designs happen to produce a
    loop, reproduce a condition the literature already measured as
    loop-inducing (Blayney et al. arXiv:2604.11791 App. C, Table 3: 0.14%
    per-token orbit rate under their "Long Persona" system prompt, vs.
    0.02% baseline) and check whether the pipeline sees anything there. If
    it doesn't, that's a pipeline bug to chase now, not a finding.
TASK: run a small GSM8K sample through Huginn under two system-prompt
    conditions, ALL token positions (not just the answer token -- Blayney's
    loops mostly occur elsewhere), classify every position's shape via
    this project's own `classify_shape`, report the loop/drift fraction
    per condition.
I/O: -> prints per-condition shape counts; saves results/blayney_repro.csv
    (one row per (condition, example_idx, token_position)).

Run: uv run python -m scripts.diag_blayney_repro [--n-examples 16]
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from tqdm import tqdm

from scripts._common import cached, load_model
from traj_geom.constants import DEFAULT_NUM_STEPS
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.gate import classify_shape

# Verbatim, from the downloaded PDF (arXiv 2604.11791, App. C.1) -- the
# exact system prompt Blayney et al. attribute to Geiping et al. (2025)'s
# own orbit plots, and confirm independently raises the non-fixed-point
# rate over their "No System Prompt" baseline.
LONG_PERSONA_SYSTEM_PROMPT = (
    "You are Huginn, an AI assistant who embodies careful thought and "
    "deliberation. Your responses demonstrate:\n"
    "Methodical reasoning, breaking complex problems into clear steps\n"
    "Mathematical and programming expertise grounded in fundamentals\n"
    "The ability to acknowledge uncertainty and correct course when needed\n"
    "Clear communication that illuminates rather than just informs\n"
    "When engaging with questions, you first seek to understand their "
    "deeper structure before answering. Like your namesake who flew the "
    "nine worlds seeking wisdom, you explore problems from multiple "
    "angles, helping users build genuine understanding rather than "
    "providing shallow answers. You express warmth and intellectual "
    "curiosity while maintaining professionalism. When faced with errors "
    "or confusion, you model honest reflection and careful correction. "
    "Your goal is not just to provide answers, but to help humans develop "
    "clearer, deeper thinking."
)

CONDITIONS = {
    "long_persona": LONG_PERSONA_SYSTEM_PROMPT,
    "no_system_prompt": None,
}


def _extract_all_positions(model, tok, prompt: str, num_steps: int, seed: int) -> torch.Tensor:
    """Minimal all-token variant of `extract_trajectory` (hook.py). Deliberately
    NOT importing/modifying hook.py: mirrors its exact return_logits=False hook
    (which already captures every position, only the return statement slices
    to one) rather than risk touching the validated production path for a
    one-off diagnostic.

    Returns:
        Tensor of shape [num_steps, n_tokens, hidden_dim].
    """
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()
    lat: list[torch.Tensor] = []

    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

    def tracking_hook(m, i, o):
        lat.append(o.detach().float().cpu())

    h = mod.register_forward_hook(tracking_hook)
    try:
        with torch.no_grad():
            model(input_ids=ids, num_steps=num_steps)
    finally:
        h.remove()

    return torch.stack([step[0] for step in lat])  # [num_steps, n_tokens, hidden]


def compute(n_examples: int, num_steps: int) -> pd.DataFrame:
    """Extract every token position for a small GSM8K sample, under both
    system-prompt conditions, and classify each position's shape.
    """
    from datasets import load_dataset

    model, tok = load_model()
    gsm8k = load_dataset("openai/gsm8k", "main", split="test").select(range(n_examples))

    rows = []
    n_failed = 0
    for cond_name, system_prompt in CONDITIONS.items():
        for ex_idx, ex in enumerate(tqdm(gsm8k, desc=cond_name)):
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": ex["question"]})
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            try:
                traj = _extract_all_positions(model, tok, prompt, num_steps, seed=0)
                n_tokens = traj.shape[1]
                for pos in range(n_tokens):
                    path = traj[:, pos, :].numpy()
                    rows.append(
                        {
                            "condition": cond_name,
                            "example_idx": ex_idx,
                            "token_position": pos,
                            "n_tokens": n_tokens,
                            "winding": abs(winding_of(path, burn=4)),
                            "shape": classify_shape(path),
                            "steps_settle": steps_to_settle(path),
                        }
                    )
            except Exception as e:  # noqa: BLE001 -- one bad example must not
                # lose the rest of this small, cheap sweep.
                n_failed += 1
                print(f"{cond_name} example {ex_idx}: skipping after error: {e!r}")
    if n_failed:
        print(f"diag_blayney_repro: {n_failed} examples failed and were skipped.")
    return pd.DataFrame(rows)


def main() -> None:
    """Report the loop/drift fraction per condition -- the actual check."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-examples", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    args = ap.parse_args()

    df = cached("blayney_repro.csv", lambda: compute(args.n_examples, args.num_steps))

    print("\n--- Shape counts per condition (all token positions pooled) ---")
    print(df.groupby("condition")["shape"].value_counts())

    print("\n--- Non-settle (loop+drift) fraction per condition ---")
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        if len(sub) == 0:
            print(f"{cond}: no data")
            continue
        non_settle = (sub["shape"] != "settle").mean()
        print(f"{cond}: {non_settle:.4%} of {len(sub)} (token, example) pairs")

    print(
        "\nNOTE: this is a pipeline sanity check (project_plan.md Phase 1.2), not "
        "a powered experiment -- see docs/power_and_preregistration.md for the "
        "N needed to expect >=5 genuine loops at these per-token rates. If "
        "long_persona shows a non-settle fraction near 0.14% and "
        "no_system_prompt near 0.02% (Blayney's own Table 3 rates), the "
        "pipeline is working as expected, even if the raw counts are small."
    )


if __name__ == "__main__":
    main()
