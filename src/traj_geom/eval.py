"""Answer correctness via Huginn's adaptive-compute generation.

OWNER: Data+Analysis
STATUS: implemented (from notebooks/01_mvp.ipynb).
TASK: decode the model's actual answer and check it against ground truth — the
    behavioural axis behind the "counting accuracy collapses" result.
I/O: (model, tok, prompt, ...) -> decoded text / bool correctness.

NOTE: uses `model.generate_with_adaptive_compute` (Huginn-specific), so it needs
    the `model` extra and a GPU. model/tok are passed in, not global.
"""

from __future__ import annotations

import re
from typing import Any


def model_answer_text(
    model: Any, tok: Any, prompt: str, num_steps: int = 32, max_new_tokens: int = 8
) -> str:
    """Generate and decode only the continuation after the prompt.

    Args:
        model: A Huginn model handle.
        tok: The matching tokenizer.
        prompt: Input prompt text.
        num_steps: Recurrent unrolls used during generation.
        max_new_tokens: Number of tokens to generate.

    Returns:
        The decoded generated text (prompt stripped).
    """
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate_with_adaptive_compute(
        ids, num_steps=num_steps, max_new_tokens=max_new_tokens, tokenizer=tok
    )
    return tok.decode(out[0][ids.shape[1]:])


def counting_correct(
    model: Any, tok: Any, prompt: str, answer: int, num_steps: int = 32
) -> bool:
    """Whether the first integer the model emits equals ``answer``.

    Args:
        model: A Huginn model handle.
        tok: The matching tokenizer.
        prompt: Input prompt text.
        answer: Ground-truth integer answer.
        num_steps: Recurrent unrolls used during generation.

    Returns:
        True iff the first parsed integer matches ``answer``.
    """
    txt = model_answer_text(model, tok, prompt, num_steps=num_steps)
    m = re.findall(r"-?\d+", txt)
    return len(m) > 0 and int(m[0]) == answer
