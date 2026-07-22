"""Synthetic reasoning-task generators (state-holding probes for H3).

OWNER: Shapes+Gate
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, make_* task builders).
TASK: generate prompts whose "reasoning depth" is a controlled integer, so
    effective compute can be regressed against it.
I/O: (n_ops, seed) -> dict with a prompt (and, for variants, matched prompts).

The counting / switch tasks force the model to HOLD a running state: to answer,
it cannot just read the last token, it must accumulate. `make_variants` is the
length-matched control — `track` and `local` share an identical body and differ
only in the question, isolating state-holding from raw prompt length.

`make_count_ones_task`, `make_projection_task`, and `make_three_scale_task` were
added later (2026-07-19), not from the notebook. Only `make_three_scale_task`
decouples prompt length from difficulty — see its own docstring. The other two
have the same collinearity as every task above them: for a fixed `n_ops`, token
length is deterministic across seeds (verified: rank-correlation(n_ops, seq_len)
== 1.0), so neither can support a length-partial correlation any more than
`make_counting_task` can.
"""

from __future__ import annotations

import random


def make_counting_task(n_ops: int, seed: int = 0) -> dict:
    """Running +/-1 sum. To answer, the model must hold a counter.

    Args:
        n_ops: Number of +/-1 operations (the reasoning depth).
        seed: Seed for the random op sequence.

    Returns:
        ``{"prompt", "n_ops", "answer"}``.
    """
    rng = random.Random(seed)
    ops = [rng.choice([1, -1]) for _ in range(n_ops)]
    text = (
        "Start at 0. "
        + " ".join("Add 1." if o > 0 else "Subtract 1." for o in ops)
        + " Final total? A:"
    )
    return {"prompt": text, "n_ops": n_ops, "answer": sum(ops)}


def make_variants(n_ops: int, seed: int = 0) -> dict:
    """Length-matched pair: identical body, different question (the killer control).

    Args:
        n_ops: Number of +/-1 operations (the reasoning depth).
        seed: Seed for the random op sequence.

    Returns:
        ``{"track", "local"}`` — ``track`` needs accumulation, ``local`` needs
        only the last instruction; both have the same token length.
    """
    rng = random.Random(seed)
    ops = [rng.choice([1, -1]) for _ in range(n_ops)]
    body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1." for o in ops)
    return {
        "track": body + " Final total? A:",  # needs accumulation
        "local": body + " What was the last instruction? A:",  # needs only the last step
    }


def make_switch_task(n_ops: int, seed: int = 0) -> dict:
    """Light on/off parity. Generalisation test beyond arithmetic counting.

    Args:
        n_ops: Number of flip/wait steps (the reasoning depth).
        seed: Seed for the random flip sequence.

    Returns:
        ``{"prompt", "answer"}`` where answer is "on"/"off" by flip parity.
    """
    rng = random.Random(seed)
    flips = [rng.choice([0, 1]) for _ in range(n_ops)]
    body = "Light is off. " + " ".join("Flip." if f else "Wait." for f in flips)
    return {"prompt": body + " Is the light on? A:", "answer": "on" if sum(flips) % 2 else "off"}


def make_max_task(n_ops: int, seed: int = 0) -> dict:
    """Running maximum. To answer, the model must hold the largest number so far.

    Args:
        n_ops: Number of digits in the stream (the reasoning depth).
        seed: Seed for the random digit stream.

    Returns:
        ``{"prompt", "answer"}`` where answer is the maximum digit.
    """
    rng = random.Random(seed)
    nums = [rng.randint(1, 9) for _ in range(n_ops)]
    return {
        "prompt": "Numbers: " + " ".join(map(str, nums)) + ". Largest so far? A:",
        "answer": max(nums),
    }


def make_count_ones_task(n_ops: int, seed: int = 0) -> dict:
    """Count ones in a sequence of zeros and ones.

    CAVEAT: like `make_counting_task`, `seq_len` is a deterministic function
    of `n_ops` (n_ops tokens, always) — no genuine length-partial control.

    Args:
        n_ops: Length of the sequence.
        seed: Random seed.

    Returns:
        ``{"prompt", "answer"}``
    """
    rng = random.Random(seed)
    seq = [rng.choice([0, 1]) for _ in range(n_ops)]
    question = f". How many ones are in the first {n_ops} symbols? A:"
    return {
        "prompt": "Sequence: " + " ".join(map(str, seq)) + question,
        "answer": sum(seq),
    }


def make_projection_task(n_ops: int, seed: int = 0) -> dict:
    """Projection in linear spaces: successive shifts along basis vectors.

    CAVEAT: like `make_counting_task`, `seq_len` is a deterministic function
    of `n_ops` (n_ops tokens, always) — no genuine length-partial control.

    Args:
        n_ops: Number of shifts.
        seed: Random seed.

    Returns:
        ``{"prompt", "answer"}``
    """
    rng = random.Random(seed)
    axes = ["x", "y", "z"]
    shifts = [rng.choice(axes) for _ in range(n_ops)]
    proj_axis = rng.choice(axes)

    body = "Start at origin. " + " ".join(f"Shift along {ax}." for ax in shifts)
    return {
        "prompt": body + f" What is the projection on the {proj_axis} axis? A:",
        "answer": shifts.count(proj_axis),
    }


def make_three_scale_task(
    irrelevant_len: int, neutral_len: int, active_len: int, seed: int = 0
) -> dict:
    """Three-scale length-ablated counting task for V6.

    Independently varies three types of prompt length:
    - irrelevant_len: padding text that doesn't require processing (filler).
    - neutral_len: number of 0s in the sequence (must be processed, but doesn't change state).
    - active_len: number of 1s in the sequence (changes internal state).

    Args:
        irrelevant_len: Number of filler words.
        neutral_len: Number of zeros.
        active_len: Number of ones.
        seed: Random seed to shuffle the 0s and 1s.

    Returns:
        ``{"prompt", "answer", "irrelevant_len", "neutral_len", "active_len"}``
    """
    rng = random.Random(seed)

    seq = [1] * active_len + [0] * neutral_len
    rng.shuffle(seq)
    seq_str = " ".join(map(str, seq))

    filler_bank = [
        "apple", "banana", "cat", "dog", "elephant", "fox", "grape", "hat", "ice", "jump",
    ]
    filler_str = " ".join(rng.choice(filler_bank) for _ in range(irrelevant_len))

    prompt = (
        f"Ignore these words: {filler_str}. Sequence: {seq_str}. "
        "How many ones are in the sequence? A:"
    )

    return {
        "prompt": prompt,
        "answer": active_len,
        "irrelevant_len": irrelevant_len,
        "neutral_len": neutral_len,
        "active_len": active_len,
    }
