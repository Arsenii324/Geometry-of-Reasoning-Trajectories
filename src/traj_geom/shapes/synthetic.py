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
    
    Args:
        n_ops: Length of the sequence.
        seed: Random seed.
        
    Returns:
        ``{"prompt", "answer"}``
    """
    rng = random.Random(seed)
    seq = [rng.choice([0, 1]) for _ in range(n_ops)]
    return {
        "prompt": "Sequence: " + " ".join(map(str, seq)) + f". How many ones are in the first {n_ops} symbols? A:",
        "answer": sum(seq),
    }


def make_projection_task(n_ops: int, seed: int = 0) -> dict:
    """Projection in linear spaces: successive shifts along basis vectors.
    
    Args:
        n_ops: Number of shifts.
        seed: Random seed.
        
    Returns:
        ``{"prompt", "answer"}``
    """
    rng = random.Random(seed)
    # Basis vectors: x, y, z
    axes = ["x", "y", "z"]
    shifts = [rng.choice(axes) for _ in range(n_ops)]
    # Target projection axis
    proj_axis = rng.choice(axes)
    answer = shifts.count(proj_axis)
    
    body = "Start at origin. " + " ".join(f"Shift along {ax}." for ax in shifts)
    return {
        "prompt": body + f" What is the projection on the {proj_axis} axis? A:",
        "answer": answer,
    }


def make_three_scale_task(irrelevant_len: int, neutral_len: int, active_len: int, seed: int = 0) -> dict:
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
    
    # 1. Active and Neutral (The logic sequence)
    seq = [1] * active_len + [0] * neutral_len
    rng.shuffle(seq)
    seq_str = " ".join(map(str, seq))
    
    # 2. Irrelevant (The padding)
    # We use a fixed bank of filler words to build irrelevant length
    filler_bank = ["apple", "banana", "cat", "dog", "elephant", "fox", "grape", "hat", "ice", "jump"]
    filler_words = [rng.choice(filler_bank) for _ in range(irrelevant_len)]
    filler_str = " ".join(filler_words)
    
    # Assemble the final prompt
    prompt = f"Ignore these words: {filler_str}. Sequence: {seq_str}. How many ones are in the sequence? A:"
    
    return {
        "prompt": prompt,
        "answer": active_len,
        "irrelevant_len": irrelevant_len,
        "neutral_len": neutral_len,
        "active_len": active_len
    }
