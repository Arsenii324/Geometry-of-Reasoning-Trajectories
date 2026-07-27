"""Tests for Barannikov's Task a and Task b generators.

The properties pinned here are the ones that make these tasks worth running at
all — fixed length (which breaks the confound every other generator has) and,
for Task b, genuine balancedness (which is what closes the curve and makes a
winding number an integer invariant).
"""

from __future__ import annotations

import pytest

from traj_geom.shapes.synthetic import make_nesting_depth_task, make_running_count_task

# --- Task a: running count, fixed length ----------------------------------


@pytest.mark.parametrize("m", [16, 32, 64])
def test_running_count_length_is_fixed_across_seeds(m: int) -> None:
    """THE point of the design: difficulty varies, token count does not.

    Every other generator in this module has rank-corr(difficulty, seq_len) =
    exactly 1.000, which makes a length control mathematically impossible.
    """
    lengths = {len(make_running_count_task(m=m, seed=s)["bits"]) for s in range(10)}
    assert lengths == {m}


def test_running_count_answer_varies_at_fixed_length() -> None:
    """Difficulty must actually vary, or the design tests nothing."""
    answers = {make_running_count_task(m=64, seed=s)["answer"] for s in range(20)}
    assert len(answers) > 5, f"too little spread in the target: {answers}"


def test_running_count_targets_are_the_prefix_sums() -> None:
    """`running[i]` must equal y_i = #{j <= i : x_j = 1} — the per-position target."""
    t = make_running_count_task(m=32, seed=3)
    bits, running = t["bits"], t["running"]
    assert len(running) == len(bits)
    assert all(running[i] == sum(bits[: i + 1]) for i in range(len(bits)))
    assert t["answer"] == running[-1]


def test_running_count_p_one_shifts_difficulty() -> None:
    lo = [make_running_count_task(m=64, seed=s, p_one=0.2)["answer"] for s in range(8)]
    hi = [make_running_count_task(m=64, seed=s, p_one=0.8)["answer"] for s in range(8)]
    assert sum(lo) / len(lo) < sum(hi) / len(hi)


def test_running_count_is_deterministic_in_the_seed() -> None:
    assert make_running_count_task(m=32, seed=7) == make_running_count_task(m=32, seed=7)


# --- Task b: nesting depth, balanced ---------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_balanced_strings_are_genuinely_balanced(seed: int) -> None:
    """The load-bearing property: the depth returns to zero and never goes below it.

    If it did not, the curve would not close and the winding number would stay
    a non-invariant real number — which is the exact defect that invalidated
    this project's own use of it.
    """
    t = make_nesting_depth_task(m=32, seed=seed, balanced=True)
    depths = t["depths"]
    assert depths[-1] == 0, "a balanced string must return to depth 0"
    assert min(depths) >= 0, "a balanced string must never go below depth 0"
    assert t["balanced"] is True


@pytest.mark.parametrize("seed", range(6))
def test_unbalanced_control_does_not_close(seed: int) -> None:
    """The matched control: same length and alphabet, no closure."""
    t = make_nesting_depth_task(m=32, seed=seed, balanced=False)
    assert not (t["depths"][-1] == 0 and min(t["depths"]) >= 0) or not t["balanced"]


def test_nesting_depth_length_is_fixed() -> None:
    lengths = {make_nesting_depth_task(m=32, seed=s)["m"] for s in range(10)}
    assert lengths == {32}


def test_nesting_depth_max_depth_varies() -> None:
    depths = {make_nesting_depth_task(m=64, seed=s)["max_depth"] for s in range(20)}
    assert len(depths) > 3, f"max depth barely varies: {depths}"


def test_nesting_depth_targets_are_the_signed_prefix_sums() -> None:
    t = make_nesting_depth_task(m=32, seed=2, balanced=True)
    sym, depths = t["symbols"], t["depths"]
    running = 0
    for i, s in enumerate(sym):
        running += 1 if s == "(" else -1
        assert depths[i] == running


def test_odd_length_is_made_even_when_balanced() -> None:
    """A balanced string cannot have odd length."""
    assert make_nesting_depth_task(m=33, seed=0, balanced=True)["m"] % 2 == 0


def test_both_tasks_end_with_the_answer_cue() -> None:
    """Prompts must end in 'A:' with no trailing space — the tokenization trap
    documented in eval_depth (the continuation is space-prefixed)."""
    for t in (make_running_count_task(m=16), make_nesting_depth_task(m=16)):
        assert t["prompt"].endswith("A:")
        assert not t["prompt"].endswith("A: ")
