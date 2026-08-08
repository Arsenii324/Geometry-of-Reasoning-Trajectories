"""Gold-answer and prompt-shape tests for the original synthetic generators.

WHY THIS FILE EXISTS. Six generators in `shapes/synthetic.py` -- counting,
variants, switch, max, count_ones, projection -- had NO test (module coverage
68%, those six bodies entirely uncovered). They produce the prompts AND the gold
answers for ledger rows B2, B3, B3c and B6, so a wrong generator would make every
claim about those tasks wrong while every downstream test still passed.

Two kinds of assertion here:

1. THE ANSWER IS THE ANSWER. Each generator's `answer` is recomputed from its own
   emitted prompt, not from a stored constant -- so the test cannot drift with the
   template, and a template edit that breaks the answer fails immediately.

2. THE LENGTH STRUCTURE IS WHAT THE LEDGER SAYS IT IS. D10/B2's correction is that
   `seq_len` is a deterministic function of `n_ops` -- exactly one distinct length
   per level, zero variation across seeds -- which makes a length-partial
   correlation mathematically degenerate rather than a clean null. That was prose
   in the module docstring; it is asserted here.

No tokenizer is used: every length property below is a whitespace-word count, so
these tests need no network and no model. The token-level check that motivated
`test_variants_differ_by_a_constant_length_offset` was run separately against the
pinned Huginn tokenizer (local is +3 tokens at n_ops 4..64, delta constant).
"""

from __future__ import annotations

import pytest

from traj_geom.shapes.synthetic import (
    make_count_ones_task,
    make_counting_task,
    make_max_task,
    make_projection_task,
    make_switch_task,
    make_variants,
)

N_OPS = [4, 8, 16, 24, 32, 48, 64]


def words(s: str) -> int:
    return len(s.split())


# --- gold answers must follow from the emitted prompt ----------------------


@pytest.mark.parametrize("n", N_OPS)
def test_counting_answer_is_the_net_sum_of_its_own_instructions(n: int) -> None:
    t = make_counting_task(n, seed=3)
    net = t["prompt"].count("Add 1.") - t["prompt"].count("Subtract 1.")
    assert t["answer"] == net
    assert t["n_ops"] == n
    assert t["prompt"].count("Add 1.") + t["prompt"].count("Subtract 1.") == n


@pytest.mark.parametrize("n", N_OPS)
def test_switch_answer_is_the_parity_of_its_own_flips(n: int) -> None:
    t = make_switch_task(n, seed=3)
    flips = t["prompt"].count("Flip.")
    assert t["answer"] == ("on" if flips % 2 else "off")
    assert flips + t["prompt"].count("Wait.") == n


@pytest.mark.parametrize("n", N_OPS)
def test_max_answer_is_the_largest_digit_in_its_own_stream(n: int) -> None:
    t = make_max_task(n, seed=3)
    stream = t["prompt"].split("Numbers: ")[1].split(".")[0].split()
    assert len(stream) == n
    assert t["answer"] == max(int(d) for d in stream)
    assert 1 <= t["answer"] <= 9


@pytest.mark.parametrize("n", N_OPS)
def test_count_ones_answer_is_the_number_of_ones_in_its_own_sequence(n: int) -> None:
    t = make_count_ones_task(n, seed=3)
    seq = t["prompt"].split("Sequence: ")[1].split(".")[0].split()
    assert len(seq) == n
    assert t["answer"] == seq.count("1")


@pytest.mark.parametrize("n", N_OPS)
def test_projection_answer_counts_shifts_along_the_asked_axis(n: int) -> None:
    t = make_projection_task(n, seed=3)
    axis = t["prompt"].split("projection on the ")[1].split(" axis")[0]
    assert axis in ("x", "y", "z")
    assert t["answer"] == t["prompt"].count(f"Shift along {axis}.")
    assert sum(t["prompt"].count(f"Shift along {a}.") for a in "xyz") == n


# --- determinism ------------------------------------------------------------


@pytest.mark.parametrize(
    "fn", [make_counting_task, make_switch_task, make_max_task,
           make_count_ones_task, make_projection_task]
)
def test_generators_are_deterministic_in_the_seed(fn) -> None:
    assert fn(16, seed=7) == fn(16, seed=7)


@pytest.mark.parametrize(
    "fn", [make_counting_task, make_switch_task, make_max_task,
           make_count_ones_task, make_projection_task]
)
def test_generators_actually_vary_with_the_seed(fn) -> None:
    """Otherwise every 'seed' in a sweep is a duplicate row, not a replicate."""
    got = {fn(32, seed=s)["prompt"] for s in range(5)}
    assert len(got) == 5, f"{fn.__name__} produced {len(got)} distinct prompts across 5 seeds"


# --- the length confound, made executable (D10 / B2) ------------------------


@pytest.mark.parametrize(
    "fn", [make_counting_task, make_switch_task, make_max_task,
           make_count_ones_task, make_projection_task]
)
def test_prompt_length_is_a_deterministic_function_of_n_ops(fn) -> None:
    """D10/B2: exactly ONE length per level, so length cannot be partialled out.

    This is the degeneracy that made `partial_spearman(winding, n_ops | seq_len)`
    correlate against floating-point noise: after ranking, n_ops and seq_len are
    literally the same array. Asserted so the confound is a known property of
    these generators rather than something to rediscover.
    """
    for n in N_OPS:
        lens = {words(fn(n, seed=s)["prompt"]) for s in range(5)}
        assert len(lens) == 1, f"{fn.__name__}(n_ops={n}) varied in length across seeds: {lens}"


@pytest.mark.parametrize(
    "fn", [make_counting_task, make_switch_task, make_max_task,
           make_count_ones_task, make_projection_task]
)
def test_prompt_length_is_strictly_increasing_in_n_ops(fn) -> None:
    lens = [words(fn(n, seed=0)["prompt"]) for n in N_OPS]
    assert lens == sorted(lens) and len(set(lens)) == len(lens)


# --- make_variants: the length-matched control ------------------------------


@pytest.mark.parametrize("n", N_OPS)
def test_variants_share_a_byte_identical_body(n: int) -> None:
    """The control's whole point: only the QUESTION may differ."""
    v = make_variants(n, seed=5)
    body_track = v["track"].rsplit(" Final total?", 1)[0]
    body_local = v["local"].rsplit(" What was the last instruction?", 1)[0]
    assert body_track == body_local
    assert body_track == make_counting_task(n, seed=5)["prompt"].rsplit(" Final total?", 1)[0]


def test_variants_differ_by_a_constant_length_offset() -> None:
    """`local` is LONGER than `track` -- but by the same amount at every n_ops.

    The docstring used to claim the two "have the same token length". They do
    not: measured against the pinned Huginn tokenizer, `local` is +3 tokens at
    every n_ops in 4..64 (and +3 whitespace words, asserted here without needing
    a tokenizer). That is not a broken control -- a CONSTANT offset shifts both
    arms equally and leaves any metric-versus-n_ops SLOPE comparison intact,
    which is what B3/B3c actually read. An offset that GREW with n_ops would
    confound the arms with difficulty, and that is what this test forbids.
    """
    deltas = {words(make_variants(n, seed=0)["local"]) - words(make_variants(n, seed=0)["track"])
              for n in N_OPS}
    assert len(deltas) == 1, f"length offset varies with n_ops: {deltas} -- control compromised"
    assert deltas.pop() > 0, "local is expected to be the longer arm"


def test_variants_are_deterministic_and_seed_sensitive() -> None:
    assert make_variants(16, seed=2) == make_variants(16, seed=2)
    assert make_variants(16, seed=2) != make_variants(16, seed=3)
