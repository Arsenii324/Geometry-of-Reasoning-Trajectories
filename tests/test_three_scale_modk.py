import pytest

from traj_geom.shapes.synthetic import make_three_scale_modk_task


def test_make_three_scale_modk_task():
    task = make_three_scale_modk_task(
        active_len=7, irrelevant_len=3, total_len=15, modulus=4, seed=42
    )

    assert "prompt" in task
    assert "answer" in task

    # neutral_len is derived, not passed in.
    assert task["active_len"] == 7
    assert task["irrelevant_len"] == 3
    assert task["neutral_len"] == 5  # 15 - 7 - 3
    assert task["total_len"] == 15
    assert task["modulus"] == 4
    assert task["answer"] == 3  # 7 % 4

    assert task["prompt"].startswith("Sequence: ")
    assert "modulo 4" in task["prompt"]

    # Exact word count for this seed (deterministic given seed=42) -- a
    # regression pin, not a lower bound, matching test_three_scale.py's
    # convention.
    words = task["prompt"].split()
    assert len(words) == 26

    seq_part = task["prompt"].split("Sequence: ")[1].split(". How")[0]
    symbols = seq_part.split()
    assert len(symbols) == 15  # total_len, constant by construction
    assert symbols.count("1") == 7
    assert symbols.count("0") == 5
    assert symbols.count("x") == 3


def test_total_length_and_answer_position_constant_across_active_len_sweep():
    """The whole point of this task over make_three_scale_task: sweeping
    active_len (at fixed total_len) must not change total sequence length
    or where the question starts -- the exact confound claims_ledger.md D11
    found in the original task.
    """
    prompts = [
        make_three_scale_modk_task(
            active_len=a, irrelevant_len=2, total_len=20, modulus=5, seed=0
        )["prompt"]
        for a in (0, 3, 8, 15, 18)
    ]
    word_counts = {len(p.split()) for p in prompts}
    assert len(word_counts) == 1  # every prompt has identical total length

    question_starts = {p.split(". How")[1] for p in prompts}
    assert len(question_starts) == 1  # the question suffix is byte-identical


def test_answer_is_always_single_token_range():
    """answer = active_len % modulus is always in [0, modulus), so it's
    representable as one digit for any modulus in 2..10 -- sidesteps the
    multi-digit-answer measurement bug (project_plan.md §9) regardless of
    how large active_len gets.
    """
    for active_len in (0, 1, 9, 10, 50, 999):
        for modulus in range(2, 11):
            task = make_three_scale_modk_task(
                active_len=active_len, irrelevant_len=0, total_len=active_len,
                modulus=modulus, seed=0,
            )
            assert 0 <= task["answer"] < modulus
            assert len(str(task["answer"])) == 1


def test_switch_task_is_the_modulus_two_case():
    """make_switch_task's parity answer ('on' iff sum(flips) is odd) is
    exactly the modulus=2 case of this family, restated as a digit: parity
    odd <=> active_len % 2 == 1.
    """
    for active_len in range(0, 8):
        task = make_three_scale_modk_task(
            active_len=active_len, irrelevant_len=0, total_len=active_len,
            modulus=2, seed=0,
        )
        is_on_equivalent = active_len % 2 == 1
        assert (task["answer"] == 1) == is_on_equivalent


def test_raises_on_modulus_out_of_single_token_range():
    with pytest.raises(ValueError, match="modulus"):
        make_three_scale_modk_task(active_len=5, irrelevant_len=0, total_len=5, modulus=11)
    with pytest.raises(ValueError, match="modulus"):
        make_three_scale_modk_task(active_len=5, irrelevant_len=0, total_len=5, modulus=1)


def test_raises_on_total_len_too_small():
    with pytest.raises(ValueError, match="neutral_len"):
        make_three_scale_modk_task(
            active_len=5, irrelevant_len=5, total_len=8, modulus=3
        )  # 5+5=10 > 8
