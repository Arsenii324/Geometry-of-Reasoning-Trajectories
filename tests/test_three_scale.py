from traj_geom.shapes.synthetic import make_three_scale_task


def test_make_three_scale_task():
    task = make_three_scale_task(irrelevant_len=3, neutral_len=2, active_len=4, seed=42)

    # Check returned dictionary keys
    assert "prompt" in task
    assert "answer" in task

    # Check the logic of the lengths
    assert task["irrelevant_len"] == 3
    assert task["neutral_len"] == 2
    assert task["active_len"] == 4
    assert task["answer"] == 4

    # Check string formatting
    assert task["prompt"].startswith("Ignore these words:")

    # Exact word count for this seed (verified: prompt is deterministic given
    # seed=42), not a lower bound — a >= check would silently pass if the
    # prompt template grew or shrank unexpectedly.
    words = task["prompt"].split()
    assert len(words) == 21

    # Check that 1s and 0s are correctly distributed in the sequence part
    seq_part = task["prompt"].split("Sequence: ")[1].split(". How")[0]
    digits = seq_part.split()
    assert digits.count("1") == 4
    assert digits.count("0") == 2
