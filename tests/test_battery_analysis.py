"""Validation of the battery analysis path, before it meets real data.

The B6-usability test is the one that matters. D72 killed the first B6 run because
correctness was a deterministic function of the answer value, and this analysis
exists partly to find a family where that is NOT true. A checker that says "usable"
for a D72-shaped family would send the next GPU hour into the same wall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scripts.run_battery import b6_usability, load, per_family


def _items(golds, correct, arm="trained", family="f", n_depth=8, chance=32768):
    """Rank curves engineered so `min(rank)==1` exactly for the correct items."""
    rows = []
    for i, (g, ok) in enumerate(zip(golds, correct, strict=True)):
        curve = [chance // 2] * n_depth
        curve[3] = 1 if ok else 77
        rows.append({"rank": curve, "logp_gold": [-1.0] * n_depth, "gold": str(g),
                     "prompt": f"p{i}", "n_gold_tok": 1})
    return {"arms": {arm: {family: rows}}}


def _write(tmp_path, blob):
    p = tmp_path / "battery.json"
    p.write_text(__import__("json").dumps(blob))
    return str(p)


def test_load_flattens_and_marks_correctness_at_best_depth(tmp_path) -> None:
    """Correctness must be best-depth, not final-depth: D68 showed accuracy is
    non-monotone in r, `echo_digit` peaking at r=4 and reading 0% by r=8."""
    blob = _items([1, 2, 3, 4], [True, False, True, False])
    df = load(_write(tmp_path, blob))
    assert len(df) == 4
    assert df["correct_best"].tolist() == [True, False, True, False]
    # every curve ends at chance/2, so final-depth accuracy is 0 for all of them
    assert df["correct_final"].sum() == 0
    assert df.loc[df["correct_best"], "argmin_depth"].eq(4).all()


def test_b6_flags_the_d72_case_as_unusable() -> None:
    """D72 exactly: correct golds {1,2} and incorrect {3,4}, zero overlap.

    Correctness is then a function of the answer, so the label-permutation null
    breaks the correctness-geometry link and the prompt-geometry link together and
    cannot separate them. This must NOT be reported as usable.
    """
    tr = pd.DataFrame({"gold": ["1", "2", "3", "4"],
                       "correct_best": [True, True, False, False]})
    out = b6_usability(tr)
    assert out["b6_usable"] is False
    assert out["b6_n_mixed_values"] == 0


def test_b6_flags_a_genuinely_mixed_family_as_usable() -> None:
    """Non-suppression: the case B6 actually needs must come back positive.

    Gold '1' carries both a success and a failure, so within that value the
    comparison holds the prompt content fixed and only correctness varies.
    """
    tr = pd.DataFrame({"gold": ["1", "1", "1", "0", "0", "0"],
                       "correct_best": [True, False, True, False, True, False]})
    out = b6_usability(tr)
    assert out["b6_usable"] is True
    assert out["b6_n_mixed_values"] == 2
    assert out["b6_n_in_mixed_gold"] == 6


def test_b6_rejects_a_ceiling_family_even_with_mixed_golds() -> None:
    """15/16 correct gives no failure class worth contrasting, mixed golds or not."""
    tr = pd.DataFrame({"gold": ["1"] * 16, "correct_best": [True] * 15 + [False]})
    assert b6_usability(tr)["b6_usable"] is False


def test_untrained_control_flags_a_family_that_is_not_at_chance(tmp_path) -> None:
    """A decode that is broken cannot put one arm at rank 3 and the other at
    chance; that asymmetry is what made D68 credible, so it is checked per family."""
    blob = _items([1, 2], [True, False])
    blob["arms"]["untrained"] = {"f": [
        {"rank": [5] * 8, "logp_gold": [-1.0] * 8, "gold": "1", "prompt": "p",
         "n_gold_tok": 1}]}
    blob["chance"] = 32768
    fam = per_family(load(_write(tmp_path, blob)))
    assert bool(fam.iloc[0]["chance_ok"]) is False


def test_untrained_control_passes_when_the_arm_sits_at_chance(tmp_path) -> None:
    blob = _items([1, 2], [True, False])
    blob["arms"]["untrained"] = {"f": [
        {"rank": [30000] * 8, "logp_gold": [-1.0] * 8, "gold": "1", "prompt": "p",
         "n_gold_tok": 1}]}
    blob["chance"] = 32768
    fam = per_family(load(_write(tmp_path, blob)))
    assert bool(fam.iloc[0]["chance_ok"]) is True
    assert np.isfinite(fam.iloc[0]["log10_gap"])
