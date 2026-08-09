"""Validation of the depth/generation analysis on planted data.

The run has one job: separate "the model knows but does not say" from "the model
does not know". Three planted worlds are required to come back distinguishable --
depth helps, depth hurts, and rank predicts nothing about the output -- because a
pipeline that cannot tell them apart would file any of them as the headline.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.run_depth_accuracy import (
    load,
    normalise,
    paired_depth_test,
    rank_vs_generation,
    report,
    score,
    sign_test,
)

FAMS = [f"fam{i:02d}" for i in range(21)]
DEPTHS = (4, 32)


def _bank(tmp_path, *, acc_by_depth, fmt_bonus=0.0, rank_tracks_output=True,
          prose=0.0, n_items=6, seed=0):
    """Write depthacc.json / depthrank.json with a planted accuracy structure.

    `acc_by_depth` maps depth -> probability an item is answered correctly.
    `prose` is the fraction of CORRECT answers wrapped in prose, which exact match
    scores wrong and containment scores right -- the discourse effect, planted.
    """
    rng = np.random.default_rng(seed)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    gen, rk = [], []
    for depth in DEPTHS:
        for fmt in ("bare", "constrained"):
            p = min(1.0, acc_by_depth[depth] + (fmt_bonus if fmt == "constrained" else 0))
            for fam in FAMS:
                for i in range(n_items):
                    gold = str((i + 1) % 10)
                    right = rng.random() < p
                    if right:
                        text = (f"The number is {gold}" if rng.random() < prose
                                else gold)
                    else:
                        text = str((int(gold) + 3) % 10)
                    gen.append({"family": fam, "item": i, "fmt": fmt, "gold": gold,
                                "prompt": "p", "depth": depth, "output": text,
                                "exact": None, "contains": None})
                    r = (1 if right else 50) if rank_tracks_output else \
                        (1 if rng.random() < 0.5 else 50)
                    rk.append({"family": fam, "item": i, "fmt": fmt, "depth": depth,
                               "rank": r, "ok": True})
    with open(out / "depthacc.json", "w", encoding="utf-8") as fh:
        json.dump(gen, fh)
    with open(out / "depthrank.json", "w", encoding="utf-8") as fh:
        json.dump(rk, fh)
    return str(out)


def test_scoring_is_redone_from_the_banked_strings(tmp_path) -> None:
    """The kernel's own `exact` field must not be trusted.

    It is written as None here; if `load` passed it through, every accuracy below
    would be undefined. Re-scoring locally is what lets the scorer change without
    another GPU hour, which matters because D60 measured an 85-point swing from a
    scoring/prompting decision.
    """
    gen, rk = load(_bank(tmp_path, acc_by_depth={4: 0.6, 32: 0.6}))
    assert gen["exact"].notna().all()
    assert gen["exact"].dtype == bool
    assert 0.4 < gen["exact"].mean() < 0.8


def test_containment_separates_prose_from_a_wrong_answer(tmp_path) -> None:
    """The exact/contains gap is the discourse effect in units of accuracy."""
    gen, _ = load(_bank(tmp_path, acc_by_depth={4: 0.8, 32: 0.8}, prose=1.0, seed=2))
    assert gen["contains"].mean() > gen["exact"].mean() + 0.5
    assert gen["exact"].mean() < 0.1, "prose-wrapped answers must fail exact match"


def test_depth_hurting_is_detected(tmp_path) -> None:
    """D68's reading, planted: accuracy peaks shallow and collapses with depth."""
    t = paired_depth_test(load(_bank(tmp_path, acc_by_depth={4: 0.8, 32: 0.1},
                                     seed=3))[0], "bare", 4, 32)
    assert t["usable"] and t["p"] < 0.01
    assert t["direction"] == "deeper is worse", t


def test_depth_helping_is_detected(tmp_path) -> None:
    """NON-SUPPRESSION. The two-sided test must find the opposite just as readily;
    a run that could only confirm D68 would not be evidence for it."""
    t = paired_depth_test(load(_bank(tmp_path, acc_by_depth={4: 0.1, 32: 0.8},
                                     seed=4))[0], "bare", 4, 32)
    assert t["usable"] and t["p"] < 0.01
    assert t["direction"] == "deeper is better", t


def test_a_flat_depth_profile_is_not_called_a_finding(tmp_path) -> None:
    t = paired_depth_test(load(_bank(tmp_path, acc_by_depth={4: 0.5, 32: 0.5},
                                     seed=5))[0], "bare", 4, 32)
    assert t["usable"] and t["p"] > 0.05, t


def test_rank_predicting_the_output_shows_up_as_a_high_phi(tmp_path) -> None:
    gen, rk = load(_bank(tmp_path, acc_by_depth={4: 0.5, 32: 0.5},
                         rank_tracks_output=True, seed=6))
    rv = rank_vs_generation(gen, rk)
    ex = rv[rv["metric"] == "exact"]
    assert (ex["phi"] > 0.8).all(), ex


def test_rank_not_predicting_the_output_shows_up_and_is_said_out_loud(tmp_path) -> None:
    """THE OUTCOME THAT WOULD REFRAME D68, D69 AND D75.

    If gold at rank 1 does not predict a correct decoded string, then every
    rank-based number in this project describes what is AVAILABLE to the readout
    rather than what the model emits. The report has to state that rather than
    leaving a reader to infer it from a phi column.
    """
    gen, rk = load(_bank(tmp_path, acc_by_depth={4: 0.5, 32: 0.5},
                         rank_tracks_output=False, seed=7))
    rv = rank_vs_generation(gen, rk)
    assert (rv[rv["metric"] == "exact"]["phi"].abs() < 0.3).all()
    assert "does NOT reliably predict" in report(gen, rk)


def test_the_format_bonus_is_reported(tmp_path) -> None:
    """D69 measured +83 points from one instruction; the table must show such a gap."""
    gen, rk = load(_bank(tmp_path, acc_by_depth={4: 0.1, 32: 0.1}, fmt_bonus=0.8,
                         seed=8))
    txt = report(gen, rk)
    assert "FORMAT INTERACTION" in txt
    assert "constrained minus bare" in txt


def test_empty_outputs_are_counted_and_flagged(tmp_path) -> None:
    """D62's symptom: a full table of zeros read as a capability finding."""
    path = _bank(tmp_path, acc_by_depth={4: 0.5, 32: 0.5}, seed=9)
    import os
    with open(os.path.join(path, "depthacc.json"), encoding="utf-8") as fh:
        recs = json.load(fh)
    for r in recs[:40]:
        r["output"] = ""
    with open(os.path.join(path, "depthacc.json"), "w", encoding="utf-8") as fh:
        json.dump(recs, fh)
    gen, rk = load(path)
    assert int(gen["empty"].sum()) == 40
    assert "D62's symptom" in report(gen, rk)


def test_sign_test_drops_ties_and_says_how_many_moved() -> None:
    """Families whose accuracy is identical at both depths carry no information.

    Counting them as evidence would dilute a real effect toward p = 1; counting
    them as agreement would manufacture one.
    """
    up, n, p = sign_test(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]))
    assert n == 5 and up == 5
    assert p < 0.07
    assert sign_test(np.zeros(10)) == (0, 0, 1.0)


def test_normalise_is_applied_to_both_sides() -> None:
    assert score("7.", "7") == (True, True)
    assert score("7", "7.") == (True, True)
    assert normalise("The 7.") == normalise("7")
    assert score("Addition", "Add")[1] is False


def test_paired_test_refuses_a_missing_depth(tmp_path) -> None:
    gen, _ = load(_bank(tmp_path, acc_by_depth={4: 0.5, 32: 0.5}, seed=10))
    t = paired_depth_test(gen, "bare", 4, 999)
    assert t["usable"] is False


@pytest.mark.parametrize("fmt", ["bare", "constrained"])
def test_both_formats_are_analysed_separately(tmp_path, fmt) -> None:
    """P4 needs the depth profile per format; pooling them would hide the
    interaction that is the whole reason both were run."""
    gen, _ = load(_bank(tmp_path, acc_by_depth={4: 0.7, 32: 0.2}, fmt_bonus=0.0,
                        seed=11))
    t = paired_depth_test(gen, fmt, 4, 32)
    assert t["usable"] and t["fmt"] == fmt
