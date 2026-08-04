"""Guards the headline findings against drift between prose and data.

Backlog item 6.4 generalised. `docs/FINDINGS.md` quotes numbers that live in
`results/*.csv`; nothing previously stopped the two from diverging as either
was edited. These tests fail if a quoted number stops matching its source, or
if a document stops citing a result it is supposed to carry.

The failure mode this prevents is specific and has already happened once in
this project's history: a summary regressed a figure its own source document
had correct (claims_ledger D21, error 1).
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
RESULTS = os.path.join(ROOT, "results")


def _doc(name: str) -> str:
    """Document text with typographic minus/dash normalised to ASCII.

    The prose deliberately uses U+2212 MINUS SIGN and en-dashes for
    readability, so a naive substring search for "-0.0091" misses "−0.0091".
    Normalising here keeps the typography free and the tests strict.
    """
    path = os.path.join(DOCS, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present")
    text = open(path, encoding="utf-8").read()
    return text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")


def _csv(name: str) -> pd.DataFrame:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present")
    return pd.read_csv(path)


# --- prose must match data -------------------------------------------------


def test_register_correlations_match_the_csv() -> None:
    """D34/D36's headline numbers, quoted in FINDINGS.md."""
    doc = _doc("FINDINGS.md")
    df = _csv("register_analysis.csv")
    for kind in ("a", "b_bal", "b_unbal"):
        got = df[df.kind == kind].register_r.mean()
        assert f"{got:.4f}" in doc, (
            f"FINDINGS.md does not quote register_r[{kind}] = {got:.4f}"
        )


def test_winding_stratified_effects_match_the_csv() -> None:
    """D28: the sign flip with num_steps is the load-bearing negative result."""
    doc = _doc("FINDINGS.md")
    df = _csv("manifold_null.csv")
    real = df[df.arm == "real"]
    for ns in (64, 128):
        got = real[real.num_steps == ns].obs_minus_null.mean()
        assert f"{got:+.4f}" in doc, f"FINDINGS.md does not quote ns={ns} effect {got:+.4f}"


def test_the_winding_sign_actually_flips() -> None:
    """If this ever stops being true, D28 and FINDINGS.md section 2.1 are wrong."""
    real = _csv("manifold_null.csv")
    real = real[real.arm == "real"]
    a = real[real.num_steps == 64].obs_minus_null.mean()
    b = real[real.num_steps == 128].obs_minus_null.mean()
    assert a > 0 > b, f"expected a sign flip, got ns64={a:+.4f} ns128={b:+.4f}"


def test_null_calibration_arm_is_unbiased() -> None:
    """Without this the real arm is uninterpretable — it is why D28 is citable."""
    df = _csv("manifold_null.csv")
    cal = df[df.arm == "calibration"]
    rate = (cal.p_value < 0.05).mean()
    assert 0.0 <= rate <= 0.15, f"calibration arm beats its null at {rate:.1%}, expected ~5%"


def test_permutation_result_is_still_null() -> None:
    """FINDINGS.md section 2.2 states no stratum survives correction."""
    assert not _csv("winding_permutation.csv").bh_pass.any()


def test_register_sign_consistency_still_holds() -> None:
    """16/16 per condition is what makes v a direction rather than noise."""
    df = _csv("register_analysis.csv")
    for kind, g in df.groupby("kind"):
        assert (g.register_r > 0).all(), f"{kind} lost sign consistency"


# --- documents must carry the results they are responsible for ------------


@pytest.mark.parametrize(
    "name",
    ["FINDINGS.md", "state_of_knowledge.md", "claims_ledger.md", "backlog_not_done.md"],
)
def test_key_documents_exist(name: str) -> None:
    assert _doc(name).strip(), f"{name} is empty"


def test_findings_states_its_negatives() -> None:
    """A findings document that reports only positives would be dishonest here —
    three of the project's four hypotheses-as-operationalised came out negative."""
    doc = _doc("FINDINGS.md")
    for phrase in ("Negative results", "does not measure rotation",
                   "does not track difficulty", "never tested"):
        assert phrase in doc, f"FINDINGS.md no longer states: {phrase!r}"


def test_findings_retains_the_honesty_section() -> None:
    """Six retractions, the leakage estimate and the modest effect sizes are
    part of the result, not decoration."""
    doc = _doc("FINDINGS.md")
    for phrase in ("retracted", "leakage", "modest"):
        assert phrase in doc.lower(), f"FINDINGS.md dropped: {phrase!r}"


# --- the ledger's own integrity -------------------------------------------


def test_claims_ledger_validates() -> None:
    """Duplicate ids, missing cells, dangling citations and unpointed retractions.

    With 55+ claims that supersede and cite one another, a typo'd "D34" or a second
    "D41" is invisible in prose and makes every downstream citation ambiguous.
    """
    from scripts.ledger import validate

    problems = validate()
    assert not problems, "claims_ledger.md problems:\n  " + "\n  ".join(problems)


def test_every_retraction_points_somewhere() -> None:
    """A retraction with no replacement leaves a reader stranded mid-argument."""
    import re

    from scripts.ledger import rows

    for cid, body in rows():
        head = body.split(" | ")[0]
        if re.search(r"\b(RETRACTED|SUPERSEDED)\b", head):
            assert re.search(r"\bD\d+\b", head) or "see (" in head.lower(), (
                f"{cid} is retracted in its headline without naming what replaced it"
            )
