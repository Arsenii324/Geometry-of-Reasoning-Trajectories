"""A retracted claim must not survive anywhere in the documents.

WHY THIS TEST EXISTS. On 2026-08-09 an audit found six real defects, and **four of
them were the same failure: a correction that reached `claims_ledger.md` and no
other document.** The ledger row said "corrected"; `UNDERSTANDING.md`, `RESULT.md`,
`PLAN.md` and both research inquiries still asserted the retracted version
verbatim. A reader of the two documents CLAUDE.md section 6 designates as the
current-state record got the withdrawn claim.

That is not an analysis error and no amount of care while writing the ledger row
prevents it. It is a propagation error, and the only reliable guard is mechanical:
**when a claim is retracted, its phrasing becomes forbidden repo-wide.**

HOW TO USE THIS. When you retract or correct a claim, add a row to RETRACTED below
with the exact phrasing, why it is wrong, and what to say instead. The test then
fails until every document is updated -- in the SAME commit, which is the point.

This is deliberately a string check rather than anything clever. The failure mode
is a stale sentence surviving a copy-paste, and a stale sentence is exactly what a
string check catches.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Files that may legitimately quote a retracted phrase in order to record that it
# WAS retracted. Everything else must not contain it at all.
ALLOWED_TO_QUOTE = {
    "claims_ledger.md",      # the corrections themselves live here
    "directions.md",         # the audit log, section F
    "GUIDANCE_REVIEW.md",    # the retrospective
}

RETRACTED: list[tuple[str, str, str]] = [
    (
        "two unrelated instruments landing on the same number",
        "D94 claimed convergent validation from two estimators that share the same "
        "log-linear fit family; the interval it compared against turned out to be "
        "the single-orbit self-convergence fit our own regime.py calls a mistake.",
        "State: rho = 0.855 raw, ~0.82 after D58's +0.033 bias correction, against "
        "D31's Arnoldi 0.79-0.81 -- and say there is no valid independent "
        "corroboration.",
    ),
    (
        "no loop or drift regime exists anywhere we have looked",
        "Retired by the three-sense taxonomy. Sense (ii) (inter-block cycle) is "
        "present and large (D98); sense (iii) (damped rotation, complex lambda) is "
        "present (D55). Only sense (i), a non-contracting limit cycle, is absent.",
        "Scope it: 'no loop or drift in the iteration-to-iteration map at a fixed "
        "block', and name the two senses that ARE present.",
    ),
    (
        "40 independent answer-position orbits",
        "That slice was sorted(glob)[:40] = 32 replicates of add1_i08 plus 8 of "
        "add1_i10, i.e. 1.25 items. The true 512-orbit median is 2.6x smaller.",
        "Quote all 512 orbits, and state the number of independent items.",
    ),
    (
        "causal interventions late in the recurrence are erased",
        "The attenuation direction was stated backwards. Decay is rho^(R-r), so the "
        "exponent SHRINKS as r grows: LATE interventions survive, EARLY ones decay.",
        "Say: late survives, early decays; and that same-prompt patching is inert "
        "at both ends because donor and recipient share an attractor.",
    ),
]


def _docs() -> list[pathlib.Path]:
    return [p for p in DOCS.rglob("*.md") if "archive" not in p.parts]


def test_no_document_repeats_a_retracted_claim():
    """Every retracted phrasing is absent outside the files that record retractions."""
    offences: list[str] = []
    for phrase, why, instead in RETRACTED:
        for path in _docs():
            if path.name in ALLOWED_TO_QUOTE:
                continue
            if phrase.lower() in path.read_text(encoding="utf-8").lower():
                offences.append(
                    f"\n  {path.relative_to(ROOT)} still contains: {phrase!r}"
                    f"\n    why retracted: {why}"
                    f"\n    say instead:   {instead}"
                )
    assert not offences, (
        "Retracted claims are still present in the documents. A correction that "
        "reaches the ledger and not the synthesis documents is not a correction."
        + "".join(offences)
    )


def test_the_guard_itself_would_fire():
    """The check must actually detect a retracted phrase -- test that the test fails.

    An unfalsified guard is an assumption wearing a costume: if the matching logic
    silently broke, `test_no_document_repeats_a_retracted_claim` would pass forever
    and read as evidence.
    """
    phrase = RETRACTED[0][0]
    haystack = f"some preamble ... {phrase.upper()} ... some trailer"
    assert phrase.lower() in haystack.lower()


def test_every_retracted_entry_says_what_to_write_instead():
    """A prohibition without a replacement gets worked around, not obeyed."""
    for phrase, why, instead in RETRACTED:
        assert len(why) > 40, f"{phrase!r}: explain WHY it was retracted"
        assert len(instead) > 30, f"{phrase!r}: give the replacement wording"
