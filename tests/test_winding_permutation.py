"""Pins the permutation-test result AND the ledger text that cites it.

This closes backlog item 6.4: until now nothing checked that a number quoted in
`claims_ledger.md` still matched the CSV it was derived from, so the two could
drift apart silently. The tests below fail if either the data changes or the
prose stops describing it.

The result being pinned is a NULL, which is exactly the kind that erodes
quietly: a later re-run with different settings could produce a nominal hit and
nobody would notice the ledger still says "null".
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "winding_permutation.csv")
LEDGER = os.path.join(ROOT, "docs", "claims_ledger.md")


@pytest.fixture(scope="module")
def perm() -> pd.DataFrame:
    if not os.path.exists(CSV):
        pytest.skip("winding_permutation.csv not present (run scripts.run_winding_permutation)")
    return pd.read_csv(CSV)


def test_schema_is_what_the_ledger_describes(perm: pd.DataFrame) -> None:
    for col in ("task", "num_steps", "n_levels", "rho", "p_perm", "bh_crit", "bh_pass"):
        assert col in perm.columns, f"missing column {col}"


def test_all_four_strata_present(perm: pd.DataFrame) -> None:
    """Stratification is load-bearing: pooling produced a spurious result before."""
    got = {(r.task, int(r.num_steps)) for r in perm.itertuples()}
    assert got == {
        ("count_ones", 64), ("count_ones", 128),
        ("projection", 64), ("projection", 128),
    }


def test_no_stratum_survives_multiplicity_correction(perm: pd.DataFrame) -> None:
    """THE claim. If this ever fails, D26(1) must be rewritten, not the test."""
    assert not perm.bh_pass.any(), (
        f"a stratum now survives BH: {perm[perm.bh_pass].to_dict('records')} -- "
        "update claims_ledger.md D26(1), which states the result is null"
    )
    assert (perm.p_perm * len(perm)).min() > 0.05, "Bonferroni now passes"


def test_signs_disagree_across_strata(perm: pd.DataFrame) -> None:
    """Cited in D26(1) as part of why the nominal hit is not compelling."""
    assert (perm.rho > 0).any() and (perm.rho < 0).any()


def test_ledger_numbers_match_the_csv(perm: pd.DataFrame) -> None:
    """Backlog 6.4: prose and data must not drift apart.

    D26(1) quotes each stratum's rho and p. Check the two most load-bearing
    figures verbatim against the file.
    """
    if not os.path.exists(LEDGER):
        pytest.skip("ledger not present")
    text = open(LEDGER, encoding="utf-8").read()
    assert "| D26 |" in text, "D26 row missing from the ledger"

    row = perm[(perm.task == "count_ones") & (perm.num_steps == 128)].iloc[0]
    assert f"{row.rho:.3f}" in text, (
        f"ledger does not quote count_ones ns=128 rho={row.rho:.3f}"
    )
    assert f"{row.p_perm:.4f}" in text, (
        f"ledger does not quote its p={row.p_perm:.4f}"
    )
    # and the conclusion itself must still be stated
    assert "NULL UNDER THE ASSUMPTION-FREE TEST" in text.upper()


def test_permutation_count_is_adequate(perm: pd.DataFrame) -> None:
    """A p-value of 0.0267 needs enough permutations to be resolved at all."""
    smallest = perm.p_perm.min()
    assert smallest > 1.0 / 20_000, (
        "smallest p is at the resolution limit of the permutation count; "
        "increase N_PERM before quoting it"
    )
