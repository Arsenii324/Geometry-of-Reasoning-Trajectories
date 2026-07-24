"""Regression-pins the verified BH-FDR sweep (claims_ledger.md D13) against
the real cached results/*.csv, so this correction can't silently drift.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.run_fdr_correction import _collect

from traj_geom.analysis.correlate import benjamini_hochberg


def test_fdr_correction_reproduces_verified_d13_result() -> None:
    """46 tests collected; 22 raw-significant; 20 survive project-wide BH-FDR;
    both 'prime target' anomalies (maxtask/dissociation-5seed winding~n_ops
    per-level, both rho=+-0.943) survive FDR itself.
    """
    tests = _collect()
    assert len(tests) == 46

    df = pd.DataFrame(tests)
    q, sig = benjamini_hochberg(df["p"].to_numpy())
    df["q"] = q
    df["sig"] = sig

    assert (df["p"] < 0.05).sum() == 22
    assert sig.sum() == 20

    prime_targets = df[df["test"].str.contains("winding~n_ops \\[per-level\\]")]
    maxtask_hit = prime_targets[prime_targets["family"] == "maxtask"].iloc[0]
    dissoc_hit = prime_targets[
        (prime_targets["family"] == "dissociation_5seed")
        & prime_targets["test"].str.startswith("local")
    ].iloc[0]

    assert maxtask_hit["stat"] == pytest.approx(0.943, abs=0.001)
    assert maxtask_hit["sig"]
    assert dissoc_hit["stat"] == pytest.approx(-0.943, abs=0.001)
    assert dissoc_hit["sig"]

    # The 15-seed replication of the same dissociation hypothesis is the
    # actual disqualifier -- fails to replicate, unlike the 5-seed draw.
    dissoc_15 = df[
        (df["family"] == "dissociation_15seed")
        & df["test"].str.startswith("local winding~n_ops [per-level]")
    ].iloc[0]
    assert dissoc_15["p"] > 0.05
    assert not dissoc_15["sig"]
