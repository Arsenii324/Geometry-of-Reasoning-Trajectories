"""Regression-pins the verified Blayney positive-control result
(claims_ledger.md D14) against the real cached results/blayney_repro.csv,
so this project's first-ever observed real loops can't silently drift.
"""

from __future__ import annotations

import pandas as pd
import pytest


def test_blayney_repro_reproduces_verified_d14_result() -> None:
    """long_persona: 7/5445 loops (0.1286%), matching Blayney et al.'s own
    0.14% per-token rate. no_system_prompt: 1/1677 (0.0596%), consistent
    with (not precisely pinning) their 0.02% baseline at this small N.
    """
    df = pd.read_csv("results/blayney_repro.csv")

    long_persona = df[df["condition"] == "long_persona"]
    no_system = df[df["condition"] == "no_system_prompt"]

    assert len(long_persona) == 5445
    assert len(no_system) == 1677

    lp_loops = int((long_persona["shape"] == "loop").sum())
    ns_loops = int((no_system["shape"] == "loop").sum())
    assert lp_loops == 7
    assert ns_loops == 1

    lp_rate = lp_loops / len(long_persona)
    ns_rate = ns_loops / len(no_system)
    assert lp_rate == pytest.approx(0.001286, abs=0.00001)
    assert ns_rate == pytest.approx(0.000596, abs=0.00001)

    # The whole point of this experiment: the observed rate is close to
    # Blayney's own reported per-token rate for this exact condition
    # (0.14%, claims_ledger.md B9) -- not an exact match (small N), but
    # same order of magnitude and not wildly off.
    assert 0.0005 < lp_rate < 0.003

    # At least one loop spans multiple full turns -- qualitatively
    # different from every prior loop this project had seen (forceloop.csv,
    # all |winding|~=0.65, under one turn).
    max_winding = long_persona.loc[long_persona["shape"] == "loop", "winding"].max()
    assert max_winding > 1.0
