"""Pins Step 5's re-derivation result (claims_ledger D29).

The load-bearing claim is that the sign of `consecutive_step_cosine` is set by
the RECORDING LENGTH rather than by anything about the model: positive when the
budget is too short to reach the arithmetic floor, negative once it is long
enough. If that stops being true the audit's central argument needs revisiting.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "regime_rederivation.csv")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not os.path.exists(CSV):
        pytest.skip("regime_rederivation.csv not present (run the script)")
    return pd.read_csv(CSV)


def test_signal_fraction_falls_as_the_budget_grows(df: pd.DataFrame) -> None:
    """Recording longer adds arithmetic, not information."""
    frac = df.groupby("num_steps")["frac_signal"].mean()
    assert frac.is_monotonic_decreasing, f"signal fraction not decreasing: {frac.to_dict()}"
    assert frac.loc[128] < 0.25, "at ns=128 well under a quarter should be signal"


def test_legacy_cosine_sign_follows_the_recording_length(df: pd.DataFrame) -> None:
    """THE claim of D29(2)."""
    by_ns = df.groupby("num_steps")["cos_legacy"].mean()
    assert by_ns.loc[16] > 0, "at ns=16 there is no floor to contaminate the second half"
    assert by_ns.loc[64] < 0 and by_ns.loc[128] < 0, "with a floor present it must go negative"


def test_regime_restriction_flips_most_trajectories(df: pd.DataFrame) -> None:
    flipped = ((df.cos_legacy < 0) & (df.cos_regime > 0)).mean()
    assert flipped > 0.8, f"only {flipped:.1%} flip; D29 reports 90.3%"


def test_conv_rate_legacy_understates_contraction(df: pd.DataFrame) -> None:
    d = df[["conv_rate_legacy", "conv_rate_regime"]].dropna()
    assert d.conv_rate_regime.mean() < d.conv_rate_legacy.mean(), (
        "the floor-aware fit must be steeper (more negative)"
    )


def test_dlr_legacy_is_inflated(df: pd.DataFrame) -> None:
    d = df[["dlr_legacy", "dlr_regime"]].dropna()
    assert d.dlr_legacy.mean() > 2 * d.dlr_regime.mean(), "D29 reports ~4x inflation"


def test_regime_length_is_roughly_budget_independent(df: pd.DataFrame) -> None:
    """~19 signal steps whether you record 64 or 128 — the mechanism behind D29(1)."""
    ends = df[df.num_steps >= 64].groupby("num_steps")["regime_end"].mean()
    assert ends.max() / ends.min() < 1.5, (
        f"regime length varies too much with budget: {ends.to_dict()}"
    )
