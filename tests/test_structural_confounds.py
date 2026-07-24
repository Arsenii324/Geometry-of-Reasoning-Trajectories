"""Regression-pins claims_ledger.md D20 against the real cached CSVs:
(a) the Blayney loops all sit in the last third of the prompt;
(b) the modk 'clean' design has a residual confound (irrelevant_len and
    neutral_len are perfectly anti-dependent at fixed active_len).
"""

from __future__ import annotations

import pandas as pd
import pytest


def test_d20a_blayney_loops_cluster_in_last_third() -> None:
    """All Long-Persona loop events fall in the last third of the prompt;
    none in the first two-thirds.
    """
    df = pd.read_csv("results/blayney_repro.csv")
    lp = df[df["condition"] == "long_persona"].copy()
    loops = lp[lp["shape"] == "loop"].copy()
    loops["rel_pos"] = loops["token_position"] / (loops["n_tokens"] - 1)

    assert len(loops) == 7
    assert loops["rel_pos"].min() >= 0.80  # last fifth (min is 0.8192)
    assert loops["rel_pos"].max() <= 0.90
    # Per-third counts: [0, 0, 7].
    early = (loops["rel_pos"] < 1 / 3).sum()
    mid = ((loops["rel_pos"] >= 1 / 3) & (loops["rel_pos"] < 2 / 3)).sum()
    late = (loops["rel_pos"] >= 2 / 3).sum()
    assert (int(early), int(mid), int(late)) == (0, 0, 7)


def test_d20b_modk_residual_confound_irrelevant_vs_neutral() -> None:
    """In the extended modk sweep, at fixed active_len, irrelevant_len and
    neutral_len are exact perfect linear opposites (corr = -1.0) -- so the
    two cannot be attributed independently. Also confirms the total_len=36
    constant (adversarial-verify correction to the ledger's earlier '54').
    """
    df = pd.read_csv("results/three_scale_modk_extended.csv")
    # neutral = total - active - irrelevant, exactly, for every row.
    residual = df["neutral_len"] + df["active_len"] + df["irrelevant_len"] - df["total_len"]
    assert residual.abs().max() == 0
    # total_len (design param) is fixed at 36; seq_len (tokenized) is the 54.
    assert df["total_len"].nunique() == 1
    assert df["total_len"].iloc[0] == 36
    assert df["seq_len"].nunique() == 1
    assert df["seq_len"].iloc[0] == 54
    # At each fixed active_len, irrelevant_len and neutral_len are corr = -1.0.
    for _, grp in df.groupby("active_len"):
        if grp["irrelevant_len"].nunique() > 1:  # need variation to define a corr
            corr = grp["irrelevant_len"].corr(grp["neutral_len"])
            assert corr == pytest.approx(-1.0, abs=1e-9)
