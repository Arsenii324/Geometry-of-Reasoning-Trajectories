"""The depth-profile analysis must not manufacture the effect it reports.

Every test here targets a way the headline could be an artefact rather than a
result: pooling clustered orbits, choosing the peak after seeing the data, reading
a length effect as a containment effect, or letting a token substring match.
"""

from __future__ import annotations

import numpy as np
import pytest
from scripts.run_depth_profile import (
    PEAK_R,
    _contains,
    containment_with_decoy,
    h0_forgotten,
    oracle_gap,
    paired_depth_test,
    preamble_split,
)


def _curves(per_family: dict[str, list[list[int]]]) -> dict[str, np.ndarray]:
    return {f: np.asarray(v, dtype=float) for f, v in per_family.items()}


def test_oracle_exceeds_every_fixed_depth_by_construction() -> None:
    """`min_r rank == 1` can never be beaten by a fixed depth. It is an upper bound.

    This is the arithmetic behind the headline: the project's `correct` label is a
    minimum over unrolls, so reporting it as capability credits the model for
    unrolls no user could have selected.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = _curves({"a": rng.integers(1, 6, size=(9, 12)).tolist(),
                     "b": rng.integers(1, 6, size=(9, 12)).tolist()})
        g = oracle_gap(c)
        assert g["oracle_acc"] >= g["best_fixed_acc"] - 1e-12
        assert g["best_fixed_acc"] >= g["final_acc"] - 1e-12


def test_paired_test_unit_is_the_family_not_the_orbit() -> None:
    """Duplicating every orbit within a family must not change the p-value.

    The 504 orbits are 21 clusters of 24 drawn from one generator each. A pooled
    binomial would read cluster size as evidence; this test fails loudly if the
    family ever stops being the unit.
    """
    base = {"a": [[1, 5]] * 3 + [[5, 5]] * 3, "b": [[1, 5]] * 4, "c": [[5, 1]] * 4,
            "d": [[1, 5]] * 2, "e": [[1, 5]] * 6}
    doubled = {k: v * 2 for k, v in base.items()}
    p1 = paired_depth_test(_curves(base), r=1, ref=2)
    p2 = paired_depth_test(_curves(doubled), r=1, ref=2)
    assert p1["p_sign"] == pytest.approx(p2["p_sign"])
    assert p1["n_fam"] == p2["n_fam"] == 5


def test_paired_test_is_null_when_depth_does_nothing() -> None:
    """Identical accuracy at both depths must give every family tied and p = 1.

    A sign test that dropped ties incorrectly would divide by zero or report a
    spurious p here.
    """
    c = _curves({chr(97 + i): [[1, 1], [4, 4], [9, 9]] for i in range(8)})
    t = paired_depth_test(c, r=1, ref=2)
    assert (t["better"], t["worse"], t["tied"]) == (0, 0, 8)
    assert t["p_sign"] == 1.0


def test_paired_test_is_calibrated_under_exchangeable_depths() -> None:
    """Under a true null the test must fire at about its nominal rate, not always.

    Asserting `p > 0.05` on a single seed is the mistake this project has already
    made twice: a calibrated test fires one time in twenty by design. The check is
    on the rate over many draws.
    """
    rng = np.random.default_rng(7)
    fired = 0
    trials = 300
    for _ in range(trials):
        c = _curves({chr(97 + i): rng.integers(1, 4, size=(24, 2)).tolist()
                     for i in range(21)})
        fired += paired_depth_test(c, r=1, ref=2)["p_sign"] < 0.05
    assert 0.005 < fired / trials < 0.12, fired / trials


def test_peak_depth_is_fixed_a_priori_not_refitted() -> None:
    """PEAK_R is a module constant, so it cannot be re-chosen per family.

    Selecting the best unroll inside each family and then testing it would be a
    winner's-curse estimate; the reported headline must use one unroll fixed in
    advance from the pooled curve.
    """
    assert isinstance(PEAK_R, int)
    # Family "a" peaks at unroll 2 and "b" at unroll 1; neither is PEAK_R. The test
    # must still read column PEAK_R for both, scoring them at an unroll that is
    # optimal for neither -- that is exactly what "fixed a priori" has to mean.
    n = PEAK_R + 2
    a = [[9] * n for _ in range(4)]
    b = [[9] * n for _ in range(4)]
    for row in a:
        row[1] = 1
    for row in b:
        row[0] = 1
    t = paired_depth_test(_curves({"a": a, "b": b}), r=PEAK_R, ref=n)
    assert t["r"] == PEAK_R
    assert (t["better"], t["worse"]) == (0, 0), "neither family is top-1 at PEAK_R"


def test_containment_control_absorbs_a_pure_length_effect() -> None:
    """If outputs merely get longer, gold and decoy must gain together.

    Specificity is the statistic that survives this: it stays ~1 when length alone
    changes and only rises when the gain is specific to the correct answer.
    """
    rows = []
    for depth, filler in ((2, ""), (32, " x" * 40)):
        for i, g in enumerate("1 2 3 4".split()):
            rows.append({"family": "f", "item": i, "fmt": "bare", "depth": depth,
                         "gold": g, "output": "1 2 3 4" + filler})
    df = containment_with_decoy(rows).set_index("depth")
    assert df.loc[32, "out_len"] > df.loc[2, "out_len"] * 3
    assert df.loc[2, "specificity"] == pytest.approx(df.loc[32, "specificity"])
    assert df.loc[32, "specificity"] == pytest.approx(1.0)


def test_containment_control_lets_a_real_specific_gain_through() -> None:
    """The control must not be so strict that a genuine effect cannot show.

    A null instrument that also refuses the positive case is D70's failure mode.
    """
    rows = []
    for depth in (2, 32):
        for i, g in enumerate("1 2 3 4".split()):
            out = g if depth == 32 else "9"
            rows.append({"family": "f", "item": i, "fmt": "bare", "depth": depth,
                         "gold": g, "output": out})
    df = containment_with_decoy(rows).set_index("depth")
    assert df.loc[32, "contains_gold"] == 1.0
    assert df.loc[32, "contains_decoy"] == 0.0
    # A zero decoy rate must not evaluate to NaN and drop the best depth out of the
    # column; it is floored at the measurement's resolution and flagged as such.
    assert bool(df.loc[32, "decoy_at_floor"])
    assert np.isfinite(df.loc[32, "specificity"])
    assert df.loc[32, "specificity"] > df.loc[2, "specificity"]


def test_containment_matches_whole_tokens_only() -> None:
    """"4" must not match inside "42" or "user4", or containment counts noise.

    The generations are chatty and full of digits; a substring match would score
    "The answer is 42." as containing 4 and 2 both.
    """
    assert _contains("the answer is 4.", "4")
    assert _contains("4user\n", "4")
    assert not _contains("the answer is 42", "4")
    assert not _contains("user4x", "4")
    assert _contains("Repeat: apple!", "apple")
    assert not _contains("applesauce", "apple")


def test_h0_reports_converged_and_transient_separately() -> None:
    """A prompt whose converged rank is fixed but whose best rank moves is the case.

    Reporting only `best_rank` gives D90's original reading ("h_0 decides the
    answer"); reporting only the converged rank hides the transient entirely. The
    frame must carry both columns for the same prompt.
    """
    rows = [{"block": "rep", "ok": True, "family": "f", "item": 0,
             "rank_curve": [k, 1 if k == 1 else 40, 7], "best_rank": min(k, 7)}
            for k in (1, 3, 9)]
    df = h0_forgotten(rows)
    assert len(df) == 1
    assert int(df.final_rank_distinct.iloc[0]) == 1      # converged rank is fixed
    assert int(df.best_rank_distinct.iloc[0]) > 1        # the transient is not
    assert df.best_rank_ratio.iloc[0] == pytest.approx(7.0)


def test_h0_reads_only_the_replicate_block() -> None:
    """Main-block orbits are one draw per item and must not enter the h_0 frame.

    Pooling them would compare different prompts and call the difference h_0.
    """
    rows = [{"block": "main", "ok": True, "family": "f", "item": i,
             "rank_curve": [1, 2], "best_rank": 1} for i in range(5)]
    assert h0_forgotten(rows).empty


def test_preamble_split_reports_both_arms_at_every_depth() -> None:
    """The refuted mechanism must stay auditable: both medians, both counts.

    It is kept in the output precisely because a reader will propose it; dropping
    the arm would leave the refutation unevidenced.
    """
    acc, rnk = [], []
    for d in (2, 32):
        for i in range(6):
            out = "The answer is 3" if (d == 32 or i % 2) else "3"
            acc.append({"family": "f", "item": i, "fmt": "bare", "depth": d,
                        "gold": "3", "output": out})
            rnk.append({"family": "f", "item": i, "fmt": "bare", "depth": d,
                        "rank": 7, "ok": True})
    df = preamble_split(acc, rnk).set_index("depth")
    assert df.loc[32, "n_preamble"] == 6
    assert df.loc[2, "n_preamble"] == 3 and df.loc[2, "n_plain"] == 3
    assert df.loc[32, "med_rank_preamble"] == pytest.approx(7.0)
    assert not np.isnan(df.loc[2, "med_rank_plain"])
