"""Validation of the variance decomposition, before geomcap's data exists.

The load-bearing test is `test_a_perfect_relation_cannot_beat_its_own_ceiling`:
if the ceiling can be exceeded by a real correlation, it is not a bound and
quoting it beside a null would be worse than quoting nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from traj_geom.analysis.reliability import (
    attenuation_ceiling,
    decompose,
    family_mean_reliability,
    icc_one_way,
)


def _planted(n_fam=21, m=24, sd_fam=1.0, sd_item=1.0, sd_h0=1.0, seed=0):
    """A `main` block with known variance components, plus family truths."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, sd_fam, n_fam)
    vals, fams = [], []
    for i in range(n_fam):
        for _ in range(m):
            vals.append(truth[i] + rng.normal(0, sd_item) + rng.normal(0, sd_h0))
            fams.append(f"f{i}")
    return np.array(vals), np.array(fams), truth


def _replicates(n_prompt=8, r=10, sd_h0=1.0, seed=1):
    """A `rep` block: same prompt r times, only h_0 differing."""
    rng = np.random.default_rng(seed)
    vals, keys = [], []
    for j in range(n_prompt):
        base = rng.normal(0, 5.0)                    # prompts sit far apart
        for _ in range(r):
            vals.append(base + rng.normal(0, sd_h0))
            keys.append(f"p{j}")
    return np.array(vals), np.array(keys)


def test_recovers_planted_components() -> None:
    """Unbiasedness over many draws, NOT accuracy on one.

    A between-group variance estimated from 21 groups has a relative standard error
    of sqrt(2/20) = 0.32, so a single-draw tolerance either passes vacuously or
    fails on an estimator that is perfectly correct. Averaging is what makes this a
    test of the estimator rather than of the seed.
    """
    bs, ws = [], []
    for s in range(40):
        v, f, _ = _planted(sd_fam=2.0, sd_item=1.0, sd_h0=0.5, seed=s)
        out = icc_one_way(v, f)
        assert out["usable"]
        bs.append(out["sigma2_between"])
        ws.append(out["sigma2_within"])
    assert float(np.mean(bs)) == pytest.approx(4.0, rel=0.15)
    assert float(np.mean(ws)) == pytest.approx(1.25, rel=0.05)


def test_between_group_variance_is_clipped_not_negative() -> None:
    """MSB < MSW happens by chance when the true between-group variance is 0.

    A negative variance passed on as-is produces a negative reliability and then a
    NaN ceiling from the sqrt, which reads as "no bound available" exactly when the
    bound is tightest (zero).
    """
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 200)
    f = np.repeat([f"g{i}" for i in range(20)], 10)
    out = icc_one_way(v, f)
    assert out["sigma2_between"] >= 0.0
    assert 0.0 <= out["icc"] <= 1.0


def test_refuses_a_design_with_no_replication() -> None:
    """One observation per group leaves the within term unidentified."""
    v = np.arange(6.0)
    f = np.array([f"g{i}" for i in range(6)])
    out = icc_one_way(v, f)
    assert out["usable"] is False


def test_reliability_rises_with_items_per_family() -> None:
    """Spearman-Brown: averaging more items shrinks noise but not signal."""
    r1 = family_mean_reliability(1.0, 4.0, m=1)
    r4 = family_mean_reliability(1.0, 4.0, m=4)
    r64 = family_mean_reliability(1.0, 4.0, m=64)
    assert r1 == pytest.approx(0.2)
    assert r4 == pytest.approx(0.5)
    assert r1 < r4 < r64 < 1.0


def test_ceiling_is_one_only_when_measurement_is_perfect() -> None:
    assert attenuation_ceiling(1.0, 1.0) == pytest.approx(1.0)
    assert attenuation_ceiling(0.25, 1.0) == pytest.approx(0.5)
    assert attenuation_ceiling(0.0, 1.0) == 0.0


def test_the_ceiling_is_the_centre_of_a_perfect_relation_not_its_bound() -> None:
    """THE POINT OF THE MODULE, and a correction to my first draft of it.

    Correlate noisy family means against the TRUE family values -- a relation that
    is perfect by construction -- and the observed |rho| must sit AROUND
    `attenuation_ceiling`, half above and half below. The first version of this
    test asserted `rho <= ceiling` and failed 15 times in 40, which is the correct
    behaviour of a correct estimator: sqrt(reliability) is where a perfect relation
    lands in EXPECTATION, and at N=21 the sampling spread is ~0.1 either way.

    Calling it a hard bound would have understated how much signal a design can
    show -- the same mistake, in the opposite direction, as quoting a bare rho.
    """
    trials, rhos, ceils = 60, [], []
    for s in range(trials):
        v, f, truth = _planted(n_fam=21, m=8, sd_fam=1.0, sd_item=2.0,
                               sd_h0=2.0, seed=100 + s)
        d = decompose(v, f, m=8, n_boot=0)
        means = np.array([v[f == f"f{i}"].mean() for i in range(21)])
        rhos.append(abs(float(spearmanr(means, truth)[0])))
        ceils.append(d["ceiling"])
    # Both averaged: the ESTIMATED ceiling is itself noisy at 21 families (one draw
    # gave 0.450 where the truth is 0.707), so a single-draw comparison would test
    # the seed, not the estimator.
    assert float(np.mean(rhos)) == pytest.approx(float(np.mean(ceils)), abs=0.12), (
        f"perfect relation averaged {np.mean(rhos):.3f} against mean ceiling "
        f"{np.mean(ceils):.3f}")
    above = sum(r > c for r, c in zip(rhos, ceils, strict=True))
    assert 0.15 * trials < above < 0.85 * trials, (
        f"{above}/{trials} above the ceiling -- it is not centred")


def test_the_band_covers_a_perfect_relation() -> None:
    """`perfect_relation_band` is the number that IS safe to compare against.

    Nominal 95%; the band is built on the ESTIMATED reliability, which itself
    varies, so the achieved rate is checked rather than assumed.
    """
    trials, hits = 60, 0
    for s in range(trials):
        v, f, truth = _planted(n_fam=21, m=8, sd_fam=1.0, sd_item=2.0,
                               sd_h0=2.0, seed=500 + s)
        d = decompose(v, f, m=8, n_levels=21, n_boot=0)
        means = np.array([v[f == f"f{i}"].mean() for i in range(21)])
        rho = abs(float(spearmanr(means, truth)[0]))
        hits += d["band_lo"] <= rho <= d["band_hi"]
    assert hits / trials >= 0.75, f"band covered {hits}/{trials}, nominal 95%"


def test_the_wide_band_covers_more_than_the_narrow_one() -> None:
    """`wide_*` adds the reliability's OWN sampling noise, which is the larger term.

    At 21 families the between-family variance carries a relative standard error of
    ~0.32, so the ceiling that centres the band moves more than the correlation
    around it does. A band built only on the correlation's noise covered 75-95% of
    perfect relations in the test above; this one has to be strictly wider or the
    bootstrap is not contributing anything.
    """
    v, f, _ = _planted(n_fam=21, m=8, sd_fam=1.0, sd_item=2.0, sd_h0=2.0, seed=3)
    d = decompose(v, f, m=8, n_levels=21, n_boot=200)
    assert d["rel_lo"] < d["reliability"] < d["rel_hi"]
    assert d["wide_lo"] <= d["band_lo"] and d["wide_hi"] >= d["band_hi"]
    assert (d["wide_hi"] - d["wide_lo"]) > (d["band_hi"] - d["band_lo"])


def test_the_band_excludes_a_real_null() -> None:
    """Non-suppression: when the truth is NO relation, the band must not contain it.

    A band wide enough to cover everything would make the bounded null vacuous --
    it has to be able to say "this observation is not what a perfect relation looks
    like", which is the entire claim geomcap rests on.
    """
    rng = np.random.default_rng(0)
    v, f, _ = _planted(n_fam=21, m=8, sd_fam=1.0, sd_item=2.0, sd_h0=2.0, seed=9)
    d = decompose(v, f, m=8, n_levels=21)
    means = np.array([v[f == f"f{i}"].mean() for i in range(21)])
    unrelated = rng.normal(0, 1, 21)
    rho = abs(float(spearmanr(means, unrelated)[0]))
    assert rho < d["band_lo"], (
        f"an unrelated variable scored {rho:.3f}, inside the perfect-relation band "
        f"[{d['band_lo']:.3f}, {d['band_hi']:.3f}]")


def test_a_high_reliability_design_yields_a_ceiling_near_one() -> None:
    """Non-suppression: the bound must not cry wolf when measurement is good.

    A ceiling that is always low would make every null unreadable, which is the
    mirror image of the failure it guards against.
    """
    v, f, _ = _planted(n_fam=21, m=24, sd_fam=3.0, sd_item=0.3, sd_h0=0.3, seed=7)
    d = decompose(v, f)
    assert d["reliability"] > 0.95
    assert d["ceiling"] > 0.97


def test_replicate_block_attributes_the_noise_to_h0() -> None:
    """With var(item)=0 the whole within-family term must land on h_0."""
    v, f, _ = _planted(n_fam=21, m=24, sd_fam=2.0, sd_item=0.0, sd_h0=1.0, seed=11)
    rv, rk = _replicates(sd_h0=1.0, seed=12)
    d = decompose(v, f, rv, rk)
    assert d["h0_share"] == pytest.approx(1.0, abs=0.25)
    assert d["sigma2_item"] == pytest.approx(0.0, abs=0.3)


def test_replicate_block_leaves_item_variance_when_there_is_some() -> None:
    """And with a real item term, h_0 must NOT absorb it."""
    v, f, _ = _planted(n_fam=21, m=24, sd_fam=2.0, sd_item=2.0, sd_h0=0.5, seed=13)
    rv, rk = _replicates(sd_h0=0.5, seed=14)
    d = decompose(v, f, rv, rk)
    assert d["h0_share"] < 0.25
    assert d["sigma2_item"] > 3.0


def test_decompose_without_replicates_still_bounds() -> None:
    """The `rep` block is optional; the pooled within term is a valid ceiling."""
    v, f, _ = _planted(seed=17)
    d = decompose(v, f)
    assert d["usable"]
    assert np.isnan(d["sigma2_h0"]) and np.isnan(d["sigma2_item"])
    assert 0.0 < d["ceiling"] <= 1.0


def test_the_d70_shape_is_flagged_as_a_low_ceiling() -> None:
    """The concrete failure this module exists to have caught.

    D70 correlated four content gaps that could not exceed 0.129 in magnitude and
    printed "CONFIRMED" from rho = +0.95. Here the same shape -- family signal far
    below measurement noise -- must return a ceiling that makes any observed rho
    unreadable, so the number cannot be quoted bare.
    """
    v, f, _ = _planted(n_fam=21, m=24, sd_fam=0.05, sd_item=1.0, sd_h0=1.0, seed=23)
    d = decompose(v, f)
    assert d["ceiling"] < 0.75, (
        f"ceiling {d['ceiling']:.3f} would let a noise correlation read as real")
