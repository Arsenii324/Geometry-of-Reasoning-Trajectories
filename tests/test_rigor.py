"""Every guard is tested against THE ACTUAL HISTORICAL BAD INPUT that motivated it.

This is the `numerical-research-code` discipline of "test that your test fails",
applied literally: for each guard, the test asserts it FIRES on the real numbers
from the real failure, and PASSES on the corrected numbers that were actually
deployed. A guard that only fires on invented input is an assumption in a costume.
"""

import pytest

from traj_geom.rigor import (
    RigorError,
    describe_range,
    exact_floor,
    independent_n,
    require_evidence,
    require_exact_test_if_small,
    require_resolvable_alpha,
    require_units,
)

# --- failure mode 1: a test that cannot fire (D93's detection floor) ---------

def test_the_real_h0within_floor_config_is_rejected():
    """40 permutations against Bonferroni alpha = 0.05/8: the actual bug.

    This configuration reported 0% detection at every planted effect size
    including 1.0 sd, which read as a hopeless design and was in fact an
    arithmetic impossibility.
    """
    with pytest.raises(RigorError, match="UNREACHABLE"):
        require_resolvable_alpha(n_perm=40, alpha=0.05 / 8, what="P8 detection floor")


def test_the_deployed_fix_is_accepted():
    """400 permutations against the same alpha -- the fix that was shipped.

    After this change the floor detected a planted 0.5 sd effect 100% of the time,
    which is what made D93's NOT-CONFIRMED verdict trustworthy.
    """
    require_resolvable_alpha(n_perm=400, alpha=0.05 / 8, what="P8 detection floor")


def test_the_error_names_the_n_perm_that_would_work():
    with pytest.raises(RigorError) as e:
        require_resolvable_alpha(n_perm=40, alpha=0.00625)
    assert "159" in str(e.value)          # ceil(1/0.00625) - 1 = 159


def test_exact_floor_matches_the_projects_p_convention():
    # p = (hits + 1) / (n_perm + 1); minimum at hits = 0
    assert exact_floor(400) == pytest.approx(1 / 401)
    assert exact_floor(40) == pytest.approx(1 / 41)


# --- failure mode 2: a p-value below its own exact floor (D96) ---------------

def test_the_real_qk_permutation_config_is_rejected():
    """12 sign-flip pairs sampled 20000 times: the actual configuration.

    It reported p = 0.00015, below the exact floor 1/4096 = 0.000244, because
    (hits+1)/(N+1) with N=20000 over a 4096-point space manufactures resolution
    that does not exist.
    """
    with pytest.raises(RigorError, match="exact floor"):
        require_exact_test_if_small(n_pairs=12, n_perm=20000, what="QK paired perm")


def test_sampling_below_the_space_size_is_fine():
    require_exact_test_if_small(n_pairs=20, n_perm=20000)   # 2**20 >> 20000


# --- failure mode 3 & 4: replicates as observations; convenience slices ------

def test_the_real_d99_glob_slice_is_rejected():
    """`sorted(glob(rep_*.npy))[:40]` was 32 of add1_i08 + 8 of add1_i10.

    Quoted in D99 as "40 independent answer-position orbits". It is 1.25 items,
    and the statistic it produced was 2.6x off the true 512-orbit population.
    """
    keys = ["add1_i08"] * 32 + ["add1_i10"] * 8
    assert independent_n(keys) == 2
    with pytest.raises(RigorError, match="pseudoreplication"):
        require_units(keys, min_units=10, what="h0bank orbit sample")


def test_the_real_d94_pair_count_is_rejected():
    """"0 of ~960 orbit pairs" over 16 prompts: contraction is a property of e."""
    keys = [f"prompt_{i:02d}" for i in range(16) for _ in range(60)]
    assert len(keys) == 960
    assert independent_n(keys) == 16
    with pytest.raises(RigorError):
        require_units(keys, min_units=100, what="contraction pairs")
    # the honest claim -- 16 units -- passes
    assert require_units(keys, min_units=16, what="contraction pairs") == 16


def test_the_real_d96_clustered_contrasts_are_rejected():
    """24 'observations' that are 12 items x 2 contrasts sharing the A value."""
    keys = [f"item{i}" for i in range(12)] * 2
    assert independent_n(keys) == 12
    with pytest.raises(RigorError):
        require_units(keys, min_units=24, what="QK confound correlation")


def test_replicates_allowed_only_when_explicitly_acknowledged():
    keys = ["one_item"] * 40
    with pytest.raises(RigorError):
        require_units(keys, min_units=5, what="sample")
    assert require_units(keys, min_units=5, what="sample", allow_replicates=True) == 1


# --- failure mode 5: a verdict over no data (ds_cyclenull) -------------------

def test_the_real_cyclenull_empty_comparison_is_rejected():
    """All three untrained arms died of CUDA OOM; the comparison ran over 0/0.

    The script nonetheless printed "D98's cycle is ARCHITECTURAL, not learned".
    """
    with pytest.raises(RigorError, match="VOID, not NULL"):
        require_evidence(0, what="trained-vs-untrained comparison")


def test_a_real_comparison_passes():
    require_evidence(3, what="trained-vs-untrained comparison")


def test_the_void_message_says_the_question_is_open_not_answered():
    with pytest.raises(RigorError) as e:
        require_evidence(0, what="control")
    msg = str(e.value)
    assert "VOID" in msg and "different finding" in msg


# --- ranges cannot be quoted without their unit count -----------------------

def test_the_real_d98_residual_range_carries_its_unit_count():
    """D98 quoted "0.09-0.11" from one prompt's four per-block values.

    The true across-prompt range was 0.052-0.402 over six prompts. The formatted
    string makes the one-unit case impossible to quote innocently.
    """
    one_prompt = describe_range([0.0999, 0.0959, 0.0949, 0.1065],
                                ["echo_digit"] * 4, what="per-block residual")
    assert "1 independent unit" in one_prompt

    six_prompts = describe_range(
        [0.1254, 0.0523, 0.2984, 0.1988, 0.4021, 0.1394],
        ["echo_digit", "add1", "count8", "parity8", "track8", "caesar1"],
        what="per-block residual")
    assert "6 independent unit" in six_prompts
    assert "0.0523" in six_prompts and "0.4021" in six_prompts


def test_describe_range_rejects_misaligned_inputs():
    with pytest.raises(RigorError, match="aligned"):
        describe_range([1.0, 2.0], ["a"], what="x")
    with pytest.raises(RigorError, match="empty"):
        describe_range([], [], what="x")


# --- the guards must be cheap enough to call unconditionally ----------------

def test_guards_are_cheap():
    import time
    keys = [f"p{i%16}" for i in range(960)]
    t0 = time.perf_counter()
    for _ in range(200):
        require_units(keys, min_units=16, what="x")
        require_resolvable_alpha(400, 0.00625)
        require_evidence(5, what="x")
    assert time.perf_counter() - t0 < 1.0
