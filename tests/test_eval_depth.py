"""Tests for the depth-threshold estimator.

`depth_curve` needs a GPU and the real model, so it is not exercised here.
`threshold_depth` is the part that carries the scientific claim -- it decides
what counts as "solved at depth r" -- and it is pure, so it is tested against
curves whose answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.eval_depth import _logsumexp, answer_surface_forms, threshold_depth


@pytest.fixture(scope="module")
def tok():
    """The pinned Huginn tokenizer, from cache; skip if unavailable."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer

        from traj_geom.constants import MODEL_ID, MODEL_REVISION

        return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    except Exception as exc:  # noqa: BLE001 - any failure means "not available here"
        pytest.skip(f"tokenizer unavailable offline: {exc}")


# --- tokenization: P(answer) is spread over more than one surface form -----


def test_both_bare_and_space_prefixed_forms_exist(tok) -> None:
    """The defect this marginalisation exists for.

    Scoring only " 2" ignores the mass the model puts on the bare "2". Both
    are real, distinct tokens in this vocabulary.
    """
    forms = answer_surface_forms(tok, 2)
    assert len(forms) == 2, f"expected two surface forms, got {forms}"
    assert [f for f, _ in forms] == [" 2", "2"]
    assert forms[0][1] != forms[1][1], "the two forms must differ in token ids"


@pytest.mark.parametrize("answer", [2, 7, 12, 64, -3, -17])
def test_surface_forms_share_a_token_length(tok, answer: int) -> None:
    """Why marginalising is cheap: the variants differ only in the first token.

    Equal length means no positional bookkeeping and one extra forward pass
    per variant. If this ever fails for some answer, `depth_curve` still works
    (each form is scored independently) but the cost assumption changes.
    """
    lens = {len(ids) for _, ids in answer_surface_forms(tok, answer)}
    assert len(lens) == 1, f"answer {answer} has ragged surface-form lengths: {lens}"


def test_surface_forms_are_deduplicated(tok) -> None:
    """If two prefixes give identical ids, the form is counted once.

    Double-counting would inflate P(answer) by up to log 2 for no reason.
    """
    for answer in (0, 5, 100):
        forms = answer_surface_forms(tok, answer)
        assert len({tuple(i) for _, i in forms}) == len(forms)


def test_marginalisation_strictly_increases_the_score() -> None:
    """log-sum-exp over forms must exceed the best single form."""
    per_form = np.array([[-3.0, -2.0], [-4.0, -5.0]])
    combined = _logsumexp(per_form, axis=0)
    assert (combined > per_form.max(axis=0)).all()
    assert combined[0] == pytest.approx(np.log(np.exp(-3.0) + np.exp(-4.0)))


def test_logsumexp_is_stable_for_very_negative_logprobs() -> None:
    """Real answer log-probs can be -700 or lower; naive exp would underflow."""
    a = np.array([[-800.0], [-802.0]])
    got = _logsumexp(a, axis=0)[0]
    assert np.isfinite(got)
    assert got == pytest.approx(-800.0 + np.log(1 + np.exp(-2.0)), abs=1e-9)


def _step_curve(r_star: int, n: int = 64, lo: float = -3.0, hi: float = 5.0) -> np.ndarray:
    """Margin that is flat-negative until r_star, then flat-positive."""
    m = np.full(n, lo)
    m[r_star - 1:] = hi
    return m


def test_recovers_a_clean_step_threshold() -> None:
    assert threshold_depth(_step_curve(20)) == 20.0


def test_threshold_is_scale_free() -> None:
    """Doubling the margin must not move the threshold -- only its shape matters."""
    a = threshold_depth(_step_curve(15, hi=4.0))
    b = threshold_depth(_step_curve(15, hi=40.0))
    assert a == b == 15.0


def test_no_threshold_when_the_model_never_prefers_the_gold_answer() -> None:
    """A margin that stays negative has no threshold; NaN, not a number."""
    assert np.isnan(threshold_depth(np.full(40, -2.0)))


def test_uniform_confidence_growth_does_not_create_a_threshold() -> None:
    """THE control this metric exists for.

    If the model merely grows more confident about everything, gold and
    distractor rise together, the margin stays flat at zero, and no threshold
    is reported. Only a *relative* preference counts.
    """
    logp_gold = np.linspace(-20, -1, 64)
    logp_dist = np.linspace(-20, -1, 64)
    assert np.isnan(threshold_depth(logp_gold - logp_dist))


def test_a_single_lucky_unroll_is_not_a_threshold() -> None:
    """One spike that immediately falls back must not be reported as solving."""
    m = np.full(64, -2.0)
    m[5] = 10.0          # transient spike
    m[40:] = 6.0         # the genuine onset
    assert threshold_depth(m) == 41.0


def test_gradual_ramp_reports_where_it_reaches_the_target_fraction() -> None:
    """A still-rising curve resolves EARLIER than its last sample, by design.

    The reference level is the tail median, not the maximum. For a curve that
    has not saturated by max_r the tail median sits below the final value, so
    the reported threshold is conservative (earlier). That is the intended
    trade: the alternative -- referencing the max -- would make the threshold
    hostage to a single noisy bf16 sample.
    """
    m = np.concatenate([np.full(10, -1.0), np.linspace(-1.0, 10.0, 54)])
    r = threshold_depth(m, frac=0.9)
    assert 45 <= r <= 60, f"ramp threshold {r} outside the expected band"
    assert r < 64, "a non-saturating ramp must not resolve only at the last unroll"


def test_short_or_degenerate_input_returns_nan() -> None:
    assert np.isnan(threshold_depth(np.array([1.0, 2.0])))
    assert np.isnan(threshold_depth(np.full(20, np.nan)))


def test_final_level_uses_the_tail_median_not_the_last_sample() -> None:
    """One noisy final unroll must not redefine 'solved'.

    bf16 logits are noisy (rigor_audit section 1); a single corrupted last
    value would otherwise move the threshold arbitrarily.
    """
    m = _step_curve(20, hi=5.0)
    m[-1] = 500.0                      # one wild sample
    assert threshold_depth(m) == 20.0


@pytest.mark.parametrize("r_star", [2, 7, 33, 48])
def test_recovers_thresholds_across_the_range(r_star: int) -> None:
    assert threshold_depth(_step_curve(r_star)) == float(r_star)


def test_onset_inside_the_reference_tail_is_reported_as_no_threshold() -> None:
    """An onset so late it falls inside the reference window yields NaN.

    With max_r = 64 the reference level is the median of the last 16 unrolls.
    A margin that only turns positive at unroll 64 leaves that median
    negative, so no threshold is claimed -- correctly: one positive sample at
    the very end is indistinguishable from noise, and the honest reading is
    "the sweep was too short", not "it solves at 64". Widen max_r instead.
    """
    assert np.isnan(threshold_depth(_step_curve(64)))
    assert np.isnan(threshold_depth(_step_curve(60)))
