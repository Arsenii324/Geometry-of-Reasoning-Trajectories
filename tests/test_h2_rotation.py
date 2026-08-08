"""Validation of the B4.15 analysis path, before it is pointed at real data.

The three kernels that produced wrong conclusions in this project (D62,
`geometry-discourse`, D70) all failed in their ANALYSIS, not their measurement,
and each would have been caught by exercising that path on synthetic input with a
known answer. So: a planted effect must be found, an absent effect must not be
invented, the null must be calibrated, and the instrument gate must be able to
fire.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scripts.run_h2_rotation import (
    analyse,
    paired_diff,
    sign_flip_p,
    total_turns,
)

N_OPS = (4, 8, 12, 16, 24, 32)
SEEDS = (0, 1)


def _frame(track_slope=0.0, local_slope=0.0, noise=0.0, seed=0, arm="trained",
           rho=0.8, complex_frac=1.0) -> pd.DataFrame:
    """Synthetic run with a KNOWN arg-vs-n_ops slope in each arm."""
    rng = np.random.default_rng(seed)
    rows = []
    n_total = len(N_OPS) * len(SEEDS) * 2
    n_real = int(round((1 - complex_frac) * n_total))
    i = 0
    for n_ops in N_OPS:
        for s in SEEDS:
            for kind, slope in (("track", track_slope), ("local", local_slope)):
                arg = 1.5 + slope * n_ops + noise * rng.normal()
                rows.append({"arm": arm, "kind": kind, "n_ops": n_ops, "seed": s,
                             "rho": rho,
                             "arg": np.nan if i < n_real else arg,
                             "arg_osc": None if i < n_real else arg})
                i += 1
    df = pd.DataFrame(rows)
    df["turns"] = total_turns(df["arg"].fillna(0.0), df["rho"])
    return df


def test_total_turns_is_a_known_answer() -> None:
    """arg=1 rad/unroll, rho=0.5 -> t* = ln(0.01)/ln(0.5) unrolls, turns = arg*t*/2pi."""
    t_star = math.log(0.01) / math.log(0.5)
    got = total_turns(np.array([1.0]), np.array([0.5]))[0]
    assert got == np.float64(1.0 * t_star / (2 * math.pi))
    # Slower contraction means more unrolls before settling, so more total turns.
    assert total_turns(np.array([1.0]), np.array([0.9]))[0] > got


def test_paired_diff_pairs_track_against_local_within_a_cell() -> None:
    df = _frame(track_slope=0.01, local_slope=0.0)
    pairs = paired_diff(df)
    assert len(pairs) == len(N_OPS) * len(SEEDS)
    # track - local must recover the planted slope difference in expectation.
    assert pairs["diff"].corr(pairs["n_ops"]) > 0.99


def test_batch_rho_matches_the_canonical_statistic() -> None:
    """The vectorised null must equal `spearman_by_level` row for row.

    The exact null enumerates 4096 arrangements, so it cannot call a pandas
    groupby per arrangement. That optimisation is only safe if it is pinned to the
    statistic the rest of the project reports.
    """
    from scripts.run_h2_rotation import _rho_by_level_batch

    from traj_geom.analysis.correlate import spearman_by_level

    rng = np.random.default_rng(0)
    levels = np.repeat(np.array(N_OPS, float), 2)
    vals = rng.normal(size=(25, len(levels)))
    fast = _rho_by_level_batch(levels, vals)
    for i in range(len(vals)):
        slow = spearman_by_level(
            pd.DataFrame({"n_ops": levels, "v": vals[i]}), "n_ops", "v")[0]
        assert fast[i] == np.float64(slow) or abs(fast[i] - slow) < 1e-12, (
            f"row {i}: fast {fast[i]} vs canonical {slow}")


def test_sign_flip_null_finds_a_planted_effect() -> None:
    df = _frame(track_slope=0.02, local_slope=0.0, noise=0.002, seed=1)
    obs, p, _ = sign_flip_p(paired_diff(df))
    assert obs > 0.8, obs
    # A monotone planted difference cannot be beaten by many sign patterns, but a
    # few (the all-flip, and level-wise negations that stay monotone) tie it, so
    # the floor itself is not reachable. What matters is that p is far below any
    # alpha this project uses.
    assert p < 0.01, f"planted effect should be highly significant, got p={p}"


def test_sign_flip_null_does_not_invent_an_effect() -> None:
    """The failure mode that matters: reporting a difference when none was planted."""
    df = _frame(track_slope=0.0, local_slope=0.0, noise=0.05, seed=7)
    _, p, _ = sign_flip_p(paired_diff(df))
    assert p > 0.05, f"no effect planted but p={p}"


def test_sign_flip_null_is_calibrated() -> None:
    """Under the null, p<0.05 must fire on about 5% of trials, not far more."""
    hits = 0
    trials = 120
    for s in range(trials):
        df = _frame(track_slope=0.0, local_slope=0.0, noise=0.05, seed=1000 + s)
        _, p, _ = sign_flip_p(paired_diff(df))
        hits += p < 0.05
    rate = hits / trials
    assert rate < 0.15, f"null rejects at {rate:.2%}, far above nominal 5%"


def test_the_null_can_actually_reject_at_this_grid_size() -> None:
    """`attainable`'s question, asked of this design before it is trusted.

    12 pairs give 2^12 = 4096 exact arrangements, so the smallest reportable p is
    2.4e-4. `geometry-correctness` was drafted with a floor of 4.98e-3 against an
    alpha of 4.17e-3, which made rejection impossible and guaranteed a null that
    read like a result (D72).
    """
    df = _frame(track_slope=0.02)
    _, _, n_arr = sign_flip_p(paired_diff(df))
    assert n_arr == 2 ** (len(N_OPS) * len(SEEDS))
    assert 1.0 / n_arr < 0.05


def test_p1_gate_fires_when_the_spectrum_is_mostly_real() -> None:
    """arg is undefined for a real eigenvalue; a mostly-real run measures nothing."""
    res = analyse(_frame(track_slope=0.02, complex_frac=0.4))["trained"]
    assert not res["p1_gate_ok"]
    assert res["complex_frac"] < 0.8


def test_p1_gate_passes_a_healthy_run() -> None:
    """Non-suppression: the gate must not fire on the case it is meant to allow."""
    res = analyse(_frame(track_slope=0.02))["trained"]
    assert res["p1_gate_ok"]
    assert res["paired"]["rho_diff_vs_n_ops"] > 0.8


def test_analyse_reports_both_arms_independently() -> None:
    a = _frame(track_slope=0.02, arm="trained", seed=2)
    b = _frame(track_slope=0.0, arm="untrained", seed=3, noise=0.05)
    res = analyse(pd.concat([a, b], ignore_index=True))
    assert set(res) == {"trained", "untrained"}
    assert res["trained"]["paired"]["p_exact"] < res["untrained"]["paired"]["p_exact"]
