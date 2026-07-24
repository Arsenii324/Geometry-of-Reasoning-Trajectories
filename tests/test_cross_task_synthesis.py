"""Regression-pins the cross-task syntheses that back claims_ledger.md D16/D18/D19
against the real cached CSVs -- these are the load-bearing anti-H2 and
init-robustness findings in docs/results_report.md, and they were assembled
by hand from multiple files, so they need a test that recomputes them.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scipy.stats import spearmanr


def _per_level_rho(path: str, xcol: str, ycol: str, kind: str | None = None) -> float:
    df = pd.read_csv(path)
    if kind is not None and "kind" in df.columns:
        df = df[df["kind"] == kind]
    gm = df.groupby(xcol)[ycol].mean()
    return float(spearmanr(gm.index.to_numpy(), gm.to_numpy())[0])


# (label, path, difficulty col, kind filter) -> expected (winding_rho, steps_rho)
_TASKS = [
    ("counting", "results/counting.csv", "n_ops", None, +0.943, +0.928),
    ("switch", "results/switch.csv", "n_ops", None, -0.086, +0.783),
    ("maxtask", "results/maxtask.csv", "n_ops", None, +0.943, -0.771),
    ("pararule", "results/pararule.csv", "depth", None, +0.200, +0.800),
    ("dissoc-track", "results/dissociation.csv", "n_ops", "track", +0.771, +0.812),
    ("three_scale", "results/three_scale.csv", "active_len", None, -0.400, +1.000),
    ("modk-N7", "results/three_scale_modk.csv", "active_len", None, +0.107, -0.739),
    ("modk-N15", "results/three_scale_modk_extended.csv", "active_len", None, -0.125, -0.664),
]


def test_d16_winding_has_no_consistent_sign_across_tasks() -> None:
    """The strongest single anti-H2 datapoint: winding~difficulty has no
    consistent sign; the only significant hits are the length-confounded ones.
    """
    winding = {}
    for label, path, xcol, kind, exp_w, _ in _TASKS:
        rho = _per_level_rho(path, xcol, "winding", kind)
        assert rho == pytest.approx(exp_w, abs=0.001), f"{label} winding rho drifted"
        winding[label] = rho

    # Signs are genuinely mixed (not all one direction).
    signs = {1 if v > 0 else -1 for v in winding.values()}
    assert signs == {1, -1}
    # The two large hits are exactly counting and maxtask (the D10-confounded pair).
    big = {k for k, v in winding.items() if v > 0.9}
    assert big == {"counting", "maxtask"}
    # Every length-clean task (three_scale + both modk) is <= 0.11 (null-to-negative).
    for clean in ("three_scale", "modk-N7", "modk-N15"):
        assert winding[clean] <= 0.11


def test_d18_steps_settle_sign_tracks_accumulate_vs_saturate() -> None:
    """steps_settle~difficulty is positive for accumulating tasks, negative for
    saturating/wrapping ones (maxtask, both modk).
    """
    accumulate = ["counting", "switch", "pararule", "dissoc-track", "three_scale"]
    saturate = ["maxtask", "modk-N7", "modk-N15"]
    steps = {}
    for label, path, xcol, kind, _, exp_s in _TASKS:
        rho = _per_level_rho(path, xcol, "steps_settle", kind)
        assert rho == pytest.approx(exp_s, abs=0.001), f"{label} steps rho drifted"
        steps[label] = rho
    assert all(steps[t] > 0 for t in accumulate)
    assert all(steps[t] < 0 for t in saturate)


def test_d19_init_robustness_ratios() -> None:
    """within-config init-noise std vs between-config signal std: winding most
    init-stable (~0.50), contraction init-noise-dominated (~1.72).
    """
    df = pd.read_csv("results/dissoc_multiinit.csv")
    keys = ["n_ops", "kind", "task_seed"]
    ratios = {}
    for metric in ("winding", "steps_settle", "contraction"):
        grp = df.groupby(keys)[metric]
        within = grp.std().mean()
        between = grp.mean().std()
        ratios[metric] = within / between
    assert ratios["winding"] == pytest.approx(0.50, abs=0.02)
    assert ratios["steps_settle"] == pytest.approx(0.84, abs=0.02)
    assert ratios["contraction"] == pytest.approx(1.72, abs=0.02)
    # The ordering is the load-bearing claim: winding most stable, contraction least.
    assert ratios["winding"] < ratios["steps_settle"] < ratios["contraction"]
    assert ratios["contraction"] > 1.0  # noise exceeds signal
