"""Regression-pins the load-bearing correctness numbers (docs/results_report.md
§4, the honesty headline) against the real cached results/counting_accuracy.csv:
the model fails the counting task at the depths where the winding "signal"
appears, and geometry does not separate correct from incorrect trajectories.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scipy.stats import pointbiserialr


def test_counting_accuracy_collapses_with_depth() -> None:
    """Accuracy is ~0 at exactly the n_ops levels where winding/steps_settle
    are reported to 'rise with reasoning depth' (n_ops>=8).
    """
    df = pd.read_csv("results/counting_accuracy.csv")
    acc = df.groupby("n_ops")["correct"].mean()
    assert acc[2] == pytest.approx(0.375, abs=0.001)
    # Every level from n_ops=8 up is at or near zero (24 is the lone 12.5%).
    assert acc[8] == 0.0
    assert acc[16] == 0.0
    assert acc[32] == 0.0
    assert acc[48] == 0.0
    assert acc[24] == pytest.approx(0.125, abs=0.001)


def test_geometry_does_not_separate_correct_from_incorrect() -> None:
    """winding does not distinguish correct vs incorrect (n.s.); steps_settle
    is HIGHER for wrong answers (negative point-biserial, significant) -- the
    opposite of a 'more effective compute -> better reasoning' reading.
    """
    df = pd.read_csv("results/counting_accuracy.csv")
    r_w, p_w = pointbiserialr(df["correct"], df["winding"])
    r_s, p_s = pointbiserialr(df["correct"], df["steps_settle"])

    assert r_w == pytest.approx(-0.087, abs=0.005)
    assert p_w > 0.05  # winding: no separation

    assert r_s == pytest.approx(-0.295, abs=0.005)
    assert p_s < 0.05  # steps_settle: significant, and the sign is negative
    assert r_s < 0  # higher steps_settle -> LESS likely correct
