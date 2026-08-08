"""Stratified permutation test: compare two classes WITHIN fixed strata.

OWNER: Data+Analysis
STATUS: implemented 2026-08-09.

WHY IT EXISTS, precisely. D72 voided B6's first run: correctness there was very
nearly a deterministic function of the ANSWER VALUE, so permuting the correctness
label destroyed the correctness-geometry link and the prompt-geometry link at the
same time, and no amount of permutation could tell them apart. The fix is not a
better statistic on the same design -- it is to permute WITHIN the answer value, so
the prompt content is held fixed by construction and only the label moves.

THE STATISTIC IS THE STRATUM-WEIGHTED MEAN DIFFERENCE. Pooling across strata would
reintroduce exactly the confound the stratification removes, because strata differ
in both their class balance and their geometry. Strata that cannot support a
contrast (all one class) contribute nothing and are reported, not silently dropped.

THE NULL'S FLOOR IS CHECKED, NOT ASSUMED. `geometry-correctness` was drafted with
n_perm=200 against a Bonferroni alpha of 0.00417, making the smallest attainable
p 0.00498 -- rejection was arithmetically impossible and the run would have returned
a clean null that read like a result. `attainable` is applied here before anything
is reported.
"""

from __future__ import annotations

import numpy as np


def stratified_diff(values: np.ndarray, labels: np.ndarray,
                    strata: np.ndarray) -> tuple[float, int]:
    """Stratum-weighted mean difference between the two classes.

    Args:
        values: metric per item.
        labels: boolean class per item (True = the "positive" class).
        strata: stratum key per item; comparisons happen only within a key.

    Returns:
        ``(difference, n_usable_items)``. A stratum containing only one class
        cannot support a contrast and is excluded from BOTH the statistic and the
        item count, so the count reports what the test actually rests on.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    strata = np.asarray(strata)
    num, den, n_used = 0.0, 0, 0
    for s in np.unique(strata):
        m = strata == s
        a, b = values[m & labels], values[m & ~labels]
        if len(a) == 0 or len(b) == 0:
            continue
        w = int(m.sum())
        num += w * (float(a.mean()) - float(b.mean()))
        den += w
        n_used += w
    return (num / den if den else float("nan")), n_used


def stratified_permutation_p(values: np.ndarray, labels: np.ndarray,
                             strata: np.ndarray, n_perm: int = 20000,
                             seed: int = 0) -> dict:
    """Two-sided p for `stratified_diff`, permuting labels WITHIN each stratum.

    Two-sided is deliberate. B6 pre-registers no direction: "correct answers settle
    sooner" and "correct answers keep moving longer" are both tellable stories, and
    choosing one after seeing the data is how D22 obtained a confirmation it later
    withdrew.

    Returns ``obs``, ``p``, ``n_perm``, ``p_floor``, ``n_used`` and ``usable`` --
    the last being False when no stratum contains both classes, which is the D72
    situation and must refuse rather than return a number.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    strata = np.asarray(strata)
    obs, n_used = stratified_diff(values, labels, strata)
    if not np.isfinite(obs):
        return {"usable": False, "why": "no stratum contains both classes",
                "obs": float("nan"), "p": float("nan"), "n_used": 0,
                "n_perm": n_perm, "p_floor": 1.0 / (n_perm + 1)}
    rng = np.random.default_rng(seed)
    idx_by_stratum = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    hits = 0
    for _ in range(n_perm):
        perm = labels.copy()
        for idx in idx_by_stratum:
            perm[idx] = rng.permutation(labels[idx])
        d, _ = stratified_diff(values, perm, strata)
        if np.isfinite(d) and abs(d) >= abs(obs) - 1e-12:
            hits += 1
    return {"usable": True, "obs": float(obs), "p": (hits + 1) / (n_perm + 1),
            "n_used": int(n_used), "n_perm": int(n_perm),
            "p_floor": 1.0 / (n_perm + 1)}


def attainable(alpha: float, n_perm: int) -> tuple[bool, float]:
    """Can a permutation test with ``n_perm`` draws ever reach ``alpha``?

    Mirrors the kernel-library guard of the same name. The smallest p a permutation
    test can report is 1/(n_perm+1); if that floor sits above the threshold,
    rejection is arithmetically impossible and the run returns "not significant" for
    every cell no matter what the data say.
    """
    floor = 1.0 / (n_perm + 1)
    return floor < alpha, floor
