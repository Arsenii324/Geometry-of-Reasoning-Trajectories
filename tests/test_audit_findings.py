"""Regression tests pinning the 2026-07-25 rigor audit's real-data findings.

Each test corresponds to a numbered section of docs/rigor_audit.md and fails if
that finding stops being true -- either because the data changed, or because a
"fix" quietly reverted the corrected behaviour.

The raw trajectories are untracked/local, so every real-data test skips (rather
than fails) when they are absent. A skipped run still exercises the synthetic
tests in test_regime.py.
"""

from __future__ import annotations

import glob
import itertools
import os

import numpy as np
import pandas as pd
import pytest
import torch

from traj_geom.metrics.convergence import consecutive_step_cosine, path_independence
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.regime import contraction_from_pair, step_cosine_converging
from traj_geom.metrics.winding import winding_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_TRAJ = os.path.join(ROOT, "results", "trajectories")
TOP_TRAJ = os.path.join(ROOT, "trajectories")


def _ns128() -> list[str]:
    return sorted(glob.glob(os.path.join(RES_TRAJ, "*ns128*.npy")))


def _need(paths: list[str], n: int = 1) -> None:
    if len(paths) < n:
        pytest.skip("raw trajectories not present (untracked/local)")


# --- section 1: compute dtype is bf16 -------------------------------------


def test_stored_trajectories_are_exactly_bf16_representable() -> None:
    """The evidence that Huginn was run in bfloat16, not float32.

    This underpins the whole noise-floor analysis; if it ever fails, the
    arithmetic-floor argument in sections 1, 2 and 6 must be re-derived.
    """
    paths = _ns128()
    _need(paths)
    for p in paths[:10]:
        t = torch.from_numpy(np.load(p))
        assert torch.equal(t, t.to(torch.bfloat16).to(t.dtype)), f"{p} is not bf16-exact"


def test_tail_jitter_is_isotropic_not_low_dimensional() -> None:
    """Section 1: the 'limit set' is noise, so its variance must not concentrate.

    Genuine low-dimensional dynamics would put most tail variance in the first
    two components. Rounding noise spreads it across all of them.
    """
    from sklearn.decomposition import PCA

    paths = _ns128()
    _need(paths)
    for p in paths[:6]:
        tail = np.load(p).astype(np.float64)[-60:]
        ev = PCA(n_components=20, svd_solver="full").fit(tail).explained_variance_ratio_
        assert ev[:2].sum() < 0.30, "tail variance concentrated -> would suggest real dynamics"


# --- section 2: the step cosine was measured on the noise regime -----------


def test_step_cosine_flips_sign_once_restricted_to_the_signal_regime() -> None:
    """Section 2, the headline correction.

    The deployed metric averages the second half of a path that settled at
    t~14, so it reports a property of rounding noise. Restricted to the
    converging regime the sign reverses: the path glides.
    """
    paths = _ns128()
    _need(paths, 10)
    old = float(np.mean([consecutive_step_cosine(np.load(p)) for p in paths[:30]]))
    new = float(np.mean([step_cosine_converging(np.load(p)) for p in paths[:30]]))
    assert old < -0.15, f"legacy second-half cosine should be clearly negative, got {old}"
    assert new > 0.0, f"signal-regime cosine should be positive, got {new}"


# --- section 3: steps_to_settle tracks rho, not difficulty -----------------


def test_steps_to_settle_is_predicted_by_the_contraction_rate_alone() -> None:
    """Section 3: t* = log(frac)/log(rho) reproduces the observed mean.

    The prediction uses no task information at all. Agreement therefore shows
    the metric cannot be carrying difficulty signal.
    """
    paths = sorted(glob.glob(os.path.join(RES_TRAJ, "*.npy")))
    _need(paths, 20)
    observed = float(np.mean([steps_to_settle(np.load(p)) for p in paths]))
    predicted = float(np.log(0.1) / np.log(0.8407))
    assert abs(observed - predicted) < 2.0, (
        f"observed {observed:.2f} vs rho-only prediction {predicted:.2f}"
    )


# --- section 4: the CSV and the banked trajectories are different data -----


def test_full_synthetic_csv_does_not_reproduce_from_banked_trajectories() -> None:
    """Section 4. Pinned so the mismatch cannot be quietly forgotten again.

    If this ever starts passing in the other direction (i.e. rows DO match),
    the provenance question has been resolved and the audit text must be
    updated -- that is a good failure, not a bad one.
    """
    csv = os.path.join(ROOT, "results", "full_synthetic_experiments.csv")
    if not os.path.exists(csv) or not _ns128():
        pytest.skip("inputs not present")
    d = pd.read_csv(csv)
    matches = 0
    checked = 0
    for _, r in d.iterrows():
        for ns in (64, 128):
            p = os.path.join(RES_TRAJ, f"{r.task}_n{r.n_ops}_ns{ns}_init{r.seed}.npy")
            if not os.path.exists(p):
                continue
            checked += 1
            if np.isclose(abs(winding_of(np.load(p))), abs(r.winding), rtol=1e-3):
                matches += 1
    assert checked > 0
    assert matches == 0, f"{matches}/{checked} now reproduce -- provenance changed"


# --- section 5: 'init{N}' is the task seed --------------------------------


def test_init_field_is_the_task_seed_not_the_init_seed() -> None:
    """Section 5, proven by a prompt collision.

    For n_ops=2, task seeds 1 and 2 generate the same bit string. If the field
    were an initialization seed the two files would differ; they are
    bit-identical, so h_0 is fixed and only the prompt varies.
    """
    a = os.path.join(RES_TRAJ, "count_ones_n2_ns128_init1.npy")
    b = os.path.join(RES_TRAJ, "count_ones_n2_ns128_init2.npy")
    if not (os.path.exists(a) and os.path.exists(b)):
        pytest.skip("collision pair not present")

    from traj_geom.shapes.synthetic import make_count_ones_task

    assert make_count_ones_task(2, seed=1)["prompt"] == make_count_ones_task(2, seed=2)["prompt"]
    assert np.array_equal(np.load(a), np.load(b)), (
        "same prompt must give the same trajectory -- if not, h_0 varies after all"
    )


# --- sections 6 & 7: contraction, floor-aware ------------------------------


def test_path_independence_understates_contraction_versus_floor_aware_fit() -> None:
    """Section 6: the whole-range fit is biased toward 'less contracting'."""
    man = os.path.join(TOP_TRAJ, "manifest.csv")
    if not os.path.exists(man):
        pytest.skip("multi-init manifest not present")
    m = pd.read_csv(man)
    old, new = [], []
    for (_, _, ns), g in m.groupby(["kind", "n_ops", "num_steps"]):
        if ns < 64:                      # ns=16 groups are not converged (section 12)
            continue
        paths = [np.load(os.path.join(TOP_TRAJ, f)).astype(np.float64) for f in g.file]
        for i, j in itertools.combinations(range(len(paths)), 2):
            old.append(float(np.exp(path_independence(paths[i], paths[j]))))
            new.append(contraction_from_pair(paths[i], paths[j])[0])
    if not old:
        pytest.skip("no converged multi-init groups")
    assert np.mean(old) > np.mean(new) + 0.02, "floor dilution should be visible"


def test_h3_premise_holds_map_contracts_on_real_huginn() -> None:
    """Section 7: rho < 1 measured from two orbits of the SAME prompt.

    This is the quantity H3 actually needs, and it is the only measurement in
    the project that bears on it. Distinct from a trajectory's approach to its
    own endpoint, which is near-tautological.
    """
    man = os.path.join(TOP_TRAJ, "manifest.csv")
    if not os.path.exists(man):
        pytest.skip("multi-init manifest not present")
    m = pd.read_csv(man)
    rhos = []
    for (_, _, ns), g in m.groupby(["kind", "n_ops", "num_steps"]):
        if ns < 64:
            continue
        paths = [np.load(os.path.join(TOP_TRAJ, f)).astype(np.float64) for f in g.file]
        for i, j in itertools.combinations(range(len(paths)), 2):
            rhos.append(contraction_from_pair(paths[i], paths[j])[0])
    if not rhos:
        pytest.skip("no converged multi-init groups")
    rho = float(np.nanmean(rhos))
    assert 0.75 < rho < 1.0, f"rho={rho} outside the audited range"


# --- section 8: the state lives on a sphere -------------------------------


def test_state_norm_is_constant_rmsnorm_sphere() -> None:
    """Section 8: every recorded state is an RMSNorm output."""
    paths = _ns128()
    _need(paths, 5)
    for p in paths[:8]:
        n = np.linalg.norm(np.load(p).astype(np.float64), axis=1)
        assert n.std() / n.mean() < 1e-3, "state norm should be effectively constant"


def test_matched_random_walk_leaves_the_sphere() -> None:
    """Section 8: why every existing null test compares on- against off-manifold."""
    from traj_geom.metrics.winding import matched_random_walk

    paths = _ns128()
    _need(paths)
    a = np.load(paths[0])
    s = matched_random_walk(a, np.random.default_rng(0)).astype(np.float64)
    real = np.linalg.norm(a.astype(np.float64), axis=1)
    sur = np.linalg.norm(s, axis=1)
    assert real.std() / real.mean() < 1e-3
    assert sur.std() / sur.mean() > 1e-2, "surrogate should drift off the sphere"


# --- section 9: PCA determinism -------------------------------------------


def test_winding_is_now_deterministic() -> None:
    """Section 9: svd_solver='full' makes repeated calls bit-identical."""
    paths = _ns128()
    _need(paths)
    a = np.load(paths[0])
    vals = [winding_of(a) for _ in range(8)]
    assert len(set(vals)) == 1, f"winding_of still nondeterministic: {set(vals)}"
