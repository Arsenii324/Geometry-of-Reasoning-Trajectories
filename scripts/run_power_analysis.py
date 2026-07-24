"""Power analysis + pre-registration for H1/H2/H3 (project_plan.md
§0.1/§7 -- "the honesty gate" that must run before any further GPU spend
on H2, since an underpowered null must never be reported as a refutation).

OWNER: Data+Analysis
STATUS: implemented 2026-07-24. CORRECTED same day after reading Blayney et
    al.'s actual PDF Appendix C (the first version relied on a WebFetch
    HTML summary that conflated two different quantities -- see the
    BLAYNEY_RATE_OPTIMISTIC comment below and claims_ledger.md B9).
TASK: two independent power questions this project had never actually
    computed:
    (1) Genuine-winding-LOOP rate power. At Blayney et al.'s (arXiv:2604.11791,
        Appendix C Table 3, see claims_ledger.md B9) measured PER-TOKEN
        non-fixed-point rates -- 0.02% with no system prompt, 0.14% under
        their "Long Persona" system-prompt condition -- how many
        (token, trajectory) draws are needed to expect >=5 genuine loops,
        and does this project's current/planned extraction scale clear
        that bar? Cross-checked against this project's own real data:
        outside the artificially-starved forceloop.csv sweep
        (num_steps=16), the answer-token-only pool (n=1294 winding/shape-
        classified real extractions, counted live below) shows ZERO
        loop/drift trajectories at full compute budget -- consistent with,
        not contradicting, the Blayney-rate power calculation (E[loops] at
        0.02% on n=1294 is ~0.26, i.e. "expect to see none" is the
        correctly-powered prediction, not evidence H2 is false).
    (2) Depth-correlation DETECTION power. At the small N (4-10) "levels"
        this project's synthetic tasks actually produce, what statistical
        power does a per-level Spearman test have to detect a range of true
        effect sizes? Via Monte Carlo simulation (bivariate-normal data,
        Spearman computed exactly as spearman_by_level does) rather than a
        closed-form approximation, so it matches what the codebase actually
        runs.

I/O: 0-GPU. Part (1) reads results/{counting,maxtask,switch,pararule,
    dissociation,dissociation_15seed,dissoc_multiinit,three_scale,
    forceloop}.csv (must already exist). Part (2) needs no data, pure
    simulation. -> prints two tables; saves results/power_loop_rate.csv and
    results/power_curve.csv.

Run: uv run python -m scripts.run_power_analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts._common import RESULTS_DIR

# Blayney et al. arXiv:2604.11791 Appendix C, Table 3 -- fraction of TOKENS
# (not examples, not trajectories) that are individually non-fixed-point.
# See claims_ledger.md B9. CORRECTED 2026-07-24: the original version of
# this script used 2.81% here, sourced from an automated HTML summary that
# turned out to conflate Table 3 (per-token) with Table 4 (per-EXAMPLE,
# "at least one question token in this example shows the behavior
# anywhere" -- a fundamentally different, much larger quantity precisely
# because it aggregates over every token in the example). Confirmed by
# reading the actual PDF: Table 3's Huginn-0125, Long-Persona-system-prompt
# PER-TOKEN rate is 0.14%, not 2.81%. Using 2.81% as a per-draw Poisson
# rate overstated the "optimistic ceiling" N-needed-for-bar by ~20x.
BLAYNEY_RATE_BASELINE = 0.0002  # 0.02%, no system prompt, Table 3
BLAYNEY_RATE_OPTIMISTIC = 0.0014  # 0.14%, "Long Persona" system prompt, Table 3
LOOP_BAR = 5  # project_plan.md §12 point 5's proposed "genuine loops" power bar

# Files with a winding/shape classification -- the current answer-token-only
# extraction pool this project has actually produced.
_WINDING_CSVS = (
    "counting.csv",
    "maxtask.csv",
    "switch.csv",
    "pararule.csv",
    "dissociation.csv",
    "dissociation_15seed.csv",
    "dissoc_multiinit.csv",
    "three_scale.csv",
    "forceloop.csv",
)


def _current_pool_size() -> int:
    """Count real Huginn extractions across every winding/shape-classified CSV."""
    total = 0
    for name in _WINDING_CSVS:
        df = pd.read_csv(RESULTS_DIR / name)
        if "winding" in df.columns or "shape" in df.columns:
            total += len(df)
    return total


def _loops_observed_outside_forceloop() -> int:
    """Real loop/drift count at full compute budget, excluding the
    artificially-starved forceloop.csv (num_steps=16) sweep.
    """
    total = 0
    for name in _WINDING_CSVS:
        if name == "forceloop.csv":
            continue
        df = pd.read_csv(RESULTS_DIR / name)
        if "shape" in df.columns:
            total += int((df["shape"] != "settle").sum())
    return total


def loop_rate_power_table() -> pd.DataFrame:
    """E[loops] and N-needed-for->=5-loops under both Blayney rates, at the
    project's current pool size and a few keystone-extraction scenarios.
    """
    n_current = _current_pool_size()
    # Keystone all-token scenarios: M prompts x S tokens/prompt, matching the
    # scale of this project's existing sweeps (100s of prompts) and the
    # token-length range its synthetic tasks actually produce (~15-250 tokens,
    # see architecture_state.md).
    scenarios = [
        ("current answer-token pool (real, this project)", n_current),
        ("keystone all-token, M=500 prompts x S=30 tok", 500 * 30),
        ("keystone all-token, M=500 prompts x S=100 tok", 500 * 100),
        ("keystone all-token, M=1000 prompts x S=50 tok", 1000 * 50),
        ("keystone all-token, M=1000 prompts x S=100 tok", 1000 * 100),
    ]
    rows = []
    for label, n in scenarios:
        for rate_name, rate in (
            ("Blayney baseline (0.02%)", BLAYNEY_RATE_BASELINE),
            ("Blayney Long Persona (0.14%)", BLAYNEY_RATE_OPTIMISTIC),
        ):
            expected = n * rate
            rows.append(
                {
                    "scenario": label,
                    "n_draws": n,
                    "rate": rate_name,
                    "expected_loops": expected,
                    "clears_bar_of_5": expected >= LOOP_BAR,
                }
            )
    return pd.DataFrame(rows)


def n_needed_for_bar() -> pd.DataFrame:
    """N of (token, trajectory) draws needed to expect >=LOOP_BAR loops,
    under each Blayney rate.
    """
    return pd.DataFrame(
        [
            {"rate": "Blayney baseline (0.02%)", "rate_value": BLAYNEY_RATE_BASELINE,
             "n_needed_for_5_loops": LOOP_BAR / BLAYNEY_RATE_BASELINE},
            {"rate": "Blayney Long Persona (0.14%)", "rate_value": BLAYNEY_RATE_OPTIMISTIC,
             "n_needed_for_5_loops": LOOP_BAR / BLAYNEY_RATE_OPTIMISTIC},
        ]
    )


def spearman_power_curve(
    n_levels: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50),
    true_rhos: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    n_trials: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Monte Carlo power of a Spearman test to detect a true correlation
    rho_true at sample size N (one point per "level", matching this
    project's per-level statistic -- N is small, 4-10, for every synthetic
    task in this project).

    Data are drawn from a bivariate normal with Pearson correlation
    rho_true (Spearman and Pearson coincide closely for normal marginals at
    these N/rho ranges); the test run on each simulated sample is exactly
    ``scipy.stats.spearmanr``, the same call every run_*.py script makes,
    so the reported power is not a closed-form approximation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_levels:
        for rho_true in true_rhos:
            cov = np.array([[1.0, rho_true], [rho_true, 1.0]])
            hits = 0
            for _ in range(n_trials):
                sample = rng.multivariate_normal([0, 0], cov, size=n)
                _, p = spearmanr(sample[:, 0], sample[:, 1])
                if p < alpha:
                    hits += 1
            power = hits / n_trials
            rows.append({"n_levels": n, "rho_true": rho_true, "power": power})
    return pd.DataFrame(rows)


def main() -> None:
    """Run both power analyses and report the pre-registration numbers."""
    n_current = _current_pool_size()
    loops_real = _loops_observed_outside_forceloop()
    print(f"Current real winding/shape-classified pool: {n_current} trajectories.")
    print(
        f"Loop/drift observed outside the starved forceloop.csv sweep: "
        f"{loops_real} (out of {n_current - 96})."
    )

    print("\n--- N needed for >=5 expected genuine loops ---")
    needed = n_needed_for_bar()
    print(needed.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    print("\n--- Loop-rate scenarios ---")
    scenarios = loop_rate_power_table()
    print(scenarios.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n--- Monte Carlo Spearman power curve (5000 trials/cell) ---")
    curve = spearman_power_curve()
    pivot = curve.pivot(index="n_levels", columns="rho_true", values="power")
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    scenarios.to_csv(RESULTS_DIR / "power_loop_rate.csv", index=False)
    curve.to_csv(RESULTS_DIR / "power_curve.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'power_loop_rate.csv'} and {RESULTS_DIR / 'power_curve.csv'}")


if __name__ == "__main__":
    main()
