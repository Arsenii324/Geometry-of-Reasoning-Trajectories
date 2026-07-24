"""Benjamini-Hochberg FDR correction across the project's existing correlation
tests (project_plan.md Phase 0, item 0.2/§15 item 1).

OWNER: Data+Analysis
STATUS: implemented 2026-07-24.
TASK: this project runs ~46-50 correlation/significance tests across 9
    experiment scripts, none previously corrected for multiple comparisons
    (see correlate.py's own note). This script collects every one of them
    from the already-cached results/*.csv (0-GPU, no re-extraction) and
    applies benjamini_hochberg. Two kinds of test are collected:
    (1) per-row spearman()/partial_spearman() calls, which already carry a
        real p-value, exactly as each script's own main() computes them;
    (2) per-level spearman_by_level() calls, which report a critical-value
        table lookup (sig/n.s.) instead of a p-value -- this script derives
        the same per-level test's actual p-value via spearmanr on the
        group means (the same computation spearman_by_level does
        internally, just not discarding p), so it can be FDR-corrected too.
    Guard-raised partial_spearman calls (the degenerate n_ops==seq_len
    tasks: counting/maxtask/switch, see claims_ledger.md D10) produce no
    p-value and are correctly excluded, not silently treated as p=1.

FAMILY DEFINITION -- an interpretive choice, not a fact. correlate.py's own
    module docstring and project_plan.md §12 curator-decision point 5 both
    flag "what counts as one family of tests" as belonging to whoever signs
    off on the paper's claims, not something to settle unilaterally. This
    script reports BOTH: a single project-wide family (every test below,
    pooled) and per-experiment families (one family per results/*.csv), so
    both readings are on the table. It leans on the project-wide family as
    the more defensible default absent other guidance, since these tests
    were not pre-registered per-experiment and a per-experiment split lets
    each experiment "reset its own budget" -- understating how many total
    chances the project took at finding a correlation. It does NOT pool
    with the not-yet-run QK probe (~7,000 tests, project_plan.md §9): that
    family doesn't exist yet, and pooling with it now would be
    scientifically meaningless, not just conservative.

I/O: reads results/{counting,maxtask,switch,pararule,dissociation,
    dissociation_15seed,dissoc_multiinit,three_scale,forceloop}.csv (must
    already exist -- this script does no extraction) -> prints a full
    table and results/fdr_correction.csv (family, test, rho_or_stat, p,
    q_project_wide, sig_raw, sig_project_wide, q_per_experiment,
    sig_per_experiment).

Run: uv run python -m scripts.run_fdr_correction
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact, spearmanr

from scripts._common import RESULTS_DIR
from traj_geom.analysis.correlate import (
    benjamini_hochberg,
    multivariate_rank_control,
    partial_spearman,
    spearman,
)


def _level_p(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[float, int, float]:
    """Per-level Spearman rho AND its p-value.

    Mirrors ``spearman_by_level``'s group-then-correlate computation exactly
    (same ``groupby(xcol)[ycol].mean()`` then ``spearmanr``) -- that function
    discards the p-value in favour of a critical-value-table lookup, which
    this FDR pass needs a real p for.
    """
    gm = df.groupby(xcol)[ycol].mean()
    rho, p = spearmanr(gm.index.to_numpy(), gm.to_numpy())
    return float(rho), int(len(gm)), float(p)


def _collect() -> list[dict]:
    """Recompute every correlation test this project currently reports,
    tagged by (family, test), from cached CSVs only -- 0-GPU.
    """
    tests: list[dict] = []

    def add(family: str, test: str, stat: float, p: float) -> None:
        tests.append({"family": family, "test": test, "stat": stat, "p": p})

    # -- counting.csv --
    cdf = pd.read_csv(RESULTS_DIR / "counting.csv")
    for metric in ("winding", "steps_settle"):
        rho, p = spearman(cdf["n_ops"], cdf[metric])
        add("counting", f"{metric}~n_ops [per-row]", rho, p)
        rho_l, _, p_l = _level_p(cdf, "n_ops", metric)
        add("counting", f"{metric}~n_ops [per-level]", rho_l, p_l)
    # partial_spearman(winding, n_ops, seq_len) raises (D10 guard) -- excluded.

    # -- maxtask.csv --
    mx = pd.read_csv(RESULTS_DIR / "maxtask.csv")
    for metric in ("steps_settle", "winding"):
        rho, p = spearman(mx["n_ops"], mx[metric])
        add("maxtask", f"{metric}~n_ops [per-row]", rho, p)
        rho_l, _, p_l = _level_p(mx, "n_ops", metric)
        add("maxtask", f"{metric}~n_ops [per-level]", rho_l, p_l)

    # -- switch.csv --
    sw = pd.read_csv(RESULTS_DIR / "switch.csv")
    for metric in ("steps_settle", "winding"):
        rho, p = spearman(sw["n_ops"], sw[metric])
        add("switch", f"{metric}~n_ops [per-row]", rho, p)
        rho_l, _, p_l = _level_p(sw, "n_ops", metric)
        add("switch", f"{metric}~n_ops [per-level]", rho_l, p_l)

    # -- pararule.csv (the one synthetic-free task with a genuinely
    #    non-degenerate confounder -- its partial_spearman does not raise) --
    pr = pd.read_csv(RESULTS_DIR / "pararule.csv")
    rho, p = spearman(pr["winding"], pr["depth"])
    add("pararule", "winding~depth [per-row]", rho, p)
    rho_l, _, p_l = _level_p(pr, "depth", "winding")
    add("pararule", "winding~depth [per-level]", rho_l, p_l)
    rho_l, _, p_l = _level_p(pr, "depth", "steps_settle")
    add("pararule", "steps_settle~depth [per-level]", rho_l, p_l)
    rho_p, p_p = partial_spearman(pr["winding"], pr["depth"], pr["seq_len"])
    add("pararule", "winding~depth [per-row, partial|seq_len]", rho_p, p_p)

    # -- dissociation.csv (5-seed) and dissociation_15seed.csv --
    dissoc_sources = (
        ("dissociation.csv", "dissociation_5seed"),
        ("dissociation_15seed.csv", "dissociation_15seed"),
    )
    for name, family in dissoc_sources:
        ds = pd.read_csv(RESULTS_DIR / name)
        for kind in ("track", "local"):
            s = ds[ds.kind == kind]
            for metric in ("winding", "steps_settle"):
                rho, p = spearman(s["n_ops"], s[metric])
                add(family, f"{kind} {metric}~n_ops [per-row]", rho, p)
                rho_l, _, p_l = _level_p(s, "n_ops", metric)
                add(family, f"{kind} {metric}~n_ops [per-level]", rho_l, p_l)

    # -- dissoc_multiinit.csv (per-level only; script computes no per-row test) --
    dm = pd.read_csv(RESULTS_DIR / "dissoc_multiinit.csv")
    for kind in ("track", "local"):
        s = dm[dm.kind == kind]
        rho_l, _, p_l = _level_p(s, "n_ops", "steps_settle")
        add("dissoc_multiinit", f"{kind} steps_settle~n_ops [per-level, pooled]", rho_l, p_l)

    # -- three_scale.csv (per-row, per-level, and the joint multivariate control) --
    ts = pd.read_csv(RESULTS_DIR / "three_scale.csv")
    for col in ("active_len", "neutral_len", "irrelevant_len", "seq_len"):
        rho, p = spearman(ts[col], ts["winding"])
        add("three_scale", f"winding~{col} [per-row]", rho, p)
        rho_l, _, p_l = _level_p(ts, col, "winding")
        add("three_scale", f"winding~{col} [per-level]", rho_l, p_l)
    mv = multivariate_rank_control(
        ts["winding"].to_numpy(),
        {
            "active_len": ts["active_len"].to_numpy(),
            "neutral_len": ts["neutral_len"].to_numpy(),
            "irrelevant_len": ts["irrelevant_len"].to_numpy(),
        },
    )
    for name, (beta, p) in mv.items():
        add("three_scale", f"winding~{name} [multivariate]", beta, p)

    # -- forceloop.csv (the Fisher exact test) --
    fl = pd.read_csv(RESULTS_DIR / "forceloop.csv")
    n_ops_levels = sorted(fl["n_ops"].unique())
    sub_a = fl[(fl["num_steps"] == 16) & (fl["n_ops"] == n_ops_levels[0])]["shape"]
    sub_b = fl[(fl["num_steps"] == 16) & (fl["n_ops"] == n_ops_levels[1])]["shape"]
    settled_a, unsettled_a = int((sub_a == "settle").sum()), int((sub_a != "settle").sum())
    settled_b, unsettled_b = int((sub_b == "settle").sum()), int((sub_b != "settle").sum())
    _, p = fisher_exact([[settled_a, unsettled_a], [settled_b, unsettled_b]])
    label = f"unsettled n_ops={n_ops_levels[0]} vs {n_ops_levels[1]} [Fisher, ns=16]"
    add("forceloop", label, float("nan"), float(p))

    return tests


def main() -> None:
    """Collect every test, apply BH under both family definitions, report."""
    tests = _collect()
    df = pd.DataFrame(tests)

    q_all, sig_all = benjamini_hochberg(df["p"].to_numpy())
    df["q_project_wide"] = q_all
    df["sig_project_wide"] = sig_all

    df["q_per_experiment"] = float("nan")
    df["sig_per_experiment"] = False
    for _family, idx in df.groupby("family").groups.items():
        q_fam, sig_fam = benjamini_hochberg(df.loc[idx, "p"].to_numpy())
        df.loc[idx, "q_per_experiment"] = q_fam
        df.loc[idx, "sig_per_experiment"] = sig_fam

    df["sig_raw"] = df["p"] < 0.05
    df = df.sort_values("p").reset_index(drop=True)

    n = len(df)
    print(f"Collected {n} correlation tests across {df['family'].nunique()} experiments.\n")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    cols = [
        "family",
        "test",
        "stat",
        "p",
        "sig_raw",
        "q_project_wide",
        "sig_project_wide",
        "sig_per_experiment",
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print(f"\nRaw p<0.05: {df['sig_raw'].sum()}/{n}")
    print(f"Survives project-wide BH-FDR (alpha=0.05): {df['sig_project_wide'].sum()}/{n}")
    print(f"Survives per-experiment BH-FDR (alpha=0.05): {df['sig_per_experiment'].sum()}/{n}")

    print("\n--- The two 'prime target' anomalies flagged in project_plan.md §0.2 ---")
    for label in ("maxtask winding~n_ops", "local winding~n_ops"):
        hit = df[df["test"].str.contains(label) & df["test"].str.contains("per-level")]
        for _, row in hit.iterrows():
            print(
                f"{row['family']:22s} {row['test']:45s} rho={row['stat']:+.3f} p={row['p']:.4g} "
                f"-> q_project_wide={row['q_project_wide']:.4g} "
                f"({'SURVIVES' if row['sig_project_wide'] else 'FDR CASUALTY'})"
            )

    df.to_csv(RESULTS_DIR / "fdr_correction.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'fdr_correction.csv'}")


if __name__ == "__main__":
    main()
