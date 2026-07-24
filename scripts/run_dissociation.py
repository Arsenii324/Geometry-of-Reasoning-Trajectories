"""Experiment D — length-matched track-vs-local dissociation (the killer, H3).

`track` and `local` share an identical body and differ only in the question, so
they are the same token length. If effective compute rises with n_ops for `track`
but not `local`, the effect is state-holding, not prompt length.

Reproduces results/dissociation.csv (--seeds 5) or dissociation_15seed.csv
(--seeds 15) and (re)draws figures/dissociation.png. OWNER: Data+Analysis.
STATUS: implemented, verified-live. The 5-seed run's "local winding~n_ops
    [per-level] = -0.943" does NOT replicate at 15 seeds (rho drops to -0.543,
    n.s., and even flips sign in the per-row statistic) -- a small-N artifact,
    not a BH-FDR casualty (it survives FDR at 5 seeds; the 15-seed rerun is
    what actually kills it). See claims_ledger.md D13. Prefer --seeds 15 for
    any claim; --seeds 5 alone is not trustworthy for this specific number.

Run: python -m scripts.run_dissociation --seeds 15
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from scripts._common import FIGURES_DIR, cached, load_model, save_partial
from traj_geom.analysis.correlate import fmt_by_level, spearman
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.synthetic import make_variants

N_OPS = (4, 8, 16, 24, 32, 48)


def _compute_fn(n_seeds: int, results_name: str):
    """Build the cache-miss compute for a given seed count."""

    def compute() -> pd.DataFrame:
        from traj_geom.extraction.hook import extract_trajectory

        model, tok = load_model()
        rows = []
        n_failed = 0
        for n_ops in tqdm(N_OPS, desc=f"{n_seeds}seed"):
            for s in range(n_seeds):
                v = make_variants(n_ops, seed=s)
                for kind in ("track", "local"):
                    try:
                        seq_len = int(tok(v[kind], return_tensors="pt").input_ids.shape[1])
                        tr = extract_trajectory(model, tok, v[kind], num_steps=64, seed=0)
                        rows.append(
                            {
                                "n_ops": n_ops,
                                "kind": kind,
                                "seq_len": seq_len,
                                "winding": abs(winding_of(tr, burn=4)),
                                "steps_settle": steps_to_settle(tr),
                            }
                        )
                    except Exception as e:  # noqa: BLE001 -- one bad config must not
                        # lose the rest of this sweep's already-completed rows.
                        n_failed += 1
                        print(f"n_ops={n_ops} seed={s} kind={kind}: skipping after error: {e!r}")
                    else:
                        save_partial(rows, results_name)
        if n_failed:
            print(f"run_dissociation: {n_failed} configs failed and were skipped.")
        return pd.DataFrame(rows)

    return compute


def plot(ctrl: pd.DataFrame, path) -> None:
    """Draw the two-panel track-vs-local figure from cached data."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"track": "#d62728", "local": "#1f77b4"}
    names = {"track": "track (needs accumulation)", "local": "local (last step only)"}
    specs = [
        ("steps_settle", "Steps-to-settle vs count length", "steps to settle"),
        ("winding", "|winding| vs count length", "|winding|"),
    ]
    for ax, (metric, ttl, ylab) in zip(axes, specs, strict=True):
        for kind in ("track", "local"):
            s = ctrl[ctrl.kind == kind]
            g = s.groupby("n_ops")[metric]
            ax.errorbar(
                g.mean().index, g.mean().values, yerr=g.sem().values,
                marker="o", capsize=3, color=colors[kind], label=names[kind],
            )
            rho, p = spearman(s.n_ops, s[metric])
            ax.text(
                0.04, 0.95 if kind == "track" else 0.06,
                f"{kind}: rho={rho:+.2f}, p={p:.3f}", transform=ax.transAxes,
                color=colors[kind], va="top" if kind == "track" else "bottom", fontsize=9,
            )
        ax.set_xlabel("n_ops (count length)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle(
        "Length-matched control: state-holding resists compression (H3)", fontweight="bold"
    )
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


def main() -> None:
    """Run the dissociation experiment at the requested seed count and plot it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5, choices=(5, 15))
    args = ap.parse_args()

    name = "dissociation.csv" if args.seeds == 5 else "dissociation_15seed.csv"
    ctrl = cached(name, _compute_fn(args.seeds, name))
    for kind in ("track", "local"):
        s = ctrl[ctrl.kind == kind]
        # Canonical: per-level Spearman + N.
        print(f"{kind} [per-level] | steps~n_ops:", fmt_by_level(s, "n_ops", "steps_settle"))
        print(f"{kind} [per-level] | winding~n_ops:", fmt_by_level(s, "n_ops", "winding"))
        # Secondary (per-row):
        print(
            f"{kind} [per-row]   | winding~n_ops:", spearman(s.n_ops, s.winding),
            "| steps~n_ops:", spearman(s.n_ops, s.steps_settle),
        )
    plot(ctrl, FIGURES_DIR / "dissociation.png")


if __name__ == "__main__":
    main()
