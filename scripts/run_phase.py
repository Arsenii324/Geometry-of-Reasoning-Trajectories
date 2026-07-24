"""Experiment — phase map of the "unsettled" fraction over (num_steps x n_ops).

Reproduces results/phase.csv and generates figures/phase.png: for each combination
of compute budget (num_steps) and count length (n_ops), the fraction of runs that
did NOT settle (loop/drift). OWNER: Shapes+Gate.
STATUS: implemented, verified-live. Same underlying phenomenon as
    `run_forceloop.py`'s B4 result (starved compute budget induces loops) --
    this script maps it over a wider (num_steps, n_ops) grid but runs no
    significance test itself; `run_forceloop.py` is the one with the real
    Fisher exact test (claims_ledger.md B4).

Run: python -m scripts.run_phase
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from scripts._common import FIGURES_DIR, cached, load_model, save_partial
from traj_geom.shapes.gate import classify_shape
from traj_geom.shapes.synthetic import make_variants

NUM_STEPS = (10, 14, 16, 18, 20, 24)
N_OPS = (4, 8, 16, 24, 32, 48)
N_SEEDS = 6


def compute() -> pd.DataFrame:
    """Classify the regime for each (num_steps, n_ops, seed)."""
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []
    n_failed = 0
    for ns in tqdm(NUM_STEPS, desc="num_steps"):
        for n_ops in N_OPS:
            for s in range(N_SEEDS):
                try:
                    v = make_variants(n_ops, seed=s)
                    tr = extract_trajectory(model, tok, v["track"], num_steps=ns, seed=0)
                    rows.append({"num_steps": ns, "n_ops": n_ops, "regime": classify_shape(tr)})
                except Exception as e:  # noqa: BLE001 -- one bad config must not lose
                    # the rest of this sweep's already-completed rows.
                    n_failed += 1
                    print(f"num_steps={ns} n_ops={n_ops} seed={s}: skipping after error: {e!r}")
                else:
                    save_partial(rows, "phase.csv")
    if n_failed:
        print(f"run_phase: {n_failed} configs failed and were skipped.")
    return pd.DataFrame(rows)


def plot(phase: pd.DataFrame, path) -> None:
    """Draw the unsettled-fraction heatmap from cached data."""
    phase = phase.copy()
    phase["unsettled"] = (phase["regime"] != "settle").astype(int)  # 1 = loops/drifts
    grid = phase.pivot_table(index="num_steps", columns="n_ops", values="unsettled", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.imshow(grid.values, aspect="auto", origin="lower", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("count length (n_ops)")
    ax.set_ylabel("compute budget (num_steps)")
    ax.set_title("Fraction 'unsettled' (loops/drifts)")
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            ax.text(j, i, f"{grid.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


def main() -> None:
    """Compute (or load) the phase grid and render the heatmap."""
    phase = cached("phase.csv", compute)
    plot(phase, FIGURES_DIR / "phase.png")


if __name__ == "__main__":
    main()
