"""Plot the regimes the trajectories actually occupy, trained against untrained.

WHAT EACH FIGURE IS FOR
    1. readout_regimes.png
        Per-instance absolute decoding error against unroll depth, log-log, one
        panel per model. The horizontal line at 0.5 counts is the DECISION
        BOUNDARY: an integer count is recovered iff the error falls below it.
        Instances are coloured by which side they end on, so the "edge of
        correctness" -- the band that is still straddling 0.5 at r=64 -- is
        visible rather than averaged away. This is D41 without the mean hiding
        the spread.

    2. answer_manifold_pca.png
        The 220 answer-token states at r=64, projected on their own top two PCs,
        coloured by the true count. If the count is carried linearly the cloud is
        a 1-D curve in count; the tighter the curve the more precisely decodable.
        Trained and untrained side by side is D41's 20x precision gap made
        visual, and the two panels are on a SHARED colour scale so they are
        comparable.

    3. register_trajectory_pca.png
        Motion through state space as the model READS the sequence -- the 74
        per-position states of a single prompt, own top two PCs, coloured by the
        running count. A register shows up as monotone drift along one axis. Task
        a (running count) against task b (nesting depth), which returns to zero,
        so the two should look different in a specific way: a drifts, b closes.

    4. log_distance_from_end.png
        ||h_i - h_end|| on a log axis. Two regimes are expected and both matter:
        an exponential approach, then a flat ARITHMETIC FLOOR where float16
        storage (these arrays are saved float16) stops resolving the difference.
        Every metric in this project that mixed the two regimes gave a wrong
        answer, so the floor is drawn explicitly rather than left implicit.

CAVEATS THAT THE FIGURES CANNOT SHOW, STATED HERE
    * The saved arrays are float16, so the floor in figure 4 is a STORAGE floor,
      not the model's compute floor (which is lower -- the runs were float32).
    * The states lie on a sphere of radius ~76.37 (RMSNorm). PCA is on centred
      data, so the leading components describe motion ON the sphere, not radial
      motion, of which there is essentially none.
    * Figures 1 and 2 are the counting task at M=64, where the trained model's
      measured accuracy is ~0 (claims_ledger D41(3)). These show what is
      DECODABLE, which this project has repeatedly found is not what is USED.

Run:  python -m scripts.plot_regimes      (no GPU, no network)
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
TRAINED = os.path.join(ROOT, "scratch", "kaggle_readout", "out")
UNTRAINED = os.path.join(ROOT, "scratch", "kaggle_untrained_depth", "out")
STATES = os.path.join(ROOT, "scratch", "kaggle_states", "out2")

DECISION = 0.5      # an integer count is recovered iff |error| < 0.5
FLOOR_C = "#b0413e"
os.makedirs(FIG, exist_ok=True)


def _pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-2 PCs of centred x, plus explained-variance fractions."""
    xc = x - x.mean(0, keepdims=True)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2 / max((s ** 2).sum(), 1e-30)
    return u[:, :2] * s[:2], var[:2]


# --- 1. readout regimes, with the decision boundary drawn ------------------


def fig_readout_regimes() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, d, name in ((axes[0], TRAINED, "trained"), (axes[1], UNTRAINED, "untrained")):
        path = os.path.join(d, "per_instance_error.json")
        if not os.path.exists(path):
            ax.set_title(f"{name}: no data")
            continue
        blob = json.load(open(path, encoding="utf-8"))
        eb = blob["error_by_depth"]
        depths = sorted(int(k) for k in eb)
        err = np.array([eb[str(r)] for r in depths], float)      # [n_depth, n_inst]
        final = err[-1]
        resolved = final < DECISION
        for j in range(err.shape[1]):
            ax.plot(depths, np.maximum(err[:, j], 1e-3),
                    color=("#2c7fb8" if resolved[j] else "#d95f02"),
                    alpha=0.18, lw=0.8)
        ax.plot(depths, np.median(err, axis=1), color="k", lw=2.2, label="median")
        ax.axhline(DECISION, color=FLOOR_C, ls="--", lw=1.6,
                   label=f"decision boundary ({DECISION} counts)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("recurrent depth r")
        ax.set_title(f"{name} — {int(resolved.sum())}/{len(final)} instances "
                     f"resolve the integer count")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("|decoding error|  (counts)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Per-instance decodability vs depth — orange instances never "
                 "resolve; the spread is the result, not the mean", fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "readout_regimes.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# --- 2. the answer manifold, trained vs untrained --------------------------


def fig_answer_manifold() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    totals = None
    for ax, d, name in ((axes[0], TRAINED, "trained"), (axes[1], UNTRAINED, "untrained")):
        sp = os.path.join(d, "answer_states_r64.npy")
        mp = os.path.join(d, "meta.json")
        if not (os.path.exists(sp) and os.path.exists(mp)):
            ax.set_title(f"{name}: no data")
            continue
        x = np.load(sp).astype(np.float64)
        totals = np.array(json.load(open(mp, encoding="utf-8"))["totals"], float)
        z, var = _pca2(x)
        sc = ax.scatter(z[:, 0], z[:, 1], c=totals, cmap="viridis", s=26,
                        edgecolor="none", vmin=totals.min(), vmax=totals.max())
        ax.set_title(f"{name} — PC1+PC2 explain {100 * var.sum():.1f}% of variance")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
    if totals is not None:
        fig.colorbar(sc, ax=axes, label="true count", fraction=0.025)
    fig.suptitle("Answer-token state at r=64, coloured by the count it encodes — "
                 "a tighter 1-D curve means a more precisely decodable count",
                 fontsize=11)
    p = os.path.join(FIG, "answer_manifold_pca.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# --- 3. reading the sequence: the register as motion -----------------------


def fig_register_trajectory() -> str:
    picks = [("a_seed0.npy", "task a — running count"),
             ("b_bal_seed0.npy", "task b — nesting depth (returns to 0)")]
    fig, axes = plt.subplots(1, len(picks), figsize=(12, 5))
    for ax, (fn, title) in zip(np.atleast_1d(axes), picks, strict=False):
        path = os.path.join(STATES, fn)
        if not os.path.exists(path):
            ax.set_title(f"{fn}: absent")
            continue
        x = np.load(path).astype(np.float64)          # [n_pos, 5280]
        z, var = _pca2(x)
        pos = np.arange(len(z))
        ax.plot(z[:, 0], z[:, 1], color="0.75", lw=0.9, zorder=1)
        sc = ax.scatter(z[:, 0], z[:, 1], c=pos, cmap="plasma", s=22, zorder=2)
        ax.scatter(*z[0], marker="o", s=110, facecolor="none", edgecolor="k",
                   lw=1.8, zorder=3, label="first token")
        ax.scatter(*z[-1], marker="s", s=110, facecolor="none", edgecolor="k",
                   lw=1.8, zorder=3, label="last token")
        ax.set_title(f"{title}\nPC1+PC2 = {100 * var.sum():.1f}% of variance")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.colorbar(sc, ax=ax, label="token position", fraction=0.04)
    fig.suptitle("State motion while READING the sequence (per-position, one prompt) "
                 "— a register shows as monotone drift along one axis", fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "register_trajectory_pca.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# --- 4. log distance from the endpoint, and the floor ----------------------


def fig_log_distance() -> str:
    import glob
    fig, ax = plt.subplots(figsize=(8, 5.2))
    kinds = {"a": "#2c7fb8", "b_bal": "#d95f02", "b_unbal": "#7570b3"}
    seen = set()
    floors = []
    for path in sorted(glob.glob(os.path.join(STATES, "*.npy"))):
        base = os.path.basename(path)
        kind = next((k for k in ("b_unbal", "b_bal", "a") if base.startswith(k)), None)
        if kind is None:
            continue
        x = np.load(path).astype(np.float64)
        d = np.linalg.norm(x - x[-1], axis=1)[:-1]
        if not len(d):
            continue
        floors.append(np.median(d[-max(3, len(d) // 8):]))
        lbl = kind if kind not in seen else None
        seen.add(kind)
        ax.plot(np.arange(len(d)), np.maximum(d, 1e-6), color=kinds[kind],
                alpha=0.45, lw=0.9, label=lbl)
    if floors:
        f = float(np.median(floors))
        ax.axhline(f, color=FLOOR_C, ls="--", lw=1.8,
                   label=f"float16 storage floor ≈ {f:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("token position i")
    ax.set_ylabel(r"$\|h_i - h_{\mathrm{end}}\|$")
    ax.set_title("Distance from the endpoint, log-scaled — the flat tail is\n"
                 "arithmetic, not the model. Mixing the two regimes is what\n"
                 "made four earlier metrics wrong.", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG, "log_distance_from_end.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    for fn in (fig_readout_regimes, fig_answer_manifold,
               fig_register_trajectory, fig_log_distance):
        try:
            print("wrote", os.path.relpath(fn(), ROOT))
        except Exception as e:                                    # noqa: BLE001
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
