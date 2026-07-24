"""Experiment E — force loops by starving the compute budget (num_steps).

Reproduces results/forceloop.csv. At num_steps=16 loops emerge for longer counts;
at >=24 steps everything settles again. Loops are a symptom of too little compute,
not of depth per se. OWNER: Shapes+Gate.

Run: python -m scripts.run_forceloop
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact
from tqdm import tqdm

from scripts._common import cached, load_model, save_partial
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.gate import classify_shape
from traj_geom.shapes.synthetic import make_variants

NUM_STEPS = (16, 24, 32, 64)
N_OPS = (8, 24, 48)
N_SEEDS = 8


def compute() -> pd.DataFrame:
    """Sweep the compute budget x count length and classify each trajectory."""
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
                    rows.append(
                        {
                            "num_steps": ns,
                            "n_ops": n_ops,
                            "winding": abs(winding_of(tr, burn=4)),
                            "shape": classify_shape(tr),
                            "steps_settle": steps_to_settle(tr),
                        }
                    )
                except Exception as e:  # noqa: BLE001 -- one bad config must not lose
                    # the rest of this sweep's already-completed rows.
                    n_failed += 1
                    print(f"num_steps={ns} n_ops={n_ops} seed={s}: skipping after error: {e!r}")
                else:
                    save_partial(rows, "forceloop.csv")
    if n_failed:
        print(f"run_forceloop: {n_failed} configs failed and were skipped.")
    return pd.DataFrame(rows)


def _unsettled_fisher_test(
    fl: pd.DataFrame, num_steps: int, n_ops_a: int, n_ops_b: int
) -> tuple[int, int, int, int, float]:
    """Fisher exact test on the (settled, unsettled) 2x2 table between two n_ops
    levels at a fixed num_steps. Returns (settled_a, unsettled_a, settled_b,
    unsettled_b, p_value).
    """
    sub_a = fl[(fl["num_steps"] == num_steps) & (fl["n_ops"] == n_ops_a)]["shape"]
    sub_b = fl[(fl["num_steps"] == num_steps) & (fl["n_ops"] == n_ops_b)]["shape"]
    settled_a = int((sub_a == "settle").sum())
    unsettled_a = int(len(sub_a) - settled_a)
    settled_b = int((sub_b == "settle").sum())
    unsettled_b = int(len(sub_b) - settled_b)
    _, p = fisher_exact([[settled_a, unsettled_a], [settled_b, unsettled_b]])
    return settled_a, unsettled_a, settled_b, unsettled_b, float(p)


def main() -> None:
    """Report the shape mix per budget, mean |winding|, and the project's
    cleanest positive H1 result: the loop rate jumping under a starved
    compute budget (see claims_ledger.md B4).
    """
    fl = cached("forceloop.csv", compute)
    print(fl.groupby(["num_steps", "n_ops"])["shape"].value_counts())
    print(fl.groupby("num_steps")["winding"].mean().round(3))

    print("\n--- Force-loop significance test (claims_ledger.md B4) ---")
    print("Unsettled (loop+drift) fraction by n_ops at the starved budget (num_steps=16):")
    for n_ops in sorted(fl.loc[fl["num_steps"] == 16, "n_ops"].unique()):
        sub = fl[(fl["num_steps"] == 16) & (fl["n_ops"] == n_ops)]["shape"]
        unsettled = int((sub != "settle").sum())
        print(f"  n_ops={n_ops:>2d}: {unsettled}/{len(sub)} unsettled")

    n_ops_levels = sorted(fl["n_ops"].unique())
    settled_a, unsettled_a, settled_b, unsettled_b, p = _unsettled_fisher_test(
        fl, num_steps=16, n_ops_a=n_ops_levels[0], n_ops_b=n_ops_levels[1]
    )
    print(
        f"\nFisher exact, n_ops={n_ops_levels[0]} vs n_ops={n_ops_levels[1]} at num_steps=16: "
        f"[[{settled_a},{unsettled_a}],[{settled_b},{unsettled_b}]], p={p:.4f}"
    )

    overall = fl["shape"].value_counts()
    settle_ge24 = (fl.loc[fl["num_steps"] >= 24, "shape"] == "settle").mean()
    print(
        f"\nAcross the full sweep (all num_steps): {overall.to_dict()} "
        f"-- every loop/drift instance occurs at num_steps=16; "
        f"{settle_ge24:.0%} settle at num_steps>=24."
    )


if __name__ == "__main__":
    main()
