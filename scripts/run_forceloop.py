"""Experiment E — force loops by starving the compute budget (num_steps).

Reproduces results/forceloop.csv. At num_steps=16 loops emerge for longer counts;
at >=24 steps everything settles again. Loops are a symptom of too little compute,
not of depth per se. OWNER: Shapes+Gate.

Run: python -m scripts.run_forceloop
"""

from __future__ import annotations

import pandas as pd
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


def main() -> None:
    """Report the shape mix per budget and mean |winding|."""
    fl = cached("forceloop.csv", compute)
    print(fl.groupby(["num_steps", "n_ops"])["shape"].value_counts())
    print(fl.groupby("num_steps")["winding"].mean().round(3))


if __name__ == "__main__":
    main()
