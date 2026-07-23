"""Experiment — counting accuracy via generate vs count length (behavioural axis).

Reproduces results/counting_accuracy.csv. Cached: if the CSV exists, only the
analysis is printed (no GPU). Result: accuracy collapses to 0% for len >= 8.
OWNER: Data+Analysis.

Run: uv run python -m scripts.run_accuracy
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model, save_partial
from traj_geom.metrics.dynamics import contraction_rate, steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.synthetic import make_counting_task

N_OPS = (2, 4, 8, 16, 24, 32, 48)
N_SEEDS = 8


def compute() -> pd.DataFrame:
    """Decode the answer and score the trajectory for each counting task."""
    from traj_geom.eval import counting_correct
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []
    for n_ops in tqdm(N_OPS, desc="accuracy"):
        for s in range(N_SEEDS):
            try:
                t = make_counting_task(n_ops, seed=s)
                ok = counting_correct(model, tok, t["prompt"], t["answer"], num_steps=32)
                tr = extract_trajectory(model, tok, t["prompt"], num_steps=64, seed=0)
                rows.append(
                    {
                        "n_ops": n_ops,
                        "correct": bool(ok),
                        "steps_settle": steps_to_settle(tr),
                        "winding": abs(winding_of(tr, 4)),
                        "contraction": contraction_rate(tr),
                    }
                )
                save_partial(rows, "counting_accuracy.csv")
            except Exception as e:  # noqa: BLE001  (skip flaky generations, keep going)
                print("skip", n_ops, s, repr(e))
    return pd.DataFrame(rows)


def main() -> None:
    """Report accuracy by count length and geometry split by correctness."""
    acc = cached("counting_accuracy.csv", compute)
    print("accuracy by length:\n", acc.groupby("n_ops")["correct"].mean().round(2))
    print(
        "\ngeometry by correctness:\n",
        acc.groupby("correct")[["steps_settle", "contraction"]].mean().round(2),
    )


if __name__ == "__main__":
    main()
