"""Which items does Huginn have DYNAMIC RANGE on -- with the estimator checked, not assumed.

WHY A SEPARATE ANALYSIS EXISTS. `geometry-census` screens every (family, item)
with SCREEN_DRAWS unseeded h_0 draws, then re-measures the ones that split -- or
sit near the decision boundary -- with DEEPEN_DRAWS more. Its in-kernel
`summarise()` pools BOTH stages, and the concern recorded in `docs/directions.md`
section E3 was that this is a **winner's curse**: an item enters stage 2 partly
because its 4 screening draws happened to look balanced, so folding those draws
back in should bias it toward looking more balanced than it is.

**THAT CONCERN WAS CHECKED BY SIMULATION AND IS WRONG FOR THIS DESIGN, and the
correction is recorded here rather than quietly dropped.** Over 4000 simulated
items with the kernel's own selection rule (both classes present in 4 screening
draws), measuring apparent balance against ground truth:

    bias in apparent balance     stage-2 only  -0.0180   pooled  -0.0014
    mean |estimate - truth|      stage-2 only   0.0778   pooled   0.0698

Pooling is better on BOTH. The selection condition is weak -- it excludes only
items pinned near 0 or 1 -- so it induces little curse, while dropping the 4
screening draws costs real precision. The residual bias is dominated not by
selection but by noise passing through the non-linear |f - 0.5| transform, which
penalises the SMALLER sample. So this script reports the pooled estimate as
primary, shows the stage-2-only estimate beside it, and prints the measured gap
so the reader can see the curse is small rather than take either on faith.

THE TRAP THAT DOES STAND, AND THIS SCRIPT CANNOT FIX IT.
Selecting items because their outcome varies and then asking whether geometry
predicts that outcome ON THE SAME DRAWS is circular, and no amount of stage
splitting repairs it. The correct pattern is the one `geometry-h0bank` used:
select from one run's ranks, bank and analyse a DIFFERENT run. **The item list
this script emits is a selector for a fresh banking run, never a dataset to
analyse on its own draws.**

WHAT THE OUTPUT IS FOR. Two experiments have now died on task supply rather than
method -- D47, and D95 whose gate wanted 4 of 9 splitting prompts and got 2. This
produces the ranked list of items whose outcome actually moves, which every
subsequent causal or within-item design selects from instead of guessing. It also
supplies the difficulty-matched candidates D96 needs: that run could not separate
"QK tracks the computation" from "QK tracks how well it is going" because its B
arm was simply an easier task, and fixing it needs markers matched on rank.

Run: uv run python -m scripts.run_census_analysis
"""

from __future__ import annotations

import json
import os

import pandas as pd

SRC = os.path.join("scratch", "kaggle_census", "out", "census.json")
OUT = os.path.join("results", "census.csv")
BAND = (0.2, 0.8)


def load(path: str = SRC) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def split_stages(rows: list, screen_draws: int) -> pd.DataFrame:
    """Tag each row stage 1 or stage 2 by APPEND ORDER within its item.

    The kernel appends stage-1 draws for every item first, then stage-2 draws for
    the selected ones, so an item's first `screen_draws` rows are its screening
    draws. This is what makes the two estimates separable at all -- the simulation
    in the module docstring then shows pooling is the better of the two.
    """
    out = []
    seen: dict[tuple, int] = {}
    for r in rows:
        if not r.get("ok"):
            continue
        key = (r["family"], r["item"])
        i = seen.get(key, 0)
        seen[key] = i + 1
        out.append({**{k: r[k] for k in
                       ("family", "item", "best_rank", "best_depth", "correct",
                        "final_rank", "n_tokens", "multi_token_gold", "gold")},
                    "stage": 1 if i < screen_draws else 2})
    return pd.DataFrame(out)


def summarise(df: pd.DataFrame, stage: int | None = None) -> pd.DataFrame:
    d = df if stage is None else df[df["stage"] == stage]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(["family", "item"])
    out = g.agg(n_draws=("correct", "size"), n_correct=("correct", "sum"),
                median_best_rank=("best_rank", "median"),
                min_best_rank=("best_rank", "min"),
                max_best_rank=("best_rank", "max"),
                best_depth_min=("best_depth", "min"),
                best_depth_max=("best_depth", "max"),
                multi_token_gold=("multi_token_gold", "first"),
                gold=("gold", "first")).reset_index()
    out["frac_correct"] = out["n_correct"] / out["n_draws"]
    out["splits"] = (out["n_correct"] > 0) & (out["n_correct"] < out["n_draws"])
    out["rank_spread"] = out["max_best_rank"] / out["min_best_rank"].clip(lower=1)
    return out


def report(s1: pd.DataFrame, s2: pd.DataFrame, both: pd.DataFrame) -> str:
    lines = ["", "  STAGE 1 (screen) -- SELECTOR ONLY, not an estimate",
             f"    items screened: {len(s1)}   splitting on screening draws: "
             f"{int(s1['splits'].sum())}"]
    if s2.empty:
        lines += ["", "  NO stage-2 rows found -- either the kernel stopped early or "
                      "no item was selected.",
                  "  Nothing here may be quoted: the stage-1 numbers are the "
                  "selection, not a measurement."]
        return "\n".join(lines)

    lines += ["", "  STAGE 2 (deepen) -- fresh draws on the selected items",
              f"    items re-measured: {len(s2)}   splitting on stage-2 draws: "
              f"{int(s2['splits'].sum())}"]

    # Both estimates side by side. Simulation (module docstring) says POOLED is
    # better on bias AND error, so it is primary; the gap is printed so the size
    # of the winner's curse is visible rather than argued about.
    m = s2.merge(both[["family", "item", "frac_correct"]],
                 on=["family", "item"], suffixes=("_s2", "_pooled"))
    if len(m):
        bal_s2 = (0.5 - (m["frac_correct_s2"] - 0.5).abs()).mean()
        bal_pool = (0.5 - (m["frac_correct_pooled"] - 0.5).abs()).mean()
        lines += ["", "  WINNER'S CURSE, MEASURED ON THE REAL DATA",
                  f"    mean apparent balance (0.5 = perfectly split), pooled  = {bal_pool:.4f}",
                  f"    mean apparent balance,                        stage-2 = {bal_s2:.4f}",
                  f"    gap = {bal_pool - bal_s2:+.4f}  (simulation predicts a small "
                  f"positive gap, ~+0.017, and that pooling is still the better "
                  f"estimator overall)"]

    # PRIMARY estimate: pooled, restricted to items that actually reached stage 2.
    deep_keys = set(zip(s2["family"], s2["item"], strict=True))
    prim = both[[(f, i) in deep_keys for f, i in zip(both["family"], both["item"], strict=True)]]
    band = prim[(prim["frac_correct"] >= BAND[0]) & (prim["frac_correct"] <= BAND[1])]
    lines += ["", f"  ITEMS IN THE {BAND[0]:.0%}-{BAND[1]:.0%} BAND (pooled estimate, primary): "
                  f"{len(band)}", "",
              f"  {'family':>16} {'item':>4} {'n':>3} {'frac_cor':>8} {'med_rank':>8} "
              f"{'spread':>7} {'depth':>9} {'multi_tok':>9}"]
    for _, r in band.sort_values("frac_correct", ascending=False).head(40).iterrows():
        lines.append(f"  {r['family']:>16} {int(r['item']):>4} {int(r['n_draws']):>3} "
                     f"{r['frac_correct']:>8.2f} {r['median_best_rank']:>8.1f} "
                     f"{r['rank_spread']:>7.2f} "
                     f"{int(r['best_depth_min']):>4}-{int(r['best_depth_max']):<4} "
                     f"{str(bool(r['multi_token_gold'])):>9}")
    clean = band[~band["multi_token_gold"].astype(bool)]
    lines += ["", f"  USABLE (in band AND single-token gold, so free of the D89 "
                  f"defect): {len(clean)}"]
    if len(clean) >= 8:
        lines.append("    Enough supply for a powered within-item design -- D95's gate "
                     "wanted 4 and got 2.")
    elif len(clean):
        lines.append(f"    Thin: {len(clean)} items. Better than D95's 2, but a powered "
                     f"design needs more -- consider the difficulty x depth grid's "
                     f"parametric families instead.")
    else:
        lines.append("    **NONE.** The hand-authored family pool does not contain "
                     "usable boundary items, and the parametric grid "
                     "(`scratch/ds_grid/`) becomes the only route.")

    fams = sorted(clean["family"].unique()) if len(clean) else []
    lines += ["", f"  families supplying usable items: {fams if fams else 'none'}"]
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not os.path.exists(SRC):
        print(f"no census at {SRC} -- pull the geometry-census output first.")
        return
    d = load()
    rows = d["rows"] if isinstance(d, dict) else d
    cfg = d.get("config", {}) if isinstance(d, dict) else {}
    screen = int(cfg.get("screen_draws", 4))
    df = split_stages(rows, screen)
    s1, s2, both = summarise(df, 1), summarise(df, 2), summarise(df)
    if len(s2):
        deep_keys = set(zip(s2["family"], s2["item"], strict=True))
        prim = both[[(f, i) in deep_keys
                     for f, i in zip(both["family"], both["item"], strict=True)]]
        save_table(OUT, prim, kind="census_pooled_primary", screen_draws=screen,
                   band=list(BAND), n_items_screened=int(len(s1)),
                   note="pooled estimate is primary; simulation showed the "
                        "winner's curse is small and pooling wins on bias and error")
        print(f"saved {OUT}")
    print(report(s1, s2, both))


if __name__ == "__main__":
    main()
