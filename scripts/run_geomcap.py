"""Does the trajectory's SHAPE track what the model can compute? Bounded either way.

Reads the banked states from `scratch/kaggle_geomcap/out/` and writes
`results/geomcap.csv` (per orbit) and `results/geomcap_summary.csv` (per statistic).
No GPU. OWNER: Data+Analysis. STATUS: implemented 2026-08-09.

THE DISSOCIATION THIS TESTS, on ONE set of forwards. The same 504 orbits carry a
capability axis spanning 21 families and four geometric statistics. If capability
varies from 0% to 100% while every geometric statistic is flat against it, the
shape of the latent path does not encode what the model is doing. D79 already found
the same at the item level within a family.

AND WHY A NULL HERE IS A RESULT RATHER THAN A SHRUG. Two bounds accompany it, both
measured in this run rather than argued:

  THE CEILING. D78 established h_0 is drawn unseeded, so the `rep` block --  ten
  forwards of one prompt, differing in h_0 alone -- measures the noise floor of
  every statistic directly. `analysis.reliability` turns that into the correlation a
  PERFECT relation would have shown. An observed rho of 0.1 against a band of
  [0.6, 0.9] is a null about the geometry; against a band of [0.1, 0.4] it is a null
  about the instrument, and the two must not be confused. D70 is the cautionary
  case: a content gap of ~0 reported from a probe that could not have moved 0.13.

  THE POSITIVE CONTROL. Prompt length is deliberately correlated against the same
  statistics. A geometry that tracks NOTHING measurable would leave the capability
  null uninterpretable; a geometry that tracks length but not capability is a
  finding, because it shows the statistics respond to the input's surface form
  while staying blind to the computation performed on it.

THE METRICS ARE D74's AND D76's, NOT THE PLANAR ONES. D74 measured the orbit at ~13
effective dimensions, so winding and its nine variants describe a 2-D shadow of an
object with ~11 other active directions. D76's consecutive-step cosine is exactly
cos(phi) for a linear map -- no window, plane, centre or surrogate -- and separates
trained from untrained completely, which is what gives it standing here.

Run: uv run python -m scripts.run_geomcap
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from traj_geom.analysis.correlate import spearman
from traj_geom.analysis.reliability import decompose
from traj_geom.metrics.dimension import effective_dimension
from traj_geom.metrics.dmd import pre_floor_window

BANK = os.path.join("scratch", "kaggle_geomcap", "out")
OUT = os.path.join("results", "geomcap.csv")
OUT_SUM = os.path.join("results", "geomcap_summary.csv")
BATTERY = os.path.join("results", "battery.csv")
METRICS = ("pr", "cos_consecutive", "contraction", "settle")
P1_MIN_RHO = 0.7


def contraction_and_settle(traj: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """Geometric decay rate of the step norm, and the unroll it reaches 1% of peak.

    Identical to `scripts.run_b6`'s, deliberately: the two banks run at the same
    depth so that their per-orbit rows pool, and a statistic computed two ways
    would silently break that.
    """
    d = np.linalg.norm(np.diff(traj, axis=0), axis=1)[lo:hi]
    if len(d) < 4:
        return float("nan"), float("nan")
    y = np.log(np.clip(d, 1e-30, None))
    slope = float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])
    below = np.flatnonzero(d < d.max() * 0.01)
    return float(np.exp(slope)), float(below[0] if len(below) else len(d))


def collect(path: str = BANK) -> pd.DataFrame:
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fh:
        recs = [r for r in json.load(fh) if r.get("ok")]
    rows = []
    for r in recs:
        traj = np.load(os.path.join(path, r["tag"] + ".npy")).astype(np.float64)
        lo, hi = pre_floor_window(traj)
        dim = effective_dimension(traj, lo=lo, hi=hi, n_null=20)
        if not dim.get("ok"):
            continue
        rho, settle = contraction_and_settle(traj, lo, hi)
        rows.append({
            "tag": r["tag"], "block": r["block"], "family": r["family"],
            "item": r["item"], "gold": r["gold"], "correct": bool(r["correct"]),
            "best_rank": r["best_rank"], "best_depth": r["best_depth"],
            "n_tokens": r["n_tokens"], "h0_seed": r.get("h0_seed"),
            "state_sha": r.get("state_sha"), "window": hi - lo,
            "log_rank": float(np.log10(r["best_rank"])),
            "pr": dim["pr"], "cos_consecutive": dim["cos_consecutive"],
            "contraction": rho, "settle": settle,
        })
    return pd.DataFrame(rows)


def determinism_verdict(df: pd.DataFrame) -> dict:
    """Did the seeded block reproduce itself, and did the unseeded one not?

    Both halves matter. Seeded orbits that differ would mean h_0 is not the only
    stochastic input and `rep` measures an upper bound rather than the h_0 term.
    Unseeded orbits that AGREE would mean the harness removed the very variation
    the ceiling is built from, and every reliability number below would be a
    tautology reading 1.0.
    """
    fix = df[df["block"] == "fix"]
    rep = df[df["block"] == "rep"]
    per_fix = fix.groupby("family")["state_sha"].nunique()
    per_rep = rep.groupby("family")["state_sha"].nunique()
    rep_n = rep.groupby("family").size()
    return {
        "n_fix_prompts": int(len(per_fix)),
        "fix_all_identical": bool((per_fix == 1).all()) if len(per_fix) else False,
        "fix_worst": int(per_fix.max()) if len(per_fix) else 0,
        "n_rep_prompts": int(len(per_rep)),
        "rep_all_distinct": bool((per_rep == rep_n).all()) if len(per_rep) else False,
        "rep_fewest_distinct": int(per_rep.min()) if len(per_rep) else 0,
    }


def family_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per family: capability, geometry means, and the length covariate."""
    m = df[df["block"] == "main"]
    agg = {"acc": ("correct", "mean"), "log_rank": ("log_rank", "mean"),
           "n_tokens": ("n_tokens", "mean"), "window": ("window", "mean"),
           "n": ("correct", "size")}
    agg.update({k: (k, "mean") for k in METRICS})
    return m.groupby("family").agg(**agg).reset_index()


def covariate_structure(fam: pd.DataFrame) -> dict:
    """How the three family-level axes relate to EACH OTHER, before any geometry.

    Two relations decide how the main result may be worded, and both are properties
    of the task suite rather than of the model:

      acc ~ n_tokens -- harder families tend to have longer prompts, so "tracks
        length but not capability" is only a dissociation if those two axes are not
        the same axis. If they were strongly correlated, no statistic could track
        one and miss the other, and the finding would be unavailable by
        construction rather than absent by measurement.

      window ~ n_tokens -- participation ratio grows with the number of samples,
        and D74(6) records `partial_spearman` REFUSING to separate `ratio` from
        window length at rho = -1.000. If the pre-floor window tracks prompt length,
        then a PR-vs-length correlation may be mediated by window rather than being
        about shape, and that has to be said rather than discovered by a reader.
    """
    out = {}
    for a, b in (("acc", "n_tokens"), ("acc", "window"), ("window", "n_tokens")):
        if fam[a].nunique() > 1 and fam[b].nunique() > 1:
            rho, p = spearman(fam[a].to_numpy(), fam[b].to_numpy())
            out[f"{a}~{b}"] = (rho, p)
        else:
            out[f"{a}~{b}"] = (float("nan"), float("nan"))
    return out


def p1_gate(fam: pd.DataFrame, battery_csv: str = BATTERY) -> dict:
    """Is this run measuring the same capability axis D75 measured?

    D78 makes this a live question rather than a formality: h_0 is unseeded, so the
    two runs cannot be expected to agree item by item, only in their family
    ordering. If even that fails, nothing here may be read against D75.
    """
    if not os.path.exists(battery_csv):
        return {"usable": False, "why": f"{battery_csv} absent"}
    b = pd.read_csv(battery_csv)
    b = b[b["arm"] == "trained"].groupby("family")["correct_best"].mean()
    j = fam.set_index("family").join(b.rename("d75_acc"), how="inner")
    if len(j) < 5:
        return {"usable": False, "why": f"only {len(j)} shared families"}
    rho, p = spearman(j["acc"].to_numpy(), j["d75_acc"].to_numpy())
    return {"usable": True, "rho": rho, "p": p, "n": int(len(j)),
            "passes": bool(rho >= P1_MIN_RHO),
            "here_range": (float(j["acc"].min()), float(j["acc"].max())),
            "d75_range": (float(j["d75_acc"].min()), float(j["d75_acc"].max()))}


def _rho(fam: pd.DataFrame, ok: pd.Series, ycol: str,
         xcol: str) -> tuple[float, float, bool]:
    """Spearman of ``ycol`` against ``xcol``, refusing when it is UNDEFINED.

    scipy returns NaN with a ConstantInputWarning when either column has no spread,
    and `NaN < alpha` is False -- so an undefined correlation silently files as
    "does not track it", which is exactly the sentence P6 is built from.
    `correlate.py` already records this for `steps_settle`, the constant 128 at
    every level in one sweep. Each axis is guarded separately: a metric with no
    spread against capability may still have spread against prompt length, and
    refusing both because one is degenerate would discard the positive control.
    """
    if int(ok.sum()) < 4 or fam.loc[ok, ycol].nunique() < 2 \
            or fam.loc[ok, xcol].nunique() < 2:
        return float("nan"), float("nan"), False
    rho, p = spearman(fam.loc[ok, ycol].to_numpy(), fam.loc[ok, xcol].to_numpy())
    return rho, p, bool(np.isfinite(rho))


def analyse(df: pd.DataFrame, n_boot: int = 400) -> pd.DataFrame:
    """Per statistic: capability correlation, length correlation, and the ceiling."""
    fam = family_table(df)
    main, rep = df[df["block"] == "main"], df[df["block"] == "rep"]
    n_lev = int(len(fam))
    # Bonferroni over the PRE-REGISTERED tests only: capability (P2/P3) and prompt
    # length (P5), one of each per statistic. `log_rank` and `window` are reported
    # as description, not tested, so they do not enlarge the correction -- counting
    # exploratory columns into alpha would penalise the pre-registered tests for
    # curiosity exercised after the fact.
    alpha = 0.05 / (len(METRICS) * 2)
    rows = []
    for k in METRICS:
        ok = fam[k].notna()
        rho_cap, p_cap, def_cap = _rho(fam, ok, k, "acc")
        rho_rank, p_rank, _ = _rho(fam, ok, k, "log_rank")
        rho_len, p_len, def_len = _rho(fam, ok, k, "n_tokens")
        rho_win, p_win, _ = _rho(fam, ok, k, "window")
        if not (def_cap or def_len):
            rows.append({"metric": k, "n_families": n_lev, "alpha": alpha,
                         "usable": False,
                         "why": (f"{k} takes {fam.loc[ok, k].nunique()} distinct "
                                 f"values over {int(ok.sum())} families; no "
                                 f"correlation is defined, which is not the same as "
                                 f"one being zero"),
                         "sig_capability": False, "sig_length": False,
                         "null_is_readable": False})
            continue
        good = main[k].notna()
        d = decompose(main.loc[good, k].to_numpy(), main.loc[good, "family"].to_numpy(),
                      rep.loc[rep[k].notna(), k].to_numpy(),
                      rep.loc[rep[k].notna(), "family"].to_numpy(),
                      m=int(main.loc[good].groupby("family").size().median()),
                      n_levels=n_lev, n_boot=n_boot)
        rows.append({
            "metric": k, "n_families": n_lev, "alpha": alpha, "usable": True,
            "capability_defined": def_cap, "length_defined": def_len,
            "rho_capability": rho_cap, "p_capability": p_cap,
            "rho_log_rank": rho_rank, "p_log_rank": p_rank,
            "rho_length": rho_len, "p_length": p_len,
            "rho_window": rho_win, "p_window": p_win,
            "sig_capability": bool(def_cap and p_cap < alpha),
            "sig_length": bool(def_len and p_len < alpha),
            **{c: d.get(c, float("nan")) for c in
               ("sigma2_family", "sigma2_within_family", "sigma2_h0", "sigma2_item",
                "h0_share", "reliability", "ceiling", "band_lo", "band_hi",
                "wide_lo", "wide_hi")},
            # A null is only readable when a perfect relation would have shown MORE
            # than what was observed. Below the band's lower edge, the design had
            # the power and did not find it; inside it, the design had no power and
            # the number says nothing about the geometry.
            "null_is_readable": bool(
                def_cap and np.isfinite(d.get("wide_lo", float("nan")))
                and abs(rho_cap) < d["wide_lo"]),
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, res: pd.DataFrame, fam: pd.DataFrame, gate: dict,
           det: dict) -> str:
    lines = ["", "  P1 GATE -- is this D75's capability axis?"]
    if not gate.get("usable"):
        lines.append(f"    NOT CHECKABLE: {gate['why']}")
    else:
        lines.append(
            f"    Spearman vs D75 over {gate['n']} families: rho = {gate['rho']:+.3f} "
            f"(p = {gate['p']:.2g}), needs >= {P1_MIN_RHO} -> "
            f"{'PASS' if gate['passes'] else 'FAIL, nothing below is comparable to D75'}")
        lines.append(f"    accuracy range here {gate['here_range'][0]:.0%}-"
                     f"{gate['here_range'][1]:.0%}, in D75 {gate['d75_range'][0]:.0%}-"
                     f"{gate['d75_range'][1]:.0%}")

    lines += ["", "  DETERMINISM CONTROL -- is h_0 the only stochastic input?"]
    lines.append(f"    seeded: {det['n_fix_prompts']} prompts, "
                 + ("all replicates bit-identical"
                    if det["fix_all_identical"]
                    else f"UP TO {det['fix_worst']} DISTINCT -- something else is "
                         f"random, so sigma2_h0 is an upper bound"))
    lines.append(f"    unseeded: {det['n_rep_prompts']} prompts, "
                 + ("every replicate distinct, as D78 predicts"
                    if det["rep_all_distinct"]
                    else f"one prompt gave only {det['rep_fewest_distinct']} distinct "
                         f"states -- the ceiling would be a tautology"))

    lines += ["", "  CAPABILITY SPREAD (the axis the geometry is tested against)"]
    o = fam.sort_values("acc")
    lines.append("    " + "  ".join(f"{r.family} {r.acc:.0%}" for r in o.itertuples()))

    cov = covariate_structure(fam)
    lines += ["", "  HOW THE AXES RELATE TO EACH OTHER (before any geometry)"]
    for key, (rho, p) in cov.items():
        note = ""
        if key == "acc~n_tokens" and np.isfinite(rho) and abs(rho) > 0.7:
            note = ("  <- capability and prompt length are nearly the same axis "
                    "here; a length/capability dissociation is not available")
        if key == "window~n_tokens" and np.isfinite(rho) and abs(rho) > 0.7:
            note = ("  <- a PR-vs-length relation may be mediated by window "
                    "length (D74(6)), not by shape")
        lines.append(f"    {key:>16}: rho = {rho:+.3f} (p = {p:.2g}){note}")

    lines += ["",
              f"  {'metric':>16} {'rho~acc':>8} {'rho~len':>8} {'h0 share':>9} "
              f"{'rel':>6} {'perfect relation would show':>28}  verdict"]
    for _, r in res.iterrows():
        if not r.get("usable", True):
            lines.append(f"  {r['metric']:>16} {'--':>8} {'--':>8} {'--':>9} "
                         f"{'--':>6} {'--':>28}  REFUSED: {r['why']}")
            continue
        band = (f"[{r['wide_lo']:+.2f}, {r['wide_hi']:+.2f}]"
                if np.isfinite(r["wide_lo"]) else "n/a")
        if not r["capability_defined"]:
            verdict = "capability correlation UNDEFINED (no spread)"
        elif r["sig_capability"]:
            verdict = "TRACKS CAPABILITY"
        elif r["null_is_readable"]:
            verdict = "flat, and the design had the power"
        else:
            verdict = "flat, but so is the instrument -- unreadable"
        lines.append(f"  {r['metric']:>16} {r['rho_capability']:>+8.3f} "
                     f"{r['rho_length']:>+8.3f} {r['h0_share']:>9.2f} "
                     f"{r['reliability']:>6.2f} {band:>28}  {verdict}")

    ok = res[res.get("usable", True).astype(bool)] if "usable" in res else res
    live = ok[ok["null_is_readable"] | ok["sig_capability"]]
    lines += ["", f"  {int(ok['sig_capability'].sum())} of {len(ok)} usable statistics "
                  f"track capability; {int(ok['sig_length'].sum())} track prompt "
                  f"length; {len(live)} of {len(ok)} are readable either way."]
    # P6 needs all three legs, and the first is the one that is easy to lose: a
    # capability axis with no spread makes "nothing tracks capability" vacuously
    # true. `capability_defined` is checked before the absence of hits is read as
    # evidence, and the null must be READABLE for at least one statistic -- an
    # unreadable null is a fact about the instrument, not about the model.
    if (len(ok) and ok["capability_defined"].any() and ok["null_is_readable"].any()
            and not ok["sig_capability"].any() and ok["sig_length"].any()):
        lines += [
            "  P6: no geometric statistic tracks what the model can compute, while",
            "      the same statistics on the same orbits DO track how long the",
            "      prompt is. The shape of the latent trajectory responds to the",
            "      input's surface form and not to the computation performed on it.",
        ]
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not os.path.exists(os.path.join(BANK, "manifest.json")):
        print(f"no banked states in {BANK} -- pull the geometry-geomcap output first.")
        return
    df = collect()
    fam = family_table(df)
    res = analyse(df)
    gate, det = p1_gate(fam), determinism_verdict(df)
    save_table(OUT, df, kind="geomcap", source=BANK)
    save_table(OUT_SUM, res, kind="geomcap_summary", source=BANK,
               p1=gate, determinism=det)
    print(f"saved {OUT} ({len(df)} orbits) and {OUT_SUM}")
    print(report(df, res, fam, gate, det))


if __name__ == "__main__":
    main()
