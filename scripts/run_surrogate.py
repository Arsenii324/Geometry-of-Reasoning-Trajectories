"""Is the orbit's geometry ANYTHING MORE than a linear contraction from a random start?

Reads the Arnoldi spectra in `scratch/ds_eigen/out/eigen.json` and the banked
trajectories in `scratch/ds_bank/out/`, which share prompts, and writes
`results/surrogate.csv`. No GPU. OWNER: Data+Analysis. STATUS: implemented 2026-08-09.

WHY A SUFFICIENCY TEST AND NOT ANOTHER NULL. D79 found no geometric difference
between correct and incorrect trajectories, and the geomcap run asks the same across
tasks. Nulls accumulate but do not explain. This asks the constructive question: a
surrogate built from the measured eigenvalues and a RANDOM start knows no prompt, no
task, no answer and no fixed point. If it reproduces the real orbit's shape
statistics, then the real orbit's shape demonstrably carries nothing beyond the
spectrum -- and the spectrum is a property of the weights.

THE COMPARISON IS THREE-WAY, because two of the three readings are easy to confuse.
  own       -- surrogate from THIS prompt's spectrum.
  swapped   -- surrogate from ANOTHER prompt's spectrum. If `own` and `swapped` land
               in the same place, even the spectrum is prompt-independent and the
               geometry is purely a weight signature.
  isotropic -- unit directions with no dynamics at all, the null D74 already uses.
               It is what "the surrogate matches" must be measured AGAINST: a
               statistic both surrogates and the real orbit share with white noise
               is not evidence for the linear picture.

WHAT WOULD REFUTE THE PICTURE. A real orbit outside the surrogate's initial-condition
band on a statistic where `own` and `isotropic` differ. That would mean the shape
carries structure the spectrum does not predict, and the founding hypothesis would
have a place to live.

THE LIMITS ARE STRUCTURAL AND ARE REPORTED WITH THE RESULT. Arnoldi returned 8
eigenvalues of a 5280-dimensional operator; `spectrum_tail` measures the sensitivity
to that cut. The surrogate uses an ORTHONORMAL frame, so it is the NORMAL
approximation -- a mismatch is evidence of non-normality (which Huginn's operator
certainly has) rather than of nonlinearity. And there are 7 usable prompts, from one
run, at one token position.

Run: uv run python -m scripts.run_surrogate
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from traj_geom.metrics.dimension import participation_ratio, step_directions
from traj_geom.metrics.dmd import pre_floor_window
from traj_geom.metrics.linear_surrogate import spectrum_tail, surrogate_statistics

EIGEN = os.path.join("scratch", "ds_eigen", "out", "eigen.json")
BANK = os.path.join("scratch", "ds_bank", "out")
OUT = os.path.join("results", "surrogate.csv")
N_DRAW = 64
N_EXTRA = 24          # the truncation control: 8 measured modes + 24 continued


def real_statistics(traj: np.ndarray, lo: int, hi: int) -> dict:
    u = step_directions(traj, lo=lo, hi=hi)
    if len(u) < 4:
        return {"ok": False}
    d = np.linalg.norm(np.diff(traj, axis=0), axis=1)[lo:hi]
    y = np.log(np.clip(d, 1e-30, None))
    rho = float(np.exp(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]))
    return {"ok": True, "n_steps": int(len(u)),
            "cos_consecutive": float(np.mean(np.sum(u[:-1] * u[1:], axis=1))),
            "pr": participation_ratio(u), "contraction": rho}


def isotropic_reference(n_steps: int, dim: int, n_draw: int = 32,
                        seed: int = 0) -> dict:
    """Unit directions with no dynamics -- the null a match must beat to mean anything.

    Simulated at the SAME sample count as the data, because with m samples in d
    dimensions isotropic directions have participation ratio ~ m rather than ~ d
    (D74). Comparing against 5280 would make every orbit look dramatically
    structured and would mean nothing.
    """
    rng = np.random.default_rng(seed)
    prs, coss = [], []
    for _ in range(n_draw):
        u = rng.normal(size=(n_steps, dim))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        prs.append(participation_ratio(u))
        coss.append(float(np.mean(np.sum(u[:-1] * u[1:], axis=1))))
    return {"pr": float(np.median(prs)), "cos_consecutive": float(np.median(coss))}


def load_pairs(eigen: str = EIGEN, bank: str = BANK) -> list[dict]:
    """Prompts for which BOTH a spectrum and a banked orbit exist."""
    with open(eigen, encoding="utf-8") as fh:
        recs = [r for r in json.load(fh)["records"] if r.get("ok")]
    with open(os.path.join(bank, "manifest.json"), encoding="utf-8") as fh:
        man = {r["tag"]: r for r in json.load(fh) if r.get("ok")}
    out = []
    for r in recs:
        tag = f"trained_{r['kind']}_n{r['n_ops']}_ns128_seed{r['seed']}"
        path = os.path.join(bank, tag + "_last.npy")
        if tag in man and os.path.exists(path):
            out.append({"tag": tag, "rec": r, "path": path})
    return out


def analyse(pairs: list[dict], n_draw: int = N_DRAW) -> pd.DataFrame:
    spectra = [np.array([complex(a, b) for a, b in p["rec"]["eigs"]]) for p in pairs]
    rows = []
    for i, p in enumerate(pairs):
        traj = np.load(p["path"]).astype(np.float64)
        lo, hi = pre_floor_window(traj)
        real = real_statistics(traj, lo, hi)
        if not real["ok"]:
            continue
        n_s, dim = real["n_steps"], traj.shape[1]
        # The surrogate is generated over the SAME number of usable steps as the
        # real orbit. Participation ratio grows with sample count, so a surrogate
        # run to a different length would be compared on a different quantity --
        # D74(6) records that confound as total (rank-collinear at rho = -1.000).
        own = surrogate_statistics(spectra[i], n_s + 2, dim, n_draw=n_draw, seed=17)
        other = spectra[(i + 1) % len(spectra)]
        swap = surrogate_statistics(other, n_s + 2, dim, n_draw=n_draw, seed=29)
        wide = surrogate_statistics(spectrum_tail(spectra[i], N_EXTRA, seed=i),
                                    n_s + 2, dim, n_draw=n_draw, seed=41)
        iso = isotropic_reference(n_s, dim, n_draw=32, seed=5)
        row = {"tag": p["tag"], "kind": p["rec"]["kind"], "n_ops": p["rec"]["n_ops"],
               "n_tokens": p["rec"]["n_tokens"], "n_steps": n_s,
               "n_eigs": len(spectra[i]), "rho_arnoldi": p["rec"]["rho"],
               "arg_lead": p["rec"]["arg"],
               "rate_over_arg": p["rec"].get("rate_over_arg", float("nan"))}
        for name, s in (("own", own), ("swap", swap), ("wide", wide)):
            for k in ("cos_consecutive", "pr", "contraction"):
                row[f"{name}_{k}"] = s.get(k, float("nan"))
            row[f"{name}_cos_lo"] = s.get("cos_lo", float("nan"))
            row[f"{name}_cos_hi"] = s.get("cos_hi", float("nan"))
            row[f"{name}_pr_lo"] = s.get("pr_lo", float("nan"))
            row[f"{name}_pr_hi"] = s.get("pr_hi", float("nan"))
        for k in ("cos_consecutive", "pr", "contraction"):
            row[f"real_{k}"] = real[k]
        row["iso_pr"] = iso["pr"]
        row["iso_cos"] = iso["cos_consecutive"]
        row["cos_in_band"] = bool(own["cos_lo"] <= real["cos_consecutive"]
                                  <= own["cos_hi"])
        row["pr_in_band"] = bool(own["pr_lo"] <= real["pr"] <= own["pr_hi"])
        # A statistic the surrogate "predicts" only because white noise predicts it
        # too is no evidence at all. `discriminating` marks the cells where the
        # linear picture and the isotropic null actually disagree.
        row["cos_discriminating"] = bool(
            abs(own["cos_consecutive"] - iso["cos_consecutive"]) > 0.1)
        row["pr_discriminating"] = bool(abs(own["pr"] - iso["pr"]) > 0.1 * iso["pr"])
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    lines = [
        "",
        "  Each row: the real orbit against a linear surrogate that knows only this",
        "  prompt's eigenvalues and a random start -- no prompt, no task, no answer.",
        "",
        f"  {'prompt':>22} {'steps':>5} | {'cos: real':>9} {'own':>17} "
        f"{'swap':>7} {'iso':>7} | {'PR: real':>8} {'own':>15} {'iso':>6}",
    ]
    for _, r in df.iterrows():
        name = f"{r['kind']}_n{r['n_ops']}"
        lines.append(
            f"  {name:>22} {r['n_steps']:>5} | {r['real_cos_consecutive']:>+9.3f} "
            f"{r['own_cos_consecutive']:>+7.3f}[{r['own_cos_lo']:+.2f},{r['own_cos_hi']:+.2f}] "
            f"{r['swap_cos_consecutive']:>+7.3f} {r['iso_cos']:>+7.3f} | "
            f"{r['real_pr']:>8.2f} {r['own_pr']:>6.2f}[{r['own_pr_lo']:.1f},{r['own_pr_hi']:.1f}] "
            f"{r['iso_pr']:>6.2f}")

    n = len(df)
    disc_cos = df[df["cos_discriminating"]]
    disc_pr = df[df["pr_discriminating"]]
    lines += ["", "  WHERE THE LINEAR PICTURE IS ACTUALLY BEING TESTED"]
    lines.append(f"    cos: {len(disc_cos)}/{n} prompts where the surrogate and the "
                 f"isotropic null differ by > 0.1; of those, "
                 f"{int(disc_cos['cos_in_band'].sum())} have the real orbit inside "
                 f"the surrogate's band")
    lines.append(f"    PR:  {len(disc_pr)}/{n} discriminating; of those, "
                 f"{int(disc_pr['pr_in_band'].sum())} inside")
    lines += ["", "  DOES THE PROMPT'S OWN SPECTRUM MATTER?"]
    d = (df["own_cos_consecutive"] - df["swap_cos_consecutive"]).abs()
    lines.append(f"    |own - swapped| on cos: median {d.median():.4f}, max {d.max():.4f}")
    lines.append("    -- a small number means the spectrum barely changes between")
    lines.append("       prompts, i.e. the geometry is a weight signature, not a")
    lines.append("       prompt signature.")
    lines += ["", "  TRUNCATION SENSITIVITY (8 measured modes vs 8 + 24 continued)"]
    d_cos = np.nanmedian((df["wide_cos_consecutive"] - df["own_cos_consecutive"]).abs())
    d_pr = np.nanmedian((df["wide_pr"] - df["own_pr"]).abs())
    lines.append(f"    cos moves {d_cos:.4f} (median), PR moves {d_pr:.2f}")
    lines.append("    -- PR is the statistic the cut bites; cos is not, because a")
    lines.append("       subdominant mode contributes little to consecutive steps.")
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not (os.path.exists(EIGEN) and os.path.exists(os.path.join(BANK, "manifest.json"))):
        print(f"need both {EIGEN} and {BANK}/manifest.json")
        return
    pairs = load_pairs()
    if not pairs:
        print("no prompt has BOTH a spectrum and a banked orbit")
        return
    df = analyse(pairs)
    save_table(OUT, df, kind="surrogate", eigen=EIGEN, bank=BANK, n_draw=N_DRAW)
    print(f"saved {OUT} ({len(df)} prompts with both a spectrum and an orbit)")
    print(report(df))


if __name__ == "__main__":
    main()
