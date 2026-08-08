"""How large could a correlation have been? Variance components and their ceiling.

OWNER: Data+Analysis
STATUS: implemented 2026-08-09.

WHY IT EXISTS. This project keeps producing nulls -- D40, D41, D48, D53, D72, D79 --
and a null is only a result if something bounds it. Twice now the bound turned out
to be the binding constraint and not the biology: D70 reported a content gap of ~0
from a probe pinned at R2 >= 0.87 in BOTH arms, where the gap could not have exceeded
0.129 whatever the truth; D72's design admitted 300 label arrangements, so no data
could have reached alpha. `stratified.combinatorial_floor` bounds a permutation test
by its DESIGN. This module bounds a CORRELATION by its MEASUREMENT NOISE.

THE DECOMPOSITION THE GEOMCAP RUN IDENTIFIES. Huginn's initial latent h_0 is drawn
from an unseeded generator (D78), so the same prompt run twice differs in h_0 and in
nothing else. Writing a per-orbit statistic as

    y_ijk = mu + F_i + I_ij + E_ijk        family i, item j, replicate k

the run's two blocks separate all three terms: the `rep` block (one item, many
replicates) gives var(E) directly, and the `main` block (many items, one replicate
each) gives var(I) + var(E) within a family, hence var(I) by subtraction. Nothing
here is assumed; every component comes from a measured contrast.

WHAT THE CEILING MEANS. A family mean over m items has reliability

    rel = var(F) / (var(F) + (var(I) + var(E)) / m)

and a correlation between two imperfectly measured variables is attenuated by the
square root of the product of their reliabilities (Spearman 1904). So an observed
|rho| must be read against `sqrt(rel)`, not against 1: if capability is measured
almost perfectly and the geometry's family means have reliability 0.25, then even a
PERFECT underlying relation could only show |rho| ~ 0.5, and an observed 0.2 is not
a null about geometry -- it is a null about the measurement. The correction is
standard psychometrics applied to the thing this project actually needs bounded.
"""

from __future__ import annotations

import numpy as np


def icc_one_way(values: np.ndarray, groups: np.ndarray) -> dict:
    """One-way random-effects variance components, tolerating unequal group sizes.

    Args:
        values: one measurement per row.
        groups: group key per row; variance is split between and within these.

    Returns:
        ``sigma2_between``, ``sigma2_within``, ``icc``, ``n_groups``, ``n`` and
        ``n0`` (the effective group size the ANOVA estimator uses). ``usable`` is
        False when fewer than two groups carry replicates, in which case the
        within-group term is not identified and no ceiling may be quoted.

    ``sigma2_between`` is clipped at zero: the ANOVA estimator ``(MSB - MSW) / n0``
    can go negative by sampling noise, and a negative variance reported as such
    would propagate into a nonsensical reliability rather than the honest "the
    between-group term is not distinguishable from zero".
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    keys = np.unique(groups)
    sizes = np.array([int((groups == g).sum()) for g in keys], dtype=np.float64)
    n, k = float(len(values)), float(len(keys))
    base = {"n_groups": int(k), "n": int(n)}
    if k < 2 or n <= k or not np.isfinite(values).all():
        return {**base, "usable": False, "why": "needs >=2 groups and >=1 replicate",
                "sigma2_between": float("nan"), "sigma2_within": float("nan"),
                "icc": float("nan"), "n0": float("nan")}
    grand = float(values.mean())
    means = np.array([values[groups == g].mean() for g in keys])
    msb = float((sizes * (means - grand) ** 2).sum() / (k - 1))
    ssw = float(sum(((values[groups == g] - m) ** 2).sum()
                    for g, m in zip(keys, means, strict=True)))
    msw = ssw / (n - k)
    n0 = float((n - (sizes**2).sum() / n) / (k - 1))
    s2b = max(0.0, (msb - msw) / n0) if n0 > 0 else 0.0
    total = s2b + msw
    return {**base, "usable": True, "sigma2_between": s2b, "sigma2_within": msw,
            "icc": (s2b / total) if total > 0 else float("nan"), "n0": n0}


def family_mean_reliability(sigma2_family: float, sigma2_noise: float,
                            m: int) -> float:
    """Reliability of a family MEAN over ``m`` items, given the per-orbit noise.

    ``sigma2_noise`` is everything that varies within a family -- item choice and
    h_0 together -- because both are noise for a question asked at the FAMILY level.
    Averaging m items divides it by m; the family term does not shrink. This is the
    Spearman-Brown step-up written in variance terms.
    """
    if m < 1 or not np.isfinite([sigma2_family, sigma2_noise]).all():
        return float("nan")
    denom = sigma2_family + sigma2_noise / m
    return float(sigma2_family / denom) if denom > 0 else float("nan")


def attenuation_ceiling(rel_x: float, rel_y: float = 1.0) -> float:
    """EXPECTED |correlation| when the underlying relation is perfect.

    Spearman's 1904 attenuation formula, rho_obs = rho_true * sqrt(rel_x * rel_y),
    at rho_true = 1. Quoting an observed rho without this is how a measurement null
    gets reported as a substantive one.

    IT IS AN EXPECTATION, NOT A HARD BOUND, and the distinction is not pedantic: at
    reliability 0.5 and N = 21 this returns 0.707, yet a perfect relation produces
    an observed |rho| above 0.757 in about a third of draws -- measured, in
    `tests/test_reliability.py`, where calling it a bound failed 15 times in 40.
    Use `perfect_relation_band` whenever the comparison needs to survive sampling
    noise; use this number as the centre it is.
    """
    if not np.isfinite([rel_x, rel_y]).all():
        return float("nan")
    return float(np.sqrt(max(0.0, rel_x) * max(0.0, rel_y)))


def perfect_relation_band(rel: float, n: int, z: float = 1.96,
                          rel_y: float = 1.0) -> tuple[float, float]:
    """Central band of |rho| a PERFECT underlying relation would produce at N = n.

    Fisher-z around `attenuation_ceiling(rel, rel_y)` with se = 1/sqrt(n - 3). This
    is the comparison an observed correlation actually needs: "the design could
    have shown 0.61 to 0.85, and it showed 0.08" is a bounded null, whereas "the
    ceiling is 0.71" invites the reader to treat 0.75 as impossible when it is
    ordinary.

    Returns ``(lo, hi)``, or ``(nan, nan)`` when n <= 3 leaves the z-transform
    undefined -- at which point no correlation from that design is interpretable
    anyway.
    """
    c = attenuation_ceiling(rel, rel_y)
    if not np.isfinite(c) or n <= 3:
        return float("nan"), float("nan")
    c = min(c, 1 - 1e-12)
    se = 1.0 / np.sqrt(n - 3)
    zc = np.arctanh(c)
    return float(np.tanh(zc - z * se)), float(np.tanh(zc + z * se))


def _bootstrap_reliability(values: np.ndarray, families: np.ndarray, m: int,
                           n_boot: int, seed: int = 0) -> tuple[float, float]:
    """95% interval for the reliability, resampling FAMILIES with replacement.

    Families are the unit the correlation is computed over, so they are the unit
    that must be resampled; resampling items would hold the family set fixed and
    understate exactly the term that is noisiest.
    """
    if n_boot < 2:
        return float("nan"), float("nan")
    values = np.asarray(values, dtype=np.float64)
    families = np.asarray(families)
    keys = np.unique(families)
    by_key = {g: values[families == g] for g in keys}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        vals = np.concatenate([by_key[keys[j]] for j in pick])
        labs = np.concatenate([np.full(len(by_key[keys[j]]), str(i))
                               for i, j in enumerate(pick)])
        a = icc_one_way(vals, labs)
        if a["usable"]:
            out.append(family_mean_reliability(a["sigma2_between"],
                                               a["sigma2_within"], m))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def decompose(main_values: np.ndarray, main_families: np.ndarray,
              rep_values: np.ndarray | None = None,
              rep_prompts: np.ndarray | None = None,
              m: int | None = None, n_levels: int | None = None,
              n_boot: int = 400) -> dict:
    """Full family/item/h_0 split for one statistic, plus the correlation ceiling.

    Args:
        main_values, main_families: one orbit per item, family key per orbit. This
            contrast identifies var(family) against var(item) + var(h_0) pooled.
        rep_values, rep_prompts: replicates of single prompts, keyed by prompt. This
            contrast identifies var(h_0) ALONE, because prompt and weights are held
            fixed and D78 leaves h_0 as the only thing that can differ. Optional:
            without it the pooled within-family term is still a valid ceiling, it
            just cannot be attributed.
        m: items per family used to form the means being correlated; defaults to the
            median family size in ``main``.

    Returns ``sigma2_family``, ``sigma2_within_family``, ``sigma2_h0``,
    ``sigma2_item`` (by subtraction, clipped at zero), ``reliability``, ``ceiling``
    and -- when ``n_levels`` is given -- ``band``, the interval a perfect relation
    would land in at that many levels.
    """
    a = icc_one_way(main_values, main_families)
    if not a["usable"]:
        return {"usable": False, "why": a.get("why", "main block not decomposable")}
    if m is None:
        fams, counts = np.unique(np.asarray(main_families), return_counts=True)
        m = int(np.median(counts)) if len(fams) else 1
    s2_within = a["sigma2_within"]
    s2_h0 = float("nan")
    if rep_values is not None and rep_prompts is not None and len(rep_values):
        b = icc_one_way(rep_values, rep_prompts)
        # The WITHIN term of the replicate block is h_0 and nothing else: same
        # prompt, same weights, same depth, same read position (D78).
        s2_h0 = b["sigma2_within"] if b["usable"] else float("nan")
    rel = family_mean_reliability(a["sigma2_between"], s2_within, m)
    nl = n_levels if n_levels else a["n_groups"]
    band = perfect_relation_band(rel, nl)
    rel_lo, rel_hi = _bootstrap_reliability(main_values, main_families, m, n_boot)
    # The band must carry BOTH uncertainties. `perfect_relation_band` covers the
    # correlation's own sampling noise; the reliability that centres it is itself
    # estimated from ~21 families, where the between-group variance has a relative
    # standard error of sqrt(2/20) = 0.32 -- measured in tests, where one draw gave
    # a ceiling of 0.45 and the average of 60 gave 0.68 for the same truth.
    wide = (min(band[0], perfect_relation_band(rel_lo, nl)[0]),
            max(band[1], perfect_relation_band(rel_hi, nl)[1]))
    return {"usable": True, "m": int(m), "band_lo": band[0], "band_hi": band[1],
            "rel_lo": rel_lo, "rel_hi": rel_hi,
            "wide_lo": wide[0], "wide_hi": wide[1], "n_levels": int(nl),
            "sigma2_family": a["sigma2_between"],
            "sigma2_within_family": s2_within,
            "sigma2_h0": s2_h0,
            "sigma2_item": (max(0.0, s2_within - s2_h0)
                            if np.isfinite(s2_h0) else float("nan")),
            "h0_share": (s2_h0 / s2_within
                         if np.isfinite(s2_h0) and s2_within > 0 else float("nan")),
            "reliability": rel, "ceiling": attenuation_ceiling(rel)}
