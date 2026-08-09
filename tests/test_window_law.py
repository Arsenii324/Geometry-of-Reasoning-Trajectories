"""Validation of the sliding-window depth analysis on constructions with known answers.

The claim the script makes -- "the shape statistics track depth, not the input" --
would be produced spuriously by three different bugs, and each is planted here:

  * a widening window instead of a sliding one (participation ratio grows with
    sample count, so the "depth effect" would be the sample count; D74(6) records
    that confound as total, rho = -1.000),
  * a thinning orbit set along the depth axis (the banks run to 64 and 128 unrolls,
    so a pooled profile changes composition as it advances),
  * a sign test that counts the wrong tail.

Nothing here reads the real banks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.run_window_law import (
    arm_contrast,
    collect,
    common_depth_range,
    depth_vs_prompt,
    family_profiles,
    paired_depth_change,
    report,
    sliding,
)

DIM = 60


def _contracting(rho=0.87, n_dir=12, n_step=80, seed=0, scale=70.0):
    """An orbit that collapses onto its dominant mode, like the real ones.

    Subdominant modes decay FASTER, so participation ratio must fall and consecutive
    steps must become parallel -- which is the pattern the script is built to detect.
    """
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(DIM, 2 * n_dir)))[0]
    t = np.arange(n_step, dtype=float)
    out = np.zeros((n_step, DIM))
    for i in range(n_dir):
        r = rho * (0.90**i)                       # mode 0 dominates asymptotically
        phi = 0.3 + 0.15 * i
        a = rng.normal() * scale
        out += np.outer(a * r**t * np.cos(phi * t), basis[:, 2 * i])
        out += np.outer(a * r**t * np.sin(phi * t), basis[:, 2 * i + 1])
    return out.astype(np.float32)


def _isotropic(n_step=80, seed=0, scale=70.0):
    """No dynamics at all: the negative control for the depth law."""
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n_step, DIM)) * scale).astype(np.float32)


def _bank(tmp_path, name, orbits, suffix=".npy"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for tag, fam, arr in orbits:
        np.save(d / (tag + suffix), arr)
        recs.append({"tag": tag, "family": fam, "correct": True, "n_tokens": 30,
                     "ok": True})
    with open(d / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh)
    return str(d)


def test_window_width_is_constant_as_it_slides() -> None:
    """THE BUG THAT WOULD MANUFACTURE THE RESULT.

    Participation ratio grows with the number of samples. If the window widened as
    it advanced, "PR falls with depth" could not even be stated -- the two effects
    push opposite ways and neither would be identified.
    """
    w = sliding(_contracting(), width=12, stride=4)
    assert len({r["start"] for r in w}) == len(w), "starts must be distinct"
    starts = sorted(r["start"] for r in w)
    assert starts == list(range(starts[0], starts[0] + 4 * len(starts), 4))
    assert {r["n_dirs"] for r in w} == {12}, (
        f"window width drifted as it slid: {sorted({r['n_dirs'] for r in w})}")


def test_a_contracting_orbit_sheds_dimensions_and_straightens() -> None:
    w = sliding(_contracting(seed=3), width=12, stride=4)
    assert len(w) >= 5
    assert w[-1]["pr"] < w[0]["pr"], [r["pr"] for r in w]
    assert w[-1]["cos_consecutive"] > w[0]["cos_consecutive"]


def test_an_orbit_with_no_dynamics_shows_no_depth_law(tmp_path) -> None:
    """NON-SUPPRESSION. If white noise also produced a depth law, the statistic
    would be measuring the sliding window rather than the model."""
    path = _bank(tmp_path, "iso", [(f"t{i:02d}", "f", _isotropic(seed=i))
                                   for i in range(20)])
    df = collect((("iso", path, ".npy"),))
    if df.empty:
        pytest.skip("isotropic orbits have no pre-floor window")
    c = paired_depth_change(df, "pr")
    if c.get("usable"):
        assert c["p_sign"] > 0.01, f"a depth law appeared in pure noise: {c}"


def test_paired_change_survives_a_thinning_bank(tmp_path) -> None:
    """THE COMPOSITION CONFOUND, planted.

    One bank runs to 90 unrolls and the other to 45, the same 2:1 split as ds_bank
    (128) against b6bank and geomcap (64). A pooled profile would change which
    orbits it describes as it advances; the paired statistic must be computed only
    over the range every orbit reaches, and must still find the effect.
    """
    deep = [(f"d{i:02d}", "deep", _contracting(n_step=90, seed=i)) for i in range(10)]
    shallow = [(f"s{i:02d}", "shallow", _contracting(n_step=45, seed=50 + i))
               for i in range(10)]
    p1 = _bank(tmp_path, "deep", deep)
    p2 = _bank(tmp_path, "shallow", shallow)
    df = collect((("deep", p1, ".npy"), ("shallow", p2, ".npy")))
    depths = common_depth_range(df)
    assert max(depths) < 40, "the common range must stop where the shallow bank does"
    c = paired_depth_change(df, "pr", depths)
    assert c["usable"] and c["n_orbits"] == 20, c
    assert c["direction"] == "down" and c["p_sign"] < 0.01, c


def test_the_sign_test_counts_the_majority_tail() -> None:
    """A unanimous DECREASE must give a tiny p, not a p of 1.

    `n_increased` is 0 there, and a test that read that number as the statistic
    would report the strongest possible result as the weakest.
    """
    import pandas as pd
    rows = [{"tag": f"t{i}", "start": s, "pr": 10.0 - s * 0.1, "bank": "b",
             "family": "f", "correct": True, "n_tokens": 1}
            for i in range(16) for s in (0, 20)]
    c = paired_depth_change(pd.DataFrame(rows), "pr", [0, 20])
    assert c["n_increased"] == 0
    assert c["n_agreeing"] == 16 and c["direction"] == "down"
    assert c["p_sign"] < 1e-4, c


def test_family_profiles_separate_shape_from_offset(tmp_path) -> None:
    """Same depth law, different offset, must read as ONE shape.

    That distinction carries the interpretation: a shared profile with per-family
    offsets says every family runs the same contraction, while genuinely different
    profiles would be the first sign that the geometry encodes something
    task-specific.
    """
    a = [(f"a{i:02d}", "A", _contracting(seed=i)) for i in range(8)]
    b = [(f"b{i:02d}", "B", _contracting(seed=100 + i, n_dir=12)) for i in range(8)]
    df = collect((("bank", _bank(tmp_path, "b", a + b), ".npy"),))
    piv = family_profiles(df, "pr")
    assert set(piv.columns) == {"A", "B"}
    assert piv["A"].corr(piv["B"]) > 0.9, "same construction gave different shapes"


def test_report_runs_and_names_the_paired_block(tmp_path) -> None:
    orbits = [(f"t{i:02d}", "f", _contracting(seed=i)) for i in range(12)]
    df = collect((("bank", _bank(tmp_path, "b", orbits), ".npy"),))
    txt = report(df)
    assert "PAIRED WITHIN EACH ORBIT" in txt
    assert "sign test p" in txt


def test_depth_vs_prompt_is_reported_as_pooled_only(tmp_path) -> None:
    """The pooled comparison is kept, but it must not be the headline."""
    orbits = [(f"t{i:02d}", "f", _contracting(seed=i)) for i in range(12)]
    df = collect((("bank", _bank(tmp_path, "b", orbits), ".npy"),))
    c = depth_vs_prompt(df, "pr")
    assert c["usable"] and c["ratio"] > 0
    assert "composition changes with depth" in report(df)


def _diffusing(n_step=80, seed=0, scale=70.0, rho=0.71):
    """A contraction with a FLAT spectrum: it shrinks without collapsing.

    Every mode at the same modulus, so none comes to dominate. This is the
    untrained arm's behaviour as D76(3) describes it -- effective dimension
    tracking the sample count rather than saturating -- and it is the construction
    that makes the trained/untrained contrast falsifiable: if the pipeline reported
    a depth law here too, D80's arm contrast would be measuring the sliding window.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n_step, DIM))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    steps *= (scale * rho ** np.arange(n_step))[:, None]
    return np.cumsum(steps, axis=0).astype(np.float32)


def test_the_arm_contrast_separates_collapse_from_diffusion(tmp_path) -> None:
    """THE CLAIM D80(7) RESTS ON, planted both ways.

    Trained orbits are built to collapse onto a dominant mode; untrained ones are
    built to shrink with a flat spectrum. The contrast must report a large paired
    change for the first and a small one for the second, at MATCHED depths -- the
    arms converge at different rates (D76(5): 104 unrolls against 39-43), so an
    unmatched comparison would contrast two different stretches.
    """
    import json

    out = tmp_path / "b"
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for i in range(12):
        np.save(out / f"tr{i:02d}.npy", _contracting(seed=i))
        recs.append({"tag": f"tr{i:02d}", "family": "f", "arm": "trained",
                     "correct": True, "n_tokens": 30, "ok": True})
    for i in range(12):
        np.save(out / f"un{i:02d}.npy", _diffusing(seed=100 + i))
        recs.append({"tag": f"un{i:02d}", "family": "f", "arm": "untrained",
                     "correct": False, "n_tokens": 30, "ok": True})
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh)

    df = collect((("bank", str(out), ".npy"),), arms=("trained", "untrained"))
    assert set(df["arm"]) == {"trained", "untrained"}
    txt = arm_contrast(df)
    assert "TRAINED AGAINST UNTRAINED" in txt

    shared = sorted(set(common_depth_range(df[df["arm"] == "trained"]))
                    & set(common_depth_range(df[df["arm"] == "untrained"])))
    tr = paired_depth_change(df[df["arm"] == "trained"], "pr", shared)
    un = paired_depth_change(df[df["arm"] == "untrained"], "pr", shared)
    assert abs(tr["median_delta"]) > 3 * abs(un["median_delta"]), (tr, un)


def test_collect_returns_only_the_arms_asked_for(tmp_path) -> None:
    """The default is trained-only, because every earlier analysis is trained-only
    and silently widening it would change published numbers."""
    import json

    out = tmp_path / "c"
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for arm, pre in (("trained", "tr"), ("untrained", "un")):
        for i in range(10):
            np.save(out / f"{pre}{i:02d}.npy", _contracting(seed=i + len(pre)))
            recs.append({"tag": f"{pre}{i:02d}", "family": "f", "arm": arm,
                         "correct": True, "n_tokens": 30, "ok": True})
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh)
    src = (("bank", str(out), ".npy"),)
    assert set(collect(src)["arm"]) == {"trained"}
    assert set(collect(src, arms=("trained", "untrained"))["arm"]) == {
        "trained", "untrained"}


def test_untrained_draws_are_kept_distinguishable(tmp_path) -> None:
    """`ds_seeds` labels its arms untrained0..untrained4, one per weight draw.

    Collapsing them to "untrained" is right for the contrast, but the draw has to
    survive in `arm_raw` -- otherwise the claim "n=6 independent weight-sets" could
    not be checked from the table, and D76(6) is the record of what one draw costs.
    """
    import json

    out = tmp_path / "d"
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for k in range(3):
        for i in range(6):
            np.save(out / f"u{k}_{i:02d}.npy", _diffusing(seed=k * 10 + i))
            recs.append({"tag": f"u{k}_{i:02d}", "family": "f",
                         "arm": f"untrained{k}", "correct": False,
                         "n_tokens": 30, "ok": True})
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh)
    df = collect((("bank", str(out), ".npy"),), arms=("untrained",))
    assert set(df["arm"]) == {"untrained"}
    assert df["arm_raw"].nunique() == 3
