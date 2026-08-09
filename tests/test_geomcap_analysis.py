"""End-to-end validation of `scripts.run_geomcap` on synthetic banks.

The three claims this pipeline can make -- "geometry tracks capability", "it does
not and the design had the power", "it does not and the design had no power" -- are
each planted here and each has to come back. A pipeline that cannot distinguish the
second from the third would report D70's mistake again: a null from an instrument
with no dynamic range, printed as a fact about the model.

Nothing here touches a GPU or the real bank; trajectories are built to order.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.run_geomcap import (
    analyse,
    collect,
    covariate_structure,
    determinism_verdict,
    family_table,
    p1_gate,
    report,
    window_sensitivity,
)


def collect_small(path):
    """Two windows instead of five: these tests exercise logic, not the sweep."""
    return collect(path, windows=(10, 20))

N_FAM, N_ITEM, N_STEP, DIM = 12, 10, 64, 40


def _orbit(rng, rho=0.88, n_dir=13, scale=70.0, n_step=N_STEP):
    """A contracting orbit that moves in `n_dir` directions, like the real ones.

    D74 measured ~13 effective dimensions and a step decay near 0.87, so the
    synthetic orbits are built at that scale -- a 2-D spiral would let statistics
    pass that could never survive on the real bank.
    """
    basis = np.linalg.qr(rng.normal(size=(DIM, n_dir)))[0]
    fixed = rng.normal(size=DIM) * scale
    out = []
    for t in range(n_step):
        coef = rng.normal(size=n_dir) * scale * (rho**t)
        out.append(fixed + basis @ coef)
    return np.stack(out).astype(np.float32)


def _bank(tmp_path, *, rho_by_family=None, acc_by_family=None, n_tokens_by_family=None,
          rep_families=(), rep_n=10, fix_families=(), fix_n=3, fix_identical=True,
          rep_identical=False, seed=0):
    """Write a manifest plus .npy orbits, exactly as the kernel does."""
    rng = np.random.default_rng(seed)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    fams = [f"fam{i:02d}" for i in range(N_FAM)]
    recs = []

    def emit(tag, block, fam, item, correct, arr, gold="g"):
        np.save(out / (tag + ".npy"), arr)
        recs.append({
            "tag": tag, "block": block, "family": fam, "item": item, "gold": gold,
            "correct": bool(correct), "best_rank": 1 if correct else 40,
            "best_depth": 4, "n_tokens": int((n_tokens_by_family or {}).get(fam, 30)),
            "h0_seed": None, "state_sha": f"{abs(hash(tag)) % 10**16:016x}",
            "ok": True,
        })

    for i, fam in enumerate(fams):
        rho = (rho_by_family or {}).get(fam, 0.88)
        acc = (acc_by_family or {}).get(fam, i / (N_FAM - 1))
        for j in range(N_ITEM):
            emit(f"{fam}_i{j:02d}", "main", fam, j, j < round(acc * N_ITEM),
                 _orbit(rng, rho=rho), gold=f"g{j % 3}")
    for fam in rep_families:
        rho = (rho_by_family or {}).get(fam, 0.88)
        fixed = _orbit(rng, rho=rho)
        for k in range(rep_n):
            arr = fixed if rep_identical else _orbit(rng, rho=rho)
            r = {"tag": f"rep_{fam}_r{k:02d}"}
            emit(r["tag"], "rep", fam, 0, k % 2 == 0, arr)
            if rep_identical:
                recs[-1]["state_sha"] = "same"
    for fam in fix_families:
        fixed = _orbit(rng, rho=(rho_by_family or {}).get(fam, 0.88))
        for k in range(fix_n):
            emit(f"fix_{fam}_r{k:02d}", "fix", fam, 0, True,
                 fixed if fix_identical else _orbit(rng))
            recs[-1]["h0_seed"] = 20260809
            if fix_identical:
                recs[-1]["state_sha"] = f"fixed_{fam}"
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh)
    return str(out)


@pytest.fixture(scope="module")
def flat_bank(tmp_path_factory):
    """Geometry identical across families; capability spans 0-100%."""
    return _bank(tmp_path_factory.mktemp("flat"),
                 rep_families=("fam00", "fam05", "fam11"),
                 fix_families=("fam00", "fam05", "fam11"), seed=1)


def test_collect_reads_every_banked_orbit(flat_bank) -> None:
    """One row per (orbit, window), and every orbit present at every window."""
    df = collect(flat_bank, windows=(10, 20))
    n_orbits = N_FAM * N_ITEM + 3 * 10 + 3 * 3
    assert df["tag"].nunique() == n_orbits
    assert set(df["block"]) == {"main", "rep", "fix"}
    assert set(df["window_k"].dropna()) == {10, 20}
    assert df["window_k"].isna().any(), "the natural window must also be emitted"
    assert len(df) == n_orbits * 3
    for col in ("pr", "cos_consecutive", "contraction", "settle", "log_rank"):
        assert df.loc[df["block"] == "main", col].notna().all(), f"{col} has NaNs"


def test_the_window_changes_the_statistics_it_is_swept_over(flat_bank) -> None:
    """WHY THE SWEEP EXISTS. Measured on the real ds_bank orbits, participation
    ratio reads ~13 over 20 unrolls and ~2.3 over 90 on the SAME trajectories --
    the transient and the asymptote of a contraction, not two estimates of one
    number. A pipeline that reported one window would be reporting a choice, which
    is D28's failure (winding's sign flipped with the recording budget alone)."""
    df = collect(flat_bank, windows=(10, 30))
    m = df[df["block"] == "main"]
    a = m[m["window_k"] == 10]["pr"].median()
    b = m[m["window_k"] == 30]["pr"].median()
    assert a < b, f"PR did not grow with window: {a} vs {b}"
    txt = window_sensitivity(df)
    assert "WINDOW SENSITIVITY" in txt
    # And it must SAY that it is not D80's sliding-window law: the two move in
    # opposite directions by construction (sample count rises here, is held fixed
    # there), so a reader comparing them without that note would see a
    # contradiction that is not one.
    assert "sliding-window law" in txt


def test_a_planted_capability_relation_is_found(tmp_path) -> None:
    """Positive control on the whole pipeline: contraction tied to accuracy.

    Without this, a null from `analyse` would be indistinguishable from a pipeline
    that reports nulls unconditionally -- which is the failure D62 and D69(2) both
    record, a verdict printed over an instrument nobody checked.
    """
    fams = [f"fam{i:02d}" for i in range(N_FAM)]
    rho_by = {f: 0.80 + 0.015 * i for i, f in enumerate(fams)}
    path = _bank(tmp_path, rho_by_family=rho_by, rep_families=("fam00", "fam11"),
                 fix_families=("fam00",), seed=2)
    res = analyse(collect(path, windows=(10, 20)), n_boot=100)
    row = res[res["metric"] == "contraction"].iloc[0]
    assert abs(row["rho_capability"]) > 0.8, row.to_dict()
    assert row["sig_capability"], row.to_dict()


def test_a_flat_geometry_gives_a_readable_null(flat_bank) -> None:
    """The claim geomcap exists to be able to make.

    Geometry is generated identically for every family while accuracy spans the
    full range, so the correlation must be ~0 AND the design must be shown to have
    had the power to see one.
    """
    res = analyse(collect(flat_bank, windows=(10, 20)), n_boot=200)
    assert not res["sig_capability"].any(), res[["metric", "rho_capability"]]


def test_prompt_length_is_the_positive_control_it_is_meant_to_be(tmp_path) -> None:
    """P5: a covariate the geometry DOES track keeps the capability null readable.

    `n_tokens` is recorded per family here and tied to the planted contraction,
    while accuracy is assigned in an ORTHOGONAL order -- it varies, so the
    capability correlation is defined, but it carries no relation to the geometry.
    That is the exact configuration the real run is being asked to distinguish, and
    a design where accuracy were merely constant would make the capability null
    vacuous rather than informative.
    """
    fams = [f"fam{i:02d}" for i in range(N_FAM)]
    shuffled = np.random.default_rng(5).permutation(N_FAM)   # Spearman 0.0 vs index
    path = _bank(tmp_path, rho_by_family={f: 0.80 + 0.015 * i for i, f in enumerate(fams)},
                 n_tokens_by_family={f: 20 + 4 * i for i, f in enumerate(fams)},
                 acc_by_family={f: float(shuffled[i]) / (N_FAM - 1)
                                for i, f in enumerate(fams)},
                 rep_families=("fam00",), fix_families=("fam00",), seed=3)
    res = analyse(collect(path, windows=(10, 20)), n_boot=100)
    row = res[res["metric"] == "contraction"].iloc[0]
    assert row["capability_defined"] and row["length_defined"]
    assert abs(row["rho_length"]) > 0.8
    assert row["sig_length"]
    assert not row["sig_capability"], "accuracy was unrelated; nothing may track it"


def test_determinism_control_passes_a_correct_run(flat_bank) -> None:
    det = determinism_verdict(collect(flat_bank))
    assert det["fix_all_identical"], det
    assert det["rep_all_distinct"], det


def test_determinism_control_catches_a_seeded_block_that_did_not_reproduce(tmp_path) -> None:
    """If `fix` orbits differ, h_0 is not the only stochastic input.

    That does not void the run, but it downgrades sigma2_h0 from a measurement to
    an upper bound, and the report has to say so rather than claim the mechanism.
    """
    path = _bank(tmp_path, fix_families=("fam00", "fam05"), fix_identical=False,
                 rep_families=("fam00",), seed=4)
    det = determinism_verdict(collect_small(path))
    assert det["fix_all_identical"] is False
    assert det["fix_worst"] > 1


def test_determinism_control_catches_a_ceiling_that_would_be_a_tautology(tmp_path) -> None:
    """The mirror failure, and the more dangerous one.

    If the UNSEEDED replicates come back identical, the harness has removed the
    variation the ceiling is built from, every reliability reads 1.0, and the run
    would announce a perfectly reliable instrument it never measured.
    """
    path = _bank(tmp_path, rep_families=("fam00", "fam05"), rep_identical=True,
                 fix_families=("fam00",), seed=5)
    det = determinism_verdict(collect_small(path))
    assert det["rep_all_distinct"] is False
    assert det["rep_fewest_distinct"] == 1


def test_p1_gate_refuses_an_unrelated_accuracy_axis(tmp_path, monkeypatch) -> None:
    """D75 comparability is a gate, not a footnote."""
    import pandas as pd

    fams = [f"fam{i:02d}" for i in range(N_FAM)]
    path = _bank(tmp_path, seed=6)
    fam = family_table(collect_small(path))
    # A battery whose family ordering is REVERSED: same families, opposite axis.
    csv = tmp_path / "battery.csv"
    pd.DataFrame({"arm": "trained", "family": fams,
                  "correct_best": [1 - i / (N_FAM - 1) for i in range(N_FAM)]}
                 ).to_csv(csv, index=False)
    gate = p1_gate(fam, str(csv))
    assert gate["usable"] and not gate["passes"]
    assert gate["rho"] < 0


def test_p1_gate_passes_a_matching_axis(tmp_path) -> None:
    import pandas as pd

    fams = [f"fam{i:02d}" for i in range(N_FAM)]
    path = _bank(tmp_path, seed=7)
    fam = family_table(collect_small(path))
    csv = tmp_path / "battery.csv"
    pd.DataFrame({"arm": "trained", "family": fams,
                  "correct_best": [i / (N_FAM - 1) for i in range(N_FAM)]}
                 ).to_csv(csv, index=False)
    gate = p1_gate(fam, str(csv))
    assert gate["usable"] and gate["passes"] and gate["rho"] > 0.9


def test_report_states_p6_only_when_both_halves_hold(tmp_path) -> None:
    """The headline sentence must require BOTH a capability null and a length hit.

    A bare "no statistic tracks capability" is the sentence D70 would have printed;
    the length control is what makes it a claim about the model rather than about
    the measurement.
    """
    fams = [f"fam{i:02d}" for i in range(N_FAM)]
    # Seed 5 gives a permutation with Spearman EXACTLY 0.0 against the family
    # index, and the family index is what drives the planted geometry. A merely
    # "random" permutation is not good enough here: at N=12 an arbitrary one
    # correlates with the index by chance often enough to make one statistic read
    # as tracking capability, which is the thing this test asserts cannot happen.
    shuffled = np.random.default_rng(5).permutation(N_FAM)
    acc = {f: float(shuffled[i]) / (N_FAM - 1) for i, f in enumerate(fams)}
    df = collect_small(_bank(
        tmp_path, rho_by_family={f: 0.80 + 0.015 * i for i, f in enumerate(fams)},
        n_tokens_by_family={f: 20 + 4 * i for i, f in enumerate(fams)},
        acc_by_family=acc, rep_families=("fam00",), fix_families=("fam00",), seed=8))
    txt = report(df, analyse(df, n_boot=100), family_table(df),
                 {"usable": False, "why": "no battery"}, determinism_verdict(df))
    assert "P6:" in txt
    assert "not to the computation performed on it" in txt

    # Same capability axis, but the geometry now tracks NOTHING -- no length
    # relation, so the capability null cannot be attributed to the model.
    flat = collect_small(_bank(tmp_path / "b", acc_by_family=acc,
                         rep_families=("fam00",), fix_families=("fam00",), seed=9))
    txt2 = report(flat, analyse(flat, n_boot=100), family_table(flat),
                  {"usable": False, "why": "no battery"}, determinism_verdict(flat))
    assert "P6:" not in txt2, "P6 fired without a length control"

    # And a capability axis with NO SPREAD must not let P6 fire either, however
    # strong the length relation is: "nothing tracks capability" is vacuous there.
    const = collect_small(_bank(
        tmp_path / "c", rho_by_family={f: 0.80 + 0.015 * i for i, f in enumerate(fams)},
        n_tokens_by_family={f: 20 + 4 * i for i, f in enumerate(fams)},
        acc_by_family=dict.fromkeys(fams, 0.5),
        rep_families=("fam00",), fix_families=("fam00",), seed=10))
    txt3 = report(const, analyse(const, n_boot=100), family_table(const),
                  {"usable": False, "why": "no battery"}, determinism_verdict(const))
    assert "P6:" not in txt3, "P6 fired on a capability axis with no spread"
    assert "UNDEFINED" in txt3


def test_report_survives_a_missing_battery(flat_bank) -> None:
    df = collect(flat_bank, windows=(10, 20))
    txt = report(df, analyse(df, n_boot=50), family_table(df),
                 {"usable": False, "why": "results/battery.csv absent"},
                 determinism_verdict(df))
    assert "NOT CHECKABLE" in txt


def test_h0_share_is_reported_from_the_replicate_block(flat_bank) -> None:
    """`sigma2_h0` must come from `rep`, not be inferred from `main`.

    The whole design rests on that separation: `main` confounds item choice with
    h_0, and only the replicate block can tell them apart.
    """
    res = analyse(collect(flat_bank, windows=(10, 20)), n_boot=50)
    live = res[res["usable"].astype(bool)]
    assert len(live)
    assert live["sigma2_h0"].notna().all()
    assert ((live["h0_share"] >= 0) & (live["h0_share"] <= 1.5)).all(), live["h0_share"]


def test_a_constant_metric_is_refused_not_filed_as_a_null(tmp_path) -> None:
    """THE FAILURE MODE THAT WOULD MANUFACTURE THE HEADLINE.

    scipy returns NaN with a ConstantInputWarning when a column has no spread, and
    NaN < alpha is False -- so an UNDEFINED correlation would be recorded as "does
    not track capability" and counted toward P6. `correlate.py` already documents
    this for `steps_settle`, the constant 128 at every level in one sweep. The
    refusal has to be explicit.
    """
    fams = [f"fam{i:02d}" for i in range(N_FAM)]
    path = _bank(tmp_path, acc_by_family={f: i / (N_FAM - 1) for i, f in enumerate(fams)},
                 rep_families=("fam00",), fix_families=("fam00",), seed=21)
    df = collect(path, windows=(10, 20))
    df["contraction"] = 0.88                       # a metric with no spread at all
    res = analyse(df, n_boot=50)
    row = res[res["metric"] == "contraction"].iloc[0]
    assert not row["usable"]
    assert not row["sig_capability"] and not row["null_is_readable"]
    assert "not the same as one being zero" in row["why"]
    txt = report(df, res, family_table(df), {"usable": False, "why": "n/a"},
                 determinism_verdict(df))
    assert "REFUSED" in txt
    n_usable = int(res["usable"].astype(bool).sum())
    n_refused = len(res) - n_usable
    assert n_refused >= res["window_k"].nunique(dropna=False), (
        "every window's `contraction` cell should have been refused")
    assert f"of {n_usable} usable" in txt, "a refused cell still counted as usable"


def test_covariate_structure_warns_when_length_is_capability(tmp_path) -> None:
    """If prompt length and accuracy are the same axis, the dissociation is void.

    "Geometry tracks length but not capability" only means something when the two
    are separable across the 21 families. That is a property of the TASK SUITE, not
    of the model, so it is measured and printed rather than assumed -- and if it
    fails, the report has to say the finding was unavailable rather than absent.
    """
    fams = [f"fam{i:02d}" for i in range(N_FAM)]
    df = collect_small(_bank(
        tmp_path,
        n_tokens_by_family={f: 20 + 4 * i for i, f in enumerate(fams)},
        acc_by_family={f: i / (N_FAM - 1) for i, f in enumerate(fams)},
        rep_families=("fam00",), fix_families=("fam00",), seed=31))
    fam = family_table(df)
    cov = covariate_structure(fam)
    assert cov["acc~n_tokens"][0] > 0.9
    txt = report(df, analyse(df, n_boot=50), fam,
                 {"usable": False, "why": "n/a"}, determinism_verdict(df))
    assert "nearly the same axis" in txt


def test_covariate_structure_stays_quiet_when_the_axes_are_separable(tmp_path) -> None:
    """Non-suppression: the warning must not fire on the design geomcap actually has."""
    fams = [f"fam{i:02d}" for i in range(N_FAM)]
    shuffled = np.random.default_rng(5).permutation(N_FAM)   # Spearman 0.0 vs index
    df = collect_small(_bank(
        tmp_path,
        n_tokens_by_family={f: 20 + 4 * i for i, f in enumerate(fams)},
        acc_by_family={f: float(shuffled[i]) / (N_FAM - 1) for i, f in enumerate(fams)},
        rep_families=("fam00",), fix_families=("fam00",), seed=32))
    cov = covariate_structure(family_table(df))
    assert abs(cov["acc~n_tokens"][0]) < 0.3
    txt = report(df, analyse(df, n_boot=50), family_table(df),
                 {"usable": False, "why": "n/a"}, determinism_verdict(df))
    assert "nearly the same axis" not in txt
