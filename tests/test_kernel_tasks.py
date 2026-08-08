"""Guards on the task generators inside Kaggle kernel bundles.

A kernel is a single file, so every generator is COPIED into it rather than
imported. That copy is exactly the kind of thing that drifts silently: the
library version gets corrected (as `make_variants`' docstring was on 2026-08-08,
when its "same token length" claim turned out to be false) while the kernel keeps
the old one, and the two experiments stop being comparable without anything
failing.

These tests are static and cheap. They load each bundle's `body.py` in a
namespace, without importing torch or touching a GPU.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(bundle: str, cut: str) -> dict:
    """Exec the pure-python prefix of a bundle's body (everything before `cut`)."""
    path = os.path.join(ROOT, "scratch", bundle, "body.py")
    if not os.path.exists(path):
        pytest.skip(f"{bundle}/body.py not present")
    src = open(path, encoding="utf-8").read()
    head = src.split(cut)[0].replace("# @needs: run load_arm free_arm", "")
    ns: dict = {}
    exec(compile(head, path, "exec"), ns)  # noqa: S102
    return ns


def test_h2rot_task_matches_library() -> None:
    """The H2 rotation kernel must use the SAME length-matched pair as everything else.

    B4.15's whole design rests on `track` and `local` sharing a byte-identical
    body, so that a difference between them is reasoning depth and not text. If
    the kernel's copy drifts from `traj_geom.shapes.synthetic.make_variants`, the
    result is not comparable with B3/B3c and the control is void.
    """
    from traj_geom.shapes.synthetic import make_variants as lib

    kern = _load("kaggle_h2rot", "def spectrum")["make_variants"]
    for n_ops in (4, 8, 12, 16, 24, 32):
        for seed in (0, 1):
            assert kern(n_ops, seed=seed) == lib(n_ops, seed=seed), (
                f"kernel make_variants drifted from the library at "
                f"n_ops={n_ops}, seed={seed}")


def test_h2rot_grid_gives_six_spearman_levels() -> None:
    """N=6 levels is load-bearing: the project's own p<0.05 threshold is 0.886 there.

    At N=4 the threshold is 1.000 and no result is interpretable -- the trap D65
    records for the PARARULE depth sweep.
    """
    from traj_geom.analysis.correlate import _SPEARMAN_CRIT_P05

    ns = _load("kaggle_h2rot", "def spectrum")
    assert len(ns["N_OPS"]) == 6
    assert _SPEARMAN_CRIT_P05[6] == pytest.approx(0.886, abs=5e-4)


def test_battery_items_are_process_stable() -> None:
    """No kernel may seed its items with Python's per-process salted `hash()`.

    `geometry-graded-readout` (D68) and `geometry-discourse` each drew a DIFFERENT
    item set for the same nominal cell because of this, which is why one reported
    96% and the other 83% on `echo_digit` (D69(3)). crc32 is stable across
    interpreters; `hash(str)` is not.
    """
    path = os.path.join(ROOT, "scratch", "kaggle_battery", "body.py")
    src = open(path, encoding="utf-8").read()
    assert "zlib.crc32" in src

    # AST, not substring: the docstring quotes `hash(task)` to explain why it is
    # banned, so a text search flags the very comment that documents the rule.
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "hash"]
    assert not calls, f"{len(calls)} call(s) to hash() in kaggle_battery/body.py"

    # The property is stability ACROSS INTERPRETERS, which one process cannot show.
    # Python's str-hash salt is per process, so two runs under different
    # PYTHONHASHSEED values are the real test (measured salts 544/92/779, D69(3)).
    prog = (
        "import json,sys;sys.path.insert(0,'.');"
        f"src=open({path!r},encoding='utf-8').read();"
        "head=src.split('def coda_head')[0].replace('# @needs: run load_arm free_arm','');"
        "ns={};exec(compile(head,'b','exec'),ns);"
        "print(json.dumps([ns['items'](t)[:2] for t in ns['TASKS']]))"
    )
    outs = []
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.append(subprocess.run([sys.executable, "-c", prog], check=True,
                                   capture_output=True, text=True, env=env,
                                   cwd=ROOT).stdout)
    assert outs[0] == outs[1] == outs[2], (
        "item sets differ across PYTHONHASHSEED values -- the kernel is not "
        "reproducible between runs")


def test_battery_distractor_never_equals_gold() -> None:
    """`logp_dist` is the margin term; a distractor equal to gold makes it zero."""
    ns = _load("kaggle_battery", "def coda_head")
    for task in ns["TASKS"]:
        for prompt, gold, dist in ns["items"](task):
            assert dist != gold, f"{task}: distractor equals gold for {prompt!r}"


def test_battery_covers_families_with_few_gold_values() -> None:
    """B6 needs a cell where success and failure can coexist AT THE SAME gold value.

    D72 killed the first B6 run because correctness was a deterministic function
    of the answer: `add1`'s correct and incorrect gold sets had ZERO overlap. A
    family with few distinct golds over many items is what makes the within-value
    contrast constructible at all, so the screen must contain some.
    """
    ns = _load("kaggle_battery", "def coda_head")
    few = {t for t in ns["TASKS"]
           if len({g for _, g, _ in ns["items"](t)}) <= 3}
    assert {"parity8", "count_mod3", "local_last"} <= few, (
        f"expected low-cardinality families for B6 stratification, got {sorted(few)}")
