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
    # Drop the build directive generically. It was matched as a literal string,
    # which silently left a stray fragment behind for any bundle declaring a
    # different block list -- an IndentationError that looked like a syntax error
    # in the kernel rather than in this helper.
    head = "\n".join(ln for ln in src.split(cut)[0].splitlines()
                     if not ln.lstrip().startswith("# @needs:"))
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


def test_battery_readout_has_not_drifted_from_the_validated_one() -> None:
    """`coda_head` must stay the exact tail D71 measured as bit-identical.

    D71 is the only positive control the per-unroll readout has: max|delta| =
    0.000000 against the model's own logits, with the known-wrong one-ln_f variant
    separating at 2.33-2.46. That validation attaches to a specific piece of code,
    so any new kernel reusing the readout has to reuse it unchanged -- otherwise it
    inherits the CLAIM without the EVIDENCE.

    Version 1 of the battery kernel is the cautionary case. It applied `lm_head`
    only to the read positions, which is exact by construction for a position-wise
    Linear, and the in-kernel check still measured 5.72e-06: cuBLAS reduces a
    3-row matmul in a different order than a 30-row one. Mathematically right,
    not bit-identical.
    """
    src_g = open(os.path.join(ROOT, "scratch", "kaggle_graded", "body.py"),
                 encoding="utf-8").read()
    src_b = open(os.path.join(ROOT, "scratch", "kaggle_battery", "body.py"),
                 encoding="utf-8").read()

    def body_of(src: str, name: str) -> str:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                node.body = [s for s in node.body
                             if not (isinstance(s, ast.Expr)
                                     and isinstance(s.value, ast.Constant)
                                     and isinstance(s.value.value, str))]
                return ast.unparse(node)
        raise AssertionError(f"{name} not found")

    assert body_of(src_b, "coda_head") == body_of(src_g, "coda_head"), (
        "kaggle_battery's coda_head has drifted from the D71-validated version in "
        "kaggle_graded; the readout would no longer carry D71's evidence")


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


# --------------------------------------------------------------------------
# geomcap: the capability-vs-geometry bank, plus its own reliability ceiling.
# --------------------------------------------------------------------------


def test_geomcap_items_are_byte_identical_to_the_battery() -> None:
    """All 336 D75 items, verbatim -- or the capability axis is a different axis.

    geomcap's entire tie to D75 is that its 21 families ARE D75's 21 families with
    D75's items. AST equality of `items` alone would not show this: `items` calls
    `WORDS`, `zlib.crc32` and per-family arithmetic that can drift independently.
    So this runs both generators and compares the emitted triples.
    """
    bat = _load("kaggle_battery", "def coda_head")
    cap = _load("kaggle_geomcap", "def coda_head")
    assert bat["TASKS"] == cap["TASKS"], "family list drifted from the battery"
    for task in bat["TASKS"]:
        assert cap["items"](task, n=16) == bat["items"](task, n=16), (
            f"{task}: geomcap's items drifted from the battery's, so its "
            f"accuracies are not D75's accuracies")


def test_geomcap_item_generator_is_prefix_stable() -> None:
    """n=24 must EXTEND the battery's 16, not redraw them.

    geomcap raises N_ITEMS above the battery's 16 to give the within-family
    stratified test more items per family. That is only legitimate if the first 16
    are unchanged -- otherwise P1's "correlate with D75" gate compares two
    different item sets and cannot fail for the right reason.
    """
    cap = _load("kaggle_geomcap", "def coda_head")
    assert cap["N_ITEMS"] == 24
    for task in cap["TASKS"]:
        wide = cap["items"](task, n=24)
        assert len(wide) == 24
        assert wide[:16] == cap["items"](task, n=16), f"{task} is not prefix-stable"


def test_geomcap_readout_has_not_drifted_from_the_validated_one() -> None:
    """Same D71 guard as the battery: the readout must carry its own evidence."""
    src_b = open(os.path.join(ROOT, "scratch", "kaggle_battery", "body.py"),
                 encoding="utf-8").read()
    src_c = open(os.path.join(ROOT, "scratch", "kaggle_geomcap", "body.py"),
                 encoding="utf-8").read()

    def body_of(src: str, name: str) -> str:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                node.body = [s for s in node.body
                             if not (isinstance(s, ast.Expr)
                                     and isinstance(s.value, ast.Constant)
                                     and isinstance(s.value.value, str))]
                return ast.unparse(node)
        raise AssertionError(f"{name} not found")

    assert body_of(src_c, "coda_head") == body_of(src_b, "coda_head")


def test_geomcap_pools_with_b6bank_at_the_same_depth() -> None:
    """NUM_STEPS must match b6bank, or the two banks cannot be analysed together.

    Every window-dependent statistic in this project moves with the number of
    unrolls -- D74(6) records `partial_spearman` refusing to correct for window
    length at rho = -1.000, i.e. the confound is total. Two banks at different
    depths are two datasets, not one.
    """
    cap = _load("kaggle_geomcap", "def coda_head")
    bank = _load("kaggle_b6bank", "def coda_head")
    assert cap["NUM_STEPS"] == bank["NUM_STEPS"] == 64


def test_geomcap_ceiling_block_spans_the_accuracy_range() -> None:
    """`pick_spanning` must take the extremes, not a clump.

    The ceiling block exists to put within-prompt spread in the same units as the
    between-family spread. If it sampled only mid-accuracy families the comparison
    would understate the between-family range it is meant to calibrate.
    """
    cap = _load("kaggle_geomcap", "def coda_head")
    acc = {f: i / 20 for i, f in enumerate(cap["TASKS"])}      # 0.00 .. 1.00
    span = cap["pick_spanning"](acc, k=8)
    assert len(span) == 8
    assert min(acc[f] for f in span) == 0.0
    assert max(acc[f] for f in span) == 1.0
    assert len(set(span)) == 8, "a family must not be replicated twice"


def test_geomcap_ceiling_block_is_deterministic_and_tie_safe() -> None:
    """All-equal accuracies must not crash or return duplicates.

    An all-zero accuracy vector is a real possibility for this model -- D75 found
    several families at exactly 0% -- and a selector that ties on every key must
    still return k distinct families.
    """
    cap = _load("kaggle_geomcap", "def coda_head")
    flat = dict.fromkeys(cap["TASKS"], 0.0)
    span = cap["pick_spanning"](flat, k=8)
    assert len(span) == len(set(span)) == 8
    assert span == cap["pick_spanning"](flat, k=8), "selection is not deterministic"


def test_geomcap_seeds_h0_only_when_asked() -> None:
    """`manual_seed` must be reachable ONLY under the `h0_seed is not None` branch.

    The `main` and `rep` blocks have to reproduce Huginn's own unseeded
    `initialize_state`, because that is what every other measurement in this
    project ran under (D78). A stray unconditional seed would silently make `rep`'s
    replicates identical and turn the reliability ceiling into a tautology --
    reporting zero within-prompt variance because the code removed it.
    """
    src = open(os.path.join(ROOT, "scratch", "kaggle_geomcap", "body.py"),
               encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "bank_one")
    seeds = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "manual_seed"]
    assert len(seeds) == 1, f"{len(seeds)} manual_seed calls in bank_one"
    guards = [n for n in ast.walk(fn)
              if isinstance(n, ast.If)
              and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                      and c.func.attr == "manual_seed" for c in ast.walk(n))]
    assert guards, "manual_seed is not inside a conditional"
    assert "h0_seed" in ast.unparse(guards[0].test), (
        f"manual_seed is guarded by {ast.unparse(guards[0].test)!r}, not by h0_seed")


# --------------------------------------------------------------------------
# depthacc: does the model SAY what its rank curve says it knows?
# --------------------------------------------------------------------------


def test_depthacc_items_match_the_battery() -> None:
    """Same 21 families and the same item prefix, or D75 is not the comparison."""
    bat = _load("kaggle_battery", "def coda_head")
    dep = _load("kaggle_depthacc", "def normalise")
    assert bat["TASKS"] == dep["TASKS"]
    for task in bat["TASKS"]:
        assert dep["items"](task, n=6) == bat["items"](task, n=6)[:6], task


def test_depthacc_scorer_is_symmetric_between_prediction_and_gold() -> None:
    """`normalise` must be applied to BOTH sides.

    Stripping punctuation from the prediction only would let "7." score wrong
    against "7" while "7" scored right against "7." -- an asymmetry that would
    move accuracy in one direction across the whole table.
    """
    ns = _load("kaggle_depthacc", "def coda_head")
    score, normalise = ns["score"], ns["normalise"]
    assert score("7.", "7") == (True, True)
    assert score("7", "7.") == (True, True)
    assert normalise("The 7.") == normalise("7")


def test_depthacc_containment_separates_the_discourse_failure() -> None:
    """The gap between exact and contains IS the effect D68(3) describes.

    "The number is 7" is scored wrong by exact match and right by containment. If
    containment did not fire there, the run could not measure how much of the 0%
    exact-match rate is a prose opener rather than a wrong answer.
    """
    score = _load("kaggle_depthacc", "def coda_head")["score"]
    assert score("The number is 7", "7") == (False, True)
    assert score("The number is 8", "7") == (False, False)
    assert score("Subtract", "Add") == (False, False)


def test_depthacc_containment_does_not_match_a_substring_of_another_word() -> None:
    """Whole-word containment for single-token golds.

    A naive `gold in pred` would score "Addition" as containing "Add", inflating
    every `local_last` cell.
    """
    score = _load("kaggle_depthacc", "def coda_head")["score"]
    assert score("Addition of terms", "Add")[1] is False
    assert score("Add", "Add")[1] is True


def test_depthacc_runs_the_decisive_depths_first() -> None:
    """Kaggle kills at the wall clock with whatever has been written.

    Every cell is persisted as it completes, so the ORDER of `DEPTHS` decides what
    survives a timeout. D68 puts the contrast at r=4 against r=32, so those two
    must not be last.
    """
    ns = _load("kaggle_depthacc", "def items")
    assert ns["DEPTHS"][:2] == (4, 32), ns["DEPTHS"]
    assert set(ns["DEPTHS"]) == {2, 4, 8, 16, 32}


def test_depthacc_gates_the_decoder_before_reading_it() -> None:
    """D62 printed a capability verdict over empty strings for want of this gate."""
    src = open(os.path.join(ROOT, "scratch", "kaggle_depthacc", "body.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [ast.unparse(n.func) for n in ast.walk(main) if isinstance(n, ast.Call)]
    assert "assert_generation_works" in calls
    assert calls.index("assert_generation_works") < calls.index("batched_generate"), (
        "the decoder is used before it is gated")


# --------------------------------------------------------------------------
# lenmatch: two computations at an IDENTICAL token count -- D84's missing test.
# --------------------------------------------------------------------------


def test_lenmatch_prompts_differ_in_exactly_one_character() -> None:
    """THE ENTIRE POINT OF THE RUN.

    D84 could not attribute its 100% family decoding to the task, because no two
    families in any bank share a prompt length -- 0 of 6 pairs in one, 0 of 3 in the
    other. Here the two forms must be byte-identical apart from the trailing marker,
    so that anything a classifier separates cannot be sequence length.
    """
    ns = _load("kaggle_lenmatch", "def coda_head")
    forms: dict = {}
    for pair, marker, prompt, _ in ns["pairs"](8):
        forms.setdefault((pair, prompt[:-1]), {})[marker] = prompt
    assert forms, "no prompts generated"
    for (pair, _), f in forms.items():
        assert set(f) == {"A", "B"}, pair
        a, b = f["A"], f["B"]
        assert len(a) == len(b), f"{pair}: {len(a)} vs {len(b)} characters"
        diff = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
        assert diff == [len(a) - 1], f"{pair}: differs at {diff}, not only the marker"


def test_lenmatch_tasks_actually_differ() -> None:
    """A length-matched pair whose two golds always agree would test nothing.

    The markers have to select genuinely different computations, or a classifier
    finding no difference would be reporting that the model was asked one question
    twice.
    """
    ns = _load("kaggle_lenmatch", "def coda_head")
    golds: dict = {}
    for pair, marker, prompt, gold in ns["pairs"](32):
        golds.setdefault((pair, prompt[:-1]), {})[marker] = gold
    by_pair: dict = {}
    for (pair, _), g in golds.items():
        by_pair.setdefault(pair, []).append(g["A"] != g["B"])
    for pair, diffs in by_pair.items():
        assert sum(diffs) / len(diffs) > 0.5, (
            f"{pair}: the two markers give the same answer in "
            f"{1 - sum(diffs) / len(diffs):.0%} of items")


def test_lenmatch_gates_length_in_the_kernel_not_in_a_docstring() -> None:
    """`make_variants` once ASSERTED length-matching and was wrong (this file's
    own opening docstring records it). This run exists because of that failure, so
    the property is measured per item and the drop count reported."""
    src = open(os.path.join(ROOT, "scratch", "kaggle_lenmatch", "body.py"),
               encoding="utf-8").read()
    main = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = ast.unparse(main)
    assert "dropped" in body and "la != lb" in body, (
        "the identical-length check is not enforced inside main()")
    assert body.index("dropped") < body.index("load_arm"), (
        "the length gate must run BEFORE the model is loaded, so a failure is free")


def test_lenmatch_pools_with_the_other_banks() -> None:
    ns = _load("kaggle_lenmatch", "def coda_head")
    assert ns["NUM_STEPS"] == _load("kaggle_b6bank", "def coda_head")["NUM_STEPS"]


def test_lenmatch_readout_has_not_drifted_from_the_validated_one() -> None:
    """Same D71 guard as every other bank."""
    def body_of(src: str, name: str) -> str:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                node.body = [s for s in node.body
                             if not (isinstance(s, ast.Expr)
                                     and isinstance(s.value, ast.Constant)
                                     and isinstance(s.value.value, str))]
                return ast.unparse(node)
        raise AssertionError(name)

    a = open(os.path.join(ROOT, "scratch", "kaggle_battery", "body.py"),
             encoding="utf-8").read()
    b = open(os.path.join(ROOT, "scratch", "kaggle_lenmatch", "body.py"),
             encoding="utf-8").read()
    assert body_of(b, "coda_head") == body_of(a, "coda_head")


# --------------------------------------------------------------------------
# clrs: the capability screen on a dataset that is IN Huginn's training mix.
# --------------------------------------------------------------------------


def test_clrs_final_answer_splits_on_the_last_separator() -> None:
    """CLRS answers are `trace | final`, and several algorithms emit `|` inside the
    trace. Splitting on the FIRST separator would score a trace step as the answer
    and the whole screen would read as failure."""
    ns = _load("kaggle_clrs", "def main")
    fa = ns["final_answer"]
    assert fa("(1,1), (5,17) | (5, 17)") == "(5, 17)"
    assert fa("a | b | c") == "c"
    assert fa("no separator here") == "no separator here"


def test_clrs_problem_size_reads_the_difficulty_knob() -> None:
    """Problem size is the difficulty parameter and the dataset does not carry it as
    a column -- only `question`, `answer`, `algo_name`. It has to be parsed, and for
    a graph algorithm the first bracketed group is a MATRIX ROW, which for an n x n
    adjacency matrix is n."""
    ns = _load("kaggle_clrs", "def main")
    ps = ns["problem_size"]
    assert ps("key: [-0.5 0.7 0.1], initial_trace: (0,0)") == 3
    assert ps("A: [[0 1], [0 0]], initial_trace: [0 1]") == 2
    assert ps("no list at all") == 0


def test_clrs_scores_the_decoded_string_not_a_first_token() -> None:
    """D89: `correct` as first-token rank measures the leading token, and CLRS
    answers are long numeric strings where that instrument is simply wrong."""
    ns = _load("kaggle_clrs", "def main")
    score = ns["score"]
    assert score("(5, 17)", "(5, 17)") == (True, True)
    assert score("the answer is (5, 17)", "(5, 17)") == (False, True)
    assert score("(5, 18)", "(5, 17)") == (False, False)


def test_clrs_brackets_the_depth_where_d86_found_the_transition() -> None:
    """D86: exact-match peaks shallow and hits 0% by r=8 while containment rises to
    r=32. A screen at one depth would report whichever side of that it landed on."""
    ns = _load("kaggle_clrs", "def main")
    assert ns["DEPTHS"] == (4, 32)


def test_clrs_gates_the_decoder_before_reading_it() -> None:
    src = open(os.path.join(ROOT, "scratch", "kaggle_clrs", "body.py"),
               encoding="utf-8").read()
    main = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [ast.unparse(n.func) for n in ast.walk(main) if isinstance(n, ast.Call)]
    assert "assert_generation_works" in calls
    assert calls.index("assert_generation_works") < calls.index("batched_generate")
