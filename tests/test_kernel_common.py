"""Tests for the shared kernel blocks, so GPU code starts from checked code.

The blocks in `scratch/_lib/kernel_common.py` are inlined into Kaggle kernels by
`scripts/build_kernel.py`. Before they existed, each kernel re-declared its own
copy: 4563 lines across 23 bundles, 28% of 8-line windows duplicated. That
duplication reintroduced a fixed bug twice -- `np.arange(1, n+1, float)`, which
passes `float` as the STEP -- and cost a GPU run. A GPU minute is expensive and
a laptop second is not, so anything that can be checked here is.

`test_fit_rho_uses_dtype_not_step` exists specifically to pin that bug.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "scratch", "_lib", "kernel_common.py")


@pytest.fixture(scope="module")
def lib() -> dict:
    """Exec the block file and hand back its namespace."""
    if not os.path.exists(LIB):
        pytest.skip("kernel_common.py absent")
    ns: dict = {}
    exec(compile(open(LIB, encoding="utf-8").read(), LIB, "exec"), ns)  # noqa: S102
    return ns


# --- fit_rho: the block that most needs to be right ------------------------


@pytest.mark.parametrize("rho", [0.60, 0.662, 0.75, 0.85, 0.91, 0.95])
def test_fit_rho_recovers_a_known_rate(lib, rho: float) -> None:
    """Under a planted arithmetic floor, recovery must be close and biased UPWARD.

    Upward bias is the safe direction: it understates any trained-vs-untrained
    gap rather than manufacturing one (claims_ledger D52).
    """
    rng = np.random.default_rng(0)
    y = rho ** np.arange(128.0) + 1e-5 * (1 + 0.05 * rng.normal(size=128))
    got, n, fit = lib["fit_rho"](y)
    assert abs(got - rho) < 0.02, f"recovered {got:.4f} for true {rho}"
    assert got >= rho - 1e-9, "bias must be upward, never downward"
    assert n >= 10 and fit > 0.99


def test_fit_rho_refuses_to_fit_too_few_points(lib) -> None:
    """A slope from three points is not a measurement and must return nan."""
    got, n, _ = lib["fit_rho"](np.concatenate([[1.0, 0.1, 0.01], np.full(20, 1e-3)]))
    assert np.isnan(got), f"returned {got} from {n} usable points"


def test_fit_rho_handles_degenerate_input(lib) -> None:
    for bad in ([1, 1, 1], np.zeros(20), np.full(20, np.nan)):
        got, _, _ = lib["fit_rho"](bad)
        assert np.isnan(got)


def test_fit_rho_uses_dtype_not_step(lib) -> None:
    """Pins the bug that was fixed once and copy-pasted back twice.

    `np.arange(1, n+1, float)` passes `float` as the STEP argument, not the dtype,
    and raises `TypeError: unsupported operand type(s) for /: 'int' and 'type'`.
    It appeared three separate times in this project.

    Checked by walking the AST, not by grepping: a regex over the source also
    matches the docstring that *describes* the bug, which is exactly what happened
    the first time this test was written.
    """
    import ast

    tree = ast.parse(open(LIB, encoding="utf-8").read())
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name != "arange" or len(node.args) < 3:
            continue
        step = node.args[2]
        if isinstance(step, ast.Name) and step.id in {"float", "int"}:
            bad.append(f"line {node.lineno}")
    assert not bad, f"np.arange with a bare type as STEP at {bad}"


# --- summarise: must expose the analyst choice, not hide it ---------------


def test_summarise_reports_all_fit_and_clean_fit(lib) -> None:
    """D52(2): dropping fits below the R^2 bar moved a trend from p=0.036 to 0.19.

    Both means must be emitted so that sensitivity is visible without a rerun.
    """
    rows = [{"rho_orbit": 0.85, "rho_step": 0.84, "r2_orbit": 0.99, "family": "a"}] * 8
    rows += [{"rho_orbit": 0.97, "rho_step": 0.96, "r2_orbit": 0.52, "family": "b"}] * 2
    out = lib["summarise"](rows)
    assert out["n_below_bar"] == 2
    assert out["rho_orbit"]["mean"] == pytest.approx(0.874, abs=1e-3)
    assert out["clean"]["rho_orbit"]["mean"] == pytest.approx(0.85, abs=1e-9)
    assert set(out["by_family"]) == {"a", "b"}


# --- cv_r2: the grouping and the null are the whole point -----------------


def test_cv_r2_null_is_negative_for_an_honest_fit(lib) -> None:
    """A permuted null at or above zero means the probe can interpolate."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 20))
    y = x[:, 0] * 3 + rng.normal(size=200) * 0.1
    r2, null = lib["cv_r2"](x, y, alpha=1.0, n_null=8)   # alpha=1e3 would shrink this away
    assert r2 > 0.8, f"planted signal not recovered: {r2:.3f}"
    assert max(null) < 0.0, f"permuted null reached {max(null):+.3f}"


def test_cv_r2_grouping_prevents_within_prompt_leakage(lib) -> None:
    """Rows from one prompt must not be split across folds.

    With a per-group constant target, a grouped CV cannot predict held-out groups
    (R^2 <= 0) while an ungrouped one trivially can. If grouping were ignored the
    two would agree.
    """
    rng = np.random.default_rng(2)
    n_g, per, dim = 12, 10, 40
    groups = np.repeat(np.arange(n_g), per)
    sig = rng.normal(size=(n_g, dim))          # each group has its own signature...
    level = rng.normal(size=n_g)               # ...unrelated to its target
    x = sig[groups] + rng.normal(size=(n_g * per, dim)) * 0.01
    y = level[groups]
    grouped, _ = lib["cv_r2"](x, y, groups=groups, n_splits=4, alpha=1.0)
    ungrouped, _ = lib["cv_r2"](x, y, n_splits=4, alpha=1.0)
    assert ungrouped > 0.9, (
        f"ungrouped should trivially succeed by matching a test row to its "
        f"training siblings, got {ungrouped:.3f}"
    )
    assert grouped < ungrouped - 0.3, (
        f"grouping is not taking effect: grouped {grouped:.3f} vs "
        f"ungrouped {ungrouped:.3f}"
    )


# --- prompts_by_family ----------------------------------------------------


def test_prompts_span_several_families_and_lengths(lib) -> None:
    """Constancy across families can only be TESTED if the families really differ."""
    ps = lib["prompts_by_family"](n_per=3)
    fams = {f for f, _ in ps}
    assert fams == {"counting", "nesting", "arith", "commonsense"}
    assert len(ps) == 12
    lens = [len(t.split()) for _, t in ps]
    assert max(lens) > 4 * min(lens), (
        "families no longer differ in length; D45's collinearity caveat and any "
        "length-vs-family separation depend on this spread"
    )


# --- the composer ---------------------------------------------------------


def test_build_kernel_resolves_dependencies_in_order() -> None:
    from scripts.build_kernel import parse_lib, resolve

    blocks = parse_lib()
    assert {"fit_rho", "measure_rho", "summarise"} <= set(blocks)
    order = resolve(["measure_rho"], blocks)
    assert order.index("fit_rho") < order.index("measure_rho"), (
        "measure_rho calls fit_rho, so fit_rho must be emitted first"
    )
    assert "capture_unrolls" in order


def test_build_kernel_rejects_an_unknown_block() -> None:
    from scripts.build_kernel import parse_lib, resolve

    with pytest.raises(KeyError):
        resolve(["no_such_block"], parse_lib())


def _fake_lm(next_tokens, torch):
    """A model whose argmax emits `next_tokens` in order, then a stop token."""
    import types

    class M:
        def __init__(self):
            self.calls = []
            self.step = 0

        def parameters(self):
            yield torch.zeros(1)

        def __call__(self, input_ids=None, num_steps=None, **kw):
            self.calls.append(tuple(input_ids.shape))
            b, _ = input_ids.shape
            tok_id = next_tokens[self.step] if self.step < len(next_tokens) else 65508
            self.step += 1
            logits = torch.full((b, input_ids.shape[1], 70000), -1e9)
            logits[:, -1, tok_id] = 0.0
            return types.SimpleNamespace(logits=logits)

    return M()


def _fake_tok(torch, decode_map=None):
    import types

    class T:
        eos_token_id, pad_token_id = 65505, 65509

        def __call__(self, p, **kw):
            return types.SimpleNamespace(
                input_ids=torch.zeros(1, len(p), dtype=torch.long))

        def decode(self, x, **kw):
            ids = [int(v) for v in x]
            if decode_map is not None:
                return "".join(decode_map.get(i, "") for i in ids)
            return "".join("w" for i in ids if i not in (65504, 65505, 65508, 65509))

    return T()


def test_batched_generate_never_mixes_lengths_in_a_batch(lib) -> None:
    """`RavenForCausalLM.forward` sets `prepared_attn_mask = None` -- the attention
    mask is commented out -- so padding is NOT masked and a mixed-length batch
    silently attends to pad tokens. Batching is only safe across identical lengths,
    and lengths cannot be assumed equal (ten six-letter words through one template
    tokenise to 15 or 16 tokens)."""
    torch = pytest.importorskip("torch")
    m = _fake_lm([100, 101], torch)
    lib["batched_generate"](m, _fake_tok(torch), ["a" * n for n in (5, 5, 5, 7, 7, 9)],
                            max_new=3, verbose=False)
    widths = {c[1] - k for c in m.calls for k in (0,)}   # prompt widths seen
    starts = sorted({c[1] for c in m.calls})
    assert min(starts) >= 5, f"unexpected batch widths {starts}"
    assert {c[0] for c in m.calls} == {1, 2, 3}, (
        f"expected one batch per distinct length with sizes 1,2,3; got "
        f"{sorted({c[0] for c in m.calls})}"
    )
    _ = widths


def test_batched_generate_stops_on_a_stop_token(lib) -> None:
    """Huginn's stop set is {65504, 65505, 65508} = begin_text, end_text, end_turn.
    A model that emits one immediately must terminate that sequence."""
    torch = pytest.importorskip("torch")
    m = _fake_lm([65508], torch)                       # end_turn on the first step
    with pytest.raises(RuntimeError, match="empty output"):
        lib["batched_generate"](m, _fake_tok(torch), ["aaa", "bbb"],
                                max_new=10, verbose=False)
    assert m.step == 1, f"should have stopped after one step, ran {m.step}"




def test_batched_generate_respects_the_activation_budget(lib) -> None:
    """`batch x sequence` must stay bounded, or the MLP OOMs.

    This loop has no KV cache, so it re-runs the full growing sequence every step.
    float32 weights are ~14.1 GB of a T4's 14.56 GB, leaving ~450 MB for
    activations against a gated MLP of inner width 17920 -- batch 16 x ~100 tokens
    died inside `nonlin(x_fc_1) * x_fc_2` in geometry-prompt-depth. Buckets are now
    split so `chunk * (prompt_len + max_new) <= max_batch_tokens`.
    """
    torch = pytest.importorskip("torch")
    m = _fake_lm([100], torch)
    # 40 same-length prompts of 50 tokens, generating 50 more: 100 tok each
    lib["batched_generate"](m, _fake_tok(torch), ["a" * 50] * 40, max_new=50,
                            max_batch_tokens=400, verbose=False)
    widest = max(c[0] for c in m.calls)
    assert widest <= 4, (
        f"largest batch was {widest}; budget 400 / (50+50) allows 4"
    )
    assert widest >= 1


def test_batched_generate_budget_does_not_split_when_unnecessary(lib) -> None:
    """A generous budget must leave whole length-buckets intact, or batching is lost."""
    torch = pytest.importorskip("torch")
    m = _fake_lm([100], torch)
    lib["batched_generate"](m, _fake_tok(torch), ["a" * 10] * 12, max_new=6,
                            max_batch_tokens=4096, verbose=False)
    assert max(c[0] for c in m.calls) == 12, "one bucket should stay one batch"


def test_nonlinear_probe_is_strong_enough_to_challenge_the_headline(lib) -> None:
    """The nonlinear probe must be able to FIND curvature, or it rigs the test.

    It exists to challenge this project's headline ("training does not change what
    the state contains") with the alternative no linear measurement can rule out:
    that training encodes the same content NONLINEARLY. A weak probe would report
    "no nonlinear lift" and falsely confirm the headline. Three designs were
    rejected on exactly this ground -- MLP(64) reached 0.42 on a CLEAN LINEAR
    target, RBF kernel ridge 0.02 with a positive null, PCA-24+poly2 0.36.

    The synthetic is LOW-RANK, matching the real states' participation ratio of
    1.0-4.8 (D48); isotropic synthetics gave two wrong verdicts before that was
    noticed.
    """
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n, d, r = 80, 400, 12
    u = rng.normal(size=(n, r))
    x = u @ rng.normal(size=(r, d)) + rng.normal(size=(n, d)) * 0.05
    z = u[:, 0]

    lin_t = z * 3 + rng.normal(size=n) * 0.05
    quad_t = (z ** 2) * 3 + rng.normal(size=n) * 0.05

    nl, _ = lib["cv_r2_nonlinear"](x, lin_t)
    assert nl > 0.5, f"probe cannot even recover a linear latent: {nl:.3f}"

    lq, _ = lib["cv_r2"](x, quad_t, alpha=1.0)
    nq, _ = lib["cv_r2_nonlinear"](x, quad_t)
    assert nq > lq + 0.3, (
        f"probe finds no curvature the linear one misses: {nq:.3f} vs {lq:.3f}"
    )

    _, null = lib["cv_r2_nonlinear"](x, rng.normal(size=n), n_null=5)
    assert max(null) < 0.0, f"permutation null is not negative: {max(null):+.3f}"


def test_every_kernel_main_py_is_current_with_its_body_and_blocks() -> None:
    """A committed `main.py` must equal what `build_kernel` produces from its
    `body.py` plus the shared blocks it declares.

    main.py IS THE FILE KAGGLE RUNS. body.py is only its source, so a rebuild
    that never happened means the experiment on the GPU is not the experiment
    in the repo -- and the divergence is invisible, because both files are
    committed and each looks fine on its own.

    Both stale bundles found on 2026-08-07 show why this needs to be automatic:

      kaggle_promptdepth  commit ffb42fc ("restore default max_batch_tokens")
                          edited body.py and did NOT rebuild, so body.py asked
                          for the default while main.py still passed
                          max_batch_tokens=384. The revert was half-applied.
      kaggle_caesar2      main.py was built BEFORE the activation budget was
                          added to kernel_common (cc4a5a3) and never rebuilt,
                          so this not-yet-launched bundle still carried the
                          unbudgeted `for idxs in buckets.values()` generator --
                          the exact configuration that OOM'd geometry-prompt-depth.

    Compares in memory via `build()`; never writes. `scripts/build_kernel.py
    --check <bundle>` is the same check as a command.
    """
    from scripts.build_kernel import build

    root = pathlib.Path(__file__).resolve().parent.parent
    stale = []
    for body in sorted(root.glob("scratch/kaggle_*/body.py")):
        bundle = body.parent
        main_py = bundle / "main.py"
        if not main_py.exists():
            stale.append(f"{bundle.name}: body.py exists but main.py was never built")
            continue
        if build(str(bundle)) != main_py.read_text(encoding="utf-8"):
            stale.append(f"{bundle.name}: main.py is STALE vs body.py + shared blocks")

    assert not stale, (
        "rebuild these with `python -m scripts.build_kernel <bundle>` -- the file "
        "Kaggle runs no longer matches its source:\n  " + "\n  ".join(stale)
    )


def _fake_lm_enforcing_huginn_state_contract(torch):
    """A fake that enforces the shape contract the REAL model imposes.

    Read from `raven_modeling_minimal.py` at the pinned revision
    bb6621b65e90b6a4b9b29ef88dc83866d450470c:

        iterate_forward (line 736):
            x = xk = self.initialize_state(input_embeds, scale=init_scale) \
                     if input_states is None else input_states.clone()
        initialize_state (line 803):
            x = torch.randn_like(input_embeds)

    So the recurrent state REPLACES a tensor shaped like `input_embeds`, i.e.
    [batch, seq_len, dim]. Passing a state whose seq_len disagrees with the
    input is a shape error in the real model. This fake raises the same way, so
    the incompatibility is reproducible on CPU with no GPU and no download.
    """
    import types

    class M:
        def __init__(self):
            self.calls = []
            self.states_seen = []

        def parameters(self):
            yield torch.zeros(1)

        def __call__(self, input_ids=None, num_steps=None, input_states=None, **kw):
            b, n = input_ids.shape
            self.calls.append((b, n))
            if input_states is not None:
                self.states_seen.append(tuple(input_states.shape))
                if input_states.shape[1] != n:
                    raise RuntimeError(
                        "shape mismatch: input_states has seq_len "
                        f"{input_states.shape[1]} but input_embeds has {n}"
                    )
            logits = torch.full((b, n, 70000), -1e9)
            logits[:, -1, 100] = 0.0
            return types.SimpleNamespace(
                logits=logits, latent_states=torch.zeros(b, n, 8)
            )

    return M()


def test_continuous_compute_is_incompatible_with_the_uncached_loop(lib) -> None:
    """`continuous_compute=True` cannot work in `batched_generate`, by construction.

    batched_generate is deliberately UNCACHED -- C11: "it re-runs the full
    GROWING sequence every step" -- so at generated token k the input is
    [b, prompt_len + k]. But the warm-start it passes is
    `lat[:, -1:, :]`, i.e. [b, 1, dim]. The real model REPLACES a
    [batch, seq_len, dim] tensor with that, so the two disagree from the second
    generated token onward, at any `max_batch_tokens`.

    Huginn's own generators do pass `latent_states[:, -1:, :]` (raven lines
    1203/1469/1488/1566) -- but inside a KV-CACHED decode loop, where each step
    feeds exactly one new token and the shapes agree. The cached path is the one
    D62 found returns empty output and never diagnosed, so neither route to
    continuous_compute currently works.

    This matters beyond the helper: `geometry-prompt-depth` stage 2 calls
    run_cell(..., cc=True), so that arm would crash, and directions.md B4.12
    records continuous_compute as the architecture's distinctive feature that no
    kernel has used. It has still not been used.
    """
    torch = pytest.importorskip("torch")
    m = _fake_lm_enforcing_huginn_state_contract(torch)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        lib["batched_generate"](m, _fake_tok(torch), ["aaa", "bbb"], max_new=4,
                                verbose=False, continuous_compute=True)
    assert m.states_seen, "no input_states was ever passed; the arm did not engage"
    assert m.states_seen[0][1] == 1, (
        f"expected a [b, 1, dim] warm-start, got {m.states_seen[0]}"
    )


def test_the_plain_arm_is_unaffected_by_that_contract(lib) -> None:
    """The control: without continuous_compute no state is passed, so the same
    fake generates happily. Isolates the failure to the cc path alone."""
    torch = pytest.importorskip("torch")
    m = _fake_lm_enforcing_huginn_state_contract(torch)
    out = lib["batched_generate"](m, _fake_tok(torch), ["aaa", "bbb"], max_new=4,
                                  verbose=False, continuous_compute=False)
    assert len(out) == 2
    assert not m.states_seen, "plain arm must not pass input_states"


# --- preflight: each guard tested against the REAL defect it would have caught --


def test_attainable_rejects_the_exact_geomcorrect_defect(lib) -> None:
    """n_perm=200 vs Bonferroni alpha 0.05/12 made rejection impossible.

    Caught by hand before that kernel ran; this is the same arithmetic as a guard,
    so it cannot recur silently. Measured power at 200 permutations was 0.00 even
    for a 2 sd shift, because the p-floor 1/201 = 0.00498 exceeds alpha = 0.00417.
    """
    ok, floor = lib["attainable"](0.05 / 12, 200)
    assert not ok and floor == pytest.approx(1 / 201)
    ok5k, floor5k = lib["attainable"](0.05 / 12, 5000)
    assert ok5k and floor5k == pytest.approx(1 / 5001)


def test_attainable_is_not_vacuously_permissive(lib) -> None:
    """A guard that always passes is worse than none."""
    assert not lib["attainable"](0.001, 100)[0]
    assert lib["attainable"](0.05, 1000)[0]


def test_degenerate_flags_the_exact_prefill_failure(lib) -> None:
    """geometry-discourse's prefill arm: byte fragments for 24/24 items.

    Token ids 6704/7909/12894 decode to partial UTF-8 ('ä¸'), rendering U+FFFD.
    The kernel printed "(B) DEGRADATION: D68 interpretation WRONG" from that arm.
    This guard makes that arm unreadable instead of authoritative.
    """
    bad, why = lib["degenerate"](["�"] * 24)
    assert bad and "unprintable" in why


def test_degenerate_flags_a_collapsed_argmax(lib) -> None:
    """The other shape: every item predicts the same token, so the measurement
    carries no per-item information even though it looks like real text."""
    bad, why = lib["degenerate"](["The"] * 24)
    assert bad and "collapsed" in why


def test_degenerate_passes_a_healthy_population(lib) -> None:
    """The constrained arm at r=64 -- real digits, several distinct. Must pass,
    or the guard would have suppressed the run's actual finding."""
    bad, why = lib["degenerate"](["1"] * 8 + ["5"] * 5 + ["8"] * 3 + ["0"] * 8)
    assert not bad, why


def test_gated_verdict_withholds_when_the_instrument_fails(lib, capsys) -> None:
    """D62's lesson, executable: a conclusion may not be printed over an arm that
    did not qualify. geometry-discourse printed the OPPOSITE of the right answer
    because P3/P4 keyed on one arm with no sanity gate."""
    msg = lib["gated_verdict"](
        "prefill recovers the answer", True,
        [("output sanity", False, "24/24 argmax are byte fragments"),
         ("threshold attainable", True, "")])
    assert "WITHHELD" in msg and "byte fragments" in msg
    assert "CONFIRMED" not in msg
    assert "WITHHELD" in capsys.readouterr().out


def test_gated_verdict_reports_when_every_gate_passes(lib) -> None:
    ok = lib["gated_verdict"]("constrained restores the answer at r=64", True,
                              [("output sanity", True, ""), ("control", True, "")])
    assert "CONFIRMED" in ok and "WITHHELD" not in ok
    bad = lib["gated_verdict"]("depth degrades the computation", False,
                               [("output sanity", True, "")])
    assert "REFUTED" in bad


def test_no_kernel_body_seeds_items_with_pythons_salted_hash() -> None:
    """`hash(str)` is salted PER PROCESS, so seeding item generation with it makes
    the item set differ on every run.

    Measured: hash("echo_digit") % 997 gave 544, 92 and 779 in three interpreters,
    so geometry-graded-readout and geometry-discourse drew DIFFERENT items and the
    same nominal cell read 96% in one and 83% in the other. Within-run comparisons
    survive; cross-run ones do not. Kernels that have already RUN are exempt --
    their body is the record of what executed, and rewriting it would falsify that.
    """
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for body in sorted(root.glob("scratch/kaggle_*/body.py")):
        if (body.parent / "out").exists():
            continue                      # already ran; frozen record
        src = body.read_text(encoding="utf-8")
        for m in re.finditer(r"random\.Random\([^)]*\bhash\(", src):
            offenders.append(f"{body.parent.name}: {src[m.start():m.start() + 60]!r}")
    assert not offenders, (
        "seed item generation with a stable digest (zlib.crc32) -- Python salts "
        "str hashes per process, so these kernels are not reproducible:\n  "
        + "\n  ".join(offenders))


def test_dynamic_range_flags_the_exact_capgraded_ceiling(lib) -> None:
    """geometry-cap-graded's content probe saturated in BOTH arms.

    Real values: trained R2 0.9691 / 0.9964 / 0.9945 / 0.8709, untrained 0.9301 /
    0.9966 / 1.0000 / 1.0000. Every cell >= 0.87 and the untrained arm hit exactly
    1.0000 twice, so the trained-minus-untrained gap could only span
    [-0.129, +0.039] -- and a Spearman over those four slivers printed +0.95
    CONFIRMED. That is D63's "a ratio of noise is not a confirmation" again.
    """
    real = [0.9691, 0.9964, 0.9945, 0.8709, 0.9301, 0.9966, 1.0000, 1.0000]
    ok, why = lib["has_dynamic_range"](real, name="content R2")
    assert not ok and "CEILING" in why


def test_dynamic_range_passes_a_measure_that_can_actually_move(lib) -> None:
    """Must not fire on a healthy spread, or it would suppress real findings."""
    ok, why = lib["has_dynamic_range"]([0.10, 0.42, 0.66, 0.88], name="content R2")
    assert ok, why


def test_dynamic_range_also_flags_a_floor(lib) -> None:
    ok, why = lib["has_dynamic_range"]([0.001, 0.004, 0.002], name="probe")
    assert not ok and "FLOOR" in why
