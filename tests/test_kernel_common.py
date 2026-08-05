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


