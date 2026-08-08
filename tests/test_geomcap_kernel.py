"""CPU dry-run of the geomcap kernel's banking loop, before it touches a GPU.

WHY A FAKE MODEL AND NOT JUST THE STATIC GUARDS. `tests/test_kernel_tasks.py`
checks that geomcap's generators, readout and depth have not drifted -- all static.
None of that exercises `bank_one`, which is where this kernel's new logic lives: a
forward hook that must fire once per unroll, a record whose `correct` field decides
the capability axis, and an `h0_seed` branch whose whole purpose is to be a
POSITIVE CONTROL. A control that silently does nothing is worse than no control, so
it is tested here against a fake that makes the difference observable.

The fake enforces the contract the real model imposes: `model(input_ids, num_steps)`
runs `core_block[-1]` exactly `num_steps` times, and the state it produces depends
on a `torch.randn` draw -- Huginn's `initialize_state` (D78). That is enough for the
loop under test; nothing here claims anything about Huginn itself.
"""

from __future__ import annotations

import ast
import os
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BODY = os.path.join(ROOT, "scratch", "kaggle_geomcap", "body.py")


def _load_body() -> dict:
    """Exec the whole body with its top-level `main()` call stripped."""
    if not os.path.exists(BODY):
        pytest.skip("kaggle_geomcap/body.py not present")
    src = open(BODY, encoding="utf-8").read().replace(
        "# @needs: run load_arm free_arm", "")
    tree = ast.parse(src)
    tree.body = [n for n in tree.body
                 if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                         and isinstance(n.value.func, ast.Name)
                         and n.value.func.id == "main")]
    ns: dict = {}
    exec(compile(tree, BODY, "exec"), ns)  # noqa: S102
    return ns


def _fake_huginn(torch, n_vocab=64, d=8, n_steps_seen=None):
    """Minimal stand-in with Huginn's hook surface and its h_0 randomness.

    `core_block[-1]` is a real `nn.Module` so `register_forward_hook` behaves as it
    does on the model, and the state carries a `randn` component so two forwards
    differ unless the global seed is set -- exactly the property `h0_seed` exists
    to control.
    """
    import torch.nn as nn

    class Block(nn.Module):
        def forward(self, x, *a, **kw):
            return x

    class Tf(nn.Module):
        def __init__(self):
            super().__init__()
            self.core_block = nn.ModuleList([Block(), Block()])
            self.coda = nn.ModuleList([Block()])

        def ln_f(self, x):
            return x

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Tf()
            self.lm_head = nn.Linear(d, n_vocab, bias=False)
            self.freqs_cis = torch.zeros(1, 512, 2)
            self.device = torch.device("cpu")

        def forward(self, input_ids=None, num_steps=None, **kw):
            b, n = input_ids.shape
            h = torch.randn(b, n, d)                       # Huginn's initialize_state
            for _ in range(num_steps):
                h = 0.9 * h + 0.1 * torch.ones(b, n, d)    # a contraction, like the real map
                self.transformer.core_block[-1](h)         # fires the hook
            if n_steps_seen is not None:
                n_steps_seen.append(num_steps)
            return types.SimpleNamespace(logits=self.lm_head(h))

    torch.manual_seed(0)
    return M()


def _fake_tok(torch):
    class T:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "<|u|>" + msgs[0]["content"] + "<|a|>"

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = torch.arange(1, min(len(text), 40) + 1).unsqueeze(0)
            if return_tensors is None:
                return types.SimpleNamespace(input_ids=ids[0].tolist())
            return types.SimpleNamespace(input_ids=ids)

    return T()


@pytest.fixture
def kern(tmp_path, monkeypatch):
    ns = _load_body()
    monkeypatch.setitem(ns, "OUTDIR", str(tmp_path))
    return ns


def _bank(ns, torch, tag, h0_seed=None, model=None, prompt="what is 1 + 1?"):
    model = model if model is not None else _fake_huginn(torch)
    return ns["bank_one"](model, _fake_tok(torch), torch, tag, "test", "fam", 0,
                          prompt, "2", "3", h0_seed=h0_seed)


def test_hook_fires_once_per_unroll(kern) -> None:
    """The rank curve and the state array must both have NUM_STEPS rows.

    A hook that fires on every core block rather than the last would silently give
    `n_layers * num_steps` rows, and every window-based statistic downstream would
    be computed on a trajectory that does not exist.
    """
    torch = pytest.importorskip("torch")
    rec = _bank(kern, torch, "t0")
    assert rec["ok"], rec.get("why")
    assert rec["shape"][0] == kern["NUM_STEPS"] == 64
    assert len(rec["rank_curve"]) == 64
    assert len(rec["logp"]) == 64


def test_seeded_h0_makes_the_orbit_reproducible(kern) -> None:
    """The `fix` block's premise, made observable.

    If this fails the determinism control cannot detect anything: `fix` would
    report distinct hashes whether or not h_0 is the only stochastic input, and
    the ceiling in `rep` would have no established cause.
    """
    torch = pytest.importorskip("torch")
    model = _fake_huginn(torch)
    a = _bank(kern, torch, "s1", h0_seed=99, model=model)
    b = _bank(kern, torch, "s2", h0_seed=99, model=model)
    assert a["ok"] and b["ok"]
    assert a["state_sha"] == b["state_sha"]
    assert a["rank_curve"] == b["rank_curve"]


def test_unseeded_h0_does_not_repeat_itself(kern) -> None:
    """The `rep` block's premise: without a seed, two forwards must DIFFER.

    This is the failure that would quietly invalidate the whole ceiling
    measurement -- a within-prompt variance of exactly zero reported as a finding
    when it was really the harness removing the variation.
    """
    torch = pytest.importorskip("torch")
    model = _fake_huginn(torch)
    shas = {_bank(kern, torch, f"u{k}", model=model)["state_sha"] for k in range(4)}
    assert len(shas) == 4, "unseeded forwards returned identical states"


def test_a_failed_item_is_recorded_not_raised(kern) -> None:
    """One bad item must not abandon the other 607 orbits mid-run.

    Kaggle gives no second chance at an 80-minute GPU run, and the manifest is
    rewritten after every item precisely so a crash costs one record, not the run.
    """
    torch = pytest.importorskip("torch")

    class Boom:
        device = torch.device("cpu")

        def __getattr__(self, k):
            raise RuntimeError("no model")

    rec = kern["bank_one"](Boom(), _fake_tok(torch), torch, "bad", "test", "fam", 0,
                           "p", "g", "d")
    assert rec["ok"] is False
    assert "RuntimeError" in rec["why"]
    assert "traceback" in rec


def test_the_read_position_is_the_one_that_predicts_the_answer(kern) -> None:
    """Geometry and rank must be read at the SAME index, n_tokens - 1.

    B6's first attempt read them one token apart, so its geometry described a
    state that never produced the scored logits. The two are taken from one
    indexed row here; this pins that they are the same row.
    """
    fn = next(n for n in ast.walk(ast.parse(open(BODY, encoding="utf-8").read()))
              if isinstance(n, ast.FunctionDef) and n.name == "bank_one")
    hook = next(n for n in ast.walk(fn)
                if isinstance(n, ast.FunctionDef) and n.name == "hook")
    subs = {ast.unparse(n.slice).strip("()")
            for n in ast.walk(hook) if isinstance(n, ast.Subscript)}
    assert "0, n_p - 1, :" in subs, f"state read at {subs}"
    assert "0, n_p - 1" in subs, f"logits read at {subs}"


def test_n_tokens_is_the_prompt_length_actually_forwarded(kern) -> None:
    """`n_tokens` is P5's covariate, so it must be the tokenised length, not len(prompt)."""
    torch = pytest.importorskip("torch")
    short = _bank(kern, torch, "short", prompt="hi")
    long = _bank(kern, torch, "long", prompt="x" * 30)
    assert short["n_tokens"] < long["n_tokens"]
    assert short["n_tokens"] == len("<|u|>" + "hi" + "<|a|>")
