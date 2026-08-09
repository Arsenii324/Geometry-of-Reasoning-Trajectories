"""CPU dry-run of the causal-patching kernel, before it touches a GPU.

WHY THIS IS THE ONE KERNEL THAT MUST BE VERIFIED MECHANICALLY, NOT JUST STATICALLY.
Every other kernel in this project reads out states and reports them; this one
MUTATES the forward pass mid-computation via a hook return value, and D47's whole
failure was trusting an intervention that never reached anything. So the tests here
check the mechanism itself: a hook that returns a modified tensor must (a) leave
every unroll BEFORE the patch point untouched, (b) actually change the state at and
after the patch point, and (c) `find_donor_recipient` must be able to tell a
seed-correct fake model's classes apart at all, or the whole design has no floor to
stand on.

The fake enforces the contract the real model imposes: `core_block[-1]` fires once
per unroll and its return value, if the hook overrides it, becomes the state fed
into the NEXT unroll -- exactly the propagation the patching design depends on.
"""

from __future__ import annotations

import ast
import os
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BODY = os.path.join(ROOT, "scratch", "kaggle_patch", "body.py")


def _load_body() -> dict:
    if not os.path.exists(BODY):
        pytest.skip("kaggle_patch/body.py not present")
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


def _fake_huginn(torch, n_vocab=6, d=8):
    # n_vocab SMALL on purpose: rank 1 by chance is ~1/6 here, so a 40-seed search
    # reliably finds both classes without the test depending on a lucky draw --
    # the real model's vocabulary (32768) makes rank 1 genuinely rare, which is
    # exactly why the real kernel restricts itself to prompts geomcap already
    # measured near the boundary (D90) rather than searching blind.
    """A contraction whose FINAL sign carries the seed's influence forward, so
    correctness is genuinely seed-dependent -- unlike the geomcap fake, which
    converges to the same fixed point regardless of h_0 and so could never test
    `find_donor_recipient`."""
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
            h = torch.randn(b, n, d)                    # Huginn's initialize_state
            for _ in range(num_steps):
                # rho=0.9 contraction that never fully erases h_0's SIGN, so which
                # seed produced h_0 keeps mattering all the way to the read -- the
                # property the real orbit has near a decision boundary (D90).
                h = 0.9 * h + 0.02 * torch.sign(h)
                # ASSIGNED BACK, mirroring `core_block_forward`'s own
                # `x = block(x, ...)`. A hook's patched return value only
                # propagates to later unrolls if the caller actually uses it --
                # dropping this assignment was the first draft's own bug, caught
                # by `test_the_patch_actually_changes_the_state_from_r_onward`
                # failing silently-not-silently (a clear, diagnosable assertion,
                # not a wrong number).
                h = self.transformer.core_block[-1](h)
            return types.SimpleNamespace(logits=self.lm_head(h))

    torch.manual_seed(0)      # fixes lm_head's / core_block's OWN init draw, so
    return M()                # the fake's weights don't depend on test execution
                               # order -- the geomcap fake does the same for the
                               # same reason.


def _fake_tok(torch):
    class T:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "<|u|>" + msgs[0]["content"] + "<|a|>"

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = torch.arange(1, min(len(text), 12) + 1).unsqueeze(0)
            if return_tensors is None:
                return types.SimpleNamespace(input_ids=ids[0].tolist())
            return types.SimpleNamespace(input_ids=ids)

    return T()


@pytest.fixture
def kern(tmp_path, monkeypatch):
    ns = _load_body()
    monkeypatch.setitem(ns, "OUTDIR", str(tmp_path))
    monkeypatch.setitem(ns, "NUM_STEPS", 12)
    return ns


def test_hook_fires_once_per_unroll_and_records_that_many_states(kern) -> None:
    torch = pytest.importorskip("torch")
    model = _fake_huginn(torch)
    r = kern["run_forward"](model, _fake_tok(torch), torch, "p", "2", h0_seed=0)
    assert r["states"].shape[0] == kern["NUM_STEPS"]
    assert len(r["rank_curve"]) == kern["NUM_STEPS"]


def test_seeded_forward_is_reproducible(kern) -> None:
    """The determinism this whole design leans on (D90): same seed, same prompt,
    same weights must give bit-identical states."""
    torch = pytest.importorskip("torch")
    model = _fake_huginn(torch)
    a = kern["run_forward"](model, _fake_tok(torch), torch, "p", "2", h0_seed=7)
    b = kern["run_forward"](model, _fake_tok(torch), torch, "p", "2", h0_seed=7)
    assert a["state_sha"] == b["state_sha"]
    assert a["rank_curve"] == b["rank_curve"]


def test_unpatched_prefix_is_untouched_by_a_later_patch(kern) -> None:
    """THE CAUSAL CLAIM'S OWN PREMISE. Unrolls strictly before the patch point must
    be bit-identical to an unpatched run of the SAME seed -- a patch cannot reach
    backward in time, and if the prefix differed, `post_patch_ranks` would not
    isolate the intervention's effect from an unrelated implementation bug."""
    torch = pytest.importorskip("torch")
    model = _fake_huginn(torch)
    unpatched = kern["run_forward"](model, _fake_tok(torch), torch, "p", "2",
                                    h0_seed=3)
    donor = kern["run_forward"](model, _fake_tok(torch), torch, "p", "2",
                                h0_seed=9)
    r = 5
    patched = kern["run_forward"](model, _fake_tok(torch), torch, "p", "2",
                                  h0_seed=3, patch_r=r, donor_states=donor["states"])
    import numpy as np
    assert np.array_equal(patched["states"][:r], unpatched["states"][:r]), (
        "the patch altered unrolls BEFORE the patch point")
    assert patched["rank_curve"][:r] == unpatched["rank_curve"][:r]


def test_the_patch_actually_changes_the_state_from_r_onward(kern) -> None:
    """And the mirror requirement: something MUST change at and after r, or the
    hook's return value is silently being ignored (the mechanical failure mode
    this whole test file exists to catch, in the spirit of D47)."""
    torch = pytest.importorskip("torch")
    import numpy as np
    model = _fake_huginn(torch)
    unpatched = kern["run_forward"](model, _fake_tok(torch), torch, "p", "2",
                                    h0_seed=3)
    donor = kern["run_forward"](model, _fake_tok(torch), torch, "p", "2",
                                h0_seed=9)
    r = 5
    patched = kern["run_forward"](model, _fake_tok(torch), torch, "p", "2",
                                  h0_seed=3, patch_r=r, donor_states=donor["states"])
    assert np.array_equal(patched["states"][r], donor["states"][r]), (
        "the state AT the patch point must equal the donor's exactly"
    )
    assert not np.array_equal(patched["states"][r + 1], unpatched["states"][r + 1]), (
        "the patch had no effect on the unroll immediately after it"
    )


def test_find_donor_recipient_separates_the_two_classes(kern) -> None:
    """If this fails, no candidate prompt in the real kernel could ever be used --
    the seed search would never find a valid pair, and P1's gate would fire on
    every prompt for a reason having nothing to do with the real model."""
    torch = pytest.importorskip("torch")
    model = _fake_huginn(torch)
    donor, recipient, n_tried = kern["find_donor_recipient"](
        model, _fake_tok(torch), torch, "p", "2", cap=40)
    assert donor is not None and recipient is not None, (
        f"no split found in {n_tried} seeds -- the fake model is not seed-sensitive "
        f"enough for this test to mean anything")
    assert donor["correct"] and not recipient["correct"]
    assert donor["seed"] != recipient["seed"]


def test_find_donor_recipient_gives_up_within_the_cap(kern) -> None:
    """A model whose state never depends on h_0 must not hang -- every seed gives
    the SAME rank, so the search must stop at the cap with at least one side
    still None, whichever class that constant rank happens to fall in."""
    torch = pytest.importorskip("torch")
    model = _fake_huginn(torch)

    # The h_0 draw is consumed (so the RNG stream advances identically to a real
    # call) but then discarded entirely, so `h` -- and therefore every downstream
    # rank -- is IDENTICAL across every seed. No search on this model can ever
    # find both classes.
    def constant_forward(self, input_ids=None, num_steps=None, **kw):
        b, n = input_ids.shape
        torch.randn(b, n, 8)
        h = torch.zeros(b, n, 8) + 1.0
        for _ in range(num_steps):
            h = self.transformer.core_block[-1](h)
        return types.SimpleNamespace(logits=self.lm_head(h))

    model.forward = types.MethodType(constant_forward, model)
    donor, recipient, n_tried = kern["find_donor_recipient"](
        model, _fake_tok(torch), torch, "p", "2", cap=5)
    assert n_tried <= 5
    assert donor is None or recipient is None, (
        "a model with no h_0 dependence somehow produced both classes")


def test_readout_position_matches_the_validated_battery_tail(kern) -> None:
    """Same D71 guard as every other bank: the coda tail must not have drifted."""
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
    b = open(BODY, encoding="utf-8").read()
    assert body_of(b, "coda_head") == body_of(a, "coda_head")


def test_items_match_h0bank_byte_for_byte(kern) -> None:
    """The candidate prompts must be the SAME prompts geomcap ranked, or the
    boundary selection (rank 1-6) is not about these items at all."""
    src = open(os.path.join(ROOT, "scratch", "kaggle_h0bank", "body.py"),
               encoding="utf-8").read().replace("# @needs: run load_arm free_arm", "")
    tree = ast.parse(src)
    tree.body = [n for n in tree.body
                 if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                         and isinstance(n.value.func, ast.Name)
                         and n.value.func.id == "main")]
    ns: dict = {}
    exec(compile(tree, BODY, "exec"), ns)  # noqa: S102
    for c in kern["CANDIDATES"]:
        assert kern["items"](c["family"])[c["item"]] == ns["items"](c["family"])[c["item"]]
