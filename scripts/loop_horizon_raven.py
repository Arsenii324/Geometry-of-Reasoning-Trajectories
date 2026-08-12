"""The loop-horizon measurements, repeated on the REAL Huginn architecture.

WHY. `scripts/loop_horizon.py` refuted the idea that contraction starves early loops of
gradient, and explained it: a normalised residual loop settles in the forward direction
while staying near-isometric in the backward one. That was measured on a 32-dimensional
toy with no attention, no tokens and a quadratic loss. Before any of it is quoted about
Huginn, it has to run on Huginn's own classes.

WHAT THIS IS. `local_smoke.py::build()` instantiates the released `RavenForCausalLM` at
toy width (~28M params) from `raven_modeling_minimal.py`. Every code path here is the
genuine one -- `initialize_state`, `adapter(cat[x, e])`, the `core_block` stack, `coda`,
`ln_f`, `lm_head` -- only the dimensions are small. Weights are RANDOM: this tests what
the ARCHITECTURE does, which is the structural claim. It says nothing about what training
does to it, and the report must not let it.

ONE THING READ OUT OF THE SOURCE, WHICH MATTERS MORE THAN THE REST.
`iterate_forward` (raven_modeling_minimal.py:746-760) runs

    with torch.no_grad():
        for no_grad_step in range(num_steps_no_grad):  ...
    for grad_step in range(num_steps_with_grad):       ...

so **Huginn's truncated backprop is in the released code, not just in a training note**:
loops outside the final window receive exactly zero gradient, not merely a small one.
`num_steps` may be passed as a pair to control the split, which is what P5 exploits.

PRE-REGISTERED PREDICTIONS, written before the first run (CLAUDE.md §1):

  P0  The no-grad / with-grad split behaves as read: passing (T, 0) leaves the core block
      with no gradient, passing (0, T) gives it gradient. Instrument check; if this fails
      the rest is meaningless.
  P1  rho_step < 1 -- the state settles across loops.
  P2  rho_jac, the spectral radius of one loop's Jacobian by power iteration on JVPs,
      sits AT OR JUST ABOVE 1, materially above rho_step. This is the toy's central
      finding and the one most likely to be an artefact of the toy.
  P3  ||dL/dh_t|| is roughly flat in t under full backprop: gradient reaches every loop.
  P4  Per-loop KL(p_t || p_T) through the real readout falls towards zero well before the
      last loop, giving an "effective depth" smaller than T. (At random init this is a
      demonstration that the diagnostic works, not a claim about a trained model.)
  P5  Truncation bites where contraction does not: with num_steps=(T-k, k), loops outside
      the window have exactly zero gradient, so the effective count tracks k rather than
      any property of the dynamics.

  FALSIFIER for the report's central claim: if rho_jac comes out clearly below 1 and
  ||dL/dh_t|| decays geometrically, the toy misled me and the conclusion re-opens.

RESULT, 2026-08-12, 27.7M random-init, T=32, float32. The CONCLUSION transfers; the
MECHANISM I gave for it does not.

  P0 PASS  num_steps=(32,0) -> 0 loops carry gradient; (0,32) -> 32. Huginn's truncated
           backprop is a code path in the released model, not only a training note.
  P1 PASS  rho_step = 0.9963: the state settles.
  P2 FAIL  rho_jac = 0.9375 (stable: 0.9375 at eps=1e-3, 0.9739 at 1e-2, 3.9% apart) --
           BELOW rho_step, not at-or-above 1. The toy's near-isometry does NOT hold here.
  P3 PASS  and this is what matters: ||dL/dh_t|| spans only 4.15x across all 32 loops
           (early ~0.003, late ~0.009). Gradient reaches every loop.
  P4 PASS  per-loop KL through the real readout falls 3.32 -> 0.37 over the first 8 loops
           and keeps falling; the diagnostic resolves a curve on the genuine decode path.
  P5 PASS  with num_steps=(T-k, k), exactly k loops carry gradient and the spread grows
           with k (1.00, 1.14, 1.40, 2.06, 3.44, 6.08 for k=1..32). Truncation is a hard
           cutoff; contraction is a soft one.

WHAT TO CARRY, STATED CAREFULLY. Early loops are not starved of gradient -- 4.15x over 32
loops is mild -- so "loops saturate because early loops get no credit" stays refuted. But
the reason is NOT that the map is isometric, as the toy suggested; it is that the per-loop
Jacobian is only *mildly* contractive (0.94), and 0.94^32 is a factor of a few, not a
vanishing. The durable cross-check is the one both experiments agree on: **rho_step and
rho_jac are different quantities and do not even disagree in a consistent direction** --
the toy had rho_jac above rho_step, the real architecture has it below. Neither can be
inferred from the other, and this project only ever measured rho_step on Huginn.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types

import torch

SRC = pathlib.Path(
    "/Users/a2mogus/build-projs/huginn-load/01-huginn-recurrent-depth_2502.05171"
    "/code/recurrent-pretraining/recpre"
)


def load_real_classes():
    if not SRC.exists():
        raise SystemExit(f"released source not found at {SRC}")
    pkg = types.ModuleType("recpre")
    pkg.__path__ = [str(SRC)]
    sys.modules["recpre"] = pkg

    def load(name):
        spec = importlib.util.spec_from_file_location(f"recpre.{name}", SRC / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"recpre.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    return load("raven_config_minimal"), load("raven_modeling_minimal")


def build(n_embd=64, n_heads=4, n_layers=8, vocab=512):
    rc, rm = load_real_classes()
    cfg = rc.RavenConfig(n_embd=n_embd, n_heads=n_heads, n_layers=n_layers,
                         block_size=256, vocab_size=vocab,
                         effective_expected_depth=4, mean_recurrence=4)
    return rm.RavenForCausalLM(cfg), cfg


def geo_mean_ratio(vals, lo_frac=0.25, hi_frac=1.0):
    v = [x for x in vals if x is not None and x == x and x > 0]
    if len(v) < 4:
        return float("nan")
    a, b = int(len(v) * lo_frac), int(len(v) * hi_frac)
    r = [v[i + 1] / v[i] for i in range(a, min(b, len(v)) - 1) if v[i] > 0]
    r = [x for x in r if x > 0]
    if not r:
        return float("nan")
    return math.exp(sum(math.log(x) for x in r) / len(r))


def capture_args(model, ids, T):
    """Run a real forward, recording core_block_forward's auxiliary arguments and each h_t."""
    orig = model.core_block_forward
    grab = {"args": None}
    states = []

    def wrapped(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step):
        if grab["args"] is None:
            grab["args"] = (input_embeds, freqs_cis, mask, past_key_values,
                            block_idx.detach().clone(), current_step)
        out = orig(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step)
        states.append(out[0].detach().clone())
        return out

    model.core_block_forward = wrapped
    with torch.no_grad():
        model(input_ids=ids, num_steps=T)
    model.core_block_forward = orig
    return grab["args"], states


def rho_jacobian(model, aux, h, n_iter=40, rel_eps=1e-3):
    """Spectral radius of one loop's Jacobian, by power iteration with finite differences.

    JVP is unavailable here: the released block uses scaled-dot-product attention, whose CPU
    kernel has no double-backward, which is what torch's jvp needs. Finite differences go
    through the real block unchanged and need no autograd at all. Stability is checked by
    re-running at a different step size (see caller) rather than assumed.
    """
    inp_e, freqs, mask, pkv, blk, step = aux

    def f(x):
        with torch.no_grad():
            out, _ = model.core_block_forward(x, inp_e, freqs, mask, None, blk.clone(), step)
        return out

    base = f(h)
    eps = rel_eps * h.norm().item() / math.sqrt(h.numel())
    v = torch.randn_like(h)
    v = v / v.norm()
    lam = float("nan")
    for _ in range(n_iter):
        jv = (f(h + eps * v) - base) / eps
        n = jv.norm().item()
        if n < 1e-30:
            return 0.0
        v = jv / n
        lam = n
    return lam


def decode(model, latent):
    """The model's own readout path (raven_modeling_minimal.py:884-892)."""
    x = model.transformer.ln_f(latent)
    for block in model.transformer.coda:
        x = block(x, model.freqs_cis[:, : x.shape[1]], torch.tensor(0), None, None)
    x = model.transformer.ln_f(x)
    return model.lm_head(x).float()


def backward_profile(model, ids, T, split=None):
    """||dL/dh_t|| for every loop, under a chosen no-grad/with-grad split."""
    orig = model.core_block_forward
    hs = []

    def wrapped(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step):
        out = orig(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step)
        h = out[0]
        if h.requires_grad:
            h.retain_grad()
            hs.append(h)
        return out

    model.core_block_forward = wrapped
    steps = torch.tensor([0, T]) if split is None else torch.tensor(list(split))
    out = model(input_ids=ids, num_steps=steps)
    logits = out.logits if hasattr(out, "logits") else out[0]
    logits.float().pow(2).mean().backward()
    model.core_block_forward = orig
    return [h.grad.norm().item() for h in hs if h.grad is not None]


def main() -> int:
    torch.manual_seed(0)
    model, cfg = build()
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    T = 32
    ids = torch.randint(0, 500, (1, 16))
    print("=" * 78)
    print(f"real RavenForCausalLM, {n_par/1e6:.1f}M params, n_embd={cfg.n_embd}, "
          f"T={T} loops, RANDOM WEIGHTS, float32")
    bad = 0

    # ---------------------------------------------------------------- P0
    print("\nP0  the no-grad / with-grad split in iterate_forward behaves as read")
    b_nograd = backward_profile(model, ids, T, split=(T, 0))
    b_full = backward_profile(model, ids, T, split=(0, T))
    ok0 = len(b_nograd) == 0 and len(b_full) == T
    print(f"    num_steps=(T,0): {len(b_nograd)} loops carry gradient (expect 0)")
    print(f"    num_steps=(0,T): {len(b_full)} loops carry gradient (expect {T})")
    print(f"    {'PASS' if ok0 else 'FAIL'} -- truncation is a real code path, not a training note")
    bad += not ok0

    # ---------------------------------------------------------------- P1
    aux, states = capture_args(model, ids, T)
    steps = [(states[i + 1] - states[i]).norm().item() for i in range(len(states) - 1)]
    rho_step = geo_mean_ratio(steps)
    print(f"\nP1  forward settling:  rho_step = {rho_step:.4f}")
    ok1 = rho_step == rho_step and rho_step < 1.0
    print(f"    step norms: " + " ".join(f"{s:.3g}" for s in steps[:6]) + " ...")
    print(f"    {'PASS' if ok1 else 'FAIL'} (need < 1: the state settles)")
    bad += not ok1

    # ---------------------------------------------------------------- P2
    rj = rho_jacobian(model, aux, states[-2], rel_eps=1e-3)
    rj_b = rho_jacobian(model, aux, states[-2], rel_eps=1e-2)
    print(f"\nP2  one-loop Jacobian spectral radius:  rho_jac = {rj:.4f}")
    print(f"    stability: eps=1e-3 -> {rj:.4f}   eps=1e-2 -> {rj_b:.4f}   "
          f"(differ by {abs(rj-rj_b)/rj:.1%})")
    print(f"    rho_jac / rho_step = {rj / rho_step:.2f}")
    ok2 = rj > rho_step
    print(f"    {'PASS' if ok2 else 'FAIL'} (toy predicted rho_jac at or just above 1, "
          f"materially above rho_step)")
    bad += not ok2

    # ---------------------------------------------------------------- P3
    rho_bwd = geo_mean_ratio(b_full[::-1])
    lo, hi = min(b_full), max(b_full)
    print(f"\nP3  backward sensitivity ||dL/dh_t|| under FULL backprop")
    print(f"    max/min across {T} loops = {hi/lo:.2f}x   rho_bwd = {rho_bwd:.4f}")
    print(f"    first 6 (loop 0..5): " + " ".join(f"{v:.3g}" for v in b_full[:6]))
    print(f"    last 6            : " + " ".join(f"{v:.3g}" for v in b_full[-6:]))
    ok3 = hi / lo < 50
    print(f"    {'PASS' if ok3 else 'FAIL'} (need < 50x spread: gradient reaches every loop)")
    bad += not ok3

    # ---------------------------------------------------------------- P4
    print(f"\nP4  per-loop KL(p_t || p_T) through the model's own readout")
    with torch.no_grad():
        lg = [decode(model, s) for s in states]
        pT = torch.log_softmax(lg[-1], -1)
        kls = []
        for l in lg:
            p = torch.log_softmax(l, -1)
            kls.append(torch.nn.functional.kl_div(pT, p, log_target=True,
                                                  reduction="batchmean").abs().item())
    eps = 0.01 * max(kls)
    eff = next((i for i, k in enumerate(kls) if k <= eps), len(kls))
    print("    " + " ".join(f"{k:.3g}" for k in kls[:8]) + " ...")
    print(f"    effective depth (first loop within 1% of final) = {eff} of {T}")
    ok4 = eff < T
    print(f"    {'PASS' if ok4 else 'FAIL'} (diagnostic resolves an effective depth below T)")
    bad += not ok4

    # ---------------------------------------------------------------- P5
    print(f"\nP5  truncation bites where contraction does not")
    print(f"    {'k':>4} {'loops with gradient':>21} {'max/min':>9}")
    for k in (1, 2, 4, 8, 16, T):
        b = backward_profile(model, ids, T, split=(T - k, k))
        sp = (max(b) / min(b)) if b else float("nan")
        print(f"    {k:>4} {len(b):>21} {sp:>9.2f}")
    print("    loops outside the window receive exactly zero gradient, not a small one.")

    print("\n" + "=" * 78)
    print(f"{'ALL GATES PASSED' if bad == 0 else str(bad) + ' GATE(S) FAILED'}")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
