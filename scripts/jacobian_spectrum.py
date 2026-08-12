"""The Jacobian-spectrum instrument, validated locally against an exact eigendecomposition.

WHY. D31 measured the recurrence's Jacobian spectrum on the real model and reported only the
magnitudes: rho = 0.7935 / 0.8042 / 0.8083. D55 later pulled the eigenvalue *arguments* out of the
same banked log at zero further compute and found the leading eigenvalue complex in 3 of 3 prompts,
rotation period 2.6-6.0 unrolls, surviving ~4 turns. D55 named its own follow-up: **a proper
multi-prompt sweep**, never run. This module is the shared instrument for that sweep, and the
kernel in `scratch/ds_jacspec/` is the sweep.

WHY IT NEEDS A LOCAL GATE. The measurement is ARPACK on a matrix-free operator built from autodiff
through the real block. Nothing about that is self-checking: ARPACK will happily return a converged
answer to the wrong operator. At the mini model's width the Jacobian is 64x64, so it can be built
DENSELY -- one JVP per basis vector -- and eigendecomposed exactly. That gives a reference the
matrix-free path must reproduce. At 5280 dimensions no such reference exists, so it has to be
established here or not at all.

METHOD, unchanged from the original kernel (`scratch/kaggle_jacobian/main.py`):
  * perturb and read at the LAST position only, giving the diagonal block whose eigenvalues are
    the recurrent dynamics at the answer token;
  * forward-mode AD where available, since torch's efficient SDPA kernel has no forward-AD rule,
    force the MATH backend; fall back to reverse mode, where eig(J^T) = eig(J) exactly and only
    the eigenvectors would differ, which we do not use.

A complex eigenvalue means the map rotates, at a period of 2*pi/|arg| unrolls. That is the number
the sweep is after.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types

import numpy as np
import torch

SRC = pathlib.Path(
    "/Users/a2mogus/build-projs/huginn-load/01-huginn-recurrent-depth_2502.05171"
    "/code/recurrent-pretraining/recpre"
)


def load_real_classes():
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


def build_mini(n_embd=64, n_heads=4, n_layers=8, vocab=512, seed=0):
    rc, rm = load_real_classes()
    torch.manual_seed(seed)
    cfg = rc.RavenConfig(n_embd=n_embd, n_heads=n_heads, n_layers=n_layers, block_size=256,
                         vocab_size=vocab, effective_expected_depth=4, mean_recurrence=4)
    return rm.RavenForCausalLM(cfg).eval(), cfg


# --------------------------------------------------------------------------- operator
def make_matvec(step_fn, hs, prefer_forward=True):
    """v -> J v at the last position. Returns (matvec, mode)."""
    dim = hs.shape[-1]

    def _fwd(v):
        vt = torch.zeros_like(hs)
        vt[0, -1, :] = torch.as_tensor(np.asarray(v, dtype=np.float32),
                                       device=hs.device, dtype=hs.dtype)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            _, jv = torch.func.jvp(step_fn, (hs,), (vt,))
        return jv[0, -1, :]

    def _rev(v):
        u = torch.zeros_like(hs)
        u[0, -1, :] = torch.as_tensor(np.asarray(v, dtype=np.float32),
                                      device=hs.device, dtype=hs.dtype)
        x = hs.detach().clone().requires_grad_(True)
        y = step_fn(x)
        (g,) = torch.autograd.grad(y, x, grad_outputs=u)
        return g[0, -1, :]

    mode = "forward"
    if prefer_forward:
        try:
            _fwd(np.zeros(dim, dtype=np.float32))
        except Exception:
            mode = "reverse"
    else:
        mode = "reverse"
    fn = _fwd if mode == "forward" else _rev

    def matvec(v):
        return fn(v).detach().float().cpu().numpy().astype(np.float64)

    return matvec, mode


def dense_jacobian(matvec, dim):
    """Build J (or J^T) column by column. Only affordable at mini width."""
    J = np.zeros((dim, dim))
    e = np.zeros(dim)
    for i in range(dim):
        e[:] = 0.0
        e[i] = 1.0
        J[:, i] = matvec(e)
    return J


def spectrum(matvec, dim, k=12, tol=1e-6, maxiter=600):
    from scipy.sparse.linalg import LinearOperator, eigs
    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    vals = eigs(op, k=k, which="LM", return_eigenvectors=False, tol=tol, maxiter=maxiter)
    return vals[np.argsort(-np.abs(vals))]


def summarise(vals):
    osc = vals[np.abs(vals.imag) > 1e-8]
    lead = vals[0]
    out = {
        "rho": float(np.abs(lead)),
        "lead_real": bool(abs(lead.imag) < 1e-8),
        "n_osc": int(len(osc)),
        "n_eigs": int(len(vals)),
    }
    if len(osc):
        a = float(abs(np.angle(osc[0])))
        out["arg_osc"] = a
        out["rho_osc"] = float(np.abs(osc[0]))
        out["period_unrolls"] = (2 * math.pi / a) if a > 1e-12 else None
        # how many turns the rotating mode survives before contraction kills it
        r = float(np.abs(osc[0]))
        out["turns_to_1pct"] = ((math.log(0.01) / math.log(r)) * a / (2 * math.pi)
                                if 0 < r < 1 else None)
    return out


# --------------------------------------------------------------------------- local gate
def core_step_factory(model, emb, freqs, blk, step):
    def step_fn(x):
        out, _ = model.core_block_forward(x, emb, freqs, None, None, blk.clone(), step)
        return out
    return step_fn


def main() -> int:
    model, cfg = build_mini()
    dim = cfg.n_embd
    ids = torch.randint(0, 500, (1, 12))

    grab = {}
    orig = model.core_block_forward

    def wrapped(x, emb, freqs, mask, pkv, blk, step):
        if "a" not in grab:
            grab["a"] = (emb, freqs, blk.detach().clone(), step)
        out = orig(x, emb, freqs, mask, pkv, blk, step)
        grab["h"] = out[0].detach().clone()
        return out

    model.core_block_forward = wrapped
    with torch.no_grad():
        model(input_ids=ids, num_steps=8)
    model.core_block_forward = orig

    emb, freqs, blk, step = grab["a"]
    hs = grab["h"]
    step_fn = core_step_factory(model, emb, freqs, blk, step)
    matvec, mode = make_matvec(step_fn, hs)

    print("=" * 74)
    print(f"mini Raven, dim={dim}, AD mode = {mode}"
          + ("  (eig(J^T)=eig(J), so the spectrum is unchanged)" if mode == "reverse" else ""))

    print("\nG1  matrix-free ARPACK must reproduce an EXACT dense eigendecomposition")
    J = dense_jacobian(matvec, dim)
    exact = np.linalg.eigvals(J)
    exact = exact[np.argsort(-np.abs(exact))]
    arp = spectrum(matvec, dim, k=min(12, dim - 2))
    n = min(6, len(arp))
    err = max(abs(abs(exact[i]) - abs(arp[i])) for i in range(n))
    print(f"    |lambda| exact vs ARPACK, top {n}: max abs diff = {err:.3e}")
    print("    exact : " + " ".join(f"{abs(v):.4f}" for v in exact[:n]))
    print("    ARPACK: " + " ".join(f"{abs(v):.4f}" for v in arp[:n]))
    ok1 = err < 1e-4
    print(f"    {'PASS' if ok1 else 'FAIL'} (need < 1e-4)")

    print("\nG2  the complex/real verdict must agree between the two routes")
    ce, ca = bool(abs(exact[0].imag) > 1e-8), bool(abs(arp[0].imag) > 1e-8)
    n_e = int((np.abs(exact.imag) > 1e-8).sum())
    print(f"    leading eigenvalue complex?  exact={ce}  ARPACK={ca}")
    print(f"    exact spectrum has {n_e}/{dim} complex eigenvalues")
    ok2 = ce == ca
    print(f"    {'PASS' if ok2 else 'FAIL'}")

    print("\nG3  summary fields, and the quantity the sweep is after")
    s = summarise(arp)
    for k_, v in s.items():
        print(f"    {k_:16s} {v}")
    ok3 = "rho" in s
    print(f"    {'PASS' if ok3 else 'FAIL'}")

    bad = int(not (ok1 and ok2 and ok3))
    print("\n" + "=" * 74)
    print("instrument validated against an exact reference" if not bad else "INSTRUMENT FAILED")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
