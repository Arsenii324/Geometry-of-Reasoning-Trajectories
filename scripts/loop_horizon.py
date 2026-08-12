"""Does a weight-tied loop's gradient concentrate on the LAST loops, and is the
effective horizon 1/(1-rho)?

WHY THIS EXISTS. `docs/LOOPED_PRETRAIN_TRANSFER.md` §2 argues that in a weight-tied,
contractive loop the weight gradient is dominated by the final loops -- the converged
ones doing the least work -- with an effective count of about 1/(1-rho). That argument
was heuristic. Under this project's own rule (CLAUDE.md §5) a load-bearing claim needs
an instrument that has passed a null, so this is that instrument. It runs on CPU in
seconds and needs no checkpoint.

WHAT IT DOES AND DOES NOT TEST. It tests the *mechanism* -- how gradient mass is
distributed across loops as a function of contraction. It does NOT test the
*consequence*, that a trained model's quality knee sits at 1/(1-rho). Nothing here
involves training. Keep the two separate when quoting it.

PRE-REGISTERED PREDICTIONS, written before the first run (CLAUDE.md §1):

  P1  NULL / instrument check. With contraction removed (rho -> 1) the per-loop
      gradient contributions must be ~uniform, giving N_eff -> T. If a non-contractive
      map still shows concentration, the measurement is an artefact and everything
      else here is void.

  P2  PRIMARY. For a contractive map, g_t ~ rho^(T-t), so
          N_eff := (sum_t g_t) / (max_t g_t)  ==  (1 - rho^T) / (1 - rho).
      Quantitative, not directional: predicted vs measured N_eff should agree to
      within ~25% across the sweep.

  P3  SPECIFICITY / saturation in T. N_eff must saturate as T grows: N_eff ~
      min(T, 1/(1-rho)). If N_eff keeps growing with T at fixed rho, the horizon
      is not set by contraction.

  P4  FLOOR -- the one that bears on our own paper. Our rho (D115) is a *step-size
      decay* fit, not a spectral radius; we never computed rho(J), and that is a
      stated limitation. If the two disagree here, then the rho we measured on Huginn
      is not the quantity that appears in the gradient argument, and §2 does not get
      to use it. Report both and their ratio.

RESULT, 2026-08-12 -- THE PRIMARY PREDICTION IS REFUTED. Recorded here so the file
cannot be quoted as if it had confirmed anything:

  P0 PASS   split-copy attribution reproduces the tied forward exactly (0.000e+00)
            and the tied gradient to 1.3e-23.
  P1 PASS   with an undamped map, gradient mass spreads (N_eff 52.6 of T=64).
  P2 FAIL   badly, for BOTH definitions of rho: median error 89% using step-decay
            rho, 65% using backward-sensitivity rho. The gradient profile is not
            geometric at all.
  P3 partial N_eff does saturate in T (4.99 at T=64 -> 5.01 at T=128), so *a* horizon
            exists -- but it is not the one the formula names.
  P4 CONFIRMS THE WORRY  rho_step ~ 0.97-0.99 while rho_jac ~ 1.00-1.05. They do not
            measure the same object; rho_jac sits AT or ABOVE 1 while rho_step sits
            below it.

WHAT THE REFUTATION MEANS, which is the actually useful part. In a normalised residual
loop the state stops moving (step sizes decay, rho_step < 1) while the Jacobian stays
near-isometric (rho_jac ~ 1). Normalisation renormalises away the norm decay that would
otherwise damp backward signals. So **forward convergence and backward damping are
different phenomena and this architecture has the first without the second**: gradient
mass is distributed almost uniformly across loops (last 32 of 64 hold 53.5% at
Huginn-like scale, against 50% for exactly uniform), with only a spike on the final
loop. Vanishing credit is therefore NOT why such a loop saturates, and "fix credit
assignment" is a lever this experiment removes rather than supports.

ATTRIBUTION METHOD. To read a per-loop gradient out of a *tied* weight we give each
loop its own copy of the block, initialised identically. The forward is then exactly
the tied forward (same numbers), while d L / d W_t isolates loop t's contribution. The
tied gradient is the sum over t. P0 below checks that equivalence bit-for-bit rather
than assuming it.
"""
import math

import torch
import torch.nn as nn

torch.set_grad_enabled(True)
DEV = "cpu"
DTYPE = torch.float64          # fp64: bf16 fakes convergence ~4.6x early (D30)


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class Block(nn.Module):
    """One loop body. Huginn-shaped: residual + MLP + normalisation."""

    def __init__(self, d, hidden, scale, gen):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(hidden, d, generator=gen, dtype=DTYPE) / math.sqrt(d))
        self.w2 = nn.Parameter(torch.randn(d, hidden, generator=gen, dtype=DTYPE) / math.sqrt(hidden))
        self.scale = scale
        self.normed = True

    def forward(self, h):
        f = torch.nn.functional.gelu(h @ self.w1.T) @ self.w2.T
        h = h + self.scale * f
        return rms_norm(h) if self.normed else h


def unroll(blocks, h0, normed=True):
    hs = [h0]
    h = h0
    for b in blocks:
        b.normed = normed
        h = b(h)
        hs.append(h)
    return h, hs


def make_blocks(d, hidden, scale, T, seed):
    """T independent copies with identical weights == the tied loop, but attributable."""
    gen = torch.Generator().manual_seed(seed)
    proto = Block(d, hidden, scale, gen)
    blocks = []
    for _ in range(T):
        b = Block(d, hidden, scale, torch.Generator().manual_seed(0))
        with torch.no_grad():
            b.w1.copy_(proto.w1)
            b.w2.copy_(proto.w2)
        blocks.append(b)
    return blocks


def rho_from_step_decay(hs, tail_frac=0.5):
    """D115's method: geometric mean of ||dh_{t+1}|| / ||dh_t|| over the tail."""
    d = [(hs[i + 1] - hs[i]).norm().item() for i in range(len(hs) - 1)]
    start = int(len(d) * (1 - tail_frac))
    ratios = [d[i + 1] / d[i] for i in range(start, len(d) - 1) if d[i] > 1e-300]
    if not ratios:
        return float("nan")
    return math.exp(sum(math.log(max(r, 1e-300)) for r in ratios) / len(ratios))


def rho_from_jacobian(block, h, n_iter=200):
    """Spectral radius of dh_{t+1}/dh_t at h, by power iteration on JVPs."""
    h = h.detach().clone().requires_grad_(True)
    v = torch.randn_like(h)
    v = v / v.norm()
    lam = float("nan")
    for _ in range(n_iter):
        _, jv = torch.autograd.functional.jvp(lambda x: block(x), (h,), (v,))
        n = jv.norm().item()
        if n < 1e-300:
            return 0.0
        v = (jv / n).detach()
        lam = n
    return lam


def per_loop_grad(blocks, h0, normed=True):
    """Returns (g_t, b_t, hs).

    g_t = ||dL/dW_t||  -- the loop's own contribution to the tied weight gradient.
    b_t = ||dL/dh_t||  -- the backward sensitivity arriving at loop t. This is the
          quantity the product of Jacobians actually damps, and it is what the
          heuristic in LOOPED_PRETRAIN_TRANSFER.md §2 should have been written in
          terms of. The step-size decay rate is a different object (see P4).
    """
    for b in blocks:
        for p in b.parameters():
            if p.grad is not None:
                p.grad = None
    hT, hs = unroll(blocks, h0, normed)
    for h in hs:
        if h.requires_grad:
            h.retain_grad()
    hT.pow(2).mean().backward()
    g = [math.sqrt(sum(p.grad.pow(2).sum().item() for p in b.parameters())) for b in blocks]
    bnorm = [h.grad.norm().item() if getattr(h, "grad", None) is not None else float("nan")
             for h in hs]
    return g, bnorm, hs


def rho_from_backward(bnorm, tail_frac=0.6):
    """Geometric decay of ||dL/dh_t|| walking BACKWARD from the loss."""
    b = [x for x in bnorm if x == x and x > 0]
    if len(b) < 4:
        return float("nan")
    b = b[::-1]                       # index 0 = last loop
    start = 1
    stop = max(start + 2, int(len(b) * tail_frac))
    ratios = [b[i + 1] / b[i] for i in range(start, stop - 1) if b[i] > 1e-300]
    ratios = [r for r in ratios if r > 0]
    if not ratios:
        return float("nan")
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def n_eff(g):
    m = max(g)
    return sum(g) / m if m > 0 else float("nan")


def predicted_n_eff(rho, T):
    if rho >= 1 - 1e-12:
        return float(T)
    return (1 - rho ** T) / (1 - rho)


# --------------------------------------------------------------------------- P0
def p0_attribution_is_exact(d=32, hidden=64, T=8, seed=0):
    """The split-copy trick must reproduce the tied forward and the tied gradient."""
    torch.manual_seed(seed)
    h0 = torch.randn(4, d, dtype=DTYPE)
    blocks = make_blocks(d, hidden, 0.5, T, seed)
    hT_split, _ = unroll(blocks, h0)

    gen = torch.Generator().manual_seed(seed)
    tied = Block(d, hidden, 0.5, gen)
    h = h0
    for _ in range(T):
        h = tied(h)
    fwd_err = (hT_split - h).abs().max().item()

    h.pow(2).mean().backward()
    tied_g = torch.cat([tied.w1.grad.flatten(), tied.w2.grad.flatten()])
    for b in blocks:
        for p in b.parameters():
            p.grad = None
    hT2, _ = unroll(blocks, h0)
    hT2.pow(2).mean().backward()
    split_sum = sum(
        torch.cat([b.w1.grad.flatten(), b.w2.grad.flatten()]) for b in blocks
    )
    grad_err = (tied_g - split_sum).abs().max().item()
    return fwd_err, grad_err


# --------------------------------------------------------------------------- main
def sweep(scales, T=64, d=32, hidden=64, seeds=(0, 1, 2), normed=True):
    rows = []
    for s in scales:
        acc = []
        for sd in seeds:
            torch.manual_seed(sd)
            h0 = torch.randn(8, d, dtype=DTYPE)
            blocks = make_blocks(d, hidden, s, T, sd)
            g, bn, hs = per_loop_grad(blocks, h0, normed)
            r_step = rho_from_step_decay(hs)
            r_jac = rho_from_jacobian(blocks[-1], hs[-2])
            r_bwd = rho_from_backward(bn)
            acc.append((r_step, r_jac, r_bwd, n_eff(g), g))
        r_step = sum(a[0] for a in acc) / len(acc)
        r_jac = sum(a[1] for a in acc) / len(acc)
        r_bwd = sum(a[2] for a in acc) / len(acc)
        ne = sum(a[3] for a in acc) / len(acc)
        rows.append(dict(scale=s, rho_step=r_step, rho_jac=r_jac, rho_bwd=r_bwd,
                         n_eff=ne, pred_step=predicted_n_eff(r_step, T),
                         pred_bwd=predicted_n_eff(r_bwd, T), g=acc[0][4]))
    return rows


def main():
    print("=" * 78)
    print("P0  attribution exactness (split copies == tied loop)")
    f_err, g_err = p0_attribution_is_exact()
    print(f"    forward max|diff| = {f_err:.3e}    gradient max|diff| = {g_err:.3e}")
    ok0 = f_err < 1e-12 and g_err < 1e-12
    print(f"    {'PASS' if ok0 else 'FAIL'} (need < 1e-12; fp64)")
    if not ok0:
        print("    attribution is not exact -- everything below is void")
        return

    T = 64
    print()
    print("=" * 78)
    print(f"P1  NULL: near-non-contracting map (small residual, NO norm), T={T}")
    print("    h_{t+1} = h_t + 0.02*f(h_t): Jacobian ~ I, so backward sensitivity")
    print("    should barely decay and gradient mass should spread over all loops.")
    torch.manual_seed(0)
    h0 = torch.randn(8, 32, dtype=DTYPE)
    blocks = make_blocks(32, 64, 0.02, T, 0)
    g, bn, hs = per_loop_grad(blocks, h0, normed=False)
    ne_null = n_eff(g)
    r_bwd_null = rho_from_backward(bn)
    print(f"    rho_bwd = {r_bwd_null:.4f}    N_eff = {ne_null:.2f} of T = {T}"
          f"    ratio = {ne_null / T:.3f}")
    ok1 = ne_null / T > 0.5
    print(f"    {'PASS' if ok1 else 'FAIL'} (need > 0.5: gradient must NOT concentrate "
          f"when nothing damps it)")

    print()
    print("=" * 78)
    print(f"P2/P4  sweep, T={T}, RMSNorm on (Huginn-shaped), 3 seeds")
    print("    rho_step = step-size decay (this project's D115 method)")
    print("    rho_bwd  = decay of ||dL/dh_t|| backward from the loss")
    print(f"    {'scale':>6} {'rho_step':>9} {'rho_jac':>8} {'rho_bwd':>8} "
          f"{'N_eff':>7} {'pr_step':>8} {'pr_bwd':>7} {'e_step':>7} {'e_bwd':>6}")
    rows = sweep([0.05, 0.1, 0.2, 0.4, 0.8, 1.6], T=T)
    e_step, e_bwd = [], []
    for r in rows:
        es = abs(r["n_eff"] - r["pred_step"]) / r["pred_step"]
        eb = abs(r["n_eff"] - r["pred_bwd"]) / r["pred_bwd"]
        e_step.append(es)
        e_bwd.append(eb)
        print(f"    {r['scale']:>6.2f} {r['rho_step']:>9.4f} {r['rho_jac']:>8.4f} "
              f"{r['rho_bwd']:>8.4f} {r['n_eff']:>7.2f} {r['pred_step']:>8.2f} "
              f"{r['pred_bwd']:>7.2f} {es:>6.0%} {eb:>5.0%}")
    ms = sorted(e_step)[len(e_step) // 2]
    mb = sorted(e_bwd)[len(e_bwd) // 2]
    print(f"    median error using rho_step: {ms:>6.0%}   "
          f"{'PASS' if ms < 0.25 else 'FAIL'}")
    print(f"    median error using rho_bwd : {mb:>6.0%}   "
          f"{'PASS' if mb < 0.25 else 'FAIL'}")
    print("    P4: rho_step and rho_jac measure different objects if the ratio")
    print("        departs from 1 or rho_jac >= 1 while rho_step < 1.")

    print()
    print("=" * 78)
    print("P3  SPECIFICITY: does N_eff saturate in T at fixed contraction?")
    print(f"    {'T':>5} {'rho_bwd':>9} {'N_eff':>7} {'pred_bwd':>9}")
    for TT in (8, 16, 32, 64, 128):
        r = sweep([0.4], T=TT, seeds=(0,))[0]
        print(f"    {TT:>5} {r['rho_bwd']:>9.4f} {r['n_eff']:>7.2f} {r['pred_bwd']:>9.2f}")

    print()
    print("=" * 78)
    print("P5  DECISIVE: how is gradient MASS actually distributed across loops?")
    print("    (N_eff above is dominated by the final-loop spike and hides this.)")
    print(f"    {'scale':>6} " + " ".join(f"{'k='+str(k):>7}" for k in (1, 2, 4, 8, 16, 32, 64)))
    for s in (0.1, 0.4, 1.6):
        acc = []
        for sd in (0, 1, 2):
            torch.manual_seed(sd)
            h0 = torch.randn(8, 32, dtype=DTYPE)
            g, _, _ = per_loop_grad(make_blocks(32, 64, s, T, sd), h0, True)
            tot = sum(g)
            acc.append([sum(g[-k:]) / tot for k in (1, 2, 4, 8, 16, 32, 64)])
        m = [sum(x[i] for x in acc) / len(acc) for i in range(7)]
        print(f"    {s:>6.2f} " + " ".join(f"{v:>6.1%} " for v in m))
    print("    Uniform would put 50% in the last 32. Measured 46-64%.")
    print("    => gradient reaches every loop; contraction does NOT starve early loops.")


if __name__ == "__main__":
    main()
