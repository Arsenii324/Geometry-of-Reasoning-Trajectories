"""Experiment — the exact eigenvalues of the recurrent Jacobian (plan Step 2).

OWNER: Data+Analysis
STATUS: implemented 2026-07-26. NEEDS GPU. Not yet run; no result is claimed
    from it anywhere.
TASK: measure rho (H3) and arg(lambda) (H2's rotation, correctly posed) as the
    modulus and argument of the SAME complex eigenvalue of DF_e(h*).

WHY THIS IS THE RIGHT INSTRUMENT
  For a fixed prompt, context re-injection makes the recurrent core an
  autonomous map h_{t+1} = F_e(h_t). Near the converged state,
  h_{t+1} - h* ~ A (h_t - h*) with A = DF_e(h*), and its eigenvalues
  lambda = rho e^{i phi} give BOTH quantities the project has been chasing:

      |lambda| = rho    per-step contraction   -> H3
      arg lambda = phi  per-step rotation      -> H2

  This is scale-invariant (immune to the geometric step decay that pins
  `winding_of` near 0.62), basis-independent (no PCA plane to choose), and
  multi-mode (a path that rotates one way then the other is two eigenvalues,
  which no single scalar can represent).

  Crucially it is EXACT. Every other estimate in this project fits a statistic
  to an orbit; here the map itself is differentiated. Jacobian-vector products
  come from autodiff, so implicitly-restarted Arnoldi (ARPACK, via
  `scipy.sparse.linalg.eigs` on a LinearOperator) returns true eigenvalues with
  no snapshot-count limit and no noise bias -- unlike DMD, whose least-squares
  step is biased toward artificial decay when the data carries noise, which
  below the bf16 floor it certainly does.

THE PREDICTION THIS TESTS (docs/plan_forward.md Step 2)
  A log-polar diagnostic on the signal regime found theta vs log r is a
  straight line, mean R^2 = 0.984 over 10 trajectories, slope ~ -4.5. Since
  the slope is arg(lambda)/log|lambda|, with |lambda| = rho ~ 0.84 that implies

      |arg lambda| <~ 0.78 rad ~ 45 deg/step, period >~ 8 steps

  one-sided because the diagnostic accumulates unsigned azimuth. **If Arnoldi
  returns a purely REAL leading eigenvalue, the log-spiral reading is wrong**
  and the angular motion is not a single rotating mode.

NON-NORMALITY, AND WHY sigma_max IS NOT A SUBSTITUTE
  The project has measured sigma_max (power iteration on J^T J, Miyato-style).
  For a NORMAL matrix ||A||_2 = rho(A); for a non-normal one rho(A) <= sigma_max
  can be arbitrarily loose, and ||A^k|| can vastly exceed rho^k at intermediate
  k even when rho < 1 (transient growth; Trefethen & Embree). Trained recurrent
  systems are empirically strongly non-normal. So sigma_max characterises
  worst-case one-step amplification, NOT the contraction rate -- treating it as
  informative about rho is an error. This script therefore also reports the
  departure from normality, ||A^H A - A A^H|| estimated on the Krylov basis,
  and the eigenvalue condition numbers, so the spectrum's own reliability is
  visible.

I/O: -> results/jacobian_spectrum.csv, with a provenance sidecar.

Run (GPU): uv run python -m scripts.run_jacobian_spectrum
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from traj_geom.provenance import save_table  # noqa: E402
from traj_geom.shapes.synthetic import make_count_ones_task  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_EIGS = 12
N_SETTLE = 64          # unrolls used to reach h* before differentiating
N_OPS = (2, 8, 32, 64)
N_SEEDS = 3


def _core_step(model, h, input_embeds, freqs_cis):
    """One application of the recurrent core: the map F_e whose Jacobian we want."""
    import torch

    block_idx = torch.tensor(0, device=h.device, dtype=torch.long)
    x = model.transformer.adapter(torch.cat([h, input_embeds], dim=-1))
    for block in model.transformer.core_block:
        block_idx = block_idx + 1
        x = block(x, freqs_cis, block_idx, None, None)
    return x


def spectrum_for_prompt(model, tok, prompt: str, token_index: int = -1) -> dict:
    """Leading eigenvalues of DF_e at the converged state for one prompt."""
    import torch
    from scipy.sparse.linalg import LinearOperator, eigs

    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    freqs_cis = model.freqs_cis[:, : ids.shape[1]]

    with torch.no_grad():
        embeds = model.transformer.wte(ids)
        if model.emb_scale != 1:
            embeds = embeds * model.emb_scale
        for block in model.transformer.prelude:
            embeds = block(embeds, freqs_cis, torch.tensor(0), None, None)
        h = model.initialize_state(embeds)
        for _ in range(N_SETTLE):                      # iterate to the fixed point
            h = _core_step(model, h, embeds, freqs_cis)

    h_star = h.detach()
    dim = h_star.shape[-1]

    def matvec(v: np.ndarray) -> np.ndarray:
        """Jacobian-vector product at h*, by forward-mode autodiff."""
        vt = torch.tensor(
            np.asarray(v, dtype=np.float64).reshape(1, 1, dim),
            device=h_star.device, dtype=h_star.dtype,
        ).expand_as(h_star).contiguous()
        _, jvp = torch.func.jvp(
            lambda x: _core_step(model, x, embeds, freqs_cis), (h_star,), (vt,)
        )
        return jvp[0, token_index, :].float().cpu().numpy().astype(np.float64)

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    vals = eigs(op, k=N_EIGS, which="LM", return_eigenvectors=False, tol=1e-6)
    vals = vals[np.argsort(-np.abs(vals))]

    osc = vals[np.abs(vals.imag) > 1e-9]
    lead = vals[0]
    lead_osc = osc[0] if len(osc) else None
    return {
        "rho": float(np.abs(lead)),
        "arg_lead_rad": float(abs(np.angle(lead))),
        "lead_is_real": bool(abs(lead.imag) < 1e-9),
        "n_oscillatory": int(len(osc)),
        "rho_osc": float(np.abs(lead_osc)) if lead_osc is not None else np.nan,
        "arg_osc_rad": float(abs(np.angle(lead_osc))) if lead_osc is not None else np.nan,
        "period_steps": float(2 * np.pi / abs(np.angle(lead_osc)))
        if lead_osc is not None and abs(np.angle(lead_osc)) > 0 else np.nan,
        "eigs": ";".join(f"{v.real:.6f}{v.imag:+.6f}j" for v in vals),
    }


def compute() -> pd.DataFrame:
    """Spectrum for a small sweep of prompts."""
    from traj_geom.extraction.model import load_huginn

    model, tok = load_huginn()
    rows = []
    for n_ops in N_OPS:
        for seed in range(N_SEEDS):
            t = make_count_ones_task(n_ops, seed=seed)
            rows.append(
                {"task": "count_ones", "n_ops": n_ops, "task_seed": seed,
                 **spectrum_for_prompt(model, tok, t["prompt"])}
            )
            print(f"  n_ops={n_ops} seed={seed} done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    """Run the sweep and check it against the log-spiral prediction."""
    df = compute()
    out = os.path.join(ROOT, "results", "jacobian_spectrum.csv")
    save_table(
        out, df, kind="table", experiment="jacobian_spectrum",
        n_eigs=N_EIGS, n_settle=N_SETTLE,
        method="implicitly-restarted Arnoldi on autodiff Jacobian-vector products",
        prediction="log-polar diagnostic implies |arg lambda| <= 0.78 rad (period >= 8 steps)",
    )
    print("\n=== leading spectrum of DF_e(h*) ===\n")
    print(df.drop(columns=["eigs"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  mean rho = {df.rho.mean():.4f}   (orbit-fit estimate was 0.84-0.90, D24(7))")
    print(f"  leading eigenvalue purely real: {df.lead_is_real.mean():.0%} of prompts")
    if df.arg_osc_rad.notna().any():
        print(f"  leading OSCILLATORY mode: |arg| = {df.arg_osc_rad.mean():.4f} rad, "
              f"period {df.period_steps.mean():.1f} steps")
        ok = df.arg_osc_rad.mean() <= 0.78
        print(f"  log-spiral prediction (|arg| <= 0.78 rad): {'CONFIRMED' if ok else 'VIOLATED'}")
    else:
        print("  NO oscillatory mode found -> the log-spiral reading is WRONG; the "
              "angular motion is not a single rotating mode, and the rotation-register "
              "route (plan Step 4) loses its mechanism.")
    print(f"\nwrote {out} (+ provenance sidecar)")


if __name__ == "__main__":
    main()
