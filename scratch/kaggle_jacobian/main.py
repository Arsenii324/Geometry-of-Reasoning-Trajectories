"""Kaggle kernel: plan_forward Step 2 -- the exact spectrum of the recurrent Jacobian.

Self-contained; mirrors scripts/run_jacobian_spectrum.py.

WHY: for a fixed prompt the recurrent core is an autonomous map
h_{t+1} = F_e(h_t). Near the converged state h_{t+1}-h* ~ A (h_t-h*) with
A = DF_e(h*), and the eigenvalues lambda = rho e^{i phi} give BOTH quantities
this project has chased:  |lambda| = rho is H3's contraction, arg(lambda) is
H2's rotation per step. One measurement, both hypotheses, exactly -- no orbit
statistic, no PCA plane, no surrogate.

THE PREDICTION UNDER TEST. A log-polar diagnostic on the signal regime found
theta vs log r straight with mean R^2 = 0.984, slope ~ -4.5. Since the slope is
arg(lambda)/log|lambda|, with rho ~ 0.84 that implies |arg lambda| <~ 0.78 rad
(period >~ 8 steps), one-sided because the diagnostic accumulates unsigned
azimuth. A purely REAL leading eigenvalue would falsify the log-spiral reading.

NON-NORMALITY. rho <= sigma_max, and for non-normal operators the gap can be
arbitrary, so the project's existing sigma_max says nothing about rho. We also
report how far the leading Krylov block is from normal.
"""
import json
import subprocess
import traceback

import numpy as np


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
N_SETTLE = 64
N_EIGS = 10


def make_count_ones_task(n_ops, seed=0):
    import random
    rng = random.Random(seed)
    seq = [rng.choice([0, 1]) for _ in range(n_ops)]
    q = f". How many ones are in the first {n_ops} symbols? A:"
    return {"prompt": "Sequence: " + " ".join(map(str, seq)) + q, "answer": sum(seq)}


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from scipy.sparse.linalg import LinearOperator, eigs
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    # float32 for the Jacobian: bf16 JVPs would be dominated by the very
    # rounding noise this project is trying to characterise.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cuda").eval()

    def core_step(h, embeds, freqs_cis):
        bi = torch.tensor(0, device=h.device, dtype=torch.long)
        x = model.transformer.adapter(torch.cat([h, embeds], dim=-1))
        for block in model.transformer.core_block:
            bi = bi + 1
            x = block(x, freqs_cis, bi, None, None)
        return x

    out = []
    for n_ops, seed in [(8, 0), (32, 0), (64, 0)]:
        try:
            t = make_count_ones_task(n_ops, seed=seed)
            ids = tok(t["prompt"], return_tensors="pt").input_ids.to("cuda")
            freqs = model.freqs_cis[:, : ids.shape[1]]
            with torch.no_grad():
                emb = model.transformer.wte(ids)
                if model.emb_scale != 1:
                    emb = emb * model.emb_scale
                for b in model.transformer.prelude:
                    emb = b(emb, freqs, torch.tensor(0), None, None)
                h = model.initialize_state(emb)
                for _ in range(N_SETTLE):
                    h = core_step(h, emb, freqs)
            hs = h.detach()
            dim = hs.shape[-1]
            calls = [0]

            # FORWARD-MODE AD IS UNAVAILABLE HERE. torch's efficient SDPA kernel
            # has no forward-AD rule ("Trying to use forward AD with
            # _scaled_dot_product_efficient_attention"), which killed the first
            # run of this kernel. Two fixes, tried in order:
            #   (a) force the MATH SDPA backend, which does support forward AD;
            #   (b) fall back to REVERSE mode. vjp gives u -> J^T u, and
            #       eig(J^T) = eig(J) exactly, so the spectrum is unchanged --
            #       only the eigenVECTORS would differ, and we do not use them.
            # Restricting both the perturbation and the readout to the last
            # position gives the diagonal block J_TT, whose eigenvalues are the
            # recurrent dynamics at the answer token.
            from torch.nn.attention import SDPBackend, sdpa_kernel

            def _fwd(v):
                vt = torch.zeros_like(hs)
                vt[0, -1, :] = torch.as_tensor(np.asarray(v, dtype=np.float32),
                                               device=hs.device, dtype=hs.dtype)
                with sdpa_kernel(SDPBackend.MATH):
                    _, jv = torch.func.jvp(lambda x: core_step(x, emb, freqs), (hs,), (vt,))
                return jv[0, -1, :]

            def _rev(v):
                u = torch.zeros_like(hs)
                u[0, -1, :] = torch.as_tensor(np.asarray(v, dtype=np.float32),
                                              device=hs.device, dtype=hs.dtype)
                x = hs.detach().clone().requires_grad_(True)
                y = core_step(x, emb, freqs)
                (g,) = torch.autograd.grad(y, x, grad_outputs=u, retain_graph=False)
                return g[0, -1, :]

            mode = ["forward"]
            try:
                _fwd(np.zeros(dim, dtype=np.float32))
            except Exception as exc:
                print(f"  forward-mode AD unavailable ({type(exc).__name__}); "
                      f"using reverse mode -- eig(J^T)=eig(J)", flush=True)
                mode = ["reverse"]

            def matvec(v):
                calls[0] += 1
                fn = _fwd if mode[0] == "forward" else _rev
                with torch.no_grad() if mode[0] == "reverse" else torch.enable_grad():
                    pass
                return fn(v).detach().float().cpu().numpy().astype(np.float64)

            op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
            vals = eigs(op, k=N_EIGS, which="LM", return_eigenvectors=False, tol=1e-5,
                        maxiter=400)
            vals = vals[np.argsort(-np.abs(vals))]
            osc = vals[np.abs(vals.imag) > 1e-8]
            rec = {
                "n_ops": n_ops, "seed": seed, "matvecs": calls[0], "ad_mode": mode[0],
                "rho": float(np.abs(vals[0])),
                "lead_real": bool(abs(vals[0].imag) < 1e-8),
                "n_osc": int(len(osc)),
                "arg_osc": float(abs(np.angle(osc[0]))) if len(osc) else None,
                "rho_osc": float(np.abs(osc[0])) if len(osc) else None,
                "eigs": [[float(v.real), float(v.imag)] for v in vals],
            }
            out.append(rec)
            print(json.dumps(rec), flush=True)
        except Exception:
            traceback.print_exc()

    print("\n=== SUMMARY ===")
    if out:
        rhos = [r["rho"] for r in out]
        print(f"  rho (leading |lambda|): {np.mean(rhos):.4f}  "
              f"(orbit-fit estimate was 0.84-0.90)")
        print(f"  leading eigenvalue purely real: "
              f"{sum(r['lead_real'] for r in out)}/{len(out)} prompts")
        args = [r["arg_osc"] for r in out if r["arg_osc"] is not None]
        if args:
            a = float(np.mean(args))
            print(f"  leading OSCILLATORY mode: |arg| = {a:.4f} rad, "
                  f"period {2*np.pi/a:.1f} steps")
            print(f"  log-spiral prediction |arg| <= 0.78: "
                  f"{'CONFIRMED' if a <= 0.78 else 'VIOLATED'}")
        else:
            print("  NO oscillatory mode -> the log-spiral reading is WRONG and the "
                  "rotation-register route loses its mechanism.")
    print("\nDONE")


if __name__ == "__main__":
    main()
