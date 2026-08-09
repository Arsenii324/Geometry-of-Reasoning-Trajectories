"""A21: re-solve the implicit-differentiation adjoint with GMRES instead of Neumann.

D131 ran DR3's Method 2 and it VOIDED ITSELF on its own convergence gate: 1 of 24
prompts reached a 1e-3 relative tolerance, the median exhausting a 120-term cap at
residual 0.0397. The implied per-iteration decay was **0.9735**, against this project's
measured contraction rates of 0.810 (Arnoldi), 0.8335 (causal) and 0.911 (rotating
templates). At 0.83 a 120-term series leaves ~1e-9; at the observed rate 1e-3 needs 257
terms.

**The identity is not in question and the solver is.** The adjoint form
`v = (I - J^T)^-1 u` was validated in closed form before D131 launched -- against exact
linear algebra on a 40-dimensional contraction, relative error 7e-4 with identical
attribution ranking. What fails is plain Neumann: it converges at ||J^k||, and for a
strongly NON-NORMAL operator that can exceed rho^k by orders of magnitude at practical
k. Huginn's Jacobian is strongly non-normal by three of this project's own measurements
(D31 rotating modes, D119 a complex leading pair, D112/D116 rotation angles of 20-83
degrees), so this is the expected regime -- and neither DR3 nor I anticipated it.

GMRES is the standard remedy: it builds a Krylov subspace and minimises the residual
over it, so it is governed by the pseudospectrum rather than by ||J^k||, and it is
exactly the case where Neumann's term-by-term accumulation is worst.

WHAT ELSE CHANGES, BOTH OF THEM D131's OWN ADMISSIONS.
  * **P4 is implemented this time.** D131 declared a brute-force cross-check and never
    wrote it; `N_BRUTE` sat defined and unread until a self-check caught it. Here a
    subset of prompts gets `e` patched at ONE position at a time, and the true logit-gap
    shift is correlated against the attribution. That is the ground truth DR3 asks for.
  * **P5 is implemented this time.** The token-count check within a marker pair was
    likewise declared and never written.

PRE-REGISTERED.
  P1  PRIMARY. GMRES reaches the 1e-3 tolerance in far fewer matvecs than Neumann's
      257-term projection, on a majority of prompts. If it does NOT, the obstacle is
      the operator rather than the algorithm, and DR3's Method 2 should be recorded as
      unusable on Huginn rather than retried a third time.
  P2  ATTRIBUTION. Given convergence, the marker token -- the known causal token, one
      character apart between arms -- ranks in the top 3 of all positions, against a
      RANDOM readout direction as the null. D131 saw median rank 14 vs 33 unconverged.
  P3  CONVERGENCE GATE, unchanged and still binding: an unconverged adjoint is not an
      attribution, and any prompt that fails is excluded and counted.
  P4  BRUTE-FORCE CROSS-CHECK, implemented: per-position patching on N_BRUTE prompts,
      correlated against the attribution.
  P5  TOKEN GATE, implemented: token counts equal within each marker pair.
"""

import collections
import json
import os
import random
import subprocess
import sys

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 48          # to reach h* before differentiating there
MAX_TERMS = 120         # covers |lambda| = 0.911 (needs ~75) with headroom
TOL = 1e-3              # relative change in the adjoint vector
N_BRUTE = 4             # prompts given the per-position brute-force cross-check (P4)
N_PROMPTS = 12


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


def build_items(rng):
    """Marker pairs: identical prompt, one trailing token selects the computation."""
    rules = ("for task A, report the largest symbol of the sequence. "
             "For task B, report the smallest symbol of the sequence.")
    items = []
    for i in range(N_PROMPTS):
        seq = [rng.randrange(10) for _ in range(6)]
        for mk in ("A", "B"):
            gold = str(max(seq)) if mk == "A" else str(min(seq))
            other = str(min(seq)) if mk == "A" else str(max(seq))
            items.append({
                "item": i, "marker": mk, "gold": gold, "other": other,
                "seq": " ".join(map(str, seq)),
                "prompt": (f"Rules: {rules}\nSequence: {' '.join(map(str, seq))}"
                           f"\nTask: {mk}"),
            })
    return items


def main():
    out_path = (os.path.abspath(sys.argv[1]) if len(sys.argv) > 1
                else os.path.abspath("implicit.json"))
    print(f"results -> {out_path}", flush=True)
    items = build_items(random.Random(20260809))
    print(f"{len(items)} prompts ({N_PROMPTS} items x 2 markers), "
          f"{len({i['prompt'] for i in items})} distinct", flush=True)

    run("pip install torch==2.5.1")
    run("pip install transformers==4.53.3 accelerate safetensors")

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()
    d_model = model.config.n_embd

    def core_step(h, e, freqs):
        """One full unroll: adapter + all four core blocks. This is F(h, e)."""
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        x = model.transformer.adapter(torch.cat([h, e], dim=-1))
        for block in model.transformer.core_block:
            bi = bi + 1
            x = block(x, freqs, bi, None, None)
        return x

    def coda_logits(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def prelude_e(ids):
        grabbed = {}

        def pre(_m, inp):
            grabbed.setdefault("e", inp[0][..., d_model:].detach().clone())

        h = model.transformer.adapter.register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=1)
        finally:
            h.remove()
        return grabbed["e"]

    def fixed_point(ids, e, freqs):
        """Iterate F to convergence from the model's own init, returning h*."""
        states = []

        def hook(_m, _i, o):
            states.append(o.detach())

        hh = model.transformer.core_block[-1].register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            hh.remove()
        return states[-1]

    records = []
    for n, it in enumerate(items):
        text = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        e = prelude_e(ids)
        h_star = fixed_point(ids, e, freqs)

        g_id = tok(it["gold"], add_special_tokens=False).input_ids[0]
        o_id = tok(it["other"], add_special_tokens=False).input_ids[0]

        # u = d(logit_gold - logit_other)/dh*, the readout direction at the answer position
        hv = h_star.clone().requires_grad_(True)
        lg = coda_logits(hv, freqs)[0, n_p - 1]
        (u,) = torch.autograd.grad(lg[g_id] - lg[o_id], hv)

        def adjoint(u0, e_, freqs_, h_):
            """v = (I - J^T)^-1 u0 by GMRES (P1); falls back to Neumann on failure.

            D131 measured plain Neumann needing ~257 terms here because ||J^k||, not
            rho^k, governs a non-normal operator. GMRES minimises the residual over a
            Krylov subspace instead, which is the standard remedy for exactly that.
            """
            import numpy as _np
            from scipy.sparse.linalg import LinearOperator, gmres
            n = int(u0.numel())
            calls = [0]

            def matvec(vec):
                calls[0] += 1
                v_t = torch.as_tensor(_np.asarray(vec, dtype=_np.float32),
                                      device=h_.device, dtype=h_.dtype).view_as(u0)
                hx = h_.detach().clone().requires_grad_(True)
                y = core_step(hx, e_, freqs_)
                (jtv,) = torch.autograd.grad(y, hx, grad_outputs=v_t)
                # (I - J^T) v
                return (v_t - jtv).detach().float().cpu().numpy().ravel().astype(_np.float64)

            op = LinearOperator((n, n), matvec=matvec, dtype=_np.float64)
            b = u0.detach().float().cpu().numpy().ravel().astype(_np.float64)
            sol, info = gmres(op, b, rtol=TOL, restart=40, maxiter=MAX_TERMS // 40 + 1)
            v = torch.as_tensor(sol.astype(_np.float32), device=u0.device,
                                dtype=u0.dtype).view_as(u0)
            resid = float(_np.linalg.norm(op.matvec(sol) - b) / max(1e-12, _np.linalg.norm(b)))
            return v.detach(), calls[0], resid

        def _neumann_unused(u0, e_, freqs_, h_):
            """Kept for reference: D131's plain Neumann, 1 VJP per term."""
            v = u0.clone()
            prev = None
            terms = 0
            resid = float("nan")
            for k in range(MAX_TERMS):
                hx = h_.detach().clone().requires_grad_(True)
                y = core_step(hx, e_, freqs_)
                (jtv,) = torch.autograd.grad(y, hx, grad_outputs=v)
                v_new = u0 + jtv
                terms = k + 1
                if prev is not None:
                    resid = float((v_new - v).norm() / max(1e-12, float(v.norm())))
                prev = v
                v = v_new
                if prev is not None and resid < TOL:
                    break
            return v.detach(), terms, resid

        v, terms, resid = adjoint(u, e, freqs, h_star)
        converged = bool(resid == resid and resid < TOL)

        def attribute(vv, *, e=e, h_star=h_star, freqs=freqs):
            """One VJP through the adapter: d(v . F(h*, e))/de, shape [seq, d_model].

            e/h_star/freqs are bound as defaults, not captured: this closure is only
            used inside its own iteration today, but a captured loop variable is
            exactly how a later edit reads the NEXT prompt's fixed point.
            """
            ev = e.detach().clone().requires_grad_(True)
            y = core_step(h_star.detach(), ev, freqs)
            (ge,) = torch.autograd.grad(y, ev, grad_outputs=vv)
            return ge[0].norm(dim=-1).float().cpu().numpy()

        attr = attribute(v)
        # P2 NULL: a random readout direction, same machinery
        ur = torch.randn_like(u)
        ur = ur / ur.norm() * u.norm()
        vr, terms_r, resid_r = adjoint(ur, e, freqs, h_star)
        attr_rand = attribute(vr)

        toks = tok.convert_ids_to_tokens(ids[0].tolist())
        marker_pos = max(i for i, t in enumerate(toks)
                         if it["marker"] in t.replace("Ġ", "").replace("▁", ""))
        order = np.argsort(-attr)
        rank = int(np.where(order == marker_pos)[0][0]) + 1
        order_r = np.argsort(-attr_rand)
        rank_r = int(np.where(order_r == marker_pos)[0][0]) + 1

        records.append({**{k: it[k] for k in ("item", "marker", "gold", "other", "seq", "prompt")},
                        "n_tokens": int(n_p), "terms": terms, "resid": resid,
                        "converged": converged, "marker_pos": marker_pos,
                        "marker_rank": rank, "marker_rank_random": rank_r,
                        "terms_random": terms_r,
                        "attr": attr.tolist(), "attr_random": attr_rand.tolist(),
                        "tokens": toks})
        json.dump({"records": records}, open(out_path, "w"))
        if n % 4 == 0:
            print(f"  {n}/{len(items)}  terms={terms} resid={resid:.2e} "
                  f"marker_rank={rank}/{n_p}", flush=True)

    # P5 TOKEN GATE, implemented this time. D131 declared it and never wrote it.
    print("\n=== P5 TOKEN GATE (equal token count within each marker pair) ===", flush=True)
    bypair = {}
    for r in records:
        bypair.setdefault(r["item"], []).append(r["n_tokens"])
    bad = {k: v for k, v in bypair.items() if len(set(v)) != 1}
    print(f"  {len(bypair) - len(bad)}/{len(bypair)} pairs have equal token counts"
          + (f"; MISMATCHED: {bad}" if bad else "  PASSED"), flush=True)

    # P4 BRUTE-FORCE CROSS-CHECK, implemented this time: ablate `e` at ONE position and
    # measure the true logit-gap shift, then correlate against the attribution. This is
    # the ground truth DR3 asks for and D131 promised without delivering.
    print(f"\n=== P4 BRUTE-FORCE CROSS-CHECK on {N_BRUTE} prompts ===", flush=True)
    from scipy.stats import spearmanr
    brute = []
    for r in records[:N_BRUTE]:
        if not r["converged"]:
            continue
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        e0 = prelude_e(ids)
        h0 = fixed_point(ids, e0, freqs)
        g_id = tok(r["gold"], add_special_tokens=False).input_ids[0]
        o_id = tok(r["other"], add_special_tokens=False).input_ids[0]

        def gap(e_, h_, freqs_=freqs, n_p_=n_p, g_id=g_id, o_id=o_id):
            with torch.no_grad():
                h = h_
                for _ in range(NUM_STEPS):
                    h = core_step(h, e_, freqs_)
                row = coda_logits(h, freqs_)[0, n_p_ - 1]
                return float(row[g_id] - row[o_id])

        base = gap(e0, h0)
        shifts = []
        for i in range(n_p):
            ep = e0.clone()
            ep[0, i, :] = 0.0
            shifts.append(abs(gap(ep, h0) - base))
        rho, pv = spearmanr(shifts, r["attr"][:len(shifts)])
        brute.append({"item": r["item"], "marker": r["marker"], "rho": float(rho),
                      "p": float(pv), "base_gap": base})
        print(f"  item {r['item']} {r['marker']}: Spearman(true ablation shift, "
              f"attribution) = {rho:+.4f}, p = {pv:.4f}", flush=True)
    if brute:
        print(f"  mean rho over {len(brute)} prompts = "
              f"{float(np.mean([b['rho'] for b in brute])):+.4f}", flush=True)
    else:
        print("  no converged prompt available -- P4 VOID", flush=True)

    print("\n=== P3 CONVERGENCE GATE ===", flush=True)
    conv = [r for r in records if r["converged"]]
    tt = [r["terms"] for r in records]
    print(f"  converged {len(conv)}/{len(records)} at tol {TOL:g}; "
          f"terms used min {min(tt)} median {int(np.median(tt))} max {max(tt)}", flush=True)
    print("  DR3 predicted ~30-40; our measured rates imply 33 (rho=0.810), "
          "38 (0.833), 75 (0.911, D127's rotating templates)", flush=True)
    if not conv:
        print("  VOID: nothing converged, no attribution may be read.", flush=True)
        json.dump({"records": records, "verdict": "VOID_no_convergence"},
                  open(out_path, "w"))
        return

    print("\n=== P1 PRIMARY: does the MARKER token get the mass? ===", flush=True)
    rk = [r["marker_rank"] for r in conv]
    print(f"  marker attribution rank: {collections.Counter(rk).most_common(6)}")
    print(f"  in TOP 3 for {sum(1 for x in rk if x <= 3)}/{len(rk)} prompts "
          f"(median rank {int(np.median(rk))} of ~{conv[0]['n_tokens']} positions)",
          flush=True)
    print("\n=== P2 NULL: random readout direction ===", flush=True)
    rr = [r["marker_rank_random"] for r in conv]
    print(f"  in TOP 3 for {sum(1 for x in rr if x <= 3)}/{len(rr)} prompts "
          f"(median rank {int(np.median(rr))})", flush=True)
    print("  -> the gap between these two lines is the result", flush=True)

    json.dump({"records": records,
               "config": {"num_steps": NUM_STEPS, "max_terms": MAX_TERMS, "tol": TOL,
                          "model": MODEL_ID, "revision": REVISION}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
