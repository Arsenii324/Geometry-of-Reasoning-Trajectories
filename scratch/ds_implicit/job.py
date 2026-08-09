"""A18 / DR3 Method 2: implicit-differentiation input attribution through the fixed point.

WHAT IT IS. At the fixed point h* = F(h*, e), the implicit function theorem gives the
exact sensitivity of the converged state to the injected input:

    dh*/de = (I - J)^-1 dF/de,     J = dF/dh at h*

`DR_3_Interp.md` Method 2 calls this "the novel, publishable core -- sound, cheap
(~30-40 JVPs), artefact-free, and unclaimed in the literature", and records the
never-used-for-interpretability status as a verified negative. It is the one Method in
that report never started.

WHY IT SUITS THIS MODEL SPECIFICALLY. Every patching result in this project fights the
rho^(R-r) damping artefact: a state perturbation applied at unroll r is attenuated by
the contraction before it reaches the readout. This quantity is DEFINED AT THE FIXED
POINT, so it sidesteps the damping entirely, and the (I - J)^-1 factor *rewards*
contraction with fast Neumann convergence. It attributes to `e` -- the map's PARAMETER,
which D111/D113 showed is the object that actually moves h* -- not to a decaying state
perturbation.

HOW IT IS COMPUTED (adjoint form, so the cost is per-READOUT not per-input-dimension).
Let u = d(readout)/dh* be the readout direction. We want

    d(readout)/de = u^T (I - J)^-1 dF/de

so first solve the adjoint system by Neumann iteration, which is 1 VJP per term:

    v_0 = u,   v_{k+1} = u + J^T v_k   ->   v = (I - J^T)^-1 u

then ONE more VJP through the adapter gives the full gradient w.r.t. e, shaped
[seq_len, d_model]. Per-token attribution is the norm of its row.

THE TRUNCATION MUST BE ADAPTIVE, AND THAT IS OUR OWN CORRECTION TO DR3's SPEC.
DR3's "~30-40 JVPs" comes from rho = 0.855. Against this project's measured rates the
term count for a 1e-3 residual is 33 (D119's full-operator 0.8098), 38 (D113's causal
0.8335) -- and **75 for the rotating templates D127 found at |lambda| = 0.911**, which
D127 postdates the DR3 report. A FIXED 40-term truncation would leave residual 0.024 on
rotating templates against 0.0007 elsewhere: a 35x accuracy difference that CORRELATES
WITH TEMPLATE TYPE, i.e. a silent bias exactly where D127 says the dynamics differ. So
this iterates to a relative tolerance, caps, and RECORDS the count and final residual
per prompt. A prompt that fails to converge is reported, never silently truncated.

GROUND TRUTH, WHICH IS WHAT MAKES THIS TESTABLE RATHER THAN MERELY NOVEL.
The marker design supplies a known causal token: two prompts identical except a single
trailing `A`/`B` that selects which computation is required. If the attribution is
real, the marker position must receive disproportionate mass.

PRE-REGISTERED.
  P1  PRIMARY. The marker token's attribution rank is in the TOP 3 of all positions,
      in a majority of prompts. Reported as the rank distribution, not a mean.
  P2  NULL. The same computation with a RANDOM unit readout direction u. The marker
      carries no special status for a random direction, so its rank should be uniform.
      This is the instrument's own null and is run first.
  P3  CONVERGENCE GATE. Per prompt, record Neumann terms used and final relative
      residual. Any prompt hitting MAX_TERMS without reaching TOL is excluded from P1
      and counted in the report -- an unconverged adjoint is not an attribution.
  P4  **DECLARED AND NOT IMPLEMENTED IN THE LAUNCHED VERSION -- recorded here rather
      than quietly dropped at write-up.** The intent was: patch `e` at ONE position
      only, measure the true logit-gap shift, and correlate it against the
      attribution, as the ground truth DR3 asks for. `N_BRUTE` is defined below and
      never used. It needs a forward pass per position, so it cannot be recovered
      from the banked records and requires a second job. **Any result from this run
      therefore rests on P1/P2/P3 only, and the attribution is validated against the
      marker-token ground truth but NOT against brute-force patching.** Caught by
      self-check while the job was already running; the launched code is unchanged.
  P5  UNIT is the prompt. **The token-count verification is NOT IMPLEMENTED in the
      launched version either** -- found by the same self-check as P4. Marker pairs
      differ by one character in a fixed template, so equal token counts are highly
      likely, but they are ASSERTED here rather than checked, which is exactly what
      D101 got wrong. Verify from the banked `n_tokens` field before reading any
      result; it is stored per record, so this one IS recoverable locally.
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
            """v = (I - J^T)^-1 u0 by Neumann iteration; 1 VJP per term (P3)."""
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

        records.append({**{k: it[k] for k in ("item", "marker", "gold", "seq", "prompt")},
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
