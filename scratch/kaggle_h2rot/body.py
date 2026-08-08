"""H2's FIRST REAL TEST, on the one instrument that is not broken, at MATCHED LENGTH.

WHY THIS EXISTS. H2 -- "when the path loops, the winding number grows with the
number of reasoning steps the problem requires" -- was retired in this project
using `winding`, and D65 established that retirement was circular: D28 showed
winding's sign FLIPS with `num_steps`, a recording-budget parameter, under an
unbiased on-manifold null, and D55 showed the trajectory is sampled at ~3.2 points
per rotation against a Nyquist floor of 2. Killing a hypothesis with a ruler you
have already proven broken is not evidence.

THE WORKING INSTRUMENT. For a fixed prompt the recurrent core is an autonomous map
h <- F_e(h). At the converged state, J = DF_e(h*) has eigenvalues lambda =
rho e^{i phi}: |lambda| is H3's contraction and arg(lambda) is H2's rotation PER
STEP. No trajectory, no recording window, no PCA plane, no surrogate -- so none of
what killed winding can touch it. D55 ran this on 3 prompts and got |arg| =
2.3974 / 2.3875 / 1.0496 at n_ops 8/32/64, which IS H2's question asked properly,
and it was never analysed as such because n=3 supports nothing.

THE CONFOUND D55 COULD NOT ESCAPE, AND THIS RUN DOES. Those three prompts differ in
n_ops AND in prompt length together, so any trend is unattributable -- exactly the
confound that made winding-vs-depth uninterpretable. Here difficulty is crossed
with a LENGTH-MATCHED control: `make_variants` emits a byte-identical body and two
questions, `track` ("Final total?") needing accumulation over all n_ops steps and
`local` ("What was the last instruction?") needing only the final step. Measured
against the pinned tokenizer, `local` is a CONSTANT +3 tokens at every n_ops, so
the arms shift equally and a slope comparison is unaffected. H2 predicts rotation
scales with REQUIRED REASONING STEPS, which `track` has and `local` does not, at
the same body and nearly the same length.

WHAT THIS KERNEL DOES NOT DO: decide anything. It measures and persists the full
spectrum, and nothing else. Three kernels in this project (D62, geometry-discourse,
D70) printed confident verdicts that were wrong, always because the verdict was
computed from the same variables as the run. The analysis lives locally in
`scripts/run_h2_rotation.py`, where it is unit-tested and can be re-run without a
GPU when a question arrives that this design did not anticipate (B4.14).

PRE-REGISTERED PREDICTIONS, written before running (CLAUDE.md section 1).

P1 -- GATE ON THE INSTRUMENT, not the hypothesis. At least 80% of prompts must
     have a COMPLEX leading oscillatory mode. arg(lambda) is undefined for a real
     eigenvalue, so if the spectrum is mostly real this kernel measures nothing and
     no other prediction may be read. D55 got 3/3 complex, so the bar is low, but
     it has never been checked off the count_ones family.
P2 -- H2, DIRECTIONAL. |arg lambda| rises with n_ops for `track`: per-level
     Spearman rho >= +0.886 over the 6 levels (the project's own N=6 threshold).
P3 -- THE CONTROL. That rise is absent or weaker for `local` at matched length.
     H2 is about REASONING STEPS, so a rise in both arms would mean the quantity
     tracks prompt length, not reasoning -- which is the winding failure again, and
     would be reported as such.
P4 -- THE CLEAN NEGATIVE. If neither arm moves with n_ops, H2 is REFUTED on an
     instrument that passes its own null -- a falsifiable negative that winding
     could never deliver, and a publishable one. D55's three points hint at this:
     |arg| FELL from 2.40 to 1.05 as n_ops went 8 -> 64.
P5 -- ARCHITECTURE vs TRAINING. The untrained arm is run last on the same grid. If
     the trained arm's n_ops dependence is reproduced by random weights, it is a
     property of the architecture and the prompt, not of anything learned.

DERIVED H2 QUANTITY, recorded so the analysis can use it. H2 is about TOTAL turns,
not rate. Total accumulated phase to convergence is Phi = |arg| * t*, with
t* = ln(eps)/ln(rho) the unrolls to shrink the gap to eps (rigor_audit section 3).
Turns = Phi / 2pi. Both factors come from this same spectrum, so no extra run.

COST. D55 measured 498-1786 matvecs per prompt at ~0.25 s each, i.e. ~250 s per
prompt including the settle. 24 trained + 12 untrained = 36 prompts, ~3 h. Every
prompt is written to disk as it completes and printed as one JSON line, so a
timeout degrades to fewer prompts rather than to nothing, and the trained arm --
the one that carries P2/P3/P4 -- runs first.
"""
# @needs: run load_arm free_arm

import json
import os
import random
import time
import traceback

import numpy as np

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"

N_SETTLE = 64          # unrolls to reach h* before linearising; D55 used 64
N_EIGS = 8             # D55 used 10; 8 still yields 3-4 complex pairs and costs less
TOL = 1e-5             # matched to D55 so the two runs are comparable
MAX_MATVEC = 4000      # ~2.5x D55's worst prompt; a cap that is REPORTED, never silent
BUDGET_S = 7.5 * 3600  # stop starting new prompts past this; Kaggle GPU cap is 9 h

N_OPS = (4, 8, 12, 16, 24, 32)   # 6 levels: the project's N=6 Spearman threshold is 0.886
KINDS = ("track", "local")
SEEDS_TRAINED = (0, 1)
SEEDS_UNTRAINED = (0,)

OUT = "/kaggle/working/h2rot.json"


def make_variants(n_ops, seed=0):
    """Verbatim from src/traj_geom/shapes/synthetic.py -- the length-matched pair.

    Copied rather than imported because a Kaggle kernel is a single file. The
    local test `tests/test_h2_rotation.py::test_kernel_task_matches_library`
    asserts this stays byte-equivalent to the library version, so the copy cannot
    silently drift from the generator every other experiment used.
    """
    rng = random.Random(seed)
    ops = [rng.choice([1, -1]) for _ in range(n_ops)]
    body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1." for o in ops)
    return {
        "track": body + " Final total? A:",
        "local": body + " What was the last instruction? A:",
    }


def spectrum(model, tok, text):
    """Top-N_EIGS eigenvalues of the recurrent Jacobian at the converged state.

    Mirrors `scratch/kaggle_jacobian/main.py`, which produced D55 and is the only
    version of this computation known to run on a T4. Deliberately unchanged in
    substance: forward-mode AD OOMs here, and reverse mode is exact for the
    spectrum because eig(J^T) = eig(J).
    """
    import torch
    from scipy.sparse.linalg import LinearOperator, eigs
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def core_step(h, embeds, freqs_cis):
        bi = torch.tensor(0, device=h.device, dtype=torch.long)
        x = model.transformer.adapter(torch.cat([h, embeds], dim=-1))
        for block in model.transformer.core_block:
            bi = bi + 1
            x = block(x, freqs_cis, bi, None, None)
        return x

    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
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

    mode = "forward"
    try:
        _fwd(np.zeros(dim, dtype=np.float32))
    except Exception as exc:
        print(f"    forward-mode AD unavailable ({type(exc).__name__}); reverse mode",
              flush=True)
        mode = "reverse"

    class _Cap(Exception):
        pass

    def matvec(v):
        calls[0] += 1
        if calls[0] > MAX_MATVEC:
            raise _Cap(f"matvec cap {MAX_MATVEC} hit")
        return (_fwd if mode == "forward" else _rev)(v).detach().float().cpu() \
            .numpy().astype(np.float64)

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    try:
        vals = eigs(op, k=N_EIGS, which="LM", return_eigenvectors=False, tol=TOL,
                    maxiter=400)
        converged = True
    except _Cap as exc:
        return {"ok": False, "why": str(exc), "matvecs": calls[0], "ad_mode": mode}
    vals = vals[np.argsort(-np.abs(vals))]
    osc = vals[np.abs(vals.imag) > 1e-8]
    lead = vals[0]
    return {
        "ok": True, "converged": converged, "matvecs": calls[0], "ad_mode": mode,
        "n_tokens": int(ids.shape[1]),
        "rho": float(np.abs(lead)),
        "lead_is_real": bool(abs(lead.imag) <= 1e-8),
        "n_osc": int(len(osc)),
        # leading OSCILLATORY mode: the largest-|lambda| complex eigenvalue. This is
        # the quantity D55 reported and the one H2 is about.
        "arg_osc": (float(abs(np.angle(osc[0]))) if len(osc) else None),
        "rho_osc": (float(np.abs(osc[0])) if len(osc) else None),
        "eigs": [[float(z.real), float(z.imag)] for z in vals],
    }


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoConfig, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)

    t0 = time.time()
    records = []

    def save():
        with open(OUT, "w") as fh:
            json.dump({"n_settle": N_SETTLE, "n_eigs": N_EIGS, "tol": TOL,
                       "max_matvec": MAX_MATVEC, "n_ops": list(N_OPS),
                       "records": records}, fh)

    for arm, seeds in (("trained", SEEDS_TRAINED), ("untrained", SEEDS_UNTRAINED)):
        if time.time() - t0 > BUDGET_S:
            print(f"=== SKIPPING {arm}: budget spent ===", flush=True)
            break
        print(f"\n=== {arm} ===", flush=True)
        model = load_arm(MODEL_ID if arm == "trained" else None, cfg,
                         REVISION if arm == "trained" else 0)
        for n_ops in N_OPS:
            for seed in seeds:
                v = make_variants(n_ops, seed=seed)
                for kind in KINDS:
                    if time.time() - t0 > BUDGET_S:
                        print("  budget spent; stopping", flush=True)
                        break
                    tag = f"{arm} {kind} n_ops={n_ops} seed={seed}"
                    ts = time.time()
                    try:
                        rec = spectrum(model, tok, v[kind])
                    except Exception as exc:
                        # D66's lesson: store the MESSAGE and the TRACEBACK, not
                        # type(e).__name__. A bare exception name cost a whole rerun.
                        rec = {"ok": False, "why": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()}
                    rec.update({"arm": arm, "kind": kind, "n_ops": n_ops, "seed": seed,
                                "secs": round(time.time() - ts, 1)})
                    records.append(rec)
                    save()
                    print(json.dumps(rec if rec.get("ok") else
                                     {k: rec[k] for k in rec if k != "traceback"}),
                          flush=True)
                    if not rec.get("ok"):
                        print(rec.get("traceback", ""), flush=True)
        free_arm(model, MODEL_ID if arm == "trained" else None)

    save()
    ok = [r for r in records if r.get("ok")]
    print(f"\n=== MEASURED {len(ok)}/{len(records)} prompts in "
          f"{(time.time() - t0) / 60:.1f} min ===", flush=True)
    # Descriptive only. Every inferential statistic lives in the local analysis;
    # a kernel that draws its own conclusion is how D62/D70 went wrong.
    for arm in ("trained", "untrained"):
        sub = [r for r in ok if r["arm"] == arm]
        if not sub:
            continue
        cx = [r for r in sub if r.get("arg_osc") is not None]
        print(f"  {arm}: {len(cx)}/{len(sub)} have a complex oscillatory mode "
              f"(P1 gate wants >= 80%)", flush=True)
    print(f"saved {OUT}", flush=True)
    print("DONE", flush=True)


main()
