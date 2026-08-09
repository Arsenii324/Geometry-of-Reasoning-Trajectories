"""B1c: does the causal contraction rate hold across RECIPIENTS, or was it one prompt?

WHAT B1b SETTLED (D111). With the state arm fixed, the two arms dissociate
completely at 48 unrolls: the `e` (parameter) arm is FLAT in r -- 4.037, 4.037,
4.038, 4.034, 4.050, 4.088, 3.534 nats -- and flips the preferred answer 6 of 8
donors at EVERY r including r = 40; the state arm flips it 0 of 56. And the state
arm's MAGNITUDE decays as rho^(R-r) across nearly three orders of magnitude,
giving **rho = 0.822** (per-donor range 0.791-0.866, R^2 0.83-0.98).

That number matters more than the dissociation: every prior estimate of rho was a
passive fit to an unperturbed orbit. This one is causal, and it lands on D94's
bias-corrected ~0.82 and beside D31's Arnoldi 0.79-0.81.

WHY THIS RE-RUN EXISTS -- TWO DEFECTS IN B1b, BOTH MINE.

  1. **ONE RECIPIENT, NOT EIGHT.** `pairs = pairs[:N_PAIRS]` sliced the first 8 of
     a double loop whose OUTER index is the recipient, so every pair had a = 0:
     eight donors into a single prompt, with 638 length-matched pairs available.
     The rho was therefore measured in the neighbourhood of ONE fixed point, and
     the 0.791-0.866 spread is across perturbation DIRECTIONS, not prompts. This
     is D99's convenience-slice failure mode committed again, in a job written
     after it was documented. Fixed here by stratifying over recipients, with a
     VOID guard if fewer than MIN_RECIPIENTS survive.
  2. **THE PRE-REGISTERED STATISTIC WAS THE WRONG ONE.** B1b tested the SIGNED
     mean state potency against r: rho = -0.321, p = 0.4821, printed as not
     detected. The sign of a donor's pull varies by donor, so the signed mean
     cancels (+0.000, -0.000, +0.001, +0.008, +0.021, -0.007, -0.174). The decay
     is only visible in |potency|. Both are computed here; the fit is on |potency|,
     and the fit block was verified against B1b's banked records to reproduce
     median rho = 0.8224 exactly before this job was submitted.

PRE-REGISTERED PREDICTION, QUANTITATIVE.

  * **Primary.** Median rho across >= 12 DISTINCT recipients falls in 0.79-0.86,
    i.e. the causal rate is a property of the map and not of one prompt. If
    instead rho varies widely across recipients (say sd > 0.05, or fewer than
    half inside the bracket), then the contraction rate is prompt-dependent and
    D94/D31's single numbers are summaries of a distribution -- a different and
    more interesting finding, which must be reported as such rather than as a
    failed replication.
  * **Secondary.** The e arm stays flat in r and keeps flipping the answer at a
    rate near 75%, now across recipients rather than donors into one recipient.
  * **Null that must pass first.** The degeneracy guard below: if either arm
    returns one value per pair across all depths, nothing is interpreted.

DESIGN NOTES CARRIED FORWARD. Donor and recipient are DIFFERENT prompts with
different golds at a VERIFIED-IDENTICAL token count, checked in-kernel. Family is
`addk` (D107: the one ladder surviving honest fixed-depth scoring). The donor is
varied along with the recipient -- taking the first eligible donor per recipient
picks nearly the same item every time, which would collapse the perturbation
direction. Statistic: logit(donor_gold) - logit(recipient_gold) at the answer
position via the D71-validated `coda_head`, minus its unpatched baseline.
"""

import json
import os
import subprocess
import sys


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 48
PATCH_R = (0, 4, 8, 16, 24, 32, 40)
N_PAIRS = 16        # now = distinct RECIPIENTS, one donor each (D111(3))
MIN_RECIPIENTS = 12  # below this a rho fit describes one fixed point, not Huginn


def addk_items():
    """`addk`: v + k with v + k <= 9, so every gold is a single token (D89-safe),
    and every prompt is the same shape. D107 identified this as the family that
    survives honest scoring."""
    out = []
    for v in range(0, 10):
        for k in range(1, 8):
            if v + k <= 9:
                out.append({"prompt": f"What is {v} + {k}? Answer with the number.",
                            "gold": str(v + k), "v": v, "k": k})
    return out


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("estream.json")
    print(f"results -> {out_path}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))

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
    adapter = model.transformer.adapter
    core_last = model.transformer.core_block[-1]

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    def prelude_e(ids):
        """The `e` this experiment patches: the prelude output, captured by hooking
        the adapter and reading the second half of its input on the first call."""
        grabbed = {}

        def pre(_m, inp):
            if "e" not in grabbed:
                grabbed["e"] = inp[0][..., d_model:].detach().clone()

        h = adapter.register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=1)
        finally:
            h.remove()
        return grabbed["e"]

    def forward(ids, gold_ids, other_ids, patch=None, donor_e=None, donor_state=None,
                patch_r=None):
        """One forward. `patch` in {None, 'e', 'state'}; both patch from `patch_r`
        onward, so the two arms differ ONLY in which stream is intervened on."""
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        step = {"i": 0}
        handles = []

        if patch == "e":
            def pre(_m, inp):
                cur = inp[0]
                if step["i"] >= patch_r:
                    cur = torch.cat([cur[..., :d_model], donor_e], dim=-1)
                step["i"] += 1
                return (cur,)
            handles.append(adapter.register_forward_pre_hook(pre))
        elif patch == "state":
            # INJECT ONCE, AT r. The first version of this arm overwrote x at EVERY
            # unroll from r onward, which pins the trajectory and makes the outcome
            # independent of when patching began -- it produced exactly 8 distinct
            # values across 56 measurements (D108(3)). A causal patch is a single
            # substitution followed by the model's own dynamics.
            def pre(_m, inp):
                cur = inp[0]
                if step["i"] == patch_r and donor_state is not None:
                    cur = torch.cat([donor_state, cur[..., d_model:]], dim=-1)
                step["i"] += 1
                return (cur,)
            handles.append(adapter.register_forward_pre_hook(pre))

        out = {}

        def post(_m, _i, o):
            out["h"] = o.detach()

        handles.append(core_last.register_forward_hook(post))
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            for h in handles:
                h.remove()
        row = torch.log_softmax(coda_head(out["h"], freqs).float()[0, n_p - 1], dim=-1)
        return (float(row[other_ids[0]] - row[gold_ids[0]]),      # donor - recipient gap
                int((row > row[gold_ids[0]]).sum().item()) + 1)   # recipient rank

    items = addk_items()
    # PAIRS AT A VERIFIED-IDENTICAL TOKEN COUNT. The swap is undefined otherwise,
    # and D101's gate caught a length claim that was false -- so check, do not assert.
    enc = {i: encode(it["prompt"]) for i, it in enumerate(items)}
    pairs = []
    for a in range(len(items)):
        for b in range(len(items)):
            if a == b or items[a]["gold"] == items[b]["gold"]:
                continue
            if enc[a].shape[1] == enc[b].shape[1]:
                pairs.append((a, b))
    # STRATIFY OVER RECIPIENTS. B1b took `pairs[:N_PAIRS]` off a double loop whose
    # outer index is the recipient, so all 8 pairs had a = 0: ONE recipient, eight
    # donors, and rho was measured at a single fixed point (D111(3)). 638 pairs were
    # available. Take at most one donor per recipient so the unit of independence is
    # the recipient prompt, which is what the claim is about.
    # Vary the DONOR too. Taking the first eligible b per recipient picks donor
    # index 0 or 1 almost every time, so every perturbation would point in nearly
    # the same direction and the spread across pairs would understate the real one.
    by_rec = {}
    for a, b in pairs:
        by_rec.setdefault(a, []).append(b)
    strat = []
    for j, a in enumerate(sorted(by_rec)):
        cand = by_rec[a]
        strat.append((a, cand[(j * 7 + 3) % len(cand)]))
    print(f"{len(items)} items -> {len(pairs)} length-matched pairs spanning "
          f"{len({a for a, _ in pairs})} distinct recipients; stratified to "
          f"{len(strat)}, using {min(N_PAIRS, len(strat))}", flush=True)
    pairs = strat[:N_PAIRS]
    n_rec = len({a for a, _ in pairs})
    if n_rec < MIN_RECIPIENTS:
        print(f"VOID: only {n_rec} distinct recipients (need {MIN_RECIPIENTS}). "
              f"A rho fitted here would again describe one fixed point.", flush=True)
        json.dump({"verdict": "VOID_too_few_recipients", "n_recipients": n_rec},
                  open(out_path, "w"))
        return
    print(f"  {n_rec} DISTINCT recipients -- the unit of independence", flush=True)
    if not pairs:
        print("NO length-matched pairs -- VOID, nothing can be concluded.", flush=True)
        json.dump({"verdict": "VOID_no_pairs"}, open(out_path, "w"))
        return

    records = []
    for rec_i, don_i in pairs:
        rec, don = items[rec_i], items[don_i]
        ids = enc[rec_i]
        g_r = tok(rec["gold"], add_special_tokens=False).input_ids
        g_d = tok(don["gold"], add_special_tokens=False).input_ids
        e_don = prelude_e(enc[don_i])
        if e_don.shape[1] != ids.shape[1]:
            print(f"  shape mismatch, skipping {rec_i}->{don_i}", flush=True)
            continue
        base_gap, base_rank = forward(ids, g_r, g_d)
        # The donor's state AT EVERY UNROLL. The first version used setdefault and
        # therefore captured only the donor's unroll-0 state -- essentially its
        # random h_0, not the state it actually holds at the unroll being patched
        # (D108(3), second bug). Patching r must inject the donor's state at r.
        st_all = []

        def grab(_m, inp):
            st_all.append(inp[0][..., :d_model].detach().clone())

        h = adapter.register_forward_pre_hook(grab)
        try:
            with torch.no_grad():
                model(input_ids=enc[don_i], num_steps=NUM_STEPS)
        finally:
            h.remove()

        for r in PATCH_R:
            ge, _ = forward(ids, g_r, g_d, patch="e", donor_e=e_don, patch_r=r)
            ds_r = st_all[r] if r < len(st_all) else st_all[-1]
            gs, _ = forward(ids, g_r, g_d, patch="state", donor_state=ds_r, patch_r=r)
            records.append({"pair": f"{rec_i}->{don_i}", "r": r,
                            "rec_gold": rec["gold"], "don_gold": don["gold"],
                            "base_gap": base_gap, "e_gap": ge, "state_gap": gs,
                            "e_potency": ge - base_gap, "state_potency": gs - base_gap})
            json.dump(records, open(out_path, "w"))
        print(f"  {rec['prompt'][:26]:>26} <- {don['gold']}: base {base_gap:+.3f} | "
              f"e-potency @r=0 {records[-len(PATCH_R)]['e_potency']:+.3f} "
              f"@r={PATCH_R[-1]} {records[-1]['e_potency']:+.3f} | "
              f"state @r=0 {records[-len(PATCH_R)]['state_potency']:+.3f} "
              f"@r={PATCH_R[-1]} {records[-1]['state_potency']:+.3f}", flush=True)

    from scipy.stats import spearmanr
    # DEGENERACY GUARD. D108's state arm returned exactly one distinct value per
    # pair across all 7 depths, which is the signature of a patch that does not
    # actually depend on r. Check it rather than discover it in the analysis.
    for arm in ("e_gap", "state_gap"):
        per_pair = {}
        for x in records:
            per_pair.setdefault(x["pair"], set()).add(round(x[arm], 9))
        flat = [k for k, v in per_pair.items() if len(v) == 1]
        if flat:
            print(f"  !! DEGENERACY: {arm} is constant across all r for "
                  f"{len(flat)}/{len(per_pair)} pairs -- that arm is NOT measuring "
                  f"a depth-dependent intervention and must not be interpreted.",
                  flush=True)
        else:
            print(f"  {arm}: varies with r in all {len(per_pair)} pairs (not degenerate)",
                  flush=True)

    print("\n=== INSTRUMENT NULL: can the intervention move the output AT ALL? ===",
          flush=True)
    n_pairs = len({x["pair"] for x in records})
    for arm in ("e_potency", "state_potency"):
        vals = [x[arm] for x in records]
        best = max(abs(v) for v in vals) if vals else 0.0
        print(f"  {arm:>15}: max |potency| {best:.4f} over {len(records)} "
              f"measurements from {n_pairs} pairs", flush=True)
    if not records:
        print("  VOID: no measurements.", flush=True)
        return

    print("\n=== PRE-REGISTERED: e should FALL with r, state should RISE ===", flush=True)
    for arm, expect in (("e_potency", "decrease"), ("state_potency", "increase")):
        per_r = {}
        for x in records:
            per_r.setdefault(x["r"], []).append(x[arm])
        xs = sorted(per_r)
        ys = [float(np.mean(per_r[k])) for k in xs]
        rho, p = spearmanr(xs, ys)
        print(f"  {arm:>15} vs r: rho={rho:+.3f} p={p:.4f} (n={len(xs)} depths, "
              f"expected to {expect})   means: "
              f"{[f'{v:+.3f}' for v in ys]}", flush=True)

    # THE STATISTIC THAT ACTUALLY MEASURES THE DECAY. B1b's signed mean above
    # returned rho = -0.321, p = 0.48 and read as "not detected", because the sign
    # of a donor's pull varies by donor and the signed mean CANCELS. The magnitude
    # rose three orders of magnitude over the same records. Fit |potency|.
    print("\n=== PRE-REGISTERED (B1c): |state_potency| ~ rho^(R-r) ===", flush=True)
    print(f"  B1b measured rho = 0.822 at ONE recipient (range 0.791-0.866 across "
          f"donor directions). Prediction: the same rate at {n_pairs} recipients, "
          f"which would make it a property of Huginn rather than of one prompt.",
          flush=True)
    print("  It must also land near the passive estimates: D94 orbit decay ~0.82 "
          "bias-corrected, D31 Arnoldi 0.79-0.81.", flush=True)

    FIT_R = [r for r in PATCH_R if r >= 8]   # r=0,4 sit on a ~5e-4 fp32 logit floor
    per_pair_rho = {}
    for x in records:
        per_pair_rho.setdefault(x["pair"], {})[x["r"]] = abs(x["state_potency"])
    rhos = []
    for pair, d in sorted(per_pair_rho.items()):
        pts = [(r, d[r]) for r in FIT_R if d.get(r, 0.0) > 0.0]
        if len(pts) < 4:
            print(f"  {pair}: only {len(pts)} usable points -- skipped", flush=True)
            continue
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.log(np.array([p[1] for p in pts], dtype=float))
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = intercept + slope * xs
        r2 = 1.0 - ((ys - pred) ** 2).sum() / max(1e-12, ((ys - ys.mean()) ** 2).sum())
        rho_fit = float(np.exp(-slope))       # potency ~ rho^(R-r) => slope = -log rho
        rhos.append(rho_fit)
        print(f"  {pair}: rho = {rho_fit:.4f}  (R^2 {r2:.3f}, {len(pts)} pts)", flush=True)

    if len(rhos) < MIN_RECIPIENTS:
        print(f"  VOID for the rho claim: only {len(rhos)} recipients yielded a "
              f"usable fit (need {MIN_RECIPIENTS}). Report the arms, not a rate.",
              flush=True)
    else:
        rr = np.array(rhos)
        print(f"\n  rho over {len(rr)} DISTINCT RECIPIENTS: median {np.median(rr):.4f}, "
              f"mean {rr.mean():.4f}, sd {rr.std(ddof=1):.4f}, "
              f"range {rr.min():.4f}-{rr.max():.4f}", flush=True)
        lo, hi = np.percentile(rr, [2.5, 97.5])
        print(f"  95% of recipients in [{lo:.4f}, {hi:.4f}]", flush=True)
        agree = int(((rr >= 0.79) & (rr <= 0.86)).sum())
        print(f"  {agree}/{len(rr)} recipients fall inside the 0.79-0.86 bracket set "
              f"by the two passive instruments", flush=True)

    json.dump({"records": records, "state_rho_per_recipient": rhos,
               "config": {"num_steps": NUM_STEPS, "patch_r": list(PATCH_R),
                          "n_pairs": n_pairs, "fit_r": FIT_R, "model": MODEL_ID}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
