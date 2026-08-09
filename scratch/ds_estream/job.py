"""B1: the potency curve. Is there ANY working causal instrument in this project?

WHY THIS IS THE HIGHEST-VALUE GPU EXPERIMENT AVAILABLE. `docs/RESULT.md` currently
states that causal evidence is ABSENT, not negative. D95's activation patch failed
its own supply gate and was recorded VOID; its state-patch design was later shown
to be inert at both ends by construction, because donor and recipient were the SAME
prompt with different h_0 and therefore share an attractor. **D95(4b), and two
independent deep-research passes reaching the same conclusion from the DEQ
literature, all name cross-prompt e-stream patching as the correct next step.**

THE MECHANISM, READ FROM THE PINNED SOURCE. `core_block_forward` does:

    x = self.transformer.adapter(torch.cat([x, input_embeds], dim=-1))

once per unroll. So `input_embeds` -- the prelude output, this project's `e` -- is
the map's **parameter**, re-injected at every step, while `x` is its **state**.
Perturbing `x` perturbs an initial condition that a contraction erases; perturbing
`e` moves the fixed point `h*(e)` itself. A forward-PRE-hook on `transformer.adapter`
can replace the `e` half of its input with a donor's, which is the intervention.

SECOND SUBMISSION (D108 -> B1b). The first run ANSWERED the e arm and VOIDED the
state arm, so the pre-registration is updated here rather than left stale:

  * **e arm -- SETTLED by D108.** The instrument works: the preferred answer flips
    in 42/56 measurements, mean potency +3.973 nats. Potency is FLAT in r
    (rho = -0.071, p = 0.879), refuting the original prediction that it would fall
    -- 87% of the effect is present with only 8 unrolls left. It is re-run here
    unchanged, as a reproducibility check on a result that changed a headline.
  * **state arm -- VOID in the first run, by two bugs of mine.** It overwrote `x`
    at EVERY unroll from r onward (so the outcome could not depend on when
    patching began: exactly 8 distinct values across 56 measurements), and it
    captured the donor's state with `setdefault` on the FIRST adapter call, i.e.
    the donor's initial state rather than its state at the unroll being patched.
    Both are fixed below: a SINGLE injection at r, using the donor's state AT r.
  * **The surviving pre-registered prediction is for the fixed state arm:
    potency should INCREASE with r**, because a state perturbation decays as
    rho^(R-r) and a later patch has less remaining budget in which to be erased.
    A flat state arm would mean state interventions are inert at every depth,
    which together with D108's potent e arm is itself the parameter-vs-state
    dissociation this pair of arms exists to test.

A degeneracy guard now runs before any interpretation: if either arm returns a
single value per pair across all depths, it is reported as NOT measuring a
depth-dependent intervention rather than being silently analysed.

THE CALIBRATION IS THE POINT, NOT A PRELIMINARY. Per CLAUDE.md section 5, an
instrument must pass its own null before its results mean anything. Here the null
is: **can the intervention move the output AT ALL, at any r?** A patching null
interpreted before that question is answered is worthless -- which is exactly what
D95 nearly became.

DESIGN. Donor and recipient are DIFFERENT prompts with different gold answers, at
a VERIFIED-IDENTICAL token count (checked in-kernel, not asserted -- the shapes
must match for the swap to be defined, and D101's token gate caught a length claim
that was false). Family is `addk`, chosen because D107 showed it is the one ladder
that survives honest fixed-depth scoring (41.7% oracle -> 38.1% held-out) after
`nth_item_k` collapsed to 4.2%.

STATISTIC. The logit gap `logit(donor_gold) - logit(recipient_gold)` at the answer
position, read via the D71-validated `coda_head` at the final unroll. Potency is
that gap minus its unpatched baseline: positive means the intervention dragged the
output toward the donor's answer. Reported per (pair, r); the independent unit is
the PAIR, and `rigor.require_units` enforces it rather than counting rows.
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
N_PAIRS = 8


def addk_items():
    """`addk`: v + k with v + k <= 9, so every gold is a single token (D89-safe),
    and every prompt is the same shape. D107 identified this as the family that
    survives honest scoring."""
    out = []
    for v in range(0, 4):
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
    print(f"{len(items)} items -> {len(pairs)} length-matched donor/recipient pairs; "
          f"using {min(N_PAIRS, len(pairs))}", flush=True)
    pairs = pairs[:N_PAIRS]
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

    json.dump({"records": records,
               "config": {"num_steps": NUM_STEPS, "patch_r": list(PATCH_R),
                          "n_pairs": n_pairs, "model": MODEL_ID}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
