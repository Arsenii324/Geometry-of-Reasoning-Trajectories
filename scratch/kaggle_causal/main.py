"""Kaggle kernel: is the counting register CAUSALLY USED, or only decodable?

THE GAP THIS CLOSES. Every positive result in this project is decoding. None is
interventional. A probe can read a feature the model does not use -- that is the
standard critique of the probing paradigm, and the standard answer is activation
patching / causal mediation. Until this runs, "Huginn maintains a counting
register" is only "count-correlated variance exists in the state."

THE TEST. Fit a readout w on unpatched answer-token states so that
w . h + b estimates the count. Then ADD alpha * w / (w . w) to the state before
the coda and head, which by construction shifts the READOUT's estimate by
exactly alpha. Decode the model's own emitted number.

    if the emitted count moves with alpha        -> the direction is causal
    if it does not move                          -> the register is a correlate

PRE-REGISTERED (fixed before running):
  * causal iff the slope of (emitted count) on alpha is > 0.3 with p < 0.01,
    over alpha in {-8,-4,-2,0,+2,+4,+8}.
  * A slope near 1.0 would mean the readout direction IS the model's own code.
  * CONTROL: the same patch along a RANDOM direction of equal norm must NOT
    move the emitted count. Without this control a positive result could be
    "any large perturbation changes the output".
  * SECOND CONTROL: patch along a direction that decodes a RANDOM SIGNED sum of
    the bits (which pass-2 showed is NOT represented, R2=0.13). If that moves
    the count as much as w does, the effect is not specific to the count.

Norm discipline: the state lies on a sphere of radius ~76.37 (RMSNorm). Patches
are reported as a fraction of that radius so their size is interpretable, and
the largest is kept small enough not to leave the shell the model operates on.
"""
import json
import random
import re
import subprocess

import numpy as np


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
M, NUM_STEPS = 64, 32
P_ONES = (0.25, 0.4, 0.5, 0.6, 0.75)
N_SEEDS = 24
ALPHAS = (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0)


def task(seed, p_one):
    rng = random.Random(seed * 7919 + int(p_one * 1000))
    bits = [1 if rng.random() < p_one else 0 for _ in range(M)]
    return {"prompt": "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:",
            "bits": bits, "total": sum(bits)}


def main():
    run("pip install -q 'transformers>=4.50,<4.54' scikit-learn scipy")
    import torch
    from sklearn.linear_model import Ridge
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cuda").eval()

    prompts = [task(s, p) for p in P_ONES for s in range(N_SEEDS)]
    print(f"{len(prompts)} prompts, all m={M}", flush=True)

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=h.device, dtype=torch.long)
        for b in model.transformer.coda:
            bi = bi - 1
            x = b(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def run_prompt(prompt, patch=None):
        """Return (answer-token state, emitted number) with an optional patch."""
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        freqs = model.freqs_cis[:, : ids.shape[1]]
        store = {}
        mod = model.transformer.core_block[-1]; mod._forward_hooks.clear()
        step = {"n": 0}

        def hook(m, i, o):
            step["n"] += 1
            if step["n"] == NUM_STEPS and patch is not None:
                o = o.clone()
                o[0, -1, :] = o[0, -1, :] + torch.as_tensor(patch, device=o.device, dtype=o.dtype)
            store["h"] = o.detach()
            return o
        h = mod.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
                logits = coda_head(store["h"], freqs)[0, -1].float()
        finally:
            h.remove()
        # decode the emitted number from the top token, space-prefixed convention
        top = int(logits.argmax())
        txt = tok.decode([top])
        m_ = re.findall(r"-?\d+", txt)
        return store["h"][0, -1, :].float().cpu().numpy(), (int(m_[0]) if m_ else None)

    # 1) unpatched pass: collect states, fit the readout
    states, totals, emitted0 = [], [], []
    for t in prompts:
        h, e = run_prompt(t["prompt"])
        states.append(h); totals.append(t["total"]); emitted0.append(e)
    X = np.stack(states); y = np.array(totals, float)
    R = float(np.linalg.norm(X, axis=1).mean())
    ok = [e for e in emitted0 if e is not None]
    print(f"state radius {R:.2f}; model emitted a number in {len(ok)}/{len(prompts)} cases; "
          f"unpatched accuracy {np.mean([e==t for e,t in zip(emitted0,totals) if e is not None]):.3f}",
          flush=True)

    ridge = Ridge(alpha=1e3).fit(X, y)
    w = ridge.coef_.astype(np.float64)
    unit = w / (w @ w)                       # adding alpha*unit shifts w.h by alpha
    rng = np.random.default_rng(0)
    rand = rng.normal(size=len(w)); rand *= np.linalg.norm(unit) / np.linalg.norm(rand)
    ybad = X @ rng.normal(size=len(w))       # a direction for an unrepresented target
    wbad = Ridge(alpha=1e3).fit(X, np.array([b for b in (np.array([p["bits"] for p in prompts],float) @ rng.choice([-1.0,1.0],M))])).coef_ \
           if False else Ridge(alpha=1e3).fit(X, np.array([p["bits"] for p in prompts],dtype=float) @ rng.choice([-1.0,1.0],M)).coef_
    ubad = wbad / (wbad @ wbad)
    ubad *= np.linalg.norm(unit) / np.linalg.norm(ubad)

    print(f"\npatch norms as a fraction of the state radius: "
          f"count {np.linalg.norm(8*unit)/R:.4f}, random {np.linalg.norm(8*rand)/R:.4f}", flush=True)

    out = []
    for name, direction in (("count_readout", unit), ("random_control", rand),
                            ("unrepresented_control", ubad)):
        print(f"\n--- patching along {name} ---", flush=True)
        for a in ALPHAS:
            em, base = [], []
            for t, h0 in zip(prompts, states):
                _, e = run_prompt(t["prompt"], patch=a * direction)
                if e is not None and abs(e) < 500:
                    em.append(e); base.append(t["total"])
            shift = float(np.mean(np.array(em) - np.array(base))) if em else float("nan")
            out.append({"direction": name, "alpha": a, "mean_emitted": float(np.mean(em)) if em else None,
                        "mean_shift_vs_truth": shift, "n": len(em)})
            print(f"   alpha={a:+6.1f}  mean emitted {np.mean(em) if em else float('nan'):7.2f}   "
                  f"shift vs truth {shift:+7.3f}  (n={len(em)})", flush=True)

    with open("causal.json", "w") as f:
        json.dump(out, f)

    from scipy.stats import linregress
    print("\n=== VERDICT ===")
    for name in ("count_readout", "random_control", "unrepresented_control"):
        r = [(o["alpha"], o["mean_shift_vs_truth"]) for o in out
             if o["direction"] == name and o["mean_shift_vs_truth"] == o["mean_shift_vs_truth"]]
        if len(r) > 2:
            lr = linregress([x for x, _ in r], [v for _, v in r])
            print(f"  {name:24s} slope {lr.slope:+.4f}  p={lr.pvalue:.4g}  R2={lr.rvalue**2:.3f}")
    print("\n  PRE-REGISTERED: causal iff count_readout slope > 0.3 with p < 0.01,")
    print("  AND both controls show a clearly smaller slope.")
    print("\nDONE")


if __name__ == "__main__":
    main()
