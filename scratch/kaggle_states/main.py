"""Kaggle kernel: DUMP per-position latents for Barannikov Task a/b.

Purpose is deliberately narrow: get the raw data off the GPU once, so every
subsequent analysis runs locally and offline. The register probe (D32) had to
re-run the model for each new question, which is both slow and a provenance
problem -- the conclusions existed but the data behind them did not.

Saves float16 to keep the output small; the states are float32-computed and
the analyses here care about directions and correlations, not the last bits.
Also saves the per-position targets and the bit/symbol sequences so nothing
has to be regenerated from a seed and hoped to match.

float32 compute, because D30 showed bf16 truncates the informative regime ~4.6x
early.
"""
import itertools
import json
import random
import subprocess

import numpy as np


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
M = 64
NUM_STEPS = 64
N_SEEDS = 16


def task_a(seed):
    rng = random.Random(seed)
    bits = [1 if rng.random() < 0.5 else 0 for _ in range(M)]
    return {"prompt": "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:",
            "bits": bits, "target": list(itertools.accumulate(bits))}


def task_b(seed, balanced):
    rng = random.Random(seed)
    m = M - (M % 2) if balanced else M
    if balanced:
        sym, op = [], 0
        for i in range(m):
            rem = m - i
            t = True if op == 0 else (False if op == rem else rng.random() < 0.5)
            sym.append("(" if t else ")"); op += 1 if t else -1
    else:
        sym = [rng.choice(["(", ")"]) for _ in range(m)]
    return {"prompt": "Sequence: " + " ".join(sym) + ". What is the maximum nesting depth? A:",
            "symbols": sym,
            "target": list(itertools.accumulate(1 if s == "(" else -1 for s in sym))}


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cuda").eval()

    def positions(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        cap = []
        mod = model.transformer.core_block[-1]; mod._forward_hooks.clear()
        h = mod.register_forward_hook(
            lambda m, i, o: cap.append(o.detach()[0].float().cpu().numpy()))
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        # token ids too: the probe must know which token each position IS
        return cap[-1], ids[0].cpu().numpy()

    meta = []
    for kind, gen in (("a", lambda s: task_a(s)),
                      ("b_bal", lambda s: task_b(s, True)),
                      ("b_unbal", lambda s: task_b(s, False))):
        for seed in range(N_SEEDS):
            t = gen(seed)
            st, ids = positions(t["prompt"])
            name = f"{kind}_seed{seed}"
            np.save(f"{name}.npy", st.astype(np.float16))
            meta.append({
                "name": name, "kind": kind, "seed": seed,
                "n_positions": int(st.shape[0]), "hidden": int(st.shape[1]),
                "token_ids": ids.tolist(),
                "target": t["target"],
                "units": t.get("bits", t.get("symbols")),
                "prompt": t["prompt"], "num_steps": NUM_STEPS, "dtype": "float32-compute",
            })
            print(f"  {name}: states {st.shape}, {len(t['target'])} targets", flush=True)

    with open("meta.json", "w") as f:
        json.dump(meta, f)
    print(f"\nsaved {len(meta)} state arrays + meta.json")
    print("DONE")


if __name__ == "__main__":
    main()
