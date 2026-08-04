"""Per-position states for the trained AND untrained model, on identical prompts.

THE QUESTION
    D50 established that Huginn carries a real counting register -- the lagged
    running count decodes at cv R^2 = 0.782 from the per-position state, after
    position and current-token identity are residualised INSIDE each fold and the
    folds are grouped by prompt. It also established that D34/D36 measured the
    wrong direction: their `v` is 97% the current-token contrast and is orthogonal
    (|cos| = 0.0004) to the direction the count actually occupies.

    D40/D46 asked whether the register is architectural by comparing the two arms
    on `register_r` -- which D50 showed is a window artefact. So the architectural
    question has never been asked about the register that actually exists.

    This kernel supplies the missing half: per-position states from a RANDOMLY
    INITIALISED model on the SAME prompts, so the D50 analysis can be rerun on
    both arms with identical code.

WHY IT ONLY DUMPS STATES
    The analysis already exists, is tested, and runs on a laptop
    (`scripts/recheck_register_window.py`). Doing it here would duplicate code that
    could drift; doing it locally means both arms go through the exact same
    function. So the GPU does the one thing only it can do.

    `token_ids` are saved per prompt, because the digit window is DERIVED from them
    rather than hardcoded -- the off-by-one that invalidated D34/D36 came from
    assuming an offset when the token ids were available all along.

BOTH MODELS IN ONE RUN, trained first: two models is the capacity that empirically
works on a T4 (a 3.5B float32 model is ~14 GB against 14.56 GB, and the previous
model is not fully freed -- see geometry-rho-ckpt's header).
"""

import gc
import json
import random
import subprocess


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
M = 64
N_PROMPTS = 16
NUM_STEPS = 32          # Huginn's mean_recurrence; where the model is meant to run


def task(seed):
    rng = random.Random(seed * 7919)
    bits = [1 if rng.random() < 0.5 else 0 for _ in range(M)]
    return "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:", bits


def capture(model, tok, prompts, label):
    """Per-position final-block states for each prompt."""
    import numpy as np
    import torch
    out = []
    for i, (text, bits) in enumerate(prompts):
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        cap = []
        mod = model.transformer.core_block[-1]
        mod._forward_hooks.clear()
        h = mod.register_forward_hook(
            lambda m, inp, o, c=cap: c.append(o.detach()[0].float().cpu().numpy()))
        try:
            torch.manual_seed(0)
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        st = cap[-1]                       # [n_positions, hidden] at the last unroll
        np.save(f"{label}_seed{i}.npy", st.astype(np.float16))
        out.append({"name": f"{label}_seed{i}", "kind": "a", "seed": i,
                    "n_positions": int(st.shape[0]), "hidden": int(st.shape[1]),
                    "token_ids": [int(t) for t in ids[0].tolist()],
                    "bits": bits, "target": list(__import__("itertools").accumulate(bits))})
        print(f"    {label} seed{i}: {st.shape}, ||h_last||="
              f"{float(np.linalg.norm(st[-1])):.2f}", flush=True)
        del ids
        torch.cuda.empty_cache()
    return out


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    prompts = [task(s) for s in range(N_PROMPTS)]
    print(f"{len(prompts)} prompts, {len(tok(prompts[0][0]).input_ids)} tokens each",
          flush=True)

    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    meta = []

    for label in ("trained", "untrained"):
        print(f"\n=== {label} ===", flush=True)
        model = None
        try:
            if label == "trained":
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, revision=REVISION, trust_remote_code=True)
            else:
                torch.manual_seed(0)
                model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            model = model.to(torch.float32).to("cuda").eval()
            meta += capture(model, tok, prompts, label)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {label} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        finally:
            try:
                model = model.to("cpu") if model is not None else None
            except Exception:                                    # noqa: BLE001, S110
                pass
            del model
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print(f"  after cleanup: {free / 2**30:.2f} GiB free of "
                  f"{total / 2**30:.2f} GiB", flush=True)
            with open("meta.json", "w") as f:
                json.dump(meta, f)

    got = {m["name"].split("_")[0] for m in meta}
    print(f"\nsaved {len(meta)} trajectories from arms: {sorted(got)}")
    print("Analyse locally with scripts/recheck_register_window.py so both arms go "
          "through identical, already-tested code.")


main()
