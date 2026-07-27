"""Kaggle kernel: plan_forward Step 1 -- is the post-convergence residual arithmetic?

Self-contained (does NOT clone the repo, so no push to the user's remote is
needed). Mirrors scripts/run_precision_check.py.

THE TEST, STRENGTHENED. The original plan compared bf16 against fp32 and
predicted the residual radius drops by 2^-16. Running THREE dtypes turns a
single ratio into a regression: if the floor is set by rounding, then

    floor  proportional to  ulp = 2^-mantissa_bits

    bfloat16 :  8 mantissa bits
    float16  : 11 mantissa bits   -> predicted floor 1/8 of bf16
    float32  : 24 mantissa bits   -> predicted floor 2^-16 of bf16

so log2(floor) against mantissa_bits should be a straight line of slope -1.
That is far more falsifiable than one ratio, and fp16 is memory-safe on a T4
even if fp32 does not fit (3.5B params x 4 bytes = 14 GB against 16 GB), so
the experiment degrades gracefully rather than failing.
"""
import json
import subprocess
import sys
import traceback

import numpy as np


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 128
N_TAIL = 20


def make_count_ones_task(n_ops, seed=0):
    import random
    rng = random.Random(seed)
    seq = [rng.choice([0, 1]) for _ in range(n_ops)]
    q = f". How many ones are in the first {n_ops} symbols? A:"
    return {"prompt": "Sequence: " + " ".join(map(str, seq)) + q, "answer": sum(seq)}


def extract(model, tok, prompt, num_steps, seed=0):
    import torch
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()
    lat = []
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    h = mod.register_forward_hook(lambda m, i, o: lat.append(o.detach()[0, -1, :].float().cpu()))
    try:
        with torch.no_grad():
            model(input_ids=ids, num_steps=num_steps)
    finally:
        h.remove()
    return torch.stack(lat).numpy()


def summarise(a):
    """Residual radius after convergence -- the quantity under test."""
    a = np.asarray(a, dtype=np.float64)
    step = np.linalg.norm(np.diff(a, axis=0), axis=1)
    floor_step = float(np.median(step[-len(step) // 4:]))
    ref = a[-N_TAIL:].mean(0)
    radius = np.linalg.norm(a - ref, axis=1)
    # converging regime: last index whose step exceeds 3x the floor
    above = np.where(step > 3 * floor_step)[0]
    end = int(above[-1]) + 1 if len(above) else 2
    return {
        "state_norm": float(np.linalg.norm(a, axis=1).mean()),
        "regime_end": end,
        "floor_step": floor_step,
        "residual_radius": float(radius[end:].mean()) if end < len(a) else float("nan"),
        "bf16_exact": bool(np.array_equal(
            a.astype(np.float32),
            __import__("torch").from_numpy(a.astype(np.float32)).to(
                __import__("torch").bfloat16).to(__import__("torch").float32).numpy())),
    }


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cases = [(8, 0), (32, 0), (32, 1)]
    dtypes = [("bfloat16", torch.bfloat16, 8), ("float16", torch.float16, 11),
              ("float32", torch.float32, 24)]

    results = []
    for name, dt, mant in dtypes:
        try:
            print(f"\n=== loading {name} ===", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, torch_dtype=dt,
                trust_remote_code=True, low_cpu_mem_usage=True).to("cuda").eval()
            for n_ops, seed in cases:
                t = make_count_ones_task(n_ops, seed=seed)
                a = extract(model, tok, t["prompt"], NUM_STEPS)
                s = summarise(a)
                s.update(dtype=name, mantissa_bits=mant, n_ops=n_ops, task_seed=seed)
                results.append(s)
                print(f"  n_ops={n_ops} seed={seed}: residual={s['residual_radius']:.6g} "
                      f"norm={s['state_norm']:.3f} regime_end={s['regime_end']}", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception:
            print(f"  {name} FAILED:", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()

    print("\n=== RESULT ===")
    for r in results:
        print(json.dumps(r), flush=True)

    by = {}
    for r in results:
        if np.isfinite(r["residual_radius"]):
            by.setdefault(r["dtype"], []).append(r["residual_radius"])
    print("\nmean residual radius by dtype:")
    for k, v in by.items():
        print(f"  {k:9s} {np.mean(v):.6g}")

    if len(by) >= 2:
        mant = {"bfloat16": 8, "float16": 11, "float32": 24}
        xs = np.array([mant[k] for k in by])
        ys = np.log2(np.array([np.mean(v) for v in by.values()]))
        if len(xs) >= 2:
            slope = np.polyfit(xs, ys, 1)[0]
            print(f"\nlog2(residual) vs mantissa_bits: slope = {slope:+.3f}")
            print("  PREDICTED -1.000 if the residual is rounding noise")
            print("  slope ~ 0 would mean a genuine invariant set, and the "
                  "regime-separation framework must be re-derived.")
    print("\nDONE")


if __name__ == "__main__":
    main()
