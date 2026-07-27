"""Kaggle kernel: is the counting register BUILT UP over recurrent depth?

THE QUESTION, and why it is the one worth asking now.
  Two results stand independently:
    * required recurrent depth scales with difficulty (D35, rho=+0.225 at
      constant prompt length),
    * a counting register exists in the per-position latents and behaves as a
      Z-action (D34/D36, r=+0.22..+0.29, 16/16 sign consistency).
  Nothing yet connects them. If the recurrent unrolling is what CONSTRUCTS the
  register, its decodability should RISE with unroll count. If the register is
  already present after one unroll and merely persists, then depth is doing
  something else and the two findings are unrelated.

  This is a within-run measurement: one forward pass per string, states captured
  at every checkpoint, so the comparison across depths has no run-to-run noise.

DESIGN NOTES
  * float32, because D30 showed bf16 truncates the informative regime ~4.6x
    early -- a bf16 depth sweep would flatten exactly the trend under test.
  * The SAME direction v is refit at each depth. Refitting is the honest choice:
    if v rotated with depth, holding it fixed at r=64 would understate early
    depths and manufacture the trend. `v_cos_to_final` records how much v
    actually moves, so the reader can see whether refitting mattered.
  * Saves raw states at each checkpoint so every downstream analysis runs
    offline, per the lesson from the earlier kernels.
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
MAX_R = 64
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64)
N_SEEDS = 12


def task_a(seed):
    rng = random.Random(seed)
    bits = [1 if rng.random() < 0.5 else 0 for _ in range(M)]
    return {"prompt": "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:",
            "units": bits, "target": list(itertools.accumulate(bits))}


def task_b(seed):
    rng = random.Random(seed)
    m = M - (M % 2)
    sym, op = [], 0
    for i in range(m):
        rem = m - i
        t = True if op == 0 else (False if op == rem else rng.random() < 0.5)
        sym.append("(" if t else ")"); op += 1 if t else -1
    return {"prompt": "Sequence: " + " ".join(sym) + ". What is the maximum nesting depth? A:",
            "units": sym,
            "target": list(itertools.accumulate(1 if s == "(" else -1 for s in sym))}


def resid(y, ctrls):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in ctrls])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def register_r(states, inc, value, offset):
    """Increment direction refit at THIS depth, then the controlled correlation."""
    n = len(inc)
    seg = states[offset:offset + n]
    d = np.diff(seg, axis=0)
    i2 = inc[1:]
    if i2.all() or not i2.any():
        return np.nan, None
    v = d[i2].mean(0) - d[~i2].mean(0)
    nv = np.linalg.norm(v)
    if nv == 0:
        return np.nan, None
    v = v / nv
    proj = seg @ v
    pos = np.arange(1, n + 1, dtype=float)
    cur = inc.astype(float)
    lagged = np.concatenate([[0.0], np.asarray(value, dtype=float)[:-1]])
    r = float(np.corrcoef(resid(proj, [pos, cur]), resid(lagged, [pos, cur]))[0, 1])
    return r, v


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cuda").eval()

    def states_by_depth(prompt):
        """One forward pass; capture the state at every unroll."""
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        cap = []
        mod = model.transformer.core_block[-1]; mod._forward_hooks.clear()
        h = mod.register_forward_hook(
            lambda m, i, o: cap.append(o.detach()[0].float().cpu().numpy()))
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=MAX_R)
        finally:
            h.remove()
        return cap

    out, meta = [], []
    for kind, gen in (("a", task_a), ("b_bal", task_b)):
        for seed in range(N_SEEDS):
            t = gen(seed)
            caps = states_by_depth(t["prompt"])
            units = np.array(t["units"])
            inc = (units == "1") | (units == 1) | (units == "(")
            value = np.asarray(t["target"], dtype=float)
            vs = {}
            for r_ in CHECKPOINTS:
                if r_ > len(caps):
                    continue
                st = caps[r_ - 1]
                # offset 4 established across all strings in the earlier run
                rr, v = register_r(st, inc, value, 4)
                vs[r_] = v
                out.append({"kind": kind, "seed": seed, "unroll": r_, "register_r": rr})
                if r_ == CHECKPOINTS[-1]:
                    np.save(f"{kind}_seed{seed}_r{r_}.npy", st.astype(np.float16))
            vf = vs.get(CHECKPOINTS[-1])
            if vf is not None:
                for r_, v in vs.items():
                    if v is not None:
                        for rec in out:
                            if rec["kind"] == kind and rec["seed"] == seed and rec["unroll"] == r_:
                                rec["v_cos_to_final"] = float(abs(v @ vf))
            meta.append({"kind": kind, "seed": seed, "prompt": t["prompt"],
                         "units": [str(u) for u in t["units"]], "target": t["target"]})
            print(f"  {kind} seed{seed} done", flush=True)

    with open("meta.json", "w") as f:
        json.dump(meta, f)
    with open("register_vs_depth.json", "w") as f:
        json.dump(out, f)

    print("\n=== register decodability vs recurrent depth ===")
    for kind in ("a", "b_bal"):
        print(f"\n  {kind}:")
        for r_ in CHECKPOINTS:
            v = [x["register_r"] for x in out
                 if x["kind"] == kind and x["unroll"] == r_ and np.isfinite(x["register_r"])]
            vc = [x.get("v_cos_to_final", np.nan) for x in out
                  if x["kind"] == kind and x["unroll"] == r_]
            if v:
                print(f"    r={r_:3d}: register_r = {np.mean(v):+.4f} "
                      f"(sd {np.std(v):.4f}, n={len(v)})   |cos(v, v_final)| = "
                      f"{np.nanmean(vc):.3f}")
        from scipy.stats import spearmanr
        xs = [x["unroll"] for x in out if x["kind"] == kind and np.isfinite(x["register_r"])]
        ys = [x["register_r"] for x in out if x["kind"] == kind and np.isfinite(x["register_r"])]
        if len(set(xs)) > 2:
            rho, p = spearmanr(xs, ys)
            print(f"    rho(unroll, register_r) = {rho:+.4f}, p = {p:.3e}, n={len(xs)}")
            print(f"    => {'the register is BUILT UP over recurrent depth' if p < 0.05 and rho > 0 else 'no evidence the register develops with depth'}")
    print("\nDONE")


if __name__ == "__main__":
    main()
