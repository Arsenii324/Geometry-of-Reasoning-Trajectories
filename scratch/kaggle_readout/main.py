"""Kaggle kernel: if the register is complete at r=1, what is depth FOR?

THE QUESTION. D37 found a sharp dissociation: the per-position counting
register's decodability is flat from unroll 1 (rho=+0.006, p=0.958) while only
its DIRECTION rotates (rho=+0.990). Yet D35 found that harder instances need
more recurrent depth (rho=+0.225 at constant prompt length). Those cannot both
be about building the register.

THE HYPOTHESIS UNDER TEST. Depth is spent on READOUT -- moving the register
from the string positions, where it already exists, into the ANSWER token,
where it has to be emitted. The register is distributed across 64 positions;
the answer is one token. Aggregating it is a different job from computing it.

  PREDICTION: decodability of the TOTAL count from the ANSWER-TOKEN state
  should RISE with unroll count, in contrast to the per-position register,
  which is flat. If it is also flat, readout is not what depth buys and the
  question stays open.

WHY THIS IS A CLEAN CONTRAST. The two measurements differ only in WHERE the
state is read:
    per-position register (D37): states at the 64 symbol positions, target
        y_i, one regression per string, flat in r.
    answer-token readout (here): state at the final position, target y_m, one
        regression ACROSS strings, measured at the same checkpoints.
Same model, same forward passes, same depths -- so a difference between them
is about location, not about method.

CONTROLS
  * Prompt length is FIXED at m=64, so nothing here can be length.
  * A permutation null (shuffled totals) at every depth, because R^2 from a
    5280-dim ridge on ~40 samples is optimistic even under cross-validation.
  * The answer token is the LAST position, which is also where every geometric
    metric in this project was measured and found to merely converge -- so a
    positive here would say that position carries content the geometry missed.

float32, because bf16 truncates the informative regime ~4.6x early (D30).
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
CHECKPOINTS = (1, 2, 4, 8, 16, 24, 32, 48, 64)
N_SEEDS = 44
P_ONES = (0.2, 0.35, 0.5, 0.65, 0.8)


def task(seed, p_one):
    rng = random.Random(seed * 7919 + int(p_one * 1000))
    bits = [1 if rng.random() < p_one else 0 for _ in range(M)]
    return {"prompt": "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:",
            "bits": bits, "total": sum(bits),
            "running": list(itertools.accumulate(bits))}


def loo_r2(X, y, alpha=1e3):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    pred = cross_val_predict(mdl, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=0))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    run("pip install -q 'transformers>=4.50,<4.54' scikit-learn scipy")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cuda").eval()

    answer_states = {r: [] for r in CHECKPOINTS}   # [n_prompts, hidden] per depth
    totals, lens = [], set()
    n = 0
    for p_one in P_ONES:
        for seed in range(N_SEEDS):
            t = task(seed, p_one)
            ids = tok(t["prompt"], return_tensors="pt").input_ids.to("cuda")
            lens.add(int(ids.shape[1]))
            cap = []
            mod = model.transformer.core_block[-1]; mod._forward_hooks.clear()
            h = mod.register_forward_hook(
                lambda m, i, o: cap.append(o.detach()[0, -1, :].float().cpu().numpy()))
            try:
                with torch.no_grad():
                    model(input_ids=ids, num_steps=MAX_R)
            finally:
                h.remove()
            for r in CHECKPOINTS:
                if r <= len(cap):
                    answer_states[r].append(cap[r - 1])
            totals.append(t["total"])
            n += 1
            if n % 40 == 0:
                print(f"  {n} prompts done", flush=True)

    y = np.array(totals, dtype=float)
    print(f"\nprompts: {n}   distinct token lengths: {sorted(lens)}   "
          f"totals span {y.min():.0f}..{y.max():.0f}")
    np.save("answer_states_r64.npy", np.stack(answer_states[CHECKPOINTS[-1]]).astype(np.float16))
    with open("meta.json", "w") as f:
        json.dump({"totals": totals, "checkpoints": list(CHECKPOINTS),
                   "token_lengths": sorted(lens), "m": M, "n_prompts": n}, f)

    rng = np.random.default_rng(0)
    out = []
    per_instance = {}   # depth -> per-prompt |CV prediction error|, for the D35 link
    print("\n=== can the TOTAL be read off the ANSWER TOKEN, and does it improve with depth? ===")
    print(f"  {'r':>4} {'R2':>9} {'null mean':>11} {'null p95':>10} {'p':>8}")
    for r in CHECKPOINTS:
        X = np.stack(answer_states[r])
        # per-instance CV error at THIS depth: does readout fail worse on harder
        # instances while the curve is still climbing? At r=64 it does not
        # (saturated); the question is whether it does at low r, which is the
        # direct link to D35's depth requirement.
        from sklearn.linear_model import Ridge as _R
        from sklearn.model_selection import KFold as _K
        from sklearn.model_selection import cross_val_predict as _cvp
        from sklearn.pipeline import make_pipeline as _mp
        from sklearn.preprocessing import StandardScaler as _S
        _p = _cvp(_mp(_S(), _R(alpha=1e3)), X, y, cv=_K(5, shuffle=True, random_state=0))
        per_instance[r] = np.abs(y - _p).tolist()
        r2 = loo_r2(X, y)
        null = np.array([loo_r2(X, rng.permutation(y)) for _ in range(60)])
        p = float(np.mean(null >= r2))
        out.append({"unroll": r, "r2": r2, "null_mean": float(null.mean()),
                    "null_p95": float(np.percentile(null, 95)), "p": p})
        print(f"  {r:>4} {r2:>+9.4f} {null.mean():>+11.4f} {np.percentile(null,95):>+10.4f} "
              f"{p:>8.3f}", flush=True)

    with open("readout_vs_depth.json", "w") as f:
        json.dump(out, f)
    with open("per_instance_error.json", "w") as f:
        json.dump({"totals": totals, "error_by_depth": per_instance}, f)

    from scipy.stats import spearmanr as _sp
    print("\n=== does readout fail worse on HARDER instances, and where? ===")
    print("    (the direct link to D35: if hard instances are read out worse while")
    print("     the curve is climbing, that is why they need more depth)")
    print(f"  {'r':>4} {'rho(count, |err|)':>19} {'p':>9} {'mean |err|':>12}")
    for r in CHECKPOINTS:
        e = np.array(per_instance[r])
        rho_, p_ = _sp(y, e)
        print(f"  {r:>4} {rho_:>+19.4f} {p_:>9.4f} {e.mean():>12.3f}")

    from scipy.stats import spearmanr
    xs = [o["unroll"] for o in out]; ys = [o["r2"] for o in out]
    rho, pv = spearmanr(xs, ys)
    print(f"\n  rho(unroll, readout R2) = {rho:+.4f}, p = {pv:.4f}")
    print(f"  R2 at r=1: {ys[0]:+.4f}   at r=64: {ys[-1]:+.4f}   change {ys[-1]-ys[0]:+.4f}")
    print()
    if pv < 0.05 and rho > 0:
        print("  => READOUT IMPROVES WITH DEPTH. The register exists from r=1 (D37) but")
        print("     getting it into the answer token takes recurrence -- which is a")
        print("     concrete answer to what the extra depth in D35 is buying.")
    else:
        print("  => readout does NOT improve with depth either. Then neither building")
        print("     the register nor aggregating it explains D35, and the question of")
        print("     what depth buys stays open -- report it as such.")
    print("\nDONE")


if __name__ == "__main__":
    main()
