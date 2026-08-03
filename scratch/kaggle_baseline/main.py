"""Kaggle kernel: the two controls this project never ran.

O1 -- ARCHITECTURE vs LEARNING. Every result in this project is from one trained
checkpoint. Nothing establishes that the counting register, the readout curve or
the positional structure are consequences of TRAINING rather than of a randomly
initialised transformer with the same architecture. An untrained model with
identical shape, tokenizer and normalisation is the control. If it shows the
same structure, the findings are about the architecture and the "the model
counts" reading collapses.

O2 -- IS THE REGISTER DIRECTION JUST A TOKEN-EMBEDDING CONTRAST?  v is defined
as mean(delta | symbol=1) - mean(delta | symbol=0). The obvious deflationary
reading is that this is simply e('1') - e('0'), the embedding difference, which
would make it a fact about the tokenizer rather than about a register. The
correlation control (against y_{i-1}) argues otherwise but does not bound the
fraction of v that is embedding contrast. Here we compute the contrast directly
and take the cosine.

Both controls are deflationary by design: they are the tests most likely to
overturn the project's positive results, which is why they are the ones worth
spending GPU on.
"""
import gc
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
M, NUM_STEPS, N_SEEDS = 64, 64, 220
P_ONES = (0.2, 0.35, 0.5, 0.65, 0.8)   # 44 seeds x 5 = 220, matching the original


def task(seed, p_one):
    rng = random.Random(seed * 7919 + int(p_one * 1000))
    bits = [1 if rng.random() < p_one else 0 for _ in range(M)]
    return {"prompt": "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:",
            "bits": bits, "running": list(itertools.accumulate(bits))}


def resid(y, ctrls):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in ctrls])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def r2_cv(X, y):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=1e3)), X, y,
                          cv=KFold(5, shuffle=True, random_state=0))
    d = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / d) if d > 1e-12 else float("nan")


def analyse(model, tok, label):
    """Register direction, register correlation, readout R2, positional k=1."""
    import torch
    answer, per_pos, bits_all = [], [], []
    prompts = [task(s, p) for p in P_ONES for s in range(N_SEEDS // len(P_ONES))]
    for t in prompts:
        ids = tok(t["prompt"], return_tensors="pt").input_ids.to("cuda")
        cap = []
        mod = model.transformer.core_block[-1]; mod._forward_hooks.clear()
        h = mod.register_forward_hook(lambda m, i, o: cap.append(o.detach()[0].float().cpu().numpy()))
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        st = cap[-1]
        answer.append(st[-1]); per_pos.append(st); bits_all.append(t["bits"])
    A = np.stack(answer); B = np.array(bits_all, float); tot = B.sum(1)

    # register direction + controlled correlation, per prompt
    vs, rs = [], []
    for st, b in zip(per_pos, bits_all):
        bb = np.array(b); n = len(bb); seg = st[4:4 + n]
        if len(seg) < n: continue
        d = np.diff(seg, axis=0); i2 = bb[1:] == 1
        if i2.all() or not i2.any(): continue
        v = d[i2].mean(0) - d[~i2].mean(0)
        nv = np.linalg.norm(v)
        if nv == 0: continue
        v /= nv; vs.append(v)
        proj = seg @ v; pos = np.arange(1, n + 1, dtype=float)
        y = np.cumsum(bb).astype(float); lag = np.concatenate([[0.0], y[:-1]])
        rs.append(np.corrcoef(resid(proj, [pos, bb.astype(float)]),
                              resid(lag, [pos, bb.astype(float)]))[0, 1])
    V = np.stack(vs); Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    cos_pair = (Vn @ Vn.T)[np.triu_indices(len(Vn), 1)]

    ctrl = np.column_stack([np.ones(len(tot)), tot])
    i = np.arange(M)
    k1 = r2_cv(A, resid(B @ np.cos(2 * np.pi * i / M), [tot]))
    out = {
        "label": label, "n_prompts": len(A),
        "v_pairwise_cos": float(cos_pair.mean()),
        "register_r": float(np.mean(rs)), "register_sign_frac": float(np.mean(np.array(rs) > 0)),
        "readout_r2_count": r2_cv(A, tot),
        "readout_r2_signed": float(np.mean([r2_cv(A, B @ np.random.default_rng(q).choice([-1.0, 1.0], M))
                                            for q in range(12)])),
        "positional_k1_resid_r2": k1,
        "v_mean": V.mean(0).tolist(),
    }
    # THE DISCRIMINATING MEASUREMENT the first version omitted: is v the raw
    # token-embedding contrast? Trained gave 0.0078 (it is not). If the untrained
    # model's v IS the contrast, the two models reach similar summary statistics
    # by DIFFERENT mechanisms and "untrained reproduces it" is a false equivalence.
    import torch as _t
    with _t.no_grad():
        _e1 = model.transformer.wte(_t.tensor([tok(" 1", add_special_tokens=False).input_ids],
                                              device="cuda"))[0, 0].float().cpu().numpy()
        _e0 = model.transformer.wte(_t.tensor([tok(" 0", add_special_tokens=False).input_ids],
                                              device="cuda"))[0, 0].float().cpu().numpy()
    _c = _e1 - _e0; _c /= np.linalg.norm(_c)
    _v = V.mean(0); _v /= np.linalg.norm(_v)
    out["cos_v_embedding_contrast"] = abs(float(_v @ _c))
    print(json.dumps({k: v for k, v in out.items() if k != "v_mean"}), flush=True)
    return out


def main():
    run("pip install -q 'transformers>=4.50,<4.54' scikit-learn scipy")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

    print("\n=== TRAINED ===", flush=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION,
        torch_dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True).to("cuda").eval()
    trained = analyse(m, tok, "trained")

    # O2: the token-embedding contrast, from the trained model
    with torch.no_grad():
        i1 = tok(" 1", add_special_tokens=False).input_ids
        i0 = tok(" 0", add_special_tokens=False).input_ids
        e1 = m.transformer.wte(torch.tensor([i1], device="cuda"))[0, 0].float().cpu().numpy()
        e0 = m.transformer.wte(torch.tensor([i0], device="cuda"))[0, 0].float().cpu().numpy()
    emb = e1 - e0; emb /= np.linalg.norm(emb)
    v = np.array(trained["v_mean"]); v /= np.linalg.norm(v)
    print(f"\n  O2: |cos(v, e('1') - e('0'))| = {abs(float(v @ emb)):.4f}")
    print("      near 1 => v IS the embedding contrast (deflationary)")
    print("      near 0 => v is not the tokenizer's doing", flush=True)

    del m; gc.collect(); torch.cuda.empty_cache()

    print("\n=== UNTRAINED (same architecture, random weights) ===", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    torch.manual_seed(0)
    mu = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(torch.float32).to("cuda").eval()
    untrained = analyse(mu, tok, "untrained")

    print("\n=== VERDICT ===")
    for k in ("v_pairwise_cos", "register_r", "register_sign_frac",
              "readout_r2_count", "readout_r2_signed", "positional_k1_resid_r2",
              "cos_v_embedding_contrast"):
        print(f"  {k:26s} trained {trained[k]:+.4f}   untrained {untrained[k]:+.4f}")
    print()
    print("  If the untrained column matches, these are architectural facts about a")
    print("  normalised residual stream, not findings about a trained counter.")
    with open("baseline.json", "w") as f:
        json.dump({"trained": trained, "untrained": untrained,
                   "cos_v_emb": abs(float(v @ emb))}, f)
    print("\nDONE")


if __name__ == "__main__":
    main()
