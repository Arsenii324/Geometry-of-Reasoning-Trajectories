"""Kaggle kernel: plan_forward Step 3 -- does required DEPTH scale with difficulty?

H2 restated behaviourally: is there a task-dependent recurrent depth r* at
which the model starts preferring the correct answer? This touches none of the
geometry the audit dismantled.

MEASURE: teacher-forced log P(gold) - log P(distractor) at every unroll.
Continuous (so a threshold is locatable), exact for multi-token and negative
answers (where a first-token argmax check inspects only the '-' sign), and one
forward pass yields the whole r-curve because teacher forcing scores all answer
positions at once.

THE MARGIN, not raw log P: a model growing uniformly more confident would lift
log P(gold) with no computation, but it lifts the distractor equally, so the
margin stays flat.

MARGINALISED OVER SURFACE FORMS: ' 2' (token 402) and '2' (token 50) are
distinct atomic tokens, neither a prefix of the other, so the two continuations
are DISJOINT events and P(answer) is their sum. Scoring only the space-prefixed
form understates it.

EXCLUDES answer == 0: correctness in the existing v6 probe is perfectly
separated by whether the answer is zero (4/5 vs 0/8, all at unroll 1) -- a
prior on emitting '0', not computation.

RUN IN float32: the gate (D30) showed bf16 rounding truncates the informative
regime ~4.6x early, so a bf16 curve would be partly masked.
"""
import json
import random
import subprocess
import traceback

import numpy as np


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MAX_R = 64
M = 64                      # FIXED length -- this is the whole point
N_SEEDS = 40
P_ONES = (0.15, 0.3, 0.5, 0.7, 0.85)   # vary the COUNT at constant length


def make_fixed_count_task(p_one, seed=0):
    """Barannikov's design: m fixed, difficulty varied by the number of ones.

    THE CONFOUND THIS BREAKS. In `make_count_ones_task` the string length IS
    the difficulty parameter, so rank-corr(n_ops, seq_len) = exactly 1.000 and
    a positive r*-vs-difficulty result cannot be distinguished from
    r*-vs-prompt-length. Here every prompt has the same 64 symbols and the same
    token count; only the ANSWER varies. Any depth scaling must then be about
    the computation.
    """
    rng = random.Random(seed * 977 + int(p_one * 1000))
    seq = [1 if rng.random() < p_one else 0 for _ in range(M)]
    q = ". How many ones are in the sequence? A:"
    return {"prompt": "Sequence: " + " ".join(map(str, seq)) + q,
            "answer": sum(seq), "p_one": p_one}


def logsumexp(a, axis=0):
    m = np.nanmax(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.nansum(np.exp(a - m), axis=axis))


def threshold_depth(margin, frac=0.9):
    m = np.asarray(margin, float)
    if len(m) < 4 or not np.isfinite(m).any():
        return float("nan")
    final = float(np.nanmedian(m[-max(1, len(m) // 4):]))
    if not np.isfinite(final) or final <= 0:
        return float("nan")
    ok = m >= frac * final
    for i in range(len(m)):
        if ok[i] and ok[i:].all():
            return float(i + 1)
    return float("nan")


def main():
    run("pip install -q 'transformers>=4.50,<4.54' scipy")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available())
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cuda").eval()

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=h.device, dtype=torch.long)
        for block in model.transformer.coda:
            bi = bi - 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def curve(prompt, ids_ans):
        ids_p = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        ans = torch.tensor([ids_ans], device="cuda", dtype=ids_p.dtype)
        ids = torch.cat([ids_p, ans], dim=1)
        npr = ids_p.shape[1]
        freqs = model.freqs_cis[:, : ids.shape[1]]
        out = []
        mod = model.transformer.core_block[-1]; mod._forward_hooks.clear()

        def hook(m, i, o):
            with torch.no_grad():
                lp = torch.log_softmax(coda_head(o.detach(), freqs).float()[0], dim=-1)
                out.append(sum(lp[npr - 1 + k, t].item() for k, t in enumerate(ids_ans)))
        h = mod.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=MAX_R)
        finally:
            h.remove()
        a = np.asarray(out, float)
        return a[:MAX_R] if len(a) >= MAX_R else np.pad(a, (0, MAX_R - len(a)), constant_values=np.nan)

    def forms(v):
        seen, out = set(), []
        for pre in (" ", ""):
            i = tuple(tok(pre + str(v), add_special_tokens=False).input_ids)
            if i and i not in seen:
                seen.add(i); out.append(list(i))
        return out

    res = []
    seen_len = set()
    for p_one in P_ONES:
        for seed in range(N_SEEDS):
            try:
                t = make_fixed_count_task(p_one, seed=seed)
                seen_len.add(len(tok(t["prompt"], add_special_tokens=False).input_ids))
                a = t["answer"]
                if a == 0:
                    continue
                gold_len = len(forms(a)[0])
                dis = next((c for c in (a + 1, a - 1, a + 2, a - 2)
                            if c != 0 and c != a and len(forms(c)[0]) == gold_len),
                           a + 1 if a + 1 != 0 else a + 2)
                lg = logsumexp(np.stack([curve(t["prompt"], f) for f in forms(a)]), 0)
                ld = logsumexp(np.stack([curve(t["prompt"], f) for f in forms(dis)]), 0)
                m = lg - ld
                r = {"n_ops": a, "p_one": p_one, "seed": seed, "answer": a, "distractor": dis,
                     "r_star": threshold_depth(m),
                     "final_margin": float(np.nanmedian(m[-MAX_R // 4:])),
                     "margin_at_1": float(m[0]), "margin_max": float(np.nanmax(m))}
                res.append(r); print(json.dumps(r), flush=True)
            except Exception:
                traceback.print_exc()

    print("\n=== SUMMARY ===")
    print(f"  DISTINCT PROMPT TOKEN LENGTHS across all instances: {sorted(seen_len)}")
    print("  (a single value means the length confound is broken by construction)")
    solved = [r for r in res if np.isfinite(r["r_star"])]
    print(f"  instances: {len(res)}   with a threshold: {len(solved)}")
    print(f"  final margin > 0 (model prefers gold): "
          f"{sum(1 for r in res if r['final_margin'] > 0)}/{len(res)}")
    if solved:
        from scipy.stats import spearmanr
        by = {}
        for r in solved:
            by.setdefault(r["n_ops"], []).append(r["r_star"])
        print("  r* by difficulty:")
        for k in sorted(by):
            print(f"    n_ops={k:3d}: mean r* = {np.mean(by[k]):5.1f}  (n={len(by[k])})")
        xi = [r["answer"] for r in solved]; yi = [r["r_star"] for r in solved]
        rho, p = spearmanr(xi, yi)
        print(f"\n  THE TEST -- per-instance rho(r*, ANSWER) = {rho:+.3f}, p = {p:.5f}, "
              f"n={len(xi)} instances")
        print("  Prompt length is CONSTANT here, so a positive result cannot be length.")
        print(f"  VERDICT: {'required depth scales with the COUNT at fixed length' if p < 0.05 and rho > 0 else 'no evidence of depth scaling at fixed length'}")
    else:
        print("  No instance ever preferred the gold answer -> H2-depth is untestable "
              "on this task set, not refuted.")
    print("\nDONE")


if __name__ == "__main__":
    main()
