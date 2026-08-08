"""D76's one-draw limit, closed: five independent untrained initialisations.

D76 found that training flips the sign of the consecutive-step cosine (-0.379
untrained -> +0.541 trained), with completely disjoint distributions at every
window length and both token positions. Every untrained orbit in it came from a
SINGLE `from_config` draw, so the effective n for any claim about untrained MODELS
is 1 -- D52 used five random inits for exactly this reason, and its headline (rho
0.7048 -> 0.8577, Mann-Whitney U=0) is credible because of it.

This job re-runs the untrained arm at five independent initialisations on the same
prompts. Two outcomes, both worth having:
  * the five draws agree -> D76's cosine gap is a property of TRAINING, and the
    comparison against the trained model stands at n=5 untrained weight-sets;
  * the draws disagree -> the gap is a property of one weight sample, and D76(6)
    must be amended rather than defended.

"Untrained" is Huginn's OWN depth-scaled scheme invoked by `from_config`
(std = sqrt(2/5d), out-projections scaled by 1/sqrt(2*d_eff) with d_eff = 132), so
these are five draws of Huginn at step 0, not a generic HF init. Varying the DRAW
is not the same as varying the init FAMILY, which directions.md B2 records as still
open.

A FREE REPRODUCIBILITY CONTROL: init seed 0 is the SAME draw `geom-bank` used, so
`untrained0`'s orbits must come back bit-identical to that job's untrained arm. If
they do not, `from_config` is not deterministic under `torch.manual_seed` and every
untrained result in this project -- D40, D41, D48, D52, D53, D70, D73, D76 -- rests
on an arm that cannot be reproduced. Costs nothing to check and would be the most
consequential possible negative.

Records states and metadata, computes nothing (B14).
"""

import json
import os
import random
import time
import traceback

import numpy as np

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 128        # matches the existing ns=128 banked data
N_OPS = (4, 8, 16, 32)
SEEDS = (0,)            # one task seed; the variation under test is the WEIGHT draw
OUTDIR = os.environ.get("RESULT_DIR", ".")
INIT_SEEDS = (0, 1, 2, 3, 4)   # the five independent weight draws


def make_variants(n_ops, seed=0):
    """Verbatim from src/traj_geom/shapes/synthetic.py; pinned by tests/test_kernel_tasks.py."""
    rng = random.Random(seed)
    ops = [rng.choice([1, -1]) for _ in range(n_ops)]
    body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1." for o in ops)
    return {
        "track": (body + " Final total? A:", str(sum(ops))),
        "local": (body + " What was the last instruction? A:",
                  "Add" if ops[-1] > 0 else "Subtract"),
    }


def make_count_ones(n_ops, seed=0):
    """The family the existing banked trajectories use, kept for continuity."""
    rng = random.Random(seed)
    bits = [rng.randint(0, 1) for _ in range(n_ops)]
    q = f". How many ones are in the first {n_ops} symbols? A:"
    return ("Sequence: " + " ".join(map(str, bits)) + q, str(sum(bits)))


def main():
    import subprocess
    subprocess.check_call("pip install -q 'transformers>=4.50,<4.54'", shell=True)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    wb = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wb = wandb.init(project="geometry-of-reasoning-trajectories",
                            name="ds-bank-" + str(int(time.time())),
                            config={"num_steps": NUM_STEPS, "n_ops": list(N_OPS),
                                    "seeds": list(SEEDS)})
        except Exception as exc:
            print(f"wandb unavailable: {exc}", flush=True)

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)

    def coda_head(model, h, freqs):
        """ln_f -> coda -> ln_f -> lm_head. TWO ln_f calls; D71 validated this exact
        tail as bit-identical to the model's own logits at the final unroll."""
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    tasks = []
    for n in N_OPS:
        for s in SEEDS:
            v = make_variants(n, seed=s)
            for kind in ("track", "local"):
                tasks.append((kind, n, s, v[kind][0], v[kind][1]))
            p, g = make_count_ones(n, seed=s)
            tasks.append(("count_ones", n, s, p, g))

    manifest = []
    t0 = time.time()
    for arm in [f"untrained{k}" for k in INIT_SEEDS]:
        print(f"\n=== {arm} ===", flush=True)
        if arm == "trained":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, torch_dtype=torch.float32,
                trust_remote_code=True, low_cpu_mem_usage=True)
        else:
            # Huginn's OWN _init_weights, invoked by from_config: std = sqrt(2/5d)
            # with out-projections scaled by 1/sqrt(2*d_eff), d_eff = 132. So
            # "untrained" here is Huginn at step 0 under the paper's scheme, not a
            # generic HF init (directions.md B2).
            torch.manual_seed(int(arm.replace('untrained', '')))
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            model = model.to(torch.float32)
        model = model.to("cuda").eval()

        for kind, n_ops, seed, prompt, gold in tasks:
            tag = f"{arm}_{kind}_n{n_ops}_ns{NUM_STEPS}_seed{seed}"
            try:
                ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
                g_ids = tok(gold, add_special_tokens=False).input_ids
                n_tok = ids.shape[1]
                mid = max(1, n_tok // 2)
                freqs = model.freqs_cis[:, :n_tok]
                states_last, ranks = [], []
                mod = model.transformer.core_block[-1]
                mod._forward_hooks.clear()

                def hook(_m, _i, o):
                    with torch.no_grad():
                        st = o.detach()
                        states_last.append(st[0, -1, :].float().cpu().numpy())
                        lg = coda_head(model, st, freqs).float()
                        row = torch.log_softmax(lg[0, -1], dim=-1)
                        ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)

                h = mod.register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        model(input_ids=ids, num_steps=NUM_STEPS)
                finally:
                    h.remove()

                last = np.stack(states_last).astype(np.float32)
                np.save(os.path.join(OUTDIR, tag + "_last.npy"), last)
                bf16_exact = bool(np.array_equal(
                    last, torch.from_numpy(last).to(torch.bfloat16).to(torch.float32).numpy()))
                rec = {"tag": tag, "arm": arm, "task": kind, "n_ops": n_ops,
                       "seed": seed, "num_steps": NUM_STEPS, "prompt": prompt,
                       "gold": gold, "n_tokens": int(n_tok), "mid_index": int(mid),
                       "rank_curve": ranks, "best_rank": int(min(ranks)),
                       "correct_any_depth": bool(min(ranks) == 1),
                       "shape": list(last.shape), "bf16_exact": bf16_exact,
                       "model_id": MODEL_ID,
                       "model_revision": REVISION if arm == "trained" else None,
                       "ok": True}
            except Exception as exc:
                rec = {"tag": tag, "arm": arm, "task": kind, "n_ops": n_ops,
                       "seed": seed, "ok": False,
                       "why": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            manifest.append(rec)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as fh:
                json.dump(manifest, fh)
            print(json.dumps({k: rec[k] for k in
                              ("tag", "best_rank", "correct_any_depth", "n_tokens",
                               "bf16_exact", "ok") if k in rec}), flush=True)
            if wb:
                wb.log({"banked": len(manifest)})
        del model
        torch.cuda.empty_cache()

    ok = sum(1 for r in manifest if r.get("ok"))
    print(f"\n=== banked {ok}/{len(manifest)} in {(time.time()-t0)/60:.1f} min ===",
          flush=True)
    for arm in ("trained", "untrained"):
        sub = [r for r in manifest if r.get("ok") and r["arm"] == arm]
        if sub:
            acc = sum(r["correct_any_depth"] for r in sub) / len(sub)
            print(f"  {arm}: {len(sub)} orbits, gold reaches rank 1 at some depth "
                  f"in {acc:.0%} (descriptive only)", flush=True)
    if wb:
        wb.finish()
    print("DONE", flush=True)


main()
