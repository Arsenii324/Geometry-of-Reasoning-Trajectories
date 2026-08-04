"""rho on the operator for TWO fresh RANDOM INITS. Closes the last hole in D44/D51.

WHY THIS EXISTS
    D44 claimed "training slows the contraction" and supported it with a Welch t
    over 12 PROMPTS. That was pseudoreplication: there was n=1 weight-set per arm,
    so the test measured prompt variation, not training. D51 fixed the TRAINED
    side -- five published checkpoints, mean 0.8609 +- 0.0164 -- but the untrained
    arm is still a single torch.manual_seed(0) draw, so the comparison remains
    one-sided.

    This measures rho for additional random inits. Random weights need no download,
    so this is the cheapest run in the project. With four more seeds the untrained
    arm has five weight-sets like the trained arm, and the contrast becomes a
    two-sample comparison across WEIGHTS -- which is what the claim is about.

    Two models per kernel, the capacity that empirically works: a 3.5B float32
    model is ~14 GB against the T4's 14.56 GB and the previous model is not freed
    (see geometry-rho-ckpt's header).

METHOD -- unchanged from geometry-rho-direct, so the numbers are directly comparable
    (A) two-orbit convergence: same prompt, two random initial latents, and
        d_t = ||h_t^(1) - h_t^(2)|| contracts as rho^t. Independent of where the
        fixed point sits.
    (B) step-norm decay along one orbit, as an independent cross-check.
    Points kept only while above 3x the empirical floor (median of the last quarter),
    since floored points drag the slope toward zero and INFLATE rho.
    float32 throughout; bfloat16's 2^-8 roundoff would swallow the decay.
"""

import gc
import json
import random
import subprocess


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


FINAL = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
# THE ONLY LINE THAT DIFFERS BETWEEN SHARDS. Seed 0 is already measured (D44).
SEEDS = (3, 4)

M = 64
MAX_R = 128
N_PER_FAMILY = 3
K_FLOOR = 3.0
TAIL_FRAC = 0.25


def prompts_by_family():
    """Four task families, so rho's constancy across tasks (D45) extends to these
    checkpoints too rather than resting only on the two anchors."""
    out = []
    for i in range(N_PER_FAMILY):
        rng = random.Random(i * 7919)
        bits = [1 if rng.random() < 0.5 else 0 for _ in range(M)]
        out.append(("counting",
                    "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:"))
    for i in range(N_PER_FAMILY):
        rng = random.Random(i * 104729)
        seq, d = [], 0
        for _ in range(M // 2):
            if d == 0 or (rng.random() < 0.5 and d < 8):
                seq.append("(")
                d += 1
            else:
                seq.append(")")
                d -= 1
        seq += [")"] * d
        out.append(("nesting",
                    "String: " + " ".join(seq) + ". What is the maximum nesting depth? A:"))
    for i in range(N_PER_FAMILY):
        rng = random.Random(i * 15485863)
        a, b, c = rng.randint(11, 99), rng.randint(3, 19), rng.randint(2, 9)
        out.append(("arith",
                    f"A shop had {a} boxes. It sold {b} boxes each day for {c} days. "
                    f"How many boxes are left? A:"))
    stems = ["The man picked up the heavy suitcase and walked toward the platform. He",
             "She opened the oven, checked the bread, and decided it needed more time. Then she",
             "The dog heard the doorbell, ran into the hallway, and started barking. Next it"]
    for i in range(N_PER_FAMILY):
        out.append(("commonsense", stems[i % len(stems)]))
    return out


def fit_rho(curve, k=K_FLOOR, tail_frac=TAIL_FRAC):
    """Fit log(curve) linear in t over the pre-floor regime -> (rho, n_used, fit_r2)."""
    import numpy as np
    y = np.asarray(curve, dtype=np.float64)
    if len(y) < 6:
        return float("nan"), 0, float("nan")
    floor = float(np.median(y[-max(3, int(len(y) * tail_frac)):]))
    if not np.isfinite(floor) or floor <= 0:
        return float("nan"), 0, float("nan")
    n = 0
    for v in (y > k * floor):
        if not v:
            break
        n += 1
    if n < 4:
        return float("nan"), int(n), float("nan")
    t = np.arange(n, dtype=np.float64)
    ly = np.log(y[:n])
    slope, icpt = np.polyfit(t, ly, 1)
    pred = slope * t + icpt
    denom = ((ly - ly.mean()) ** 2).sum()
    r2 = float(1 - ((ly - pred) ** 2).sum() / denom) if denom > 0 else float("nan")
    return float(np.exp(slope)), int(n), r2


def orbit(model, ids, seed, max_r):
    import numpy as np
    import torch
    cap = []
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()
    h = mod.register_forward_hook(
        lambda m, i, o, c=cap: c.append(o.detach()[0, -1, :].float().cpu().numpy()))
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            model(input_ids=ids, num_steps=max_r)
    finally:
        h.remove()
    return np.stack(cap)


def measure(model, tok, label, prompts):
    import numpy as np
    import torch
    rows = []
    for i, (family, p) in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        a = orbit(model, ids, 1000 + i, MAX_R)
        b = orbit(model, ids, 2000 + i, MAX_R)
        d = np.linalg.norm(a - b, axis=1)
        s = np.linalg.norm(np.diff(a, axis=0), axis=1)
        rd, nd, fd = fit_rho(d)
        rs, ns, fs = fit_rho(s)
        rows.append({"prompt": i, "family": family, "n_tok": int(ids.shape[1]),
                     "rho_orbit": rd, "n_orbit": nd, "r2_orbit": fd,
                     "rho_step": rs, "n_step": ns, "r2_step": fs,
                     "d0": float(d[0]), "d_end": float(d[-1]),
                     "norm_h": float(np.linalg.norm(a[-1]))})
        print(f"    {label} {family[:5]}{i}: rho_orbit={rd:.4f}(n={nd},fit {fd:.3f})  "
              f"rho_step={rs:.4f}(n={ns},fit {fs:.3f})  d0={d[0]:.2f}  "
              f"||h||={rows[-1]['norm_h']:.2f}", flush=True)
        del ids
        torch.cuda.empty_cache()
    return rows


def summarise(rows):
    import numpy as np

    def agg(rs, key):
        v = np.array([r[key] for r in rs], float)
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()) if len(v) else float("nan"),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                "n": int(len(v))}

    out = {k: agg(rows, k) for k in ("rho_orbit", "rho_step")}
    out["by_family"] = {}
    for fam in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == fam]
        out["by_family"][fam] = {k: agg(rs, k) for k in ("rho_orbit", "rho_step")}
    return out



def main():
    run("pip install -q 'transformers>=4.50,<4.54' scipy")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), "devices:", torch.cuda.device_count(),
          flush=True)

    tok = AutoTokenizer.from_pretrained(FINAL, revision=REVISION)
    prompts = prompts_by_family()
    print(f"{len(prompts)} prompts across {len({f for f, _ in prompts})} families",
          flush=True)

    final_cfg = AutoConfig.from_pretrained(FINAL, revision=REVISION,
                                           trust_remote_code=True)
    results = {}

    for seed in SEEDS:
        label = f"init{seed}"
        print(f"\n=== {label} (random init, torch seed {seed}) ===", flush=True)
        model = None
        try:
            torch.manual_seed(seed)
            model = AutoModelForCausalLM.from_config(final_cfg, trust_remote_code=True)
            model = model.to(torch.float32).to("cuda").eval()
            rows = measure(model, tok, label, prompts)
            results[label] = {"step": 0, "init_seed": seed, "rows": rows,
                              "summary": summarise(rows)}
            sm = results[label]["summary"]
            print(f"  {label}: rho_orbit={sm['rho_orbit']['mean']:.4f}"
                  f"+-{sm['rho_orbit']['sd']:.4f}  "
                  f"rho_step={sm['rho_step']['mean']:.4f}+-{sm['rho_step']['sd']:.4f}",
                  flush=True)
        except Exception as e:                                      # noqa: BLE001
            print(f"  {label} FAILED: {type(e).__name__}: {str(e)[:250]}", flush=True)
            results[label] = {"step": 0, "init_seed": seed,
                              "error": f"{type(e).__name__}: {e}"}
        finally:
            try:
                model = model.to("cpu") if model is not None else None
            except Exception:                                       # noqa: BLE001, S110
                pass
            del model
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            print(f"  after cleanup: {free / 2**30:.2f} GiB free of "
                  f"{total / 2**30:.2f} GiB", flush=True)
            with open("rho_seed.json", "w") as f:
                json.dump(results, f, indent=1)

    print("\n=== SHARD RESULT ===")
    for _k, v in sorted(results.items(), key=lambda x: x[1]["init_seed"]):
        if "summary" in v:
            sm = v["summary"]
            print(f"  init seed {v['init_seed']:>3}  rho_orbit="
                  f"{sm['rho_orbit']['mean']:.4f}+-{sm['rho_orbit']['sd']:.4f}  "
                  f"rho_step={sm['rho_step']['mean']:.4f}")
        else:
            print(f"  init seed {v['init_seed']:>3}  ERROR {str(v.get('error'))[:150]}")
    print("\n  for reference:")
    print("    untrained seed 0 (D44)      rho_orbit=0.7150 +-0.0069")
    print("    5 trained weight-sets (D51) rho_orbit=0.8609 +-0.0164")


main()
