"""Contraction rate rho ACROSS TRAINING: random init -> 8 checkpoints -> final model.

WHAT THIS DECIDES
    D42 claims training SLOWS the contraction (rho ~0.66 untrained -> ~0.91
    trained) and that this one parameter explains both effects in D41. Two
    weaknesses: (a) the untrained 0.662 is INFERRED from a readout curve via
    observable_convergence.py eq.(3), never measured on the operator; (b) two
    endpoints cannot distinguish "training raises rho" from "these two particular
    weight-sets happen to differ".

    tomg-group-umd published eight intermediate checkpoints of the SAME run --
    step-00006144 ... step-00041728, all with configs byte-identical to
    huginn-0125 (verified: n_embd=5280, mean_recurrence=32, same architecture).
    So rho can be measured as a FUNCTION OF TRAINING STEP. A monotone rise turns
    D42 from a two-point contrast into a trend; a flat or non-monotone curve
    falsifies it.

    All models measured by the SAME code, on the SAME prompts, in the SAME
    process, so differences are attributable to the weights alone.

TWO INDEPENDENT METHODS, because either alone has a known failure mode
    (A) TWO-ORBIT CONVERGENCE.  Same prompt, two different random initial
        latents; d_t = ||h_t^(1) - h_t^(2)|| contracts as rho^t. Measures the
        contraction of the MAP, independent of where the fixed point sits --
        the D24(7) method.
    (B) STEP-NORM DECAY.  s_t = ||h_{t+1} - h_t|| along one orbit, also ~ rho^t.
        Cheaper, but contaminated if the orbit has not reached the linear regime.
    Agreement is the internal check; disagreement is reported, not hidden.

THE FLOOR, WHICH BIASES ANY SUCH FIT IF IGNORED
    Finite arithmetic floors both curves; once there, decay stops, and including
    those points drags the slope toward zero and INFLATES rho. Points are kept
    only while above k * (empirical floor), floor = median of the last quarter --
    the rule in src/traj_geom/metrics/regime.py. n_used is reported, because a
    fit on three points is not a measurement. Validated locally against known
    rates 0.60-0.95: recovery bias +0.002..+0.015, always upward, so an
    UNDERstated difference, never an overstated one.

float32 throughout -- bfloat16 (unit roundoff 2^-8) would raise the floor far
enough to swallow the decay. Results are written after EVERY checkpoint, so a
timeout still yields a usable partial curve; the checkpoint order is chosen so
the most informative points land first.
"""

import gc
import json
import os
import random
import shutil
import subprocess


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


FINAL = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"

# (label, repo_id, training_step, revision). Revisions pinned 2026-08-04, same
# provenance rule the rest of the project uses. Ordered so a timeout still leaves a usable
# curve: the two anchors and the extreme intermediates go first.
SWEEP = [
    ("untrained",  None,                                              0, None),
    ("final",      FINAL,                                        100000, REVISION),
    ("s06144",     "tomg-group-umd/step-00006144-recurrence_full_512_0",  6144, "b5b1f9d44fdf3f1b91cb301dfaded77196e7bf3d"),
    ("s41728",     "tomg-group-umd/step-00041728-recurrence_full_512_0", 41728, "0fb03f39328917a92ca3059cc77b3e2115a35c94"),
    ("s17920",     "tomg-group-umd/step-00017920-recurrence_full_512_0", 17920, "9c8576dc13cffdf93ec7da08eecf5b200bc8b47d"),
    ("s29824",     "tomg-group-umd/step-00029824-recurrence_full_512_0", 29824, "252da2591e39f1ca27c063e018e254e470833c39"),
    ("s10752",     "tomg-group-umd/step-00010752-recurrence_full_512_0", 10752, "5e35596c8e8c79ecae57a8fab34625fe8ac640a8"),
    ("s23808",     "tomg-group-umd/step-00023808-recurrence_full_512_0", 23808, "b531a3366a7d2a4171403038b785639606cb1297"),
    ("s35840",     "tomg-group-umd/step-00035840-recurrence_full_512_0", 35840, "41580a97ec90f282e6e9b72f83073779808a53af"),
    ("s11904",     "tomg-group-umd/step-00011904-recurrence_full_512_0", 11904, "aadca23e6829ea0bcf96b7965ebc61ad9fada17c"),
]

M = 64
MAX_R = 128
N_PROMPTS = 8
K_FLOOR = 3.0
TAIL_FRAC = 0.25


def task(seed):
    rng = random.Random(seed * 7919)
    bits = [1 if rng.random() < 0.5 else 0 for _ in range(M)]
    return "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:"


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
    for v in (y > k * floor):        # leading contiguous run; once floored it stays
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
    """h_1..h_max_r at the final position, from a controlled random initial latent."""
    import numpy as np
    import torch
    cap = []
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()
    h = mod.register_forward_hook(
        lambda m, i, o: cap.append(o.detach()[0, -1, :].float().cpu().numpy()))
    try:
        torch.manual_seed(seed)      # controls h_0
        with torch.no_grad():
            model(input_ids=ids, num_steps=max_r)
    finally:
        h.remove()
    return np.stack(cap)


def measure(model, tok, label, prompts):
    import numpy as np
    import torch
    rows = []
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        a = orbit(model, ids, 1000 + i, MAX_R)
        b = orbit(model, ids, 2000 + i, MAX_R)
        d = np.linalg.norm(a - b, axis=1)
        s = np.linalg.norm(np.diff(a, axis=0), axis=1)
        rd, nd, fd = fit_rho(d)
        rs, ns, fs = fit_rho(s)
        rows.append({"prompt": i, "rho_orbit": rd, "n_orbit": nd, "r2_orbit": fd,
                     "rho_step": rs, "n_step": ns, "r2_step": fs,
                     "d0": float(d[0]), "d_end": float(d[-1]),
                     "norm_h": float(np.linalg.norm(a[-1]))})
        print(f"    {label} p{i}: rho_orbit={rd:.4f}(n={nd},fit {fd:.3f})  "
              f"rho_step={rs:.4f}(n={ns},fit {fs:.3f})  ||h||={rows[-1]['norm_h']:.2f}",
              flush=True)
        del ids
        torch.cuda.empty_cache()
    return rows


def summarise(rows):
    import numpy as np
    out = {}
    for key in ("rho_orbit", "rho_step"):
        v = np.array([r[key] for r in rows], float)
        v = v[np.isfinite(v)]
        out[key] = {"mean": float(v.mean()) if len(v) else float("nan"),
                    "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                    "n": int(len(v))}
    return out


def purge(repo_id):
    """Drop the HF cache for one repo -- 10 x ~7GB would otherwise exhaust the disk."""
    if not repo_id:
        return
    root = os.path.expanduser("~/.cache/huggingface/hub")
    d = os.path.join(root, "models--" + repo_id.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print(f"    purged cache {d}", flush=True)


def main():
    run("pip install -q 'transformers>=4.50,<4.54' scipy")
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(FINAL, revision=REVISION)
    prompts = [task(s) for s in range(N_PROMPTS)]
    print(f"{len(prompts)} prompts, token lengths "
          f"{sorted({len(tok(p).input_ids) for p in prompts})}", flush=True)

    cfg = AutoConfig.from_pretrained(FINAL, revision=REVISION, trust_remote_code=True)
    results = {}

    for label, repo, step, rev in SWEEP:
        print(f"\n=== {label} (step {step}) ===", flush=True)
        model = None
        try:
            if repo is None:
                torch.manual_seed(0)
                model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    repo, revision=rev, trust_remote_code=True)
            model = model.to(torch.float32).to("cuda").eval()
            rows = measure(model, tok, label, prompts)
            results[label] = {"step": step, "repo": repo, "revision": rev, "rows": rows,
                              "summary": summarise(rows)}
            s = results[label]["summary"]
            print(f"  {label}: rho_orbit={s['rho_orbit']['mean']:.4f}"
                  f"+-{s['rho_orbit']['sd']:.4f}  "
                  f"rho_step={s['rho_step']['mean']:.4f}+-{s['rho_step']['sd']:.4f}",
                  flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {label} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            results[label] = {"step": step, "repo": repo, "revision": rev, "error": f"{type(e).__name__}: {e}"}
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            purge(repo)
            with open("rho_vs_training.json", "w") as f:      # partial-safe
                json.dump(results, f, indent=1)

    print("\n=== rho VERSUS TRAINING STEP ===")
    ok = [(v["step"], k, v["summary"]) for k, v in results.items() if "summary" in v]
    ok.sort()
    print(f"  {'step':>8} {'label':>10} {'rho_orbit':>18} {'rho_step':>18}")
    for st, lab, s in ok:
        print(f"  {st:>8} {lab:>10} {s['rho_orbit']['mean']:>10.4f}"
              f" +-{s['rho_orbit']['sd']:<6.4f} {s['rho_step']['mean']:>10.4f}"
              f" +-{s['rho_step']['sd']:<6.4f}")
    if len(ok) >= 4:
        from scipy.stats import spearmanr
        steps = np.array([o[0] for o in ok], float)
        for key in ("rho_orbit", "rho_step"):
            vals = np.array([o[2][key]["mean"] for o in ok], float)
            m = np.isfinite(vals)
            if m.sum() >= 4:
                rho_s, p = spearmanr(steps[m], vals[m])
                print(f"\n  spearman(training step, {key}) = {rho_s:+.4f}, p = {p:.4g}"
                      f"   [n={int(m.sum())} checkpoints]")
        print("\n  D42 SURVIVES if rho rises with training step. A flat or")
        print("  non-monotone curve falsifies 'training slows the contraction'.")


main()
