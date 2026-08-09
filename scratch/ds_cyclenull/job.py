"""The control D98 needs: is the block-cycle LEARNED, or just four different matrices?

WHY THIS RUN MUST HAPPEN BEFORE ANYTHING ELSE IS BUILT ON D98. D98 found that
Huginn's four core blocks each converge to their own fixed point, with the four
separated by ~52% of the state norm (separation/residual ratio 95-730) -- a large,
stable, period-4 cycle in block-space that every prior instrument in this project
was blind to. It corrected the scope of D94 and D97, and a follow-up experiment
(`scratch/ds_cyclegeom/`) is already testing H2 against that cycle.

**But four blocks with four DIFFERENT weight matrices will have four different
fixed points for a completely trivial reason.** If a randomly initialised model
shows the same thing, the cycle is a fact about the architecture, not about what
training built, and D98 is a much weaker claim than it currently reads as. This
project has made the untrained-control move before and it paid: D76/D80(7) found
the trained orbit sheds directions 8x faster than untrained and the step cosine
FLIPS SIGN (-0.379 -> +0.541) with completely disjoint distributions -- so
"trained and untrained differ" is not a foregone conclusion in either direction,
which is exactly why it has to be measured.

CLAUDE.md section 1: the cheapest check that could refute a premise comes before
the work that depends on it. This is that check, and it is one job.

DESIGN. The SAME measurement as D98, on the SAME prompts, run twice:
  * TRAINED arm -- the pinned public checkpoint (reproduces D98, so this run also
    serves as an independent replication of it at a different h_0).
  * UNTRAINED arm -- `AutoModelForCausalLM.from_config`, three independent seeds,
    identical architecture and identical prompts. Three seeds because one random
    draw cannot distinguish "random weights do this" from "this random draw did".
The arms are loaded and freed one at a time; two 3.5B models do not fit together.

PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE THE RUN (CLAUDE.md section 1):
  1. **The untrained arm ALSO shows four distinct fixed points**, because four
     different random matrices have no reason to agree. So the mere EXISTENCE of
     a cycle is expected to be architectural and NOT a finding about training.
  2. **The trained cycle differs QUANTITATIVELY -- and this is the real test.**
     The candidate discriminators, all reported: the separation/residual ratio
     (D98: 95-730 trained), planarity, the regularity of the four pairwise
     distances (trained 22-58 is irregular), and the cycle's size relative to the
     distance travelled from h_0 (D98: perimeter 1.6x). If NONE of these
     separates the arms, **D98 must be restated as an architectural observation
     rather than a property of the trained model, and the H2-on-the-cycle
     follow-up loses most of its motivation.**
  3. **The untrained arm may not converge at all.** D80(7) found untrained orbits
     diffuse at roughly constant effective dimension rather than collapsing. If
     the per-block residual does not fall, then "four fixed points" is not even
     well defined there, and that is itself the cleanest possible answer: the
     CONVERGENCE is learned even if the separation is not.

Whatever comes back, D98's ledger row gets amended with it, including if the
answer is that D98 was overstated.
"""

import json
import os
import subprocess
import sys


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 48
UNTRAINED_SEEDS = (0, 1, 2)

PROMPTS = [
    ("echo_digit", "Repeat this number exactly.\nNumber: 7"),
    ("add1", "What is 5 + 1? Answer with the number."),
    ("count8", "Count how many ones are in this sequence. Answer with the number.\n"
               "Sequence: 1 0 1 1 0 0 1 0"),
    ("parity8", "Is the number of ones in this sequence even or odd? "
                "Answer 0 for even and 1 for odd.\nSequence: 1 0 1 1 0 1 0 1"),
    ("track8", "Start at 0. Add 1. Add 1. Subtract 1. Add 1. Subtract 1. Add 1. "
               "Add 1. Subtract 1. What is the final total? Answer with the number."),
    ("caesar1", "Shift the letter c forward by 1 in the alphabet. "
                "Answer with one letter."),
]


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("cyclenull.json")
    print(f"results -> {out_path}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))

    import gc

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)

    def measure(model, prompt):
        """D98's measurement verbatim, so the arms are directly comparable."""
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        nb = len(model.transformer.core_block)
        seen = {j: [] for j in range(nb)}
        handles = []

        def mk(j):
            def hook(_m, _i, o):
                seen[j].append(o.detach()[0, n_p - 1, :].float().cpu().numpy())
            return hook

        for j, blk in enumerate(model.transformer.core_block):
            blk._forward_hooks.clear()
            handles.append(blk.register_forward_hook(mk(j)))
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            for h in handles:
                h.remove()

        n = min(len(v) for v in seen.values())
        x = np.stack([np.stack(seen[j][:n]) for j in range(nb)], axis=1).astype(np.float64)
        star = x[-1]
        resid = [float(np.linalg.norm(np.diff(x[:, j, :], axis=0), axis=1)[-5:].mean())
                 for j in range(nb)]
        resid0 = [float(np.linalg.norm(np.diff(x[:, j, :], axis=0), axis=1)[0])
                  for j in range(nb)]
        pair = [float(np.linalg.norm(star[a] - star[b]))
                for a in range(nb) for b in range(a + 1, nb)]
        perim = float(sum(np.linalg.norm(star[(j + 1) % nb] - star[j]) for j in range(nb)))
        cen = star - star.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(cen, compute_uv=False)
        travelled = float(np.linalg.norm(x[-1, -1] - x[0, -1]))
        med_res = float(np.median(resid))
        return {"median_pairwise": float(np.median(pair)),
                "pairwise_min": float(min(pair)), "pairwise_max": float(max(pair)),
                "pairwise_irregularity": float(max(pair) / min(pair)),
                "final_block_residual": med_res,
                "first_block_residual": float(np.median(resid0)),
                "residual_drop": float(np.median(resid0) / med_res) if med_res else float("inf"),
                "sep_over_resid": float(np.median(pair) / med_res) if med_res else float("inf"),
                "perimeter": perim,
                "perimeter_over_travelled": perim / travelled if travelled else float("nan"),
                "planarity": float((sv[:2] ** 2).sum() / (sv ** 2).sum()),
                "state_norm": float(np.linalg.norm(star, axis=1).mean()),
                "pairwise_over_norm": float(np.median(pair) /
                                            np.linalg.norm(star, axis=1).mean()),
                "ok": True}

    rows = []

    def sweep(model, arm, seed=None):
        for name, prompt in PROMPTS:
            try:
                r = measure(model, prompt)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
            r.update({"arm": arm, "seed": seed, "prompt": name})
            rows.append(r)
        cell = [r for r in rows if r["arm"] == arm and r.get("seed") == seed and r.get("ok")]
        if cell:
            print(f"  [{arm} seed={seed}] sep/resid median "
                  f"{float(np.median([c['sep_over_resid'] for c in cell])):9.1f} | "
                  f"resid drop {float(np.median([c['residual_drop'] for c in cell])):8.1f}x | "
                  f"pair/norm {float(np.median([c['pairwise_over_norm'] for c in cell])):.3f} | "
                  f"planarity {float(np.median([c['planarity'] for c in cell])):.4f}",
                  flush=True)
        json.dump(rows, open(out_path, "w"))

    print("\n=== TRAINED ARM (also an independent replication of D98) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()
    sweep(model, "trained")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== UNTRAINED ARM, three independent weight draws ===", flush=True)
    for s in UNTRAINED_SEEDS:
        torch.manual_seed(s)
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        m = m.to(torch.float32).to("cuda").eval()
        sweep(m, "untrained", seed=s)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== PRE-REGISTERED COMPARISON ===", flush=True)
    tr = [r for r in rows if r["arm"] == "trained" and r.get("ok")]
    un = [r for r in rows if r["arm"] == "untrained" and r.get("ok")]
    from scipy.stats import mannwhitneyu
    verdicts = []
    for stat in ("sep_over_resid", "residual_drop", "pairwise_over_norm",
                 "planarity", "perimeter_over_travelled", "pairwise_irregularity"):
        a = [r[stat] for r in tr if np.isfinite(r[stat])]
        b = [r[stat] for r in un if np.isfinite(r[stat])]
        if len(a) < 3 or len(b) < 3:
            continue
        u, p = mannwhitneyu(a, b)
        disj = (max(a) < min(b)) or (min(a) > max(b))
        verdicts.append((stat, float(np.median(a)), float(np.median(b)), p, disj))
        print(f"  {stat:>26}: trained {np.median(a):10.3f}  untrained {np.median(b):10.3f}"
              f"  p={p:.5f}{'  DISJOINT' if disj else ''}", flush=True)
    sig = [v for v in verdicts if v[3] < 0.05 / max(1, len(verdicts))]
    print(f"\n  statistics separating the arms after Bonferroni: {len(sig)}/{len(verdicts)}",
          flush=True)
    if not sig:
        print("  **NO statistic separates trained from untrained.**", flush=True)
        print("  => D98's cycle is ARCHITECTURAL, not learned. Its ledger row must be "
              "restated and the H2-on-the-cycle follow-up loses its motivation.",
              flush=True)
    else:
        print("  The block cycle is quantitatively DIFFERENT in the trained model on "
              f"{len(sig)} statistic(s):", flush=True)
        for s in sig:
            print(f"      {s[0]}: {s[1]:.3f} vs {s[2]:.3f} (p={s[3]:.5f})", flush=True)
        print("  => D98 survives as a statement about the TRAINED model, with the "
              "existence of a cycle being architectural and its GEOMETRY learned.",
              flush=True)

    json.dump({"rows": rows, "config": {"num_steps": NUM_STEPS,
                                        "untrained_seeds": list(UNTRAINED_SEEDS),
                                        "model": MODEL_ID, "revision": REVISION}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
