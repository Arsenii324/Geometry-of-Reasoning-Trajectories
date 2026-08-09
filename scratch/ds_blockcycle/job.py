"""Is the "loop" an inter-BLOCK cycle that our single-block read cannot see?

THE THREAT THIS RUN EXISTS TO TEST, STATED BLUNTLY. D94 (contraction rho = 0.855,
0 of ~960 orbit pairs at rho >= 1) and D97 (179 of 179 token positions classify as
`settle`, zero loop, zero drift) are the project's strongest H1 evidence. **Both
read the state at `core_block[-1]` and nowhere else.** A published mechanistic
analysis of looped reasoning LMs reports that *each layer in the cycle converges to
a distinct fixed point; consequently, the recurrent block follows a consistent
cyclic trajectory in the latent space.* If that is what "loop" means in this
literature, then:

  * every block converging is CONSISTENT with a cycle, not evidence against it;
  * reading ONE block per iteration samples one point of that cycle, and would
    show clean convergence no matter how large the cycle is;
  * **D94 and D97 would be answering a question nobody asked, and H1 would be
    open rather than settled.**

This is the single largest interpretive risk to the current headline. It is also
cheap to resolve, because the distinction is geometric and fully determined by
data we have never recorded: the state after EACH of the 4 core blocks.

THE ARCHITECTURE FACT THAT MAKES THE TEST WELL-POSED. One iteration applies
`core_block[0..3]` in sequence, and the output of block 3 becomes the input of
block 0 at the next iteration. So in the "unrolled by block" view the trajectory
is *inherently* cyclic in ORDER -- it visits four slots forever. The real question
is whether that cycle is **non-degenerate**: are the four per-block fixed points
far apart relative to how tightly each one converges?

  * **Non-degenerate** (pairwise separation >> per-block convergence residual):
    there IS a genuine limit cycle of period 4 in block-space, each block
    converges, and our single-block read is one sample of it. **H1's "loop"
    would then exist at a granularity this project has never measured, and the
    D94/D97 nulls would need restating rather than retracting.**
  * **Degenerate** (the four fixed points nearly coincide): the four blocks
    converge to essentially the same point, there is no meaningful cycle, and
    D94/D97 measured the right object after all.

WHAT IS MEASURED, at the answer position, for each prompt:
  * `x[r, j]` -- the state after core block j at iteration r, for j = 0..3.
  * per-block convergence residual: ||x[r,j] - x[r-1,j]|| as r grows -- does each
    block settle to its own fixed point, as the cited paper claims?
  * pairwise distances between the four converged points, ||x*_j - x*_k||.
  * **the ratio that decides it**: (median pairwise separation) / (final
    per-block residual). Large ratio -> real cycle. Order 1 -> degenerate.
  * the cycle's own geometry: its perimeter, and its diameter relative to the
    distance the state travels from iteration 0 to convergence.

PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE THE RUN (CLAUDE.md section 1):
  1. **Each block converges** -- per-block residuals decay to the noise floor.
     This is the cited paper's claim and I expect it to reproduce.
  2. **The four fixed points are DISTINCT and well separated**, giving a
     separation/residual ratio >> 1. The four blocks have different weights and
     no reason to agree, so a non-degenerate cycle is the likely outcome.
  3. **Therefore the honest reading is that D94/D97 measured convergence WITHIN a
     block-slot, which is a real result but is NOT the same as "no loop exists".**
     If prediction 2 holds, the ledger must say so explicitly, and the H1 claim
     must be restated as "no loop/drift in the iteration-to-iteration map at a
     fixed block" rather than "no loop anywhere".

**This run is designed to be able to embarrass the project's own headline, which
is why it is worth its GPU minutes.**
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
        os.path.abspath("blockcycle.json")
    print(f"results -> {out_path}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    n_blocks = len(model.transformer.core_block)
    print(f"core_block has {n_blocks} blocks -- hooking ALL of them, not just [-1]",
          flush=True)

    def all_block_states(prompt):
        """[n_iter, n_blocks, d] at the answer position.

        A hook on EVERY core block, tagged by block index. The blocks fire in
        order within each iteration, so appending per-block gives the full
        block-resolved trajectory that a `core_block[-1]`-only read collapses.
        """
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        seen = {j: [] for j in range(n_blocks)}
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
        return np.stack([np.stack(seen[j][:n]) for j in range(n_blocks)], axis=1)

    records = []
    for name, prompt in PROMPTS:
        try:
            x = all_block_states(prompt).astype(np.float64)   # [n_iter, n_blocks, d]
            n_iter = x.shape[0]

            # (1) does each block converge to its OWN fixed point?
            resid = {}
            for j in range(n_blocks):
                d = np.linalg.norm(np.diff(x[:, j, :], axis=0), axis=1)
                resid[j] = d
            final_resid = float(np.median([resid[j][-5:].mean() for j in range(n_blocks)]))

            # (2) how far apart are the converged per-block points?
            star = x[-1]                                   # [n_blocks, d]
            pair = [float(np.linalg.norm(star[a] - star[b]))
                    for a in range(n_blocks) for b in range(a + 1, n_blocks)]
            med_pair = float(np.median(pair))

            # (3) THE DECIDING RATIO
            ratio = med_pair / final_resid if final_resid > 0 else float("inf")

            # (4) the cycle's own scale, against the distance travelled overall
            perim = float(sum(np.linalg.norm(star[(j + 1) % n_blocks] - star[j])
                              for j in range(n_blocks)))
            travelled = float(np.linalg.norm(x[-1, -1] - x[0, -1]))
            norms = [float(np.linalg.norm(star[j])) for j in range(n_blocks)]

            rec = {"name": name, "ok": True, "n_iter": int(n_iter),
                   "n_blocks": int(n_blocks),
                   "final_block_residual": final_resid,
                   "pairwise_fixedpoint_dists": pair,
                   "median_pairwise": med_pair,
                   "separation_over_residual": ratio,
                   "cycle_perimeter": perim,
                   "distance_travelled_block3": travelled,
                   "perimeter_over_travelled": perim / travelled if travelled else float("nan"),
                   "state_norms": norms,
                   "resid_first": {str(j): float(resid[j][0]) for j in range(n_blocks)},
                   "resid_last": {str(j): float(resid[j][-1]) for j in range(n_blocks)}}
            records.append(rec)
            print(f"  {name:>10}: per-block final residual {final_resid:9.4f} | "
                  f"median pairwise fixed-point distance {med_pair:9.4f} | "
                  f"RATIO {ratio:8.1f} | perimeter/travelled "
                  f"{rec['perimeter_over_travelled']:.3f}", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            records.append({"name": name, "ok": False,
                            "why": f"{type(exc).__name__}: {exc}"})
        json.dump({"records": records}, open(out_path, "w"))

    ok = [r for r in records if r.get("ok")]
    print(f"\n=== banked {len(ok)}/{len(records)} ===", flush=True)
    if ok:
        import numpy as np
        ratios = [r["separation_over_residual"] for r in ok]
        print("\n=== PRE-REGISTERED VERDICT ===", flush=True)
        print(f"  separation/residual ratio: min {min(ratios):.1f}  "
              f"median {float(np.median(ratios)):.1f}  max {max(ratios):.1f}", flush=True)
        if float(np.median(ratios)) > 10:
            print("  **NON-DEGENERATE CYCLE.** The four per-block fixed points are far "
                  "apart relative to how tightly each converges.", flush=True)
            print("  So a period-4 cycle in BLOCK-space is real, every block still "
                  "converges, and a core_block[-1]-only read samples ONE point of it.",
                  flush=True)
            print("  => D94/D97 must be restated as 'no loop/drift in the "
                  "iteration-to-iteration map AT A FIXED BLOCK', not 'no loop anywhere'.",
                  flush=True)
        else:
            print("  **DEGENERATE.** The four fixed points nearly coincide, so there is "
                  "no meaningful inter-block cycle", flush=True)
            print("  and D94/D97 measured the right object. H1's null stands as stated.",
                  flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
