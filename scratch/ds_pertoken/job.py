"""G1: record EVERY token's path at every depth, and chart which shapes appear.

WHY THIS IS THE PROJECT'S LARGEST BLIND SPOT. `project_plan.md` G1 asks to
"instrument Huginn to record **each token's** path at **every depth t<=r**; chart
which shapes appear and when (tests H1)". Every instrument this project has built --
winding (D28), effective dimensionality (D74), the Jacobian argument (D83), the
shape code (D84/D87/D88), the correctness decode (D92/D93), the contraction rate
(D94) -- reads exactly ONE position: the answer token, at `core_block[-1]`.
**G1 is 0% done, and its object is the variation this project has never looked at.**

WHY IT IS NOT MERELY COMPLETENESS. Two independent published results say the thing
we are blind to is real and large: *Per-Token Fixed-Point Convergence in
Depth-Recurrent Transformers* (arXiv:2607.14427) reports the median token converged
by loop 6 while **~10% of tokens are still updating at depth 8**, and
arXiv:2607.20594 argues the algorithm lives in the trajectory's HEAD, before the
tail instruments saturate. D80 established the same thing at our one position from
the other side: shape statistics are dominated by *where in the convergence* they
are measured. **If tokens sit at very different phases of that clock, then a
single-position read averages over a mixture, and H1's regimes could be present per
token while invisible in every measurement made so far.** That is a live
alternative explanation for this project's H1/H2 nulls, and it is cheap to test.

WHAT IS MEASURED. For each prompt, at every token position and every unroll, the
hidden state at `core_block[-1]`. Then per POSITION, using the project's own
validated instruments, unmodified:
  * `classify_shape_regime` -> settle / loop / drift  (the H1 object itself)
  * `converging_regime` -> when that position stops moving (the per-token clock)
  * participation ratio and step cosine over a fixed window (D80's two movers)

PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE THE RUN (CLAUDE.md section 1):
  1. **Convergence depth varies substantially across positions within one prompt.**
     If arXiv:2607.14427 transfers, the spread should be tens of unrolls, not a
     few. A flat profile would contradict that paper on this model and is the
     more interesting outcome if it happens.
  2. **The answer position is NOT typical.** It is the last token and the one every
     other result reads; if its convergence depth sits at an extreme of the
     distribution, then every prior single-position claim is a claim about an
     atypical position, which must be stated in the ledger.
  3. **`classify_shape_regime` returns `settle` for the large majority of
     positions.** D94 measured rho = 0.855 with 0.0% of ~960 orbit pairs at
     rho >= 1, so a non-settling position would have to be a local exception to a
     globally contracting map. **Any position returning `loop` or `drift` is the
     first direct evidence for H1's other regimes on this model and would be the
     single most consequential observation this project has made** -- so the
     count of such positions is the headline number of this run, whichever way it
     falls.

SCOPE. 12 prompts spanning the difficulty range, NUM_STEPS=48, every position.
Per-position statistics are computed on the GPU and returned as JSON; raw states
are returned for 2 prompts only, since all-position states are ~50 MB/prompt.
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
RAW_PROMPTS = 2          # how many prompts also return their full state tensor

PROMPTS = [
    ("echo_digit", "Repeat the digit. Digit: 7. Answer:", "7"),
    ("last_item", "List: 4 9 2 6. What is the last item? Answer:", "6"),
    ("compare", "Which is larger, 3 or 8? Answer:", "8"),
    ("add1", "Compute 5 + 1. Answer:", "6"),
    ("add_2d", "Compute 23 + 41. Answer:", "64"),
    ("count4", "Count the 1s: 1 0 1 1. Answer:", "3"),
    ("count8", "Count the 1s: 1 0 1 1 0 0 1 0. Answer:", "4"),
    ("count_mod3", "Count the 1s in 1 1 0 1 1 and give the remainder mod 3. Answer:", "1"),
    ("caesar1", "Shift the letter c forward by 1. Answer:", "d"),
    ("parity8", "Is the number of 1s in 1 0 1 1 0 1 0 1 even or odd? Answer:", "odd"),
    ("track8", "Start at 0. Add 1. Add 1. Subtract 1. Add 1. Final total? A:", "2"),
    ("nth_item", "List: 5 8 3 9 1. What is the 3rd item? Answer:", "3"),
]


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("pertoken.json")
    print(f"results -> {out_path}", flush=True)

    run(
        "git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo"
    )
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")

    # `pip install -e .` reports success but the package can still be missing from
    # THIS interpreter's sys.path -- pip and `python` are not guaranteed to be the
    # same environment under `env.python.type: manual`, and the first submission of
    # this job died on `ModuleNotFoundError: No module named 'traj_geom'` several
    # lines after pip printed "Successfully installed ... traj-geom-0.1.0".
    # Putting the source root on the path directly makes the import independent of
    # which interpreter pip chose. (The QK probe never hit this because it imports
    # nothing from the project.)
    sys.path.insert(0, os.path.abspath("src"))

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # The project's OWN validated instruments, imported not reimplemented.
    from traj_geom.metrics.convergence import consecutive_step_cosine
    from traj_geom.metrics.dimension import participation_ratio
    from traj_geom.metrics.regime import classify_shape_regime, converging_regime

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def all_position_states(prompt):
        """[num_steps, n_tokens, d] -- the whole grid G1 asks for, not one column."""
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        states = []
        mod = model.transformer.core_block[-1]
        mod._forward_hooks.clear()

        def hook(_m, _i, o):
            states.append(o.detach()[0].float().cpu().numpy())

        h = mod.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        return np.stack(states), ids.shape[1]

    records, raw_bank = [], {}
    for pi, (name, prompt, gold) in enumerate(PROMPTS):
        try:
            grid, n_tok = all_position_states(prompt)     # [T, S, d]
            toks = tok.convert_ids_to_tokens(
                tok(prompt, return_tensors="pt").input_ids[0].tolist())
            per_pos = []
            for p in range(n_tok):
                traj = grid[:, p, :].astype(np.float64)   # this token's own path
                try:
                    regime = classify_shape_regime(traj)
                except Exception as exc:
                    regime = f"ERROR:{type(exc).__name__}"
                try:
                    lo, hi = converging_regime(traj)
                except Exception:
                    lo, hi = -1, -1
                per_pos.append({
                    "pos": p,
                    "token": toks[p] if p < len(toks) else "?",
                    "is_answer_pos": bool(p == n_tok - 1),
                    "regime": str(regime),
                    "converge_lo": int(lo), "converge_hi": int(hi),
                    "pr_early": float(participation_ratio(traj[0:12])),
                    "pr_late": float(participation_ratio(traj[24:36])),
                    "cos_early": float(consecutive_step_cosine(traj[0:12])),
                    "cos_late": float(consecutive_step_cosine(traj[24:36])),
                })
            regimes = [r["regime"] for r in per_pos]
            n_settle = sum(1 for r in regimes if r == "settle")
            his = [r["converge_hi"] for r in per_pos if r["converge_hi"] >= 0]
            ans = per_pos[-1]
            print(f"  {name:>11}: {n_tok:>3} tokens | settle {n_settle}/{len(regimes)} "
                  f"| converge_hi min/med/max = "
                  f"{min(his) if his else -1}/{int(np.median(his)) if his else -1}/"
                  f"{max(his) if his else -1} | answer-pos regime={ans['regime']} "
                  f"hi={ans['converge_hi']}", flush=True)
            records.append({"name": name, "prompt": prompt, "gold": gold,
                            "n_tokens": int(n_tok), "ok": True, "per_pos": per_pos})
            if pi < RAW_PROMPTS:
                raw_bank[name] = grid.astype(np.float16).tolist()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            records.append({"name": name, "ok": False,
                            "why": f"{type(exc).__name__}: {exc}"})

    ok = sum(1 for r in records if r.get("ok"))
    print(f"\n=== banked {ok}/{len(records)} prompts ===", flush=True)

    # Headline counts, printed so a reader sees them without loading the JSON.
    all_reg = [p["regime"] for r in records if r.get("ok") for p in r["per_pos"]]
    from collections import Counter
    print("  regime counts over ALL positions:", dict(Counter(all_reg)), flush=True)

    with open(out_path, "w") as fh:
        json.dump({"records": records, "raw": raw_bank,
                   "config": {"num_steps": NUM_STEPS, "model": MODEL_ID,
                              "revision": REVISION}}, fh)
    print(f"wrote {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
