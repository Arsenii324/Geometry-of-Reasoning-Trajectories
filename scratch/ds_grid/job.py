"""The difficulty x depth grid: accuracy as a surface, not a number.

WHY THIS DESIGN, AND WHY NOT A BINARY SEARCH. The first version of this plan
proposed binary-searching the difficulty knob for the level where accuracy crosses
the measurable band. The supervisor pointed out the flaw: accuracy at a given level
is a noisy Bernoulli estimate, so a staircase that commits to a direction at each
step chases its own noise -- and the noise is LARGEST exactly in the 20-80% band we
want to resolve. It also discards every level it steps past. So: sweep the whole
grid, few draws per cell, and report the surface.

THE DEPTH AXIS IS FREE, AND THIS IS THE POINT. The recurrence is iterated, so ONE
forward at num_steps=48 yields the gold rank at ALL 48 depths -- the read hook
applies the D71-validated `coda_head` at each unroll of a single continuous run.
Never re-run a prompt per depth. So a (difficulty x depth) surface costs the same
as a difficulty-only scan, and the h_0 spread within each cell comes along too.

WHAT THREE THINGS FALL OUT OF ONE SWEEP:
  (a) WHERE THE MEASURABLE BAND IS, per family -- the binding constraint on this
      whole project. Two experiments (D47, D95) have now died because too few
      items had outcomes that varied at all; D95's gate wanted 4 of 9 splitting
      prompts and got 2.
  (b) WHETHER HARDER PROBLEMS NEED MORE DEPTH -- directly the object H2 is about
      ("more reasoning steps, more turning"), but measured BEHAVIOURALLY as
      "does the depth at which the answer first becomes available increase with
      n?" rather than geometrically. Every geometric H2 instrument in this
      project (D28, D74(6), D83) has returned nothing; this is the behavioural
      version and it has never been run.
  (c) THE h_0 CONFOUND (D90), visible as within-cell variance rather than
      needing its own experiment.

CRITERION 3 IS THE HARD ONE AND DRIVES THE FAMILY CHOICE. Trajectory geometry in
this model demonstrably tracks PROMPT LENGTH (D26, D84, 6 of 18 cells in D85). So
a difficulty knob that also lengthens the prompt is confounded by construction.
**Families are therefore split into two groups and the kernel CHECKS the token
count itself rather than assuming it** (the gate D87 passed absolutely):

  LENGTH-CONSTANT (the valuable ones -- difficulty rises, token count does not):
    `caesar_k`    -- shift a letter forward by k, k = 1..12. The supervisor's own
                     example. Single-character gold, and "by 7" is as long as
                     "by 2".
    `addk`        -- what is v + k, k = 1..9.
    `nth_item_k`  -- which item is at position k of a FIXED-length list, k = 1..8.
                     Indexing depth varies; the list never changes size.
  LENGTH-GROWING (kept deliberately, as the contrast that shows what the confound
  looks like -- not as the primary evidence):
    `count_n`, `parity_n`, `track_n`.

PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE THE RUN (CLAUDE.md section 1):
  1. **At least one LENGTH-CONSTANT family has a level with accuracy in 20-80%.**
     If none does, the difficulty-knob approach fails on this model and the
     project must fall back to in-distribution data (CLRS-Text), which is the
     subject of a separate research inquiry.
  2. **Accuracy falls monotonically with k in `caesar_k`.** If it does not -- if
     k=1 and k=12 are equally hard -- then the knob is not a difficulty knob and
     the family is measuring something else (e.g. token identity of the target
     letter).
  3. **best_depth does NOT increase with difficulty.** Every geometric H2
     instrument has returned nothing, and D86 found the rank->output link is gone
     by r=32 while D97 found positions converge in a tight 31-36 band regardless
     of task. So the honest prior is that this model does not spend more
     iterations on harder problems. **A positive result here -- harder levels
     peaking at systematically later depth -- would be the first behavioural
     support for H2 in the project and would matter more than anything else in
     this run.**

The gold for every family is single-token where possible, so the rank readout is
not measuring a leading token (the D89 defect); `multi_token_gold` is recorded on
every row regardless so it stays filterable.
"""

import json
import os
import string
import subprocess
import sys


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 48
N_ITEMS = 3          # distinct items per (family, level), so one odd item cannot carry a cell
N_DRAWS = 4          # unseeded h_0 draws per item -> 12 forwards per cell

L = string.ascii_lowercase

# (family, length_constant, levels, item_fn) -- item_fn(level, i) -> (prompt, gold)
FAMILIES = {}


def _caesar(k, i):
    ch = L[(i * 7 + 2) % 26]
    return (f"Shift the letter {ch} forward by {k} in the alphabet. Answer with one letter.",
            L[(L.index(ch) + k) % 26])


def _addk(k, i):
    """v is chosen so v + k NEVER reaches 10.

    Otherwise the gold silently becomes two tokens exactly at the harder levels,
    which would make D89's multi-token defect correlate with the difficulty knob
    -- the readout would degrade with k for a tokenisation reason and look like a
    capability decline. Caught before running.
    """
    v = i % 3
    return (f"What is {v} + {k}? Answer with the number.", str(v + k))


def _nth(k, i):
    xs = [(i * 5 + j * 3) % 10 for j in range(8)]
    return (f"What is item number {k} in this list? Answer with the number.\n"
            f"List: {' '.join(map(str, xs))}", str(xs[k - 1]))


def _bits(n, i, salt):
    """Deterministic but genuinely varied bits.

    The first version used `(i*a + j*b) % 3 % 2`, which is CONSTANT in j whenever
    b is a multiple of 3 -- `track_n` came out with every operation identical, so
    its "difficulty" levels were all the same trivial task. Use an explicit seeded
    RNG instead of hand-rolled arithmetic.
    """
    import random
    rng = random.Random(1000 * salt + i)
    return [rng.randint(0, 1) for _ in range(n)]


def _count(n, i):
    bits = _bits(n, i, 1)
    return ("Count how many ones are in this sequence. Answer with the number.\n"
            f"Sequence: {' '.join(map(str, bits))}", str(sum(bits)))


def _parity(n, i):
    bits = _bits(n, i, 2)
    return ("Is the number of ones in this sequence even or odd? "
            "Answer 0 for even and 1 for odd.\n"
            f"Sequence: {' '.join(map(str, bits))}", str(sum(bits) % 2))


def _track(n, i):
    ops = [1 if b else -1 for b in _bits(n, i, 3)]
    body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1." for o in ops)
    return (body + " What is the final total? Answer with the number.", str(sum(ops)))


FAMILIES["caesar_k"] = (True, list(range(1, 13)), _caesar)
FAMILIES["addk"] = (True, list(range(1, 8)), _addk)   # v+k <= 9, single-token gold
FAMILIES["nth_item_k"] = (True, list(range(1, 9)), _nth)
FAMILIES["count_n"] = (False, [2, 4, 6, 8, 12, 16, 20], _count)
FAMILIES["parity_n"] = (False, [2, 4, 6, 8, 12, 16, 20], _parity)
FAMILIES["track_n"] = (False, [2, 4, 6, 8, 12, 16], _track)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("grid.json")
    print(f"results -> {out_path}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))   # pip and python may differ; see REMOTE_RUNS.md

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

    def coda_head(h_state, freqs_cis):
        """Verbatim D71-validated tail: TWO ln_f calls, coda blocks, lm_head."""
        x = model.transformer.ln_f(h_state)
        block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            block_idx -= 1
            x = block(x, freqs_cis, block_idx, None, None)
        x = model.transformer.ln_f(x)
        return model.lm_head(x)

    def one(prompt, gold):
        """One forward -> the gold rank at EVERY depth. The whole design rests on
        this being a single continuous run rather than one run per depth."""
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        g_ids = tok(gold, add_special_tokens=False).input_ids
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        ranks = []
        mod = model.transformer.core_block[-1]
        mod._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                row = torch.log_softmax(
                    coda_head(o.detach(), freqs).float()[0, n_p - 1], dim=-1)
                ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)

        h = mod.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        return {"rank_curve": ranks, "n_tokens": int(n_p),
                "best_rank": int(min(ranks)), "best_depth": int(np.argmin(ranks)) + 1,
                "correct": bool(min(ranks) == 1),
                "multi_token_gold": bool(len(g_ids) > 1)}

    rows = []
    for fam, (len_const, levels, fn) in FAMILIES.items():
        print(f"\n=== {fam}  (length_constant={len_const}, "
              f"levels {levels[0]}..{levels[-1]}) ===", flush=True)
        for lv in levels:
            for i in range(N_ITEMS):
                prompt, gold = fn(lv, i)
                for _ in range(N_DRAWS):
                    try:
                        r = one(prompt, gold)
                    except Exception as exc:
                        r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
                    r.update({"family": fam, "level": lv, "item": i,
                              "length_constant": len_const, "gold": gold})
                    r.setdefault("ok", True)
                    rows.append(r)
            cell = [r for r in rows
                    if r["family"] == fam and r["level"] == lv and r.get("ok")]
            acc = float(np.mean([r["correct"] for r in cell]))
            bd = float(np.median([r["best_depth"] for r in cell]))
            ntok = sorted({r["n_tokens"] for r in cell})
            print(f"  level {lv:>3}: acc {acc:6.1%}  median best_depth {bd:5.1f}  "
                  f"n_tokens {ntok[0]}-{ntok[-1]}  (n={len(cell)})", flush=True)
        json.dump(rows, open(out_path, "w"))

    # THE TOKEN-COUNT GATE, checked rather than assumed (D87's discipline).
    print("\n=== TOKEN-COUNT GATE: is 'length_constant' actually true? ===", flush=True)
    for fam, (len_const, _lv, _f) in FAMILIES.items():
        ns = sorted({r["n_tokens"] for r in rows if r["family"] == fam and r.get("ok")})
        ok = (len(ns) == 1)
        verdict = "CONSTANT" if ok else f"VARIES {ns[0]}-{ns[-1]}"
        flag = "" if ok == len_const else "   <-- CLAIM DOES NOT MATCH MEASUREMENT"
        print(f"  {fam:>12}: claimed length_constant={len_const}, measured {verdict}{flag}",
              flush=True)

    print("\n=== PRE-REGISTERED CHECKS ===", flush=True)
    band = []
    for fam, (len_const, levels, _f) in FAMILIES.items():
        for lv in levels:
            cell = [r for r in rows
                    if r["family"] == fam and r["level"] == lv and r.get("ok")]
            if cell:
                a = float(np.mean([r["correct"] for r in cell]))
                if 0.2 <= a <= 0.8:
                    band.append((fam, lv, a, len_const))
    lc = [b for b in band if b[3]]
    print(f"  (1) cells in the 20-80% band: {len(band)} total, "
          f"{len(lc)} of them LENGTH-CONSTANT", flush=True)
    for b in band:
        print(f"        {b[0]:>12} level {b[1]:>3}: {b[2]:.1%}"
              f"{'  [length-constant]' if b[3] else ''}", flush=True)
    if not lc:
        print("      NO length-constant cell lands in the band -- the difficulty-knob "
              "route fails on this model and CLRS-Text becomes the fallback.", flush=True)

    from scipy.stats import spearmanr
    for fam, (len_const, levels, _f) in FAMILIES.items():
        pts = [(r["level"], r["best_depth"]) for r in rows
               if r["family"] == fam and r.get("ok")]
        accs = [(lv, float(np.mean([r["correct"] for r in rows
                                    if r["family"] == fam and r["level"] == lv
                                    and r.get("ok")]))) for lv in levels]
        if len(set(p[0] for p in pts)) > 2:
            rho_d, p_d = spearmanr([p[0] for p in pts], [p[1] for p in pts])
            rho_a, p_a = spearmanr([a[0] for a in accs], [a[1] for a in accs])
            print(f"  (2,3) {fam:>12}: acc-vs-level rho={rho_a:+.3f} (p={p_a:.3f}) | "
                  f"best_depth-vs-level rho={rho_d:+.3f} (p={p_d:.4f})", flush=True)

    json.dump({"rows": rows,
               "config": {"num_steps": NUM_STEPS, "n_items": N_ITEMS,
                          "n_draws": N_DRAWS, "model": MODEL_ID,
                          "revision": REVISION,
                          "families": {k: [v[0], v[1]] for k, v in FAMILIES.items()}}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
