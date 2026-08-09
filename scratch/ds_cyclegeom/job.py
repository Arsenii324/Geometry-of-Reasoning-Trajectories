"""H2, asked about the object D98 actually found: does the CYCLE grow with difficulty?

WHY THIS IS THE RIGHT NEXT EXPERIMENT. Every H2 instrument this project has built
-- winding (D28), effective dimensionality (D74(6)), the Jacobian rotation argument
(D83) -- returned nothing, and D80 explained why: they were all reading a clock,
because the object they measured was a CONVERGING PATH whose statistics are
dominated by how far the contraction has run.

D98 changed the object. Hooking all four core blocks instead of only
`core_block[-1]` showed each block converges to its OWN fixed point, and the four
fixed points sit ~52% of a state norm apart (separation/residual ratio 95-730,
cycle perimeter 1.6x the whole distance travelled from h_0 to convergence). **There
is a large, stable, period-4 cycle in block-space, and every prior instrument was
blind to it by construction.** H2 -- "more reasoning steps, more turning" -- is a
claim about a loop. Now there is a loop to measure.

**THE CATCH THAT MAKES THIS A REAL TEST RATHER THAN A FISHING TRIP.** D98 already
noted the cycle perimeter is strikingly STABLE across 6 different prompts
(1.602-1.643 in units of distance travelled). A quantity that barely moves across
tasks is not obviously carrying task information. So the honest prediction is
NEGATIVE, and it is pre-registered below.

DIFFICULTY MUST NOT CHANGE TOKEN COUNT, or the whole thing is D84's length
confound again. Three length-constant ladders, and the kernel CHECKS the claim:

  `binsearch_depth` -- **the construction the deep-research pass identified as the
      single most valuable property available**: hold a sorted list FIXED and vary
      only the target. The prompt is byte-identical except one 2-digit number, yet
      the number of binary-search comparisons needed depends on where the target
      sits in the search tree. Difficulty = that depth (1..4 for a 9-element list).
      Gold is a single digit (the 1-based position).
      **If accuracy is FLAT in target-depth, that means the model is not iterating
      a search at all but looking the answer up -- which is itself an H1/H3 result
      about contraction-to-a-fixed-point rather than a failed experiment.**
  `caesar_k`   -- shift a letter forward by k, k = 1..9 (k>=10 is two digits and
      would make length covary with difficulty). Single-letter gold.
  `nth_item_k` -- position k of a FIXED 8-element list, k = 1..8. Single-digit gold.

WHAT IS MEASURED, per prompt, at the answer position:
  * the four per-block converged states x*_0..x*_3 (the cycle's vertices);
  * `perimeter`   -- sum of consecutive vertex distances;
  * `pairwise`    -- all 6 vertex-vertex distances;
  * `planarity`   -- fraction of the centred vertices' variance in the top 2
                     singular directions. A planar cycle is a genuine loop in a
                     2-plane; a non-planar one is four scattered attractors.
  * `area2`       -- product of the top two singular values, a scale for the loop;
  * `radius`      -- mean distance from the vertex centroid;
  * plus the gold rank curve, so accuracy and cycle geometry come from the SAME
    forward and can be related per item.

PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE THE RUN (CLAUDE.md section 1):
  1. **No cycle statistic varies monotonically with difficulty at matched token
     count.** Spearman(level, perimeter) will not clear a Bonferroni threshold
     over the statistics tested. Basis: D98's perimeter stability, plus every
     prior H2 instrument returning nothing.
  2. **The cycle is highly planar** (top-2 variance fraction > 0.9). Four points
     always fit a 3-space exactly, so this is only informative if it is near 1.0
     -- and it is reported with that caveat rather than as a discovery.
  3. **`binsearch_depth` accuracy is FLAT in target depth.** If it instead falls
     with depth, Huginn is genuinely iterating a search and this becomes the
     project's first difficulty ladder with a mechanistic interpretation.

**A positive on (1) would be the first support for H2 in this project's history,
and it would be on an object no prior instrument could see. A negative closes H2
on the one object where it was still live.** Either way it is a result.

THE CONFOUND THIS DESIGN CONTROLS, AND THE ONE IT CANNOT. It controls LENGTH:
token count is constant within each family and the kernel checks it. It does NOT
control TOKEN IDENTITY -- in `binsearch_depth` the target numeral necessarily
changes with difficulty, and D88 established that this model's trajectory geometry
responds to a changed input token whether or not that token changes the
computation. **So a POSITIVE on prediction 1 would be ambiguous between "the cycle
tracks difficulty" and "the cycle reads the target token", and would need a
D88-style same-computation control before being claimed.** A NEGATIVE is
unambiguous, which is the asymmetry that makes this worth running now.
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
N_ITEMS = 3
L = string.ascii_lowercase

# A FIXED sorted list. Only the target changes, so prompts are byte-identical
# except one 2-digit number -- the deep-research pass's "fixed body" construction.
BS_LIST = [11, 23, 34, 42, 56, 61, 78, 85, 97]


def _bs_depth(idx0, lo=0, hi=len(BS_LIST) - 1, d=1):
    """Binary-search comparison depth for 0-based index `idx0` in a 9-element list.

    This is the DIFFICULTY: how many comparisons an actual binary search needs.
    The middle element is depth 1; the deepest are depth 4.
    """
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == idx0:
            return d
        d += 1
        if idx0 < mid:
            hi = mid - 1
        else:
            lo = mid + 1
    return d


def _binsearch(level, i):
    """All items at a given search depth; `i` selects among them (cyclically)."""
    cands = [j for j in range(len(BS_LIST)) if _bs_depth(j) == level]
    j = cands[i % len(cands)]
    body = " ".join(str(v) for v in BS_LIST)
    return (f"Sorted list: {body}\nAt what position is {BS_LIST[j]}? "
            f"Answer with the position number.", str(j + 1))


def _caesar(k, i):
    ch = L[(i * 7 + 2) % 26]
    return (f"Shift the letter {ch} forward by {k} in the alphabet. "
            f"Answer with one letter.", L[(L.index(ch) + k) % 26])


NTH_LIST = [4, 9, 2, 7, 1, 8, 3, 6]


def _nth(k, i):
    xs = NTH_LIST[i:] + NTH_LIST[:i]
    return (f"List: {' '.join(map(str, xs))}\nWhat is item number {k} in this list? "
            f"Answer with the number.", str(xs[k - 1]))


FAMILIES = {
    "binsearch_depth": (list(range(1, 5)), _binsearch),
    # k <= 9 ONLY: k=10..12 are two digits, which makes prompt length
    # covary with difficulty (72 vs 73 chars) -- the exact confound this
    # run exists to avoid. Caught by dry-running the generators.
    "caesar_k": (list(range(1, 10)), _caesar),
    "nth_item_k": (list(range(1, 9)), _nth),
}


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("cyclegeom.json")
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

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def one(prompt, gold):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        g_ids = tok(gold, add_special_tokens=False).input_ids
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        seen = {j: [] for j in range(n_blocks)}
        ranks = []
        handles = []

        def mk(j):
            def hook(_m, _i, o):
                st = o.detach()
                seen[j].append(st[0, n_p - 1, :].float().cpu().numpy())
                if j == n_blocks - 1:      # rank read at the same place as every other bank
                    with torch.no_grad():
                        row = torch.log_softmax(
                            coda_head(st, freqs).float()[0, n_p - 1], dim=-1)
                        ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)
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
        x = np.stack([np.stack(seen[j][:n]) for j in range(n_blocks)], axis=1)
        star = x[-1].astype(np.float64)                 # [n_blocks, d] cycle vertices

        pair = [float(np.linalg.norm(star[a] - star[b]))
                for a in range(n_blocks) for b in range(a + 1, n_blocks)]
        perim = float(sum(np.linalg.norm(star[(j + 1) % n_blocks] - star[j])
                          for j in range(n_blocks)))
        cen = star - star.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(cen, compute_uv=False)
        planarity = float((sv[:2] ** 2).sum() / (sv ** 2).sum())
        resid = float(np.median([np.linalg.norm(x[-1, j] - x[-2, j])
                                 for j in range(n_blocks)]))
        return {"perimeter": perim, "pairwise": pair,
                "median_pairwise": float(np.median(pair)),
                "planarity": planarity, "area2": float(sv[0] * sv[1]),
                "radius": float(np.linalg.norm(cen, axis=1).mean()),
                "block_residual": resid,
                "sep_over_resid": float(np.median(pair) / resid) if resid else float("inf"),
                "n_tokens": int(n_p), "best_rank": int(min(ranks)),
                "best_depth": int(np.argmin(ranks)) + 1,
                "correct": bool(min(ranks) == 1),
                "multi_token_gold": bool(len(g_ids) > 1), "ok": True}

    rows = []
    for fam, (levels, fn) in FAMILIES.items():
        print(f"\n=== {fam} ===", flush=True)
        for lv in levels:
            for i in range(N_ITEMS):
                prompt, gold = fn(lv, i)
                try:
                    r = one(prompt, gold)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
                r.update({"family": fam, "level": lv, "item": i, "gold": gold})
                rows.append(r)
            cell = [r for r in rows if r["family"] == fam and r["level"] == lv and r.get("ok")]
            acc = float(np.mean([r["correct"] for r in cell]))
            per = float(np.mean([r["perimeter"] for r in cell]))
            pl = float(np.mean([r["planarity"] for r in cell]))
            nt = sorted({r["n_tokens"] for r in cell})
            print(f"  level {lv:>2}: acc {acc:6.1%}  perimeter {per:9.3f}  "
                  f"planarity {pl:.4f}  n_tokens {nt[0]}-{nt[-1]}", flush=True)
        json.dump(rows, open(out_path, "w"))

    print("\n=== TOKEN-COUNT GATE (difficulty must not change length) ===", flush=True)
    for fam in FAMILIES:
        ns = sorted({r["n_tokens"] for r in rows if r["family"] == fam and r.get("ok")})
        print(f"  {fam:>16}: n_tokens {ns[0]}-{ns[-1]}  "
              f"{'CONSTANT' if len(ns) == 1 else 'VARIES -- confounded, report separately'}",
              flush=True)

    from scipy.stats import spearmanr
    STATS = ("perimeter", "median_pairwise", "planarity", "area2", "radius")
    alpha = 0.05 / (len(STATS) * len(FAMILIES))
    print(f"\n=== PRE-REGISTERED TEST: does any cycle statistic track difficulty? ===",
          flush=True)
    print(f"  Bonferroni alpha over {len(STATS)}x{len(FAMILIES)} = {alpha:.5f}", flush=True)
    hits = 0
    for fam in FAMILIES:
        cell = [r for r in rows if r["family"] == fam and r.get("ok")]
        if len(set(r["level"] for r in cell)) < 3:
            continue
        lv = [r["level"] for r in cell]
        accs = [r["correct"] for r in cell]
        rho_a, p_a = spearmanr(lv, accs)
        print(f"  {fam:>16}  accuracy vs level: rho={rho_a:+.3f} p={p_a:.4f}", flush=True)
        for s in STATS:
            rho, p = spearmanr(lv, [r[s] for r in cell])
            mark = "  <== CLEARS" if p < alpha else ""
            hits += int(p < alpha)
            print(f"      {s:>16}: rho={rho:+.3f}  p={p:.5f}{mark}", flush=True)
    print(f"\n  statistics clearing the corrected threshold: {hits}", flush=True)
    if hits == 0:
        print("  PREDICTION 1 HOLDS: no cycle statistic tracks difficulty at matched "
              "token count.", flush=True)
        print("  H2 returns nothing even on the cycle -- the one object where it was "
              "still live.", flush=True)
    else:
        print("  **PREDICTION 1 FAILS: something on the cycle tracks difficulty.** "
              "First H2 support in the project.", flush=True)

    json.dump({"rows": rows, "config": {"num_steps": NUM_STEPS, "n_items": N_ITEMS,
                                        "bs_list": BS_LIST, "model": MODEL_ID,
                                        "revision": REVISION}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
