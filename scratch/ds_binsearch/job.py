"""A16 / DR1 Stage 2: the fixed-body binary-search ladder. The one this project kept missing.

WHY THIS EXISTS, AND WHY IT IS EMBARRASSING THAT IT DOES NOT ALREADY.
D110 found H2's behavioural arm running backwards on `addk`, but `addk` obeys
`gold = v + k`, so difficulty and answer carry only two degrees of freedom. B4b was
built to fix that with a fixed-sum design and went **VOID**: Huginn scores 0.0% on
sums of two or more nonzero digits (D118), so the ladder had one live level.

`DR_1_tasks.md` Stage 2 had already named the construction, and `PLAN.md` line 61
already called it *"the one escape"*:

    "Fix one sorted key array of length n; sweep only `target` so prompts are
     byte-identical except one number. Difficulty = comparison depth for that
     target. This isolates geometry-vs-computation with prompt length exactly
     constant."

THE PROPERTY THAT MAKES IT WORK, CHECKED BEFORE BUILDING. Binary-search comparison
depth is NOT monotone in position: the midpoint is depth 1, the quartiles depth 2,
and the extremes depth 4. At n = 15 the rank correlation between position and depth
is **exactly 0.0000**; at n = 9 it is +0.2108. n = 15 would give perfect decoupling
but 2-digit answers for positions 10-15, which is D89's multi-token defect. **n = 9
is chosen so every answer is a single token**, and the residual +0.21 is handled the
way D110 handled `addk`'s: measured, stratified, and reported.

DIFFICULTY IS THE ALGORITHM'S, NOT NECESSARILY THE MODEL'S -- and DR1 says so, which
is why both outcomes are informative:

    "If binary_search fixed-body accuracy is flat in target-depth -> Huginn is
     shortcutting; that is itself a publishable H1/H3 result (contraction to a
     fixed point rather than iteration)."

So a null here is a finding about retrieval-versus-iteration, not a failed run.

PRE-REGISTERED.
  P1  PRIMARY, directional, carrying D110's direction: rho(depth, best_depth) < 0
      at the (array, target) unit. Two-sided p also printed.
  P2  CAPABILITY GATE, using `rigor.require_dynamic_range` -- the guard D118 forced
      into existence. At least TWO depth levels must clear 5% accuracy, else the
      ladder has no difficulty axis and the run is VOID, not null. B4b's gate asked
      whether ANY level cleared floor and passed a dead ladder.
  P3  TOKEN GATE. Within an array only the target number changes and every key is
      two digits, so the token count is constant by construction -- and is VERIFIED
      per array in-kernel, because D101 carried a length claim that was false.
  P4  UNIT. One value per (array, target) PROBLEM, keyed on the prompt string.
      D110 found `require_units` passing 42 keys that were 36 prompts because it
      compared labels, not content.
  P5  THE RESIDUAL CONFOUND, MEASURED. rho(position, depth) = +0.21 by construction.
      Position is recorded per item and the primary statistic is re-run stratified
      by it, exactly as D110 identified k by holding v and gold fixed in turn.
"""

import collections
import json
import os
import random
import subprocess
import sys


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 48
N_ARRAYS = 12
N = 9                 # positions 1..9, so every gold is a single token (D89-safe)
N_DRAWS = 4          # 108 x 4 = 432 forwards, ~50 min on gt4.1; grant money
FLOOR_ACC = 0.05


def depths(n):
    """Binary-search comparison depth for each 0-based index of a sorted array."""
    d = {}

    def rec(lo, hi, k):
        if lo > hi:
            return
        mid = (lo + hi) // 2
        d[mid] = k
        rec(lo, mid - 1, k + 1)
        rec(mid + 1, hi, k + 1)

    rec(0, n - 1, 1)
    return [d[i] for i in range(n)]


def build_items():
    """N_ARRAYS fixed bodies; within one body ONLY the target number changes."""
    rng = random.Random(20260809)
    dep = depths(N)
    items = []
    for a in range(N_ARRAYS):
        keys = sorted(rng.sample(range(10, 100), N))      # 9 distinct TWO-DIGIT keys
        body = " ".join(str(k) for k in keys)
        for pos, key in enumerate(keys):
            items.append({
                "array": a, "pos": pos, "depth": dep[pos], "key": key,
                "gold": str(pos + 1),                      # 1-based, single token
                "body": body,
                "prompt": (f"The list is sorted: {body}\n"
                           f"At what position is {key}? Answer with the position "
                           f"number."),
            })
    return items


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("binsearch.json")
    print(f"results -> {out_path}", flush=True)

    items = build_items()
    print(f"{len(items)} items = {N_ARRAYS} fixed arrays x {N} targets", flush=True)
    dep = depths(N)
    print(f"  depth by position: {dep}   levels {sorted(set(dep))}", flush=True)
    # The design property, asserted before any GPU is spent.
    per_depth = dict(sorted(collections.Counter(i["depth"] for i in items).items()))
    print(f"  items per depth: {per_depth}", flush=True)
    if len({i["prompt"] for i in items}) != len(items):
        print("DESIGN BROKEN: duplicate prompts -- VOID.", flush=True)
        json.dump({"verdict": "VOID_duplicate_prompts"}, open(out_path, "w"))
        return
    print(f"  {len({i['prompt'] for i in items})} distinct prompts (no collisions)",
          flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))

    # IMPORT THE REPO GUARDS HERE, NOT IN THE ANALYSIS BLOCK. A4b died at
    # `ImportError: require_null_can_move` after 2.5 h and 256 completed forwards,
    # because the branch was 31 commits ahead of origin and remote jobs clone
    # from GitHub. At the top this fails in 30 seconds instead.
    from traj_geom.rigor import require_dynamic_range, require_units
    print(f"repo guards imported OK: {require_dynamic_range.__name__}, "
          f"{require_units.__name__}", flush=True)

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

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
        ranks = []
        mod = model.transformer.core_block[-1]
        mod._forward_hooks.clear()

        def hook(_m, _i, o, *, n_p=n_p, freqs=freqs, g_ids=g_ids, ranks=ranks):
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
    for n, it in enumerate(items):
        for _ in range(N_DRAWS):
            try:
                r = one(it["prompt"], it["gold"])
            except Exception as exc:  # noqa: BLE001
                r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
            r.setdefault("ok", True)
            r.update({k: it[k] for k in ("array", "pos", "depth", "key", "gold",
                                         "prompt")})
            rows.append(r)
        if n % 20 == 0:
            print(f"  {n}/{len(items)}", flush=True)
            json.dump({"rows": rows}, open(out_path, "w"))
    ok = [r for r in rows if r.get("ok")]
    print(f"\n{len(ok)}/{len(rows)} forwards ok", flush=True)

    from scipy.stats import spearmanr


    print("\n=== P3 TOKEN GATE (per fixed array) ===", flush=True)
    bad = []
    for a in range(N_ARRAYS):
        nt = sorted({r["n_tokens"] for r in ok if r["array"] == a})
        if len(nt) != 1:
            bad.append((a, nt))
    if bad:
        print(f"  !! token count varies WITHIN a fixed array: {bad[:3]} -- the whole "
              f"design rests on this being constant. VOID.", flush=True)
        json.dump({"rows": rows, "verdict": "VOID_token_varies"}, open(out_path, "w"))
        return
    print(f"  PASSED: constant within every array; values "
          f"{sorted({r['n_tokens'] for r in ok})}", flush=True)

    print("\n=== P2 CAPABILITY GATE (D118's guard) ===", flush=True)
    acc = {}
    for d in sorted({r["depth"] for r in ok}):
        sub = [r for r in ok if r["depth"] == d]
        acc[d] = float(np.mean([r["correct"] for r in sub]))
        print(f"  depth {d}: acc {acc[d]:6.1%}  median best_depth "
              f"{float(np.median([r['best_depth'] for r in sub])):5.1f}  (n={len(sub)})",
              flush=True)
    try:
        require_dynamic_range(acc, floor=FLOOR_ACC, what="binary-search ladder")
        print(f"  PASSED: {sum(1 for v in acc.values() if v >= FLOOR_ACC)} of "
              f"{len(acc)} depth levels above floor", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  {exc}", flush=True)
        json.dump({"rows": rows, "verdict": "VOID_capability_floor", "acc": acc},
                  open(out_path, "w"))
        return

    print("\n=== P4 UNIT: one value per (array, target) PROBLEM ===", flush=True)
    cells = {}
    for r in ok:
        cells.setdefault(r["prompt"], []).append(r)
    require_units(list(cells), min_units=60, what="binsearch problems")
    units = [(v[0]["depth"], v[0]["pos"],
              float(np.median([x["best_depth"] for x in v])),
              float(np.mean([x["correct"] for x in v]))) for v in cells.values()]
    print(f"  {len(units)} problems from {len(ok)} forwards", flush=True)

    dd = [u[0] for u in units]
    bd = [u[2] for u in units]
    pos = [u[1] for u in units]
    ac = [u[3] for u in units]

    print("\n=== P1 PRIMARY: rho(search depth, best_depth), D110 predicts NEGATIVE ===",
          flush=True)
    rho, p = spearmanr(dd, bd)
    print(f"  n = {len(units)}   rho = {rho:+.4f}   two-sided p = {p:.4f}", flush=True)
    print(f"  one-sided in D110's direction: {p / 2 if rho < 0 else 1 - p / 2:.4f}",
          flush=True)

    print("\n=== P5 RESIDUAL CONFOUND: position correlates +0.21 with depth ===",
          flush=True)
    rp, pp = spearmanr(pos, dd)
    print(f"  rho(position, search depth) = {rp:+.4f} p={pp:.4f}  (expected ~+0.21)",
          flush=True)
    rb, pb = spearmanr(pos, bd)
    print(f"  rho(position, best_depth)   = {rb:+.4f} p={pb:.4f}", flush=True)
    strata = {}
    for u in units:
        strata.setdefault(u[1], []).append(u)
    rs = [spearmanr([x[0] for x in v], [x[2] for x in v])[0]
          for v in strata.values() if len({x[0] for x in v}) >= 2]
    rs = [x for x in rs if x == x]
    if rs:
        print(f"  holding POSITION fixed: mean rho(depth, best_depth) = "
              f"{float(np.mean(rs)):+.4f} over {len(rs)} strata "
              f"({sum(1 for x in rs if x < 0)} negative)", flush=True)

    ra, pa = spearmanr(dd, ac)
    print(f"\n  accuracy vs search depth: rho = {ra:+.4f} p={pa:.4f}", flush=True)
    print("  (DR1: if accuracy is FLAT in depth, Huginn is shortcutting rather than "
          "iterating -- itself a result about retrieval vs computation)", flush=True)

    json.dump({"rows": rows, "acc_by_depth": acc,
               "primary": {"rho": float(rho), "p": float(p), "n_units": len(units)},
               "config": {"num_steps": NUM_STEPS, "n_arrays": N_ARRAYS, "n": N,
                          "n_draws": N_DRAWS, "model": MODEL_ID,
                          "revision": REVISION}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
