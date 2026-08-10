"""B4c: the orthogonal difficulty ladder, on a task the census says Huginn can do.

THE THIRD ATTEMPT, AND THE FIRST WITH BOTH PROPERTIES AT ONCE.
  `addk`  (D110) -- difficulty and answer collinear by construction (gold = v + k),
                    so no stratification separates them.
  B4b     (D118) -- orthogonality achieved, but VOID: 0% accuracy at k >= 2, one live
                    level, no difficulty axis at all.
  A16     (D128) -- every structural gate passed, but binary-search depth runs OPPOSITE
                    to Huginn's difficulty (accuracy RISES 25% -> 58%), so the model is
                    not running the algorithm whose depth was varied.

`nth_item_k` gives both, once its generator is fixed. Difficulty is the requested INDEX
k; the answer is `xs[k-1]` for a randomly drawn list, so **the answer is independent of
the difficulty by construction** -- measured on the generated set at rho(k, gold) =
+0.064 -- and the prompt is the same length at every k.

WHY IT WAS NOT USABLE BEFORE, AND WHAT CHANGED.
D110 found the old generator built `xs = [(i*5 + j*3) % 10 ...]`; since `i*5 % 10`
cycles 0,5,0,5, **items 0, 2 and 4 were the LITERALLY IDENTICAL prompt** -- 6 slots
carrying 2 distinct problems. That defect is why D100 and D101 disagreed about this
family's accuracy (D100: 3 forwards per level; D101: 2 distinct prompts per level;
neither able to resolve it). The generator here draws each list independently and
asserts distinctness.

And D130's census says the task is alive: `nth_item` supplies **4 items inside the
20-80% band with single-token golds**, one of five families that do.

PRE-REGISTERED.
  P1  PRIMARY, directional, carrying D110's direction: rho(k, best_depth) < 0 at the
      problem unit. This is the first ladder able to test it without a confound.
  P2  CAPABILITY GATE via `rigor.require_dynamic_range` -- at least TWO difficulty
      levels above 5%, else VOID not null. This is the gate B4b's version lacked.
  P3  TOKEN GATE: constant across all k, checked not asserted.
  P4  ORTHOGONALITY GATE: |rho(k, gold)| < 0.15 on the realised item set, checked
      in-kernel. If the draw happens to correlate them, the run says so.
  P5  UNIT: the problem, keyed on the PROMPT STRING, and distinctness asserted at
      generation -- the specific failure D110 found in this family.
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
N_ITEMS = 12
LIST_LEN = 8          # 8 difficulty levels; every gold is one digit (D89-safe)
N_DRAWS = 4          # 108 x 4 = 432 forwards, ~50 min on gt4.1; grant money
FLOOR_ACC = 0.05


def build_items():
    """Independently drawn lists; difficulty = requested index; answer orthogonal to it."""
    rng = random.Random(20260810)
    items, seen = [], set()
    for i in range(N_ITEMS):
        while True:
            xs = [rng.randrange(10) for _ in range(LIST_LEN)]
            if tuple(xs) not in seen:
                seen.add(tuple(xs))
                break
        for k in range(1, LIST_LEN + 1):
            items.append({
                "item": i, "k": k, "depth": k, "pos": k - 1,
                "gold": str(xs[k - 1]), "xs": " ".join(map(str, xs)),
                "prompt": (f"What is item number {k} in this list? "
                           f"Answer with the number.\nList: {' '.join(map(str, xs))}"),
            })
    return items


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("binsearch.json")
    print(f"results -> {out_path}", flush=True)

    items = build_items()
    print(f"{len(items)} items = {N_ITEMS} lists x {LIST_LEN} difficulty levels", flush=True)
    ks = [i["k"] for i in items]
    gs = [int(i["gold"]) for i in items]
    # P4 ORTHOGONALITY GATE on the REALISED set, not on the design intent.
    mk, mg = sum(ks) / len(ks), sum(gs) / len(gs)
    num = sum((a - mk) * (b - mg) for a, b in zip(ks, gs, strict=True))
    den = (sum((a - mk) ** 2 for a in ks) * sum((b - mg) ** 2 for b in gs)) ** 0.5
    r_kg = num / den if den else 0.0
    print(f"  P4 orthogonality: r(k, gold) = {r_kg:+.4f} on the realised set", flush=True)
    if abs(r_kg) >= 0.15:
        print("  VOID: difficulty and answer are correlated in this draw.", flush=True)
        json.dump({"verdict": "VOID_not_orthogonal", "r_kg": r_kg}, open(out_path, "w"))
        return
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
            # keys must match build_items() exactly: it emits `item` and `k`, NOT
            # `array` and `key`. The first launch died here with KeyError: 'array'
            # on the very first forward, after paying the full 12-minute setup.
            r.update({key: it[key] for key in ("item", "pos", "depth", "k", "gold",
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
    for a in range(N_ITEMS):
        nt = sorted({r["n_tokens"] for r in ok if r["item"] == a})
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
               "config": {"num_steps": NUM_STEPS, "n_items": N_ITEMS,
                          "list_len": LIST_LEN,
                          "n_draws": N_DRAWS, "model": MODEL_ID,
                          "revision": REVISION}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
