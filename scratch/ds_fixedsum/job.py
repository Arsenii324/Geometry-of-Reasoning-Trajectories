"""B4b: a difficulty ladder where the ANSWER IS INDEPENDENT OF THE DIFFICULTY.

WHY THIS EXISTS. D110 found H2's behavioural arm running backwards on `addk`:
harder problems reach their best readout depth EARLIER (rho = -0.440 at the true
problem unit, two-sided p = 0.0103 holding the answer token fixed). But `addk`
obeys `gold = v + k` exactly, so difficulty, first operand and answer carry only
TWO degrees of freedom, and no stratification separates all three. D110 identified
k as the driver by an indirect argument -- two arms that move `v` in OPPOSITE
directions both give ~-0.5, while holding k fixed gives +0.07 -- but the direction
was not pre-registered and the family is one. **Every other candidate ladder is
unusable and this was checked, not assumed:** `caesar_k` is clean but sits at ~0%
accuracy at every level (0.00, 0.00, 0.08, 0.00, ... in D101's banked rows), and
`nth_item_k`'s generator makes items 0, 2 and 4 the LITERALLY IDENTICAL prompt.

THE DESIGN. Six addend slots, always six, every addend a single digit:

    What is 0+1+1+0+3+0? Answer with the number.

Difficulty k is the number of NONZERO addends. The gold is the sum, held FIXED
while k varies. Every gold in 5..9 admits a composition into exactly k positive
parts for every k in 1..5 (checked: the 5x5 cell table is full), so the design is
fully crossed and **rho(k, gold) = 0 BY CONSTRUCTION** rather than by
stratification after the fact. The token count is constant by construction too --
six single digits in a fixed template -- and is nonetheless VERIFIED in-kernel,
because D101 asserted a length claim that was false.

This is the thing `addk` cannot be: a difficulty axis orthogonal to the readout
token.

PRE-REGISTERED PREDICTIONS.

  P1  PRIMARY, directional, and it is D110's direction carried over rather than a
      fresh two-sided fishing trip: rho(k, best_depth) < 0 at the (k, gold) unit.
      If D110's reversal is a fact about Huginn, it appears here with the answer
      token fully decoupled. If it was a fact about `addk`'s collinearity, it
      vanishes or flips.
  P2  NULL THAT MUST PASS FIRST -- the capability gate. If accuracy is at floor
      (< 5% at every k) the ladder measures nothing and `best_depth` is the argmin
      of rank curves that never reach 1. Report VOID, not a null. D110(c) showed a
      63%-unsolved pool did not bias the estimate, but a 100%-unsolved one is not
      a measurement at all.
  P3  TOKEN GATE. n_tokens constant across k, checked not asserted.
  P4  UNIT. One value per (k, gold) problem, never per forward. D101's error, and
      D110 found `require_units` passing 42 keys that were 36 prompts -- so the
      key here is the PROMPT STRING, not an index into a generator.
  P5  RESIDUAL CONFOUND, MEASURED NOT IGNORED. k anti-correlates with the LARGEST
      addend (k=1 means one big term; k=5 means five small ones). This is recorded
      per item as `max_addend` along with `n_distinct` and `first_nz_pos`, so the
      analysis can stratify on it the way D110 stratified on v. A ladder whose
      confound is measured beats one whose confound is argued away.
"""

import itertools
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
SLOTS = 6
GOLDS = (5, 6, 7, 8, 9)
LEVELS = (1, 2, 3, 4, 5)
N_ITEMS = 3          # distinct compositions per (k, gold) cell
N_DRAWS = 6          # unseeded h_0 draws per item
FLOOR_ACC = 0.05     # below this at EVERY level the ladder is VOID, not null
OVERLAP_BAND = (2, 3, 4)  # max_addend values several difficulties can all reach


def compositions(g, k):
    """All ways to write g as an ordered sum of exactly k POSITIVE parts."""
    out = []
    for c in itertools.combinations(range(1, g), k - 1):
        parts, prev = [], 0
        for x in list(c) + [g]:
            parts.append(x - prev)
            prev = x
        out.append(tuple(parts))
    return out


def build_items():
    """One list of items, each carrying its own covariates.

    The rng is seeded so the item set is reproducible; h_0 is what stays unseeded.

    COMPOSITION CHOICE IS NOT ARBITRARY. `max_addend >= ceil(g/k)` and
    `max_addend <= g - k + 1`, so with a fixed single-token sum the largest addend
    MUST fall as k rises -- the P5 confound cannot be designed away (checked: it
    sits at rho ~ -0.75 under every level/gold window tried). What CAN be arranged
    is overlap: compositions whose max lies in the band several difficulties can all
    reach are preferred, and within a cell distinct max values are taken before
    repeats. That turns P5's stratified test from decorative into powered -- it
    yields 5 strata spanning >= 3 difficulties each, covering 58 of 69 items, which
    is the same identification argument D110 used on `addk`.
    """
    rng = random.Random(20260809)
    items = []
    seen_prompts = set()      # two different compositions can shuffle to the SAME
    #                           slot arrangement -- (1,2,2) and (2,1,2) both reach
    #                           0+1+2+2+0+0. Left in, that is D110(b) again: keys
    #                           distinct, prompts not. Dedupe at generation.
    for g in GOLDS:
        for k in LEVELS:
            comps = compositions(g, k)
            if not comps:
                continue
            ranked = sorted(comps,
                            key=lambda c: (0 if max(c) in OVERLAP_BAND else 1, rng.random()))
            picks, seen = [], set()
            for c in ranked:
                if len(picks) >= N_ITEMS:
                    break
                if max(c) not in seen or len(ranked) <= N_ITEMS:
                    picks.append(c)
                    seen.add(max(c))
            for c in ranked:                      # top up if diversity ran out
                if len(picks) >= N_ITEMS:
                    break
                picks.append(c)
            kept = 0
            for parts in picks:
                if kept >= N_ITEMS:
                    break
                slots = list(parts) + [0] * (SLOTS - k)
                rng.shuffle(slots)
                prompt = (f"What is {'+'.join(map(str, slots))}? "
                          f"Answer with the number.")
                if prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)
                nz = [i for i, s in enumerate(slots) if s > 0]
                items.append({
                    "prompt": prompt, "gold": str(g), "k": k, "g": g, "ci": kept,
                    "slots": slots, "max_addend": max(slots),
                    "n_distinct": len({s for s in slots if s > 0}),
                    "first_nz_pos": nz[0] if nz else -1,
                })
                kept += 1
    return items


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("fixedsum.json")
    print(f"results -> {out_path}", flush=True)

    items = build_items()
    print(f"{len(items)} items over {len(GOLDS)} golds x {len(LEVELS)} difficulties",
          flush=True)
    # The balance that justifies the whole design, asserted before any GPU is spent.
    cells = {(it["k"], it["g"]) for it in items}
    missing = [(k, g) for g in GOLDS for k in LEVELS if (k, g) not in cells]
    if missing:
        print(f"DESIGN BROKEN: empty cells {missing} -- gold would correlate with k. "
              f"VOID.", flush=True)
        json.dump({"verdict": "VOID_unbalanced_design", "missing": missing},
                  open(out_path, "w"))
        return
    print(f"  design is FULLY CROSSED: all {len(cells)} (k, gold) cells populated, "
          f"so gold is orthogonal to difficulty by construction", flush=True)

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
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
          flush=True)

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
        """One forward -> the gold rank at EVERY depth, from a single continuous run."""
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
    for it in items:
        for _ in range(N_DRAWS):
            try:
                r = one(it["prompt"], it["gold"])
            except Exception as exc:
                r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
            r.update({k: it[k] for k in ("prompt", "gold", "k", "g", "ci",
                                         "max_addend", "n_distinct", "first_nz_pos")})
            r.setdefault("ok", True)
            rows.append(r)
        json.dump({"rows": rows}, open(out_path, "w"))
    ok_rows = [r for r in rows if r.get("ok")]
    print(f"\n{len(ok_rows)}/{len(rows)} forwards ok", flush=True)

    print("\n=== P2 CAPABILITY GATE (must pass before best_depth means anything) ===",
          flush=True)
    acc_by_k = {}
    for k in LEVELS:
        sub = [r for r in ok_rows if r["k"] == k]
        acc_by_k[k] = float(np.mean([r["correct"] for r in sub])) if sub else 0.0
        ntok = sorted({r["n_tokens"] for r in sub})
        bds = [r["best_depth"] for r in sub]
        print(f"  k={k}: acc {acc_by_k[k]:6.1%}  median best_depth "
              f"{float(np.median(bds)):5.1f}  n_tokens {ntok}  (n={len(sub)})",
              flush=True)
    if max(acc_by_k.values()) < FLOOR_ACC:
        print(f"  VOID: accuracy is at floor everywhere (max {max(acc_by_k.values()):.1%} "
              f"< {FLOOR_ACC:.0%}). best_depth would be the argmin of rank curves that "
              f"never reach 1 -- this measures nothing about difficulty. Report VOID, "
              f"NOT a null.", flush=True)
        json.dump({"rows": rows, "verdict": "VOID_capability_floor",
                   "acc_by_k": acc_by_k}, open(out_path, "w"))
        return
    print(f"  PASSED: peak accuracy {max(acc_by_k.values()):.1%}", flush=True)

    print("\n=== P3 TOKEN GATE ===", flush=True)
    ntok = sorted({r["n_tokens"] for r in ok_rows})
    if len(ntok) != 1:
        print(f"  VOID: n_tokens varies {ntok} across the ladder, so any best_depth "
              f"shift could be a length effect (D80). Nothing is interpreted.",
              flush=True)
        json.dump({"rows": rows, "verdict": "VOID_token_count_varies",
                   "n_tokens": ntok}, open(out_path, "w"))
        return
    print(f"  PASSED: n_tokens == {ntok[0]} at every difficulty", flush=True)

    print("\n=== P4 UNIT: one value per (k, gold) PROBLEM, keyed on the prompt ===",
          flush=True)
    from traj_geom.rigor import require_units
    cellmap = {}
    for r in ok_rows:
        cellmap.setdefault((r["k"], r["gold"], r["ci"]), []).append(r)
    prompts = [r["prompt"] for r in ok_rows]
    n_distinct_prompts = len(set(prompts))
    print(f"  {len(cellmap)} problems from {n_distinct_prompts} DISTINCT prompt "
          f"strings ({len(ok_rows)} forwards)", flush=True)
    if n_distinct_prompts != len(cellmap):
        print(f"  !! {len(cellmap)} keys but {n_distinct_prompts} prompts -- the "
              f"generator collides, exactly D110(b). Keying on the prompt.", flush=True)
    bykey = {}
    for r in ok_rows:
        bykey.setdefault(r["prompt"], []).append(r)
    require_units(list(bykey), min_units=20, what="fixedsum problems")

    units = [(v[0]["k"], v[0]["max_addend"], float(np.median([x["best_depth"] for x in v])),
              float(np.mean([x["correct"] for x in v])), int(v[0]["g"]))
             for v in bykey.values()]

    from scipy.stats import spearmanr
    kk = [u[0] for u in units]
    bd = [u[2] for u in units]
    gg = [u[4] for u in units]
    mx = [u[1] for u in units]
    ac = [u[3] for u in units]

    print("\n=== P1 PRIMARY: rho(k, best_depth), D110 predicts NEGATIVE ===", flush=True)
    rho, p = spearmanr(kk, bd)
    print(f"  n = {len(units)} problems   rho = {rho:+.4f}   p(two-sided) = {p:.4f}",
          flush=True)
    print(f"  one-sided in D110's predicted NEGATIVE direction: "
          f"{p / 2 if rho < 0 else 1 - p / 2:.4f}", flush=True)

    print("\n=== THE POINT OF THIS LADDER: gold must be orthogonal to k ===", flush=True)
    rg, pg = spearmanr(kk, gg)
    print(f"  rho(k, gold) = {rg:+.4f} p={pg:.4f}   <- ~0 by construction "
          f"(`addk` had +0.68 to +0.93)", flush=True)
    rb, pb = spearmanr(gg, bd)
    print(f"  rho(gold, best_depth) = {rb:+.4f} p={pb:.4f}", flush=True)

    print("\n=== P5 RESIDUAL CONFOUND: k anti-correlates with the largest addend ===",
          flush=True)
    rm, pm = spearmanr(kk, mx)
    print(f"  rho(k, max_addend) = {rm:+.4f} p={pm:.4f}", flush=True)
    rmb, pmb = spearmanr(mx, bd)
    print(f"  rho(max_addend, best_depth) = {rmb:+.4f} p={pmb:.4f}", flush=True)
    strata = {}
    for u in units:
        strata.setdefault(u[1], []).append(u)
    rs = [spearmanr([x[0] for x in v], [x[2] for x in v])[0]
          for v in strata.values() if len({x[0] for x in v}) >= 3]
    rs = [x for x in rs if x == x]
    if rs:
        print(f"  holding max_addend FIXED: mean rho(k, best_depth) = "
              f"{float(np.mean(rs)):+.4f} over {len(rs)} strata "
              f"({sum(1 for x in rs if x < 0)} negative)", flush=True)
    else:
        print("  no max_addend stratum spans >= 3 difficulties -- cannot stratify",
              flush=True)

    ra, pa = spearmanr(kk, ac)
    print(f"\n  accuracy control: rho(k, accuracy) = {ra:+.4f} p={pa:.4f}", flush=True)

    json.dump({"rows": rows, "acc_by_k": acc_by_k,
               "primary": {"rho_k_best_depth": float(rho), "p": float(p),
                           "n_units": len(units)},
               "orthogonality": {"rho_k_gold": float(rg)},
               "config": {"num_steps": NUM_STEPS, "slots": SLOTS, "golds": list(GOLDS),
                          "levels": list(LEVELS), "n_items": N_ITEMS,
                          "n_draws": N_DRAWS, "model": MODEL_ID, "revision": REVISION}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}\nDONE", flush=True)


if __name__ == "__main__":
    main()
