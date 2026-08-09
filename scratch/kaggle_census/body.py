"""Which items does Huginn actually have DYNAMIC RANGE on? The observability census.

WHY THIS RUN EXISTS, AND WHAT IT UNBLOCKS. D95 -- the project's first causal
experiment -- did not fail on method. Its patching instrument validated itself
(P2 no-op replay reproduced the original rank curve exactly). It failed on SUPPLY:
its pre-registered gate wanted >= 4 of 9 prompts to split on correctness across
h_0 draws, and only 2 did, so the kernel declared itself VOID rather than null.
D90 saw 3 of 8 split at 10 draws; D95 saw 2 of 9 at 20. **Every causal or
within-prompt design this project can run is bottlenecked on the same scarce
resource: items where the outcome actually moves.** Guessing which those are has
now cost two runs.

WHAT IS MEASURED. For 21 families x 12 items, several unseeded h_0 draws each,
the per-unroll gold rank curve -- and from it, per ITEM:
  * `frac_correct`   -- share of draws reaching gold rank 1
  * `splits`         -- both classes present (the D90/D95 handle, the scarce thing)
  * `rank_spread`    -- max/min best_rank across draws (how much h_0 moves it)
  * `median_best_rank` and `best_depth` spread (when the answer is available)
An item is OBSERVABLE when its outcome is neither pinned at 0 nor at 1.

NO STATES ARE SAVED. This is deliberate and is what makes the census affordable:
`geometry-h0bank` returned 705 MB because it persisted the answer-position state
for every orbit. Rank curves alone are a few KB per forward, so breadth costs
nothing here, and the selected items can be re-run WITH states later.

TWO STAGES, so the budget goes where the signal is.
  Stage 1 (SCREEN): every family x every item x SCREEN_DRAWS draws. Cheap and wide.
  Stage 2 (DEEPEN): only items that either split in stage 1 or sit near the
    boundary (median best_rank <= BOUNDARY_RANK) get DEEPEN_DRAWS more draws, so
    their split fraction is estimated with real precision rather than from 4 draws.

PRE-REGISTERED PREDICTIONS, WRITTEN BEFORE THE RUN (CLAUDE.md section 1):
  1. **Splitting items are rare and concentrated.** D90 and D95 together suggest
     well under a third of items split. If the census finds a comparable rate,
     that is a stable property of the model and not the bad luck of two draws.
  2. **Splitting items cluster at intermediate median best_rank (roughly 1-8).**
     Items pinned at rank 1 (`echo_digit`) or at rank ~1000s (`rot13_word`) cannot
     split however much h_0 moves them -- that is D90(6)'s argument, and this run
     tests it across 21 families instead of asserting it.
  3. **`add1` supplies splitting items**, since it is the one family that did in
     D95. If it does not reproduce here, D95's two survivors were noise and the
     h_0 handle is even scarcer than stated.

THE OUTPUT IS A REUSABLE ASSET, not just a claim: a ranked table of the most
observable items, which every subsequent causal / within-prompt design selects
from instead of guessing.
"""

# @needs: run load_arm free_arm

import hashlib
import json
import os
import random
import string
import time
import traceback
import zlib

import numpy as np

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 48          # census reads ranks only; 48 spans every best_depth seen (D90: 7-25)
N_ITEMS = 24            # items() keeps the battery's item set so indices match other banks
OUTDIR = "/kaggle/working"

WORDS = ("apple", "chair", "river", "stone", "bread", "cloud", "green", "horse",
         "light", "money", "night", "paper", "queen", "table", "water", "youth")

def items(task, n=N_ITEMS):
    """(prompt, gold, distractor) triples. Gold is single-token where possible.

    `zlib.crc32`, NOT `hash()`: Python salts str hashes per process, so a kernel
    seeded with `hash(task)` draws a different item set on every run (D69(3)).
    """
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + zlib.crc32(task.encode()) % 997)
        L = string.ascii_lowercase
        if task == "echo_digit":                                  # trivial retrieval
            v = rng.randint(0, 9)
            out.append((f"Repeat this number exactly.\nNumber: {v}", str(v),
                        str((v + 3) % 10)))
        elif task == "echo_word":
            w = rng.choice(WORDS)
            out.append((f"Repeat this word exactly.\nWord: {w}", w,
                        rng.choice([x for x in WORDS if x != w])))
        elif task == "nth_item":                                  # indexing
            xs = [rng.randint(0, 9) for _ in range(5)]
            k = rng.randint(1, 5)
            out.append((f"What is item number {k} in this list?\n"
                        f"List: {' '.join(map(str, xs))}", str(xs[k - 1]),
                        str((xs[k - 1] + 3) % 10)))
        elif task == "last_item":                                 # recency
            xs = [rng.randint(0, 9) for _ in range(6)]
            out.append((f"What is the last number in this list?\n"
                        f"List: {' '.join(map(str, xs))}", str(xs[-1]),
                        str((xs[-1] + 3) % 10)))
        elif task == "add1":
            v = rng.randint(0, 8)
            out.append((f"What is {v} + 1?", str(v + 1), str((v + 4) % 10)))
        elif task == "sub1":
            v = rng.randint(1, 9)
            out.append((f"What is {v} - 1?", str(v - 1), str((v + 4) % 10)))
        elif task == "add_2d":                                    # harder arithmetic
            a, b = rng.randint(10, 49), rng.randint(10, 49)
            out.append((f"What is {a} + {b}?", str(a + b), str(a + b + 3)))
        elif task == "compare":
            a, b = rng.sample(range(1, 100), 2)
            out.append((f"Which number is larger, {a} or {b}?", str(max(a, b)),
                        str(min(a, b))))
        elif task == "count4":
            b = [rng.randint(0, 1) for _ in range(4)]
            g = sum(b)
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 2) % 5)))
        elif task == "count8":
            b = [rng.randint(0, 1) for _ in range(8)]
            g = sum(b)
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 3) % 9)))
        elif task == "count16":
            b = [rng.randint(0, 1) for _ in range(16)]
            g = sum(b)
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 3) % 17)))
        elif task == "count_mod3":                                # modular state
            b = [rng.randint(0, 1) for _ in range(9)]
            g = sum(b) % 3
            out.append(("Count how many ones are in this sequence, then give the "
                        "remainder when divided by 3.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 1) % 3)))
        elif task == "parity8":                                   # theory: impossible
            b = [rng.randint(0, 1) for _ in range(8)]
            g = sum(b) % 2
            out.append(("Is the number of ones in this sequence even or odd? "
                        "Answer 0 for even and 1 for odd.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str(1 - g)))
        elif task == "track_total":                               # accumulation (H2's track)
            ops = [rng.choice([1, -1]) for _ in range(8)]
            g = sum(ops)
            body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1."
                                             for o in ops)
            out.append((body + " Final total?", str(g), str(g + 2)))
        elif task == "local_last":                                # length-matched control
            ops = [rng.choice([1, -1]) for _ in range(8)]
            g = "Add" if ops[-1] > 0 else "Subtract"
            body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1."
                                             for o in ops)
            out.append((body + " What was the last instruction?", g,
                        "Subtract" if g == "Add" else "Add"))
        elif task == "max_run":                                   # order-dependent
            b = [rng.randint(0, 1) for _ in range(12)]
            best = cur = 1
            for i in range(1, len(b)):
                cur = cur + 1 if b[i] == b[i - 1] else 1
                best = max(best, cur)
            out.append(("What is the length of the longest run of identical "
                        "symbols in this sequence?\n"
                        f"Sequence: {' '.join(map(str, b))}", str(best),
                        str(best + 1)))
        elif task == "sort_min":
            xs = rng.sample(range(1, 100), 3)
            out.append((f"What is the smallest of these numbers: "
                        f"{xs[0]}, {xs[1]}, {xs[2]}?", str(min(xs)),
                        str(sorted(xs)[1])))
        elif task == "succ_letter":
            c = rng.choice(L[:25])
            out.append((f"What letter comes after '{c}' in the alphabet?",
                        L[L.index(c) + 1], L[(L.index(c) + 5) % 26]))
        elif task == "caesar1_letter":                            # easiest cipher cell
            c = rng.choice(L[:25])
            out.append((f"Shift this letter forward by 1 in the alphabet.\n"
                        f"Letter: {c}", L[L.index(c) + 1], L[(L.index(c) + 7) % 26]))
        elif task == "caesar1_word":
            w = "".join(rng.choice(L) for _ in range(4))
            enc = "".join(L[(L.index(c) + 1) % 26] for c in w)
            out.append((f"Shift each letter of this text backward by 1 in the "
                        f"alphabet.\nText: {enc}", w,
                        "".join(rng.choice(L) for _ in range(4))))
        else:                                                     # rot13_word
            w = "".join(rng.choice(L) for _ in range(4))
            enc = "".join(L[(L.index(c) + 13) % 26] for c in w)
            out.append((f"Decode this ROT13 text.\nText: {enc}", w,
                        "".join(rng.choice(L) for _ in range(4))))
    return out

TASKS = ("echo_digit", "echo_word", "nth_item", "last_item",
         "add1", "sub1", "add_2d", "compare",
         "count4", "count8", "count16", "count_mod3", "parity8",
         "track_total", "local_last", "max_run", "sort_min", "succ_letter",
         "caesar1_letter", "caesar1_word", "rot13_word")

def coda_head(model, h_state, freqs_cis):
    """model's own ln_f -> coda -> ln_f -> lm_head on an intermediate state.

    TWO ln_f calls, not one: D71 validated exactly this tail as bit-identical to the
    model's own logits (max|delta| = 0.000000), with the known-wrong one-ln_f variant
    separating at 2.33-2.46. Do not shortcut it.
    """
    import torch
    x = model.transformer.ln_f(h_state)
    block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
    for block in model.transformer.coda:
        block_idx -= 1
        x = block(x, freqs_cis, block_idx, None, None)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)



SCREEN_DRAWS = 4          # stage 1: wide and cheap
DEEPEN_DRAWS = 20         # stage 2: only for candidates
BOUNDARY_RANK = 8         # "near the boundary" for stage-2 selection
CENSUS_ITEMS = 12         # items per family (of the battery's 24)


def rank_curve_only(model, tok, torch, fam, item, prompt, gold, dist, h0_seed=None):
    """One forward, rank curve only -- no state persisted (see the docstring).

    Read position and readout are IDENTICAL to `bank_one` in
    `scratch/kaggle_h0bank/body.py` (answer position n_p-1, D71-validated
    `coda_head`), so this census's ranks are directly comparable to every other
    bank in the project. `h0_seed=None` leaves Huginn's own unseeded
    `initialize_state` alone, which is the whole point.
    """
    try:
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
                    coda_head(model, o.detach(), freqs).float()[0, n_p - 1], dim=-1)
                ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)

        h = mod.register_forward_hook(hook)
        try:
            if h0_seed is not None:
                torch.manual_seed(h0_seed)
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        return {"family": fam, "item": item, "h0_seed": h0_seed,
                "n_tokens": int(n_p), "gold": gold,
                # D89's defect made explicit on EVERY row rather than rediscovered
                "multi_token_gold": bool(len(g_ids) > 1),
                "best_rank": int(min(ranks)), "final_rank": int(ranks[-1]),
                "correct": bool(min(ranks) == 1),
                "best_depth": int(np.argmin(ranks)) + 1,
                "rank_curve": ranks, "ok": True}
    except Exception as exc:
        return {"family": fam, "item": item, "h0_seed": h0_seed, "ok": False,
                "why": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


def summarise(rows):
    """Per-item observability, from whatever draws that item has."""
    by = {}
    for r in rows:
        if r.get("ok"):
            by.setdefault((r["family"], r["item"]), []).append(r)
    out = []
    for (fam, item), rs in sorted(by.items()):
        cor = [r["correct"] for r in rs]
        br = [r["best_rank"] for r in rs]
        bd = [r["best_depth"] for r in rs]
        n_cor = sum(cor)
        out.append({
            "family": fam, "item": item, "n_draws": len(rs),
            "n_correct": n_cor, "frac_correct": n_cor / len(rs),
            "splits": bool(0 < n_cor < len(rs)),
            "median_best_rank": float(np.median(br)),
            "min_best_rank": int(min(br)), "max_best_rank": int(max(br)),
            "rank_spread": float(max(br) / max(1, min(br))),
            "best_depth_min": int(min(bd)), "best_depth_max": int(max(bd)),
            "multi_token_gold": bool(rs[0].get("multi_token_gold", False)),
        })
    return out


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoConfig, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    t0 = time.time()
    # Identical construction to `scratch/kaggle_h0bank/body.py`'s main(), so the
    # weights, dtype and tokenisation this census reports are the same objects
    # every other bank in the project measured.
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = load_arm(MODEL_ID, cfg, REVISION)

    rows = []
    print(f"=== STAGE 1: screen {len(TASKS)} families x {CENSUS_ITEMS} items "
          f"x {SCREEN_DRAWS} draws ===", flush=True)
    for fam in TASKS:
        its = items(fam)[:CENSUS_ITEMS]
        for item, (prompt, gold, dist) in enumerate(its):
            for d in range(SCREEN_DRAWS):
                rows.append(rank_curve_only(model, tok, torch, fam, item,
                                            prompt, gold, dist, h0_seed=None))
        s = [x for x in summarise(rows) if x["family"] == fam]
        n_split = sum(1 for x in s if x["splits"])
        acc = float(np.mean([x["frac_correct"] for x in s])) if s else float("nan")
        print(f"  {fam:>16}: acc {acc:5.1%}  splitting items {n_split}/{len(s)}  "
              f"[{time.time()-t0:6.0f}s]", flush=True)
        json.dump(rows, open(os.path.join(OUTDIR, "census_raw.json"), "w"))

    stage1 = summarise(rows)
    cands = [x for x in stage1
             if x["splits"] or x["median_best_rank"] <= BOUNDARY_RANK]
    print(f"\n=== STAGE 2: deepen {len(cands)} candidate items "
          f"x {DEEPEN_DRAWS} more draws ===", flush=True)
    for c in cands:
        fam, item = c["family"], c["item"]
        prompt, gold, dist = items(fam)[item]
        for d in range(DEEPEN_DRAWS):
            rows.append(rank_curve_only(model, tok, torch, fam, item,
                                        prompt, gold, dist, h0_seed=None))
        json.dump(rows, open(os.path.join(OUTDIR, "census_raw.json"), "w"))

    final = summarise(rows)
    deep = [x for x in final if x["n_draws"] > SCREEN_DRAWS]
    splitting = [x for x in final if x["splits"]]
    splitting.sort(key=lambda x: -min(x["frac_correct"], 1 - x["frac_correct"]))

    print(f"\n=== CENSUS RESULT: {len(splitting)}/{len(final)} items split "
          f"({len(deep)} measured deeply) ===", flush=True)
    print(f"  {'family':>16} {'item':>4} {'draws':>5} {'frac_cor':>8} "
          f"{'med_rank':>8} {'spread':>7} {'multi_tok':>9}", flush=True)
    for x in splitting[:40]:
        print(f"  {x['family']:>16} {x['item']:>4} {x['n_draws']:>5} "
              f"{x['frac_correct']:>8.2f} {x['median_best_rank']:>8.1f} "
              f"{x['rank_spread']:>7.2f} {str(x['multi_token_gold']):>9}", flush=True)

    # PREDICTION CHECKS, printed so the raw log carries them (CLAUDE.md section 4)
    print("\n  PRE-REGISTERED CHECKS", flush=True)
    print(f"    (1) splitting rate = {len(splitting)}/{len(final)} = "
          f"{len(splitting)/max(1,len(final)):.1%}", flush=True)
    if splitting:
        mr = [x["median_best_rank"] for x in splitting]
        print(f"    (2) splitting items' median_best_rank: min {min(mr):.1f} "
              f"med {np.median(mr):.1f} max {max(mr):.1f} "
              f"(predicted to cluster in 1-{BOUNDARY_RANK})", flush=True)
        fams = sorted({x["family"] for x in splitting})
        print(f"    (3) add1 present: {'add1' in fams}; splitting families: {fams}",
              flush=True)
    else:
        print("    (2,3) NO splitting items at all -- the h_0 handle does not exist "
              "at this scale, which retires the whole D90-derived design family.",
              flush=True)

    json.dump({"rows": rows, "summary": final,
               "config": {"screen_draws": SCREEN_DRAWS, "deepen_draws": DEEPEN_DRAWS,
                          "boundary_rank": BOUNDARY_RANK, "items": CENSUS_ITEMS,
                          "num_steps": NUM_STEPS, "tasks": list(TASKS)}},
              open(os.path.join(OUTDIR, "census.json"), "w"))
    print(f"\nwrote census.json  [{time.time()-t0:.0f}s]", flush=True)
    free_arm(model, MODEL_ID)
    print("DONE", flush=True)


main()

