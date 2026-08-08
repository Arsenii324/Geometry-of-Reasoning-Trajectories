"""Does trajectory geometry track the COMPUTATION -- and if not, what does it track?

THE QUESTION THIS COMPLETES. D79 answered the within-task version and got a bounded
null: at matched answer value, correct and incorrect trajectories do not differ on
effective dimensionality, step cosine, contraction or settling time, powered to 1.5
within-stratum sd. D76 answered the across-ARM version and got a large positive:
training flips the step cosine from -0.37 to +0.54 with completely disjoint
distributions over five independent weight draws.

Between those sits the question neither answers: **within the trained model, does
the geometry differ between tasks it CAN do and tasks it cannot?** D75 supplies the
capability axis -- 21 families spanning 0% to 100% -- and this run supplies geometry
for the same items, from the same forward.

AND A BARE NULL WOULD NOT BE ENOUGH, SO THIS RUN ALSO MEASURES ITS OWN CEILING.
D78 established that Huginn's initial latent h_0 is `torch.randn_like(input_embeds)`
from an UNSEEDED generator, so two forwards of the SAME prompt differ in h_0 and in
nothing else. That makes the reliability ceiling of every geometric statistic
directly measurable rather than assumed: run one prompt R times and the spread is
initial-condition noise, in the same units as the between-family spread the
capability correlation is built from. A null with no ceiling is uninterpretable --
it cannot distinguish "the geometry does not track capability" from "this statistic
does not track anything". A null WITH a ceiling is a mechanism:

    var(statistic | same prompt, different h_0)  vs  var(statistic | family)

If the first is comparable to the second, the trajectory's shape is dominated by
where it started, not by what it is computing, and D79's within-task null is
explained rather than merely observed.

THREE BLOCKS, ONE MODEL LOAD.
  main  -- 21 families x N_ITEMS items, unseeded h_0. The capability axis AND the
           geometry, from one pass, so D78's ~8-point cross-run label noise cannot
           enter: every family's accuracy and geometry come from the same forward.
  rep   -- REP_N replicates of ONE item from each of 8 families spanning the
           MEASURED accuracy range (chosen after `main`, not from D75, so no
           cross-run dependency). Same prompt, same weights, different h_0.
  fix   -- the same 8 prompts with `torch.manual_seed(FIX_SEED)` before each
           forward. POSITIVE CONTROL on the mechanism: if these come back
           bit-identical, h_0 is proven to be the only stochastic input and `rep`'s
           spread has exactly one cause. If they do NOT, something else is random
           too and `rep` is an upper bound rather than a measurement -- either way
           the claim is bounded by evidence from this run.

Generators are the battery's verbatim (all 336 items sha256-identical, checked
before launch), so the families and their difficulty ordering are D75's.

PRE-REGISTERED (CLAUDE.md section 1).
P1 -- GATE. Per-family accuracy here must correlate with D75's across the 21
      families (Spearman >= 0.7). If not, the runs are not measuring the same thing
      and nothing below may be compared to D75.
P2 -- Spearman of per-family mean step cosine against per-family acc@1 over 21
      families. Two-sided: no direction is pre-registered, because "capable tasks
      glide more" and "capable tasks need more turning" are both tellable.
P3 -- The same for effective dimensionality and for settling time.
P4 -- CEILING. The between-family spread of each statistic must exceed its
      within-prompt spread for the P2/P3 correlations to be interpretable at all.
      Reported whatever P2/P3 return, because it is what makes a null readable.
P5 -- POSITIVE CONTROL ON THE INSTRUMENT. At least one NON-capability covariate --
      prompt length is the pre-registered one -- must correlate with the geometry.
      A statistic that correlates with nothing is not evidence about capability.
P6 -- THE KILL CONDITION. If no geometric statistic tracks capability across 21
      families, while P5 shows the same statistics do track prompt length, and D79
      found nothing within a family, then the shape of the latent trajectory is a
      property of the weights and the input's surface form, not of the computation.

Computes nothing (B14): states, rank curves, prompts and golds, then stops.
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
NUM_STEPS = 64          # b6bank's depth, so the two banks pool for within-family work
N_ITEMS = 24            # a superset of the battery's 16; `items` is prefix-stable
VOCAB_CHANCE = 32768
OUTDIR = "/kaggle/working"

# The ceiling block. 8 prompts is enough to span the measured accuracy range while
# leaving REP_N=10 replicates each affordable; 10 gives 9 df per prompt, which is
# what a variance ratio needs to be worth reporting.
REP_FAMS = 8
REP_N = 10
FIX_N = 3               # determinism control; 3 pairwise comparisons, not 1
FIX_SEED = 20260809

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


def pick_spanning(acc_by_fam, k=REP_FAMS):
    """k families evenly spaced along the MEASURED accuracy ordering.

    Chosen from this run's own accuracies rather than D75's, so the ceiling block
    spans the very axis the capability correlation is built on even if D78's h_0
    noise moved a family since D75. Ties break on the family name, so an all-zero
    accuracy vector -- which D75 makes entirely possible -- still yields k distinct
    families rather than a crash or a duplicate.
    """
    order = sorted(acc_by_fam, key=lambda f: (acc_by_fam[f], f))
    if len(order) <= k:
        return order
    idx = sorted({round(i * (len(order) - 1) / (k - 1)) for i in range(k)})
    return [order[j] for j in idx]


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


def bank_one(model, tok, torch, tag, block, fam, item, prompt, gold, dist,
             h0_seed=None):
    """One forward; persist the per-unroll state at the answer position and the
    per-unroll gold rank. Returns the manifest record, never raises.

    `h0_seed` is the ONLY knob: None leaves Huginn's own unseeded `initialize_state`
    alone, which is what every other run in this project did, and an int makes h_0
    reproducible via `torch.manual_seed` (the pattern `_lib.capture_unrolls` already
    relies on). Everything else -- prompt, weights, dtype, depth, read position --
    is identical across calls by construction.
    """
    try:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        g_ids = tok(gold, add_special_tokens=False).input_ids
        d_ids = tok(dist, add_special_tokens=False).input_ids
        n_p = ids.shape[1]
        # BOTH geometry and correctness are read at n_p - 1, the position that
        # predicts the answer. The first B6 attempt read them one token apart
        # (self-review, 2026-08-08); nothing is appended here at all.
        freqs = model.freqs_cis[:, :n_p]
        states, ranks, lps = [], [], []
        mod = model.transformer.core_block[-1]
        mod._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                st = o.detach()
                states.append(st[0, n_p - 1, :].float().cpu().numpy())
                row = torch.log_softmax(
                    coda_head(model, st, freqs).float()[0, n_p - 1], dim=-1)
                ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)
                lps.append([float(row[g_ids[0]].item()),
                            float(row[d_ids[0]].item())])

        h = mod.register_forward_hook(hook)
        try:
            if h0_seed is not None:
                torch.manual_seed(h0_seed)
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()

        arr = np.stack(states).astype(np.float32)
        np.save(os.path.join(OUTDIR, tag + ".npy"), arr)
        return {"tag": tag, "block": block, "family": fam, "item": item,
                "prompt": prompt, "gold": gold, "distractor": dist,
                "n_tokens": int(n_p), "num_steps": NUM_STEPS, "h0_seed": h0_seed,
                "rank_curve": ranks, "logp": lps, "best_rank": int(min(ranks)),
                "correct": bool(min(ranks) == 1),
                "best_depth": int(np.argmin(ranks)) + 1,
                "state_sha": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
                "shape": list(arr.shape), "ok": True}
    except Exception as exc:
        return {"tag": tag, "block": block, "family": fam, "item": item,
                "h0_seed": h0_seed, "ok": False,
                "why": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoConfig, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = load_arm(MODEL_ID, cfg, REVISION)

    manifest = []
    t0 = time.time()

    def flush(rec):
        manifest.append(rec)
        with open(os.path.join(OUTDIR, "manifest.json"), "w") as fh:
            json.dump(manifest, fh)
        return rec

    # ---- block `main`: the capability axis and the geometry, one forward each ----
    acc_by_fam = {}
    for fam in TASKS:
        for i, (prompt, gold, dist) in enumerate(items(fam)):
            flush(bank_one(model, tok, torch, f"{fam}_i{i:02d}", "main", fam, i,
                           prompt, gold, dist))
        done = [r for r in manifest if r["block"] == "main"
                and r["family"] == fam and r.get("ok")]
        acc = float(np.mean([r["correct"] for r in done])) if done else float("nan")
        acc_by_fam[fam] = acc
        # Reported per family: how many items sit in gold values carrying BOTH
        # classes. Anything else cannot support D79's within-value contrast, and
        # with 24 items instead of b6bank's 32 across 4 families this is the count
        # that decides how many families the within-task test can use at all.
        ok_g = {r["gold"] for r in done if r["correct"]}
        bad_g = {r["gold"] for r in done if not r["correct"]}
        mixed = ok_g & bad_g
        n_in = sum(1 for r in done if r["gold"] in mixed)
        print(f"  {fam:>14}: {len(done)} items, acc {acc:.0%}, "
              f"{len(mixed)} gold values carry both classes, {n_in} items in them "
              f"(within-family test wants >= 6)", flush=True)

    # ---- block `rep`: the ceiling. Same prompt, different h_0, REP_N times. ----
    span = pick_spanning(acc_by_fam)
    print(f"\n=== ceiling block: {REP_N} replicates each of item 0 from "
          f"{len(span)} families spanning acc "
          f"{min(acc_by_fam[f] for f in span):.0%}-{max(acc_by_fam[f] for f in span):.0%} "
          f"===", flush=True)
    print("    " + ", ".join(f"{f} {acc_by_fam[f]:.0%}" for f in span), flush=True)
    for fam in span:
        prompt, gold, dist = items(fam)[0]
        for k in range(REP_N):
            flush(bank_one(model, tok, torch, f"rep_{fam}_r{k:02d}", "rep", fam, 0,
                           prompt, gold, dist))
        rs = [r["best_rank"] for r in manifest
              if r["block"] == "rep" and r["family"] == fam and r.get("ok")]
        shas = {r["state_sha"] for r in manifest
                if r["block"] == "rep" and r["family"] == fam and r.get("ok")}
        print(f"  {fam:>14}: {len(rs)} replicates, best_rank {min(rs)}-{max(rs)}, "
              f"{len(shas)} distinct state hashes (expect {len(rs)}: unseeded h_0)",
              flush=True)

    # ---- block `fix`: is h_0 the ONLY stochastic input? ----
    print(f"\n=== determinism control: {FIX_N} forwards per prompt at "
          f"manual_seed({FIX_SEED}) ===", flush=True)
    for fam in span:
        prompt, gold, dist = items(fam)[0]
        for k in range(FIX_N):
            flush(bank_one(model, tok, torch, f"fix_{fam}_r{k:02d}", "fix", fam, 0,
                           prompt, gold, dist, h0_seed=FIX_SEED))
        shas = {r["state_sha"] for r in manifest
                if r["block"] == "fix" and r["family"] == fam and r.get("ok")}
        verdict = "IDENTICAL" if len(shas) == 1 else f"{len(shas)} DISTINCT -- h_0 is not the only source"
        print(f"  {fam:>14}: {verdict}", flush=True)

    free_arm(model, MODEL_ID)
    print(f"\n=== banked {sum(1 for r in manifest if r.get('ok'))}/{len(manifest)} "
          f"in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print("no analysis here by design; see scripts/run_geomcap.py", flush=True)
    print("DONE", flush=True)


main()
