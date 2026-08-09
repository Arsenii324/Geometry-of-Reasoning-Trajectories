"""Does the trajectory's shape predict the answer WITHIN a prompt, where the only
thing that varies is the random initial latent?

WHY THIS IS THE EXPERIMENT THE LITERATURE DEMANDS. Published results reporting
correctness decodable from trajectory geometry all obtain their within-prompt
variance from stochastic ROLLOUTS. This project's decode is deterministic and its
main block draws one h_0 per item, so it has had no within-prompt contrast at all --
its correctness null is currently a statement about BETWEEN-prompt variation. D90
found the missing handle: h_0 is unseeded (D78), gold rank varies in 6 of 8 prompts
across h_0 alone (up to 6x), and 3 of 8 prompts SPLIT on correctness -- `parity8` at
5 correct / 5 incorrect over ten draws. With h_0 seeded, 0 of 8 vary.

D90 then ran the test on what geomcap happened to bank: shape -> log10 gold rank,
centred within prompt, Spearman +0.295 against a null of -0.005, p = 0.076 over 80
orbits; shape -> correctness 33.3% leave-one-out over 30, p = 0.22. Not significant,
and not a clean zero either. n = 80 and n = 30 are simply too small, and that is the
only thing this run changes.

THE DESIGN. Prompts chosen AT THE DECISION BOUNDARY, because only those can split.
A prompt whose gold sits at rank 1 in every draw (`echo_digit`) or at rank ~2000 in
every draw (`rot13_word`) carries no correctness information however much h_0 moves
it. The battery's own per-item best_rank from `geometry-geomcap` names them, so the
selection is data-driven rather than guessed -- but it is made from a DIFFERENT run's
ranks, so the selection cannot be circular with this run's labels.

PRE-REGISTERED (CLAUDE.md section 1).
P1 -- GATE. At least 8 prompts must SPLIT on correctness here (both classes present
      across their replicates), or the binary arm is not analysed. Reported either way.
P2 -- Within-prompt shape -> log10 gold rank, ridge, cross-validated, against a
      null that permutes the outcome WITHIN prompt. The continuous arm; it uses every
      orbit, splitting or not.
P3 -- Within-prompt shape -> correctness on the splitting prompts, balanced-accuracy,
      same null.
P4 -- POSITION as the comparison arm at the same feature count, as in D84.
P5 -- A DETECTION FLOOR, planted on within-prompt-permuted labels. A null here
      without one repeats D70, and this run exists precisely because D90's version
      was underpowered.

Computes nothing (B14): states, rank curves, prompts and golds, then stops.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run load_arm free_arm


def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


def load_arm(spec, cfg, revision=None):
    """Load one weight-set. `spec` is None for a fresh random init, else a repo id.

    Backfills config attributes ABSENT from an older checkpoint from the final
    model's config -- intermediate Huginn checkpoints predate fields the current
    modeling code reads (`test_time_noise`), and without this they raise
    AttributeError. Only missing keys are copied, and every backfill is logged.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    if spec is None:
        torch.manual_seed(revision if isinstance(revision, int) else 0)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    else:
        ck = AutoConfig.from_pretrained(spec, revision=revision, trust_remote_code=True)
        added = [k for k, v in vars(cfg).items()
                 if not hasattr(ck, k) and not k.startswith("_")]
        for k in added:
            setattr(ck, k, getattr(cfg, k))
        if added:
            print(f"  backfilled {len(added)} config attrs: {sorted(added)}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec, revision=revision, config=ck, trust_remote_code=True,
            low_cpu_mem_usage=True)
    return model.to(torch.float32).to("cuda").eval()


def free_arm(model, repo_id=None):
    """Release a weight-set and report what was actually reclaimed.

    Printing free memory is the point: a silent cleanup is how eight checkpoints
    were lost to OOM with 13.46 GiB still held (see `geometry-rho-direct`). Also
    purges the HF cache for `repo_id`, since ten 7GB checkpoints exhaust the disk.
    """
    import gc
    import os
    import shutil

    import torch
    try:
        model = model.to("cpu") if model is not None else None
    except Exception:                                          # noqa: BLE001, S110
        pass
    del model
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"  after cleanup: {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB",
          flush=True)
    if repo_id:
        d = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                         "models--" + repo_id.replace("/", "--"))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"  purged {d}", flush=True)

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
NUM_STEPS = 64          # matches every other bank, so all of them pool
N_ITEMS = 24            # the battery's item set, so BOUNDARY prompts can be named
VOCAB_CHANCE = 32768
OUTDIR = "/kaggle/working"

# The ceiling block. 8 prompts is enough to span the measured accuracy range while
# leaving REP_N=10 replicates each affordable; 10 gives 9 df per prompt, which is
# what a variance ratio needs to be worth reporting.
# The whole budget goes into REPLICATION, not breadth. 16 boundary prompts x 32
# unseeded draws = 512 orbits, against geomcap's 8 x 10. D90's continuous arm had
# n = 80 and reached p = 0.076; this is 6.4x the orbits and 4x the draws per prompt.
REP_FAMS = 16
REP_N = 32
FIX_N = 2               # the determinism control still runs, cheaply
FIX_SEED = 20260809

# Chosen from `geometry-geomcap`'s per-item best_rank -- a DIFFERENT run, so the
# selection cannot be circular with this run's labels. A prompt can only split on
# correctness if h_0 can move its gold across rank 1, and D90 measured the movement
# at roughly a factor of 2, so the band is ranks 1-6 with ties broken toward 2-4.
BOUNDARY_MIN, BOUNDARY_MAX = 1, 6

# Inlined so the artefact that RAN carries its own selection. Ranks are
# geomcap's, from a different run, so they cannot be circular with this
# run's labels.
BOUNDARY = [
    {
        "family": "add1",
        "item": 8,
        "geomcap_rank": 2
    },
    {
        "family": "add1",
        "item": 10,
        "geomcap_rank": 3
    },
    {
        "family": "add_2d",
        "item": 0,
        "geomcap_rank": 2
    },
    {
        "family": "add_2d",
        "item": 1,
        "geomcap_rank": 3
    },
    {
        "family": "caesar1_letter",
        "item": 2,
        "geomcap_rank": 3
    },
    {
        "family": "caesar1_letter",
        "item": 12,
        "geomcap_rank": 3
    },
    {
        "family": "compare",
        "item": 2,
        "geomcap_rank": 2
    },
    {
        "family": "compare",
        "item": 7,
        "geomcap_rank": 2
    },
    {
        "family": "count16",
        "item": 17,
        "geomcap_rank": 2
    },
    {
        "family": "count4",
        "item": 1,
        "geomcap_rank": 2
    },
    {
        "family": "count8",
        "item": 21,
        "geomcap_rank": 2
    },
    {
        "family": "count_mod3",
        "item": 2,
        "geomcap_rank": 2
    },
    {
        "family": "count_mod3",
        "item": 4,
        "geomcap_rank": 2
    },
    {
        "family": "echo_word",
        "item": 3,
        "geomcap_rank": 2
    },
    {
        "family": "echo_word",
        "item": 10,
        "geomcap_rank": 2
    },
    {
        "family": "last_item",
        "item": 0,
        "geomcap_rank": 3
    }
]

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
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
          flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)

    # Resolve the boundary selection against the battery's own generators, so the
    # prompts are byte-identical to the ones geomcap ranked.
    chosen = []
    for b in BOUNDARY:
        prompt, gold, dist = items(b["family"])[b["item"]]
        chosen.append((f"{b['family']}_i{b['item']:02d}", b["family"], b["item"],
                       prompt, gold, dist, b["geomcap_rank"]))
    print(f"{len(chosen)} boundary prompts, {REP_N} unseeded draws + {FIX_N} seeded "
          f"each = {len(chosen) * (REP_N + FIX_N)} orbits", flush=True)
    for tag, _, _, _, gold, _, r in chosen:
        print(f"    {tag:>20}  gold {gold!r:>8}  geomcap rank {r}", flush=True)

    model = load_arm(MODEL_ID, cfg, REVISION)
    manifest = []
    t0 = time.time()

    def flush(rec):
        manifest.append(rec)
        with open(os.path.join(OUTDIR, "manifest.json"), "w") as fh:
            json.dump(manifest, fh)

    for tag, fam, item, prompt, gold, dist, gc_rank in chosen:
        for k in range(REP_N):
            rec = bank_one(model, tok, torch, f"rep_{tag}_r{k:02d}", "rep", fam,
                           item, prompt, gold, dist)
            rec["geomcap_rank"] = gc_rank
            flush(rec)
        for k in range(FIX_N):
            rec = bank_one(model, tok, torch, f"fix_{tag}_r{k:02d}", "fix", fam,
                           item, prompt, gold, dist, h0_seed=FIX_SEED)
            rec["geomcap_rank"] = gc_rank
            flush(rec)

        rep = [r for r in manifest
               if r.get("ok") and r["block"] == "rep" and r["tag"].startswith(f"rep_{tag}_")]
        fix = [r for r in manifest
               if r.get("ok") and r["block"] == "fix" and r["tag"].startswith(f"fix_{tag}_")]
        if rep:
            ranks = [r["best_rank"] for r in rep]
            n_ok = sum(r["correct"] for r in rep)
            split = 0 < n_ok < len(rep)
            n_sha = len({r["state_sha"] for r in rep})
            fix_sha = len({r["state_sha"] for r in fix})
            print(f"  {tag:>20}: {n_ok}/{len(rep)} correct, rank {min(ranks)}-{max(ranks)}, "
                  f"{n_sha} distinct states (expect {len(rep)}); seeded gives {fix_sha} "
                  f"(expect 1)" + ("   <- SPLITS" if split else ""), flush=True)

    ok = [r for r in manifest if r.get("ok")]
    rep = [r for r in ok if r["block"] == "rep"]
    by = {}
    for r in rep:
        by.setdefault(r["tag"].rsplit("_r", 1)[0], []).append(r["correct"])
    n_split = sum(1 for v in by.values() if 0 < sum(v) < len(v))
    print(f"\n=== P1 GATE: {n_split}/{len(by)} prompts split on correctness "
          f"(wants >= 8) ===", flush=True)
    free_arm(model, MODEL_ID)
    print(f"=== banked {len(ok)}/{len(manifest)} in {(time.time() - t0) / 60:.1f} min ===",
          flush=True)
    print("no analysis here by design; see scripts/run_h0_within.py", flush=True)
    print("DONE", flush=True)


main()
