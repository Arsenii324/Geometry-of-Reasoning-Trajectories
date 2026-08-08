"""A WIDE capability screen: 21 short task families, graded readout, both arms.

WHY WIDE RATHER THAN DEEP. This project has spent most of its GPU on one family
at a time -- counting, then Caesar -- and re-attacked each after it read 0%. D68
showed why that was the wrong loop: exact-match cannot tell "the model cannot do
this" from "the readout cannot see it", so a 0% cell licensed another attempt on
the same family instead of a move to a different one. With full-vocab rank the
screen is graded, and with the per-unroll hook ONE forward at num_steps=64 yields
the entire r=1..64 curve, so breadth costs almost nothing. On power: separating an
intrinsic 95% from an intrinsic 5% needs about a dozen items, not hundreds, so 16
items x 20 families buys far more information than 320 items on one family.

WHAT IT IS FOR, CONCRETELY -- three blocked things at once.

  * B6 (geometry vs correctness) is blocked by D72: correctness there was a
    deterministic function of the ANSWER VALUE, so permuting the label broke the
    prompt-geometry link at the same time. B6 needs a cell where success and
    failure COEXIST AT THE SAME GOLD VALUE. That cannot be designed a priori --
    it has to be found, and finding it is what a wide screen does.
  * B1/Caesar was dropped on two blunt metrics, one of which (char_acc) had the
    UNTRAINED arm beating the trained one in 4 of 6 cells (D57, B5). Three cipher
    cells of graded difficulty settle it properly.
  * The capability ladder D68 needed had five rungs, four of which the model
    cannot do. The families here span copy, indexing, arithmetic, counting,
    modular state, parity, accumulation, order and cipher, with a difficulty
    gradient inside several of them.

TWO DEFECTS FROM PRIOR KERNELS ARE FIXED HERE, BOTH RECORDED IN THE LEDGER.

  * `zlib.crc32`, never `hash()`. Python salts str hashes per process, so
    `geometry-graded-readout` (D68) and `geometry-discourse` drew DIFFERENT item
    sets for the same nominal cell -- measured at 96% vs 83% on one of them
    (D69(3)).
  * The PROMPT AND GOLD ARE PERSISTED per item. `geometry-correctness` saved
    neither, so establishing D72's confound needed the item set regenerated from
    the kernel's own seeding and matched back by index. Anything that wants to
    stratify on the answer value later should not have to do that.

THE READOUT, AND THE ONE CHANGE TO IT. `coda_head` is unchanged from the version
D71 validated as BIT-IDENTICAL to the model's own logits (max|delta| = 0.000000,
with the known-wrong one-ln_f variant separating at 2.3). The single change is
that `lm_head` is applied only to the positions actually read instead of the whole
sequence -- a ~10-30x saving on the dominant matmul. `lm_head` is a position-wise
Linear, so this is exact by construction rather than approximately equal; but
"exact by construction" is the kind of claim this project has been wrong about
before, so the kernel MEASURES it on the first item and aborts if it is not
literally zero (SLICE CHECK below).

WHAT THIS KERNEL DOES NOT DO: draw a conclusion. It prints descriptive per-task
numbers and persists every curve. D62, geometry-discourse and D70 each printed a
confident verdict that was wrong, always because the verdict was computed from the
same variables as the run.

PRE-REGISTERED PREDICTIONS (CLAUDE.md section 1).

P1 -- GATE, on the instrument. The slice check must report max|delta| == 0.0
      exactly. If not, nothing else in this run may be read, because the readout
      would no longer be the one D71 validated.
P2 -- GATE, on the control arm. The untrained arm must sit near chance (median
      rank within a factor of ~3 of 32768) on every family. An untrained arm that
      is not at chance means the decode is broken, which is the control that made
      D68 credible in the first place.
P3 -- At least 4 families show trained acc@1 >= 50% at some depth. D68/D69 found
      `echo_digit` at 96% and D70 found `add1` at 100%, so 2 are known; if a wide
      net over 18 more families adds fewer than 2, the model's competence really
      is confined to near-trivial retrieval and the paper should say so plainly.
P4 -- At least one family lands in the 20-80% band, which is where success and
      failure coexist and therefore where B6 becomes runnable. If NOTHING lands
      there, B6 is not rescuable by task selection and that should be recorded as
      a negative rather than retried.
P5 -- Cipher: `caesar1_letter` (a single letter, shift 1) is the easiest cipher
      cell constructible. If even that is at chance, the Caesar family is dead for
      reasons that have nothing to do with the readout, and B1 closes.

RANK IS ON THE FIRST GOLD TOKEN, which is exact for the 15 families whose gold is
a single token and a WEAKER test for the 6 where it is 2-3 (`add_2d`, `compare`,
`sort_min`, `caesar1_word`, `rot13_word`, and some `echo_word`/`count16`/
`track_total` items): first-token rank 1 does not prove the whole answer. This is
why `logp_gold` sums over EVERY gold token -- the full-answer likelihood is
recorded even though the headline statistic is first-token rank. Verified before
launch that all 336 golds recompute from the emitted prompt text under a parser
written separately from the generator, so the check is not circular.

COST. `geometry-graded-readout` measured 5 families x 24 items x 2 arms in about
1 GPU-h. This is 21 x 16 x 2 = 672 items, but prompts here are 6-34 tokens and the
lm_head slice makes each item several times cheaper. Budget ~2 GPU-h. Written to
disk after every family, so a timeout degrades to fewer families rather than to
nothing.
"""
# @needs: run load_arm free_arm

import json
import random
import string
import time
import zlib

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MAX_R = 64
N_ITEMS = 16
VOCAB_CHANCE = 32768           # half of Huginn's 65536-token vocabulary
OUT = "/kaggle/working/battery.json"

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


def coda_head(model, h_state, freqs_cis, pos=None):
    """model's own ln_f -> coda -> ln_f -> lm_head on an intermediate state.

    TWO ln_f calls, not one: skipping the first feeds coda a state it was never
    built to accept, measured at 1.7-1.9 max abs logit error. D71 validated this
    exact tail as bit-identical to the model's own logits.

    `pos` slices AFTER coda and BEFORE lm_head. coda is attention, so it must see
    the whole sequence; lm_head is position-wise, so restricting it to the read
    positions is exact. The `SLICE CHECK` in `curve` measures that rather than
    assuming it.
    """
    import torch
    x = model.transformer.ln_f(h_state)
    block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
    for block in model.transformer.coda:
        block_idx -= 1
        x = block(x, freqs_cis, block_idx, None, None)
    x = model.transformer.ln_f(x)
    if pos is not None:
        x = x[:, pos[0]:pos[1]]
    return model.lm_head(x)


def curve(model, tok, prompt, gold, distractor, slice_check=False):
    """Per-unroll (rank of gold's first token, logp gold, logp distractor, top1).

    ONE forward at num_steps=MAX_R; the hook fires once per unroll, so this
    returns the whole depth curve for the cost of a single pass.
    """
    import torch
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids_p = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    g_ids = tok(gold, add_special_tokens=False).input_ids
    d_ids = tok(distractor, add_special_tokens=False).input_ids
    ids = torch.cat([ids_p, torch.tensor([g_ids], device=model.device,
                                         dtype=ids_p.dtype)], dim=1)
    n_p = ids_p.shape[1]
    lo, hi = n_p - 1, n_p - 1 + len(g_ids)          # every position actually read
    freqs = model.freqs_cis[:, : ids.shape[1]]
    rec = {"rank": [], "logp_gold": [], "logp_dist": [], "top1": []}
    checks = []
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()

    def hook(_m, _i, o):
        with torch.no_grad():
            lg = coda_head(model, o.detach(), freqs, pos=(lo, hi)).float()
            if slice_check and not checks:
                full = coda_head(model, o.detach(), freqs).float()[:, lo:hi]
                checks.append(float((full - lg).abs().max().item()))
            lp = torch.log_softmax(lg[0], dim=-1)
            row = lp[0]                                   # the gold-predicting position
            rec["rank"].append(int((row > row[g_ids[0]]).sum().item()) + 1)
            rec["top1"].append(int(row.argmax().item()))
            rec["logp_gold"].append(float(sum(lp[k, t].item()
                                              for k, t in enumerate(g_ids))))
            rec["logp_dist"].append(float(row[d_ids[0]].item()))

    h = mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids, num_steps=MAX_R)
    finally:
        h.remove()
    rec["n_gold_tok"] = len(g_ids)
    if slice_check:
        rec["slice_max_abs_delta"] = checks[0] if checks else None
    return rec


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)

    t0 = time.time()
    out = {"vocab": len(tok), "max_r": MAX_R, "n_items": N_ITEMS,
           "chance": VOCAB_CHANCE, "tasks": list(TASKS), "arms": {}}

    def save():
        with open(OUT, "w") as f:
            json.dump(out, f)

    for arm, spec in (("trained", MODEL_ID), ("untrained", None)):
        print(f"\n=== {arm} ===", flush=True)
        model = load_arm(spec, cfg, REVISION if spec else 0)
        out["arms"][arm] = {}
        first = True
        for task in TASKS:
            rows = []
            for prompt, gold, dist in items(task):
                r = curve(model, tok, prompt, gold, dist, slice_check=first)
                if first:
                    d = r.get("slice_max_abs_delta")
                    print(f"  SLICE CHECK (P1 gate): max|delta| = {d} "
                          f"-- must be exactly 0.0", flush=True)
                    if d is None or d != 0.0:
                        raise RuntimeError(
                            f"SLICE CHECK FAILED: max|delta| = {d}. The readout is no "
                            f"longer the one D71 validated; refusing to produce numbers.")
                    first = False
                # PERSIST THE PROMPT AND THE GOLD. geometry-correctness saved neither,
                # so D72's confound had to be established by regenerating the item set
                # and matching by index (D72(2)).
                r.update({"prompt": prompt, "gold": gold, "distractor": dist})
                rows.append(r)
            out["arms"][arm][task] = rows
            rk = np.array([min(x["rank"]) for x in rows], float)
            fin = np.array([x["rank"][-1] for x in rows], float)
            acc_best = float(np.mean([min(x["rank"]) == 1 for x in rows]))
            acc_64 = float(np.mean([x["rank"][-1] == 1 for x in rows]))
            print(f"  {task:>15}: best-depth median rank {np.median(rk):>8.0f}  "
                  f"acc@1(best r) {acc_best:>5.1%}  acc@1(r64) {acc_64:>5.1%}  "
                  f"median rank@r64 {np.median(fin):>8.0f}", flush=True)
            save()
        free_arm(model, spec)

    save()
    print(f"\n=== MEASURED in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    # Descriptive only. Every inferential statistic lives in the local analysis.
    print("  family            trained_acc@1(best r)   untrained_median_rank(r64)")
    for task in TASKS:
        tr = out["arms"].get("trained", {}).get(task)
        un = out["arms"].get("untrained", {}).get(task)
        if not tr:
            continue
        a = float(np.mean([min(x["rank"]) == 1 for x in tr]))
        u = float(np.median([x["rank"][-1] for x in un])) if un else float("nan")
        print(f"  {task:>15}   {a:>10.1%}            {u:>12.0f}", flush=True)
    print(f"saved {OUT}", flush=True)
    print("DONE", flush=True)


main()
