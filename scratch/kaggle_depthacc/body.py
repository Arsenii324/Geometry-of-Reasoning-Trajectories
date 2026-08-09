"""Does the model SAY the answer it demonstrably knows -- and does depth help or hurt?

THE GAP THIS CLOSES IS ONE THIS PROJECT NAMED ITSELF. D68(5) states it outright:
"WHAT IS NOT ESTABLISHED. That generation accuracy would be higher at r=4 -- this
measures teacher-forced RANK at the FIRST answer position, not a decoded string, and
a model that says 'The number is 7' is not ignorant." Everything downstream of D68
and D69 -- that the model knows answers it never emits, that depth converges to a
discourse choice, that one instruction moves echo_digit from 0% to 83% -- rests on
rank. Rank is not capability, and the difference is exactly what a reader will ask
about first.

THE DESIGN. Greedy decoding at several depths on the SAME 21 families D75 measured,
in two prompt formats, with the teacher-forced rank recorded from the SAME run so
the two instruments are compared within a run rather than across runs (D78: h_0 is
unseeded, so cross-run labels carry ~8 accuracy points of noise).

WHY BOTH FORMATS. D69 established that "Reply with only the answer" moves echo_digit
from 0% to 83% at r=64 while the bare prompt reads 0%. If the depth profile of
GENERATION differs between formats, then "depth hurts" is a statement about the bare
prompt's discourse convergence rather than about the computation -- which is the
distinction D68(3) drew and could not test.

PRE-REGISTERED (CLAUDE.md section 1).
P1 -- GATE ON THE INSTRUMENT. `assert_generation_works` must pass before anything is
      read. D62 printed a capability verdict over empty strings because no gate
      tested whether the decoder produced output at all; that failure is not
      repeated by inspection alone.
P2 -- THE POINT AT ISSUE. On the BARE prompt, does exact-match accuracy FALL with
      depth? D68 measured rank-1 accuracy peaking at r=4 and reaching 0% by r=8 on
      echo_digit. Two-sided over 21 families: a rise would refute the depth-hurts
      reading as squarely as a fall confirms it.
P3 -- RANK AGAINST GENERATION, per item and per depth. If gold rank 1 does not
      predict a correct decoded string, then every rank-based claim in D68/D69/D75
      is about a quantity that does not reach the output, and this run says so.
P4 -- THE FORMAT INTERACTION. Under "Reply with only the answer", the depth profile
      is predicted to flatten or invert relative to bare. If it does not, the
      discourse explanation is incomplete.
P5 -- CONTAINMENT, not just exact match. A model that answers "The number is 7" is
      scored wrong by exact match and right by containment; both are recorded,
      because the gap between them IS the discourse effect in units of accuracy.

Computes accuracy but banks every decoded string, so scoring can be redone offline.
"""
# @needs: run load_arm free_arm batched_generate assert_generation_works preflight

import json
import os
import random
import string
import time
import traceback
import zlib

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
N_ITEMS = 6             # a prefix of the battery's 24; `items` is prefix-stable
MAX_NEW = 8
# ORDERED BY DECISIVENESS, NOT BY SIZE. Kaggle kills a kernel at its wall clock
# with whatever it has written; every cell is persisted as it completes, so the
# order decides what survives a timeout. D68 puts the contrast at r=4 (accuracy
# peak) against r=32 (this project's default everywhere else), so those two run
# first and the interpolating depths fill in afterwards.
DEPTHS = (4, 32, 8, 16, 2)
OUTDIR = "/kaggle/working"

# D69's instruction, verbatim: it is the arm that moved echo_digit 0% -> 83%.
CONSTRAINT = "Reply with only the answer."
WORDS = ('apple', 'chair', 'river', 'stone', 'bread', 'cloud', 'green', 'horse', 'light', 'money', 'night', 'paper', 'queen', 'table', 'water', 'youth')


def items(task, n=N_ITEMS):
    """(prompt, gold, distractor) triples. Gold is single-token where possible.

    `zlib.crc32`, NOT `hash()`: Python salts str hashes per process, so a kernel
    seeded with `hash(task)` draws a different item set on every run (D69(3)).
    """
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + zlib.crc32(task.encode()) % 997)
        L = string.ascii_lowercase
        if task == 'echo_digit':
            v = rng.randint(0, 9)
            out.append((f'Repeat this number exactly.\nNumber: {v}', str(v), str((v + 3) % 10)))
        elif task == 'echo_word':
            w = rng.choice(WORDS)
            out.append((f'Repeat this word exactly.\nWord: {w}', w, rng.choice([x for x in WORDS if x != w])))
        elif task == 'nth_item':
            xs = [rng.randint(0, 9) for _ in range(5)]
            k = rng.randint(1, 5)
            out.append((f"What is item number {k} in this list?\nList: {' '.join(map(str, xs))}", str(xs[k - 1]), str((xs[k - 1] + 3) % 10)))
        elif task == 'last_item':
            xs = [rng.randint(0, 9) for _ in range(6)]
            out.append((f"What is the last number in this list?\nList: {' '.join(map(str, xs))}", str(xs[-1]), str((xs[-1] + 3) % 10)))
        elif task == 'add1':
            v = rng.randint(0, 8)
            out.append((f'What is {v} + 1?', str(v + 1), str((v + 4) % 10)))
        elif task == 'sub1':
            v = rng.randint(1, 9)
            out.append((f'What is {v} - 1?', str(v - 1), str((v + 4) % 10)))
        elif task == 'add_2d':
            a, b = (rng.randint(10, 49), rng.randint(10, 49))
            out.append((f'What is {a} + {b}?', str(a + b), str(a + b + 3)))
        elif task == 'compare':
            a, b = rng.sample(range(1, 100), 2)
            out.append((f'Which number is larger, {a} or {b}?', str(max(a, b)), str(min(a, b))))
        elif task == 'count4':
            b = [rng.randint(0, 1) for _ in range(4)]
            g = sum(b)
            out.append((f"Count how many ones are in this sequence.\nSequence: {' '.join(map(str, b))}", str(g), str((g + 2) % 5)))
        elif task == 'count8':
            b = [rng.randint(0, 1) for _ in range(8)]
            g = sum(b)
            out.append((f"Count how many ones are in this sequence.\nSequence: {' '.join(map(str, b))}", str(g), str((g + 3) % 9)))
        elif task == 'count16':
            b = [rng.randint(0, 1) for _ in range(16)]
            g = sum(b)
            out.append((f"Count how many ones are in this sequence.\nSequence: {' '.join(map(str, b))}", str(g), str((g + 3) % 17)))
        elif task == 'count_mod3':
            b = [rng.randint(0, 1) for _ in range(9)]
            g = sum(b) % 3
            out.append((f"Count how many ones are in this sequence, then give the remainder when divided by 3.\nSequence: {' '.join(map(str, b))}", str(g), str((g + 1) % 3)))
        elif task == 'parity8':
            b = [rng.randint(0, 1) for _ in range(8)]
            g = sum(b) % 2
            out.append((f"Is the number of ones in this sequence even or odd? Answer 0 for even and 1 for odd.\nSequence: {' '.join(map(str, b))}", str(g), str(1 - g)))
        elif task == 'track_total':
            ops = [rng.choice([1, -1]) for _ in range(8)]
            g = sum(ops)
            body = 'Start at 0. ' + ' '.join(('Add 1.' if o > 0 else 'Subtract 1.' for o in ops))
            out.append((body + ' Final total?', str(g), str(g + 2)))
        elif task == 'local_last':
            ops = [rng.choice([1, -1]) for _ in range(8)]
            g = 'Add' if ops[-1] > 0 else 'Subtract'
            body = 'Start at 0. ' + ' '.join(('Add 1.' if o > 0 else 'Subtract 1.' for o in ops))
            out.append((body + ' What was the last instruction?', g, 'Subtract' if g == 'Add' else 'Add'))
        elif task == 'max_run':
            b = [rng.randint(0, 1) for _ in range(12)]
            best = cur = 1
            for i in range(1, len(b)):
                cur = cur + 1 if b[i] == b[i - 1] else 1
                best = max(best, cur)
            out.append((f"What is the length of the longest run of identical symbols in this sequence?\nSequence: {' '.join(map(str, b))}", str(best), str(best + 1)))
        elif task == 'sort_min':
            xs = rng.sample(range(1, 100), 3)
            out.append((f'What is the smallest of these numbers: {xs[0]}, {xs[1]}, {xs[2]}?', str(min(xs)), str(sorted(xs)[1])))
        elif task == 'succ_letter':
            c = rng.choice(L[:25])
            out.append((f"What letter comes after '{c}' in the alphabet?", L[L.index(c) + 1], L[(L.index(c) + 5) % 26]))
        elif task == 'caesar1_letter':
            c = rng.choice(L[:25])
            out.append((f'Shift this letter forward by 1 in the alphabet.\nLetter: {c}', L[L.index(c) + 1], L[(L.index(c) + 7) % 26]))
        elif task == 'caesar1_word':
            w = ''.join((rng.choice(L) for _ in range(4)))
            enc = ''.join((L[(L.index(c) + 1) % 26] for c in w))
            out.append((f'Shift each letter of this text backward by 1 in the alphabet.\nText: {enc}', w, ''.join((rng.choice(L) for _ in range(4)))))
        else:
            w = ''.join((rng.choice(L) for _ in range(4)))
            enc = ''.join((L[(L.index(c) + 13) % 26] for c in w))
            out.append((f'Decode this ROT13 text.\nText: {enc}', w, ''.join((rng.choice(L) for _ in range(4)))))
    return out


TASKS = ('echo_digit', 'echo_word', 'nth_item', 'last_item', 'add1', 'sub1', 'add_2d', 'compare', 'count4', 'count8', 'count16', 'count_mod3', 'parity8', 'track_total', 'local_last', 'max_run', 'sort_min', 'succ_letter', 'caesar1_letter', 'caesar1_word', 'rot13_word')

def normalise(text):
    """Lowercase, strip punctuation and articles -- the scorer, written once.

    Exact match on a raw string would score "7." wrong against "7", which measures
    punctuation rather than capability. Kept deliberately simple and applied to BOTH
    prediction and gold, so it cannot favour one side.
    """
    t = text.strip().lower()
    for ch in ".,!?;:'\"()[]":
        t = t.replace(ch, " ")
    # Articles and copulas are discourse, not content, and dropping them is
    # standard QA normalisation. It does make exact match slightly lenient -- "The
    # 12" scores exact against "12" while "The number is 12" does not -- which is
    # deliberate: what `contains` is meant to catch is SUBSTANTIVE prose wrapped
    # around the answer, not a leading article.
    words = [w for w in t.split() if w not in ("the", "a", "an", "is", "are")]
    return " ".join(words)


def score(pred, gold):
    """(exact, contains) after normalisation.

    Both, because the gap between them IS the discourse effect measured in units of
    accuracy: "the number is 7" fails exact and passes contains, and D68(3) says
    that is the dominant failure mode on the bare prompt.
    """
    p, g = normalise(pred), normalise(gold)
    return (p == g), (g in p.split() if " " not in g else g in p)


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


def gold_rank_at(model, tok, torch, text, gold, depth):
    """Teacher-forced rank of gold's first token, from the SAME weights and depth.

    Recorded beside the decoded string so P3 compares the two instruments WITHIN a
    run. Across runs they would not be comparable: h_0 is drawn unseeded (D78), so
    two forwards of one prompt differ, and a cross-run rank/generation correlation
    would be attenuated by that alone.
    """
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device)
    g_ids = tok(gold, add_special_tokens=False).input_ids
    n_p = ids.shape[1]
    with torch.no_grad():
        out = model(input_ids=ids, num_steps=depth)
        row = torch.log_softmax(out.logits[0, n_p - 1].float(), dim=-1)
    return int((row > row[g_ids[0]]).sum().item()) + 1


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoConfig, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
          flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION,
                                     trust_remote_code=True)
    model = load_arm(MODEL_ID, cfg, REVISION)

    # P1: the decoder must be shown to work before any number below is read. D62
    # printed a capability verdict over empty strings for want of exactly this.
    assert_generation_works(model, tok, chat=True)
    print("P1: generation gate PASSED", flush=True)

    def build(prompt, fmt):
        body = prompt if fmt == "bare" else prompt + "\n" + CONSTRAINT
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)

    cells = []
    for fam in TASKS:
        for i, (prompt, gold, dist) in enumerate(items(fam)):
            for fmt in ("bare", "constrained"):
                cells.append({"family": fam, "item": i, "fmt": fmt, "gold": gold,
                              "distractor": dist, "prompt": prompt,
                              "text": build(prompt, fmt)})

    records, ranks = [], []
    t0 = time.time()
    for depth in DEPTHS:
        # Ranks BEFORE generation at each depth, and both persisted per depth: a
        # timeout then costs whole depths rather than the entire rank arm, which is
        # what P3 needs and which is far cheaper to compute than the generations.
        for c in cells:
            try:
                r, ok = gold_rank_at(model, tok, torch, c["text"], c["gold"], depth), True
            except Exception as exc:
                r, ok = -1, False
                if len(ranks) < 3:
                    print(f"  rank failed: {type(exc).__name__}: {exc}", flush=True)
            ranks.append({"family": c["family"], "item": c["item"], "fmt": c["fmt"],
                          "depth": depth, "rank": r, "ok": ok})
        with open(os.path.join(OUTDIR, "depthrank.json"), "w") as fh:
            json.dump(ranks, fh)
        at_d = [x for x in ranks if x["depth"] == depth and x["ok"]]
        top1 = sum(x["rank"] == 1 for x in at_d) / max(1, len(at_d))
        print(f"  depth {depth:>3}: gold is top-1 in {top1:.1%} of {len(at_d)} cells "
              f"(teacher-forced)", flush=True)

        for fmt in ("bare", "constrained"):
            sub = [c for c in cells if c["fmt"] == fmt]
            try:
                outs = batched_generate(model, tok, [c["text"] for c in sub],
                                        max_new=MAX_NEW, num_steps=depth,
                                        verbose=False)
            except Exception as exc:
                print(f"  depth {depth} {fmt}: FAILED {type(exc).__name__}: {exc}",
                      flush=True)
                print(traceback.format_exc(), flush=True)
                continue
            deg = degenerate(outs)
            for c, o in zip(sub, outs):
                ex, ct = score(o, c["gold"])
                records.append({**{k: c[k] for k in
                                   ("family", "item", "fmt", "gold", "prompt")},
                                "depth": depth, "output": o, "exact": bool(ex),
                                "contains": bool(ct)})
            with open(os.path.join(OUTDIR, "depthacc.json"), "w") as fh:
                json.dump(records, fh)
            here = [r for r in records if r["depth"] == depth and r["fmt"] == fmt]
            n = len(here)
            print(f"  depth {depth:>3} {fmt:>12}: exact "
                  f"{sum(r['exact'] for r in here) / max(1, n):.1%}, contains "
                  f"{sum(r['contains'] for r in here) / max(1, n):.1%}, "
                  f"{n} items, degenerate={deg}", flush=True)

    free_arm(model, MODEL_ID)
    print(f"\n=== {len(records)} generations + {len(ranks)} ranks in "
          f"{(time.time() - t0) / 60:.1f} min ===", flush=True)
    print("scoring is redone offline from the banked strings; see "
          "scripts/run_depth_accuracy.py", flush=True)
    print("DONE", flush=True)


main()
