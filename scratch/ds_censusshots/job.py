"""A42: is this project's whole capability record an artefact of ZERO-SHOT prompting?

THE QUESTION D169 FORCES. A34 measured ARC-Easy letter-argmax at **0.416 zero-shot, 0.713 at
two shots, 0.723 at five**, against a published 0.699 -- paired, 35 items fixed against 4
broken, exact McNemar p = 3.35e-07. The gain is protocol-specific: over the same shots the
option-TEXT likelihood arms moved ~2 points while the letter arm moved **30.7**. A likelihood
score never asks the model to emit anything; the letter arm asks for a bare option letter at
the first generated position, and zero-shot Huginn opens with prose there (D166: `The` x12 of
32). **The answer was available and could not get out.**

**Every capability number in this project is zero-shot.** The observability census (D130), its
final-unroll rescoring (D157, D158), the displacement curve (D159), the ladder (D143), the
regime-behaviour task (D148) -- all of them. If zero-shot understates this model by thirty
points on ARC, the census's picture of what Huginn can and cannot do is in question, and with
it the screen D168 applies and the verdict that H2 is "untestable on this model" because every
axis is either shortcuttable or beyond Huginn.

THE ITEMS ARE THE CENSUS'S OWN, BY CONSTRUCTION. `items()` and `TASKS` below are copied
verbatim from `scratch/kaggle_census/body.py`, including its `zlib.crc32` seeding -- which
exists precisely because Python salts `hash()` per process (D69(3)). Item i of family f here is
**the same item** the census scored, so this is a paired comparison against banked numbers
rather than a fresh sample.

PREREGISTERED PREDICTIONS -- and D169's mechanism makes them sharply different:

  P1  REPLICATION GATE. The 0-shot arm must reproduce the census's banked per-family oracle
      accuracy (`min(rank) == 1` over 48 unrolls). Same items, same depth, same scoring. More
      than a few points off on the well-populated families and this run is measuring something
      else.

  P2  PRIMARY, AND THE TWO AXES ARE PREDICTED TO MOVE DIFFERENTLY. D169 says few-shot fixes
      EMISSION, not computation. So: **final-unroll accuracy should rise substantially** on the
      19 families that read exactly 0.000 (D158), while **oracle accuracy should barely move**,
      because availability was never the thing blocked. Registered before running.

  P3  THE OUTCOME THAT WOULD MATTER MORE. If ORACLE accuracy also rises substantially, then
      few-shot improves the computation itself, not just its expression -- and "the task is
      beyond the model" was never established for those families. That would reopen the
      capability screen D168 applies, and weaken the H2 verdict, which rests on every available
      difficulty axis being shortcuttable or out of reach.

  P4  WHAT WINS AT THE FINAL UNROLL, at every shot count. If D166's mechanism is right, the
      prose opener should be displaced BY THE ANSWER as shots increase. Literal strings, not a
      category count -- D110 printed a verdict over unread output.

  P5  EXEMPLARS NEVER LEAK. Shots are drawn from item indices at the END of each family's pool
      and test items from the START, with an assertion that the two sets are disjoint.
"""

import json
import os
import random
import string
import subprocess
import sys
import time
import zlib

DEPTH = 48                 # the census's depth, so the 0-shot arm is comparable
SHOTS = (0, 2, 5)
N_TEST = 8                 # test items per family, from the START of the pool
N_ITEMS = 24               # census pool size; items() keeps the battery's item set
TOPK = 5
SEED = 20260811
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 11000

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


def build_items(rng=None):
    """Test items: the FIRST N_TEST of each family's census pool (P5)."""
    out = []
    for task in TASKS:
        for i, (prompt, gold, _distractor) in enumerate(items(task)[:N_TEST]):
            out.append({"family": task, "item": i, "prompt": prompt, "gold": gold})
    return out


def shot_pool(task, k_max=max(SHOTS)):
    """Exemplars whose PROMPTS are disjoint from the test set -- filtered, not assumed.

    The first version of this took `items(task)[N_TEST:]` and asserted disjointness. The
    assertion fired on `echo_digit`, and it was right to: that family has only **ten**
    distinct prompts ("Repeat this number exactly. Number: d"), so a pool of 24 items
    repeats them, and an exemplar block would have shown the model the exact test item
    together with its answer. Few-shot accuracy would then measure memorisation of the
    prompt it is about to be asked. `add1`, `sub1`, `count4`, `succ_letter` and
    `caesar1_letter` all have small answer alphabets and the same exposure.

    So exemplars are drawn from a much larger pool, filtered against the test prompts and
    deduplicated. Families that cannot supply `k_max` distinct non-test prompts return
    fewer, and the caller records how many were actually used rather than assuming k.
    """
    test = {p for p, _, _ in items(task)[:N_TEST]}
    seen, out = set(), []
    for prompt, gold, distractor in items(task, n=400):
        if prompt in test or prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, gold, distractor))
        if len(out) >= k_max:
            break
    return out


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("censusshots.json")
    mnt = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"results -> {out_path}; weights mount -> {mnt}", flush=True)

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

    tok = AutoTokenizer.from_pretrained(mnt)
    cfg = AutoConfig.from_pretrained(mnt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mnt, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    core_last = model.transformer.core_block[-1]

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    # P5: exemplars must not overlap the test set, and short families must SAY they are short
    shot_avail = {}
    for task in TASKS:
        te = {p for p, _, _ in items(task)[:N_TEST]}
        sh = [p for p, _, _ in shot_pool(task)]
        assert not (te & set(sh)), f"{task}: exemplar/test overlap of {len(te & set(sh))}"
        shot_avail[task] = len(sh)
    short = {k: v for k, v in shot_avail.items() if v < max(SHOTS)}
    print(f"P5 OK: exemplar and test prompts disjoint for all {len(TASKS)} families",
          flush=True)
    if short:
        print(f"  SHORT POOLS (fewer than {max(SHOTS)} distinct non-test prompts "
              f"exist): {short} -- these families run at the k they can supply, and "
              f"`k_actual` is banked per row.", flush=True)

    def one(prompt, gold, k, task):
        pool = shot_pool(task)[:k]
        pre = "".join(f"{p}\nAnswer: {g}\n\n" for p, g, _ in pool)
        text = tok.apply_chat_template(
            [{"role": "user", "content": pre + prompt}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        g0 = tok(gold, add_special_tokens=False).input_ids[0]
        ranks, tops = [], []
        core_last._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                row = torch.log_softmax(
                    coda_head(o.detach(), freqs).float()[0, n_p - 1], dim=-1)
                ranks.append(int((row > row[g0]).sum().item()) + 1)
                if len(ranks) == DEPTH:
                    lp, ix = torch.topk(row, TOPK)
                    tops.append([[tok.decode([int(j)]), round(float(p), 3)]
                                 for p, j in zip(lp.tolist(), ix.tolist())])

        h = core_last.register_forward_hook(hook)
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=DEPTH)
        finally:
            h.remove()
        return {"k_actual": len(pool), "rank_curve": ranks, "best_rank": int(min(ranks)),
                "best_depth": int(np.argmin(ranks)) + 1, "final_rank": int(ranks[-1]),
                "oracle": int(min(ranks) == 1), "final_correct": int(ranks[-1] == 1),
                "final_top5": tops[-1] if tops else None, "n_tokens": int(n_p)}

    test = build_items()
    print(f"{len(test)} test items over {len(TASKS)} families", flush=True)
    t0, rows = time.time(), []
    for k in SHOTS:
        for n, it in enumerate(test):
            try:
                r = one(it["prompt"], it["gold"], k, it["family"])
                r["ok"] = True
            except Exception as exc:  # noqa: BLE001
                r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
            r.update({kk: it[kk] for kk in ("family", "item", "gold")})
            r["shots"] = k
            rows.append(r)
            if n % 40 == 0:
                print(f"  k={k} {n}/{len(test)} ({time.time() - t0:.0f}s)", flush=True)
                with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                    json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok = [r for r in rows if r.get("ok")]
    print("\n=== P1/P2/P3 ORACLE vs FINAL, by family and shots ===", flush=True)
    print(f"{'family':15s} " + " ".join(f"{'k=' + str(k):>17s}" for k in SHOTS))
    print(f"{'':15s} " + " ".join(f"{'oracle/final':>17s}" for _ in SHOTS))
    tot = {k: [0, 0, 0] for k in SHOTS}
    for fam in TASKS:
        cells = []
        for k in SHOTS:
            v = [r for r in ok if r["family"] == fam and r["shots"] == k]
            if not v:
                cells.append(f"{'-':>17s}"); continue
            o = sum(r["oracle"] for r in v) / len(v)
            f_ = sum(r["final_correct"] for r in v) / len(v)
            tot[k][0] += sum(r["oracle"] for r in v)
            tot[k][1] += sum(r["final_correct"] for r in v)
            tot[k][2] += len(v)
            cells.append(f"{o:8.3f}/{f_:<8.3f}")
        print(f"{fam:15s} " + " ".join(cells), flush=True)
    print(f"\n{'POOLED':15s} " + " ".join(
        f"{tot[k][0] / max(1, tot[k][2]):8.3f}/{tot[k][1] / max(1, tot[k][2]):<8.3f}"
        for k in SHOTS), flush=True)

    print("\n=== P2/P3 VERDICT ===", flush=True)
    if all(tot[k][2] for k in SHOTS):
        o0, f0 = tot[0][0] / tot[0][2], tot[0][1] / tot[0][2]
        oK, fK = tot[SHOTS[-1]][0] / tot[SHOTS[-1]][2], tot[SHOTS[-1]][1] / tot[SHOTS[-1]][2]
        print(f"  oracle {o0:.3f} -> {oK:.3f}  (d {oK - o0:+.3f})", flush=True)
        print(f"  final  {f0:.3f} -> {fK:.3f}  (d {fK - f0:+.3f})", flush=True)
        if fK - f0 > 0.15 and oK - o0 < 0.10:
            print("  -> P2: few-shot fixes EMISSION, not computation. D169's mechanism "
                  "generalises from ARC to the census.", flush=True)
        elif oK - o0 >= 0.10:
            print("  -> P3: ORACLE moves too. Few-shot improves the computation, so 'the "
                  "task is beyond the model' was not established for these families, and "
                  "the H2 verdict needs revisiting.", flush=True)
        else:
            print("  -> NEITHER axis moves: the census families do not behave like ARC, "
                  "and D169's gain does not generalise.", flush=True)

    print("\n=== P4 WHAT WINS AT THE FINAL UNROLL (literal) ===", flush=True)
    for fam in ("echo_digit", "add1", "sort_min", "count_mod3", "compare"):
        for k in SHOTS:
            v = [r for r in ok if r["family"] == fam and r["shots"] == k and r["final_top5"]]
            if v:
                print(f"  {fam:12s} k={k}: gold={v[0]['gold']!r} "
                      f"top5={v[0]['final_top5']}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "shots": list(SHOTS), "depth": DEPTH, "seed": SEED,
                   "n_test": N_TEST, "tasks": list(TASKS),
                   "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} forwards, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
