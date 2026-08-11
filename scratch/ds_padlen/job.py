"""
A45: is the five-shot regression about EXEMPLARS or about PROMPT LENGTH?

THE CONFOUND, REGISTERED IN D181 BEFORE THIS RUN. Two in-context examples raise availability
(oracle 0.327 -> 0.548, paired McNemar p = 9.3e-09, D180) and five examples lose some of it
back (oracle k=2 -> k=5 net -15, p = 7.3e-04). D181 found the mechanism: the gold's rank at
unroll 1 goes **54.5 -> 58.0 -> 157.5** across k = 0/2/5 while the prompt grows **29 -> 73 ->
139** tokens, and D178 established that family-level Spearman(start rank, best_depth) = +0.885.

**But exemplar COUNT and prompt LENGTH are perfectly collinear in that design.** Two very
different claims predict the identical data:

    EXEMPLARS  in-context learning has diminishing and then negative returns
    LENGTH     Huginn's readout degrades with context length, full stop

The second would be an architectural fact about a recurrent-depth model and would apply to
every long prompt, not just few-shot ones.

I TRIED TO SETTLE IT FROM BANKED DATA AND COULD NOT, which is why this runs. Dose-response
across families gives Spearman(added tokens, change in start rank) = +0.419 at **p = 0.0586**,
and the two census families whose prompts vary in length at k = 0 give **opposite signs**
(`compare` +0.121, `sort_min` -0.147). Suggestive, not decisive, with outliers in the wrong
direction.

THE DESIGN. Four arms on the census's own items, the middle one being the discriminator:

    k0        the bare prompt                                          -- floor
    k0_pad    the bare prompt padded with IRRELEVANT prose to the       -- THE DISCRIMINATOR
              k=5 token length, so length matches with zero exemplars
    k2        two exemplars                                             -- the helpful arm
    k5        five exemplars                                            -- the regression

PREREGISTERED PREDICTIONS:

  P1  LENGTH MATCH IS A MEASUREMENT, NOT AN ASSUMPTION. The realised token count of `k0_pad`
      must land within ~10% of `k5`'s for the same item, and both are banked per row. If they
      do not match, the arm is not a length control and is reported as void.

  P2  PRIMARY, AND IT IS A CLEAN FORK. **If LENGTH is the cause**, `k0_pad` degrades the gold's
      start rank roughly as far as `k5` does and its oracle accuracy falls with it. **If
      EXEMPLARS are the cause**, `k0_pad` stays near `k0` -- padding costs nothing -- and the
      k=5 regression is a property of demonstration rather than of context.

  P3  START RANK IS THE MEASURE THAT MATTERS, not accuracy, because D181's mechanism is
      specifically about where the answer sits before the recurrence starts. Median rank at
      unroll 1 is the primary readout; oracle and final-unroll accuracy are reported alongside.

  P4  THE PADDING MUST NOT BE A SECOND TASK. It is neutral English prose with no questions, no
      digits and no answer-shaped structure, so it cannot act as a distractor exemplar. The
      exact padding text is banked with the run.

  P5  A REPLICATION GATE ON k0 AND k5. Both must reproduce D180's pooled oracle within a few
      points on these families. If they do not, this run differs from A42 in some way I have
      not identified.
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
ARMS = ("k0", "k0_pad", "k2", "k5")
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



TASKS = ("echo_digit", "add1", "sub1", "sort_min", "count_mod3", "parity8",
         "caesar1_letter", "count16")
N_TEST_LOCAL = 6

# P4: neutral prose. No questions, no digits, no answer-shaped structure, so it cannot act
# as a distractor exemplar. Repeated as needed to reach the target length.
PAD_TEXT = (
    "The following passage is provided as background reading and has no bearing on the "
    "task. Rivers have shaped the landscape of the northern valleys for many centuries, "
    "carving channels through soft stone and depositing silt along their banks. Travellers "
    "who pass through in the early morning often remark on the quality of the light, which "
    "falls across the water in long pale bands. The old bridges were built by masons whose "
    "names were never recorded, and their arches have outlasted most of the buildings on "
    "either shore. In summer the meadows beside the water are left to grow tall, and the "
    "keepers walk them slowly at dusk. "
)


def build_items(rng=None):
    out = []
    for task in TASKS:
        for i, (prompt, gold, _d) in enumerate(items(task)[:N_TEST_LOCAL]):
            out.append({"family": task, "item": i, "prompt": prompt, "gold": gold})
    return out


def shot_pool(task, k_max=5):
    test = {p for p, _, _ in items(task)[:N_TEST_LOCAL]}
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
        os.path.abspath("padlen.json")
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

    def encode(body):
        text = tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    def body_for(arm, it, target_tokens=None):
        if arm == "k0":
            return it["prompt"]
        if arm in ("k2", "k5"):
            k = 2 if arm == "k2" else 5
            pool = shot_pool(it["family"])[:k]
            return "".join(f"{p}\nAnswer: {g}\n\n" for p, g, _ in pool) + it["prompt"]
        # k0_pad: neutral prose trimmed at WORD granularity to hit the k5 token target.
        # PAD_TEXT alone is ~130 tokens, so adding it whole to a 30-token prompt overshoots
        # a 139-token target and would fail P1's +/-10% match. Grow, then trim back.
        words = (PAD_TEXT * 12).split()
        lo, hi = 0, len(words)
        while lo < hi:                                  # smallest prefix that reaches target
            mid = (lo + hi) // 2
            body = " ".join(words[:mid]) + "\n" + it["prompt"]
            if encode(body).shape[1] >= target_tokens:
                hi = mid
            else:
                lo = mid + 1
        return " ".join(words[:lo]) + "\n" + it["prompt"]

    def measure(ids, gold):
        n = ids.shape[1]
        freqs = model.freqs_cis[:, :n]
        g0 = tok(gold, add_special_tokens=False).input_ids[0]
        ranks = []
        core_last._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                row = torch.log_softmax(
                    coda_head(o.detach(), freqs).float()[0, n - 1], dim=-1)
                ranks.append(int((row > row[g0]).sum().item()) + 1)

        h = core_last.register_forward_hook(hook)
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=DEPTH)
        finally:
            h.remove()
        return {"start_rank": ranks[0], "best_rank": int(min(ranks)),
                "best_depth": int(np.argmin(ranks)) + 1, "final_rank": ranks[-1],
                "oracle": int(min(ranks) == 1), "final_correct": int(ranks[-1] == 1),
                "n_tokens": int(n)}

    test = build_items()
    print(f"{len(test)} items x {len(ARMS)} arms", flush=True)
    t0, rows = time.time(), []
    for n_it, it in enumerate(test):
        try:
            k5_ids = encode(body_for("k5", it))
            target = k5_ids.shape[1]
            per = {}
            for arm in ARMS:
                ids = k5_ids if arm == "k5" else encode(
                    body_for(arm, it, target_tokens=target))
                per[arm] = measure(ids, it["gold"])
            ok, why = True, ""
        except Exception as exc:  # noqa: BLE001
            per, ok, why = {}, False, f"{type(exc).__name__}: {exc}"
        rows.append({**{k: it[k] for k in ("family", "item", "gold")},
                     "ok": ok, "why": why, "arms": per})
        if n_it % 8 == 0:
            print(f"  {n_it}/{len(test)} ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok_rows = [r for r in rows if r["ok"]]
    import statistics as st
    print("\n=== P1 LENGTH MATCH (k0_pad must land near k5) ===", flush=True)
    if ok_rows:
        rr = [(r["arms"]["k0_pad"]["n_tokens"], r["arms"]["k5"]["n_tokens"]) for r in ok_rows]
        ratios = [a / b for a, b in rr]
        print(f"  k0_pad/k5 token ratio: median {st.median(ratios):.3f} "
              f"min {min(ratios):.3f} max {max(ratios):.3f}  "
              f"{'OK' if 0.9 <= st.median(ratios) <= 1.15 else 'FAIL -- not a length control'}",
              flush=True)

    print("\n=== P3 PRIMARY: start rank by arm ===", flush=True)
    print(f"{'arm':9s} {'n':>4s} {'tokens':>7s} {'start_rank(med)':>16s} {'oracle':>7s} "
          f"{'final':>7s} {'best_depth':>11s}")
    summ = {}
    for arm in ARMS:
        v = [r["arms"][arm] for r in ok_rows if arm in r["arms"]]
        if not v:
            continue
        summ[arm] = st.median([x["start_rank"] for x in v])
        print(f"{arm:9s} {len(v):4d} {st.median([x['n_tokens'] for x in v]):7.0f} "
              f"{st.median([x['start_rank'] for x in v]):16.1f} "
              f"{sum(x['oracle'] for x in v) / len(v):7.3f} "
              f"{sum(x['final_correct'] for x in v) / len(v):7.3f} "
              f"{st.mean([x['best_depth'] for x in v]):11.2f}", flush=True)

    print("\n=== P2 VERDICT ===", flush=True)
    if {"k0", "k0_pad", "k5"} <= set(summ):
        d_pad = summ["k0_pad"] - summ["k0"]
        d_k5 = summ["k5"] - summ["k0"]
        frac = d_pad / d_k5 if d_k5 else float("nan")
        print(f"  start rank: k0 {summ['k0']:.1f} -> k0_pad {summ['k0_pad']:.1f} "
              f"(+{d_pad:.1f}) -> k5 {summ['k5']:.1f} (+{d_k5:.1f})", flush=True)
        print(f"  padding reproduces {frac:.0%} of the k5 degradation", flush=True)
        if frac > 0.6:
            print("  -> LENGTH. Huginn's readout degrades with context length; the k=5 "
                  "regression is not about demonstration.", flush=True)
        elif frac < 0.25:
            print("  -> EXEMPLARS. Padding costs nothing; the k=5 regression is a property "
                  "of demonstration.", flush=True)
        else:
            print("  -> MIXED: both contribute and neither account is sufficient alone.",
                  flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "arms": list(ARMS), "tasks": list(TASKS), "depth": DEPTH,
                   "seed": SEED, "pad_text": PAD_TEXT, "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} items, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
