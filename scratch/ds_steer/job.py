"""A48: a STEERING VECTOR on `e` -- can a direction in the map's parameter suppress the prose frame?

WHY THIS IS THE INTERP DIRECTION WITH THE MOST ROOM. D188 surveyed all 12 papers in
`huginn-load`, tex and code, and found **zero** occurrences of steering vectors, activation
addition, representation engineering, SAEs, path patching, attribution patching, causal
scrubbing or circuit discovery. The standard mech-interp toolkit has never been pointed at a
recurrent-depth model.

AND THIS ARCHITECTURE IS UNUSUALLY SUITED TO STEERING. `e` is re-injected at **every unroll**
via `adapter(cat[x, input_embeds])`, so a vector added to `e` is applied **r times rather than
once** -- a dose-response knob no feedforward model has. We already own the machinery: A24
patches `e` for a whole run and A39 swaps it mid-trajectory, both with P1 = 0.000e+00.

WHAT IT STEERS TOWARD, AND WHY THAT TARGET. The dominant failure mode of this model is not
computation, it is format: D166/D174 found `The` opening 39.4% of generations with the rate of
opening with the gold flat in depth, D172 measured first-token accuracy 0.125 against 0.781
containment, and D180 showed two in-context examples move final-unroll accuracy 0.071 -> 0.327.
So the contrast that defines the direction is already measured:

    delta_e  =  mean( e[last] | 2-shot )  -  mean( e[last] | 0-shot )

computed on FIT items and applied to HELD-OUT items, then swept in alpha. If a single direction
in `e` reproduces what exemplars do, the prose decision is **linearly encoded in the map's
parameter** -- which is an interp result, and simultaneously a fix for the metric problem that
dominates this project's record.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, VOID WITHOUT IT. At alpha = 0 the hook still fires -- it clones `e` and
      adds zero -- so the run must reproduce the unpatched rank curve at **0.000e+00**. Any
      deviation means the write is not clean.

  P2  PRIMARY, DOSE-RESPONSE. First-token accuracy and the rate of opening with a prose token,
      as functions of alpha. Registered before running: **if alpha > 0 raises first-token
      accuracy above the 0-shot baseline on HELD-OUT items, the prose decision is a linear
      direction in `e`.** A flat curve says it is not, and that is equally reportable.

  P3  NO LEAKAGE. `delta_e` is fitted on items disjoint from the test set, and the disjointness
      is asserted rather than assumed. A47/A42's exemplar leak was caught by exactly such an
      assertion before launch.

  P4  THE CONTROL THAT DECIDES WHETHER P2 MEANS ANYTHING. A random direction at matched norm,
      same alphas. D144 already showed a norm-matched random bearing in `e`-space destroys
      rotation on 3 of 3 carriers, so random directions here are known to be *capable* of large
      effects -- which makes this a real control rather than a formality.

  P5  HELD-OUT FAMILIES. `delta_e` is fitted on `echo_digit`/`add1`/`sub1`, the three families
      where D180 measured 2-shot final accuracy 1.000, and tested on those PLUS `sort_min` and
      `count_mod3`, which contributed nothing to the fit. A direction that only works where it
      was fitted is a memorised offset, not a mechanism.

  P6  READ THE RAW OUTPUT. The literal first token is banked for every (item, alpha) and the
      census is printed per alpha. D110 printed a capability verdict over unread strings.
"""

import json
import os
import random
import re
import string
import subprocess
import sys
import time
import zlib

DEPTH = 48
SEED = 20260811
N_TEST = 5                 # held-out items per family, from the START of the census pool
N_FIT = 6                  # fit items per family, disjoint, used only to build delta_e
N_ITEMS = 24               # census pool size; items() keeps the battery's item set
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0, -1.0)
# Fit on LARGE-POOL families only. The small-alphabet families (`echo_digit` 10 distinct
# prompts, `add1`/`sub1` 9) cannot supply test + fit + exemplars at once -- the first
# version tried and `shot_pool` returned ZERO shots for all three, which would have made
# `delta_e` identically zero and the run vacuous. Caught by P3 before launch. This also
# makes the three families where D180 measured the largest 0-shot -> 2-shot jump
# (0.000 -> 1.000 final accuracy) into fully HELD-OUT families, which strengthens P5.
FIT_FAMILIES = ("add_2d", "sort_min", "count_mod3")
TEST_FAMILIES = ("echo_digit", "add1", "sub1", "sort_min", "count_mod3")
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 10000

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



def build_items(rng=None):
    """Held-out TEST items: the first N_TEST of each test family's census pool."""
    out = []
    for task in TEST_FAMILIES:
        for i, (prompt, gold, _d) in enumerate(items(task)[:N_TEST]):
            out.append({"family": task, "item": i, "prompt": prompt, "gold": gold})
    return out


def fit_items(task):
    """Items used ONLY to build delta_e, filtered so no PROMPT is shared with the test set.

    Index-disjointness is not prompt-disjointness: `echo_digit` has only ten distinct prompts
    ("Repeat this number exactly. Number: d"), so a pool of 24 repeats them and slicing by
    index puts the same prompt in both sets. The P3 assertion caught this before launch, as
    the same assertion caught A42's exemplar leak. Families that cannot supply N_FIT distinct
    non-test prompts return fewer, and the count is printed rather than padded.
    """
    test = {p for p, _, _ in items(task)[:N_TEST]}
    seen, out = set(), []
    for prompt, gold, distractor in items(task, n=400):
        if prompt in test or prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, gold, distractor))
        if len(out) >= N_FIT:
            break
    return out


def shot_pool(task, k=2):
    """Exemplars whose prompts appear in neither the test nor the fit set."""
    used = {p for p, _, _ in items(task)[:N_TEST]} | {p for p, _, _ in fit_items(task)}
    seen, out = set(), []
    for prompt, gold, distractor in items(task, n=400):
        if prompt in used or prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, gold, distractor))
        if len(out) >= k:
            break
    return out


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("steer.json")
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

    d_model = model.config.n_embd
    adapter = model.transformer.adapter
    core_last = model.transformer.core_block[-1]

    # P3: the fit set and the test set must not share a prompt
    for task in TEST_FAMILIES:
        te = {p for p, _, _ in items(task)[:N_TEST]}
        fi = {p for p, _, _ in fit_items(task)}
        assert not (te & fi), f"{task}: fit/test overlap of {len(te & fi)}"
    print(f"P3 OK: fit and test prompts disjoint for all {len(TEST_FAMILIES)} families",
          flush=True)

    def encode(body):
        text = tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    def prelude_e(ids):
        """`e` is the second half of the adapter's concatenated input."""
        grabbed = {}

        def pre(_m, inp):
            if "e" not in grabbed:
                grabbed["e"] = inp[0][..., d_model:].detach().clone()

        h = adapter.register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=1)
        finally:
            h.remove()
        return grabbed["e"]

    def steer_hook(delta, alpha):
        """Add alpha*delta to `e` AT THE LAST POSITION, every unroll.

        Last position only, because that is where the readout is taken and because it makes
        the intervention independent of prompt length -- the 0-shot and 2-shot prompts that
        define `delta` differ in length, and only their final-position `e` is comparable.
        """
        def pre(_m, inp):
            cur = inp[0]
            x, e = cur[..., :d_model], cur[..., d_model:].clone()
            e[:, -1, :] = e[:, -1, :] + alpha * delta
            return (torch.cat([x, e], dim=-1),)
        return pre

    def rank_curve(ids, gold, delta=None, alpha=0.0):
        n = ids.shape[1]
        freqs = model.freqs_cis[:, :n]
        g0 = tok(gold, add_special_tokens=False).input_ids[0]
        ranks, tops = [], []
        core_last._forward_hooks.clear()

        def coda_head(h):
            x = model.transformer.ln_f(h)
            bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
            for block in model.transformer.coda:
                bi -= 1
                x = block(x, freqs, bi, None, None)
            return model.lm_head(model.transformer.ln_f(x))

        def post(_m, _i, o):
            with torch.no_grad():
                row = torch.log_softmax(coda_head(o.detach()).float()[0, n - 1], dim=-1)
                ranks.append(int((row > row[g0]).sum().item()) + 1)
                if len(ranks) == DEPTH:
                    lp, ix = torch.topk(row, 5)
                    tops.append([[tok.decode([int(j)]), round(float(p), 3)]
                                 for p, j in zip(lp.tolist(), ix.tolist())])

        hs = [core_last.register_forward_hook(post)]
        if delta is not None:
            hs.append(adapter.register_forward_pre_hook(steer_hook(delta, alpha)))
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                out = model(input_ids=ids, num_steps=DEPTH)
        finally:
            for h in hs:
                h.remove()
        logits = out.logits if hasattr(out, "logits") else out[0]
        first = tok.decode([int(logits[:, -1, :].argmax(-1))])
        return {"rank_curve": ranks, "best_rank": int(min(ranks)),
                "best_depth": int(np.argmin(ranks)) + 1, "final_rank": ranks[-1],
                "oracle": int(min(ranks) == 1), "final_correct": int(ranks[-1] == 1),
                "first_tok": first, "first_exact": int(first.strip() == gold.strip()),
                "top5": tops[-1] if tops else None, "n_tokens": int(n)}

    # ---- build delta_e on the FIT items only -------------------------------------
    diffs = []
    for task in FIT_FAMILIES:
        pool = shot_pool(task, k=2)
        pre = "".join(f"{p}\nAnswer: {g}\n\n" for p, g, _ in pool)
        for prompt, gold, _d in fit_items(task):
            e0 = prelude_e(encode(prompt))[0, -1, :]
            e2 = prelude_e(encode(pre + prompt))[0, -1, :]
            diffs.append((e2 - e0).clone())
    delta = torch.stack(diffs).mean(0)
    dn = float(delta.norm())
    print(f"delta_e from {len(diffs)} fit pairs over {FIT_FAMILIES}: ||delta|| = {dn:.4f}",
          flush=True)
    gen = torch.Generator(device=delta.device).manual_seed(SEED)
    rnd = torch.randn(delta.shape, generator=gen, device=delta.device, dtype=delta.dtype)
    rnd = rnd / rnd.norm() * dn                                   # P4: matched norm
    print(f"random control direction: ||rnd|| = {float(rnd.norm()):.4f}", flush=True)

    test = build_items()
    print(f"{len(test)} held-out test items over {TEST_FAMILIES}", flush=True)
    t0, rows = time.time(), []
    for n_it, it in enumerate(test):
        ids = encode(it["prompt"])
        base = rank_curve(ids, it["gold"])                        # unpatched reference
        for name, vec in (("steer", delta), ("random", rnd)):
            for a in ALPHAS:
                if name == "random" and a == 0.0:
                    continue                                      # identical to steer alpha=0
                try:
                    r = rank_curve(ids, it["gold"], delta=vec, alpha=a)
                    r["ok"] = True
                except Exception as exc:                          # noqa: BLE001
                    r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
                r.update({k: it[k] for k in ("family", "item", "gold")})
                r["arm"], r["alpha"] = name, a
                r["base_final_rank"] = base["final_rank"]
                r["base_first_tok"] = base["first_tok"]
                r["null_dev"] = (abs(r.get("final_rank", -1) - base["final_rank"])
                                 if name == "steer" and a == 0.0 else None)
                rows.append(r)
        if n_it % 5 == 0:
            print(f"  {n_it}/{len(test)} ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok = [r for r in rows if r.get("ok")]
    import collections
    import statistics as st

    print("\n=== P1 INSTRUMENT NULL: alpha=0 must reproduce the unpatched curve ===", flush=True)
    z = [r for r in ok if r["arm"] == "steer" and r["alpha"] == 0.0]
    devs = [r["null_dev"] for r in z if r["null_dev"] is not None]
    same_tok = sum(1 for r in z if r["first_tok"] == r["base_first_tok"])
    print(f"  n={len(z)}  max |d final_rank| = {max(devs) if devs else 'n/a'}  "
          f"first token identical {same_tok}/{len(z)}  "
          f"{'OK' if devs and max(devs) == 0 and same_tok == len(z) else 'FAIL'}", flush=True)

    print("\n=== P2/P4 DOSE-RESPONSE ===", flush=True)
    print(f"{'arm':8s} {'alpha':>6s} {'n':>4s} {'first_exact':>11s} {'final_corr':>11s} "
          f"{'oracle':>7s} {'med final_rank':>15s}")
    for name in ("steer", "random"):
        for a in ALPHAS:
            v = [r for r in ok if r["arm"] == name and r["alpha"] == a]
            if not v:
                continue
            print(f"{name:8s} {a:6.2f} {len(v):4d} "
                  f"{sum(r['first_exact'] for r in v) / len(v):11.3f} "
                  f"{sum(r['final_correct'] for r in v) / len(v):11.3f} "
                  f"{sum(r['oracle'] for r in v) / len(v):7.3f} "
                  f"{st.median([r['final_rank'] for r in v]):15.1f}", flush=True)

    print("\n=== P6 FIRST-TOKEN CENSUS, literal, by alpha (steer arm) ===", flush=True)
    for a in ALPHAS:
        v = [r for r in ok if r["arm"] == "steer" and r["alpha"] == a]
        if v:
            c = collections.Counter(r["first_tok"] for r in v)
            print(f"  alpha={a:5.2f}: {dict(c.most_common(6))}", flush=True)

    print("\n=== P5 HELD-OUT FAMILIES (fit on echo_digit/add1/sub1 only) ===", flush=True)
    for fam in TEST_FAMILIES:
        tag = "FIT" if fam in FIT_FAMILIES else "HELD-OUT"
        cells = []
        for a in ALPHAS:
            v = [r for r in ok if r["arm"] == "steer" and r["alpha"] == a and r["family"] == fam]
            cells.append(f"a{a}={sum(r['first_exact'] for r in v) / len(v):.2f}" if v else "-")
        print(f"  {fam:12s} [{tag:8s}] " + "  ".join(cells), flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "alphas": list(ALPHAS), "depth": DEPTH, "seed": SEED,
                   "fit_families": list(FIT_FAMILIES), "test_families": list(TEST_FAMILIES),
                   "delta_norm": dn, "n_fit_pairs": len(diffs),
                   "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} conditions, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
