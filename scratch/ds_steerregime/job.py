"""A49: does the PROSE-SUPPRESSION direction in `e` also move the ROTATION REGIME?

WHY THIS IS ADDITIVE TO A48 AND NOT PART OF IT. A48 steers along
`delta_e = mean(e | 2-shot) - mean(e | 0-shot)` on census prompts, where the readout is the
formatting one. But `e` is the SAME object the regime line manipulates: D140 interpolates `e`
between nouns to flip settling<->rotating, D161 does it from the `wte` row, D173 swaps `e`
mid-trajectory with 62/64 switches taking at P1 = 0.000e+00. So `e`-space already contains one
known, causally-verified, behaviourally-measurable direction. **A48 cannot test the relation,
because its base prompts are census items and the regime is defined on D132's template.** This
runs the same steering vector on prompts that sit on both sides of the regime boundary.

THE FORK, AND BOTH BRANCHES ARE WORTH HAVING:

  SEPARATE AXES -- if R stays on whichever side of D141's threshold (0.6677) the base prompt
    started, then formatting and regime are **independent directions in `e`-space**. That is a
    clean dissociation and it bounds what D161's "the embedding controls the regime" means: it
    controls the regime, not everything.

  ONE PHENOMENON -- if R crosses the threshold as alpha grows, then the prose frame and the
    period-6 rotation are **the same thing seen two ways**, which ties D166/D174's format
    findings directly to the geometry line and is a much larger claim.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, VOID WITHOUT IT. At alpha = 0 the hook fires and adds zero, so R and the
      rank curve must reproduce the unpatched run at **0.000e+00**, as A25/A32/A39/A41 achieved.

  P2  BOTH SIDES OF THE BOUNDARY, WHICH IS THE POINT. Base prompts are D134's rotating set
      (`symbol`, `symptom`, `cymbal`) and settling set (`element`, `token`, `digit`) in D132's
      template. **Steering only from settling prompts cannot detect a crossing in one
      direction**, so both are run and reported separately.

  P3  PRIMARY. `rotation_power` at the last position, per alpha, against threshold 0.6677 --
      imported from `traj_geom.metrics.dynamics`, never reimplemented, because D137's
      instrument null rests on the kernel-side and analysis-side values being the same
      function.

  P4  THE CONTROL WITH TEETH. A norm-matched random direction, measured on R as well as on the
      readout. **D144 found a norm-matched random bearing in `e`-space destroys rotation on 3
      of 3 carriers** -- so without this arm, "steering changed R" is indistinguishable from
      "any large `e` perturbation changes R". This is the arm that decides whether P3 means
      anything.

  P5  THE FORMAT READOUT IS TRACKED TOO, so the two axes can be compared on the same runs: the
      gold's rank curve and the literal first token at every alpha.
"""

import json
import os
import subprocess
import sys
import time
import zlib
import random
import string

DEPTH = 64                 # the regime banks' depth, so R is comparable to D132/D141
SEED = 20260811
THRESHOLD = 0.6677         # D141
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0, -1.0)
ROT_NOUNS = ("symbol", "symptom", "cymbal")     # D134's rotating set
SET_NOUNS = ("element", "token", "digit")       # D134's settling set
SEQS = ("1 6 0 1 7 7 8 1", "3 5 5 3 6 1 5 2")
MARKER = "A"
N_FIT = 6
N_ITEMS = 24
FIT_FAMILIES = ("add_2d", "sort_min", "count_mod3")   # large pools; A48's construction
WORDS = ("apple", "chair", "river", "stone", "bread", "cloud", "green", "horse",
         "light", "money", "night", "paper", "queen", "table", "water", "youth")
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 9000

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



def prompt_for(word, seq, mk=MARKER):
    """D132's template verbatim -- the one the whole regime line is defined on."""
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def gold_for(seq, mk=MARKER):
    d = [int(x) for x in seq.split()]
    return str(max(d) if mk == "A" else min(d))


def build_items(rng=None):
    out = []
    for kind, nouns in (("rotating", ROT_NOUNS), ("settling", SET_NOUNS)):
        for w in nouns:
            for i, s in enumerate(SEQS):
                out.append({"kind": kind, "noun": w, "item": i, "seq": s,
                            "prompt": prompt_for(w, s), "gold": gold_for(s)})
    return out


def fit_items(task):
    """delta_e fit items, prompt-filtered. Same construction as A48."""
    seen, out = set(), []
    for prompt, gold, distractor in items(task, n=400):
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, gold, distractor))
        if len(out) >= N_FIT:
            break
    return out


def shot_pool(task, k=2):
    used = {p for p, _, _ in fit_items(task)}
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
        os.path.abspath("steerregime.json")
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

    # ONE implementation, never reimplemented: D137's instrument null rests on the
    # kernel-side and analysis-side values being the same function.
    from traj_geom.metrics.dynamics import rotation_power

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

    def encode(body):
        text = tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    def prelude_e(ids):
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
        def pre(_m, inp):
            cur = inp[0]
            x, e = cur[..., :d_model], cur[..., d_model:].clone()
            e[:, -1, :] = e[:, -1, :] + alpha * delta
            return (torch.cat([x, e], dim=-1),)
        return pre

    def measure(ids, gold, delta=None, alpha=0.0):
        """R at the last position AND the gold's rank curve, from one forward."""
        n = ids.shape[1]
        freqs = model.freqs_cis[:, :n]
        g0 = tok(gold, add_special_tokens=False).input_ids[0]
        ranks, traj = [], []
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
                h = o.detach()
                traj.append(h[0, -1].float().cpu().numpy().copy())
                row = torch.log_softmax(coda_head(h).float()[0, n - 1], dim=-1)
                ranks.append(int((row > row[g0]).sum().item()) + 1)

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
        R = float(rotation_power(np.array(traj, dtype=np.float32)))
        return {"R": R, "rotating": int(R > THRESHOLD), "best_rank": int(min(ranks)),
                "best_depth": int(np.argmin(ranks)) + 1, "final_rank": ranks[-1],
                "first_tok": first, "first_exact": int(first.strip() == gold.strip()),
                "n_tokens": int(n)}

    # ---- delta_e, A48's construction ---------------------------------------------
    diffs = []
    for task in FIT_FAMILIES:
        pool = shot_pool(task, k=2)
        assert len(pool) == 2, f"{task}: only {len(pool)} exemplars -- delta would be degenerate"
        pre = "".join(f"{p}\nAnswer: {g}\n\n" for p, g, _ in pool)
        for prompt, gold, _d in fit_items(task):
            e0 = prelude_e(encode(prompt))[0, -1, :]
            e2 = prelude_e(encode(pre + prompt))[0, -1, :]
            diffs.append((e2 - e0).clone())
    delta = torch.stack(diffs).mean(0)
    dn = float(delta.norm())
    print(f"delta_e from {len(diffs)} fit pairs over {FIT_FAMILIES}: ||delta|| = {dn:.4f}",
          flush=True)
    g = torch.Generator(device=delta.device).manual_seed(SEED)
    rnd = torch.randn(delta.shape, generator=g, device=delta.device, dtype=delta.dtype)
    rnd = rnd / rnd.norm() * dn
    print(f"random control: ||rnd|| = {float(rnd.norm()):.4f}", flush=True)

    test = build_items()
    print(f"{len(test)} base prompts: "
          f"{sum(1 for x in test if x['kind'] == 'rotating')} rotating, "
          f"{sum(1 for x in test if x['kind'] == 'settling')} settling", flush=True)
    t0, rows = time.time(), []
    for n_it, it in enumerate(test):
        ids = encode(it["prompt"])
        base = measure(ids, it["gold"])
        for name, vec in (("steer", delta), ("random", rnd)):
            for a in ALPHAS:
                if name == "random" and a == 0.0:
                    continue
                try:
                    r = measure(ids, it["gold"], delta=vec, alpha=a)
                    r["ok"] = True
                except Exception as exc:                       # noqa: BLE001
                    r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
                r.update({k: it[k] for k in ("kind", "noun", "item", "gold")})
                r["arm"], r["alpha"] = name, a
                r["base_R"], r["base_rotating"] = base["R"], base["rotating"]
                r["base_final_rank"] = base["final_rank"]
                rows.append(r)
        if n_it % 3 == 0:
            print(f"  {n_it}/{len(test)} ({time.time() - t0:.0f}s) {it['noun']}: "
                  f"base R {base['R']:.3f}", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok = [r for r in rows if r.get("ok")]
    import statistics as st
    import collections

    print("\n=== P1 INSTRUMENT NULL ===", flush=True)
    z = [r for r in ok if r["arm"] == "steer" and r["alpha"] == 0.0]
    dev = max((abs(r["R"] - r["base_R"]) for r in z), default=float("nan"))
    print(f"  n={len(z)}  max |dR| at alpha=0: {dev:.3e}  "
          f"{'OK' if dev == 0.0 else 'FAIL -- nothing below is readable'}", flush=True)

    print("\n=== P3/P4 PRIMARY: R by alpha, per arm, split by base regime ===", flush=True)
    print(f"{'kind':10s} {'arm':7s} {'alpha':>6s} {'n':>3s} {'med R':>7s} {'rotating':>9s} "
          f"{'first_exact':>11s} {'med final_rank':>14s}")
    for kind in ("rotating", "settling"):
        for name in ("steer", "random"):
            for a in ALPHAS:
                v = [r for r in ok if r["kind"] == kind and r["arm"] == name
                     and r["alpha"] == a]
                if not v:
                    continue
                print(f"{kind:10s} {name:7s} {a:6.2f} {len(v):3d} "
                      f"{st.median([r['R'] for r in v]):7.3f} "
                      f"{sum(r['rotating'] for r in v):4d}/{len(v):<4d} "
                      f"{sum(r['first_exact'] for r in v) / len(v):11.3f} "
                      f"{st.median([r['final_rank'] for r in v]):14.1f}", flush=True)

    print("\n=== THE FORK ===", flush=True)
    for kind in ("rotating", "settling"):
        base_side = 1 if kind == "rotating" else 0
        flips = {}
        for name in ("steer", "random"):
            f = [r for r in ok if r["kind"] == kind and r["arm"] == name
                 and r["alpha"] != 0.0 and r["rotating"] != base_side]
            tot = [r for r in ok if r["kind"] == kind and r["arm"] == name and r["alpha"] != 0.0]
            flips[name] = (len(f), len(tot))
        print(f"  {kind:10s} crossings: steer {flips['steer'][0]}/{flips['steer'][1]}, "
              f"random {flips['random'][0]}/{flips['random'][1]}", flush=True)
    print("  -> steer >> random means the format direction moves the regime; "
          "steer ~= random means any large perturbation does (D144); "
          "both ~0 means the axes are SEPARATE.", flush=True)

    print("\n=== P5 FIRST-TOKEN CENSUS by alpha (steer arm) ===", flush=True)
    for a in ALPHAS:
        v = [r for r in ok if r["arm"] == "steer" and r["alpha"] == a]
        if v:
            print(f"  alpha={a:5.2f}: "
                  f"{dict(collections.Counter(r['first_tok'] for r in v).most_common(5))}",
                  flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "alphas": list(ALPHAS), "depth": DEPTH, "seed": SEED,
                   "threshold": THRESHOLD, "delta_norm": dn, "n_fit_pairs": len(diffs),
                   "rot_nouns": list(ROT_NOUNS), "set_nouns": list(SET_NOUNS),
                   "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} conditions, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
