"""A50: does the instruction work from the SYSTEM turn, where D193 never looked?

WHAT D193 ESTABLISHED AND WHAT IT EXPLICITLY DID NOT. A43 swept four format instructions at
zero shots and every arm sat at the floor: parsed-exact **0.000** bare, **0.000** for *"Reply
with only the answer."*, **0.056** for a `\\boxed{}` request, **0.028** for the harness prompt,
against a registered bar of 0.30. Its own scope line is the reason this run exists: **all four
arms are USER-turn instructions**, so D193 licenses *"no user-turn format instruction works"*
and nothing wider.

**Another group solved exactly our problem from the system turn.** Paper 02 (`coda_lens_exp.py:83`,
`logit_lens_exp.py:137`) prefixes every prompt with:

    {"role": "system", "content": "You are a concise and helpful assistant.
                                   Always return only the final answer straightway."}

They needed terse output for a rank-trajectory analysis and this is how they got it. It is the
one form of the instruction this project has never tried (D189c).

THE DESIGN, AND ITS POINT IS ONE CONTRAST. Arms `U_concise` and `S_concise` carry **the same
sentence, verbatim**, and differ **only** in which turn it sits in. Everything else is held:
same items, same depth, same parser, same seed. So a difference between them is turn placement
and nothing else.

    A_bare      no instruction at all                      -- replication gate against D193
    U_concise   paper 02's sentence, USER turn             -- the control for placement
    S_concise   paper 02's sentence, SYSTEM turn           -- THE TEST
    E_reason    "work through it, then write Answer: ..."  -- A43's fifth arm, which never ran

PREREGISTERED PREDICTIONS:

  P1  REPLICATION GATE. `A_bare` must reproduce A43's floor -- parsed-exact at or near **0.000**
      on these same families. If bare suddenly works, this run differs from A43 in a way I have
      not identified and nothing below may be read.

  P2  PRIMARY, REGISTERED BOTH WAYS. Does `S_concise` clear bare by more than **0.15**?
      **Yes** -> the instruction works from the system turn and D193's negative is a statement
      about turn placement, not about instruction-following; the practical recommendation for
      every future eval in this project changes. **No** -> D193 generalises from "user-turn" to
      "any turn", Huginn does not follow format instructions at all, and paper 02's system
      prompt was doing less work for them than it appears to.

  P3  THE GATE THAT STOPS THIS BEING VACUOUS. **The system text must actually survive
      templating.** If `apply_chat_template` silently drops an unsupported `system` role we
      would measure a bare prompt and call it a system prompt. The rendered string is asserted
      to contain the sentence, the assertion is checked for every arm before any generation,
      and the full rendered prompt is banked. A run that fails this prints VOID and scores
      nothing -- D110 printed a capability verdict over empty strings and that is the failure
      being guarded.

  P4  CHANCE-RATE CONTROL, as in A38/A43. Containment against the item's own gold is scored
      beside containment against the OTHER items' golds in the same family. Digit golds make
      containment cheap and the lift is the only interpretable quantity.

  P5  LITERAL OUTPUT. Three generations per arm printed in full, plus the rendered prompt for
      one item per arm. Every claim here is checkable against the strings that produced it.
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

DEPTH = 32
MAX_NEW = 16               # A43 used 32; halved for turnaround, still past the answer position
N_TEST = 4
N_ITEMS = 24
SEED = 20260811
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 7000

FAMILIES = ("echo_digit", "add1", "sort_min", "count_mod3", "track_total", "max_run")

CONCISE = ("You are a concise and helpful assistant. "
           "Always return only the final answer straightway.")

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





ARMS = ("A_bare", "U_concise", "S_concise", "E_reason")
REASON = "Work through it, then write 'Answer: <answer>' on the last line."


def build_items(rng=None):
    out = []
    for task in FAMILIES:
        for i, (prompt, gold, _d) in enumerate(items(task)[:N_TEST]):
            out.append({"family": task, "item": i, "prompt": prompt, "gold": gold})
    return out


def parse_answer(text, arm=None):
    """A43's parser verbatim -- one implementation, so the arms are comparable to D193."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip(), "boxed"
    m = re.search(r"(?i)answer\s*[:=]\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip().rstrip(".").strip(), "marker"
    return text.strip().split("\n")[0].strip().rstrip(".").strip(), "wholeline"


def contains(text, gold):
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(gold)}(?![A-Za-z0-9])", text,
                          re.IGNORECASE))


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("sysprompt.json")
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

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tok = AutoTokenizer.from_pretrained(mnt)
    cfg = AutoConfig.from_pretrained(mnt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mnt, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    stop = {65504, 65505, 65508}
    if getattr(tok, "eos_token_id", None) is not None:
        stop.add(tok.eos_token_id)

    def render(prompt, arm):
        if arm == "A_bare":
            msgs = [{"role": "user", "content": prompt}]
        elif arm == "U_concise":
            msgs = [{"role": "user", "content": prompt + "\n" + CONCISE}]
        elif arm == "S_concise":
            msgs = [{"role": "system", "content": CONCISE},
                    {"role": "user", "content": prompt}]
        else:
            msgs = [{"role": "user", "content": prompt + "\n" + REASON}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---- P3: the system text must survive templating, checked BEFORE any generation ----
    print("\n=== P3 GATE: does the SYSTEM role survive apply_chat_template? ===", flush=True)
    probe = build_items()[0]["prompt"]
    rendered = {a: render(probe, a) for a in ARMS}
    for a in ARMS:
        print(f"  --- {a} ---\n{rendered[a]!r}", flush=True)
    sys_ok = CONCISE in rendered["S_concise"]
    usr_ok = CONCISE in rendered["U_concise"]
    print(f"\n  system text present in S_concise: {sys_ok}", flush=True)
    print(f"  same text present in U_concise:   {usr_ok}", flush=True)
    differ = rendered["S_concise"] != rendered["U_concise"]
    print(f"  the two renderings actually differ: {differ}", flush=True)
    if not (sys_ok and usr_ok and differ):
        print("\nVOID: the system prompt does not survive templating, or the two arms render "
              "identically. Scoring would compare a prompt against itself. Refusing to score.",
              flush=True)
        with open(out_path, "w") as f:
            json.dump({"void": True, "rendered": rendered, "sys_ok": sys_ok,
                       "usr_ok": usr_ok, "differ": differ}, f)
        return 1
    print("  GATE PASSES -- the system turn is real and distinct from the user turn.",
          flush=True)

    def generate(prompt, arm):
        ids = tok(render(prompt, arm), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        first = None
        for _ in range(MAX_NEW):
            with torch.no_grad():
                torch.manual_seed(SEED)
                res = model(input_ids=ids, num_steps=DEPTH)
            logits = res.logits if hasattr(res, "logits") else res[0]
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            if first is None:
                first = tok.decode([int(nxt[0, 0])])
            if int(nxt[0, 0]) in stop:
                break
            ids = torch.cat([ids, nxt], dim=1)
        return tok.decode(ids[0, n_p:], skip_special_tokens=True), first, int(n_p)

    test = build_items()
    golds = {}
    for it in test:
        golds.setdefault(it["family"], set()).add(it["gold"])
    print(f"\n{len(test)} items x {len(ARMS)} arms = {len(test) * len(ARMS)} generations",
          flush=True)

    t0, rows = time.time(), []
    for arm in ARMS:
        for n, it in enumerate(test):
            try:
                gen, first, n_p = generate(it["prompt"], arm)
                ok, why = True, ""
            except Exception as exc:  # noqa: BLE001
                gen, first, n_p, ok, why = "", "", 0, False, f"{type(exc).__name__}: {exc}"
            parsed, how = parse_answer(gen) if ok else ("", "none")
            others = sorted(golds[it["family"]] - {it["gold"]})
            rows.append({**{k: it[k] for k in ("family", "item", "gold", "prompt")},
                         "arm": arm, "gen": gen, "first_tok": first, "ok": ok, "why": why,
                         "n_prompt": n_p, "parsed": parsed, "parse_how": how,
                         "parsed_exact": bool(ok and parsed.strip().lower()
                                              == it["gold"].strip().lower()),
                         "first_exact": bool(first and first.strip() == it["gold"].strip()),
                         "contains_gold": bool(ok and contains(gen, it["gold"])),
                         "n_other_hits": sum(contains(gen, g) for g in others) if ok else 0,
                         "n_others": len(others), "gen_chars": len(gen)})
            if n % 8 == 0:
                print(f"  {arm} {n}/{len(test)} ({time.time() - t0:.0f}s)", flush=True)
                with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                    json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok_rows = [r for r in rows if r["ok"]]
    if not any(r["gen"].strip() for r in ok_rows):
        print("FATAL: every generation empty. Refusing to score (D62/D110).", flush=True)
        with open(out_path, "w") as f:
            json.dump({"rows": rows, "fatal": "all generations empty"}, f)
        return 1

    import statistics as st
    print("\n=== P5 LITERAL GENERATIONS, 3 per arm ===", flush=True)
    for a in ARMS:
        for r in [x for x in ok_rows if x["arm"] == a][:3]:
            print(f"  [{a:10s}] {r['family']:11s} gold={r['gold']!r:6s} "
                  f"first={r['first_tok']!r:10s} gen={r['gen']!r}", flush=True)

    print("\n=== P1/P2/P4 BY ARM ===", flush=True)
    print(f"{'arm':11s} {'n':>4s} {'first_exact':>11s} {'parsed_exact':>12s} {'contains':>9s} "
          f"{'CHANCE':>7s} {'med_chars':>9s} {'med_prompt_tok':>14s}")
    S = {}
    for a in ARMS:
        v = [r for r in ok_rows if r["arm"] == a]
        if not v:
            continue
        fe = sum(r["first_exact"] for r in v) / len(v)
        pe = sum(r["parsed_exact"] for r in v) / len(v)
        ct = sum(r["contains_gold"] for r in v) / len(v)
        ch = sum(r["n_other_hits"] for r in v) / max(1, sum(r["n_others"] for r in v))
        S[a] = (fe, pe, ct, ch)
        print(f"{a:11s} {len(v):4d} {fe:11.3f} {pe:12.3f} {ct:9.3f} {ch:7.3f} "
              f"{st.median([r['gen_chars'] for r in v]):9.1f} "
              f"{st.median([r['n_prompt'] for r in v]):14.0f}", flush=True)

    print("\n=== P1 GATE vs D193 (A43 bare: parsed 0.000) ===", flush=True)
    if "A_bare" in S:
        print(f"  A_bare parsed_exact {S['A_bare'][1]:.3f}  "
              f"{'consistent with A43' if S['A_bare'][1] < 0.15 else 'INCONSISTENT -- read nothing below'}",
              flush=True)

    print("\n=== P2 PRIMARY: does the SYSTEM turn beat the USER turn? ===", flush=True)
    if {"A_bare", "U_concise", "S_concise"} <= set(S):
        b, u, s = S["A_bare"][1], S["U_concise"][1], S["S_concise"][1]
        print(f"  bare {b:.3f} | same sentence USER turn {u:.3f} | SYSTEM turn {s:.3f}",
              flush=True)
        print(f"  system - bare = {s - b:+.3f}   system - user = {s - u:+.3f}", flush=True)
        if s - b > 0.15:
            print("  -> THE SYSTEM TURN WORKS. D193's negative is about turn placement, not "
                  "about instruction-following, and every future eval here should use it.",
                  flush=True)
        else:
            print("  -> the system turn does NOT work either. D193 generalises from "
                  "'user-turn' to 'any turn': Huginn does not follow format instructions.",
                  flush=True)

    print("\n=== P4 CONTAINMENT vs CHANCE ===", flush=True)
    for a, (fe, pe, ct, ch) in S.items():
        print(f"  {a:11s} contains {ct:.3f} chance {ch:.3f} lift {ct - ch:+.3f} "
              f"-> {'INFORMATIVE' if ct > ch + 0.2 else 'AT CHANCE'}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "arms": list(ARMS), "families": list(FAMILIES),
                   "depth": DEPTH, "max_new": MAX_NEW, "seed": SEED, "concise": CONCISE,
                   "rendered_probe": rendered, "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} generations, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
