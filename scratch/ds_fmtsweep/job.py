"""A43: which answer FORMAT actually gets a parseable answer out of Huginn?

WHAT IS ALREADY KNOWN, AND IT SHAPES THE ARMS. The model has the answer and cannot emit it in
the position we score. D172: first-token exact-match reads **0.125** where the answer is
actually produced in **0.781** of generations, against a measured chance rate of 0.042-0.152.
D174: by unroll ~8 the model has committed to a prose frame -- `The` opens **39.4%** of 1260
banked generations, and the rate at which it opens with the gold does not move with depth
(p = 0.515). So the gap is elicitation, and two routes out of it are already measured:

  * **INSTRUCTION FAILS.** `kaggle_depthacc` ran `"Reply with only the answer."` as a second
    arm at every depth. Exact-match **0.000 at r = 32**, identical to bare; containment 0.389
    against bare's 0.333. Telling Huginn to emit only the answer does not work.
  * **DEMONSTRATION WORKS.** D169: five in-context examples move ARC-Easy letter-argmax
    0.416 -> 0.723 against a published 0.699, paired, exact McNemar p = 3.35e-07.

WHY THESE PARTICULAR ARMS, AND THE EVIDENCE FOR EACH. The banked generations show Huginn
reaching for two answer conventions **unprompted**: `$\\boxed{...}$` in **21 of 1260**
(`'The final total is $\\boxed{-'`, `'There are $\\boxed{4}$'`) and an `Answer:` marker in 8
more. Those are conventions it learned, so asking for them is a different request from asking
it to suppress prose. And D175 showed the harness's bare continuation format -- no chat
template, no option list -- reaching 0.658 on ARC at zero shots.

    A  bare                     the census's own prompt                    -- control
    B  "Reply with only the answer."                                      -- REPLICATION GATE
    C  "Put your final answer in \\boxed{}."                              -- a convention it uses
    D  "Question: ...\\nAnswer:" with NO chat template                     -- D175's winner
    E  "Work through it, then write 'Answer: <answer>' on the last line." -- reason-then-mark

PREREGISTERED PREDICTIONS:

  P1  REPLICATION GATE, VOID WITHOUT IT. Arm B must reproduce `kaggle_depthacc`'s failure:
      exact-match at or near **0.000**, not meaningfully above arm A. If arm B suddenly works,
      this run differs from that one in some way I have not identified and nothing else may be
      read.

  P2  PRIMARY. Does any arm yield a **parseable** answer -- extracted from `\\boxed{}`, from
      after an `Answer:` marker, or as the whole trimmed output -- at a rate well above arm A's?
      Registered before running: **arm C or E above 0.30 parsed-exact** would be a usable
      format; all arms within ~0.05 of bare means format instructions do not work on this model
      at zero shots and **demonstration is the only route**, which is a real finding and makes
      A42's few-shot arm the whole story.

  P3  PARSING IS REPORTED SEPARATELY FROM PRODUCING. Containment is scored on every arm too. An
      arm can raise containment (the answer is in there) without raising parsed-exact (we can
      find it), and those are different problems with different fixes. Both are printed.

  P4  THE CHANCE-RATE CONTROL, as in A38. For every generation, how often does some OTHER
      item's gold appear? Digit golds make containment cheap and this is the only thing that
      makes a containment number mean anything.

  P5  COST IS PART OF THE ANSWER. Arm E asks for reasoning first, so it will be longer. Median
      generated characters and the truncation rate are reported per arm -- a format that works
      only by spending 3x the tokens is a different recommendation from one that does not.

  P6  READ THE RAW OUTPUT. Three literal generations per arm are printed in full. D110 printed
      a capability verdict over empty strings; every format claim here is checkable against the
      strings that produced it.
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
MAX_NEW = 32
N_TEST = 6
N_ITEMS = 24
SEED = 20260811
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 10500

# six families spanning the range: three the model does well, two mid, one it cannot do
FAMILIES = ("echo_digit", "add1", "sort_min", "count_mod3", "track_total", "max_run")

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



ARMS = {
    "A_bare":      None,
    "B_only":      "Reply with only the answer.",
    "C_boxed":     "Put your final answer in \\boxed{}.",
    "D_harness":   "__HARNESS__",          # no chat template, "Question: ...\nAnswer:"
    "E_reason":    "Work through it, then write 'Answer: <answer>' on the last line.",
}


def build_items(rng=None):
    out = []
    for task in FAMILIES:
        for i, (prompt, gold, _d) in enumerate(items(task)[:N_TEST]):
            out.append({"family": task, "item": i, "prompt": prompt, "gold": gold})
    return out


def parse_answer(text, arm):
    """The ONE parser, banked with the row. Tries the arm's own convention, then falls back.

    Order matters and is fixed here rather than per-arm, so a `\\boxed{}` that appears
    spontaneously in another arm is still credited -- 21 of 1260 banked generations produce
    one unprompted, so restricting the parser by arm would undercount the control arms.
    """
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
        os.path.abspath("fmtsweep.json")
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

    def make_text(prompt, arm):
        suffix = ARMS[arm]
        if suffix == "__HARNESS__":
            return f"Question: {prompt}\nAnswer:"          # bare, no chat template (D175)
        body = prompt if suffix is None else prompt + "\n" + suffix
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)

    def generate(prompt, arm):
        ids = tok(make_text(prompt, arm), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        first, truncated = None, True
        for step in range(MAX_NEW):
            with torch.no_grad():
                torch.manual_seed(SEED)
                res = model(input_ids=ids, num_steps=DEPTH)
            logits = res.logits if hasattr(res, "logits") else res[0]
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            if first is None:
                first = tok.decode([int(nxt[0, 0])])
            if int(nxt[0, 0]) in stop:
                truncated = False
                break
            ids = torch.cat([ids, nxt], dim=1)
        return tok.decode(ids[0, n_p:], skip_special_tokens=True), first, truncated

    test = build_items()
    golds_by_family = {}
    for it in test:
        golds_by_family.setdefault(it["family"], set()).add(it["gold"])
    print(f"{len(test)} items x {len(ARMS)} arms = {len(test) * len(ARMS)} generations",
          flush=True)

    t0, rows = time.time(), []
    for arm in ARMS:
        for n, it in enumerate(test):
            try:
                gen, first, trunc = generate(it["prompt"], arm)
                ok, why = True, ""
            except Exception as exc:  # noqa: BLE001
                gen, first, trunc, ok, why = "", "", True, False, f"{type(exc).__name__}: {exc}"
            parsed, how = parse_answer(gen, arm) if ok else ("", "none")
            others = sorted(golds_by_family[it["family"]] - {it["gold"]})
            rows.append({**{k: it[k] for k in ("family", "item", "gold", "prompt")},
                         "arm": arm, "gen": gen, "first_tok": first, "ok": ok, "why": why,
                         "truncated": trunc, "parsed": parsed, "parse_how": how,
                         "parsed_exact": bool(ok and parsed.strip().lower()
                                              == it["gold"].strip().lower()),
                         "first_exact": bool(first and first.strip() == it["gold"].strip()),
                         "contains_gold": bool(ok and contains(gen, it["gold"])),
                         "n_other_hits": sum(contains(gen, g) for g in others) if ok else 0,
                         "n_others": len(others), "gen_chars": len(gen)})
            if n % 12 == 0:
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

    print("\n=== P6 LITERAL GENERATIONS, 3 per arm ===", flush=True)
    for arm in ARMS:
        for r in [x for x in ok_rows if x["arm"] == arm][:3]:
            print(f"  [{arm:10s}] {r['family']:11s} gold={r['gold']!r:7s} "
                  f"parsed={r['parsed']!r:14s} ({r['parse_how']})\n"
                  f"      gen={r['gen']!r}", flush=True)

    print("\n=== P1/P2/P3/P5 BY ARM ===", flush=True)
    print(f"{'arm':11s} {'n':>4s} {'first_exact':>11s} {'parsed_exact':>12s} {'contains':>9s} "
          f"{'CHANCE':>7s} {'med_chars':>9s} {'trunc':>6s}")
    import statistics as st
    summary = {}
    for arm in ARMS:
        v = [r for r in ok_rows if r["arm"] == arm]
        if not v:
            continue
        fe = sum(r["first_exact"] for r in v) / len(v)
        pe = sum(r["parsed_exact"] for r in v) / len(v)
        ct = sum(r["contains_gold"] for r in v) / len(v)
        ch = sum(r["n_other_hits"] for r in v) / max(1, sum(r["n_others"] for r in v))
        summary[arm] = (fe, pe, ct, ch)
        print(f"{arm:11s} {len(v):4d} {fe:11.3f} {pe:12.3f} {ct:9.3f} {ch:7.3f} "
              f"{st.median([r['gen_chars'] for r in v]):9.1f} "
              f"{sum(r['truncated'] for r in v) / len(v):6.2f}", flush=True)

    print("\n=== P1 GATE ===", flush=True)
    if "A_bare" in summary and "B_only" in summary:
        da = summary["B_only"][1] - summary["A_bare"][1]
        print(f"  arm B parsed_exact {summary['B_only'][1]:.3f} vs bare "
              f"{summary['A_bare'][1]:.3f}  (d {da:+.3f}) -- depthacc found NO gain; "
              f"{'consistent' if abs(da) < 0.15 else 'INCONSISTENT, read nothing below'}",
              flush=True)

    print("\n=== P2 VERDICT ===", flush=True)
    if summary:
        base = summary.get("A_bare", (0, 0, 0, 0))[1]
        best = max(summary, key=lambda a: summary[a][1])
        print(f"  best arm: {best} at parsed_exact {summary[best][1]:.3f} "
              f"(bare {base:.3f}, gain {summary[best][1] - base:+.3f})", flush=True)
        if summary[best][1] > 0.30 and summary[best][1] - base > 0.15:
            print("  -> a format instruction DOES work at zero shots; use it.", flush=True)
        else:
            print("  -> no format instruction works at zero shots. Demonstration (A42's "
                  "few-shot arm) is the only route, and that is the finding.", flush=True)

    print("\n=== P4 CONTAINMENT vs CHANCE, per arm ===", flush=True)
    for arm, (fe, pe, ct, ch) in summary.items():
        print(f"  {arm:11s} contains {ct:.3f} chance {ch:.3f} lift {ct - ch:+.3f} "
              f"-> {'INFORMATIVE' if ct > ch + 0.2 else 'AT CHANCE'}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "arms": list(ARMS), "families": list(FAMILIES),
                   "depth": DEPTH, "max_new": MAX_NEW, "seed": SEED,
                   "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} generations, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
