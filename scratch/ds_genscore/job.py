"""A38: if the answer is one token behind `The`, does it appear in the GENERATION?

THE PREDICTION D166 MAKES AND CANNOT TEST. D166 banked the top-5 tokens at every unroll and
found that what displaces the gold at the final unroll is a **prose sentence-opener** -- `The`
x12 of 32, `One` x3, `Number` x2, `To` x2, with `The` taking `sort_min` 8 of 8 at logprob
-0.079 while the gold sat at rank 2. The gold's final rank was median **2.5** and it was in the
top 5 for 24 of 32 items. So the reading is that Huginn **has** these answers and opens a
sentence instead of emitting the bare token, and that D157/D158's final-unroll zeros measure
answer FORMATTING rather than capability.

That is a prediction about text the model would actually produce, and every number behind it
is a rank at a single position. **This generates.** If the reading is right, the continuation
says something like *"The smallest number is 1"* and the gold is in it. If the reading is
wrong -- if the model opens with `The` and then produces the wrong answer, or wanders -- then
the zeros were capability after all and D166(2) must be narrowed to a statement about ranks.

WHY THE OBVIOUS SCORING IS A TRAP, AND WHAT GUARDS IT. Three of this project's worst errors
were containment scorers read without their controls:

  - D110 printed a capability verdict over empty strings.
  - D145's `contains` field disagreed with a reimplementation of itself by up to 13 points,
    because one stripped role markers and the other did not (RC4).
  - `echo_*` golds are IN THE PROMPT. A model that echoes its own prompt scores 100% on
    containment while answering nothing.

So: the raw generations are banked verbatim and printed (P3); containment is computed once,
here, and never recomputed downstream (RC4); and the **cross-item false-positive rate** is
measured rather than assumed (P4) -- for every generation, how often does some OTHER item's
gold appear in it? For single-digit families that rate is the whole ballgame.

PREREGISTERED PREDICTIONS:

  P1  REPLICATION GATE, tied to A35. The first generated token must reproduce D166: `The` (or
      another sentence opener) dominant on `sort_min`/`add1`/`echo_digit`, and the gold's own
      first token dominant on `echo_word`. Same items, same seed, same prompts as A35 by
      construction. If the first token does not reproduce, this run is measuring something
      else and nothing below may be read.

  P2  PRIMARY. Containment accuracy against first-token accuracy, per family. D166's reading
      predicts a LARGE gap -- first-token near 0 on `echo_digit`/`add1`/`sort_min` and
      containment well above it. Registered before running: **containment >= 0.5 on at least
      two of those three families** supports the reading; containment near first-token accuracy
      on all three refutes it and the zeros stand as capability.

  P3  LITERAL OUTPUT, NOT A COUNT. Every generation is banked and a sample of them is printed
      in full. A containment number without its strings is exactly the D110 failure.

  P4  THE CONTROL THAT DECIDES WHETHER P2 MEANS ANYTHING. For every generation, test the golds
      of all OTHER items in the same family. That is the chance rate of the containment test.
      If cross-item containment is nearly as high as own-gold containment -- which is entirely
      possible for `echo_digit`, where golds are single characters -- then P2's positive is
      noise and must be reported as such.

  P5  DEPTH. Everything is run at r = 32 (Huginn's default) and r = 48 (A35's depth). D157's
      inflation is defined across depth, so if containment is depth-insensitive while the
      rank-based axis is not, that is a further sign the rank axis is reading format.
"""

import json
import os
import re
import subprocess
import sys
import time

SEED = 20260811           # A35's seed: identical items, so P1 is a true replication
DEPTHS = (32, 48)
MAX_NEW = 24
N_ITEMS = 16
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 9000
FAMILIES = ("echo_word", "echo_digit", "add1", "sort_min")

# From `scratch/kaggle_depthacc/main.py`, which is the hardened path: begin_text, end_text,
# end_turn. Copied rather than imported because that file is a Kaggle kernel with no import
# guard; the values are the model's, not a derived quantity.
STOP_IDS = {65504, 65505, 65508}


def build_items(rng=None):
    """A35's generator verbatim, at more items. Same seed, so the first 8 coincide."""
    import random as _r
    rng = rng or _r.Random(SEED)
    words = ["cloud", "paper", "river", "stone", "candle", "garden", "window", "silver",
             "mirror", "forest", "bottle", "planet", "copper", "shadow", "market", "pencil"]
    out = []
    for i in range(N_ITEMS):
        w = words[i % len(words)]
        d = rng.randrange(10)
        a, b = rng.randrange(10), rng.randrange(10)
        seq = [rng.randrange(10) for _ in range(6)]
        out += [
            {"family": "echo_word", "item": i, "gold": w,
             "prompt": f"Repeat this word exactly.\nWord: {w}"},
            {"family": "echo_digit", "item": i, "gold": str(d),
             "prompt": f"Repeat this number exactly.\nNumber: {d}"},
            {"family": "add1", "item": i, "gold": str((a + 1) % 10),
             "prompt": f"Add one to this number, modulo ten.\nNumber: {a}"},
            {"family": "sort_min", "item": i, "gold": str(min(seq)),
             "prompt": f"Report the smallest number in this list.\n"
                       f"List: {' '.join(map(str, seq))}"},
        ]
    return out


def contains(text, gold):
    """The ONE containment definition in this kernel; banked, never recomputed (RC4).

    Word-boundary rather than substring: bare `in` would score `1` inside `10` and `river`
    inside `rivers`, and for single-digit golds that difference is most of the signal.
    """
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(gold)}(?![A-Za-z0-9])", text,
                          re.IGNORECASE))


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("genscore.json")
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

    stop = set(STOP_IDS)
    if getattr(tok, "eos_token_id", None) is not None:
        stop.add(tok.eos_token_id)

    def generate(prompt, depth):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        first = None
        for _ in range(MAX_NEW):
            with torch.no_grad():
                torch.manual_seed(SEED)
                res = model(input_ids=ids, num_steps=depth)
            logits = res.logits if hasattr(res, "logits") else res[0]
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            if first is None:
                first = tok.decode([int(nxt[0, 0])])
            if int(nxt[0, 0]) in stop:
                break
            ids = torch.cat([ids, nxt], dim=1)
        return tok.decode(ids[0, n_p:], skip_special_tokens=True), first

    items = build_items()
    golds_by_family = {}
    for it in items:
        golds_by_family.setdefault(it["family"], set()).add(it["gold"])

    t0, rows = time.time(), []
    for depth in DEPTHS:
        for n, it in enumerate(items):
            try:
                gen, first = generate(it["prompt"], depth)
                ok, why = True, ""
            except Exception as exc:  # noqa: BLE001
                gen, first, ok, why = "", "", False, f"{type(exc).__name__}: {exc}"
            others = sorted(golds_by_family[it["family"]] - {it["gold"]})
            rows.append({**{k: it[k] for k in ("family", "item", "gold", "prompt")},
                         "depth": depth, "gen": gen, "first_tok": first, "ok": ok, "why": why,
                         "first_exact": bool(first and first.strip() == it["gold"].strip()),
                         "first_prefix": bool(first and first.strip() and
                                              it["gold"].lower().startswith(
                                                  first.strip().lower())),
                         "contains_gold": contains(gen, it["gold"]),
                         # P4: how many OTHER golds of this family also appear?
                         "n_other_hits": sum(contains(gen, g) for g in others),
                         "n_others": len(others)})
            if n % 16 == 0:
                print(f"  d{depth} {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
                with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                    json.dump(rows, f)
            if time.time() - t0 > WALL_BUDGET_S:
                print("WALL BUDGET -- banking and stopping cleanly", flush=True)
                break

    ok_rows = [r for r in rows if r["ok"]]
    if not any(r["gen"].strip() for r in ok_rows):
        # the D62 failure mode: refuse to report over empty strings, as D110 did
        print("FATAL: every generation is empty. Refusing to score. (D62/D110)", flush=True)
        with open(out_path, "w") as f:
            json.dump({"rows": rows, "fatal": "all generations empty"}, f)
        return 1

    print("\n=== P3 LITERAL GENERATIONS (first 3 per family, r=32) ===", flush=True)
    for fam in FAMILIES:
        for r in [x for x in ok_rows if x["family"] == fam and x["depth"] == 32][:3]:
            print(f"  [{fam}] gold={r['gold']!r} first={r['first_tok']!r}\n"
                  f"      gen={r['gen']!r}", flush=True)

    print("\n=== P1/P2/P4/P5 ===", flush=True)
    print(f"{'family':11s} {'r':>3s} {'n':>3s} {'first==gold':>11s} {'first_prefix':>12s} "
          f"{'contains':>9s} {'CHANCE(P4)':>11s}")
    for depth in DEPTHS:
        for fam in FAMILIES:
            v = [r for r in ok_rows if r["family"] == fam and r["depth"] == depth]
            if not v:
                continue
            fe = sum(r["first_exact"] for r in v) / len(v)
            fp = sum(r["first_prefix"] for r in v) / len(v)
            ct = sum(r["contains_gold"] for r in v) / len(v)
            chance = (sum(r["n_other_hits"] for r in v) /
                      max(1, sum(r["n_others"] for r in v)))
            print(f"{fam:11s} {depth:3d} {len(v):3d} {fe:11.3f} {fp:12.3f} {ct:9.3f} "
                  f"{chance:11.3f}", flush=True)

    print("\n=== P4 READING ===", flush=True)
    for fam in FAMILIES:
        v = [r for r in ok_rows if r["family"] == fam]
        ct = sum(r["contains_gold"] for r in v) / max(1, len(v))
        ch = sum(r["n_other_hits"] for r in v) / max(1, sum(r["n_others"] for r in v))
        verdict = ("INFORMATIVE" if ct > ch + 0.2 else
                   "AT CHANCE -- containment here means nothing")
        print(f"  {fam:11s} contains {ct:.3f} vs chance {ch:.3f}  -> {verdict}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "depths": list(DEPTHS), "max_new": MAX_NEW, "seed": SEED,
                   "families": list(FAMILIES), "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} generations, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
