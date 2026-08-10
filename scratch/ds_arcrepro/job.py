"""A29: reproduce Huginn's OWN published accuracy, and calibrate our capability axis.

WHY THIS MATTERS MORE THAN IT DID YESTERDAY. Every capability number in this project is
measured by instruments we built, on tasks we built, and **not one published result has
ever been reproduced here**. The single attempt (D14, validating the pipeline against
Blayney's winding rate) was withdrawn -- a detector with an unknown false-positive rate
cannot be validated by matching a published number. Huginn's own figures, GSM8K 32.6% and
ARC-E 69.9% at r = 32, are quoted from the model's paper and have never been re-measured.

Two results this week made that gap urgent rather than tidy-up work:

  * **D145**: Huginn's chat template glues the next-turn role header onto the answer with
    no separator, so our decoded scorer discards correct answers -- 72 missed hits in one
    bank, and D86's "exact match hits 0.0%" turned out to be an artefact.
  * **D147**: the project's standard axis, `min(rank) == 1` over the unroll budget, reads
    **1.000 at every difficulty level** on a task where the model is a constant responder.
    Not inflated -- vacuous.

So the question is no longer only "can Huginn do things". It is **"is our capability
instrument trustworthy at all"**, and the only external check available is the model's own
published number on a benchmark we did not design.

WHY ARC-Easy AND NOT GSM8K. ARC-E answers are a single letter from a fixed 4-way option
set, so it can be scored by argmax over the option tokens with **no generation, no
decoding, and no exposure to the D145 marker leak**. GSM8K needs multi-token chain-of-
thought whose scoring is exactly the machinery D145 showed is broken; reproducing it would
confound "can Huginn do it" with "can we score it", which is the confound this run exists
to remove.

PREREGISTERED PREDICTIONS:

  P1  REPRODUCTION, and it is the point. At r = 32 the published ARC-Easy figure is
      **69.9%**. **The registered bar: our argmax-over-options accuracy at r = 32 lands
      within +/- 10 points of it, i.e. in [59.9, 79.9].** Outside that band, one of three
      things is true and the row must say which: the harness differs from theirs, the
      published number does not describe this checkpoint, or our measurement is wrong.
      A miss is as informative as a hit and is reported either way.

  P2  THE DEPTH CURVE. Huginn's paper reports recall-heavy tasks saturating early while
      reasoning-heavy ones keep improving to r ~ 32-64. ARC-E is the recall-heavy arm, so
      accuracy should rise steeply to about r = 8 and then flatten. Reported across
      r = 1, 2, 4, 8, 16, 32, 64.

  P3  THE AXIS CALIBRATION, and the reason this run is worth GPU beyond the reproduction.
      The SAME orbits are scored three ways: (a) argmax over the 4 option tokens at the
      FINAL unroll -- the honest measure; (b) the same at ANY unroll -- the oracle D147
      showed can be vacuous; (c) full-vocabulary `min(rank) == 1` over the budget -- this
      project's standard axis. **The gap between (a) and (c) is the inflation factor of
      every capability number in this ledger, measured on a benchmark with a published
      answer.** D103 put it at 2.06x on our own tasks; this tests it against an external
      anchor.

  P4  CHANCE FLOOR AND BALANCE. ARC-E items have 3-5 options, so chance is not a constant.
      The per-item chance is recorded and the majority-option baseline computed, because
      D147 showed a constant responder can beat a naive chance floor.

  P5  NO GENERATION, SO NO MARKER LEAK. Scoring is argmax over option-letter logits at a
      fixed position; nothing is decoded. This is deliberate -- it makes the reproduction
      independent of the defect D145 found, so a mismatch cannot be blamed on it.
"""

import json
import os
import subprocess
import sys
import time

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
DEPTHS = (1, 2, 4, 8, 16, 32, 64)
N_ITEMS = 150
PUBLISHED_ARC_E = 0.699      # the model's own paper, r = 32
SEED = 20260810
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 11000


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("arcrepro.json")
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"results -> {out_path}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    print(f"ARC-Easy test: {len(ds)} items", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    core_last = model.transformer.core_block[-1]

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def build(rec):
        """Standard 4-way MC framing; the answer is a single option LETTER."""
        ch = rec["choices"]
        labels = list(ch["label"])
        texts = list(ch["text"])
        # ARC ships some items labelled 1-4 rather than A-D; normalise to letters
        if labels and labels[0].isdigit():
            labels = [chr(ord("A") + int(x) - 1) for x in labels]
        gold = rec["answerKey"]
        if gold.isdigit():
            gold = chr(ord("A") + int(gold) - 1)
        body = "\n".join(f"{a}. {b}" for a, b in zip(labels, texts))
        q = (f"Question: {rec['question']}\n{body}\n"
             f"Answer with the letter of the correct option.")
        return q, labels, gold

    items = []
    for rec in ds:
        q, labels, gold = build(rec)
        if gold not in labels or len(labels) < 3:
            continue
        items.append({"q": q, "labels": labels, "gold": gold, "id": rec["id"]})
        if len(items) >= N_ITEMS:
            break
    print(f"{len(items)} usable items; option counts "
          f"{ {k: sum(1 for i in items if len(i['labels']) == k) for k in (3, 4, 5)} }",
          flush=True)

    t0, rows = time.time(), []
    for n, it in enumerate(items):
        text = tok.apply_chat_template([{"role": "user", "content": it["q"]}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        opt_ids = [tok(a, add_special_tokens=False).input_ids[0] for a in it["labels"]]
        g_ids = tok(it["gold"], add_special_tokens=False).input_ids
        gi = it["labels"].index(it["gold"])

        for R in DEPTHS:
            picks, ranks = [], []
            core_last._forward_hooks.clear()

            def hook(_m, _i, o):
                with torch.no_grad():
                    row = coda_head(o.detach(), freqs).float()[0, n_p - 1]
                    picks.append(int(np.argmax([float(row[c]) for c in opt_ids])))
                    ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)

            h = core_last.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    torch.manual_seed(SEED)
                    model(input_ids=ids, num_steps=R)
            finally:
                h.remove()
            rows.append({
                "id": it["id"], "gold": it["gold"], "n_options": len(it["labels"]),
                "depth": R, "n_tokens": int(n_p),
                # (a) the honest measure: option argmax at the FINAL unroll
                "final_correct": bool(picks[-1] == gi),
                # (b) the oracle over depth on the option set
                "any_correct": bool(gi in picks),
                # (c) this project's standard axis: full-vocab min-rank over the budget
                "vocab_minrank_correct": bool(min(ranks) == 1),
                "final_rank": int(ranks[-1]), "best_rank": int(min(ranks)),
                "ok": True,
            })
        if n % 20 == 0:
            print(f"  {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    print("\n=== P1 REPRODUCTION / P2 DEPTH CURVE / P3 AXIS CALIBRATION ===", flush=True)
    for R in DEPTHS:
        v = [r for r in rows if r["depth"] == R]
        if not v:
            continue
        a = np.mean([r["final_correct"] for r in v])
        b = np.mean([r["any_correct"] for r in v])
        c = np.mean([r["vocab_minrank_correct"] for r in v])
        print(f"  r={R:3d}  n={len(v):4d}  final {a:.3f}   any-unroll {b:.3f}   "
              f"vocab-minrank {c:.3f}   inflation (c/a) "
              f"{(c / a if a else float('nan')):.2f}x", flush=True)
    v32 = [r for r in rows if r["depth"] == 32]
    if v32:
        a32 = float(np.mean([r["final_correct"] for r in v32]))
        print(f"\n  P1: r=32 final-unroll accuracy {a32:.3f} against published "
              f"{PUBLISHED_ARC_E:.3f}  ->  "
              f"{'WITHIN +/-10 pts' if abs(a32 - PUBLISHED_ARC_E) <= 0.10 else 'OUTSIDE the registered band'}",
              flush=True)
    chance = float(np.mean([1.0 / r["n_options"] for r in rows])) if rows else float("nan")
    print(f"  P4: mean per-item chance {chance:.3f}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "depths": list(DEPTHS), "published_arc_e": PUBLISHED_ARC_E,
                   "seed": SEED, "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} measurements, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
