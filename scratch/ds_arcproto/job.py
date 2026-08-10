"""A30: is D153's 29-point miss a PROTOCOL difference or a capability difference?

THE QUESTION. A29 scored ARC-Easy by argmax over the option-LETTER tokens at the final
unroll and got **0.407 at r = 32** against Huginn's published **0.699** -- a 29-point miss
of a pre-registered +/-10 band (D153). D153 named protocol as the most likely explanation
and recorded it explicitly as a hypothesis, not evidence. This runs the test.

The standard multiple-choice protocol in common evaluation harnesses does not ask the
model to emit a letter at all. It scores each option by the **log-likelihood of the option
TEXT as a continuation**, and picks the argmax -- usually reported twice, once on the raw
summed log-probability and once length-normalised (`acc` and `acc_norm`). A base model
that understands the content but has never learned the "answer with the letter A/B/C/D"
convention will score far lower under A29's protocol than under that one, **without any
difference in what it knows**.

So the two arms are run on the SAME 150 items, at the same depths, in one job:

  * **arm L (letter)** -- reproduce A29 exactly: argmax over option-letter tokens at the
    final unroll.
  * **arm T (text)** -- for each option, teacher-force the option text as a continuation
    of the prompt and sum its token log-probabilities; report raw-sum argmax and
    length-normalised argmax separately, because the two disagree in the literature and
    reporting only the flattering one would be the error this run exists to avoid.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT IDENTITY, VOID WITHOUT IT. Arm L must reproduce A29's numbers on the
      same items to within Monte-Carlo zero -- h_0 is seeded and the items are rebuilt by
      the same deterministic procedure, so **r = 32 must come out at 0.407**. If it does
      not, the two runs are not measuring the same thing and no comparison is licensed.

  P2  PRIMARY. Arm T at r = 32 against the published **0.699**. **The registered
      reading:** if arm T lands inside [0.599, 0.799] then the gap was PROTOCOL, D153's
      hypothesis is confirmed, and this project reproduces a published number for the
      first time. If arm T also misses low, protocol is refuted as the explanation and
      the discrepancy is real -- which is a more serious finding and must be reported as
      such rather than buried.

  P3  BOTH NORMALISATIONS, decided in advance. Raw-sum favours short options, length
      normalisation favours long ones; harnesses report both. **Both are reported and
      neither is designated the headline after the fact.**

  P4  A LENGTH SANITY CHECK, because it is the known failure mode of arm T. If the
      correct option is systematically shorter or longer than the distractors, raw-sum
      and normalised will diverge for a reason that has nothing to do with the model.
      Mean option token lengths are recorded per item.

  P5  CHANCE AND BASELINE. Mean per-item chance and the majority-option baseline are
      recomputed here rather than assumed, since D147 showed a constant responder can
      clear a naive chance floor.
"""

import json
import os
import subprocess
import sys
import time

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
DEPTHS = (16, 32, 64)
N_ITEMS = 150
PUBLISHED_ARC_E = 0.699
SEED = 20260810
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 12000


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("arcproto.json")
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
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    # A29's item construction, reproduced verbatim so P1 can be an identity check
    def build(rec):
        ch = rec["choices"]
        labels, texts = list(ch["label"]), list(ch["text"])
        if labels and labels[0].isdigit():
            labels = [chr(ord("A") + int(x) - 1) for x in labels]
        gold = rec["answerKey"]
        if gold.isdigit():
            gold = chr(ord("A") + int(gold) - 1)
        body = "\n".join(f"{a}. {b}" for a, b in zip(labels, texts))
        q = (f"Question: {rec['question']}\n{body}\n"
             f"Answer with the letter of the correct option.")
        return q, labels, texts, gold

    items = []
    for rec in ds:
        q, labels, texts, gold = build(rec)
        if gold not in labels or len(labels) < 3:
            continue
        items.append({"q": q, "labels": labels, "texts": texts, "gold": gold,
                      "id": rec["id"], "question": rec["question"]})
        if len(items) >= N_ITEMS:
            break
    print(f"{len(items)} items", flush=True)

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    def letter_pick(it, R):
        """Arm L: A29's protocol, reproduced."""
        ids = tok(chat(it["q"]), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        with torch.no_grad():
            torch.manual_seed(SEED)
            out = model(input_ids=ids, num_steps=R)
        logits = out.logits if hasattr(out, "logits") else out[0]
        row = logits.float()[0, n_p - 1]
        opt = [tok(a, add_special_tokens=False).input_ids[0] for a in it["labels"]]
        return int(np.argmax([float(row[c]) for c in opt]))

    def text_scores(it, R):
        """Arm T: sum log P(option text | question), teacher-forced. One forward each."""
        raw, norm, ntok = [], [], []
        # the continuation is scored after the SAME chat-formatted prompt, so the only
        # difference from arm L is what is being asked for, not how it is framed
        base = chat(it["q"])
        for txt in it["texts"]:
            full = base + " " + txt
            fids = tok(full, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(model.device)
            bids = tok(base, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(model.device)
            k = bids.shape[1]
            with torch.no_grad():
                torch.manual_seed(SEED)
                out = model(input_ids=fids, num_steps=R)
            logits = out.logits if hasattr(out, "logits") else out[0]
            lp = torch.log_softmax(logits.float()[0], dim=-1)
            tgt = fids[0, k:]
            if tgt.numel() == 0:
                raw.append(-1e9); norm.append(-1e9); ntok.append(0); continue
            s = float(sum(lp[k + i - 1, int(tgt[i])] for i in range(tgt.numel())))
            raw.append(s); norm.append(s / tgt.numel()); ntok.append(int(tgt.numel()))
        return raw, norm, ntok

    t0, rows = time.time(), []
    for n, it in enumerate(items):
        gi = it["labels"].index(it["gold"])
        for R in DEPTHS:
            li = letter_pick(it, R)
            raw, norm, ntok = text_scores(it, R)
            rows.append({
                "id": it["id"], "depth": R, "n_options": len(it["labels"]),
                "gold": it["gold"], "gold_idx": gi,
                "letter_correct": bool(li == gi),
                "text_raw_correct": bool(int(np.argmax(raw)) == gi),
                "text_norm_correct": bool(int(np.argmax(norm)) == gi),
                "gold_ntok": ntok[gi], "mean_ntok": float(np.mean(ntok)),
                "ok": True,
            })
        if n % 15 == 0:
            print(f"  {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    print("\n=== P1 IDENTITY / P2 PRIMARY / P3 BOTH NORMALISATIONS ===", flush=True)
    for R in DEPTHS:
        v = [r for r in rows if r["depth"] == R]
        if not v:
            continue
        print(f"  r={R:3d} n={len(v):4d}  letter {np.mean([r['letter_correct'] for r in v]):.3f}"
              f"   text-raw {np.mean([r['text_raw_correct'] for r in v]):.3f}"
              f"   text-norm {np.mean([r['text_norm_correct'] for r in v]):.3f}", flush=True)
    v32 = [r for r in rows if r["depth"] == 32]
    if v32:
        for k, nm in (("text_raw_correct", "raw-sum"), ("text_norm_correct", "length-norm")):
            a = float(np.mean([r[k] for r in v32]))
            print(f"  P2 {nm:12s} at r=32: {a:.3f} vs published {PUBLISHED_ARC_E:.3f} -> "
                  f"{'INSIDE the band -- the gap was PROTOCOL' if abs(a - PUBLISHED_ARC_E) <= 0.10 else 'still outside'}",
                  flush=True)
        print(f"  P4 gold option length {np.mean([r['gold_ntok'] for r in v32]):.1f} tokens "
              f"vs mean option {np.mean([r['mean_ntok'] for r in v32]):.1f}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "depths": list(DEPTHS),
                   "published_arc_e": PUBLISHED_ARC_E, "seed": SEED,
                   "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} measurements, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
