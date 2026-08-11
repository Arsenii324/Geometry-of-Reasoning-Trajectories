"""A34: is D153's 29-point ARC gap closed by FEW-SHOT prompting?

D162 refuted protocol: scoring the option TEXT instead of the option letter moved accuracy
2.6 and 0.6 points against the 29.2 needed, and the three protocols agree within ~5 points
at every depth. So the gap is real, and the remaining candidates are, in order, the number
of in-context examples, the item subset, and the checkpoint.

SHOTS FIRST, because it is the only candidate with a measured precedent of the right size:
D60 found a trivial `copy` task going 15% -> 100% on prompt format alone, worth about 85
points. If ARC-Easy behaves that way, few-shot closes 29 points easily. If it moves the
number by a few points as protocol did, elicitation is not the explanation and the
checkpoint becomes the leading candidate.

Same 150 items, same three scorings, same seeded h_0 as A30, so P1 is again an identity
check.

This is also the FIRST kernel to MOUNT the prebuilt weights dataset
`bt102r0j5cb8r6r6nb36` instead of downloading from HuggingFace, which replaces the
262-282 s download with a 7.7 s mount. No `revision=` is passed -- it is baked in.

  P1  IDENTITY. The 0-shot letter arm at r = 32 must reproduce A29/A30 at 0.407, or the
      comparison across shot counts is not licensed.
  P2  PRIMARY. Does k = 5 reach the published 0.699 within +/- 0.10?
  P3  All three scorings at every k, so a PARTIAL move is visible rather than rounded to
      a verdict.
  P4  Exemplars come from the TRAIN split and are never test items.
"""
import json
import os
import subprocess
import sys
import time

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
DEPTHS = (32,)
SHOTS = (0, 2, 5)
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
    tr = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    # MOUNTED weights (REMOTE_RUNS.md): no revision= -- it is baked into the dataset.
    mnt = os.path.abspath(sys.argv[2])
    print(f"weights mount: {mnt}", flush=True)
    tok = AutoTokenizer.from_pretrained(mnt)
    cfg = AutoConfig.from_pretrained(mnt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mnt, config=cfg, trust_remote_code=True,
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

    shot_pool = []
    for rec in tr:
        q, labels, texts, gold = build(rec)
        if gold in labels and len(labels) >= 3:
            shot_pool.append((q, gold))
        if len(shot_pool) >= max(SHOTS):
            break
    print(f"{len(shot_pool)} exemplars from the TRAIN split (P4)", flush=True)

    def preamble(k):
        return "".join(f"{q}\nAnswer: {g}\n\n" for q, g in shot_pool[:k])

    def chat(p, k=0):
        return tok.apply_chat_template([{"role": "user", "content": preamble(k) + p}],
                                       tokenize=False, add_generation_prompt=True)

    def letter_pick(it, R, k=0):
        """Arm L: A29's protocol, reproduced."""
        ids = tok(chat(it["q"], k), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        with torch.no_grad():
            torch.manual_seed(SEED)
            out = model(input_ids=ids, num_steps=R)
        logits = out.logits if hasattr(out, "logits") else out[0]
        row = logits.float()[0, n_p - 1]
        opt = [tok(a, add_special_tokens=False).input_ids[0] for a in it["labels"]]
        return int(np.argmax([float(row[c]) for c in opt]))

    def text_scores(it, R, k=0):
        """Arm T: sum log P(option text | question), teacher-forced. One forward each."""
        raw, norm, ntok = [], [], []
        # the continuation is scored after the SAME chat-formatted prompt, so the only
        # difference from arm L is what is being asked for, not how it is framed
        base = chat(it["q"], k)
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
          for k in SHOTS:
            li = letter_pick(it, R, k)
            raw, norm, ntok = text_scores(it, R, k)
            rows.append({
                "id": it["id"], "depth": R, "shots": k, "n_options": len(it["labels"]),
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
    for k in SHOTS:
        v = [r for r in rows if r["shots"] == k]
        if not v:
            continue
        print(f"  shots={k} n={len(v):4d}  letter {np.mean([r['letter_correct'] for r in v]):.3f}"
              f"   text-raw {np.mean([r['text_raw_correct'] for r in v]):.3f}"
              f"   text-norm {np.mean([r['text_norm_correct'] for r in v]):.3f}"
              f"   [published {PUBLISHED_ARC_E:.3f}]", flush=True)
    v32 = [r for r in rows if r["shots"] == 0]
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
