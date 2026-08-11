"""A40: score ARC-Easy the way the HARNESS does. The gap is the only external calibration.

WHY THIS IS THE MOST IMPORTANT REMAINING RUN. After 166 rows this project cannot reproduce a
single published number about the model it studies. Huginn's paper reports **0.699** on
ARC-Easy at r = 32. D153 measured **0.407** by option-letter argmax; D162 added the option-TEXT
likelihood arm and got **0.433 raw / 0.413 length-normalised**, concluded that the three
protocols agree with each other within ~5 points, and declared **"protocol is refuted as the
explanation."**

**That conclusion is too strong, and the reason is visible in the kernels rather than in the
numbers.** All three ARC runs -- `ds_arcrepro`, `ds_arcproto`, `ds_arcshots` -- call
`apply_chat_template(..., add_generation_prompt=True)`. Every arm D162 compared was
chat-templated. The standard `lm-evaluation-harness` `arc_easy` task, which is what a reported
0.699 almost certainly comes from, is **not**:

    harness:  "Question: {question}\\nAnswer:"  +  " {choice_text}"      # bare, no options listed
    ours:     chat("Question: {q}\\nA. ...\\nB. ...\\nAnswer with the letter ...")

So our runs deviate from the harness on **three axes at once** and D162 varied only one:

    (i)   chat template vs bare completion
    (ii)  the option list present in the prompt vs absent
    (iii) length normalisation by TOKEN count (ours) vs by CHARACTER count (harness acc_norm)

And D166 has just shown that this model is extremely format-sensitive: stripping the chat
template moved the gold's median rank from 2.5 to 10, with one item at 509. A format axis that
large cannot be assumed away.

THE DESIGN. Four arms on the SAME 120 items at r = 32, so every comparison is within one run:

    L-chat   A29/D153's letter argmax, chat-templated              -- replication gate
    T-chat   A30/D162's option-text likelihood, chat-templated     -- replication gate
    H-chat   harness prompt and target, but chat-templated         -- isolates (ii)
    H-bare   the harness protocol as written                       -- isolates (i)

`acc` (raw summed loglikelihood) and both normalisations are reported for every likelihood arm,
which isolates (iii) at no extra forward passes.

PREREGISTERED PREDICTIONS:

  P1  REPLICATION GATE, VOID WITHOUT IT. `L-chat` must land near D153's **0.407** and `T-chat`
      near D162's **0.433 / 0.413** at r = 32. Different item sample (first 120 vs first 150 of
      the same split, same builder), so exact equality is not expected; more than ~8 points off
      means this run is not measuring what those did and nothing else may be read.

  P2  PRIMARY. Does `H-bare` reach the published **0.699**? Registered before running:
      **within 0.10 of 0.699** means the gap was protocol after all, D162's headline
      ("protocol is refuted") must be amended to "protocol, but not the axis we varied", and
      this project finally has a validated external anchor. **Still near 0.43** means the
      published number does not reproduce under the standard protocol either, and that is a
      strong, defensible negative rather than an open question.

  P3  WHICH AXIS, IF ANY, CARRIES IT. `H-chat` sits between the two: harness prompt, our
      framing. If `H-bare` > `H-chat` ~ `T-chat`, the chat template is the culprit (axis i). If
      `H-chat` ~ `H-bare` > `T-chat`, the option list in the prompt is (axis ii). If all three
      agree, neither is, and axis (iii) is all that is left.

  P4  THE NORMALISATION IS FREE, SO MEASURE IT. Character-normalised and token-normalised
      accuracy are computed from the same forwards. ARC is usually reported as `acc_norm`, and
      D162 used tokens where the harness uses characters.

  P5  DEGENERACY CHECK, BECAUSE D110. A likelihood arm that picks the same option index for
      nearly every item is broken, not accurate. The distribution of chosen indices is printed
      per arm; a mode above ~60% is flagged in the output.
"""

import json
import os
import subprocess
import sys
import time

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
DEPTH = 32                     # the published setting
N_ITEMS = 120
SEED = 20260810                # A29/A30's seed
PUBLISHED_ARC_E = 0.699
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 12000


def build_items(rng=None):
    """Shape-only stub for preflight; the real items need the dataset (built in main)."""
    return [{"q": "", "labels": [], "texts": [], "gold": "", "id": ""}]


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("archarness.json")
    mnt = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"results -> {out_path}; weights mount -> {mnt}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model] datasets")
    sys.path.insert(0, os.path.abspath("src"))

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    src = mnt or MODEL_ID
    kw = {} if mnt else {"revision": REVISION}
    tok = AutoTokenizer.from_pretrained(src, **kw)
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        src, config=cfg, trust_remote_code=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, **kw).to("cuda").eval()

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

    # A29's item construction, reproduced verbatim so P1 is a real replication gate
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

    def letter_pick(it):
        ids = tok(chat(it["q"]), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        with torch.no_grad():
            torch.manual_seed(SEED)
            out = model(input_ids=ids, num_steps=DEPTH)
        logits = out.logits if hasattr(out, "logits") else out[0]
        row = logits.float()[0, n_p - 1]
        opt = [tok(a, add_special_tokens=False).input_ids[0] for a in it["labels"]]
        return int(np.argmax([float(row[c]) for c in opt]))

    def score_continuations(base, conts):
        """sum log P(cont | base), teacher-forced, one forward per continuation.

        Returns (raw, per_token, per_char). The three differ only in the denominator, so
        computing all of them costs nothing and settles axis (iii) for free.
        """
        raw, ptok, pchar = [], [], []
        bids = tok(base, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)
        k = bids.shape[1]
        for txt in conts:
            fids = tok(base + txt, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(model.device)
            if fids.shape[1] <= k:
                raw.append(-1e9); ptok.append(-1e9); pchar.append(-1e9); continue
            with torch.no_grad():
                torch.manual_seed(SEED)
                out = model(input_ids=fids, num_steps=DEPTH)
            logits = out.logits if hasattr(out, "logits") else out[0]
            lp = torch.log_softmax(logits.float()[0], dim=-1)
            tgt = fids[0, k:]
            s = float(sum(lp[k + i - 1, int(tgt[i])] for i in range(tgt.numel())))
            raw.append(s)
            ptok.append(s / int(tgt.numel()))
            pchar.append(s / max(1, len(txt)))
        return raw, ptok, pchar

    t0, rows = time.time(), []
    for n, it in enumerate(items):
        gi = it["labels"].index(it["gold"])
        try:
            li = letter_pick(it)
            # T-chat: A30's arm -- our prompt (options listed), chat-templated
            t_raw, t_tok, t_chr = score_continuations(chat(it["q"]) + " ",
                                                      it["texts"])
            # harness prompt: no option list, "Answer:" then " {text}"
            hb = f"Question: {it['question']}\nAnswer:"
            h_raw, h_tok, h_chr = score_continuations(hb, [" " + t for t in it["texts"]])
            hc = chat(f"Question: {it['question']}\nAnswer:")
            c_raw, c_tok, c_chr = score_continuations(hc, [" " + t for t in it["texts"]])
            row = {"id": it["id"], "gold_idx": gi, "n_options": len(it["labels"]), "ok": True,
                   "L_chat": int(li == gi),
                   "T_chat_raw": int(int(np.argmax(t_raw)) == gi),
                   "T_chat_tok": int(int(np.argmax(t_tok)) == gi),
                   "T_chat_chr": int(int(np.argmax(t_chr)) == gi),
                   "H_bare_raw": int(int(np.argmax(h_raw)) == gi),
                   "H_bare_tok": int(int(np.argmax(h_tok)) == gi),
                   "H_bare_chr": int(int(np.argmax(h_chr)) == gi),
                   "H_chat_raw": int(int(np.argmax(c_raw)) == gi),
                   "H_chat_tok": int(int(np.argmax(c_tok)) == gi),
                   "H_chat_chr": int(int(np.argmax(c_chr)) == gi),
                   "pick_L": int(li),
                   "pick_T_chat_chr": int(np.argmax(t_chr)),
                   "pick_H_bare_chr": int(np.argmax(h_chr)),
                   "pick_H_chat_chr": int(np.argmax(c_chr))}
        except Exception as exc:  # noqa: BLE001
            row = {"id": it["id"], "ok": False, "why": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        if n % 10 == 0:
            print(f"  {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok = [r for r in rows if r.get("ok")]
    print(f"\n=== ACCURACY at r={DEPTH}, n={len(ok)} items, published {PUBLISHED_ARC_E} ===",
          flush=True)
    arms = ["L_chat", "T_chat_raw", "T_chat_tok", "T_chat_chr",
            "H_chat_raw", "H_chat_tok", "H_chat_chr",
            "H_bare_raw", "H_bare_tok", "H_bare_chr"]
    acc = {}
    for a in arms:
        v = [r[a] for r in ok if a in r]
        acc[a] = sum(v) / len(v) if v else float("nan")
        gate = ""
        if a == "L_chat":
            gate = "   <- P1 gate vs D153 0.407"
        elif a in ("T_chat_raw", "T_chat_tok"):
            gate = "   <- P1 gate vs D162 0.433 / 0.413"
        elif a == "H_bare_chr":
            gate = "   <- P2 PRIMARY: the harness's own acc_norm"
        print(f"  {a:12s} {acc[a]:.3f}   d(published) {acc[a] - PUBLISHED_ARC_E:+.3f}{gate}",
              flush=True)

    print("\n=== P2 VERDICT ===", flush=True)
    prim = acc.get("H_bare_chr", float("nan"))
    if abs(prim - PUBLISHED_ARC_E) <= 0.10:
        print(f"  H_bare_chr {prim:.3f} is WITHIN 0.10 of {PUBLISHED_ARC_E}: the gap WAS "
              f"protocol, on an axis D162 did not vary. D162's headline needs amending.",
              flush=True)
    else:
        print(f"  H_bare_chr {prim:.3f} is {abs(prim - PUBLISHED_ARC_E):.3f} from "
              f"{PUBLISHED_ARC_E}: the published number does NOT reproduce under the "
              f"standard protocol either.", flush=True)

    print("\n=== P3 WHICH AXIS ===", flush=True)
    print(f"  template  (H_bare_chr {acc.get('H_bare_chr', float('nan')):.3f} vs H_chat_chr "
          f"{acc.get('H_chat_chr', float('nan')):.3f}): "
          f"{acc.get('H_bare_chr', 0) - acc.get('H_chat_chr', 0):+.3f}", flush=True)
    print(f"  option list (H_chat_chr {acc.get('H_chat_chr', float('nan')):.3f} vs T_chat_chr "
          f"{acc.get('T_chat_chr', float('nan')):.3f}): "
          f"{acc.get('H_chat_chr', 0) - acc.get('T_chat_chr', 0):+.3f}", flush=True)
    print(f"  normalisation (chr vs tok, H_bare): "
          f"{acc.get('H_bare_chr', 0) - acc.get('H_bare_tok', 0):+.3f}", flush=True)

    print("\n=== P5 DEGENERACY ===", flush=True)
    import collections
    for p in ("pick_L", "pick_T_chat_chr", "pick_H_bare_chr", "pick_H_chat_chr"):
        c = collections.Counter(r[p] for r in ok if p in r)
        if c:
            top, cnt = c.most_common(1)[0]
            flag = "  <-- DEGENERATE" if cnt / len(ok) > 0.6 else ""
            print(f"  {p:18s} {dict(sorted(c.items()))}  mode {top} at "
                  f"{cnt / len(ok):.2f}{flag}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "depth": DEPTH, "n_items": N_ITEMS, "seed": SEED,
                   "published": PUBLISHED_ARC_E, "acc": acc,
                   "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} items, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
