"""Does the prompt FORMAT explain every near-zero accuracy in this project?

THE PROBLEM (claims_ledger D57)
    Huginn ships a chat template -- `<|begin_header|>`, `<|end_header|>`,
    `<|end_turn|>`, assistant role `Huginn` -- in its `tokenizer_config.json`.
    EVERY task in this project (counting, parity, running-max, three-scale, the
    Barannikov tasks, ParaRule, Caesar) was prompted as raw continuation text.

    The Caesar screen returned 0% in all six cells, and the trained model answered
    almost every prompt with the memorised pangram "The quick brown fox jumps over
    the lazy dog" -- which is exactly what a chat-formatted model does when handed
    a base-format prompt. So "Huginn cannot do task X" is not established anywhere
    in this project; what is established is "Huginn does not do task X when
    prompted as raw text".

    Until this runs, every accuracy figure here is qualified: D35's depth-scaling,
    D41(3)'s retraction, D47's 0/120, D54, D56(2).

DESIGN -- three arms on IDENTICAL items, one model, one process
    raw       exactly as every previous kernel prompted (the reproduction arm)
    chat      the same text through `tok.apply_chat_template(..., add_generation_prompt=True)`
    fewshot   raw, but preceded by two solved examples -- because Geiping et al.
              report ARC-C saturation shifting with 25-50 few-shot examples, so
              few-shot is how this model is meant to be used, and a base model
              given a zero-shot instruction is the weakest possible setup

    The reproduction arm is the control: if `raw` does not reproduce the earlier
    0%, something other than format changed and no comparison is valid.

TASKS -- the two that carry load elsewhere in the ledger
    caesar    ROT13, shift given, single word     (D57's cell `given_rot13_word`)
    counting  count the ones in 64 bits           (the task behind D35/D41/D47/D54)
    A trivially easy control (`copy`) is included so that a 0% everywhere result
    can be distinguished from a broken harness -- if the model cannot even echo a
    word back, the generation path is wrong, not the prompting.

Only the trained model is run: the untrained arm scored 0% exact under every
condition already, and format cannot rescue random weights.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run


def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)

import json
import random
import string

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 32
MAX_NEW = 20
N_ITEMS = 20
ALPHA = string.ascii_lowercase

WORDS = ["banana", "orange", "puzzle", "rocket", "silver", "meadow", "candle",
         "forest", "kitten", "planet", "guitar", "window", "yellow", "dragon",
         "pillow", "market", "summer", "castle", "bridge", "wizard"]


def rot(w, k=13):
    return "".join(chr((ord(c) - 97 + k) % 26 + 97) for c in w)


def items():
    rng = random.Random(0)
    out = {"caesar": [], "counting": [], "copy": []}
    for w in WORDS[:N_ITEMS]:
        out["caesar"].append((
            "Decode this Caesar cipher. Each letter was shifted forward by 13. "
            f"Shift each letter back by 13 to recover the original word.\nCiphertext: {rot(w)}",
            w))
        out["copy"].append((f"Repeat this word exactly.\nWord: {w}", w))
    for i in range(N_ITEMS):
        r = random.Random(1000 + i)
        bits = [1 if r.random() < 0.5 else 0 for _ in range(64)]
        out["counting"].append((
            "Count how many ones are in this sequence and reply with just the number.\n"
            "Sequence: " + " ".join(map(str, bits)), str(sum(bits))))
    _ = rng
    return out


FEWSHOT = {
    "caesar": [("Ciphertext: uryyb", "hello"), ("Ciphertext: jbeyq", "world")],
    "counting": [("Sequence: 1 0 1 1", "3"), ("Sequence: 0 0 1 0", "1")],
    "copy": [("Word: apple", "apple"), ("Word: table", "table")],
}


def render(task, body, arm, tok):
    """Build the actual model input for one arm."""
    if arm == "raw":
        return body + "\nAnswer:"
    if arm == "fewshot":
        shots = "".join(f"{q}\nAnswer: {a}\n\n" for q, a in FEWSHOT[task])
        return shots + body + "\nAnswer:"
    msgs = [{"role": "user", "content": body}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(model, tok, text):
    import torch
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    gen = []
    for _ in range(MAX_NEW):
        with torch.no_grad():
            out = model(input_ids=ids, num_steps=NUM_STEPS)
        logits = out.logits if hasattr(out, "logits") else out[0]
        nxt = int(logits[0, -1].argmax())
        gen.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        dec = tok.decode(gen)
        if "\n" in dec or "<|end_turn|>" in dec:
            break
    del ids
    torch.cuda.empty_cache()
    return tok.decode(gen).split("\n")[0].replace("<|end_turn|>", "").strip()


def score(task, pred, gold):
    p = pred.lower().strip().strip(".").strip()
    if task == "counting":
        digits = "".join(c if c.isdigit() else " " for c in p).split()
        return float(bool(digits) and digits[0] == gold)
    return float("".join(c for c in p if c in ALPHA) == gold)


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    print("chat template present:", tok.chat_template is not None, flush=True)
    data = items()
    demo = render("caesar", data["caesar"][0][0], "chat", tok)
    print(f"\nCHAT-RENDERED EXAMPLE:\n{demo!r}\n", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True).to(torch.float32).to("cuda").eval()

    results, samples = {}, {}
    for arm in ("raw", "chat", "fewshot"):
        results[arm] = {}
        for task, its in data.items():
            sc, ex = [], []
            for body, gold in its:
                pred = generate(model, tok, render(task, body, arm, tok))
                sc.append(score(task, pred, gold))
                if len(ex) < 3:
                    ex.append({"gold": gold, "pred": pred})
            results[arm][task] = {"acc": float(np.mean(sc)), "n": len(sc)}
            samples[f"{arm}/{task}"] = ex
            print(f"  {arm:>8} {task:>9}: {np.mean(sc):>6.1%}   "
                  f"e.g. {ex[0]['gold']!r} -> {ex[0]['pred']!r}", flush=True)
        with open("chatfmt.json", "w") as f:
            json.dump({"results": results, "samples": samples}, f, indent=1)

    print("\n=== VERDICT ===")
    print(f"  {'task':>9} " + " ".join(f"{a:>9}" for a in ("raw", "chat", "fewshot")))
    for task in data:
        print(f"  {task:>9} " + " ".join(
            f"{results[a][task]['acc']:>8.1%} " for a in ("raw", "chat", "fewshot")))
    copy_best = max(results[a]["copy"]["acc"] for a in results)
    if copy_best < 0.5:
        print("\n  The COPY control failed. The generation path is broken; no")
        print("  conclusion about prompting or capability may be drawn from this run.")
    else:
        best = {t: max(results[a][t]["acc"] for a in results) for t in data}
        raw = {t: results["raw"][t]["acc"] for t in data}
        moved = [t for t in data if best[t] > raw[t] + 0.10]
        if moved:
            print(f"\n  FORMAT MATTERS on: {moved}. Every accuracy figure in this")
            print("  project was measured under `raw` and must be re-read (D57(5)).")
        else:
            print("\n  Format does NOT rescue accuracy. The near-zero results stand as")
            print("  capability statements, and D57(5)'s qualification can be lifted.")


main()
