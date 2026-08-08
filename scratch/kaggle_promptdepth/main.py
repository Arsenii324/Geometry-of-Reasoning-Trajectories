"""Five prompt phrasings x five recurrence depths, in Huginn's own chat format.

TWO BLIND SPOTS THIS CLOSES, both found by auditing my own kernels

  1. EVERY accuracy experiment in this project fixed `num_steps = 32`. Not one
     varied recurrent depth against CORRECTNESS. The kernels that do sweep depth
     (readout, register, eps_sweep, untrained_depth) all score geometry. So on a
     RECURRENT-DEPTH model -- whose entire premise is test-time compute scaling,
     and whose authors report GSM8K rising 9-10% at r=4 to 28-38% at r=32 -- I
     never asked whether more depth makes it more accurate. That is the single
     most obvious experiment for this architecture and it has not been run.

  2. One phrasing per task, greedy, and mostly at n=64 where the task is hopeless
     (D56: counting is 37.5% at n=2 and 0% from n=8). A single phrasing cannot
     separate "cannot do this" from "was not asked well", and D60 already showed
     an 85-point swing on `copy` from format alone.

DESIGN -- two stages, so depth is swept on a prompt that works rather than on a
straw man
  STAGE 1  5 phrasings x 3 tasks x 16 items, at r=32. Picks the best phrasing.
  STAGE 2  best phrasing per task x depths {4, 8, 16, 32, 64} x 16 items, PLUS a
           `continuous_compute` arm -- Huginn's warm-start / "continuous CoT" mode,
           which this project has never used in any kernel.

THE FIVE PHRASINGS, chosen to differ in what they ask the model to DO, not merely
in wording
  bare        the question, nothing else
  constrained "Reply with only the number." -- removes the prose-wrapper failures
              seen in D61 mode 6 ("There are 19 ones in the sequence.")
  cot         "Think step by step, then give the final answer." -- the obvious
              thing to try on a reasoning model, never tried here
  decompose   a task-specific hint at the algorithm (e.g. "go through the sequence
              one item at a time, keeping a running total")
  fewshot2    two worked examples AS PRIOR CHAT TURNS. D60's few-shot arm pasted
              examples into raw text, which is not what the template is for; this
              puts them in user/assistant turns as the model was trained to see.

SIZES ARE SMALL ON PURPOSE. n=4 for the sequence tasks, because D56 puts the model
at 37.5% at n=2 and 0% by n=8. If accuracy is zero even at n=4 under the best
phrasing at r=64, that is a much stronger negative than anything measured so far.

SCORING takes the LAST integer in the reply, not the first, because the CoT and
decompose phrasings put working before the answer. `copy` is the control at every
cell; if it fails the harness is broken and nothing else may be read.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run batched_generate assert_generation_works


def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


def batched_generate(model, tok, prompts, max_new=24, num_steps=32, verbose=True,
                     continuous_compute=False, max_batch_tokens=1024):
    """Greedy generation over length-homogeneous batches, WITHOUT the KV cache.

    WHY NOT THE MODEL'S OWN `generate_minimal`. It is batched and cache-backed and
    should be strictly better. It returned an EMPTY STRING for every prompt in
    `geometry-task-accuracy` (D62) while the naive batch-1 loop reached 100% on the
    same task and prompts (D60). Two candidate mechanisms were checked against the
    source and BOTH REFUTED: it returns a plain tensor unless `return_dict_in_generate`
    is set (it was not), and the stop check reads `next_token[i,0]` only, so the
    prompt's own `<|begin_text|>` cannot trip it. What remains untested is the
    cache+batch path itself. Rather than debug someone else's decode loop on a
    borrowed GPU, this keeps the generator that is KNOWN to work and takes the
    speedup from batching alone.

    Batching still pays: the screen ran 240 completions at ~1 min each because it
    was batch-1 (C9). Bucket sizes here are 8-16, so most of the win survives.

    ONE ARCHITECTURE-SPECIFIC TRAP, verified in the source: `forward` sets
    `prepared_attn_mask = None` -- the attention mask is commented out -- so PADDING
    IS NOT MASKED and a padded batch silently attends to pad tokens. Batching is
    therefore only safe across prompts of IDENTICAL token length, and lengths are
    measured rather than assumed (ten six-letter words through one template tokenise
    to 15 OR 16 tokens).

    `continuous_compute` warm-starts each new token's latent from the previous
    token's final latent instead of re-initialising it randomly -- Huginn's
    "continuous CoT" mode, which no kernel in this project had ever used. It needs
    `output_details` to return latents, so it is requested explicitly.

    `max_batch_tokens` caps `batch x sequence_length`, because this loop has no KV
    cache and therefore re-runs the FULL growing sequence every step. float32
    weights are ~14.1 GB of a T4's 14.56 GB, leaving ~450 MB for activations, and
    the gated MLP's inner width is 17920 -- so batch 16 x ~100 tokens OOM'd inside
    `nonlin(x_fc_1) * x_fc_2`. Buckets are split into chunks satisfying
    `chunk * (prompt_len + max_new) <= max_batch_tokens`, which keeps the peak
    bounded regardless of how long the prompts or completions are.

    Raises if every output is empty: that is the D62 symptom, and returning it
    silently is what let a full table of zeros read as a capability finding.
    """
    import torch

    dev = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    stop = {65504, 65505, 65508}                      # begin_text, end_text, end_turn
    if getattr(tok, "eos_token_id", None) is not None:
        stop.add(tok.eos_token_id)

    enc = [tok(p, return_tensors="pt", add_special_tokens=False).input_ids[0] for p in prompts]
    buckets: dict[int, list[int]] = {}
    for i, e in enumerate(enc):
        buckets.setdefault(int(e.shape[0]), []).append(i)
    # split each length-bucket so batch x seq stays inside the activation budget
    chunks: list[list[int]] = []
    for width, idxs in buckets.items():
        per = max(1, max_batch_tokens // max(width + max_new, 1))
        chunks += [idxs[k:k + per] for k in range(0, len(idxs), per)]
    if verbose:
        print(f"    {len(prompts)} prompts -> {len(buckets)} length-buckets -> "
              f"{len(chunks)} chunks (max {max(len(c) for c in chunks)} per chunk, "
              f"budget {max_batch_tokens} tok)", flush=True)

    out: list[str] = [""] * len(prompts)
    for idxs in chunks:
        ids = torch.stack([enc[i] for i in idxs]).to(dev)
        n_prompt = ids.shape[1]
        live = [True] * len(idxs)
        state = None
        for _ in range(max_new):
            kw = {"num_steps": num_steps}
            if continuous_compute:
                kw["output_details"] = {"return_logits": True, "return_latents": True,
                                        "return_head": False, "return_stats": False}
                if state is not None:
                    kw["input_states"] = state
            with torch.no_grad():
                res = model(input_ids=ids, **kw)
            logits = res.logits if hasattr(res, "logits") else res[0]
            if continuous_compute:
                lat = getattr(res, "latent_states", None)
                state = lat[:, -1:, :].clone() if lat is not None else None
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            for b in range(len(idxs)):
                if int(nxt[b, 0]) in stop:
                    live[b] = False
            ids = torch.cat([ids, nxt], dim=1)
            if not any(live):
                break
        for b, i in enumerate(idxs):
            out[i] = tok.decode(ids[b, n_prompt:], skip_special_tokens=True)
        del ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not any(o.strip() for o in out):
        raise RuntimeError(
            f"batched_generate produced empty output for all {len(prompts)} prompts. "
            "This is the D62 failure mode; refusing to return silently.")
    return out


def assert_generation_works(model, tok, chat=True):
    """Smoke-test the generator against a task the model provably does, before use.

    D62: `batched_generate` returned an empty string for every prompt and the whole
    task-accuracy run was scored on it. Its unit tests passed -- they asserted the
    length-bucketing the author designed and never that the function returns correct
    text. A block that talks to a real model needs a real check against that model.

    `copy` is used because D60 measured it at 100% under the chat template, so a
    failure here is unambiguous. Raises rather than warns: a silent generator is
    exactly what produced a full run of zeros that read as a scientific result.
    """
    words = ["banana", "orange", "puzzle", "kitten"]
    bodies = [f"Repeat this word exactly.\nWord: {w}" for w in words]
    if chat:
        texts = [tok.apply_chat_template([{"role": "user", "content": b}],
                                         tokenize=False, add_generation_prompt=True)
                 for b in bodies]
    else:
        texts = [b + "\nAnswer:" for b in bodies]
    got = batched_generate(model, tok, texts, max_new=8, verbose=False)
    hits = sum(w in g.lower() for w, g in zip(words, got, strict=True))
    print(f"  generation smoke-test: {hits}/{len(words)} copied  -> {got}", flush=True)
    if hits < len(words) // 2:
        raise RuntimeError(
            f"generation smoke-test FAILED ({hits}/{len(words)}); got {got}. "
            "Refusing to run an experiment on a generator that cannot copy a word.")
    return hits

import json
import random

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
N_ITEMS = 16
N_SEQ = 4                      # sequence length for the counting/parity tasks
DEPTHS = (4, 8, 16, 32, 64)
PICK_DEPTH = 32


def item(task, seed):
    rng = random.Random(seed * 7919 + hash(task) % 1000)
    if task == "count_ones":
        b = [rng.randint(0, 1) for _ in range(N_SEQ)]
        return " ".join(map(str, b)), str(sum(b))
    if task == "parity":
        b = [rng.randint(0, 1) for _ in range(N_SEQ)]
        return " ".join(map(str, b)), "even" if sum(b) % 2 == 0 else "odd"
    w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
    return w, w


QUESTION = {
    "count_ones": "Count how many ones are in this sequence.\nSequence: {x}",
    "parity": "Is the number of ones in this sequence even or odd?\nSequence: {x}",
    "copy": "Repeat this word exactly.\nWord: {x}",
}
SUFFIX = {
    "bare": "",
    "constrained": "\nReply with only the answer, nothing else.",
    "cot": "\nThink step by step, then give the final answer on the last line.",
    "decompose": {
        "count_ones": "\nGo through the sequence one item at a time, keeping a "
                      "running total, then state the total.",
        "parity": "\nGo through the sequence one item at a time, flipping between "
                  "even and odd for each 1, then state the result.",
        "copy": "\nOutput the word and nothing else.",
    },
}
SHOTS = {
    "count_ones": [("Count how many ones are in this sequence.\nSequence: 1 0 1 1", "3"),
                   ("Count how many ones are in this sequence.\nSequence: 0 0 1 0", "1")],
    "parity": [("Is the number of ones in this sequence even or odd?\nSequence: 1 0 1 1",
                "odd"),
               ("Is the number of ones in this sequence even or odd?\nSequence: 0 0 1 0",
                "odd")],
    "copy": [("Repeat this word exactly.\nWord: apple", "apple"),
             ("Repeat this word exactly.\nWord: table", "table")],
}
PHRASINGS = ("bare", "constrained", "cot", "decompose", "fewshot2")


def render(task, x, phrasing, tok):
    q = QUESTION[task].format(x=x)
    if phrasing == "fewshot2":
        msgs = []
        for u, a in SHOTS[task]:
            msgs += [{"role": "user", "content": u},
                     {"role": "Huginn", "content": a}]
        msgs.append({"role": "user", "content": q})
    else:
        suf = SUFFIX[phrasing]
        if isinstance(suf, dict):
            suf = suf[task]
        msgs = [{"role": "user", "content": q + suf}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def score(task, pred, gold):
    p = pred.lower()
    if task == "parity":
        toks = [w for w in p.replace(".", " ").split() if w in ("even", "odd")]
        return float(bool(toks) and toks[-1] == gold)
    if task == "copy":
        letters = "".join(c for c in p if c.isalpha())
        return float(gold in letters)
    nums = []
    cur = ""
    for c in p:
        if c.isdigit():
            cur += c
        else:
            if cur:
                nums.append(cur)
            cur = ""
    if cur:
        nums.append(cur)
    return float(bool(nums) and nums[-1] == gold)      # LAST number: CoT puts it last


def run_cell(model, tok, task, phrasing, depth, max_new, cc=False):
    import numpy as np
    items = [item(task, s) for s in range(N_ITEMS)]
    texts = [render(task, x, phrasing, tok) for x, _ in items]
    preds = batched_generate(model, tok, texts, max_new=max_new, num_steps=depth,
                             verbose=False, continuous_compute=cc)
    sc = [score(task, p, g) for p, (_, g) in zip(preds, items, strict=True)]
    return float(np.mean(sc)), preds[:2], [g for _, g in items[:2]]


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True).to(torch.float32).to("cuda").eval()
    assert_generation_works(model, tok)

    print("\n=== example renderings ===")
    for ph in PHRASINGS:
        t = render("count_ones", "0 1 0 1", ph, tok)
        print(f"  {ph:>12}: {t[-118:]!r}", flush=True)

    out = {"stage1": {}, "stage2": {}}
    print("\n=== STAGE 1: which phrasing? (r=32, n=4) ===")
    print(f"  {'task':>11} " + " ".join(f"{p:>12}" for p in PHRASINGS))
    best = {}
    for task in QUESTION:
        row = {}
        for ph in PHRASINGS:
            mx = 64 if ph in ("cot", "decompose") else 12
            acc, ex, gold = run_cell(model, tok, task, ph, PICK_DEPTH, mx)
            row[ph] = acc
            out["stage1"].setdefault(task, {})[ph] = {
                "acc": acc, "sample": [{"gold": g, "pred": p[:70]}
                                       for g, p in zip(gold, ex, strict=True)]}
        best[task] = max(row, key=row.get)
        print(f"  {task:>11} " + " ".join(f"{row[p]:>11.0%} " for p in PHRASINGS)
              + f"  best={best[task]}", flush=True)
        with open("prompt_depth.json", "w") as f:
            json.dump({"out": out, "best": best}, f, indent=1)

    print("\n=== STAGE 2: does DEPTH help, on the best phrasing? ===")
    print(f"  {'task':>11} {'phrasing':>12} " + " ".join(f"r={d:<5}" for d in DEPTHS)
          + "  cc@32")
    for task in QUESTION:
        ph = best[task]
        mx = 64 if ph in ("cot", "decompose") else 12
        row = {}
        for d in DEPTHS:
            row[d], _, _ = run_cell(model, tok, task, ph, d, mx)
        cc, _, _ = run_cell(model, tok, task, ph, PICK_DEPTH, mx, cc=True)
        out["stage2"][task] = {"phrasing": ph, "by_depth": row, "continuous": cc}
        print(f"  {task:>11} {ph:>12} " + " ".join(f"{row[d]:>6.0%} " for d in DEPTHS)
              + f"  {cc:>5.0%}", flush=True)
        with open("prompt_depth.json", "w") as f:
            json.dump({"out": out, "best": best}, f, indent=1)

    print("\n=== VERDICT ===")
    ctrl = out["stage2"].get("copy", {}).get("by_depth", {})
    if ctrl and max(ctrl.values()) < 0.5:
        print("  COPY CONTROL FAILED -- harness broken, nothing else may be read.")
        return
    from scipy.stats import spearmanr
    for task in ("count_ones", "parity"):
        d = out["stage2"].get(task, {}).get("by_depth", {})
        if len(d) >= 4 and max(d.values()) > 0:
            rho, p = spearmanr(list(d), list(d.values()))
            print(f"  {task}: spearman(depth, accuracy) = {rho:+.3f}, p={p:.3f}"
                  f"   range {min(d.values()):.0%}..{max(d.values()):.0%}")
        else:
            print(f"  {task}: zero at every depth under the best phrasing "
                  f"({out['stage2'].get(task, {}).get('phrasing')}) -- "
                  "test-time compute does not rescue it at n=4.")


main()
