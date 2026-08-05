"""Accuracy on every task family whose GEOMETRY this project measured but whose
correctness it never did.

THE GAP (claims_ledger D56(2))
    A census of `results/*.csv` found geometry measured on EIGHT task families and
    model correctness on ONE. `switch` (parity), `maxtask` (running max),
    `projection`, `three_scale`, `three_scale_modk`, `running_count`,
    `nesting_depth` and `count_ones` all have winding and steps-to-settle rows and
    no accuracy column anywhere. So for eight of ten tasks we do not know whether
    the model can do the thing whose "reasoning trajectory" was characterised --
    and if it cannot, those trajectories are the geometry of a model FAILING,
    which is a legitimate object but not what the project set out to describe.

    D56 called this the single highest-value gap and estimated it at one
    generation pass per task. It was not run then because generation cost ~1
    minute per completion (D57/C9); `batched_generate` removed that.

DESIGN
    * CHAT FORMAT, because D60 showed a trivial copy task scores 15% under raw
      prompting and 100% under Huginn's own chat template. Every earlier accuracy
      number here was measured raw and is a lower bound. Raw is kept as one arm so
      the size of that gap is measured per task rather than assumed from `copy`.
    * A DIFFICULTY LADDER per task, because D56 showed counting is 37.5% at n=2 and
      0% from n=8 onward. A single hard setting cannot distinguish "cannot do this"
      from "cannot do this at this size", and the geometry runs used the large
      sizes. Each task is asked at n = 2, 4, 8, 16, 32.
    * THE UNTRAINED ARM, so any non-zero cell is immediately checkable against
      chance -- the control that reshaped this project four times (D40/41/48/53).
    * A `copy` control at every ladder rung, so a row of zeros can be told apart
      from a broken harness.

WHAT WOULD CHANGE THE PICTURE
    If any family is well above zero at the sizes the geometry was measured at,
    that family's geometry describes a working computation and is worth revisiting.
    If all are at zero by n=8, then D56(2)'s worry is confirmed: the project
    characterised the trajectories of a model that could not do the tasks, and the
    honest scope of every geometric result narrows accordingly.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run load_arm free_arm batched_generate assert_generation_works


def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


def load_arm(spec, cfg, revision=None):
    """Load one weight-set. `spec` is None for a fresh random init, else a repo id.

    Backfills config attributes ABSENT from an older checkpoint from the final
    model's config -- intermediate Huginn checkpoints predate fields the current
    modeling code reads (`test_time_noise`), and without this they raise
    AttributeError. Only missing keys are copied, and every backfill is logged.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    if spec is None:
        torch.manual_seed(revision if isinstance(revision, int) else 0)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    else:
        ck = AutoConfig.from_pretrained(spec, revision=revision, trust_remote_code=True)
        added = [k for k, v in vars(cfg).items()
                 if not hasattr(ck, k) and not k.startswith("_")]
        for k in added:
            setattr(ck, k, getattr(cfg, k))
        if added:
            print(f"  backfilled {len(added)} config attrs: {sorted(added)}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec, revision=revision, config=ck, trust_remote_code=True,
            low_cpu_mem_usage=True)
    return model.to(torch.float32).to("cuda").eval()


def free_arm(model, repo_id=None):
    """Release a weight-set and report what was actually reclaimed.

    Printing free memory is the point: a silent cleanup is how eight checkpoints
    were lost to OOM with 13.46 GiB still held (see `geometry-rho-direct`). Also
    purges the HF cache for `repo_id`, since ten 7GB checkpoints exhaust the disk.
    """
    import gc
    import os
    import shutil

    import torch
    try:
        model = model.to("cpu") if model is not None else None
    except Exception:                                          # noqa: BLE001, S110
        pass
    del model
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"  after cleanup: {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB",
          flush=True)
    if repo_id:
        d = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                         "models--" + repo_id.replace("/", "--"))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"  purged {d}", flush=True)


def batched_generate(model, tok, prompts, max_new=24, num_steps=32, verbose=True):
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
    if verbose:
        sizes = sorted((len(v) for v in buckets.values()), reverse=True)
        print(f"    batching {len(prompts)} prompts into {len(buckets)} length-buckets "
              f"(sizes {sizes[:6]}{'...' if len(sizes) > 6 else ''})", flush=True)

    out: list[str] = [""] * len(prompts)
    for idxs in buckets.values():
        ids = torch.stack([enc[i] for i in idxs]).to(dev)
        n_prompt = ids.shape[1]
        live = [True] * len(idxs)
        for _ in range(max_new):
            with torch.no_grad():
                res = model(input_ids=ids, num_steps=num_steps)
            logits = res.logits if hasattr(res, "logits") else res[0]
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
NUM_STEPS = 32
MAX_NEW = 12
N_ITEMS = 16
LADDER = (2, 4, 8, 16, 32)


def make(task, n, seed):
    """(question_body, gold_answer_string) for one item."""
    rng = random.Random(seed * 7919 + n)
    if task == "count_ones":
        b = [rng.randint(0, 1) for _ in range(n)]
        return (f"Count how many ones are in this sequence.\nSequence: "
                f"{' '.join(map(str, b))}", str(sum(b)))
    if task == "parity":
        b = [rng.randint(0, 1) for _ in range(n)]
        return (f"Is the number of ones in this sequence even or odd?\nSequence: "
                f"{' '.join(map(str, b))}", "even" if sum(b) % 2 == 0 else "odd")
    if task == "running_max":
        v = [rng.randint(0, 99) for _ in range(n)]
        return (f"What is the largest number in this list?\nList: "
                f"{' '.join(map(str, v))}", str(max(v)))
    if task == "running_count":
        b = [rng.randint(0, 1) for _ in range(n)]
        return (f"Start at 0. For each 1 add one, for each 0 subtract one. "
                f"What is the final total?\nSequence: {' '.join(map(str, b))}",
                str(sum(1 if x else -1 for x in b)))
    if task == "nesting_depth":
        seq, d, mx = [], 0, 0
        for _ in range(n):
            if d == 0 or (rng.random() < 0.5 and d < 8):
                seq.append("(")
                d += 1
                mx = max(mx, d)
            else:
                seq.append(")")
                d -= 1
        seq += [")"] * d
        return (f"What is the maximum nesting depth of these brackets?\nBrackets: "
                f"{' '.join(seq)}", str(mx))
    if task == "modk":
        b = [rng.randint(0, 1) for _ in range(n)]
        return (f"Count the ones in this sequence, then give the remainder when "
                f"divided by 5.\nSequence: {' '.join(map(str, b))}", str(sum(b) % 5))
    if task == "copy":
        w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        return (f"Repeat this word exactly.\nWord: {w}", w)
    raise ValueError(task)


TASKS = ("count_ones", "parity", "running_max", "running_count",
         "nesting_depth", "modk", "copy")


def render(body, arm, tok):
    if arm == "raw":
        return body + "\nAnswer:"
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True)


def score(task, pred, gold):
    p = pred.lower().strip().strip(".").strip()
    if task == "parity":
        has_e, has_o = "even" in p, "odd" in p
        return float(has_e != has_o and (has_e if gold == "even" else has_o))
    if task == "copy":
        return float("".join(c for c in p if c.isalpha()) == gold)
    digits = "".join(c if (c.isdigit() or c == "-") else " " for c in p).split()
    return float(bool(digits) and digits[0] == gold)


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    print(f"{len(TASKS)} tasks x {len(LADDER)} sizes x {N_ITEMS} items x 2 formats "
          f"x 2 arms = {len(TASKS) * len(LADDER) * N_ITEMS * 4} completions", flush=True)
    ex = make("parity", 8, 0)
    print(f"  example: {ex[0]!r} -> {ex[1]!r}", flush=True)

    out, samples = {}, {}
    for arm in ("trained", "untrained"):
        print(f"\n=== {arm} ===", flush=True)
        model = None
        try:
            model = load_arm(None if arm == "untrained" else MODEL_ID, cfg,
                             0 if arm == "untrained" else REVISION)
            if arm == "trained":
                assert_generation_works(model, tok)   # D62: never score a dead generator
            out[arm] = {}
            for fmt in ("chat", "raw"):
                out[arm][fmt] = {}
                for task in TASKS:
                    row = {}
                    for n in LADDER:
                        items = [make(task, n, s) for s in range(N_ITEMS)]
                        preds = batched_generate(
                            model, tok, [render(b, fmt, tok) for b, _ in items],
                            max_new=MAX_NEW, num_steps=NUM_STEPS, verbose=False)
                        sc = [score(task, p.split("\n")[0], g)
                              for p, (_, g) in zip(preds, items, strict=True)]
                        row[n] = float(np.mean(sc))
                        if n == LADDER[0]:
                            samples[f"{arm}/{fmt}/{task}"] = [
                                {"gold": items[i][1], "pred": preds[i].split("\n")[0][:40]}
                                for i in range(3)]
                    out[arm][fmt][task] = row
                    print(f"  {fmt:>4} {task:>14}: " + "  ".join(
                        f"n={n}:{row[n]:>5.0%}" for n in LADDER), flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        finally:
            free_arm(model, None if arm == "untrained" else MODEL_ID)
            with open("task_accuracy.json", "w") as f:
                json.dump({"acc": out, "samples": samples, "ladder": list(LADDER),
                           "n_items": N_ITEMS}, f, indent=1)

    print("\n=== DOES ANY FAMILY WORK AT THE SIZES THE GEOMETRY USED? ===")
    t = out.get("trained", {}).get("chat", {})
    u = out.get("untrained", {}).get("chat", {})
    print(f"  {'task':>14} " + " ".join(f"{'n=' + str(n):>7}" for n in LADDER)
          + "   untrained@n=8")
    for task in TASKS:
        if task in t:
            print(f"  {task:>14} " + " ".join(f"{t[task][n]:>6.0%} " for n in LADDER)
                  + f"   {u.get(task, {}).get(8, float('nan')):>6.0%}")
    copy_ok = t.get("copy", {}).get(2, 0.0)
    print()
    if copy_ok < 0.5:
        print(f"  THE COPY CONTROL FAILED ({copy_ok:.0%} at n=2). The harness is broken;")
        print("  no conclusion about capability may be drawn. This gate exists because")
        print("  D62 printed a capability verdict on a run where every output was empty.")
        return
    live = [k for k, v in t.items() if k != "copy" and v.get(8, 0) > 0.25]
    if live:
        print(f"  FAMILIES ALIVE AT n=8: {live}. Their geometry describes a working")
        print("  computation and is worth revisiting.")
    else:
        print("  NO family except `copy` clears 25% at n=8. D56(2) is confirmed: the")
        print("  geometry in this project was measured on tasks the model cannot do,")
        print("  and every geometric result must be scoped accordingly.")


main()
