"""WHERE in its own output does the model say the answer, and does depth move it?

Writes `answerpos.json` to /kaggle/working. Joins `geometry-depthacc` on
(family, item, fmt, depth), so the two banks are directly comparable.

THE QUESTION THIS SETTLES. `scripts/run_depth_profile.py` establishes three things
off banked data: fixed-depth top-1 accuracy at the answer position PEAKS AT r=5
(16.1%) and collapses to a flat 6.2% by r=8 (family-paired sign test, 14 of 15
discordant families, p = 9.8e-4); the answer is nonetheless MORE present in the
generated text at depth 32 than at depth 2 (containment 21.4% -> 33.3%) and far more
SPECIFIC (a length-matched decoy -- the gold of another item in the same family --
falls 15.1% -> 2.7%, so specificity rises 1.41x -> 12.35x); and the converged rank is
identical across ten unseeded h_0 draws in 7 of 8 prompts, so h_0 is forgotten by
convergence and only moves the transient.

Those three cannot all be true unless DEPTH RELOCATES THE ANSWER rather than
destroying it. But "relocates" is an inference, because every rank in this project is
measured at ONE position -- the first answer token. This run measures the rank of
gold at EVERY position the model generates, so the relocation is either seen or it is
not.

THE PROPOSED MECHANISM WAS ALREADY REFUTED ONCE and must not be smuggled back in.
"Depth prepends conversational framing and pushes gold out of first place" predicts
that, at a fixed depth, generations opening with a framing word show a worse gold
rank than those that do not. Measured: at depth 32 the median rank is 7.0 with
framing and 7.5 without. The word-list proxy is not the mechanism, and this run
replaces it with the position itself rather than a proxy for the position.

PRE-REGISTERED (CLAUDE.md section 1). Fixed before the run, from the banked
containment curve, which is a different instrument on different data.
Q1 -- GATE. `assert_generation_works` must pass, else nothing below is read (D62
      printed a capability verdict over empty strings for want of exactly this).
Q2 -- PRIMARY. `first_top1_pos`, the first generated position where gold is rank 1.
      Prediction: it is 0 at depth 2-4 and > 0 at depth 16-32, tested family-paired
      with an exact sign test on the median position per family.
Q3 -- PRIMARY. `any_top1`, whether gold is rank 1 at ANY generated position.
      Prediction: RISES with depth, tracking containment (21.4% -> 33.3%). This is
      the claim that depth helps once you stop scoring only the first token.
Q4 -- CONTROL. The same two statistics for the DISTRACTOR. Both must stay flat or
      fall; a distractor that also gains would mean the readout drifts with output
      length rather than tracking the answer.
Q5 -- CONTROL. `first_top1_pos` is undefined when gold is never top-1; it is
      recorded as -1 and Q2 is computed only over cells where BOTH depths resolve,
      so the paired test cannot be carried by differential missingness.

Banks the full per-position rank vector for gold and distractor, so any scoring
decision can be redone offline without a GPU (B14).
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run load_arm free_arm batched_generate assert_generation_works preflight


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


def attainable(alpha, n_perm):
    """Can a permutation test with `n_perm` draws ever reach `alpha`?

    The smallest p a permutation test can report is 1/(n_perm+1). If that floor
    sits above the significance threshold, REJECTION IS ARITHMETICALLY
    IMPOSSIBLE and the run returns "not significant" for every cell no matter
    what the data say -- a guaranteed null that reads like a scientific result.

    Measured instance: `geometry-correctness` was drafted with n_perm=200 against
    a Bonferroni alpha of 0.05/12 = 0.00417. Floor = 1/201 = 0.00498 > alpha, and
    synthetic power was 0.00 even for a 2 sd shift. Same class as the winding
    null's p=0.024 floor at n=40.

    Returns (ok, floor).
    """
    floor = 1.0 / (n_perm + 1)
    return floor < alpha, floor


def degenerate(decoded, min_distinct=2):
    """Is an argmax population degenerate rather than informative?

    Two failure shapes, both seen on real runs:
      * COLLAPSE -- every item predicts the same token, so the measurement
        carries no per-item information.
      * UNPRINTABLE -- the argmax decodes to a partial UTF-8 byte fragment
        (U+FFFD after decode), which means the distribution is not on words at
        all. `geometry-discourse`'s prefill arm did BOTH: token ids 6704/7909/
        12894 ('ä¸') for 24/24 items on every task, and the kernel still printed
        a confident verdict from that arm.

    Returns (is_degenerate, reason).
    """
    uniq = set(decoded)
    bad = sum("�" in d for d in decoded)
    if bad > len(decoded) // 2:
        return True, f"{bad}/{len(decoded)} argmax tokens are unprintable byte fragments"
    if len(uniq) < min_distinct:
        return True, f"argmax collapsed to {len(uniq)} distinct token(s): {sorted(uniq)[:3]}"
    return False, ""


def gated_verdict(claim, passed, gates):
    """Print a conclusion ONLY if every precondition holds; else say why not.

    D62: a capability verdict printed over empty strings because the analysis
    excluded the control that would have caught it. `geometry-discourse` repeated
    it -- P3/P4 keyed on one arm and printed the OPPOSITE of the right answer
    without ever checking that arm's output was sane.

    `gates` is a list of (name, ok, detail). A verdict computed from the same
    variables as the run will agree with the run's mistakes, so the gates must
    test the INSTRUMENT, not the hypothesis.

    Returns the verdict string, and prints it.
    """
    failed = [(n, d) for n, ok, d in gates if not ok]
    if failed:
        msg = (f"  VERDICT WITHHELD -- {claim}\n"
               + "\n".join(f"    gate FAILED: {n} -- {d}" for n, d in failed)
               + "\n    the instrument did not qualify; this arm may not be read.")
    else:
        msg = f"  {claim}: {'CONFIRMED' if passed else 'REFUTED'}"
    print(msg, flush=True)
    return msg

import json
import os
import random
import string
import time
import traceback
import zlib

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
N_ITEMS = 6             # the depthacc prefix exactly, so the two banks join
MAX_NEW = 16            # depthacc used 8; the framing openers alone eat 3-4 tokens
# Ordered by decisiveness: Kaggle kills at wall clock with whatever is written, and
# every depth is persisted as it completes. The prediction lives in the 2-vs-32
# contrast, so those two run first and the rest interpolate.
DEPTHS = (2, 32, 8, 16, 4)
OUTDIR = "/kaggle/working"

CONSTRAINT = "Reply with only the answer."

WORDS = ('apple', 'chair', 'river', 'stone', 'bread', 'cloud', 'green', 'horse', 'light', 'money', 'night', 'paper', 'queen', 'table', 'water', 'youth')

TASKS = ('echo_digit', 'echo_word', 'nth_item', 'last_item', 'add1', 'sub1', 'add_2d', 'compare', 'count4', 'count8', 'count16', 'count_mod3', 'parity8', 'track_total', 'local_last', 'max_run', 'sort_min', 'succ_letter', 'caesar1_letter', 'caesar1_word', 'rot13_word')

def items(task, n=N_ITEMS):
    """(prompt, gold, distractor) triples. Gold is single-token where possible.

    `zlib.crc32`, NOT `hash()`: Python salts str hashes per process, so a kernel
    seeded with `hash(task)` draws a different item set on every run (D69(3)).
    """
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + zlib.crc32(task.encode()) % 997)
        L = string.ascii_lowercase
        if task == 'echo_digit':
            v = rng.randint(0, 9)
            out.append((f'Repeat this number exactly.\nNumber: {v}', str(v), str((v + 3) % 10)))
        elif task == 'echo_word':
            w = rng.choice(WORDS)
            out.append((f'Repeat this word exactly.\nWord: {w}', w, rng.choice([x for x in WORDS if x != w])))
        elif task == 'nth_item':
            xs = [rng.randint(0, 9) for _ in range(5)]
            k = rng.randint(1, 5)
            out.append((f"What is item number {k} in this list?\nList: {' '.join(map(str, xs))}", str(xs[k - 1]), str((xs[k - 1] + 3) % 10)))
        elif task == 'last_item':
            xs = [rng.randint(0, 9) for _ in range(6)]
            out.append((f"What is the last number in this list?\nList: {' '.join(map(str, xs))}", str(xs[-1]), str((xs[-1] + 3) % 10)))
        elif task == 'add1':
            v = rng.randint(0, 8)
            out.append((f'What is {v} + 1?', str(v + 1), str((v + 4) % 10)))
        elif task == 'sub1':
            v = rng.randint(1, 9)
            out.append((f'What is {v} - 1?', str(v - 1), str((v + 4) % 10)))
        elif task == 'add_2d':
            a, b = (rng.randint(10, 49), rng.randint(10, 49))
            out.append((f'What is {a} + {b}?', str(a + b), str(a + b + 3)))
        elif task == 'compare':
            a, b = rng.sample(range(1, 100), 2)
            out.append((f'Which number is larger, {a} or {b}?', str(max(a, b)), str(min(a, b))))
        elif task == 'count4':
            b = [rng.randint(0, 1) for _ in range(4)]
            g = sum(b)
            out.append((f"Count how many ones are in this sequence.\nSequence: {' '.join(map(str, b))}", str(g), str((g + 2) % 5)))
        elif task == 'count8':
            b = [rng.randint(0, 1) for _ in range(8)]
            g = sum(b)
            out.append((f"Count how many ones are in this sequence.\nSequence: {' '.join(map(str, b))}", str(g), str((g + 3) % 9)))
        elif task == 'count16':
            b = [rng.randint(0, 1) for _ in range(16)]
            g = sum(b)
            out.append((f"Count how many ones are in this sequence.\nSequence: {' '.join(map(str, b))}", str(g), str((g + 3) % 17)))
        elif task == 'count_mod3':
            b = [rng.randint(0, 1) for _ in range(9)]
            g = sum(b) % 3
            out.append((f"Count how many ones are in this sequence, then give the remainder when divided by 3.\nSequence: {' '.join(map(str, b))}", str(g), str((g + 1) % 3)))
        elif task == 'parity8':
            b = [rng.randint(0, 1) for _ in range(8)]
            g = sum(b) % 2
            out.append((f"Is the number of ones in this sequence even or odd? Answer 0 for even and 1 for odd.\nSequence: {' '.join(map(str, b))}", str(g), str(1 - g)))
        elif task == 'track_total':
            ops = [rng.choice([1, -1]) for _ in range(8)]
            g = sum(ops)
            body = 'Start at 0. ' + ' '.join(('Add 1.' if o > 0 else 'Subtract 1.' for o in ops))
            out.append((body + ' Final total?', str(g), str(g + 2)))
        elif task == 'local_last':
            ops = [rng.choice([1, -1]) for _ in range(8)]
            g = 'Add' if ops[-1] > 0 else 'Subtract'
            body = 'Start at 0. ' + ' '.join(('Add 1.' if o > 0 else 'Subtract 1.' for o in ops))
            out.append((body + ' What was the last instruction?', g, 'Subtract' if g == 'Add' else 'Add'))
        elif task == 'max_run':
            b = [rng.randint(0, 1) for _ in range(12)]
            best = cur = 1
            for i in range(1, len(b)):
                cur = cur + 1 if b[i] == b[i - 1] else 1
                best = max(best, cur)
            out.append((f"What is the length of the longest run of identical symbols in this sequence?\nSequence: {' '.join(map(str, b))}", str(best), str(best + 1)))
        elif task == 'sort_min':
            xs = rng.sample(range(1, 100), 3)
            out.append((f'What is the smallest of these numbers: {xs[0]}, {xs[1]}, {xs[2]}?', str(min(xs)), str(sorted(xs)[1])))
        elif task == 'succ_letter':
            c = rng.choice(L[:25])
            out.append((f"What letter comes after '{c}' in the alphabet?", L[L.index(c) + 1], L[(L.index(c) + 5) % 26]))
        elif task == 'caesar1_letter':
            c = rng.choice(L[:25])
            out.append((f'Shift this letter forward by 1 in the alphabet.\nLetter: {c}', L[L.index(c) + 1], L[(L.index(c) + 7) % 26]))
        elif task == 'caesar1_word':
            w = ''.join((rng.choice(L) for _ in range(4)))
            enc = ''.join((L[(L.index(c) + 1) % 26] for c in w))
            out.append((f'Shift each letter of this text backward by 1 in the alphabet.\nText: {enc}', w, ''.join((rng.choice(L) for _ in range(4)))))
        else:
            w = ''.join((rng.choice(L) for _ in range(4)))
            enc = ''.join((L[(L.index(c) + 13) % 26] for c in w))
            out.append((f'Decode this ROT13 text.\nText: {enc}', w, ''.join((rng.choice(L) for _ in range(4)))))
    return out


def rank_walk(model, tok, torch, text, gold, dist, depth, max_new=MAX_NEW):
    """Greedy-decode `max_new` tokens; at every step record gold's and the
    distractor's rank in that step's distribution.

    The model's OWN continuation is what gets walked -- not a teacher-forced gold
    prefix. The question is where the model chooses to put the answer, so feeding it
    the answer would answer a different question.

    One `num_steps=depth` forward per token, no KV cache reuse across steps: the
    cache is indexed by `block_idx` and is rebuilt anyway when `num_steps` changes,
    and MAX_NEW=16 at N_ITEMS=6 is small enough that correctness beats speed here.
    """
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device)
    g0 = tok(gold, add_special_tokens=False).input_ids[0]
    d0 = tok(dist, add_special_tokens=False).input_ids[0]
    g_ranks, d_ranks, toks = [], [], []
    with torch.no_grad():
        for _ in range(max_new):
            row = torch.log_softmax(
                model(input_ids=ids, num_steps=depth).logits[0, -1].float(), dim=-1)
            g_ranks.append(int((row > row[g0]).sum().item()) + 1)
            d_ranks.append(int((row > row[d0]).sum().item()) + 1)
            nxt = int(torch.argmax(row).item())
            toks.append(nxt)
            if nxt == tok.eos_token_id:
                break
            ids = torch.cat(
                [ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    def first1(rs):
        # Q5: -1 encodes "never top-1", and the analysis drops such cells pairwise
        # rather than imputing them, so missingness cannot carry the paired test.
        return next((i for i, r in enumerate(rs) if r == 1), -1)
    return {"gold_ranks": g_ranks, "dist_ranks": d_ranks,
            "text_out": tok.decode(toks, skip_special_tokens=True),
            "first_top1_pos": first1(g_ranks), "any_top1": bool(min(g_ranks) == 1),
            "dist_first_top1_pos": first1(d_ranks),
            "dist_any_top1": bool(min(d_ranks) == 1)}


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoConfig, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
          flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION,
                                     trust_remote_code=True)
    model = load_arm(MODEL_ID, cfg, REVISION)

    assert_generation_works(model, tok, chat=True)      # Q1
    print("Q1: generation gate PASSED", flush=True)

    def build(prompt, fmt):
        body = prompt if fmt == "bare" else prompt + "\n" + CONSTRAINT
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)

    cells = []
    for fam in TASKS:
        for i, (prompt, gold, dist) in enumerate(items(fam)):
            for fmt in ("bare", "constrained"):
                cells.append({"family": fam, "item": i, "fmt": fmt, "gold": gold,
                              "distractor": dist, "text": build(prompt, fmt)})
    print(f"{len(cells)} cells x {len(DEPTHS)} depths", flush=True)

    out, t0 = [], time.time()
    for depth in DEPTHS:
        for c in cells:
            rec = {"family": c["family"], "item": c["item"], "fmt": c["fmt"],
                   "gold": c["gold"], "distractor": c["distractor"],
                   "depth": depth, "ok": True}
            try:
                rec.update(rank_walk(model, tok, torch, c["text"], c["gold"],
                                     c["distractor"], depth))
            except Exception as exc:
                rec.update({"ok": False, "why": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()})
                if sum(not r["ok"] for r in out) < 3:
                    print(f"  cell failed: {exc}", flush=True)
            out.append(rec)
        with open(os.path.join(OUTDIR, "answerpos.json"), "w") as fh:
            json.dump(out, fh)
        d = [r for r in out if r["depth"] == depth and r["ok"]]
        res = [r["first_top1_pos"] for r in d if r["first_top1_pos"] >= 0]
        print(f"  depth {depth:>3}: any_top1 {sum(r['any_top1'] for r in d)/max(1,len(d)):.1%}"
              f"  at pos 0 {sum(r['first_top1_pos'] == 0 for r in d)/max(1,len(d)):.1%}"
              f"  median pos {sorted(res)[len(res)//2] if res else float('nan')}"
              f"  distractor any_top1 {sum(r['dist_any_top1'] for r in d)/max(1,len(d)):.1%}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    free_arm(model, MODEL_ID)
    print("DONE", flush=True)


main()
