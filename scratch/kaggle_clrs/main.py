"""Capability screen on CLRS-Text, which is IN Huginn's own training mixture.

THE GAP THIS ADDRESSES, NAMED BY THE SUPERVISOR. The project has never had a task
family with a difficulty PARAMETER and success in a readable band. Difficulty varies
across the 21-family battery, but the two parametric ladders inside it are broken:
`count4/count8/count16` reads 21% / 0% / 12% -- floored and non-monotone -- and
`caesar1_letter/caesar1_word/rot13_word` reads 8% / 0% / 0%. So "more reasoning
steps" has never been testable against a graded axis that the model can actually
climb.

WHY CLRS-TEXT AND NOT ANOTHER SYNTHETIC FAMILY.
  * It is in Huginn's pre-training data (`tomg-group-umd/CLRS-Text-train`, listed on
    the model card). Every task this project has used was invented here, so "the
    model looks incapable because the prompts are off-distribution" has been a live
    and untested explanation. This removes it.
  * Two independent difficulty knobs: the ALGORITHM (30 of them) and the problem
    SIZE n.
  * It can break the length/difficulty collinearity, which is the property that
    matters most here. D85 measured the geometry tracking prompt length and D88
    showed the shape reading a single token, so a difficulty axis collinear with
    token count is unusable. Array algorithms scale ~n in tokens and graph
    algorithms ~n^2, so (algorithm, n) cells exist with matched token counts and
    very different difficulty. This run measures the token counts so that design can
    be built rather than assumed.

SCORED ON THE DECODED STRING, NOT ON FIRST-TOKEN RANK. D89 found that `correct` --
the first token of gold reaching rank 1 -- measures the leading token for 8 of 21
battery families, and CLRS answers are long numeric strings where that instrument is
simply wrong. Every completion is banked verbatim so scoring can be redone offline
without another GPU hour.

PRE-REGISTERED (CLAUDE.md section 1).
P1 -- GATE ON THE INSTRUMENT. `assert_generation_works` must pass before any number
      is read. D62 printed a capability verdict over empty strings for want of it.
P2 -- THE SCREEN. Per (algorithm, size) cell, exact-match on the FINAL answer -- the
      field after the last "|" -- and a containment score, over 6 items. The output
      is the list of cells landing in 20-80%.
P3 -- THE LENGTH MAP, which is the part that makes the screen usable. Token count is
      recorded per item, so cells with matched token count and different difficulty
      can be identified. Reported whatever P2 returns.
P4 -- REFUSAL. If no cell lands in 20-80% at any depth, that is reported as the
      result and the next run is a different dataset, not a different window. The
      project has twice mistaken a floored axis for a measurement.

Two depths, because D86 found exact-match peaking shallow and collapsing by r=8
while containment rises to r=32: r=4 and r=32 bracket that.
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
import re
import time
import traceback

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
DATASET = "tomg-group-umd/CLRS-Text-train"
OUTDIR = "/kaggle/working"

N_ITEMS = 4              # per (algorithm, size) cell -- was 6; see DEPTHS below
MAX_NEW = 24
# DEPTHS WAS (4, 32) AND COULD NOT FINISH (D105). Depth 4 took 6126 s on 222 items;
# depth 32 is 8x the unrolls, so the pair needed ~55500 s against Kaggle's 43200 s
# limit, and because the dump happened only at the END of a depth the overrun
# produced NOTHING. (4, 8) costs ~6126 + ~12250 = ~18400 s, which fits with margin
# even after few-shot exemplars lengthen every prompt.
# KEPT AT (4, 32) DELIBERATELY. An earlier fix dropped it to (4, 8) to fit the
# budget, and `test_clrs_brackets_the_depth_where_d86_found_the_transition` caught
# that this destroys the design: D86's finding is that exact-match peaks SHALLOW
# while containment rises DEEP, and only a 4-vs-32 bracket tests it. The budget is
# met by cutting N_ITEMS 6 -> 4 (222 -> ~148 items) instead, and by the two
# protections D105 lacked: per-batch dumps and the wall guard below, which together
# mean an overrun costs the tail of a depth rather than everything.
DEPTHS = (4, 32)
WALL_BUDGET_S = 34000    # stop cleanly well inside Kaggle's 12 h, having saved
N_ALGOS = 10
SIZE_BINS = (4, 8, 12, 16)   # problem sizes, read off the input list length
N_SHOTS = 2              # same-algorithm exemplars, selected PER ITEM excluding itself
INSTRUCTION = ("Answer with only the final value after the last '|'. "
               "Do not restate the trace.")
ANSWER_PREFIX = "Final answer: "


def final_answer(ans):
    """CLRS answers are `trace | final`; the screen scores the FINAL field.

    Splitting on the LAST separator, not the first: several algorithms emit `|`
    inside the trace, and taking the first would score a trace step as the answer.
    """
    return ans.rsplit("|", 1)[-1].strip() if "|" in ans else ans.strip()


def problem_size(question):
    """Number of elements in the first bracketed list -- CLRS's difficulty knob.

    Read from the question text rather than from a metadata column, because the
    dataset carries only `question`, `answer` and `algo_name`. Returns 0 when no
    list is found, and those items are dropped rather than binned as size 0.
    """
    m = re.search(r"\[([^\[\]]*)\]", question)
    if not m:
        return 0
    inner = m.group(1).strip()
    return len(inner.split()) if inner else 0


def normalise(text):
    """Same normalisation on both sides, as in `scripts/run_depth_accuracy.py`."""
    t = str(text).strip().lower()
    for ch in ".,!?;:'\"":
        t = t.replace(ch, " ")
    return " ".join(t.split())


def score(pred, gold):
    p, g = normalise(pred), normalise(gold)
    return (p == g), (g in p)


def main():
    run("pip install -q 'transformers>=4.50,<4.54' datasets")
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
          flush=True)
    ds = load_dataset(DATASET, split="train", streaming=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

    # Build the (algorithm, size) grid by streaming until each cell is full, so the
    # whole 430k-row parquet never has to be materialised on a Kaggle worker.
    want = N_ITEMS
    cells, seen_algos = {}, []
    for row in ds:
        algo = row["algo_name"]
        if algo not in seen_algos:
            if len(seen_algos) >= N_ALGOS:
                continue
            seen_algos.append(algo)
        n = problem_size(row["question"])
        if n == 0:
            continue
        size = min(SIZE_BINS, key=lambda b: abs(b - n))
        if abs(size - n) > 2:
            continue
        key = (algo, size)
        cur = cells.setdefault(key, [])
        if len(cur) >= want:
            continue
        cur.append({"algo": algo, "size": size, "n_raw": n,
                    "question": row["question"],
                    "gold": final_answer(row["answer"])})
        if len(cells) >= N_ALGOS * len(SIZE_BINS) and all(
                len(v) >= want for v in cells.values()):
            break
    items = [x for v in cells.values() for x in v if len(v) >= 3]
    print(f"{len(items)} items over {len({(i['algo'], i['size']) for i in items})} "
          f"(algorithm, size) cells, {len({i['algo'] for i in items})} algorithms",
          flush=True)

    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = load_arm(MODEL_ID, cfg, REVISION)
    assert_generation_works(model, tok, chat=True)
    print("P1: generation gate PASSED", flush=True)

    # FEW-SHOT + A STRICT ANSWER PREFIX. D105 scored 0.0% exact with a bare
    # question and no exemplars, which the deep-research pass says cannot be read
    # as a capability floor -- and this project's own D69 measured one added
    # instruction moving a family 0% -> 83%. Exemplars are drawn from the SAME
    # algorithm as the item so the format is demonstrated, and are taken from
    # items NOT scored, so no item ever sees its own answer.
    by_algo: dict[str, list] = {}
    for it in items:
        by_algo.setdefault(it["algo"], []).append(it)

    for it in items:
        # Exemplars from the SAME algorithm but never this item: every item is
        # scored, so drawing shots from a fixed prefix of the group would hand
        # items 0..N_SHOTS-1 their own gold. Selecting per item is the fix, and
        # the leakage gate below verifies it rather than trusting it.
        pool = [d for d in by_algo[it["algo"]] if d is not it][:N_SHOTS]
        shot_text = "".join(
            f"{d['question']}\n{ANSWER_PREFIX}{d['gold']}\n\n" for d in pool)
        prompt = (f"{INSTRUCTION}\n\n{shot_text}"
                  f"{it['question']}\n{ANSWER_PREFIX}")
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        it["text"] = text
        it["n_tokens"] = int(tok(text, add_special_tokens=False,
                                 return_tensors="pt").input_ids.shape[1])

    # NO ITEM MAY SEE ITS OWN ANSWER. An exemplar drawn from the scored set would
    # hand the model the gold it is being tested on; check it rather than assume.
    # Check QUESTION IDENTITY, not gold value. CLRS golds are small integers, so
    # two items in one cell routinely share one; a value-based check would read
    # that coincidence as leakage and abort a legitimate run. What actually
    # matters is whether an item appears as its OWN exemplar.
    leaked = [it["algo"] for it in items
              if it["text"].count(it["question"]) > 1]
    if leaked:
        print(f"  LEAKAGE GATE FAILED: {len(leaked)} items contain their own gold "
              f"in an exemplar -- refusing to score. Algos: {sorted(set(leaked))}",
              flush=True)
        return
    print(f"  leakage gate PASSED ({len(items)} items, {N_SHOTS}-shot)", flush=True)

    records = []
    t0 = time.time()
    BATCH = 24
    for depth in DEPTHS:
        if time.time() - t0 > WALL_BUDGET_S:
            print(f"  wall budget {WALL_BUDGET_S}s reached before depth {depth} -- "
                  f"stopping cleanly with {len(records)} records SAVED", flush=True)
            break
        try:
            # PER-BATCH, NOT PER-DEPTH. D105 lost 4.8 hours of depth-32 work because
            # the dump happened only after a whole depth finished.
            outs = []
            for b0 in range(0, len(items), BATCH):
                chunk = items[b0:b0 + BATCH]
                outs.extend(batched_generate(
                    model, tok, [i["text"] for i in chunk],
                    max_new=MAX_NEW, num_steps=depth, verbose=False))
                for it, o in zip(chunk, outs[b0:b0 + len(chunk)]):
                    ex, ct = score(o, it["gold"])
                    records.append({k: it[k] for k in ("algo", "size", "n_raw",
                                                       "gold", "n_tokens", "question")}
                                   | {"depth": depth, "output": o,
                                      "exact": bool(ex), "contains": bool(ct)})
                with open(os.path.join(OUTDIR, "clrs.json"), "w") as fh:
                    json.dump(records, fh)
                done = b0 + len(chunk)
                print(f"    depth {depth:>3} [{done}/{len(items)}] "
                      f"elapsed {time.time()-t0:.0f}s, saved", flush=True)
                if time.time() - t0 > WALL_BUDGET_S:
                    print("    wall budget reached mid-depth -- saved and stopping",
                          flush=True)
                    break
        except Exception as exc:
            print(f"  depth {depth}: FAILED {type(exc).__name__}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            continue
        deg = degenerate(outs)
        here = [r for r in records if r["depth"] == depth]
        n = max(1, len(here))
        print(f"  depth {depth:>3}: exact {sum(r['exact'] for r in here) / n:.1%}, "
              f"contains {sum(r['contains'] for r in here) / n:.1%}, {len(here)} items, "
              f"degenerate={deg}", flush=True)

    print("\n=== P2: per-cell accuracy (best over depth) ===", flush=True)
    band = []
    for (algo, size) in sorted({(r["algo"], r["size"]) for r in records}):
        sub = [r for r in records if r["algo"] == algo and r["size"] == size]
        by_depth = {d: [r for r in sub if r["depth"] == d] for d in DEPTHS}
        best_ex = max((sum(r["exact"] for r in v) / max(1, len(v))
                       for v in by_depth.values()), default=0.0)
        best_ct = max((sum(r["contains"] for r in v) / max(1, len(v))
                       for v in by_depth.values()), default=0.0)
        ntok = sum(r["n_tokens"] for r in sub) / max(1, len(sub))
        flag = "  <- IN BAND" if 0.2 <= best_ex <= 0.8 else ""
        if flag:
            band.append((algo, size, best_ex))
        print(f"  {algo:>28} n={size:<3} exact {best_ex:>5.0%} contains "
              f"{best_ct:>5.0%}  {ntok:>5.0f} tok{flag}", flush=True)

    print(f"\n=== P2 RESULT: {len(band)} cells land in 20-80% exact ===", flush=True)
    if not band:
        print("  P4: NO cell is in band. The screen's answer is that this dataset,",
              flush=True)
        print("  at these sizes and depths, does not give a readable difficulty axis",
              flush=True)
        print("  either -- report it and change the dataset, not the window.", flush=True)

    print("\n=== P3: token count against size, per algorithm ===", flush=True)
    for algo in sorted({r["algo"] for r in records}):
        per = {}
        for r in records:
            if r["algo"] == algo:
                per.setdefault(r["size"], []).append(r["n_tokens"])
        print(f"  {algo:>28}: " + "  ".join(
            f"n={s}:{sum(v) / len(v):.0f}tok" for s, v in sorted(per.items())),
            flush=True)

    free_arm(model, MODEL_ID)
    print(f"\n=== {len(records)} generations in {(time.time() - t0) / 60:.1f} min ===",
          flush=True)
    print("scoring is redone offline from the banked strings", flush=True)
    print("DONE", flush=True)


main()
