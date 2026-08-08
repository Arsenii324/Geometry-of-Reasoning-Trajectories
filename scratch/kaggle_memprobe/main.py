"""What activation budget does the uncached generator actually need on a T4?

WHY THIS EXISTS. `geometry-prompt-depth` OOM'd (D62-era run, kernel status ERROR)
inside `nonlin(x_fc_1) * x_fc_2` while generating. The fix was `max_batch_tokens`,
which splits each length-bucket so `chunk * (prompt_len + max_new) <= budget`. But
the VALUE has never been measured -- 1024 is the library default and 384 was set,
then reverted, for promptdepth, and commit ffb42fc left body.py and main.py
disagreeing about which. Choosing it by argument is guesswork; this measures it.

WHAT IS MEASURED. Peak CUDA allocation for the WORST cell in promptdepth's stage 1
-- parity/decompose, the longest prompt (56 tokens) at the longest completion
(max_new=64), i.e. 1920 tokens of batch x sequence if left unsplit. Each budget is
run in isolation with `reset_peak_memory_stats`, and an OOM is caught so one run
answers the whole ladder instead of dying at the first failure.

The `continuous_compute` arm is measured too, at the largest surviving budget: it
requests `return_logits` AND `return_latents` via output_details, which the plain
arm does not, and no kernel in this project has ever profiled it.

PRE-REGISTERED PREDICTIONS, written before the run (CLAUDE.md section 1):
  P1. The unbudgeted config OOMs, reproducing the original failure. Its peak would
      be 1920 tokens; the original died at an allocation of 98 MiB, which is
      exactly [b*s, 17920] float32 with b*s = 1433, so failure is expected BEFORE
      the nominal peak is reached.
  P2. budget=1024 SURVIVES. It caps the peak at 1017 tokens (computed locally from
      the real tokenizer), which is below the ~1433 where the original died.
      Margin is only ~1.4x, so this is the prediction most likely to be wrong --
      and the reason for running this at all.
  P3. Peak allocated memory grows monotonically with the budget, and roughly
      linearly in `chunk * (prompt_len + max_new)`.
  P4. The continuous_compute arm peaks HIGHER than the plain arm at equal budget,
      because it additionally retains latents.
If P2 fails, the library default is wrong for this workload and promptdepth must
carry an explicit smaller budget; that is a result either way.
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
import traceback

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
N_ITEMS = 16                      # promptdepth's batch before any split
MAX_NEW = 64                      # promptdepth's cot/decompose completion length
NUM_STEPS = 32                    # promptdepth's PICK_DEPTH
UNBUDGETED = 10**9          # a sentinel LARGE budget; see the note below
BUDGETS = (UNBUDGETED, 1024, 384)

# V2, after reading v1's raw numbers rather than its verdict. Two flaws found:
#
#  (a) v1 spelled "unbudgeted" as `kw = {}`, which takes batched_generate's
#      DEFAULT max_batch_tokens=1024 -- it does not mean unlimited. So v1 never
#      tested the unbudgeted path at all, and P1 was never evaluated. The tell was
#      in the raw table, not the verdict: the "none" and "1024" arms reported a
#      BYTE-IDENTICAL peak of 13977.0517578125 MiB, and "none" showed 0.191
#      MiB/token against ~0.39 for every real arm -- exactly half, because its
#      labelled chunk of 16 was my own arithmetic while the code ran chunk 8.
#      Fixed by passing an explicit huge budget so the split is a no-op.
#
#  (b) v1 recorded only `type(e).__name__` for a failure, so the
#      continuous_compute arm came back as a bare "RuntimeError" with no message
#      and no mechanism. D62's lesson exactly: a failure you cannot read is a
#      failure you will guess about. Now the message and traceback are captured.

Q = ("Is the number of ones in this sequence even or odd?\nSequence: {x}"
     "\nGo through the sequence one item at a time, flipping between even and odd "
     "for each 1, then state the result.")


def prompts(tok):
    """promptdepth's parity/decompose cell: its longest prompt, 16 items."""
    import random
    out = []
    for s in range(N_ITEMS):
        rng = random.Random(s * 7919)
        x = " ".join(str(rng.randint(0, 1)) for _ in range(4))
        out.append(tok.apply_chat_template([{"role": "user", "content": Q.format(x=x)}],
                                           tokenize=False, add_generation_prompt=True))
    return out


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"total GPU memory: {total:.0f} MiB", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True).to(torch.float32).to("cuda").eval()
    weights = torch.cuda.memory_allocated() / 2**20
    print(f"weights allocated: {weights:.0f} MiB   headroom: {total - weights:.0f} MiB", flush=True)
    assert_generation_works(model, tok)

    texts = prompts(tok)
    widths = sorted({len(tok(t, add_special_tokens=False).input_ids) for t in texts})
    print(f"prompt token widths: {widths}  max_new={MAX_NEW}  "
          f"unsplit batch*seq = {N_ITEMS * (max(widths) + MAX_NEW)}", flush=True)

    rec = {"device": torch.cuda.get_device_name(0), "total_mib": total,
           "weights_mib": weights, "widths": widths, "arms": []}

    print(f"\n{'budget':>8} {'chunk':>6} {'peak b*s':>9} {'peak MiB':>9} {'headroom MiB':>13}  outcome")
    for b in BUDGETS:
        kw = {"max_batch_tokens": b}
        per = max(1, min(N_ITEMS, b // (max(widths) + MAX_NEW)))
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            batched_generate(model, tok, texts, max_new=MAX_NEW, num_steps=NUM_STEPS,
                             verbose=False, **kw)
            peak = torch.cuda.max_memory_allocated() / 2**20
            ok = "OK"
        except torch.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated() / 2**20
            ok = "OOM"
            torch.cuda.empty_cache()
        except Exception as e:                       # noqa: BLE001 - report, do not abort the ladder
            peak, ok = float("nan"), f"{type(e).__name__}: {e}"
            print(traceback.format_exc(), flush=True)
        label = "unbudget" if b == UNBUDGETED else str(b)
        print(f"{label:>8} {per:>6} {per * (max(widths) + MAX_NEW):>9} "
              f"{peak:>9.0f} {total - peak:>13.0f}  {ok[:44]}", flush=True)
        rec["arms"].append({"budget": b, "chunk": per, "peak_mib": peak, "outcome": ok})
        with open("mem_probe.json", "w") as f:
            json.dump(rec, f, indent=1)

    # continuous_compute at the largest budget that survived
    rec["cc"] = []
    for b in (1024, 384):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            batched_generate(model, tok, texts, max_new=MAX_NEW, num_steps=NUM_STEPS,
                             verbose=False, max_batch_tokens=b, continuous_compute=True)
            peak, ok = torch.cuda.max_memory_allocated() / 2**20, "OK"
        except torch.OutOfMemoryError:
            peak, ok = torch.cuda.max_memory_allocated() / 2**20, "OOM"
            torch.cuda.empty_cache()
        except Exception as e:                       # noqa: BLE001
            peak, ok = float("nan"), f"{type(e).__name__}: {e}"
            print(traceback.format_exc(), flush=True)
        print(f"\ncontinuous_compute @ budget {b}: peak {peak:.0f} MiB  {ok}", flush=True)
        rec["cc"].append({"budget": b, "peak_mib": peak, "outcome": ok})

    with open("mem_probe.json", "w") as f:
        json.dump(rec, f, indent=1)

    print("\n=== VERDICT ===")
    by = {a["budget"]: a["outcome"] for a in rec["arms"]}
    unb = by.get(UNBUDGETED, "not run")
    print(f"  P1 (UNBUDGETED reproduces the original OOM): "
          f"{'CONFIRMED' if unb == 'OOM' else f'REFUTED -- unbudgeted came back {unb}'}")
    print(f"  P2 (budget 1024 survives): "
          f"{'CONFIRMED' if by.get(1024) == 'OK' else 'REFUTED -- promptdepth needs a smaller budget'}")
    cc_ok = [c['budget'] for c in rec['cc'] if c['outcome'] == 'OK']
    print(f"  P4 (continuous_compute runs at all): "
          f"{'yes, at ' + str(cc_ok) if cc_ok else 'NO -- it fails at every budget tried'}")
    for c in rec["cc"]:
        print(f"     cc@{c['budget']}: {c['outcome'][:100]}")
    safe = [a["budget"] for a in rec["arms"] if a["outcome"] == "OK"]
    print(f"  largest budget that survived: {max(safe) if safe else 'NONE'}")


main()
