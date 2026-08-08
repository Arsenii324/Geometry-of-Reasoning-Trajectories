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
# @needs: run batched_generate assert_generation_works

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
