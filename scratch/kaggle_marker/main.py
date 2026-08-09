"""Does the shape encode the COMPUTATION, or just the marker token?

WHAT `geometry-lenmatch` ESTABLISHED, AND WHAT IT LEFT OPEN. At verified-identical
token counts -- three pairs, 188 of 188 items kept, token multisets identical -- a
classifier on the orbit's rotation/translation/scale-invariant shape separates the
two task markers at **96.9-100% balanced accuracy**, p = 0.0025, against nulls at
50%. That removes D84's length confound completely. It does not remove a simpler
explanation: the marker is part of the INPUT, so the shape may be responding to one
token rather than to the different computation that token selects.

THE CONTROL, AND IT IS THE WHOLE POINT. Three markers instead of two, with the rule
block defining **A and C to do exactly the same thing** and B something different:

    A vs B  -- different marker, DIFFERENT computation
    A vs C  -- different marker, SAME computation

Both contrasts differ by exactly one token and are identical in length, so anything
that separates A from C is token sensitivity and nothing else. The comparison of the
two accuracies is the answer:

  * A-vs-B decodable and A-vs-C NOT -> the shape encodes the computation, and
    `geometry-lenmatch`'s result is about what the model is doing.
  * both decodable and roughly equally so -> the shape is reading the marker token,
    and the lenmatch result is input sensitivity wearing a task label.
  * neither -> something is wrong with this run and it says so rather than being
    read as evidence for the first branch.

P2 IS A GATE, NOT A FOOTNOTE. The model must actually treat A and C alike and B
differently, measured as gold rank per marker from the same forward. If it treats A
and C differently behaviourally, they are not the same computation for this model
and the control is void whatever the classifier says. `geometry-lenmatch` measured
exactly this kind of separation (0% vs 47%, 28% vs 6%, 3% vs 19%), so the
measurement is known to work.

Computes nothing (B14): states, rank curves, prompts and markers, then stops.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run load_arm free_arm


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

import hashlib
import json
import os
import random
import time
import traceback

import numpy as np

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 64          # matches b6bank and geomcap, so all three banks pool
N_ITEMS = 24
OUTDIR = "/kaggle/working"

RULES = ("Rules: for task A, {a}. For task B, {b}. For task C, {a}.\n"
         "Sequence: {seq}\n"
         "Task: {marker}")


def pairs(n=N_ITEMS):
    """(pair, marker, prompt, gold) for markers A, B, C.

    The rule block defines A and C IDENTICALLY and B differently, so `A` and `C`
    prompts differ in exactly one character and select the same computation, while
    `A` and `B` differ in exactly one character and select different computations.
    Golds are single characters, so rank does not depend on tokenisation.
    """
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + 104729)

        bits = [rng.randint(0, 1) for _ in range(8)]
        base = {"a": "report how many 1s are in the sequence",
                "b": "report the last symbol of the sequence",
                "seq": " ".join(map(str, bits))}
        g = {"A": str(sum(bits)), "B": str(bits[-1]), "C": str(sum(bits))}
        for m in ("A", "B", "C"):
            out.append(("count_vs_last", m, RULES.format(marker=m, **base), g[m]))

        xs = [rng.randint(0, 9) for _ in range(6)]
        base = {"a": "report the first symbol of the sequence",
                "b": "report the last symbol of the sequence",
                "seq": " ".join(map(str, xs))}
        g = {"A": str(xs[0]), "B": str(xs[-1]), "C": str(xs[0])}
        for m in ("A", "B", "C"):
            out.append(("first_vs_last", m, RULES.format(marker=m, **base), g[m]))

        ys = [rng.randint(0, 4) for _ in range(5)]
        base = {"a": "report the largest symbol of the sequence",
                "b": "report the smallest symbol of the sequence",
                "seq": " ".join(map(str, ys))}
        g = {"A": str(max(ys)), "B": str(min(ys)), "C": str(max(ys))}
        for m in ("A", "B", "C"):
            out.append(("max_vs_min", m, RULES.format(marker=m, **base), g[m]))
    return out


def coda_head(model, h_state, freqs_cis):
    """model's own ln_f -> coda -> ln_f -> lm_head on an intermediate state.

    TWO ln_f calls, not one: D71 validated exactly this tail as bit-identical to the
    model's own logits (max|delta| = 0.000000), with the known-wrong one-ln_f variant
    separating at 2.33-2.46. Do not shortcut it.
    """
    import torch
    x = model.transformer.ln_f(h_state)
    block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
    for block in model.transformer.coda:
        block_idx -= 1
        x = block(x, freqs_cis, block_idx, None, None)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


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

    def encode(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt", add_special_tokens=False).input_ids

    # P1: the identical-length gate, applied BEFORE the model is loaded so a
    # failure costs nothing. Items are kept only when BOTH markers tokenise to the
    # same length -- the property this whole run depends on, measured per item
    # because `make_variants` once asserted it and was wrong.
    items = pairs()
    by_key = {}
    for pair, marker, prompt, gold in items:
        by_key.setdefault((pair, prompt.split("Task:")[0]), {})[marker] = (prompt, gold)
    kept, dropped = [], 0
    for (pair, _), forms in by_key.items():
        if set(forms) != {"A", "B", "C"}:
            dropped += 1
            continue
        lens = {m: int(encode(forms[m][0]).shape[1]) for m in ("A", "B", "C")}
        la = lens["A"]
        if len(set(lens.values())) != 1:
            dropped += 1
            continue
        for marker in ("A", "B", "C"):
            kept.append((pair, marker, *forms[marker], la))
    n_by_pair = {}
    for pair, *_ in kept:
        n_by_pair[pair] = n_by_pair.get(pair, 0) + 1
    print(f"P1 length gate: kept {len(kept)} prompts, dropped {dropped} item(s) "
          f"whose two markers tokenised differently", flush=True)
    for p, n in sorted(n_by_pair.items()):
        status = "OK" if n >= 45 else "TOO FEW -- will not be analysed"
        print(f"    {p:>16}: {n} prompts ({n // 2} matched items)  {status}",
              flush=True)
    if not kept:
        print("no length-matched items survived; nothing to bank", flush=True)
        return

    model = load_arm(MODEL_ID, cfg, REVISION)
    manifest = []
    t0 = time.time()
    for pair, marker, prompt, gold, n_tok in kept:
        tag = f"{pair}_{marker}_{hashlib.sha256(prompt.encode()).hexdigest()[:8]}"
        try:
            ids = encode(prompt).to(model.device)
            g_ids = tok(gold, add_special_tokens=False).input_ids
            n_p = ids.shape[1]
            freqs = model.freqs_cis[:, :n_p]
            states, ranks = [], []
            mod = model.transformer.core_block[-1]
            mod._forward_hooks.clear()

            def hook(_m, _i, o):
                with torch.no_grad():
                    st = o.detach()
                    states.append(st[0, n_p - 1, :].float().cpu().numpy())
                    row = torch.log_softmax(
                        coda_head(model, st, freqs).float()[0, n_p - 1], dim=-1)
                    ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)

            h = mod.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    model(input_ids=ids, num_steps=NUM_STEPS)
            finally:
                h.remove()

            arr = np.stack(states).astype(np.float32)
            np.save(os.path.join(OUTDIR, tag + ".npy"), arr)
            rec = {"tag": tag, "family": f"{pair}_{marker}", "pair": pair,
                   "marker": marker, "item": 0, "prompt": prompt, "gold": gold,
                   "n_tokens": int(n_p), "matched_len": int(n_tok),
                   "num_steps": NUM_STEPS, "rank_curve": ranks,
                   "best_rank": int(min(ranks)), "correct": bool(min(ranks) == 1),
                   "best_depth": int(np.argmin(ranks)) + 1,
                   "state_sha": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
                   "shape": list(arr.shape), "ok": True}
        except Exception as exc:
            rec = {"tag": tag, "family": f"{pair}_{marker}", "pair": pair,
                   "marker": marker, "ok": False,
                   "why": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
        manifest.append(rec)
        with open(os.path.join(OUTDIR, "manifest.json"), "w") as fh:
            json.dump(manifest, fh)

    # P2: does the model distinguish the tasks BEHAVIOURALLY? If not, "the shape
    # does not encode the task" is untestable, because the model is not performing
    # two different computations for the classifier to have missed.
    print("", flush=True)
    for pair in sorted(n_by_pair):
        for marker in ("A", "B", "C"):
            sub = [r for r in manifest
                   if r.get("ok") and r["pair"] == pair and r["marker"] == marker]
            if not sub:
                continue
            acc = sum(r["correct"] for r in sub) / len(sub)
            med = float(np.median([r["best_rank"] for r in sub]))
            print(f"  {pair:>16} {marker}: {len(sub)} prompts, gold reaches rank 1 "
                  f"in {acc:.0%}, median best rank {med:.0f}", flush=True)

    free_arm(model, MODEL_ID)
    print(f"\n=== banked {sum(1 for r in manifest if r.get('ok'))}/{len(manifest)} "
          f"in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    print("no analysis here by design; see scripts/run_shape_decode.py", flush=True)
    print("DONE", flush=True)


main()
