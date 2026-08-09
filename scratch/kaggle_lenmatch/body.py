"""Two different computations on the SAME token count. The test D84 demands.

WHY THIS RUN EXISTS. D84 found that a classifier on the orbit's rotation-,
translation- and scale-invariant shape decodes the task family at 100% balanced
accuracy within every bank. It also found that the result cannot be attributed to
the TASK: across every orbit this project has banked, no two families share a prompt
length -- 0 of 6 pairs in one bank, 0 of 3 in the other -- so task identity and
sequence length are PERFECTLY collinear. Even `track`/`local`, built to be
length-matched and byte-identical in body, differ by 3 tokens in the appended
question and have disjoint counts (22/34/58/106 against 25/37/61/109). No
statistical control can separate perfectly collinear variables; D74(6) records
`partial_spearman` refusing at rho = -1.000 for exactly this shape of problem. It
needs a different experiment, and this is it.

THE DESIGN. Each pair presents an IDENTICAL context and an identical instruction
block, and selects the computation with a single trailing marker token -- `A` or
`B`. The two prompts therefore differ in exactly one token position and agree in
length by construction. The computations differ completely: counting versus
retrieval, first versus last, sum versus maximum.

THE GATE IS ABSOLUTE AND IS ENFORCED IN-KERNEL. Every item is tokenised BOTH ways
and DROPPED unless the two forms have identical token counts. `make_variants`'
docstring once claimed length-matching that turned out to be false (recorded in
`tests/test_kernel_tasks.py`), and this run exists precisely because that claim
failed; so the property is measured per item, not asserted, and the count of
dropped items is reported.

WHAT EACH OUTCOME MEANS.
  * shape decodes A-vs-B at matched length -> the trajectory's shape carries WHICH
    COMPUTATION is being performed, and D84's family result is not merely length.
  * it does not -> D84's 100% was sequence length, and the shape identifies the
    input's surface form and nothing about the operation applied to it.
Both are worth having, and the design cannot deliver a third answer.

PRE-REGISTERED (CLAUDE.md section 1).
P1 -- GATE. At least 24 items per pair must survive the identical-length check, or
      that pair is not analysed at all.
P2 -- GATE. The model must distinguish the tasks BEHAVIOURALLY, else "the shape does
      not encode the task" is untestable because the model is not doing the task.
      Measured as gold rank per marker from the same forward.
P3 -- THE TEST, run offline on the banked states: balanced accuracy of a classifier
      on the shape code, A versus B, within a pair. Chance is 0.5 by construction.
P4 -- CONTROL. The same classifier on raw states. If position also fails, the pair
      is too hard for the readout rather than uninformative about shape.

Computes nothing (B14): states, rank curves, prompts and markers, then stops.
"""
# @needs: run load_arm free_arm

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
N_ITEMS = 32
OUTDIR = "/kaggle/working"

RULES = ("Rules: for task A, {a}. For task B, {b}.\n"
         "Sequence: {seq}\n"
         "Task: {marker}")


def pairs(n=N_ITEMS):
    """(pair, marker, prompt, gold) with the two markers differing in ONE token.

    The context and the instruction block are byte-identical between markers, so
    everything except the final character is shared. Golds are single characters
    wherever possible, since a multi-token gold would make rank depend on
    tokenisation as well as on the model.
    """
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + 104729)

        bits = [rng.randint(0, 1) for _ in range(8)]
        seq = " ".join(map(str, bits))
        base = {"a": "report how many 1s are in the sequence",
                "b": "report the last symbol of the sequence", "seq": seq}
        out.append(("count_vs_last", "A", RULES.format(marker="A", **base),
                    str(sum(bits))))
        out.append(("count_vs_last", "B", RULES.format(marker="B", **base),
                    str(bits[-1])))

        xs = [rng.randint(0, 9) for _ in range(6)]
        seq = " ".join(map(str, xs))
        base = {"a": "report the first symbol of the sequence",
                "b": "report the last symbol of the sequence", "seq": seq}
        out.append(("first_vs_last", "A", RULES.format(marker="A", **base),
                    str(xs[0])))
        out.append(("first_vs_last", "B", RULES.format(marker="B", **base),
                    str(xs[-1])))

        ys = [rng.randint(0, 4) for _ in range(5)]
        seq = " ".join(map(str, ys))
        base = {"a": "report the largest symbol of the sequence",
                "b": "report the smallest symbol of the sequence", "seq": seq}
        out.append(("max_vs_min", "A", RULES.format(marker="A", **base),
                    str(max(ys))))
        out.append(("max_vs_min", "B", RULES.format(marker="B", **base),
                    str(min(ys))))
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
        if set(forms) != {"A", "B"}:
            dropped += 1
            continue
        la = int(encode(forms["A"][0]).shape[1])
        lb = int(encode(forms["B"][0]).shape[1])
        if la != lb:
            dropped += 1
            continue
        for marker in ("A", "B"):
            kept.append((pair, marker, *forms[marker], la))
    n_by_pair = {}
    for pair, *_ in kept:
        n_by_pair[pair] = n_by_pair.get(pair, 0) + 1
    print(f"P1 length gate: kept {len(kept)} prompts, dropped {dropped} item(s) "
          f"whose two markers tokenised differently", flush=True)
    for p, n in sorted(n_by_pair.items()):
        status = "OK" if n >= 48 else "TOO FEW -- will not be analysed"
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
        for marker in ("A", "B"):
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
