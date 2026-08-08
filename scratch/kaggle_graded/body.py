"""Does the model KNOW answers it never EMITS? Graded readout vs exact match.

THE PROBLEM WITH EVERY CAPABILITY CLAIM IN THIS PROJECT
    Each one -- D56, D60, D61, the Caesar screen, and the still-unrun capcontent
    -- scores capability by GREEDY GENERATION plus string match. That instrument
    has failed loudly at least twice (D62: empty output for every prompt; D60:
    format alone moved `copy` 15% -> 100%) and it has no dynamic range: once
    accuracy is 0 every task looks identical, so "the model cannot do this" and
    "the readout cannot see it" are indistinguishable. UNDERSTANDING.md 6.1 makes
    that the project's largest open threat, because every "content is
    architectural" result was measured where accuracy is ~0.

    The Caesar screen is the sharpest case. Trained exact-match is 0.0% in all six
    cells, and the SECOND metric it added to avoid exactly this trap -- char_acc --
    is uninterpretable: the untrained arm BEATS the trained one in four of six
    cells (5.8% vs 0.0%), and the trained arm's best cells come from emitting a
    memorised pangram ('the cat sat on the mat' -> 'The quick brown fox jumps over
    the lazy dog'), i.e. D61's retrieval mode, not partial competence. Both of that
    screen's metrics are blunt.

THE INSTRUMENT THE PROJECT ALREADY BUILT AND NEVER POINTED HERE
    `traj_geom/eval_depth.py` scores TEACHER-FORCED log P(gold) per unroll by
    hooking `core_block[-1]` and decoding each unroll's state through a replicated
    coda+head. It was used for D35 (required depth scales with difficulty) and
    never for capability. Two properties make it the right tool:

      * ONE forward at num_steps=R yields the WHOLE depth curve r=1..R, because
        the hook fires once per unroll. Generation needs a separate multi-token
        decode per depth. That is the difference between a depth x task x arm
        sweep costing minutes and costing hours (C9: generation was the entire
        cost, 1 completion/min).
      * It already computes log_softmax over the FULL vocabulary. RANK is one
        line away and was simply never recorded.

WHY RANK AND NOT Acc@5
    Acc@k truncates exactly where the interesting variation lives. A model at 0%
    accuracy might hold the gold token at rank 3 or at rank 8000; Acc@5 calls the
    first a miss and the second a miss. Full-vocab rank separates them by three
    orders of magnitude, and log-rank is the natural scale (chance = 32768).
    Recorded per unroll, so "does depth help" becomes a question with a gradient
    instead of a step.

PRE-REGISTERED PREDICTIONS, written before the run (CLAUDE.md section 1)
  P1. On cells where exact-match is 0 (count16, rot13_word), the TRAINED model
      still places gold far below chance rank (median rank << 32768). If instead
      median rank ~ chance, then "the model cannot do these tasks" is robust, the
      graded readout adds nothing, and this whole direction is refuted cheaply.
  P2. The trained-minus-untrained log-rank gap is positive everywhere and GROWS
      with capability. This is UNDERSTANDING 6.1's moderation test with a
      continuous moderator instead of a binary one.
  P3. Rank improves with depth r on tasks the model can do and is flat on tasks it
      cannot -- B4.11 (depth vs accuracy, never run) with dynamic range, from one
      forward per item.
  P4. On rot13_word, low gold rank together with a pangram top-1 would separate
      "knows but does not emit" from "does not know", which is exactly what D61's
      retrieval modes could not distinguish.

Per B4.14 this persists the CURVES (rank and logp at every unroll), not just
fitted summaries, so any question this analysis did not anticipate can be asked
locally without another GPU run.
"""
# @needs: run load_arm free_arm

import json
import random
import string

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MAX_R = 64
N_ITEMS = 24
VOCAB_CHANCE = 32768          # half of Huginn's 65536-token vocabulary


def items(task, n=N_ITEMS):
    """(prompt_body, gold, distractor). Gold is single-token where possible."""
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + hash(task) % 997)
        if task == "echo_digit":
            v = rng.randint(0, 9)
            out.append((f"Repeat this number exactly.\nNumber: {v}", str(v),
                        str((v + 3) % 10)))
        elif task == "add1":
            v = rng.randint(0, 8)
            out.append((f"What is {v} + 1?", str(v + 1), str((v + 4) % 10)))
        elif task == "count4":
            b = [rng.randint(0, 1) for _ in range(4)]
            g = sum(b)
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 2) % 5)))
        elif task == "count16":
            b = [rng.randint(0, 1) for _ in range(16)]
            g = sum(b)
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 3) % 17)))
        else:                                            # rot13_word
            w = "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
            enc = "".join(chr((ord(c) - 97 + 13) % 26 + 97) for c in w)
            other = "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
            out.append((f"Decode this ROT13 text.\nText: {enc}", w, other))
    return out


TASKS = ("echo_digit", "add1", "count4", "count16", "rot13_word")


def coda_head(model, h_state, freqs_cis):
    """model's own ln_f -> coda -> ln_f -> lm_head on an intermediate state.

    TWO ln_f calls, not one: the first is the normalisation `iterate_forward`
    applies before returning, and skipping it feeds coda a state it was never
    built to accept -- measured as a consistent 1.7-1.9 max abs logit error on
    real hardware (traj_geom/extraction/hook.py, 2026-07-23). Do not shortcut.
    """
    import torch
    x = model.transformer.ln_f(h_state)
    block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
    for block in model.transformer.coda:
        block_idx -= 1
        x = block(x, freqs_cis, block_idx, None, None)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


def curve(model, tok, prompt, gold, distractor):
    """Per-unroll (rank of gold's first token, logp gold, logp distractor, top1).

    ONE forward at num_steps=MAX_R; the hook fires once per unroll, so this
    returns the whole depth curve for the cost of a single pass.
    """
    import torch
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids_p = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    g_ids = tok(gold, add_special_tokens=False).input_ids
    d_ids = tok(distractor, add_special_tokens=False).input_ids
    ids = torch.cat([ids_p, torch.tensor([g_ids], device=model.device,
                                         dtype=ids_p.dtype)], dim=1)
    n_p = ids_p.shape[1]
    freqs = model.freqs_cis[:, : ids.shape[1]]
    rec = {"rank": [], "logp_gold": [], "logp_dist": [], "top1": []}
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()

    def hook(_m, _i, o):
        with torch.no_grad():
            lg = coda_head(model, o.detach(), freqs).float()
            lp = torch.log_softmax(lg[0], dim=-1)
            row = lp[n_p - 1]                                  # predicts gold token 0
            rec["rank"].append(int((row > row[g_ids[0]]).sum().item()) + 1)
            rec["top1"].append(int(row.argmax().item()))
            rec["logp_gold"].append(float(sum(lp[n_p - 1 + k, t].item()
                                              for k, t in enumerate(g_ids))))
            rec["logp_dist"].append(float(lp[n_p - 1, d_ids[0]].item()))

    h = mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids, num_steps=MAX_R)
    finally:
        h.remove()
    rec["n_gold_tok"] = len(g_ids)
    return rec


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)

    out = {"vocab": len(tok), "max_r": MAX_R, "n_items": N_ITEMS, "arms": {}}
    for arm, spec in (("trained", MODEL_ID), ("untrained", None)):
        print(f"\n=== {arm} ===", flush=True)
        model = load_arm(spec, cfg, REVISION if spec else 0)
        out["arms"][arm] = {}
        for task in TASKS:
            rows = []
            for prompt, gold, dist in items(task):
                rows.append(curve(model, tok, prompt, gold, dist))
            out["arms"][arm][task] = rows
            r_final = np.array([r["rank"][-1] for r in rows], float)
            r_best = np.array([min(r["rank"]) for r in rows], float)
            top1 = float(np.mean([r["rank"][-1] == 1 for r in rows]))
            print(f"  {task:>11}: median rank@r64 {np.median(r_final):>8.0f}  "
                  f"best-depth median {np.median(r_best):>8.0f}  "
                  f"acc@1 {top1:>5.1%}  chance={VOCAB_CHANCE}", flush=True)
            with open("graded.json", "w") as f:
                json.dump(out, f)
        free_arm(model)

    print("\n=== VERDICT ===")
    for task in TASKS:
        t = np.array([r["rank"][-1] for r in out["arms"]["trained"][task]], float)
        u = np.array([r["rank"][-1] for r in out["arms"]["untrained"][task]], float)
        gap = float(np.median(np.log10(u)) - np.median(np.log10(t)))
        p1 = "BELOW chance" if np.median(t) < VOCAB_CHANCE / 10 else "~chance"
        print(f"  {task:>11}: trained median rank {np.median(t):>8.0f} ({p1})  "
              f"untrained {np.median(u):>8.0f}  log10 gap {gap:+.2f}")
    with open("graded.json", "w") as f:
        json.dump(out, f)


main()
