"""B6 re-run: geometry conditioned on correctness, WITHIN a fixed answer value.

THE FOUNDING HYPOTHESIS' STRONGEST AVAILABLE TEST, and it has been blocked twice.
Every geometric measurement in this project has pooled correct and incorrect
trajectories, and since accuracy read ~0 almost everywhere, the geometry on record
describes failure. The project set out to ask whether trajectory shape encodes
reasoning and has never compared a trajectory that got the answer right against one
that got it wrong.

WHY THE FIRST ATTEMPT WAS VOID, AND WHAT FIXES IT. D72: in `geometry-correctness`,
correctness was very nearly a deterministic function of the ANSWER VALUE -- `add1`'s
correct and incorrect gold sets had ZERO overlap, `count4`'s successes all sat at
gold=1. Permuting the correctness label therefore broke the correctness-geometry
link and the prompt-geometry link at the same time, and no null could separate them.
The fix is not a better statistic; it is items where success and failure COEXIST AT
THE SAME GOLD VALUE, so the contrast holds the answer fixed by construction.
D75's 21-family screen found them: `count_mod3` (44%), `sub1` (31%), `nth_item`
(31%), `local_last` (25%) each have at least one gold value carrying both classes.

WHAT GEOMETRY, GIVEN D74. D74 measured that the orbit spans ~13 effective dimensions
against an isotropic null of ~21, 140/140 orbits, so PLANAR descriptions of it are
inadequate by construction -- which is why `winding_of`, all nine variants W1-W9 and
`classify_shape`'s settle/loop/drift taxonomy failed, and why the eigenplane rate
came out 0.24x its prediction. So this run deliberately measures NO planar quantity.
It banks the states and lets the analysis use D74's instrument (effective
dimensionality of the step directions, which is budget-robust: its sign held between
num_steps 64 and 128 where winding's FLIPPED) plus contraction rate and
settling time.

POSITION, corrected from the first attempt. Self-review found `geometry-correctness`
captured geometry at `[0, -1, :]` -- the appended gold-token position -- while
reading correctness at `n_p - 1`. Those are one token apart, and `-1` no longer
matched the project's own prior convention. Here BOTH are read at `n_p - 1`, the
position that actually predicts the answer.

A FREE POSITIVE CONTROL, from reusing D75's generators verbatim. Items 0-15 of
each family are the SAME items D75 screened (the seeding is per-index), so their
per-unroll ranks must reproduce D75's. The one deliberate difference is that D75
appended the gold token to the sequence and read position `n_p - 1`, while this
kernel appends nothing and reads the same position. Under causal attention those
are identical by construction -- appended tokens cannot influence an earlier
position -- so agreement confirms it, and DISAGREEMENT would itself be a finding:
it would mean the appended gold token was leaking into the readout that D68, D69,
D71 and D75 all rest on. Either way the check costs nothing and is worth more than
an assumption.

THIS KERNEL COMPUTES NOTHING (directions.md B14). It records states, the per-unroll
rank of gold, the prompt and the gold, and stops. Three kernels in this project
printed confident conclusions that were wrong, in every case because the verdict was
computed from the same variables as the run; and three separate questions (6.9,
D73's rank-1 objection, D72's confound) were unanswerable because a kernel stored
its conclusions instead of its data.

PRE-REGISTERED (CLAUDE.md section 1).
P1 -- GATE. Each family must yield at least 6 items in gold values carrying BOTH
      classes, else the within-value contrast is not constructible and that family
      is dropped with the count reported, not quietly pooled.
P2 -- The honest prediction is NO DIRECTION. "Correct answers settle sooner" and
      "correct answers keep moving longer" are both tellable stories, and picking
      one after the fact is how D22 got a confirmation it later withdrew. The test
      is two-sided on every metric.
P3 -- If correct and incorrect trajectories do not differ on ANY of effective
      dimensionality, contraction rate or settling time at matched answer value,
      then trajectory geometry does not track the computation -- a clean falsifiable
      negative that no amount of pooled measurement could produce, and the founding
      hypothesis' clearest available answer.
"""
# @needs: run load_arm free_arm

import json
import os
import random
import string
import time
import traceback
import zlib

import numpy as np

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 64          # D74's signal window is ~21 unrolls; 64 covers it with room
N_ITEMS = 32
FAMILIES = ("count_mod3", "sub1", "nth_item", "local_last")
OUTDIR = "/kaggle/working"


def items(task, n=N_ITEMS):
    """The D75 generators, verbatim, so the screen's accuracy numbers transfer.

    `zlib.crc32`, never `hash()`: Python salts str hashes per process, which made
    D68 and D69 draw different item sets for the same nominal cell (D69(3)).
    """
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + zlib.crc32(task.encode()) % 997)
        L = string.ascii_lowercase
        if task == "sub1":
            v = rng.randint(1, 9)
            out.append((f"What is {v} - 1?", str(v - 1), str((v + 4) % 10)))
        elif task == "nth_item":
            xs = [rng.randint(0, 9) for _ in range(5)]
            k = rng.randint(1, 5)
            out.append((f"What is item number {k} in this list?\n"
                        f"List: {' '.join(map(str, xs))}", str(xs[k - 1]),
                        str((xs[k - 1] + 3) % 10)))
        elif task == "count_mod3":
            b = [rng.randint(0, 1) for _ in range(9)]
            g = sum(b) % 3
            out.append(("Count how many ones are in this sequence, then give the "
                        "remainder when divided by 3.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 1) % 3)))
        else:                                                   # local_last
            ops = [rng.choice([1, -1]) for _ in range(8)]
            g = "Add" if ops[-1] > 0 else "Subtract"
            body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1."
                                             for o in ops)
            out.append((body + " What was the last instruction?", g,
                        "Subtract" if g == "Add" else "Add"))
        _ = L
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
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = load_arm(MODEL_ID, cfg, REVISION)

    manifest = []
    t0 = time.time()
    for fam in FAMILIES:
        for i, (prompt, gold, dist) in enumerate(items(fam)):
            tag = f"{fam}_i{i:02d}"
            try:
                text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
                ids = tok(text, return_tensors="pt",
                          add_special_tokens=False).input_ids.to(model.device)
                g_ids = tok(gold, add_special_tokens=False).input_ids
                d_ids = tok(dist, add_special_tokens=False).input_ids
                n_p = ids.shape[1]
                # BOTH geometry and correctness are read at n_p - 1, the position
                # that predicts the answer. The first B6 attempt read them one token
                # apart (self-review, 2026-08-08); nothing is appended here at all.
                freqs = model.freqs_cis[:, :n_p]
                states, ranks, lps = [], [], []
                mod = model.transformer.core_block[-1]
                mod._forward_hooks.clear()

                def hook(_m, _i, o):
                    with torch.no_grad():
                        st = o.detach()
                        states.append(st[0, n_p - 1, :].float().cpu().numpy())
                        row = torch.log_softmax(
                            coda_head(model, st, freqs).float()[0, n_p - 1], dim=-1)
                        ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)
                        lps.append([float(row[g_ids[0]].item()),
                                    float(row[d_ids[0]].item())])

                h = mod.register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        model(input_ids=ids, num_steps=NUM_STEPS)
                finally:
                    h.remove()

                arr = np.stack(states).astype(np.float32)
                np.save(os.path.join(OUTDIR, tag + ".npy"), arr)
                rec = {"tag": tag, "family": fam, "item": i, "prompt": prompt,
                       "gold": gold, "distractor": dist, "n_tokens": int(n_p),
                       "num_steps": NUM_STEPS, "rank_curve": ranks,
                       "logp": lps, "best_rank": int(min(ranks)),
                       "correct": bool(min(ranks) == 1),
                       "best_depth": int(np.argmin(ranks)) + 1,
                       "shape": list(arr.shape), "ok": True}
            except Exception as exc:
                rec = {"tag": tag, "family": fam, "item": i, "ok": False,
                       "why": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            manifest.append(rec)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as fh:
                json.dump(manifest, fh)
        done = [r for r in manifest if r["family"] == fam and r.get("ok")]
        acc = np.mean([r["correct"] for r in done]) if done else float("nan")
        # P1's gate, reported per family: how many items sit in gold values that
        # carry BOTH classes. Anything else cannot support a within-value contrast.
        ok_g = {r["gold"] for r in done if r["correct"]}
        bad_g = {r["gold"] for r in done if not r["correct"]}
        mixed = ok_g & bad_g
        n_in = sum(1 for r in done if r["gold"] in mixed)
        print(f"  {fam:>12}: {len(done)} items, acc {acc:.0%}, "
              f"{len(mixed)} gold values carry both classes, {n_in} items in them "
              f"(P1 gate wants >= 6)", flush=True)

    free_arm(model, MODEL_ID)
    print(f"\n=== banked {sum(1 for r in manifest if r.get('ok'))}/{len(manifest)} "
          f"in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print("no analysis here by design; see scripts/run_b6.py", flush=True)
    print("DONE", flush=True)


main()
