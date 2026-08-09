"""Does substituting the STATE at unroll r causally flip the answer -- and where?

WHY THIS IS THE FIRST REAL CAUSAL EXPERIMENT THE PROJECT HAS. D47 attempted
activation patching once and it was VOID, not null: the outcome variable had zero
dynamic range before any intervention (the model emitted the same wrong token in
all 21 conditions), so the run could not distinguish "the direction is unused" from
"the readout is degenerate". Every other result in this project (D79, D84, D85, D91,
D92) is a CORRELATION between the state and the outcome, never an intervention.

D90 supplies the mechanism this design needs. Huginn's initial latent h_0 is
`torch.randn_like(input_embeds)`, unseeded (D78) -- so the SAME prompt, run twice,
differs ONLY in h_0, and for prompts near the decision boundary that alone flips the
answer between correct and incorrect (D90: 3 of 8 tested prompts split, `parity8` at
5/10). That gives a donor and a recipient trajectory for the SAME prompt, the SAME
weights and the SAME depth, differing in nothing but the random seed and therefore
in nothing but the state itself from unroll 0 onward.

THE INTERVENTION. For each boundary prompt: find one h_0 seed that answers correctly
(the DONOR) and one that answers incorrectly (the RECIPIENT), by deterministic seed
search (`torch.manual_seed`, D90's determinism control already proved this is
bit-reproducible). Then re-run the RECIPIENT's exact forward, and at a chosen unroll
r, REPLACE its state at the answer position with the DONOR's state at that same r --
recorded from a prior, separate forward -- and let the recipient's own forward
continue causally from there (its own KV cache, its own remaining unrolls). Only the
ANSWER-POSITION state is patched, not the whole sequence: a minimal, precise
intervention, stated explicitly so a coarser patch is never confused with this one.

WHY THE r-SWEEP IS WHERE THIS TEST CONNECTS TO D91/D92. Those found correctness
CORRELATIONALLY decodable from the shape specifically at unrolls 6-24 (D91: 68.4% at
6-18, 65.9% at 12-24, chance elsewhere) and via a properly stratified classifier
(D92). If that correlation reflects something the state CAUSALLY carries, patching
in the donor's state at r in [6,24] should flip the recipient's answer more often
than patching at r=0 (too early, the donor's own trajectory has barely diverged from
noise) or r>=32 (D91's tail, where the correlation vanishes). This design tests that
directly rather than inferring it.

PRE-REGISTERED (CLAUDE.md section 1). D47's failure is fixed by construction: every
candidate cell is checked for real dynamic range (a genuine correct AND incorrect
draw) BEFORE it is used, so a repeat of "the outcome never varies" is structurally
impossible here.
P1 -- GATE. At least half the candidate prompts must yield a valid donor/recipient
      pair (both classes found within the seed-search budget) or the run is VOID and
      says so, per D47's own lesson.
P2 -- SANITY. Re-running the recipient's seed with NO patch must reproduce its
      original incorrect label exactly (D90's determinism control, checked again
      here because a patching experiment that cannot even replay itself proves
      nothing about the patch).
P3 -- THE TEST. Does patching at ANY r flip the recipient to correct, scored on
      unrolls FROM r ONWARD ONLY -- ranks before the patch point are causally
      untouched by construction and must not be allowed to count as evidence for it.
P4 -- THE WINDOW. Is flip rate higher for r in {8,16,24} than for r in {0,48}? The
      causal analogue of D91's correlational window, tested rather than assumed.

Candidate prompts are the D90/`geometry-h0bank` boundary set (geomcap rank 1-6),
RESTRICTED to families D89 found have single-token golds -- `add_2d`, `compare`,
`count16` and `echo_word` are dropped here because their rank axis measures the
first token of a multi-token answer, not the answer, and a patching target should
not inherit that ambiguity.

Computes and reports (not B14): this is the one kernel in the project whose entire
point is the intervention's effect, not banked states for later offline analysis.
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
import string
import time
import traceback
import zlib

import numpy as np

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 64
N_ITEMS = 24
OUTDIR = "/kaggle/working"

SEARCH_SEED_CAP = 20        # per class, per prompt; early-stopped once found
# D91's grid restricted to the points that matter for THIS test: 0 (too early),
# the window D91/D92 flagged (8, 16, 24), and two tail points D91 found flat
# (32, 48) -- six points, not eight, to keep the run inside a T4-hour budget
# alongside the seed search.
PATCH_R = (0, 8, 16, 24, 32, 48)

# D89's multi-token-gold families dropped: add_2d, caesar1_word, compare, count16,
# echo_word, rot13_word, sort_min, track_total. What remains from the boundary set.
CANDIDATES = [
    {"family": "add1", "item": 8},
    {"family": "add1", "item": 10},
    {"family": "caesar1_letter", "item": 2},
    {"family": "caesar1_letter", "item": 12},
    {"family": "count4", "item": 1},
    {"family": "count8", "item": 21},
    {"family": "count_mod3", "item": 2},
    {"family": "count_mod3", "item": 4},
    {"family": "last_item", "item": 0},
]


def items(task, n=N_ITEMS):
    """(prompt, gold, distractor) triples. Gold is single-token where possible.

    `zlib.crc32`, NOT `hash()`: Python salts str hashes per process, so a kernel
    seeded with `hash(task)` draws a different item set on every run (D69(3)).
    Verbatim from the battery / `geometry-h0bank`, restricted here to the branches
    CANDIDATES actually uses -- WORDS is dropped since `echo_word` is not called.
    """
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + zlib.crc32(task.encode()) % 997)
        L = string.ascii_lowercase
        if task == "add1":
            v = rng.randint(0, 8)
            out.append((f"What is {v} + 1?", str(v + 1), str((v + 4) % 10)))
        elif task == "count4":
            b = [rng.randint(0, 1) for _ in range(4)]
            g = sum(b)
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 2) % 5)))
        elif task == "count8":
            b = [rng.randint(0, 1) for _ in range(8)]
            g = sum(b)
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 3) % 9)))
        elif task == "count_mod3":                                # modular state
            b = [rng.randint(0, 1) for _ in range(9)]
            g = sum(b) % 3
            out.append(("Count how many ones are in this sequence, then give the "
                        "remainder when divided by 3.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 1) % 3)))
        elif task == "caesar1_letter":                            # easiest cipher cell
            c = rng.choice(L[:25])
            out.append((f"Shift this letter forward by 1 in the alphabet.\n"
                        f"Letter: {c}", L[L.index(c) + 1], L[(L.index(c) + 7) % 26]))
        elif task == "last_item":                                 # recency
            xs = [rng.randint(0, 9) for _ in range(6)]
            out.append((f"What is the last number in this list?\n"
                        f"List: {' '.join(map(str, xs))}", str(xs[-1]),
                        str((xs[-1] + 3) % 10)))
        else:
            raise ValueError(f"items() not defined for task {task!r} in this bundle")
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


def run_forward(model, tok, torch, prompt, gold, h0_seed, patch_r=None,
                donor_states=None):
    """One forward. If `patch_r` is given, the ANSWER-POSITION state at that unroll
    is replaced by `donor_states[patch_r]` and the forward continues causally from
    there -- its own KV cache, its own remaining unrolls. Returns the full per-unroll
    rank curve and state array, so both the untouched prefix and the patched
    continuation are visible in the record.

    The hook fires once per unroll (`core_block[-1]` runs once per recurrent
    iteration), so a closure counter is what turns "the hook fired" into "this is
    unroll i" -- there is no other index available inside the hook.
    """
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device)
    g_ids = tok(gold, add_special_tokens=False).input_ids
    n_p = ids.shape[1]
    freqs = model.freqs_cis[:, :n_p]
    states, ranks = [], []
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()
    step = {"i": 0}

    def hook(_m, _i, o):
        i = step["i"]
        step["i"] += 1
        with torch.no_grad():
            st = o.detach()
            out = None
            if patch_r is not None and i == patch_r:
                # THE INTERVENTION: only the answer-token position is overwritten,
                # not the whole sequence -- a minimal, precise patch, stated in the
                # module docstring so it is never mistaken for a coarser one.
                st = st.clone()
                st[0, n_p - 1, :] = torch.from_numpy(
                    donor_states[patch_r]).to(st.dtype).to(st.device)
                out = st
            states.append(st[0, n_p - 1, :].float().cpu().numpy())
            row = torch.log_softmax(
                coda_head(model, st, freqs).float()[0, n_p - 1], dim=-1)
            ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)
            return out

    h = mod.register_forward_hook(hook)
    try:
        torch.manual_seed(h0_seed)
        with torch.no_grad():
            model(input_ids=ids, num_steps=NUM_STEPS)
    finally:
        h.remove()
    arr = np.stack(states).astype(np.float32)
    return {"n_tokens": int(n_p), "rank_curve": ranks, "states": arr,
            "best_rank": int(min(ranks)), "correct": bool(min(ranks) == 1),
            "state_sha": hashlib.sha256(arr.tobytes()).hexdigest()[:16]}


def find_donor_recipient(model, tok, torch, prompt, gold, cap=SEARCH_SEED_CAP):
    """Deterministic seed search for one CORRECT (donor) and one INCORRECT
    (recipient) h_0 draw of the SAME prompt. Early-stopped per class.

    Seeds are small consecutive integers starting at 0, which doubles as the
    determinism control D90 already validated: `torch.manual_seed(s)` on the SAME
    prompt at the SAME weights reproduces bit-identical states, so two calls with
    the same seed here would (and later do, in the P2 sanity check) agree exactly.
    """
    donor = recipient = None
    for s in range(cap):
        if donor is not None and recipient is not None:
            break
        r = run_forward(model, tok, torch, prompt, gold, h0_seed=s)
        if r["correct"] and donor is None:
            donor = {"seed": s, **r}
        elif not r["correct"] and recipient is None:
            recipient = {"seed": s, **r}
    return donor, recipient, s + 1


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

    manifest = {"pairs": [], "patches": []}
    t0 = time.time()

    def flush():
        with open(os.path.join(OUTDIR, "manifest.json"), "w") as fh:
            json.dump(manifest, fh)

    n_split = 0
    for c in CANDIDATES:
        prompt, gold, _dist = items(c["family"])[c["item"]]
        tag = f"{c['family']}_i{c['item']:02d}"
        try:
            donor, recipient, n_tried = find_donor_recipient(model, tok, torch,
                                                              prompt, gold)
        except Exception as exc:
            print(f"  {tag:>20}: SEARCH FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            print(traceback.format_exc(), flush=True)
            continue
        if donor is None or recipient is None:
            print(f"  {tag:>20}: no split within {n_tried} seeds "
                  f"(donor {'found' if donor else 'MISSING'}, "
                  f"recipient {'found' if recipient else 'MISSING'})", flush=True)
            manifest["pairs"].append({"tag": tag, "family": c["family"],
                                      "item": c["item"], "split": False,
                                      "n_tried": n_tried})
            flush()
            continue

        n_split += 1
        print(f"  {tag:>20}: donor seed {donor['seed']} (rank {donor['best_rank']}), "
              f"recipient seed {recipient['seed']} (rank {recipient['best_rank']}), "
              f"found within {n_tried} seeds", flush=True)
        manifest["pairs"].append({
            "tag": tag, "family": c["family"], "item": c["item"], "split": True,
            "n_tried": n_tried, "donor_seed": donor["seed"],
            "donor_rank_curve": donor["rank_curve"], "donor_sha": donor["state_sha"],
            "recipient_seed": recipient["seed"],
            "recipient_rank_curve": recipient["rank_curve"],
            "recipient_sha": recipient["state_sha"],
        })
        flush()

        # P2 SANITY: replay the recipient's seed with NO patch. Must reproduce the
        # ORIGINAL incorrect result exactly, or nothing below means anything.
        try:
            replay = run_forward(model, tok, torch, prompt, gold,
                                 h0_seed=recipient["seed"])
            sane = (replay["state_sha"] == recipient["state_sha"]
                   and replay["correct"] == recipient["correct"])
        except Exception as exc:
            sane = False
            print(f"    P2 SANITY FAILED: {type(exc).__name__}: {exc}", flush=True)
        print(f"    P2 sanity (no-patch replay reproduces the original): "
              f"{'PASS' if sane else 'FAIL -- results below are not trustworthy'}",
              flush=True)
        if not sane:
            continue

        for r in PATCH_R:
            if r >= len(donor["states"]):
                continue
            try:
                patched = run_forward(model, tok, torch, prompt, gold,
                                      h0_seed=recipient["seed"], patch_r=r,
                                      donor_states=donor["states"])
                post_ranks = patched["rank_curve"][r:]
                flipped = bool(min(post_ranks) == 1) if post_ranks else False
                rec = {"tag": tag, "family": c["family"], "item": c["item"],
                       "patch_r": r, "post_patch_ranks": post_ranks,
                       "flipped_post_patch": flipped,
                       "flipped_full_curve": patched["correct"],
                       "best_rank_post_patch": (min(post_ranks) if post_ranks
                                                else None), "ok": True}
            except Exception as exc:
                rec = {"tag": tag, "family": c["family"], "item": c["item"],
                       "patch_r": r, "ok": False,
                       "why": f"{type(exc).__name__}: {exc}"}
            manifest["patches"].append(rec)
            flush()
            print(f"      r={r:>3}: "
                  + (f"flipped(post)={rec.get('flipped_post_patch')} "
                     f"best_rank_post={rec.get('best_rank_post_patch')}"
                     if rec["ok"] else f"FAILED {rec['why']}"), flush=True)

    print(f"\n=== P1 GATE: {n_split}/{len(CANDIDATES)} prompts split "
          f"(wants >= {len(CANDIDATES) // 2}) ===", flush=True)
    if n_split < len(CANDIDATES) // 2:
        print("  VOID, not null, per D47's lesson: too few prompts had real "
              "dynamic range in the outcome for a patching effect to be visible "
              "even if one exists.", flush=True)

    ok_patches = [p for p in manifest["patches"] if p.get("ok")]
    by_r = {}
    for p in ok_patches:
        by_r.setdefault(p["patch_r"], []).append(p["flipped_post_patch"])
    print("\n=== flip rate by patch unroll (post-patch ranks only) ===", flush=True)
    for r in PATCH_R:
        v = by_r.get(r, [])
        rate = sum(v) / len(v) if v else float("nan")
        print(f"  r={r:>3}: {sum(v)}/{len(v)} flipped ({rate:.0%})" if v
              else f"  r={r:>3}: no data", flush=True)

    free_arm(model, MODEL_ID)
    print(f"\n=== done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    print("DONE", flush=True)


main()
