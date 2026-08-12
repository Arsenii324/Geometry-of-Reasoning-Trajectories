"""A47: does a donor state injected at h_0 reach the READOUT before the map forgets it?

THE WINDOW THIS EXPLOITS, AND WHY THE NAIVE VERSION IS WORTHLESS. D173 showed the regime is
memoryless in the PARAMETER: 62 of 64 mid-trajectory `e` swaps take, at switch points from
unroll 1 to 48, with relaxation median **15.5** unrolls entering rotation and **1.0** leaving.
With `e` left intact, an injected STATE should be washed out by the same contraction
(rho ~ 0.79-0.87). So "inject h_0, read the final answer" has a confident prior of no effect,
and a 0% arm could not be told from the capability floor that made D148's null vacuous.

**But the answer is settled at median unroll ~4** (D159; A44 measures position-0 `best_depth`
= 4.07) **while relaxation takes ~15**. That leaves roughly ten unrolls in which history could
still reach the readout even though the trajectory demonstrably ends up in the same place.
Nobody has looked there: D173's statistic is period-6 power over the LAST 24 unrolls, which is
blind to exactly this window.

**So this is the sharpest available test of H3.** Contraction is supposed to exclude a running
register, yet the limit set has diameter ~1 in 5279 tangent dimensions -- room to spare. If an
injected state moves the transient readout, that is a demonstrated information channel through
`h`. If it does not, while P2 confirms the injection took, H3's strong form gets its first
direct behavioural test instead of an inference from rho.

DESIGN CONSTRAINTS, EACH FROM A NAMED PAST FAILURE.

  * **Inject ONCE, at h_0.** `ds_estream` documents that overwriting the state every unroll
    pins the trajectory and measures nothing. `e` is a parameter and may be held fixed; `h` is
    the state and may not. Done by monkeypatching `initialize_state`, which is called exactly
    once per forward.
  * **Seed every forward.** `initialize_state` draws from the global RNG (D78) and no kernel
    seeded it before that row; `torch.manual_seed(SEED)` precedes every forward here, so the
    unpatched baseline is reproducible rather than noise.
  * **Norms are measured, not assumed.** A natural h_0 is `trunc_normal(std=0.0087) *
    embed_scale`, per-component sd 0.632, so **||h_0|| ~ 45.3** -- while mid-trajectory states
    sit at **76.386** by RMSNorm (D99, D187a). Injecting a converged h* as h_0 therefore
    injects something ~1.7x oversized. Both norms are banked, and a **norm-matched arm** is
    included so content and magnitude can be told apart.
  * **Run where the model is right.** RC6 and D148: an arm with no variance proves nothing.
    A42 at k = 2 gives `echo_digit`, `add1`, `sub1` final-unroll accuracy **1.000**, so those
    three run two-shot here and the measured base rate is printed in the output.

PREREGISTERED GATES -- THE RUN IS VOID WITHOUT P1.

  P1  Injecting the run's OWN h_0 must reproduce the unpatched orbit at **0.000e+00** on the
      whole rank curve, as A25, A32, A39 and A41 each achieved on their own null. Any
      deviation means the monkeypatch is not writing what `initialize_state` would have.

  P2  RELAXATION, REPORTED PER UNROLL: `||h_t - h_t(unpatched)||`. **If the injection is gone
      by unroll 2, nothing downstream is interpretable.** If it persists past 15, D173's
      timescale does not transfer from parameter to state, and that alone is the finding.

  P3  PRIMARY -- the recipient gold's rank curve, patched vs unpatched, at every unroll.
      Registered both ways: **a change at unrolls 1-8 that vanishes by 32 is a transient memory
      channel**; **no change at any unroll, with P2 confirming the injection took, is direct
      behavioural evidence for H3's strong form.**

  P4  THE DONOR'S OWN ANSWER, and this is the informative arm. The DONOR item's gold is tracked
      in the RECIPIENT's rank curve. If the recipient briefly ranks the donor's answer, the
      state carries **content**, not merely perturbation.

  P6  IS h_0 ITSELF GOOD OR BAD? Six independent h_0 draws per item, nothing injected, only
      the seed changed. If correctness and `best_depth` vary across draws, the model's answer
      depends on a random tensor nobody controls -- and every single-seed number in this
      project inherits that variance. If they are identical, h_0 is irrelevant and D78's
      seeding requirement is about reproducibility only, not about outcomes.

  P7  DOES h_0 MATTER MORE AT THE ANSWER POSITION? h_0 has shape [1, n_tokens, d]. Two hybrid
      arms: the run's own h_0 everywhere EXCEPT the last position (redrawn there), and the
      converse. The readout is taken at the last position, so if only `mix_last` moves the
      curve the state's influence is local to where it is read; if only `mix_stmt` moves it,
      the statement's positions carry it.

  P5  DONOR LADDER, weakest to strongest: the run's own h_0 (gate) -> a redrawn h_0 from a
      different seed -> a same-family different-item state at unroll 4 -> that item's converged
      h* -> a DIFFERENT-family h* -> the same-family h* rescaled to the natural h_0 norm. If
      only the cross-family arm moves anything, the channel is coarse; if the rescaled arm
      behaves like the unscaled one, the effect is content rather than magnitude.
"""

import json
import os
import random
import string
import subprocess
import sys
import time
import zlib

DEPTH = 48
SHOTS = 2
N_TEST = 6
N_ITEMS = 24
SEED = 20260811
ALT_SEED = 76543
TOPK = 5
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 10000

FAMILIES = ("echo_digit", "add1", "sub1")     # A42 k=2 final-unroll accuracy 1.000 on all three

WORDS = ("apple", "chair", "river", "stone", "bread", "cloud", "green", "horse",
         "light", "money", "night", "paper", "queen", "table", "water", "youth")

def items(task, n=N_ITEMS):
    """(prompt, gold, distractor) triples. Gold is single-token where possible.

    `zlib.crc32`, NOT `hash()`: Python salts str hashes per process, so a kernel
    seeded with `hash(task)` draws a different item set on every run (D69(3)).
    """
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + zlib.crc32(task.encode()) % 997)
        L = string.ascii_lowercase
        if task == "echo_digit":                                  # trivial retrieval
            v = rng.randint(0, 9)
            out.append((f"Repeat this number exactly.\nNumber: {v}", str(v),
                        str((v + 3) % 10)))
        elif task == "echo_word":
            w = rng.choice(WORDS)
            out.append((f"Repeat this word exactly.\nWord: {w}", w,
                        rng.choice([x for x in WORDS if x != w])))
        elif task == "nth_item":                                  # indexing
            xs = [rng.randint(0, 9) for _ in range(5)]
            k = rng.randint(1, 5)
            out.append((f"What is item number {k} in this list?\n"
                        f"List: {' '.join(map(str, xs))}", str(xs[k - 1]),
                        str((xs[k - 1] + 3) % 10)))
        elif task == "last_item":                                 # recency
            xs = [rng.randint(0, 9) for _ in range(6)]
            out.append((f"What is the last number in this list?\n"
                        f"List: {' '.join(map(str, xs))}", str(xs[-1]),
                        str((xs[-1] + 3) % 10)))
        elif task == "add1":
            v = rng.randint(0, 8)
            out.append((f"What is {v} + 1?", str(v + 1), str((v + 4) % 10)))
        elif task == "sub1":
            v = rng.randint(1, 9)
            out.append((f"What is {v} - 1?", str(v - 1), str((v + 4) % 10)))
        elif task == "add_2d":                                    # harder arithmetic
            a, b = rng.randint(10, 49), rng.randint(10, 49)
            out.append((f"What is {a} + {b}?", str(a + b), str(a + b + 3)))
        elif task == "compare":
            a, b = rng.sample(range(1, 100), 2)
            out.append((f"Which number is larger, {a} or {b}?", str(max(a, b)),
                        str(min(a, b))))
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
        elif task == "count16":
            b = [rng.randint(0, 1) for _ in range(16)]
            g = sum(b)
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 3) % 17)))
        elif task == "count_mod3":                                # modular state
            b = [rng.randint(0, 1) for _ in range(9)]
            g = sum(b) % 3
            out.append(("Count how many ones are in this sequence, then give the "
                        "remainder when divided by 3.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str((g + 1) % 3)))
        elif task == "parity8":                                   # theory: impossible
            b = [rng.randint(0, 1) for _ in range(8)]
            g = sum(b) % 2
            out.append(("Is the number of ones in this sequence even or odd? "
                        "Answer 0 for even and 1 for odd.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(g), str(1 - g)))
        elif task == "track_total":                               # accumulation (H2's track)
            ops = [rng.choice([1, -1]) for _ in range(8)]
            g = sum(ops)
            body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1."
                                             for o in ops)
            out.append((body + " Final total?", str(g), str(g + 2)))
        elif task == "local_last":                                # length-matched control
            ops = [rng.choice([1, -1]) for _ in range(8)]
            g = "Add" if ops[-1] > 0 else "Subtract"
            body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1."
                                             for o in ops)
            out.append((body + " What was the last instruction?", g,
                        "Subtract" if g == "Add" else "Add"))
        elif task == "max_run":                                   # order-dependent
            b = [rng.randint(0, 1) for _ in range(12)]
            best = cur = 1
            for i in range(1, len(b)):
                cur = cur + 1 if b[i] == b[i - 1] else 1
                best = max(best, cur)
            out.append(("What is the length of the longest run of identical "
                        "symbols in this sequence?\n"
                        f"Sequence: {' '.join(map(str, b))}", str(best),
                        str(best + 1)))
        elif task == "sort_min":
            xs = rng.sample(range(1, 100), 3)
            out.append((f"What is the smallest of these numbers: "
                        f"{xs[0]}, {xs[1]}, {xs[2]}?", str(min(xs)),
                        str(sorted(xs)[1])))
        elif task == "succ_letter":
            c = rng.choice(L[:25])
            out.append((f"What letter comes after '{c}' in the alphabet?",
                        L[L.index(c) + 1], L[(L.index(c) + 5) % 26]))
        elif task == "caesar1_letter":                            # easiest cipher cell
            c = rng.choice(L[:25])
            out.append((f"Shift this letter forward by 1 in the alphabet.\n"
                        f"Letter: {c}", L[L.index(c) + 1], L[(L.index(c) + 7) % 26]))
        elif task == "caesar1_word":
            w = "".join(rng.choice(L) for _ in range(4))
            enc = "".join(L[(L.index(c) + 1) % 26] for c in w)
            out.append((f"Shift each letter of this text backward by 1 in the "
                        f"alphabet.\nText: {enc}", w,
                        "".join(rng.choice(L) for _ in range(4))))
        else:                                                     # rot13_word
            w = "".join(rng.choice(L) for _ in range(4))
            enc = "".join(L[(L.index(c) + 13) % 26] for c in w)
            out.append((f"Decode this ROT13 text.\nText: {enc}", w,
                        "".join(rng.choice(L) for _ in range(4))))
    return out



def build_items(rng=None):
    out = []
    for task in FAMILIES:
        for i, (prompt, gold, _d) in enumerate(items(task)[:N_TEST]):
            out.append({"family": task, "item": i, "prompt": prompt, "gold": gold})
    return out


def shot_pool(task, k=SHOTS):
    """Exemplars whose prompts are disjoint from the test set (A42's P5 leak, fixed there)."""
    test = {p for p, _, _ in items(task)[:N_TEST]}
    seen, out = set(), []
    for prompt, gold, distractor in items(task, n=400):
        if prompt in test or prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, gold, distractor))
        if len(out) >= k:
            break
    return out


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("h0inject.json")
    mnt = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"results -> {out_path}; weights mount -> {mnt}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tok = AutoTokenizer.from_pretrained(mnt)
    cfg = AutoConfig.from_pretrained(mnt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mnt, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    core_last = model.transformer.core_block[-1]
    orig_init = model.initialize_state
    DONOR = {"v": None}
    CAPTURED = {}

    def patched_init(input_embeds, scale: float = 1.0):
        """Injects ONCE, because initialize_state is called once per forward."""
        if DONOR["v"] is not None:
            CAPTURED["h0"] = DONOR["v"].detach().clone()
            return DONOR["v"].clone()
        h = orig_init(input_embeds, scale)
        CAPTURED["h0"] = h.detach().clone()
        return h

    model.initialize_state = patched_init

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def encode(body):
        text = tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    def prompt_for(it):
        pre = "".join(f"{p}\nAnswer: {g}\n\n" for p, g, _ in shot_pool(it["family"]))
        return pre + it["prompt"]

    def orbit(ids, gold_ids, donor=None, seed=SEED):
        """Full trajectory + per-unroll rank of every id in `gold_ids` (recipient, donor)."""
        n = ids.shape[1]
        freqs = model.freqs_cis[:, :n]
        ranks = {g: [] for g in gold_ids}
        traj, tops = [], []
        core_last._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                h = o.detach()[0, -1]
                traj.append(h.float().cpu().numpy().copy())
                row = torch.log_softmax(
                    coda_head(o.detach(), freqs).float()[0, n - 1], dim=-1)
                for g in gold_ids:
                    ranks[g].append(int((row > row[g]).sum().item()) + 1)
                if len(traj) in (1, 2, 4, 8, 16, 32, DEPTH):
                    lp, ix = torch.topk(row, TOPK)
                    tops.append([len(traj), [[tok.decode([int(j)]), round(float(p), 3)]
                                             for p, j in zip(lp.tolist(), ix.tolist())]])

        DONOR["v"] = donor
        CAPTURED.pop("h0", None)
        h = core_last.register_forward_hook(hook)
        try:
            with torch.no_grad():
                torch.manual_seed(seed)
                model(input_ids=ids, num_steps=DEPTH)
        finally:
            h.remove()
            DONOR["v"] = None
        return {"ranks": {str(k): v for k, v in ranks.items()}, "tops": tops,
                "traj": np.array(traj, dtype=np.float32), "h0": CAPTURED.get("h0")}

    test = build_items()
    print(f"{len(test)} recipients over {FAMILIES}, {SHOTS}-shot, depth {DEPTH}", flush=True)

    t0, rows, dropped = time.time(), [], []
    for n_it, it in enumerate(test):
        try:
            ids = encode(prompt_for(it))
            g_self = tok(it["gold"], add_special_tokens=False).input_ids[0]

            # ---- donor sources -------------------------------------------------
            same = [x for x in test if x["family"] == it["family"] and x["item"] != it["item"]]
            xfam = [x for x in test if x["family"] != it["family"]]
            d_same = same[0] if same else None
            d_xfam = xfam[0] if xfam else None
            if d_same is None or d_xfam is None:
                dropped.append({"item": it, "why": "no donor available"}); continue
            ids_same, ids_xfam = encode(prompt_for(d_same)), encode(prompt_for(d_xfam))
            # NO LENGTH GATE, and the first two runs died for want of this comment.
            # The original required every donor prompt to match the recipient's token count.
            # `echo_digit` encodes to 49 tokens and `add1`/`sub1` to 40, so the CROSS-FAMILY
            # donor can never match -- 18 of 18 items were dropped, `ok` was empty, and the
            # kernel hit its own "NO USABLE ITEMS" guard and returned 1. DataSphere reported
            # ERROR with no output file, which is exactly what was observed twice.
            # The gate was never needed: a donor is injected as ONE [d] vector tiled by
            # `tile()` to the RECIPIENT's length, so the donor prompt's own length never
            # enters the injection. The donor's orbit is a separate forward on its own ids.
            g_don = tok(d_same["gold"], add_special_tokens=False).input_ids[0]
            g_don_x = tok(d_xfam["gold"], add_special_tokens=False).input_ids[0]
            # De-duplicate: `ranks` is keyed by token id, so a recipient and donor sharing a
            # gold (3 of 18 items here -- both are single digits) made the hook append twice
            # per unroll and doubled the rank curve. Verified against the toy model.
            #
            # THE THIRD RUN DIED HERE, INDIRECTLY. De-duplication means `base_ranks` has ONE
            # key whenever the donor and recipient share a gold, and the P4 scoring block then
            # did `list(r["base_ranks"].keys())[1]` -> IndexError, unhandled, after 77 minutes
            # of compute and before the results file was written. Both golds are now recorded
            # by id so no scoring step indexes this dict by position. Also: the cross-family
            # donor has its OWN gold, which the old code never tracked -- P4 was scoring the
            # star_xfam arm against the same-family donor's answer.
            gold_ids = sorted({g_self, g_don, g_don_x})

            base = orbit(ids, gold_ids)                       # unpatched reference
            h0_own = base["h0"]
            don_same = orbit(ids_same, [g_don])
            don_xfam = orbit(ids_xfam, [g_don_x])

            def as_state(arr):
                return torch.tensor(arr, device=model.device, dtype=torch.float32)

            # a donor state must be shaped like h_0: [1, n_tokens, d]. The banked traj is the
            # LAST position only, so tile it across positions -- stated, not hidden.
            def tile(vec):
                return as_state(vec).view(1, 1, -1).repeat(1, ids.shape[1], 1)

            h0_alt = None
            DONOR["v"] = None
            torch.manual_seed(ALT_SEED)
            with torch.no_grad():
                model(input_ids=ids, num_steps=1)
            h0_alt = CAPTURED.get("h0")

            star_same = tile(don_same["traj"][-1])
            n_h0 = float(h0_own.norm())
            n_star = float(star_same.norm())
            arms = {
                "own_h0":      h0_own,                                   # P1 gate
                "redraw_h0":   h0_alt,                                   # rung 2
                "mid4_same":   tile(don_same["traj"][3]),                # rung 3
                "star_same":   star_same,                                # rung 4
                "star_xfam":   tile(don_xfam["traj"][-1]),               # rung 5
                "star_scaled": star_same * (n_h0 / max(n_star, 1e-9)),   # rung 6, norm-matched
            }
            # P7: hybrids -- own h_0 except the last position, and the converse
            mix_stmt, mix_last = h0_own.clone(), None
            if h0_alt is not None:
                mix_stmt[:, -1, :] = h0_alt[:, -1, :]
                mix_last = h0_alt.clone()
                mix_last[:, -1, :] = h0_own[:, -1, :]
            arms["mix_lastpos_redrawn"] = mix_stmt
            arms["mix_stmtpos_redrawn"] = mix_last

            # P6: six independent h_0 draws, nothing injected, only the seed changed
            seed_sweep = []
            for si, sd in enumerate((SEED, ALT_SEED, 11111, 22222, 33333, 44444)):
                o = orbit(ids, gold_ids, donor=None, seed=sd)
                rc = o["ranks"][str(g_self)]
                seed_sweep.append({"seed": sd, "best_rank": int(min(rc)),
                                   "best_depth": int(min(range(len(rc)), key=lambda i: rc[i])) + 1,
                                   "final_rank": rc[-1], "oracle": int(min(rc) == 1),
                                   "final_correct": int(rc[-1] == 1),
                                   "h0_norm": float(o["h0"].norm()) if o["h0"] is not None else -1.0})

            rec = {"family": it["family"], "item": it["item"], "gold": it["gold"],
                   "seed_sweep": seed_sweep,
                   "donor_gold": d_same["gold"], "donor_gold_xfam": d_xfam["gold"],
                   "gid_self": int(g_self), "gid_don": int(g_don), "gid_don_xfam": int(g_don_x),
                   "n_tokens": int(ids.shape[1]),
                   "norm_h0": n_h0, "norm_star": n_star, "ok": True,
                   "base_ranks": base["ranks"], "base_tops": base["tops"], "arms": {}}
            for name, donor in arms.items():
                if donor is None:
                    continue
                o = orbit(ids, gold_ids, donor=donor)
                dev = [float(np.linalg.norm(a - b))
                       for a, b in zip(o["traj"], base["traj"])]
                rec["arms"][name] = {
                    "ranks": o["ranks"], "tops": o["tops"], "dev": dev,
                    "donor_norm": float(donor.norm()),
                    "rank_delta": [a - b for a, b in
                                   zip(o["ranks"][str(g_self)], base["ranks"][str(g_self)])],
                    "donor_gold_shared": bool(g_don == g_self),
                }
            rows.append(rec)
        except Exception as exc:  # noqa: BLE001
            rows.append({"family": it["family"], "item": it["item"], "ok": False,
                         "why": f"{type(exc).__name__}: {exc}"})
        if n_it % 4 == 0:
            print(f"  {n_it}/{len(test)} ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump([{k: v for k, v in r.items() if k != "arms"} for r in rows], f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok = [r for r in rows if r.get("ok")]
    import statistics as st
    print(f"\n{len(ok)} usable, {len(dropped)} dropped for length mismatch", flush=True)
    if dropped:
        print(f"  drop reasons: {[d.get('why') for d in dropped][:4]}", flush=True)
    if not ok:
        print("NO USABLE ITEMS -- refusing to score.", flush=True)
        with open(out_path, "w") as f:
            json.dump({"rows": rows, "dropped": dropped}, f)
        return 1

    # BANK FIRST. The third run spent 77 minutes computing, then died in the scoring block
    # below with an IndexError, and DataSphere reported ERROR with no output file because
    # out_path was only written at the very end. Results are now on disk before anything is
    # printed, and the whole scoring section is wrapped so a summary bug can never again cost
    # a run. RC9's lesson, applied one level up: a kernel should not be able to lose data it
    # has already computed.
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "dropped": dropped, "banked_before_scoring": True,
                   "families": list(FAMILIES), "depth": DEPTH, "shots": SHOTS,
                   "seed": SEED, "alt_seed": ALT_SEED,
                   "elapsed_s": time.time() - t0}, f)
    print(f"\nBANKED {len(ok)} usable rows to {out_path} before scoring", flush=True)

    try:
        _score(ok, rows, dropped, st)
    except Exception as exc:  # noqa: BLE001
        print(f"\nSCORING FAILED ({type(exc).__name__}: {exc}) -- data is already banked, "
              f"re-score offline from the JSON.", flush=True)
    return 0


def _score(ok, rows, dropped, st):
    """Printing only. Must not write files and must not raise into main():
    the data is already on disk by the time this runs."""
    print("\n=== BASE RATE (RC6: an arm with no variance proves nothing) ===", flush=True)
    for fam in FAMILIES:
        v = [r for r in ok if r["family"] == fam]
        if v:
            g = [r["base_ranks"][str(list(map(int, r["base_ranks"].keys()))[0])] for r in v]
            fin = sum(1 for r in v
                      if list(r["base_ranks"].values())[0][-1] == 1) / len(v)
            orc = sum(1 for r in v if min(list(r["base_ranks"].values())[0]) == 1) / len(v)
            print(f"  {fam:11s} n={len(v)}  unpatched oracle {orc:.3f}  final-unroll {fin:.3f}",
                  flush=True)

    print("\n=== P1 GATE: own_h0 must reproduce the unpatched rank curve exactly ===",
          flush=True)
    worst = 0
    for r in ok:
        a = r["arms"].get("own_h0")
        if not a:
            continue
        gs = list(r["base_ranks"].keys())[0]
        d = max(abs(x - y) for x, y in zip(a["ranks"][gs], r["base_ranks"][gs]))
        worst = max(worst, d)
    print(f"  max rank deviation over {len(ok)} items: {worst}  "
          f"{'OK (0.000e+00)' if worst == 0 else 'FAIL -- nothing below may be read'}",
          flush=True)

    print("\n=== P2 RELAXATION: ||h_t - h_t(unpatched)|| by unroll ===", flush=True)
    for name in ("redraw_h0", "mid4_same", "star_same", "star_xfam", "star_scaled"):
        devs = [r["arms"][name]["dev"] for r in ok if name in r["arms"]]
        if not devs:
            continue
        m = [st.mean(d[i] for d in devs) for i in range(min(len(x) for x in devs))]
        idx = [0, 1, 3, 7, 15, 23, 31, 47]
        print(f"  {name:12s} " + "  ".join(f"u{i+1}={m[i]:7.2f}" for i in idx if i < len(m)),
              flush=True)

    print("\n=== P3 PRIMARY: recipient gold rank delta (patched - unpatched) ===", flush=True)
    for name in ("redraw_h0", "mid4_same", "star_same", "star_xfam", "star_scaled"):
        ds = [r["arms"][name]["rank_delta"] for r in ok if name in r["arms"]]
        if not ds:
            continue
        m = [st.mean(d[i] for d in ds) for i in range(min(len(x) for x in ds))]
        nz = sum(1 for d in ds if any(x != 0 for x in d[:8]))
        idx = [0, 1, 3, 7, 15, 31, 47]
        print(f"  {name:12s} " + "  ".join(f"u{i+1}={m[i]:+7.2f}" for i in idx if i < len(m))
              + f"   items moved in u1-8: {nz}/{len(ds)}", flush=True)

    print("\n=== P4 DONOR CONTENT: donor gold's rank in the RECIPIENT's curve ===", flush=True)
    for name in ("redraw_h0", "star_same", "star_xfam", "star_scaled"):
        rows_d = [r for r in ok if name in r["arms"]]
        if not rows_d:
            continue
        out = []
        for i in (0, 1, 3, 7, 15, 47):
            base_v, pat_v = [], []
            for r in rows_d:
                # by recorded id, never by dict position: the two can coincide (see above).
                gd = str(r.get("gid_don_xfam" if name == "star_xfam" else "gid_don", ""))
                if gd == str(r.get("gid_self")) or gd not in r["base_ranks"]:
                    continue          # donor and recipient share a gold: no content signal
                if i < len(r["base_ranks"][gd]):
                    base_v.append(r["base_ranks"][gd][i])
                    pat_v.append(r["arms"][name]["ranks"][gd][i])
            if base_v:
                out.append(f"u{i+1}: {st.median(base_v):.0f}->{st.median(pat_v):.0f}")
        print(f"  {name:12s} donor-gold median rank  " + "  ".join(out), flush=True)

    print("\n=== P6 IS h_0 ITSELF GOOD OR BAD? six draws per item, nothing injected ===",
          flush=True)
    sw = [r["seed_sweep"] for r in ok if r.get("seed_sweep")]
    if sw:
        n_var_or = sum(1 for s in sw if len({x["oracle"] for x in s}) > 1)
        n_var_fc = sum(1 for s in sw if len({x["final_correct"] for x in s}) > 1)
        n_var_bd = sum(1 for s in sw if len({x["best_depth"] for x in s}) > 1)
        print(f"  items whose ORACLE differs across h_0 draws:      {n_var_or}/{len(sw)}",
              flush=True)
        print(f"  items whose FINAL-unroll answer differs:          {n_var_fc}/{len(sw)}",
              flush=True)
        print(f"  items whose best_depth differs:                   {n_var_bd}/{len(sw)}",
              flush=True)
        spread = [max(x["best_depth"] for x in s) - min(x["best_depth"] for x in s) for s in sw]
        print(f"  best_depth spread across draws: median {st.median(spread):.1f} "
              f"max {max(spread)}", flush=True)
        h0n = [x["h0_norm"] for s in sw for x in s if x["h0_norm"] > 0]
        if h0n:
            print(f"  ||h_0|| across all draws: mean {st.mean(h0n):.2f} "
                  f"(predicted ~45.3 from init_values std x embed_scale, D187a)", flush=True)

    print("\n=== P7 DOES h_0 MATTER MORE AT THE ANSWER POSITION? ===", flush=True)
    for name in ("mix_lastpos_redrawn", "mix_stmtpos_redrawn", "redraw_h0"):
        ds = [r["arms"][name]["rank_delta"] for r in ok if name in r["arms"]]
        if not ds:
            continue
        nz = sum(1 for d in ds if any(x != 0 for x in d[:8]))
        m = [st.mean(d[i] for d in ds) for i in range(min(len(x) for x in ds))]
        print(f"  {name:22s} moved in u1-8: {nz}/{len(ds)}   "
              + "  ".join(f"u{i+1}={m[i]:+6.2f}" for i in (0, 3, 7, 47) if i < len(m)),
              flush=True)

    print("\n=== P5/raw: literal top-5 where anything moved (first 3 items, star_xfam) ===",
          flush=True)
    for r in ok[:3]:
        a = r["arms"].get("star_xfam")
        if not a:
            continue
        print(f"  {r['family']} item{r['item']} gold={r['gold']!r} donor_gold="
              f"{r['donor_gold']!r}  ||h0||={r['norm_h0']:.1f} ||h*||={r['norm_star']:.1f}",
              flush=True)
        for (u, t_) in a["tops"][:3]:
            print(f"     patched  u{u}: {t_}", flush=True)
        for (u, t_) in r["base_tops"][:3]:
            print(f"     baseline u{u}: {t_}", flush=True)

    print(f"SCORED {len(ok)} items", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
