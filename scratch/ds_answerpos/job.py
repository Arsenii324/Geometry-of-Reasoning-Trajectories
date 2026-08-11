"""A44: what is the trajectory when the answer is REASONED, not emitted at position 1?

THE BLIND SPOT, STATED PLAINLY. Every geometry measurement in this project is **one forward
pass at the last prompt position**: hook `core_block[-1]`, unroll r times, read the gold's rank
at `n_prompt - 1`. That is the trajectory for producing the FIRST generated token. When Huginn
generates 24 tokens there are 24 such forward passes, each with its own r-unroll trajectory,
and only the first has ever been measured.

D166 and D174 established what that first token is: **`The`**, opening 39.4% of 1260 banked
generations, with the rate of opening with the gold flat in depth (p = 0.515). So the whole
rank-based apparatus -- D157, D158, D159, D168, D177, D178, D181 -- measures the readout **while
the model is deciding to write `The`**. The token that carries the answer has never been looked
at.

Recurrent depth is latent test-time scaling; generated reasoning is explicit test-time scaling.
They compose, and D181 shows them trading against each other. This asks the question that
separates them: **when the answer arrives at generated position 8 instead of position 0, does
that forward pass look like computation, or like decoding something already decided?**

THE TWO HYPOTHESES, AND THEY PREDICT OPPOSITE THINGS:

  DECORATION -- the computation finished in the prompt's forward pass (consistent with D159's
    answer being rank-1 by unroll ~4, and D168's gold never leaving the top 35). Then the
    answer-carrying position should be **shallower** than position 0: the token is already
    determined, and the unrolls have nothing left to do.

  COMPUTATION -- the model works while it writes, which is what the literal text suggests:
    `sub1` at r = 32 produces *'The answer is 4 - 1 = 3'*, and the `3` appears only after the
    subtraction has been written out. Then the answer-carrying position should be **deeper**,
    or geometrically distinct, from the prose positions around it.

WHAT IS MEASURED, at EVERY generated position rather than only the first: the gold's rank curve
over all `DEPTH` unrolls, the position's own `rotation_power`, and its last-step residual
`||h_r - h_{r-1}||`. The answer position is located by the same rules D179 ranked -- exact token
match, and a word-boundary containment scan over the text so far.

PREREGISTERED PREDICTIONS:

  P1  REPLICATION GATE. Position 0's rank curve must reproduce the census's shape for these
      families -- `best_depth` around 2-6, `echo_digit` shallow, `add1` around 5. Same items,
      same depth, same scoring. If position 0 does not reproduce, this run is not comparable to
      anything and nothing below may be read.

  P2  PRIMARY -- DEPTH AT THE ANSWER POSITION vs AT POSITION 0, paired within item. Registered
      before running: **DECORATION predicts the answer position is shallower** (its `best_depth`
      smaller), **COMPUTATION predicts deeper or equal with a different geometry**. Either is a
      finding; a null with a stated floor is also a finding.

  P3  THE PROSE POSITIONS ARE THE CONTROL, and they are what makes P2 mean anything. Compare
      the answer position against the OTHER generated positions in the same generation, not
      only against position 0. If every generated position looks alike, the answer position is
      not special and 'the model reasons' is not supported by geometry.

  P4  GEOMETRY, NOT JUST DEPTH. `rotation_power` and the final residual per position. D146
      found rotating orbits carry a 157x larger per-block residual than settling ones, so these
      statistics do separate regimes when a difference exists.

  P5  READ THE RAW OUTPUT. The full generation, the located answer position, and the per-position
      statistics are banked and printed for a sample. D110 printed a verdict over empty strings.
"""

import json
import os
import random
import re
import string
import subprocess
import sys
import time
import zlib

DEPTH = 48
MAX_NEW = 20
N_TEST = 6
N_ITEMS = 24
SEED = 20260811
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 10500

# families where D172/D180 show the answer IS produced, so an answer position exists to find
FAMILIES = ("echo_digit", "add1", "sub1", "sort_min", "compare", "track_total")

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


def contains(text, gold):
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(gold)}(?![A-Za-z0-9])", text,
                          re.IGNORECASE))


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("answerpos.json")
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

    from traj_geom.metrics.dynamics import rotation_power

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tok = AutoTokenizer.from_pretrained(mnt)
    cfg = AutoConfig.from_pretrained(mnt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mnt, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    core_last = model.transformer.core_block[-1]
    stop = {65504, 65505, 65508}
    if getattr(tok, "eos_token_id", None) is not None:
        stop.add(tok.eos_token_id)

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def step(ids, g0):
        """One generated position: its full r-unroll trajectory, read at the LAST position.

        This is the same hook every other kernel uses -- the difference is only that it is
        applied at every generated position rather than once at the prompt's end.
        """
        n = ids.shape[1]
        freqs = model.freqs_cis[:, :n]
        ranks, traj = [], []
        core_last._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                h = o.detach()[0, -1]
                traj.append(h.float().cpu().numpy().copy())
                row = torch.log_softmax(
                    coda_head(o.detach(), freqs).float()[0, n - 1], dim=-1)
                ranks.append(int((row > row[g0]).sum().item()) + 1)

        hh = core_last.register_forward_hook(hook)
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                out = model(input_ids=ids, num_steps=DEPTH)
        finally:
            hh.remove()
        logits = out.logits if hasattr(out, "logits") else out[0]
        nxt = int(logits[:, -1, :].argmax(-1))
        T = np.array(traj, dtype=np.float32)
        return {"next": nxt, "rank_curve": ranks, "best_rank": int(min(ranks)),
                "best_depth": int(np.argmin(ranks)) + 1, "final_rank": int(ranks[-1]),
                "rot": float(rotation_power(T)),
                "resid": float(np.linalg.norm(T[-1] - T[-2]))}

    test = build_items()
    print(f"{len(test)} items over {sorted(set(i['family'] for i in test))}", flush=True)
    t0, rows = time.time(), []
    for n_it, it in enumerate(test):
        text = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        g0 = tok(it["gold"], add_special_tokens=False).input_ids[0]
        per_pos, gen_ids = [], []
        try:
            for pos in range(MAX_NEW):
                s = step(ids, g0)
                s["pos"] = pos
                s["tok"] = tok.decode([s["next"]])
                per_pos.append(s)
                if s["next"] in stop:
                    break
                gen_ids.append(s["next"])
                ids = torch.cat([ids, torch.tensor([[s["next"]]], device=ids.device)], dim=1)
            gen = tok.decode(gen_ids, skip_special_tokens=True) if gen_ids else ""
            ok, why = True, ""
        except Exception as exc:  # noqa: BLE001
            gen, ok, why = "", False, f"{type(exc).__name__}: {exc}"

        # locate the answer: first position whose emitted token IS the gold, else the first
        # position after which the running text contains the gold at a word boundary
        ans_pos = -1
        if ok:
            for s in per_pos:
                if s["tok"].strip().lower() == it["gold"].strip().lower():
                    ans_pos = s["pos"]
                    break
            if ans_pos < 0:
                run_txt = ""
                for s in per_pos:
                    run_txt += s["tok"]
                    if contains(run_txt, it["gold"]):
                        ans_pos = s["pos"]
                        break
        rows.append({**{k: it[k] for k in ("family", "item", "gold", "prompt")},
                     "ok": ok, "why": why, "gen": gen, "n_prompt": int(n_p),
                     "n_pos": len(per_pos), "ans_pos": ans_pos, "per_pos": per_pos})
        if n_it % 6 == 0:
            print(f"  {n_it}/{len(test)} ({time.time() - t0:.0f}s) last gen={gen!r} "
                  f"ans_pos={ans_pos}", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    okr = [r for r in rows if r["ok"]]
    found = [r for r in okr if r["ans_pos"] >= 0]
    print(f"\n=== located the answer in {len(found)}/{len(okr)} generations ===", flush=True)

    print("\n=== P5 RAW: generation, answer position, per-position depth ===", flush=True)
    for r in found[:8]:
        print(f"  {r['family']:12s} gold={r['gold']!r:6s} ans_pos={r['ans_pos']:2d}  "
              f"gen={r['gen']!r}", flush=True)
        print("     pos/tok/best_depth/rot: " + "  ".join(
            f"{s['pos']}:{s['tok']!r}:{s['best_depth']}:{s['rot']:.2f}"
            for s in r["per_pos"][:8]), flush=True)

    import statistics as st
    print("\n=== P1/P2 DEPTH: position 0 vs the ANSWER position (paired) ===", flush=True)
    if found:
        p0 = [r["per_pos"][0]["best_depth"] for r in found]
        pa = [r["per_pos"][r["ans_pos"]]["best_depth"] for r in found]
        print(f"  n={len(found)}  position 0 mean {st.mean(p0):.2f} (median {st.median(p0):.1f})"
              f"   answer position mean {st.mean(pa):.2f} (median {st.median(pa):.1f})",
              flush=True)
        try:
            from scipy.stats import wilcoxon
            print(f"  Wilcoxon paired p = {wilcoxon(p0, pa).pvalue:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  (wilcoxon unavailable: {exc})", flush=True)
        d = st.mean(pa) - st.mean(p0)
        print(f"  -> answer position is {'DEEPER' if d > 0 else 'SHALLOWER'} by {abs(d):.2f} "
              f"unrolls. DECORATION predicted shallower; COMPUTATION predicted deeper.",
              flush=True)

    print("\n=== P3 THE PROSE CONTROL: answer position vs OTHER generated positions ===",
          flush=True)
    if found:
        pa = [r["per_pos"][r["ans_pos"]]["best_depth"] for r in found]
        po = [s["best_depth"] for r in found for s in r["per_pos"]
              if s["pos"] != r["ans_pos"] and s["pos"] != 0]
        if po:
            print(f"  answer positions n={len(pa)} mean {st.mean(pa):.2f} | other generated "
                  f"positions n={len(po)} mean {st.mean(po):.2f}", flush=True)
            try:
                from scipy.stats import mannwhitneyu
                print(f"  Mann-Whitney p = {mannwhitneyu(pa, po).pvalue:.4f}", flush=True)
            except Exception:
                pass

    print("\n=== P4 GEOMETRY per position type ===", flush=True)
    if found:
        for label, sel in (("position 0", lambda r: [r["per_pos"][0]]),
                           ("answer position", lambda r: [r["per_pos"][r["ans_pos"]]]),
                           ("other positions",
                            lambda r: [s for s in r["per_pos"]
                                       if s["pos"] not in (0, r["ans_pos"])])):
            vals = [s for r in found for s in sel(r)]
            if vals:
                print(f"  {label:17s} n={len(vals):4d}  rot {st.mean([v['rot'] for v in vals]):.3f}"
                      f"   resid {st.mean([v['resid'] for v in vals]):.4f}"
                      f"   best_depth {st.mean([v['best_depth'] for v in vals]):.2f}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "families": list(FAMILIES), "depth": DEPTH,
                   "max_new": MAX_NEW, "seed": SEED, "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} items, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
