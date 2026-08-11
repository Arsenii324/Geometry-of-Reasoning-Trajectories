"""A41: does crossing the regime boundary change what the model OUTPUTS? D148, repaired.

WHY THIS EXISTS. D148 is the geometry track's headline behavioural negative -- *a dramatic,
causally-controlled change of dynamical regime produces no change in what the model answers* --
and half of it is now withdrawn. Re-reading its banked run: **`correct` is False in 432 of 432
orbits.** The gold never reaches rank 1 anywhere, in either regime, at any grid point; the best
rank any orbit achieves is 3. "0 of 48 units changed correctness" describes a variable that is
identically zero on both sides of the boundary. There was nothing for the regime to change.

The task was "report the largest/smallest {noun} of the sequence" with golds 6/7/8/9, and
Huginn does not do it. That is the check CLAUDE.md §1 exists for and it was not run.

THE REPAIR IS NOT A HARDER-OR-EASIER TASK, IT IS THE RIGHT DEPENDENT VARIABLE. D148 asked
whether the ANSWER changes and measured whether CORRECTNESS changes. Those come apart exactly
when the model is wrong on both sides -- which is the case here, 432 times out of 432. So this
run measures the **emitted token and the generated text**, which answers D148's actual question
without requiring the model to be right about anything.

D166 makes that measurable and also explains D148's floor: the top ranks are held by prose
openers (`The` x12 of 32, `One` x3, `Number` x2), and a best rank of 3 is exactly what two
permanently-ahead format tokens produce.

THE DESIGN, INHERITED FROM A26 SO THE COMPARISON IS THE SAME ONE. The same four chords, the
same interpolation of `e` between a rotating and a settling noun, the same D141 threshold. For
each chord and sequence, R is measured on a five-point grid straddling D140's crossing t*, the
ADJACENT PAIR that actually straddles the threshold is identified from the measured R (not
assumed from t*), and the model is then run to generation at both members of that pair.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, VOID WITHOUT IT. The t = 0 patch substitutes `e_s` for `e_s` and must
      reproduce the unpatched orbit's R to floating point (|dR| < 1e-9), as A25 and A32 both
      achieved. If it fails, the hook writes something other than `e`.

  P2  THE BOUNDARY MUST ACTUALLY BE CROSSED, per unit, from measured R. Units where no adjacent
      pair on the grid straddles the threshold are DROPPED and counted, not stretched to fit.
      D148 got 48 of 48 with exactly one crossing; a much lower rate here means the grid is
      wrong and the run is void.

  P3  PRIMARY -- DOES THE OUTPUT CHANGE? Across the straddling pair: does the first generated
      token differ? Does the generated text differ at all? Registered before running: D148's
      withdrawn claim predicts **no change in either**. Any appreciable rate of change refutes
      "dynamical epiphenomenon" outright, and the rate is the finding.

  P4  THE CAPABILITY CHECK D148 SKIPPED, RUN FIRST AND REPORTED WHETHER OR NOT IT IS FLATTERING.
      Does the gold appear in the generated text, on either side? If it appears in ~0 of the
      units, this task is beyond the model and **P3's null would be as vacuous as D148's** --
      in which case that is the finding and it is reported as such, not buried.

  P5  WHAT HOLDS THE TOP RANKS, since D166 predicts it is prose. The top-5 tokens at the final
      unroll are banked at both grid points. If the argmax is a format token on both sides, the
      rank-based measures D148 used were reading formatting, and the one-rank shift it found
      has a candidate explanation that has nothing to do with the answer.
"""

import json
import os
import subprocess
import sys
import time

NUM_STEPS = 64
GEN_DEPTH = 32
MAX_NEW = 16
SEED = 20260810
THRESHOLD = 0.6677
TOPK = 5
# (rotating noun, settling noun, D140's crossing t*) -- A26's chords verbatim
CHORDS = (("array", "token", 0.283), ("block", "digit", 0.435),
          ("signal", "letter", 0.528), ("symbol", "element", 0.737))
SEQS = ("1 6 0 1 7 7 8 1", "3 5 5 3 6 1 5 2", "2 4 4 1 7 5 0 9",
        "9 1 3 8 2 6 4 7", "5 0 5 2 8 8 1 3", "7 3 9 4 0 2 6 5")
MARKER = "A"
OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10)
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 11000


def build_items(rng=None):
    return [{"seq": s, "item": i} for i, s in enumerate(SEQS)]


def prompt_for(word, seq, mk=MARKER):
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def gold_for(seq, mk=MARKER):
    d = [int(x) for x in seq.split()]
    return str(max(d) if mk == "A" else min(d))


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("regimeout.json")
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

    d_model = model.config.n_embd
    adapter = model.transformer.adapter
    core_last = model.transformer.core_block[-1]
    stop = {65504, 65505, 65508}
    if getattr(tok, "eos_token_id", None) is not None:
        stop.add(tok.eos_token_id)

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    def prelude_e(ids):
        grabbed = {}

        def pre(_m, inp):
            if "e" not in grabbed:
                grabbed["e"] = inp[0][..., d_model:].detach().clone()

        h = adapter.register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=1)
        finally:
            h.remove()
        return grabbed["e"]

    def patched(donor_e):
        def pre(_m, inp):
            cur = inp[0]
            if donor_e is not None:
                cur = torch.cat([cur[..., :d_model], donor_e], dim=-1)
            return (cur,)
        return pre

    def orbit_R(ids, donor_e):
        traj = []

        def post(_m, _i, o):
            traj.append(o.detach()[0, -1].float().cpu().numpy().copy())

        hs = [adapter.register_forward_pre_hook(patched(donor_e)),
              core_last.register_forward_hook(post)]
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            for h in hs:
                h.remove()
        return float(rotation_power(np.array(traj, dtype=np.float32)))

    def generate_and_top(ids, donor_ids, t, gold):
        """Generation under the patched parameter, plus the final-unroll top-5 (P5).

        THE DONOR `e` IS RECOMPUTED AT EVERY STEP, and it has to be. `e` has shape
        [1, n_tokens, d_model], so a donor captured on the prompt cannot be concatenated
        onto a sequence that has grown by a generated token -- the first version of this
        kernel did exactly that and died at the second token. The intervention is "run this
        prompt with the OTHER noun's parameter", so as the continuation grows the donor
        must grow with it: the donor prompt plus the same generated tokens, re-run through
        the prelude. Both prompts are asserted equal length, so the shapes stay matched.
        """
        cur, dcur = ids, donor_ids
        n_p = ids.shape[1]
        first, top5, gold_rank = None, None, -1
        for step in range(MAX_NEW):
            # both prompts carry the same continuation, so the two `e`s stay shape-matched
            e_t = (1.0 - t) * prelude_e(cur) + t * prelude_e(dcur)
            h = adapter.register_forward_pre_hook(patched(e_t))
            try:
                with torch.no_grad():
                    torch.manual_seed(SEED)
                    out = model(input_ids=cur, num_steps=GEN_DEPTH)
            finally:
                h.remove()
            logits = out.logits if hasattr(out, "logits") else out[0]
            row = torch.log_softmax(logits.float()[0, -1], dim=-1)
            if step == 0:
                lp, ix = torch.topk(row, TOPK)
                top5 = [[tok.decode([int(j)]), round(float(p), 3)]
                        for p, j in zip(lp.tolist(), ix.tolist())]
                g = tok(gold, add_special_tokens=False).input_ids[0]
                gold_rank = int((row > row[g]).sum().item()) + 1
                first = tok.decode([int(row.argmax())])
            nxt = int(row.argmax())
            if nxt in stop:
                break
            nt = torch.tensor([[nxt]], device=cur.device)
            cur = torch.cat([cur, nt], dim=1)
            dcur = torch.cat([dcur, nt], dim=1)
        gen = tok.decode(cur[0, n_p:], skip_special_tokens=True)
        return {"gen": gen, "first": first, "top5": top5, "gold_rank": gold_rank}

    t0, rows, dropped = [], [], []
    t0 = time.time()
    for rot_w, set_w, tstar in CHORDS:
        for it in build_items():
            gold = gold_for(it["seq"])
            ids_r = encode(prompt_for(rot_w, it["seq"]))
            ids_s = encode(prompt_for(set_w, it["seq"]))
            if ids_r.shape[1] != ids_s.shape[1]:
                dropped.append({"chord": f"{rot_w}/{set_w}", "item": it["item"],
                                "why": f"token counts {ids_r.shape[1]} vs {ids_s.shape[1]}"})
                continue
            e_r, e_s = prelude_e(ids_r), prelude_e(ids_s)

            # P1: t = 0 substitutes e_s for e_s on the SETTLING prompt
            null_dev = abs(orbit_R(ids_s, e_s) - orbit_R(ids_s, None))

            grid = []
            for off in OFFSETS:
                t = min(1.0, max(0.0, tstar + off))
                e_t = (1.0 - t) * e_s + t * e_r
                grid.append((t, orbit_R(ids_s, e_t), e_t))

            # P2: find the ADJACENT pair that actually straddles, from measured R
            pair = None
            for a, b in zip(grid, grid[1:]):
                if (a[1] > THRESHOLD) != (b[1] > THRESHOLD):
                    pair = (a, b)
                    break
            if pair is None:
                dropped.append({"chord": f"{rot_w}/{set_w}", "item": it["item"],
                                "why": "no adjacent grid pair straddles the threshold",
                                "Rs": [round(g[1], 3) for g in grid]})
                print(f"  DROP {rot_w}/{set_w} seq{it['item']}: R "
                      f"{[round(g[1], 3) for g in grid]}", flush=True)
                continue

            side = {}
            for (t, R, e_t) in pair:
                side["rot" if R > THRESHOLD else "set"] = \
                    {"t": t, "R": R,
                     **generate_and_top(ids_s, ids_r, t, gold)}
            if len(side) != 2:
                dropped.append({"chord": f"{rot_w}/{set_w}", "item": it["item"],
                                "why": "pair did not resolve to two distinct sides"})
                continue

            rows.append({"chord": f"{rot_w}/{set_w}", "rot_w": rot_w, "set_w": set_w,
                         "tstar": tstar, "item": it["item"], "seq": it["seq"], "gold": gold,
                         "null_dev": null_dev, "grid_R": [round(g[1], 4) for g in grid],
                         "rot": side["rot"], "set": side["set"],
                         "first_differs": side["rot"]["first"] != side["set"]["first"],
                         "gen_differs": side["rot"]["gen"] != side["set"]["gen"],
                         "gold_in_rot": gold in side["rot"]["gen"],
                         "gold_in_set": gold in side["set"]["gen"],
                         "n_tokens": int(ids_s.shape[1]), "ok": True})
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
            r = rows[-1]
            print(f"  {r['chord']} seq{r['item']} gold={gold}: "
                  f"R {r['set']['R']:.3f}->{r['rot']['R']:.3f}  "
                  f"first {r['set']['first']!r}->{r['rot']['first']!r}  "
                  f"differs={r['first_differs']}/{r['gen_differs']}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok = [r for r in rows if r.get("ok")]
    print(f"\n=== P1 INSTRUMENT NULL ===", flush=True)
    if ok:
        mx = max(r["null_dev"] for r in ok)
        print(f"  max |dR| at t=0 over {len(ok)} units: {mx:.3e} "
              f"{'OK' if mx < 1e-9 else 'FAIL -- nothing below may be read'}", flush=True)
    print(f"\n=== P2 BOUNDARY CROSSED ===\n  {len(ok)} units usable, "
          f"{len(dropped)} dropped", flush=True)

    print(f"\n=== P4 CAPABILITY CHECK (the one D148 skipped) ===", flush=True)
    if ok:
        gr = sum(r["gold_in_rot"] for r in ok)
        gs = sum(r["gold_in_set"] for r in ok)
        print(f"  gold appears in the generation: rotating {gr}/{len(ok)}, "
              f"settling {gs}/{len(ok)}", flush=True)
        if gr == 0 and gs == 0:
            print("  ZERO on both sides -- this task is beyond the model, and P3's null "
                  "would be as vacuous as D148's. That is the finding.", flush=True)

    print(f"\n=== P3 PRIMARY: DOES THE OUTPUT CHANGE ACROSS THE BOUNDARY? ===", flush=True)
    if ok:
        print(f"  first token differs: {sum(r['first_differs'] for r in ok)}/{len(ok)}",
              flush=True)
        print(f"  generated text differs: {sum(r['gen_differs'] for r in ok)}/{len(ok)}",
              flush=True)

    print(f"\n=== P5 WHAT HOLDS THE TOP RANK (literal, both sides) ===", flush=True)
    for r in ok[:8]:
        print(f"  {r['chord']} seq{r['item']} gold={r['gold']}", flush=True)
        print(f"     settling top5 {r['set']['top5']}", flush=True)
        print(f"     rotating top5 {r['rot']['top5']}", flush=True)
        print(f"     gen set={r['set']['gen']!r}\n     gen rot={r['rot']['gen']!r}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "dropped": dropped, "chords": [list(c) for c in CHORDS],
                   "threshold": THRESHOLD, "seed": SEED, "num_steps": NUM_STEPS,
                   "gen_depth": GEN_DEPTH, "max_new": MAX_NEW,
                   "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} units, {len(dropped)} dropped, {time.time() - t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
