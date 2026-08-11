"""A52: does the regime EXIST YET when the answer is decided?

THE STRUCTURAL MISMATCH, REGISTERED AS §N3 BEFORE THIS RUN. Two numbers in this record are
measured on different halves of the same trajectory and have always been compared as though
they were not:

  * **the regime** is `rotation_power(period=6, tail=24)` -- the **last 24** unrolls. A49's
    first pass failed its own gate precisely because a 48-unroll window let the damping
    transient dominate the low-frequency bins, so the statistic is deliberately about the
    **settled** part.
  * **the answer** is best-ranked at median unroll **~4** (D159), and position-0 `best_depth`
    is **4.07** (D184).

Every test of *"does the regime affect the answer"* compares one window to the other. D148
found 0 of 48 (on a variable with no variance); D176 re-ran it properly and found **1 of 18**,
p = 1.0. **If the regime is a property of unrolls 40-64 and the answer is chosen around unroll
4, those nulls may say nothing about the regime's causal role -- the dynamics being manipulated
had not started when the answer was decided.**

WHAT THIS MEASURES. `rotation_power` at the same period on two windows of the same orbit:

    R_early = rotation_power(traj[:12], period=6, tail=12)   # unrolls 0-11, the decision window
    R_tail  = rotation_power(traj,      period=6, tail=24)   # unrolls 40-63, the regime as defined

`tail=12` is the smallest window the statistic accepts at period 6 -- exactly two cycles -- so
this is the earliest a period-6 orbit is even definable. **That bound is itself part of the
answer: if the answer is settled at unroll 4, no period-6 statistic can describe the state at
the moment of decision, because two cycles have not elapsed.**

PREREGISTERED PREDICTIONS:

  P1  REPLICATION GATE, VOID WITHOUT IT. `R_tail` must reproduce this project's regime labels:
      `symbol`-type prompts above D141's 0.6677, `element`-type below, as in D132/D141/D192. If
      the tail statistic does not reproduce, this run is not measuring the same object.

  P2  PRIMARY. Is there ANY period-6 power in the decision window? Registered before running:
      **if `R_early` is near zero for rotating and settling prompts alike, the regime does not
      exist yet when the answer is chosen**, and D148/D176's nulls are explained without
      appealing to the regime being epiphenomenal. **If `R_early` already separates the two
      populations, the regime is present early and those nulls stand as genuine behavioural
      nulls.**

  P3  WHEN DOES IT SWITCH ON? `R` over a sliding 12-unroll window across the whole orbit, so
      the onset time is measured rather than inferred from two endpoints. D192 found the
      regime is positionally structured; this asks whether it is temporally structured too.

  P4  AGAINST CORRECTNESS, which is the point of asking. On census families where correctness
      varies, does `R_early` predict `oracle` where `R_tail` does not? Both are tested against
      the same items with the same test, so a difference between them is not a difference of
      design.

  P5  THE FLOOR, because a fraction of nothing is meaningless. Total non-DC power is banked for
      both windows. If the early window carries almost no power at all, `R_early` is a ratio of
      noise and must be reported as such rather than as a low value.
"""

import json
import os
import subprocess
import sys
import time

NUM_STEPS = 64
EARLY = 12                 # two period-6 cycles: the earliest the statistic is defined
TAIL = 24                  # rotation_power's default, and D141's calibration
SEED = 20260811
THRESHOLD = 0.6677
MARKER = "A"
SEQS = ("1 6 0 1 7 7 8 1", "3 5 5 3 6 1 5 2", "2 4 4 1 7 5 0 9")
ROT_NOUNS = ("symbol", "symptom")
SET_NOUNS = ("element", "token")
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 8000


CENSUS = ("echo_digit", "add1", "sub1", "sort_min", "compare", "count_mod3")
N_CENSUS = 6


def prompt_for(word, seq, mk=MARKER):
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def gold_for(seq, mk=MARKER):
    d = [int(x) for x in seq.split()]
    return str(max(d) if mk == "A" else min(d))


def census_items():
    """Small self-contained generators; correctness varies here, which P4 needs."""
    import random as _r
    rng = _r.Random(SEED)
    out = []
    for i in range(N_CENSUS):
        a, b = rng.randrange(1, 9), rng.randrange(1, 9)
        xs = rng.sample(range(1, 100), 3)
        bits = [rng.randrange(2) for _ in range(9)]
        d = rng.randrange(10)
        out += [
            {"family": "echo_digit", "gold": str(d),
             "prompt": f"Repeat this number exactly.\nNumber: {d}"},
            {"family": "add1", "gold": str(a + 1), "prompt": f"What is {a} + 1?"},
            {"family": "sub1", "gold": str(a - 1 if a > 0 else 0),
             "prompt": f"What is {a} - 1?"},
            {"family": "sort_min", "gold": str(min(xs)),
             "prompt": f"What is the smallest of these numbers: {xs[0]}, {xs[1]}, {xs[2]}?"},
            {"family": "compare", "gold": str(max(a, b) if a != b else a),
             "prompt": f"Which number is larger, {a} or {b}?"},
            {"family": "count_mod3", "gold": str(sum(bits) % 3),
             "prompt": ("Count how many ones are in this sequence, then give the remainder "
                        f"when divided by 3.\nSequence: {' '.join(map(str, bits))}")},
        ]
    return out


def build_items(rng=None):
    out = []
    for n in ROT_NOUNS + SET_NOUNS:
        for i, s in enumerate(SEQS):
            out.append({"kind": "regime", "label": n, "item": i,
                        "prompt": prompt_for(n, s), "gold": gold_for(s)})
    for i, it in enumerate(census_items()):
        out.append({"kind": "census", "label": it["family"], "item": i,
                    "prompt": it["prompt"], "gold": it["gold"]})
    return out


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("transrot.json")
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

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def nondc(x):
        """Total non-DC power of a window -- P5's floor, so a ratio of noise is visible."""
        y = x - x.mean(0, keepdims=True)
        f = np.abs(np.fft.rfft(y, axis=0)) ** 2
        return float(f[1:].sum())

    def measure(prompt, gold):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n = ids.shape[1]
        freqs = model.freqs_cis[:, :n]
        g0 = tok(gold, add_special_tokens=False).input_ids[0]
        traj, ranks = [], []
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
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            hh.remove()
        T = np.array(traj, dtype=np.float32)
        # P3: sliding 12-unroll window, so onset is measured rather than inferred
        slide = [(s, float(rotation_power(T[s:s + EARLY], period=6, tail=EARLY)))
                 for s in range(0, T.shape[0] - EARLY + 1, 4)]
        return {"R_early": float(rotation_power(T[:EARLY], period=6, tail=EARLY)),
                "R_tail": float(rotation_power(T, period=6, tail=TAIL)),
                "pow_early": nondc(T[:EARLY]), "pow_tail": nondc(T[-TAIL:]),
                "slide": slide, "rank_curve": ranks,
                "best_rank": int(min(ranks)), "best_depth": int(np.argmin(ranks)) + 1,
                "oracle": int(min(ranks) == 1), "n_tokens": int(n)}

    items = build_items()
    print(f"{len(items)} prompts ({sum(1 for i in items if i['kind'] == 'regime')} regime, "
          f"{sum(1 for i in items if i['kind'] == 'census')} census)", flush=True)
    t0, rows = time.time(), []
    for k, it in enumerate(items):
        try:
            rows.append({**it, "ok": True, **measure(it["prompt"], it["gold"])})
        except Exception as exc:  # noqa: BLE001
            rows.append({**it, "ok": False, "why": f"{type(exc).__name__}: {exc}"})
        if k % 8 == 0:
            print(f"  {k}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok = [r for r in rows if r["ok"]]
    import statistics as st
    reg = [r for r in ok if r["kind"] == "regime"]
    cen = [r for r in ok if r["kind"] == "census"]

    print("\n=== P1 GATE: R_tail must reproduce the regime labels ===", flush=True)
    for lab in ROT_NOUNS + SET_NOUNS:
        v = [r["R_tail"] for r in reg if r["label"] == lab]
        if v:
            exp = "rotating" if lab in ROT_NOUNS else "settling"
            got = "rotating" if st.median(v) > THRESHOLD else "settling"
            print(f"  {lab:9s} R_tail median {st.median(v):.3f}  expected {exp:9s} got {got:9s}"
                  f"  {'OK' if exp == got else 'FAIL'}", flush=True)

    print("\n=== P2 PRIMARY: is there any period-6 power in the DECISION window? ===",
          flush=True)
    print(f"{'label':10s} {'R_early':>8s} {'R_tail':>8s} {'pow_early':>11s} {'pow_tail':>11s} "
          f"{'best_depth':>11s}")
    for r in reg:
        print(f"{r['label']:10s} {r['R_early']:8.3f} {r['R_tail']:8.3f} {r['pow_early']:11.4g} "
              f"{r['pow_tail']:11.4g} {r['best_depth']:11d}", flush=True)
    for grp, nm in ((ROT_NOUNS, "rotating"), (SET_NOUNS, "settling")):
        v = [r for r in reg if r["label"] in grp]
        if v:
            print(f"  {nm:9s}: R_early median {st.median([x['R_early'] for x in v]):.3f}, "
                  f"R_tail median {st.median([x['R_tail'] for x in v]):.3f}", flush=True)
    e_rot = [r["R_early"] for r in reg if r["label"] in ROT_NOUNS]
    e_set = [r["R_early"] for r in reg if r["label"] in SET_NOUNS]
    if e_rot and e_set:
        sep = abs(st.median(e_rot) - st.median(e_set))
        print(f"  -> early separation {sep:.3f}; tail separation "
              f"{abs(st.median([r['R_tail'] for r in reg if r['label'] in ROT_NOUNS]) - st.median([r['R_tail'] for r in reg if r['label'] in SET_NOUNS])):.3f}",
              flush=True)
        print("     " + ("THE REGIME IS ALREADY PRESENT EARLY -- D148/D176's nulls stand as "
                         "behavioural nulls" if sep > 0.2 else
                         "THE REGIME DOES NOT EXIST YET IN THE DECISION WINDOW -- D148/D176's "
                         "nulls are explained without the regime being epiphenomenal"),
              flush=True)

    print("\n=== P3 ONSET: sliding 12-unroll R, rotating vs settling ===", flush=True)
    for grp, nm in ((ROT_NOUNS, "rotating"), (SET_NOUNS, "settling")):
        v = [r for r in reg if r["label"] in grp]
        if v:
            starts = [s for s, _ in v[0]["slide"]]
            med = [st.median([dict(x["slide"])[s] for x in v]) for s in starts]
            print(f"  {nm:9s} " + " ".join(f"u{s}:{m:.2f}" for s, m in zip(starts, med)),
                  flush=True)

    print("\n=== P4 AGAINST CORRECTNESS (census families) ===", flush=True)
    if cen:
        c1 = [r for r in cen if r["oracle"]]
        c0 = [r for r in cen if not r["oracle"]]
        print(f"  n={len(cen)}, oracle base rate {len(c1) / len(cen):.3f}", flush=True)
        if len(c1) > 2 and len(c0) > 2:
            for key in ("R_early", "R_tail"):
                print(f"  {key:8s} correct median {st.median([r[key] for r in c1]):.3f} | "
                      f"wrong median {st.median([r[key] for r in c0]):.3f}", flush=True)
        else:
            print("  outcome has no variance -- P4 not evaluable (RC6)", flush=True)

    print("\n=== P5 FLOOR ===", flush=True)
    print(f"  early-window non-DC power: median {st.median([r['pow_early'] for r in ok]):.4g}; "
          f"tail: {st.median([r['pow_tail'] for r in ok]):.4g}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "early": EARLY, "tail": TAIL, "threshold": THRESHOLD,
                   "seed": SEED, "num_steps": NUM_STEPS, "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} prompts, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
