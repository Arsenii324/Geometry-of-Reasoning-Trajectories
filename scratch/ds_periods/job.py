"""A49 (2nd pass): does their word problem orbit at SOME period, or not at all?

WHAT A46 FOUND AND WHY IT IS NOT YET AN ANSWER. A46 measured `rotation_power` at every token
position of 21 prompts. It vindicated the record -- rotating prompts rotate at **62.0%** of
positions, settling prompts at **0.000**, no overlap (D192). But on the paper's own example
shape, *"Claire makes a 3 egg omelette every morning for breakfast..."*, it found **0 of 35**
positions rotating, max R **0.119** against a 0.6677 threshold. The paper says orbiting happens
*'when responding to prompts requiring numerical reasoning'* and draws its figures on exactly
that kind of prompt.

**Our detector is the obvious suspect.** `rotation_power(traj, period=6, tail=24)` measures the
fraction of non-DC tail power sitting in the **period-6 bin only** -- it was tuned to D112's
finding on our own prompts. An orbit at any other period reads ~0 **by construction**. Their
figures are PCA projections judged visually, with no period stated anywhere.

So the question is not "do they orbit" but "at what period, if any", and it is answerable in
one run.

THE DESIGN CHANGE THAT MAKES IT REUSABLE. A46 banked R at period 6. This banks the **entire
rfft power spectrum** of the tail at every position -- 33 bins for a 64-unroll trajectory --
so **any** period can be read offline, now or later, without another GPU hour. The spectrum is
the primitive; `rotation_power` is one bin of it.

PREREGISTERED PREDICTIONS:

  P1  REPLICATION GATE, VOID WITHOUT IT. The period-6 bin fraction computed from the banked
      spectrum must reproduce A46's `rotation_power` values to floating point on the same
      prompts -- `symbol` ~0.77, `element` ~0.08, their word problem ~0.045. If the spectrum
      does not reproduce the statistic that was derived from it, one of the two is wrong.

  P2  PRIMARY. On their word problem, is there ANY period whose power fraction clears 0.6677 at
      ANY position? Registered both ways: **a clear peak at some period p != 6 means our
      instrument was narrow and D192's failure-to-reproduce is ours, not theirs**; **no peak at
      any period means their example does not orbit on this checkpoint at all**, and D185's
      contradiction of their attribution strengthens from "wording, not content" to "their own
      exemplar does not do it".

  P3  THE POSITIVE CONTROL, and it is what makes P2 readable. On `symbol`, the spectrum must
      peak **at period 6** and not merely be large somewhere. If our own rotating prompts do
      not show a clean period-6 peak, the whole regime line rests on a statistic that was
      measuring something broader than we said.

  P4  IS PERIOD 6 SPECIAL TO THE MODEL OR TO OUR PROMPTS? The dominant period is reported per
      prompt family. Four core blocks and a period-6 orbit are not obviously related; if
      rotating prompts differ among themselves in dominant period, "period-6 rotation" is a
      property of a prompt set rather than of the architecture.

  P5  SETTLING PROMPTS ARE THE FLOOR. `element` should show no peak anywhere -- a settled orbit
      has almost no non-DC power at all. Its spectrum is banked so "flat" can be checked rather
      than assumed, and the total non-DC power is reported alongside the fractions, because a
      fraction of nearly nothing is meaningless.
"""

import json
import os
import subprocess
import sys
import time

NUM_STEPS = 64
TAILS = (24, 48)           # 24 is rotation_power's default and D141's calibration;
                           # 48 was A49's first pass and FAILED P1 because the damping
                           # transient dominates its low-frequency bins. Bank both so
                           # the tail-dependence is measured rather than assumed.
SEED = 20260811
THRESHOLD = 0.6677
MARKER = "A"
SEQS = ("1 6 0 1 7 7 8 1", "3 5 5 3 6 1 5 2")
ROT_NOUNS = ("symbol", "symptom")
SET_NOUNS = ("element", "token")
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 8000

THEIRS = (
    "Claire makes a 3 egg omelette every morning for breakfast. How many eggs will "
    "she eat in 4 weeks?",
    "What do you think of Goethe's Faust?",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?",
)


def prompt_for(word, seq, mk=MARKER):
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def build_items(rng=None):
    out = []
    for n in ROT_NOUNS:
        for i, s in enumerate(SEQS):
            out.append({"kind": "rot_noun", "label": n, "item": i, "prompt": prompt_for(n, s)})
    for n in SET_NOUNS:
        for i, s in enumerate(SEQS):
            out.append({"kind": "set_noun", "label": n, "item": i, "prompt": prompt_for(n, s)})
    for i, p in enumerate(THEIRS):
        out.append({"kind": "theirs", "label": f"theirs{i}", "item": i, "prompt": p})
    return out


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("periods.json")
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

    def spectra(prompt):
        """Full rfft power spectrum of the tail, at EVERY position. The spectrum is the
        primitive; `rotation_power` is one bin of it, so banking this makes every period
        readable offline forever."""
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        traj = []
        core_last._forward_hooks.clear()

        def hook(_m, _i, o):
            traj.append(o.detach()[0].float().cpu().numpy().copy())

        h = core_last.register_forward_hook(hook)
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        T = np.stack(traj, axis=0)                       # [unrolls, pos, dim]
        out = {"tokens": [tok.decode([int(i)]) for i in ids[0]]}
        for TAIL in TAILS:
            x = T[-TAIL:]
            x = x - x.mean(0, keepdims=True)
            f = np.abs(np.fft.rfft(x, axis=0)) ** 2      # [bins, pos, dim]
            pw = f.sum(-1)                               # [bins, pos]
            total = pw[1:].sum(0)                        # non-DC power per position
            frac = np.where(total > 0, pw / np.maximum(total, 1e-30), 0.0)
            out[f"frac{TAIL}"] = frac.tolist()
            out[f"total{TAIL}"] = total.tolist()
            # P1 must compare LIKE WITH LIKE: rotation_power at the SAME tail as the spectrum.
            out[f"r6_spec{TAIL}"] = float(frac[TAIL // 6][-1])
            out[f"r6_fn{TAIL}"] = float(rotation_power(T[:, -1, :], period=6, tail=TAIL))
        out["n_pos"] = int(np.array(out[f"frac{TAILS[0]}"]).shape[1])
        return out

    items = build_items()
    print(f"{len(items)} prompts, TAIL={TAIL} -> periods readable: "
          f"{[TAIL // k for k in range(1, TAIL // 2 + 1) if TAIL % k == 0]}", flush=True)
    t0, rows = time.time(), []
    for n, it in enumerate(items):
        try:
            s = spectra(it["prompt"])
            rows.append({**it, "ok": True, **s})
        except Exception as exc:  # noqa: BLE001
            rows.append({**it, "ok": False, "why": f"{type(exc).__name__}: {exc}"})
        if n % 3 == 0:
            print(f"  {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    import numpy as np
    ok = [r for r in rows if r["ok"]]
    print("\n=== P1 GATE, at EACH tail: spectrum's period-6 bin vs rotation_power(tail=) ===",
          flush=True)
    for TAIL in TAILS:
        worst = max((abs(r[f"r6_spec{TAIL}"] - r[f"r6_fn{TAIL}"]) for r in ok), default=0.0)
        print(f"  tail={TAIL}: max |spectrum - function| = {worst:.3e}  "
              f"{'OK' if worst < 1e-6 else 'FAIL -- nothing at this tail may be read'}",
              flush=True)
        for r in ok[:3]:
            print(f"     {r['label']:10s} spec {r[f'r6_spec{TAIL}']:.4f}  "
                  f"fn {r[f'r6_fn{TAIL}']:.4f}", flush=True)

    for TAIL in TAILS:
        print(f"\n=== dominant period at tail={TAIL} (threshold-free) ===", flush=True)
        print(f"{'kind':10s} {'label':10s} {'best_frac':>10s} {'at_period':>10s} "
              f"{'medNonDC':>11s}")
        for r in ok:
            F = np.array(r[f"frac{TAIL}"]); sub = F[1:]
            bi, pi = np.unravel_index(int(np.argmax(sub)), sub.shape)
            print(f"{r['kind']:10s} {r['label']:10s} {float(sub[bi, pi]):10.3f} "
                  f"{TAIL / (bi + 1):10.2f} "
                  f"{np.median(np.array(r[f'total{TAIL}'])):11.4g}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "tail": TAIL, "num_steps": NUM_STEPS, "seed": SEED,
                   "threshold": THRESHOLD, "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} prompts, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
