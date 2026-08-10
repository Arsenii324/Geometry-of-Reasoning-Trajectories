"""A28: what SHAPE is the regime boundary in `e`-space? Bisect it in many directions.

WHY THIS IS THE RIGHT NEXT GEOMETRY EXPERIMENT. A24/A25 established that the regime is
causally set by `e`, that the transition is sharper than a 0.05 grid step, that direction
rather than distance is what matters (0 of 8 norm-matched random bearings rotate, and a
random bearing destroys rotation on 3 of 3 rotating carriers), and that the rotating set
is non-convex but not generically so (2 of 5 settling->settling chords cross). All of that
says *where* the regime lives and nothing about *what* it is.

This asks the shape question directly, and it has a sharp null. **Take one rotating
carrier and walk outward in many random unit directions, bisecting for the distance at
which the orbit leaves the rotating regime.** The geometry of the boundary is then read
off the distribution of those distances:

  * **If the boundary is a HYPERPLANE** n.e = c at distance h from the carrier, then a
    direction u crosses it only when u.n > 0 -- so **about half the directions never
    cross** -- and for those that do, d(u) = h / (u.n). Equivalently **1/d(u) = (u.n)/h
    is a LINEAR function of u**, which for a random unit vector in high dimension is
    approximately Gaussian about 0 with the crossing half being its positive tail.
  * **If the rotating set is a bounded BLOB**, essentially all directions cross, and 1/d
    is concentrated away from 0 rather than piling up near it.
  * **If it is neither**, the crossing fraction and the 1/d distribution say so, and the
    honest answer is that the shape is not one of the two simple ones.

WHY IT MATTERS BEYOND THE SHAPE. D135 and D136 found the regime unreadable from the token
EMBEDDING by six classifiers, linear and nonlinear, at n = 49. If the boundary turns out
to be a hyperplane in `e`, those two nulls stop being puzzling: `e` is the prelude's
NONLINEAR image of the embedding, so a structure that is exactly linear in `e` need not be
even approximately linear in the embedding. A hyperplane here would convert two of this
project's nulls from unexplained into predicted.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, VOID WITHOUT IT. h_0 is seeded, so a zero-displacement patch is a
      no-op and must reproduce the unpatched orbit's R to 0.0 exactly (A25 got 0.000e+00
      on 8 chords; this must hold here too or the bisection is measuring the wrong thing).

  P2  BRACKETING GATE. A direction is only bisected if the endpoint at MAX_DIST is on the
      far side of the threshold. Directions that never leave the regime within MAX_DIST
      are recorded as CENSORED, never as "very far" -- averaging a cap in as an
      observation is the aggregation error this project has already made once.

  P3  PRIMARY -- THE CROSSING FRACTION. A hyperplane predicts ~50% of random directions
      cross; a bounded blob predicts ~100%. Reported with a binomial interval.

  P4  SECONDARY -- THE SHAPE OF 1/d. Under a hyperplane, 1/d is the positive half of a
      near-Gaussian centred at 0, so its density RISES toward 0 (many directions barely
      cross, few cross close in). Under a blob, 1/d is concentrated at a typical radius
      and its density FALLS toward 0. The two make opposite predictions about the same
      histogram, which is what makes this a test rather than a description.

  P5  ANTIPODAL CONTROL. Each direction is also walked in -u. Under a hyperplane exactly
      one of {u, -u} crosses (they are on opposite sides of the normal), so the count of
      pairs where BOTH cross should be near 0. Under a blob both always cross. **This is
      the cleanest single discriminator in the design** and it needs no distributional
      assumption at all.
"""

import json
import os
import subprocess
import sys
import time

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 64
SEED = 20260810
THRESHOLD = 0.6677          # D141
CARRIERS = (("symbol", "1 6 0 1 7 7 8 1"), ("block", "3 5 5 3 6 1 5 2"))
N_DIR = 16                  # random directions per carrier; each also walked antipodally
MAX_DIST_MULT = 3.0         # cap, as a multiple of ||e_rot - e_set|| for a known chord
BISECT_STEPS = 8            # 2^-8 of the bracket ~ 0.4% of MAX_DIST
MARKER = "A"
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 11000


def build_items(rng=None):
    return [{"carrier": w, "seq": s, "item": i} for i, (w, s) in enumerate(CARRIERS)]


def prompt_for(word, seq, mk=MARKER):
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("boundary.json")
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"results -> {out_path}", flush=True)

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

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    d_model = model.config.n_embd
    adapter = model.transformer.adapter
    core_last = model.transformer.core_block[-1]

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
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=1)
        finally:
            h.remove()
        return grabbed["e"]

    def R_at(ids, e_new):
        traj = []

        def pre(_m, inp):
            cur = inp[0]
            return (torch.cat([cur[..., :d_model], e_new], dim=-1),)

        def post(_m, _i, o):
            traj.append(o.detach()[0, -1].float().cpu().numpy().copy())

        handles = [adapter.register_forward_pre_hook(pre),
                   core_last.register_forward_hook(post)]
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            for h in handles:
                h.remove()
        return rotation_power(np.array(traj, dtype=np.float32))

    t0, rows = time.time(), []
    for it in build_items():
        ids = encode(prompt_for(it["carrier"], it["seq"]))
        e0 = prelude_e(ids)

        # scale: use a real chord's length so MAX_DIST is in natural units
        ids_ref = encode(prompt_for("element", it["seq"]))
        scale = float((prelude_e(ids_ref) - e0).norm()) if ids_ref.shape[1] == ids.shape[1] else float(e0.norm()) * 0.1
        max_dist = MAX_DIST_MULT * scale

        # P1: zero displacement must be an exact no-op
        r_base = R_at(ids, e0)
        r_unp = None
        traj = []

        def pre(_m, inp):
            return (inp[0],)

        def post(_m, _i, o):
            traj.append(o.detach()[0, -1].float().cpu().numpy().copy())

        hs = [adapter.register_forward_pre_hook(pre),
              core_last.register_forward_hook(post)]
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            for h in hs:
                h.remove()
        r_unp = rotation_power(np.array(traj, dtype=np.float32))
        print(f"P1 {it['carrier']}: patched-at-zero {r_base:.12f} vs unpatched "
              f"{r_unp:.12f}  d={abs(r_base - r_unp):.3e}  scale={scale:.3f}", flush=True)

        g = torch.Generator(device=e0.device).manual_seed(SEED + it["item"])
        for k in range(N_DIR):
            u = torch.randn(e0.shape, generator=g, device=e0.device, dtype=e0.dtype)
            u = u / u.norm()
            for sign, tag in ((1.0, "u"), (-1.0, "-u")):
                far = R_at(ids, e0 + sign * max_dist * u)
                if (far > THRESHOLD) == (r_base > THRESHOLD):
                    rows.append({"carrier": it["carrier"], "seq": it["seq"], "dir": k,
                                 "sign": tag, "censored": True, "d": None,
                                 "R_base": r_base, "R_far": far, "scale": scale,
                                 "max_dist": max_dist, "ok": True})
                    print(f"  dir{k}{tag}: CENSORED (R_far={far:.3f})", flush=True)
                    continue
                lo, hi = 0.0, max_dist
                for _ in range(BISECT_STEPS):
                    mid = 0.5 * (lo + hi)
                    rm = R_at(ids, e0 + sign * mid * u)
                    if (rm > THRESHOLD) == (r_base > THRESHOLD):
                        lo = mid
                    else:
                        hi = mid
                d = 0.5 * (lo + hi)
                rows.append({"carrier": it["carrier"], "seq": it["seq"], "dir": k,
                             "sign": tag, "censored": False, "d": d,
                             "d_rel": d / scale, "R_base": r_base, "R_far": far,
                             "scale": scale, "max_dist": max_dist, "ok": True})
                print(f"  dir{k}{tag}: crossed at d={d:.3f} ({d / scale:.2f} chords) "
                      f"({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
            if time.time() - t0 > WALL_BUDGET_S:
                print("WALL BUDGET -- banking and stopping cleanly", flush=True)
                break

    summary = {"rows": rows, "num_steps": NUM_STEPS, "threshold": THRESHOLD,
               "seed": SEED, "n_dir": N_DIR, "bisect_steps": BISECT_STEPS,
               "max_dist_mult": MAX_DIST_MULT, "elapsed_s": time.time() - t0}
    with open(out_path, "w") as f:
        json.dump(summary, f)
    n_c = sum(1 for r in rows if r["censored"])
    print(f"DONE {len(rows)} rays, {n_c} censored, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
