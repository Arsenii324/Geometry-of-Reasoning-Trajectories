"""A39: swap `e` MID-TRAJECTORY. Is the regime a function of the parameter, or of history?

WHY THIS IS THE RIGHT SHAPE OF EXPERIMENT NOW. D165 established that the regime cannot be
dated by linear probing: the design fails on a label it is mathematically guaranteed to
contain (0.690 against a 0.600 baseline, where the correct 1-D statistic gets 0.980). So the
question "when is the regime decided?" has to be asked causally. This project already has the
tool -- D140/A24 replaced `e` for a whole run and flipped the regime -- and it has never once
been used **partway through**.

`e` is the map's PARAMETER, not its input (D111, D113): the model re-injects it at every
unroll via `adapter(cat[x, input_embeds])`. So it can be changed at unroll k and left changed,
which asks a question no static design can:

    run the ROTATING prompt, and from unroll k onward feed the SETTLING prompt's `e`.
    Does the orbit become settling -- and if so, how many unrolls does it take?

**AND IT MAKES A QUANTITATIVE PREDICTION THAT CONNECTS TWO OTHERWISE UNCONNECTED NUMBERS.**
UNDERSTANDING.md §6.2 says flatly that ρ "has never been connected to behaviour". If the map
contracts at ρ ~ 0.83 (D31, D94, D113, D119 all land in 0.79-0.87) and the attractor is unique
given the parameter, then after switching the parameter the state must approach the new
attractor geometrically, and the unrolls needed to close a fraction eps is

    n  =  ln(eps) / ln(rho)     ->   ln(0.01)/ln(0.83)  =  **24.7 unrolls**  for eps = 1%
                                     ln(0.05)/ln(0.83)  =  **16.1 unrolls**  for eps = 5%

That is registered here, before running, as the prediction for P4's relaxation time. A measured
relaxation in the range **16-25 unrolls** would be the first time this project's contraction
rate predicted an observable, rather than merely being measured.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, VOID WITHOUT IT. Two identities must hold to floating point. **(a)** No
      swap at all must reproduce the unpatched source orbit (|dR| < 1e-9). **(b)** Swapping at
      k = 0 -- the donor's `e` from the very first unroll -- must reproduce the donor's own
      orbit, which is exactly A24's t = 1 arm and is known to hold. If either fails the hook
      writes something other than `e` and nothing else may be read.

  P2  DOES THE REGIME FOLLOW THE PARAMETER AT ALL? For k = 1, R over the final 24 unrolls must
      take the DONOR's regime. If it does not, `e` is not sufficient to set the regime once the
      state has moved even one step, and D140's whole-run result was about initialisation
      rather than about the parameter.

  P3  PRIMARY -- HYSTERESIS OR NOT. R over the final 24 unrolls as a function of the switch
      point k, in BOTH directions (rotating -> settling and settling -> rotating). **A
      contraction with a unique attractor per parameter predicts NO hysteresis**: every k gives
      the donor's regime, and the curve is flat in k. **Hysteresis -- late switches failing to
      take -- would mean the map is MULTISTABLE at fixed `e`**, which contradicts the
      contraction picture that four instruments agree on, and would be the more interesting
      outcome. Registered: flat in k is the prediction; any k at which the switch fails to take
      is the finding.

  P4  THE RELAXATION TIME, AND THE ρ TEST ABOVE. Sliding-window R after the switch gives the
      number of unrolls before the new regime is established. Predicted **16-25** from ρ ~ 0.83
      as derived above. Measured well below 16 would mean the regime change is faster than
      contraction and is not simply the state falling into a new basin; well above 25 would
      mean it is slower, and the gap is the finding.

  P5  ASYMMETRY. Rotating -> settling and settling -> rotating need not be equally fast. A
      periodic orbit is not a point, so entering one may cost more than leaving one. Both
      directions are run for exactly this reason, and any asymmetry is reported rather than
      averaged away.
"""

import json
import os
import subprocess
import sys
import time

NUM_STEPS = 96            # so the final-24 window is post-switch for every k tested
SEED = 20260810
THRESHOLD = 0.6677        # D141
KS = (0, 1, 2, 4, 8, 16, 32, 48)
PAIRS = (("symbol", "element"), ("array", "digit"))   # (rotating, settling)
SEQS = ("1 6 0 1 7 7 8 1", "3 5 5 3 6 1 5 2")
MARKER = "A"
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 10000


def build_items(rng=None):
    return [{"seq": s, "item": i} for i, s in enumerate(SEQS)]


def prompt_for(word, seq, mk=MARKER):
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("hyster.json")
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

    d_model = cfg.n_embd if hasattr(cfg, "n_embd") else model.config.n_embd
    adapter = model.transformer.adapter
    core_last = model.transformer.core_block[-1]

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    def prelude_e(ids):
        """`e` is the second half of the concatenation `adapter(cat[x, input_embeds])`.

        Same mechanism as `ds_einterp`/`ds_estream`; captured on the first adapter call.
        """
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

    def orbit(ids, donor_e=None, switch_at=None):
        """Trajectory of the last position, with `e` replaced from unroll `switch_at` on.

        `switch_at=None` never swaps; `switch_at=0` swaps from the first unroll, which is
        A24's whole-run patch and therefore the known-good identity in P1(b). The counter
        is the adapter call index, and the adapter is called exactly once per unroll.
        """
        traj = []
        n = {"i": 0}

        def pre(_m, inp):
            cur = inp[0]
            if donor_e is not None and switch_at is not None and n["i"] >= switch_at:
                cur = torch.cat([cur[..., :d_model], donor_e], dim=-1)
            n["i"] += 1
            return (cur,)

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
        return np.array(traj, dtype=np.float32)

    def sliding_R(traj, width=30, step=3):
        """R as a function of time. Width 30 >= tail 24, so each window resolves period 6."""
        return [(s, float(rotation_power(traj[s:s + width])))
                for s in range(0, traj.shape[0] - width + 1, step)]

    def relax(traj, k, target_rot, width=30, step=3):
        """Unrolls after the switch before the window R first sits on the donor's side."""
        for s, R in sliding_R(traj, width, step):
            if s < k:
                continue
            if (R > THRESHOLD) == target_rot:
                return s - k
        return -1

    t0, rows = time.time(), []
    for rot_w, set_w in PAIRS:
        for it in build_items():
            ids_r = encode(prompt_for(rot_w, it["seq"]))
            ids_s = encode(prompt_for(set_w, it["seq"]))
            if ids_r.shape[1] != ids_s.shape[1]:
                print(f"  DROPPED {rot_w}/{set_w} seq{it['item']}: token counts "
                      f"{ids_r.shape[1]} vs {ids_s.shape[1]} differ", flush=True)
                continue
            e_r, e_s = prelude_e(ids_r), prelude_e(ids_s)
            base_r = rotation_power(orbit(ids_r))
            base_s = rotation_power(orbit(ids_s))
            print(f"\n  {rot_w}/{set_w} seq{it['item']}: base rotating {base_r:.3f}, "
                  f"base settling {base_s:.3f}", flush=True)

            for direction, ids, donor, want_rot, base_self, base_donor in (
                    ("rot->set", ids_r, e_s, False, base_r, base_s),
                    ("set->rot", ids_s, e_r, True, base_s, base_r)):
                for k in KS:
                    tr = orbit(ids, donor_e=donor, switch_at=k)
                    R = rotation_power(tr)
                    rows.append({
                        "rot_w": rot_w, "set_w": set_w, "direction": direction,
                        "seq": it["seq"], "item": it["item"], "k": k, "R": R,
                        "base_self": base_self, "base_donor": base_donor,
                        "took": bool((R > THRESHOLD) == want_rot),
                        "relax": relax(tr, k, want_rot),
                        "sliding": sliding_R(tr),
                        "n_tokens": int(ids.shape[1]), "ok": True})
                    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                        json.dump(rows, f)
                seg = [r for r in rows[-len(KS):]]
                print(f"    {direction}: R by k " +
                      " ".join(f"k{r['k']}={r['R']:.3f}" for r in seg), flush=True)
                print(f"    {direction}: took " +
                      " ".join(f"k{r['k']}={'Y' if r['took'] else 'N'}" for r in seg) +
                      "   relax " + " ".join(f"{r['relax']}" for r in seg), flush=True)
            if time.time() - t0 > WALL_BUDGET_S:
                print("WALL BUDGET -- banking and stopping cleanly", flush=True)
                break

    ok = [r for r in rows if r.get("ok")]
    print("\n=== P1 INSTRUMENT NULL ===", flush=True)
    for r in [x for x in ok if x["k"] == 0]:
        d = abs(r["R"] - r["base_donor"])
        print(f"  {r['rot_w']}/{r['set_w']} {r['direction']} seq{r['item']}: "
              f"k=0 R {r['R']:.6f} vs donor's own {r['base_donor']:.6f}  |d| {d:.2e}"
              f"  {'OK' if d < 1e-6 else 'FAIL'}", flush=True)

    print("\n=== P3 HYSTERESIS: did the switch take, by k? ===", flush=True)
    for direction in ("rot->set", "set->rot"):
        v = [r for r in ok if r["direction"] == direction]
        for k in KS:
            w = [r for r in v if r["k"] == k]
            if w:
                print(f"  {direction} k={k:3d}: took {sum(r['took'] for r in w)}/{len(w)}, "
                      f"median R {sorted(r['R'] for r in w)[len(w) // 2]:.3f}", flush=True)

    print("\n=== P4 RELAXATION (predicted 16-25 unrolls from rho ~ 0.83) ===", flush=True)
    for direction in ("rot->set", "set->rot"):
        v = [r["relax"] for r in ok if r["direction"] == direction
             and r["took"] and r["relax"] >= 0]
        if v:
            print(f"  {direction}: n={len(v)} median {sorted(v)[len(v) // 2]} "
                  f"min {min(v)} max {max(v)}", flush=True)
        else:
            print(f"  {direction}: no switch took; relaxation undefined", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "pairs": [list(p) for p in PAIRS], "ks": list(KS),
                   "threshold": THRESHOLD, "seed": SEED, "num_steps": NUM_STEPS,
                   "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} orbits, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
