"""A26: does crossing the regime boundary change what the model DOES?

THE QUESTION THIS CLOSES. D132 compared the rotating and settling arms and found no
behavioural difference -- 16.0 vs 14.5 unrolls to best availability (p = 0.156), best rank
6.5 vs 7.0 (p = 0.880). A full-ledger audit then flagged the fatal weakness: **no detection
floor was ever computed for that test**, so it is UNREFUTED rather than bounded, and it
rests on 36 pairs where rank-1 accuracy is 6-17% -- meaning `best_depth` there is mostly
the argmin of rank curves that never reach 1. It is the weakest link in the whole project:
a dramatic dynamical phenomenon with no demonstrated consequence, and no bound on how
large a consequence we could have missed.

WHY THIS DESIGN IS DIFFERENT, AND WHY IT IS THE RIGHT ONE. D132 was observational: it
compared DIFFERENT PROMPTS (`symbol` vs `element`), so any behavioural difference could
have been the word rather than the regime. D140/D144 give something better -- a **causal
handle on the regime that leaves the prompt untouched**. The carrier prompt is fixed; only
`e`, the map's parameter, moves. And because the transition is sharper than a 0.05 grid
step (D140: all 8 cross-regime paths cross exactly once, each inside one step), we can
compare **two points 0.05 apart in `e`-space that sit on opposite sides of the boundary.**

That is a regression discontinuity. Between the two points the parameter barely moves,
every token is identical, and the required answer is identical -- but the dynamics flip
regime. If behaviour is going to depend on the regime anywhere, it depends on it here. If
it does not, "the regime is a dynamical epiphenomenon" stops being a hedge and becomes the
finding.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, VOID WITHOUT IT. h_0 is seeded (D78), so the t = 0 patch is a no-op
      by construction and must reproduce the unpatched orbit's rank curve EXACTLY --
      identical best_depth, best_rank and gold rank at every unroll. A25 established this
      for the state; here it must also hold for the READOUT, which is a different code
      path (coda + lm_head) and has never been checked under patching.

  P2  THE REGIME MUST ACTUALLY FLIP, or there is nothing to test. Each (chord, sequence)
      must contain exactly one crossing of R > 0.6677 (D141's threshold) inside the swept
      window. Units with zero or multiple crossings are DROPPED and counted, not analysed
      -- an RD design with no discontinuity measures nothing.

  P3  PRIMARY. Within each surviving unit, compare the last sub-threshold grid point
      against the first supra-threshold point on: (a) `best_depth`, (b) `best_rank`,
      (c) correctness. Paired, within-unit, so the sequence and the carrier cancel.
      **The comparison is between two states of the SAME prompt 0.05 apart in `e`.**

  P4  DETECTION FLOOR -- the thing D132 never had, and the reason this run exists.
      The analysis plants effects of known size into the observed distribution and
      reports what fraction this design would catch. A null without this is not a bound.
      Computed locally from the banked rank curves, so it costs no GPU and cannot be
      skipped for time.

  P5  A POSITIVE CONTROL ON THE READOUT ITSELF. Across the full sweep from t = 0 to t = 1
      the parameter travels all the way from one noun to another. If the gold rank does not
      move AT ALL over that range, the readout is insensitive to `e` in this setup and P3's
      null is uninterpretable. So the sweep is banked whole, not only at the boundary.

The kernel measures the READOUT (gold rank at every unroll) as well as the state, which
D140 and D144 did not -- they banked geometry only. That is the one thing needed to turn
the causal handle into a behavioural test.
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
THRESHOLD = 0.6677          # D141: reproduces the binary period-6 label on 691/696 orbits

# Chords with a t* measured in D140, so the local window is centred where the crossing is.
# (rotating noun, settling carrier, t* from D140)
CHORDS = (
    ("array", "token", 0.283),
    ("block", "digit", 0.435),
    ("signal", "letter", 0.528),
    ("symbol", "element", 0.737),
)
# local grid: t* +/- 0.15 in 0.05 steps, plus the two anchors
def grid_for(tstar):
    loc = [round(tstar + d * 0.05, 4) for d in (-3, -2, -1, 0, 1, 2, 3)]
    return [0.0] + [t for t in loc if 0.0 < t < 1.0] + [1.0]

N_SEQ = 12                  # 12 sequences x 4 chords = 48 paired units
SEQ_LEN = 8
MARKER = "A"
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 12000


def build_items(rng=None):
    """Sequences only -- the prompt text is assembled per chord from the carrier noun."""
    import random as _r
    rng = rng or _r.Random(20260810)
    seqs, seen = [], set()
    while len(seqs) < N_SEQ:
        s = tuple(rng.randrange(10) for _ in range(SEQ_LEN))
        if s in seen:
            continue
        seen.add(s)
        seqs.append(s)
    return [{"seq_id": i, "seq": " ".join(map(str, s)),
             "gold": str(max(s)), "item": i} for i, s in enumerate(seqs)]


def prompt_for(word, seq, mk=MARKER):
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("regimebehav.json")
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

    # ONE implementation of R, shared with the local analysis (D141's threshold is
    # defined against exactly this function).
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

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

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

    def orbit_and_ranks(ids, gold_id, donor_e=None):
        """Bank the last-position state trajectory AND the gold rank at every unroll.

        The rank is computed INSIDE the core-block hook, matching the pattern `ds_nth`
        and `ds_addk` already use, rather than storing 64 full [1, n_p, 5280] activations
        and running the coda afterwards. Same numbers, ~71 MB less resident, and it keeps
        one implementation of the rank curve across kernels. `coda_head` needs the full
        position axis (the coda blocks attend), so only the SAVED state is sliced.
        """
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        traj, ranks = [], []
        core_last._forward_hooks.clear()

        def pre(_m, inp):
            cur = inp[0]
            if donor_e is not None:
                cur = torch.cat([cur[..., :d_model], donor_e], dim=-1)
            return (cur,)

        def post(_m, _i, o):
            with torch.no_grad():
                row = torch.log_softmax(
                    coda_head(o.detach(), freqs).float()[0, n_p - 1], dim=-1)
                ranks.append(int((row > row[gold_id]).sum().item()) + 1)
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
        return np.array(traj, dtype=np.float32), ranks

    items = build_items()
    t0 = time.time()
    rows, dropped = [], []

    for (rot_w, set_w, tstar) in CHORDS:
        ts = grid_for(tstar)
        for it in items:
            ids_r = encode(prompt_for(rot_w, it["seq"]))
            ids_s = encode(prompt_for(set_w, it["seq"]))
            if ids_r.shape[1] != ids_s.shape[1]:
                dropped.append({"chord": f"{rot_w}/{set_w}", "seq_id": it["seq_id"],
                                "n_a": int(ids_r.shape[1]), "n_b": int(ids_s.shape[1])})
                continue
            g_ids = tok(it["gold"], add_special_tokens=False).input_ids
            gold_id = g_ids[0]
            e_r, e_s = prelude_e(ids_r), prelude_e(ids_s)

            # P1: the unpatched reference, same seed, for the exact no-op comparison
            base_states, base_ranks = orbit_and_ranks(ids_s, gold_id, donor_e=None)
            base_R = rotation_power(base_states)

            for t in ts:
                e_t = (1.0 - t) * e_s + t * e_r
                st, rk = orbit_and_ranks(ids_s, gold_id, donor_e=e_t)
                R = rotation_power(st)
                rows.append({
                    "rot_w": rot_w, "set_w": set_w, "tstar_d140": tstar,
                    "seq_id": it["seq_id"], "seq": it["seq"], "gold": it["gold"],
                    "t": t, "R": R, "rotating": bool(R > THRESHOLD),
                    "rank_curve": rk,
                    "best_rank": int(min(rk)), "best_depth": int(np.argmin(rk)) + 1,
                    "correct": bool(min(rk) == 1),
                    "final_rank": int(rk[-1]),
                    "multi_token_gold": bool(len(g_ids) > 1),
                    "base_R": base_R, "base_ranks": base_ranks,
                    "base_best_depth": int(np.argmin(base_ranks)) + 1,
                    "base_best_rank": int(min(base_ranks)),
                    "n_tokens": int(ids_s.shape[1]), "ok": True,
                })
                with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                    json.dump(rows, f)
            print(f"  {rot_w}<-{set_w} seq{it['seq_id']}: "
                  f"{sum(1 for r in rows[-len(ts):] if r['rotating'])}/{len(ts)} rotating, "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    summary = {"rows": rows, "dropped": dropped, "num_steps": NUM_STEPS,
               "threshold": THRESHOLD, "seed": SEED, "chords": [list(c) for c in CHORDS],
               "elapsed_s": time.time() - t0}
    with open(out_path, "w") as f:
        json.dump(summary, f)
    print(f"DONE {len(rows)} orbits, {len(dropped)} dropped, {time.time() - t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
