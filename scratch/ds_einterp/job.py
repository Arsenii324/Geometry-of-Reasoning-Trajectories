"""A24: interpolate `e` between a rotating and a settling noun and find the boundary.

WHY THIS DESIGN. D135 and D136 both tried to PREDICT the regime from the token, and
both were null -- six classifiers, best-of-six 0.551 against a null-of-max of 0.589.
The obstacle is not the hypothesis, it is n: 51 nouns is too few to fit any rule in
5280 dimensions, and adding nouns costs GPU linearly for a power gain that goes as
sqrt(n). **So stop fitting and start manipulating.** `e` is the map's PARAMETER
(D111, D113): the prelude's output, re-injected at every unroll via
`adapter(cat[x, input_embeds])`. It can be set to anything, including points that no
token produces.

This walks a straight line in `e`-space from a settling noun to a rotating one and
measures the rotation strength R along it. One pair suffices to locate a boundary,
so the classifier power problem does not arise.

D137 makes this sharp. It found the R distribution GAPPED (largest relative gap
0.1128, p = 0.0002) -- two regimes, not a continuum -- with exactly one token (`set`)
in the gap. A gap in the population predicts a SHARP transition along the
interpolation. A smooth ramp would contradict D137 and mean the population gap is a
sampling artifact of 51 nouns.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, AND THE RUN IS VOID WITHOUT IT. The carrier prompt is the
      SETTLING noun's at every t, so at t = 0 the patch substitutes `e_s` for `e_s`
      and is a **no-op by construction**. R(0) must therefore equal the unpatched
      settling orbit's R to floating-point tolerance. **If it does not, the hook is
      not writing what it claims and nothing else may be read.** Both are computed
      in-kernel so this needs no external file.

      **P1' is a PREDICTION, not an instrument check, and must not be read as one.**
      At t = 1 the parameter is the rotating noun's `e` while the initial state came
      from the settling noun's prompt. R(1) matching the rotating noun's banked orbit
      would confirm that the attractor is fixed by `e` alone -- the DEQ claim of D111
      and D113 -- and its FAILURE would be a finding about the model, not a broken
      instrument. Conflating the two is exactly the error CLAUDE.md §5 exists to stop,
      so they are separated here and reported separately.

  P2  PRIMARY -- SHARPNESS. Fit R(t) and report the 10-90% transition width in t.
      D137's gap predicts a step: width < 0.25. A width above 0.5 is a smooth ramp
      and CONTRADICTS D137's gap, which would then be withdrawn as a 51-noun
      sampling artifact.

  P3  WITHIN-REGIME CONTROL, AND IT IS THE ONE THAT MATTERS. Interpolating between
      two nouns in the SAME regime (rotating->rotating, settling->settling) must
      produce NO transition -- R stays flat at its endpoint value. Without this,
      "any interpolation of `e` produces a flip somewhere" explains the result just
      as well, and it is the reading D111's own history warns about: a hook that
      perturbs `e` at all will move the dynamics.

  P4  CONSISTENCY. Four cross-regime pairs. If the boundary is a real surface in
      `e`-space, the crossing points t* are a property of the pair, and a pair whose
      endpoints are further apart should not systematically cross earlier or later.
      Reported, not gated -- 4 pairs cannot support a test, only a description.

  P5  WHERE DOES `set` SIT? D137 identified `set` as the one token inside the gap.
      Its `e` is captured and projected onto each pair's interpolation axis. The
      prediction is that it lands near t*. This is the only prediction here that
      could fail while everything else passes, and it is the one that would tie the
      causal boundary to the observational gap.

TOKEN COUNT IS GATED, NOT ASSERTED. `e` has shape [1, n_tokens, d_model]; the
interpolation is undefined if the two prompts tokenise to different lengths, and this
project has already shipped one length claim that was false (D101) and one capability
axis with a tokenisation defect (D89). Pairs of unequal length are DROPPED and named.
"""

import json
import os
import subprocess
import sys
import time

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 64

# A23's two sequences, verbatim from its manifest, so P1 compares against banked orbits
SEQS = ("1 6 0 1 7 7 8 1", "3 5 5 3 6 1 5 2")

# Labels from A23/D135. Cross-regime pairs first, then the P3 within-regime controls.
PAIRS = (
    ("symbol", "element", "cross"),   # D132's original pair
    ("array", "token", "cross"),
    ("block", "digit", "cross"),
    ("signal", "letter", "cross"),
    ("symbol", "array", "within-rot"),
    ("element", "token", "within-set"),
)
PROBE = "set"                  # D137's one in-gap token, for P5
TS = tuple(i / 20 for i in range(21))    # 0.00 .. 1.00 step 0.05
MARKER = "A"
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 9000


def prompt_for(word, seq, mk=MARKER):
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("einterp.json")
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

    # ONE implementation of R, shared with scripts/run_rotation_continuum.py.
    # Imported after the clone+install above, per docs/REMOTE_RUNS.md.
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
        """The prelude output, captured from the adapter's input on the first call.

        Same mechanism as ds_estream's `prelude_e`; `e` is the second half of the
        concatenation `adapter(cat[x, input_embeds])`.
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

    def orbit(ids, donor_e=None):
        """Full trajectory of the last position under a patched `e`.

        `donor_e` replaces the `e` half at EVERY unroll, which is correct here and is
        NOT the error ds_estream documents for the state arm: `e` is a parameter the
        model re-injects every unroll by construction, so holding it fixed is running
        the model's own map at a different parameter. The state arm is different --
        overwriting the STATE every unroll pins the trajectory, and that is why
        ds_estream injects the state once.
        """
        traj = []

        def pre(_m, inp):
            cur = inp[0]
            if donor_e is not None:
                cur = torch.cat([cur[..., :d_model], donor_e], dim=-1)
            return (cur,)

        def post(_m, _i, o):
            traj.append(o.detach()[0, -1].float().cpu().numpy().copy())

        handles = [adapter.register_forward_pre_hook(pre),
                   core_last.register_forward_hook(post)]
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            for h in handles:
                h.remove()
        return np.array(traj, dtype=np.float32)

    t0 = time.time()
    rows, dropped = [], []
    e_probe = {}

    # capture the P5 probe token's `e` on each sequence
    for si, seq in enumerate(SEQS):
        e_probe[si] = prelude_e(encode(prompt_for(PROBE, seq)))

    for (rot_w, set_w, kind) in PAIRS:
        for si, seq in enumerate(SEQS):
            ids_r = encode(prompt_for(rot_w, seq))
            ids_s = encode(prompt_for(set_w, seq))
            if ids_r.shape[1] != ids_s.shape[1]:
                dropped.append({"pair": f"{rot_w}/{set_w}", "seq": si,
                                "n_a": int(ids_r.shape[1]), "n_b": int(ids_s.shape[1])})
                print(f"DROPPED {rot_w}/{set_w} seq{si}: "
                      f"{ids_r.shape[1]} != {ids_s.shape[1]} tokens", flush=True)
                continue
            e_r, e_s = prelude_e(ids_r), prelude_e(ids_s)

            # P5: where the probe token falls on this axis, as a scalar t.
            # Gated on shape -- `set` need not tokenise to the same length as the
            # pair, and a silent broadcast would produce a number that means nothing.
            axis = (e_r - e_s).flatten()
            den = float(axis @ axis)
            if e_probe[si].shape == e_s.shape and den > 1e-12:
                proj = float((e_probe[si] - e_s).flatten() @ axis / den)
            else:
                proj = None

            # the unpatched settling orbit: P1's reference, computed here so the
            # instrument check needs no external file
            base_R = rotation_power(orbit(ids_s, donor_e=None))
            base_rot_R = rotation_power(orbit(ids_r, donor_e=None))

            for t in TS:
                e_t = (1.0 - t) * e_s + t * e_r
                # the CARRIER prompt is held fixed at the settling noun for every t,
                # so the only thing that moves across the sweep is `e`
                arr = orbit(ids_s, donor_e=e_t)
                R = rotation_power(arr)
                tag = f"{rot_w}-{set_w}_s{si}_t{t:.2f}"
                np.save(os.path.join(OUTDIR, tag + ".npy"), arr.astype(np.float16))
                rows.append({"rot_w": rot_w, "set_w": set_w, "kind": kind, "seq": si,
                             "t": t, "R": R, "tag": tag, "probe_t": proj,
                             "base_R": base_R, "base_rot_R": base_rot_R,
                             "n_tokens": int(ids_s.shape[1]), "ok": True})
                print(f"{tag}  R={R:.4f}  ({time.time() - t0:.0f}s)", flush=True)
                with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                    json.dump(rows, f)
            if time.time() - t0 > WALL_BUDGET_S:
                print("WALL BUDGET -- banking and stopping cleanly", flush=True)
                break

    summary = {"rows": rows, "dropped": dropped, "num_steps": NUM_STEPS,
               "seqs": list(SEQS), "ts": list(TS), "probe": PROBE,
               "elapsed_s": time.time() - t0}
    with open(out_path, "w") as f:
        json.dump(summary, f)
    with open(os.path.join(OUTDIR, "einterp.json"), "w") as f:
        json.dump(summary, f)
    print(f"DONE {len(rows)} orbits, {len(dropped)} dropped pairs, "
          f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
