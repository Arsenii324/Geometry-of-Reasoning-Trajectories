"""A25: is the rotating region in `e`-space non-convex, or was that one pair?

WHY THIS EXISTS. D140's novel structural claim rests on ONE pair. Interpolating `e`
between `element` and `token` -- both SETTLING nouns -- produced an orbit that crossed
into the rotating regime and back, twice, on both sequences. If that generalises, the
rotating set in `e`-space is not convex and the regime is not a half-space; if it is
specific to that pair, it is an anecdote and D140(4) must be narrowed.

This is the cheapest test that could refute it: six settling->settling chords instead
of one, plus three rotating->rotating chords as the control that a chord inside a
region stays inside it.

AND IT FIXES D140's UNEVALUABLE GATE. D140's P1 required R(t=0) to equal the unpatched
orbit, which is right about the patch and wrong about the forward: D78 records that
`initialize_state` draws h_0 from an UNSEEDED RNG and that no kernel seeds it, so two
forwards of one prompt cannot agree. **Here every forward is preceded by
`torch.manual_seed(SEED)`**, which is the prescription `constants.py` has carried all
along. P1 then becomes a real identity check rather than a confound.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, AND THE RUN IS VOID WITHOUT IT. With h_0 seeded, the t = 0 patch
      substitutes `e_s` for `e_s` and must reproduce the unpatched orbit to floating
      point: |R(0) - R_base| < 1e-9. D140 could not make this check; if it fails HERE,
      the hook writes something other than `e` and nothing else may be read.

  P2  PRIMARY -- DOES NON-CONVEXITY GENERALISE? Count settling->settling chords whose R
      crosses the rotation threshold (R > 0.6677, D141). D140 saw 2 of 2. The
      prediction registered before running: **at least 3 of 6 cross.** Fewer than 2 and
      D140(4) is withdrawn as a one-pair artefact; 0 of 6 would mean the `element`/
      `token` chord was special and should be investigated on its own terms.

  P3  CONTROL. Rotating->rotating chords must NOT leave the regime, as in D140 (0
      crossings in 21 points). If these DO leave it, the finding is not non-convexity
      but "interpolated `e` is off-distribution", which is a different and duller
      claim -- and it is the reading this control exists to exclude.

  P4  OFF-DISTRIBUTION CONTROL, and it is the one that could kill the whole design.
      An interpolated `e` is not any token's `e`. If the midpoint of a chord is simply
      an implausible parameter, the model might rotate there for reasons unrelated to
      the boundary. So each chord's midpoint `e` is also compared against a
      NORM-MATCHED RANDOM direction from `e_s`: same distance travelled, random
      bearing. If random directions rotate as often as real chords, P2 means nothing.
"""

import json
import os
import subprocess
import sys
import time

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 64
SEED = 20260810          # D78/constants.py: h_0 is random; seeding it is the prescription

# A23's two sequences, verbatim from its manifest, so P1 compares against banked orbits
SEQS = ("1 6 0 1 7 7 8 1",)   # ONE sequence: h_0 is seeded, so a second adds
#                               nothing here. (An `N_SEQ = 1` constant was
#                               declared alongside this and never read; the
#                               preregistration guard caught it. A constant
#                               that parameterises nothing is how a gate gets
#                               declared and not written -- A18's failure.)

# Labels from A23/D135. Cross-regime pairs first, then the P3 within-regime controls.
# labels from A23/D135. `a` is the far end of the chord, `b` the carrier.
PAIRS = (
    ("element", "token", "within-set"),    # D140's pair, replicated first
    ("digit", "letter", "within-set"),
    ("cell", "field", "within-set"),
    ("glyph", "numeral", "within-set"),
    ("entry", "point", "within-set"),
    ("form", "image", "within-set"),
    ("symbol", "array", "within-rot"),      # P3 controls
    ("block", "signal", "within-rot"),
    ("chunk", "node", "within-rot"),
)
PROBE = "set"                  # D141's one straddling token
# P2's threshold (D141's R > 0.6677, which reproduces the binary label on 691/696
# orbits) is applied in the ANALYSIS, not here: this kernel banks R at every grid
# point, so the crossing count is fully recoverable downstream. It was originally
# declared as a module constant here and never read -- the preregistration guard
# caught that, twice on this file, which is exactly the A18 failure mode (a gate
# declared in the docstring and never written).
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
        os.path.abspath("nonconvex.json")
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
                torch.manual_seed(SEED)      # P1: h_0 is drawn here (D78)
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
                torch.manual_seed(SEED)      # P1: identical h_0 for every forward
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

            # P4 OFF-DISTRIBUTION CONTROL: a norm-matched random bearing from e_s,
            # travelling the same distance as the chord's midpoint. If a random
            # direction rotates as often as the real chord, P2 measures nothing.
            g = torch.Generator(device=e_s.device).manual_seed(SEED + si)
            step = 0.5 * (e_r - e_s)
            rnd = torch.randn(e_s.shape, generator=g, device=e_s.device, dtype=e_s.dtype)
            rnd = rnd / rnd.norm() * step.norm()
            rand_R = rotation_power(orbit(ids_s, donor_e=e_s + rnd))

            for t in TS:
                e_t = (1.0 - t) * e_s + t * e_r
                # the CARRIER prompt is held fixed at the settling noun for every t,
                # so the only thing that moves across the sweep is `e`
                arr = orbit(ids_s, donor_e=e_t)
                R = rotation_power(arr)
                tag = f"{rot_w}-{set_w}_s{si}_t{t:.2f}"
                np.save(os.path.join(OUTDIR, tag + ".npy"), arr.astype(np.float16))
                rows.append({"far_w": rot_w, "carrier_w": set_w, "kind": kind, "seq": si,
                             "t": t, "R": R, "tag": tag, "probe_t": proj,
                             "base_R": base_R, "base_rot_R": base_rot_R, "rand_R": rand_R,
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
    with open(os.path.join(OUTDIR, "nonconvex.json"), "w") as f:
        json.dump(summary, f)
    print(f"DONE {len(rows)} orbits, {len(dropped)} dropped pairs, "
          f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
