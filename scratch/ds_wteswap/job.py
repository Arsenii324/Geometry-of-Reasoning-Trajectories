"""A32: control the regime from the EMBEDDING MATRIX, not from `e`. The causal version.

WHY NOW. D134 refuted meaning, form and tokenisation as the selector of the rotation
regime, then inferred *by elimination* that the property lives in the token's learned
embedding. D135 and D136 appeared to refute that survivor by fitting classifiers to 49
banked embeddings. **D155 shows they did no such thing**: the design cannot resolve a
linear class separation below Cohen's d ~ 10 — where 0.8 is conventionally large — and the
observed data is indistinguishable from planted separations of 0.25, 0.5 and 1.0. So the
embedding hypothesis is **untested, not refuted**, and D149 has since removed the
hyperplane picture that would have explained a linear-in-`e` structure being invisible in
the embedding.

The lesson of A24/A25 applies exactly: **stop fitting and manipulate.** There the object
was `e`, the prelude's output. Here it is `wte`, the embedding matrix itself — one row of
it, for one token.

THE DESIGN. Hold the prompt text fixed at the ROTATING noun, and interpolate that token's
**embedding row** toward a settling noun's:

    wte[tok(symbol)]  <-  (1 - t) * wte[tok(symbol)] + t * wte[tok(element)]

Nothing else changes: same string, same token ids, same positions, same length, same h_0.
At t = 1 the rotating token's embedding IS the settling token's, so the forward pass is
input-identical to the settling prompt and the regime MUST flip — that is not a finding,
it is a sanity check (P2). **The finding is the SHAPE of the crossing**, and it is a direct
causal measurement of what the fitted classifiers could not see.

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, VOID WITHOUT IT. At t = 0 the substitution is the identity, so R
      must reproduce the unpatched orbit EXACTLY (0.000e+00, as A25 achieved on 8 chords).
      Any deviation means the write is not clean and nothing else may be read.

  P2  SANITY, NOT A RESULT. At t = 1 the run must match the settling noun's own orbit,
      because the embedding is then literally that token's. If it does not, the mechanism
      is not what this docstring says and the run is void.

  P3  PRIMARY — SHARPNESS. Where between 0 and 1 does R cross D141's threshold, and how
      sharply? A24 found the `e`-space transition sharper than a 0.05 grid step. **If the
      wte-space transition is comparably sharp, the property is carried by a small,
      localised part of the embedding difference**; if it is a slow ramp, the regime
      depends on the embedding diffusely and the fitting failure of D135/D136 becomes
      unsurprising for a reason other than sample size.

  P4  IS THE CROSSING POINT A PROPERTY OF THE PAIR OR OF THE TARGET? Four settling targets
      are swept from the same rotating source. If t* is roughly constant across targets,
      what matters is distance travelled away from the source; if t* varies widely, the
      destination matters and the boundary is not a sphere around the source.

  P5  WITHIN-REGIME CONTROL, the load-bearing one. Interpolating toward another ROTATING
      noun must NOT flip the regime. Without it, "any large edit to an embedding row
      destroys rotation" explains P3 just as well — and D144 already showed that a
      norm-matched random bearing in `e`-space destroys rotation on 3 of 3 carriers, so
      this failure mode is real and observed, not hypothetical.
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
THRESHOLD = 0.6677
SOURCE = "symbol"                      # rotating; its wte row is the one edited
SET_TARGETS = ("element", "token", "digit", "letter")
ROT_TARGETS = ("array", "block")       # P5 controls
SEQS = ("1 6 0 1 7 7 8 1", "3 5 5 3 6 1 5 2")
TS = tuple(i / 20 for i in range(21))
MARKER = "A"
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 11000


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
        os.path.abspath("wteswap.json")
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

    wte = model.transformer.wte.weight
    core_last = model.transformer.core_block[-1]

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    def orbit_R(ids):
        traj = []

        def post(_m, _i, o):
            traj.append(o.detach()[0, -1].float().cpu().numpy().copy())

        h = core_last.register_forward_hook(post)
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        return rotation_power(np.array(traj, dtype=np.float32))

    def one_token(word):
        """The word must be a SINGLE token for a row swap to be well defined."""
        t_ids = tok(f" {word}", add_special_tokens=False).input_ids
        t2 = tok(word, add_special_tokens=False).input_ids
        return (t_ids[0], " " + word) if len(t_ids) == 1 else \
               ((t2[0], word) if len(t2) == 1 else (None, None))

    src_id, src_form = one_token(SOURCE)
    print(f"source `{SOURCE}` -> token id {src_id} (form {src_form!r})", flush=True)
    if src_id is None:
        print("VOID: the source noun is not a single token; a row swap is undefined")
        return 1

    t0, rows, dropped = time.time(), [], []
    orig = wte[src_id].detach().clone()

    for target in SET_TARGETS + ROT_TARGETS:
        kind = "within-rot" if target in ROT_TARGETS else "cross"
        tgt_id, tgt_form = one_token(target)
        if tgt_id is None:
            dropped.append({"target": target, "why": "not a single token"})
            print(f"  DROPPED {target}: not a single token", flush=True)
            continue
        tgt_vec = wte[tgt_id].detach().clone()
        for it in build_items():
            ids = encode(prompt_for(SOURCE, it["seq"]))
            with torch.no_grad():
                wte[src_id] = orig
            base = orbit_R(ids)
            ids_t = encode(prompt_for(target, it["seq"]))
            tgt_own = orbit_R(ids_t) if ids_t.shape[1] == ids.shape[1] else float("nan")
            for t in TS:
                with torch.no_grad():
                    wte[src_id] = (1.0 - t) * orig + t * tgt_vec
                R = orbit_R(ids)
                rows.append({"source": SOURCE, "target": target, "kind": kind,
                             "seq": it["seq"], "item": it["item"], "t": t, "R": R,
                             "base_R": base, "target_own_R": tgt_own,
                             "n_tokens": int(ids.shape[1]), "ok": True})
                with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                    json.dump(rows, f)
            with torch.no_grad():
                wte[src_id] = orig          # always restore before the next pair
            n_rot = sum(1 for r in rows[-len(TS):] if r["R"] > THRESHOLD)
            print(f"  {SOURCE}->{target} ({kind}) seq{it['item']}: base {base:.3f}, "
                  f"target's own {tgt_own:.3f}, {n_rot}/{len(TS)} rotating "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    with torch.no_grad():
        wte[src_id] = orig
    summary = {"rows": rows, "dropped": dropped, "source": SOURCE,
               "set_targets": list(SET_TARGETS), "rot_targets": list(ROT_TARGETS),
               "threshold": THRESHOLD, "seed": SEED, "num_steps": NUM_STEPS,
               "elapsed_s": time.time() - t0}
    with open(out_path, "w") as f:
        json.dump(summary, f)
    print(f"DONE {len(rows)} orbits, {len(dropped)} dropped, {time.time() - t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
