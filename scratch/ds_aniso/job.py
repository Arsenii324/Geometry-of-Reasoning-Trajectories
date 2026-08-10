"""A31: WHAT is the rotating region elongated along? A targeted test, not blind PCA.

WHERE THIS STANDS. D149 established that the rotating set in `e`-space is bounded and
anisotropic -- 81.2% of random rays escape it, both members of 20 of 32 antipodal pairs
escape, and crossing distances vary by a factor of 49 (0.053 to 2.572 chord units). It
also refuted the hyperplane hypothesis, which had been the tidiest available explanation
of D135/D136. Its own limits section names the gap: *"the anisotropy is described, not
quantified -- no principal axes were extracted."*

WHY NOT PCA. `e` has 53 x 5280 = 279,840 dimensions and a run of this size affords tens
of rays. Estimating a full shape operator from that is hopeless, and a top-k PCA of 64
boundary points in 280k dimensions would return directions determined by sampling noise.
**So this asks a specific question with a specific prediction instead.**

THE HYPOTHESIS, and it comes from D144 rather than from nothing. D144 found that a chord
toward another *rotating* noun keeps all 21 of 21 grid points inside the regime, while
norm-matched random bearings escape and, on 3 of 3 rotating carriers, destroy the rotation
outright. **That says the region is extended along the directions that connect rotating
tokens.** If so, a random direction's survival distance should be predicted by how much of
it lies in the span of those token-difference vectors.

  Let `V = span{ e(rot_i) - e(carrier) }` over several rotating nouns.
  For a random unit direction `u`, let `a = ||P_V u||` be its alignment with that span.
  **Prediction: d(u) rises with a.** Directions inside V stay in the regime far longer.

  The control is the matching span built from SETTLING nouns, `W = span{ e(set_j) -
  e(carrier) }`. If alignment with W predicts d equally well, then what is being measured
  is "alignment with any token-difference direction", not anything about the rotating
  regime, and the result means much less. **Both are computed on the same rays.**

PREREGISTERED PREDICTIONS:

  P1  INSTRUMENT NULL, VOID WITHOUT IT. A zero-displacement patch must reproduce the
      unpatched orbit's R exactly (0.000e+00), as it did in A25 on 8 chords.

  P2  PRIMARY. Spearman(alignment with the ROTATING span, crossing distance) > 0 at
      permutation p < 0.05, over the uncensored rays. Censored rays -- those that never
      leave within the cap -- are handled explicitly in P4, never averaged in as a large
      distance.

  P3  CONTROL. The same correlation against the SETTLING span. The finding is only about
      rotating structure if P2 exceeds P3; if they match, that is reported as the result
      and P2 is not claimed.

  P4  CENSORING. Rays that never escape are the most informative points for this
      hypothesis, not a nuisance -- under it they should be the MOST aligned. So the
      analysis reports (a) the correlation on uncensored rays and (b) the alignment of
      censored rays against uncensored ones, which is a censoring-free test of the same
      prediction and does not require a distance at all.

  P5  DIMENSION SANITY. Both spans are built from the same number of nouns so their
      dimensions match; a larger span captures more of any random vector by construction,
      and comparing spans of different rank would manufacture P2 > P3.
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
CARRIER = "symbol"
SEQ = "1 6 0 1 7 7 8 1"
# equal counts, so the two spans have equal rank (P5)
ROT_NOUNS = ("array", "block", "signal", "chunk", "node", "pattern")
SET_NOUNS = ("element", "token", "digit", "letter", "cell", "field")
N_DIR = 28
MAX_DIST_MULT = 3.0
BISECT_STEPS = 8
MARKER = "A"
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 11000


def build_items(rng=None):
    return [{"carrier": CARRIER, "seq": SEQ, "item": 0}]


def prompt_for(word, seq, mk=MARKER):
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("aniso.json")
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
            if e_new is None:
                return (cur,)
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

    ids = encode(prompt_for(CARRIER, SEQ))
    e0 = prelude_e(ids)
    n_tok = ids.shape[1]
    print(f"carrier `{CARRIER}` at {n_tok} tokens", flush=True)

    # P1: zero displacement must be an exact no-op
    r_patched, r_unp = R_at(ids, e0), R_at(ids, None)
    print(f"P1 no-op: patched {r_patched:.12f} vs unpatched {r_unp:.12f}  "
          f"d={abs(r_patched - r_unp):.3e}", flush=True)

    def span_basis(nouns):
        """Orthonormal basis of span{e(noun) - e(carrier)}, dropped if length differs."""
        cols, used = [], []
        for w in nouns:
            i2 = encode(prompt_for(w, SEQ))
            if i2.shape[1] != n_tok:
                print(f"  dropped {w}: {i2.shape[1]} != {n_tok} tokens", flush=True)
                continue
            cols.append((prelude_e(i2) - e0).flatten())
            used.append(w)
        if not cols:
            return None, []
        M = torch.stack(cols, dim=1)                 # [D, k]
        Q, _ = torch.linalg.qr(M.float())
        return Q, used

    Qr, rot_used = span_basis(ROT_NOUNS)
    Qs, set_used = span_basis(SET_NOUNS)
    kr = Qr.shape[1] if Qr is not None else 0
    ks = Qs.shape[1] if Qs is not None else 0
    print(f"P5 span ranks: rotating {kr} ({rot_used}), settling {ks} ({set_used})",
          flush=True)
    if kr == 0 or ks == 0:
        print("VOID: a span is empty")
        return 1

    scale = float(torch.linalg.norm(Qr[:, 0])) or 1.0
    ref = encode(prompt_for("element", SEQ))
    chord = float((prelude_e(ref) - e0).norm()) if ref.shape[1] == n_tok else float(e0.norm()) * 0.1
    max_dist = MAX_DIST_MULT * chord
    print(f"chord scale {chord:.3f}, max_dist {max_dist:.3f}", flush=True)

    g = torch.Generator(device=e0.device).manual_seed(SEED)
    t0, rows = time.time(), []
    for k in range(N_DIR):
        u = torch.randn(e0.shape, generator=g, device=e0.device, dtype=e0.dtype)
        u = u / u.norm()
        uf = u.flatten().float()
        a_rot = float(torch.linalg.norm(Qr.T @ uf))     # ||P_V u||, u is unit
        a_set = float(torch.linalg.norm(Qs.T @ uf))
        far = R_at(ids, e0 + max_dist * u)
        rec = {"dir": k, "align_rot": a_rot, "align_set": a_set,
               "R_far": far, "chord": chord, "max_dist": max_dist,
               "rank_rot": kr, "rank_set": ks, "ok": True}
        if (far > THRESHOLD) == (r_patched > THRESHOLD):
            rec.update({"censored": True, "d": None, "d_rel": None})
            print(f"  dir{k:2d} a_rot={a_rot:.4f} a_set={a_set:.4f}  CENSORED", flush=True)
        else:
            lo, hi = 0.0, max_dist
            for _ in range(BISECT_STEPS):
                mid = 0.5 * (lo + hi)
                if (R_at(ids, e0 + mid * u) > THRESHOLD) == (r_patched > THRESHOLD):
                    lo = mid
                else:
                    hi = mid
            d = 0.5 * (lo + hi)
            rec.update({"censored": False, "d": d, "d_rel": d / chord})
            print(f"  dir{k:2d} a_rot={a_rot:.4f} a_set={a_set:.4f}  d={d / chord:.3f} chords "
                  f"({time.time() - t0:.0f}s)", flush=True)
        rows.append(rec)
        with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
            json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    unc = [r for r in rows if not r["censored"]]
    cen = [r for r in rows if r["censored"]]
    print(f"\n{len(unc)} uncensored, {len(cen)} censored", flush=True)
    if len(unc) >= 6:
        def sp(x, y):
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            return float(np.corrcoef(rx, ry)[0, 1])
        dd = np.array([r["d_rel"] for r in unc])
        print(f"P2 Spearman(align_rot, d) = {sp(np.array([r['align_rot'] for r in unc]), dd):+.4f}",
              flush=True)
        print(f"P3 Spearman(align_set, d) = {sp(np.array([r['align_set'] for r in unc]), dd):+.4f}",
              flush=True)
    if cen and unc:
        print(f"P4 mean align_rot: censored {np.mean([r['align_rot'] for r in cen]):.4f} "
              f"vs uncensored {np.mean([r['align_rot'] for r in unc]):.4f}", flush=True)
        print(f"   mean align_set: censored {np.mean([r['align_set'] for r in cen]):.4f} "
              f"vs uncensored {np.mean([r['align_set'] for r in unc]):.4f}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "carrier": CARRIER, "seq": SEQ, "threshold": THRESHOLD,
                   "rot_nouns": list(rot_used), "set_nouns": list(set_used),
                   "seed": SEED, "p1_noop_delta": abs(r_patched - r_unp),
                   "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} rays, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
