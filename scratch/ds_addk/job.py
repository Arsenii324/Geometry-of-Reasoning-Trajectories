"""B4: the powered, pre-registered test of the project's ONLY positive H2 direction.

WHAT IS BEING RETESTED. D101 found the depth at which the gold answer is best
available rising with difficulty -- the behavioural form of H2, after every
geometric instrument returned nothing (D28, D74(6), D83, D100). But its kernel's
p-values were pseudoreplicated ~12x (3 items x 4 draws per level pooled as
independent), and at the correct unit `nth_item_k` gave rho = +0.437, p = 0.033
and `addk` rho = +0.362, p = 0.107 -- neither clearing Bonferroni over 6 families.

WHY `addk` AND NOT `nth_item_k`. D107 re-scored the capability axis at a readout
depth chosen PER FAMILY on held-out items, as D103 requires. `nth_item_k`
collapsed from 19.8% to 4.2% -- its apparent dynamic range was largely an artefact
of maximising over depth -- while `addk` held 41.7% -> 38.1%. **So the ladder that
survives honest scoring is `addk`, and D100's recommendation of `nth_item_k` is
superseded.** `addk` is also length-constant by construction and single-token-gold
by constraint (v + k <= 9), so it is free of the D89 defect.

PRE-REGISTERED, FIXED BEFORE THIS RUNS:
P1  UNIT. One value per (level, item). Never per forward -- that was D101's error,
    and `traj_geom.rigor.require_units` enforces it here.
P2  STATISTIC. Spearman(k, best_depth) at that unit. ONE statistic, not five.
P3  NULL. `best_depth` permuted WITHIN level, 2000 draws; p = (hits+1)/(n+1).
P4  THRESHOLD. alpha = 0.05. One family, one statistic, so no correction is owed
    -- which is the point of narrowing from D101's six families.
P5  GATE. The token count must be CONSTANT across levels, checked in-kernel. If it
    is not, the run reports VOID rather than a confounded correlation.
P6  ACCURACY CONTROL. Spearman(k, accuracy) is reported alongside. If accuracy
    falls with k, a best_depth shift could be an artefact of harder items being
    answered wrongly rather than later; D101 found accuracy FLAT for its ladder,
    which is what made the depth shift interesting.

THE PREDICTION. D101's direction was positive at rho = +0.362 for this family.
This run has ~6x the draws per cell. **If H2 has any behavioural footing, it shows
here; if it does not clear alpha = 0.05 with this power on its best family, the
behavioural arm joins the geometric ones.**
"""

import json
import os
import string
import subprocess
import sys


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 48
N_ITEMS = 6          # distinct v values per level (P1's independent unit is (level,item))
N_DRAWS = 8          # unseeded h_0 draws per item -> 48 forwards per level, ~6x D101
N_PERM = 2000        # P3

L = string.ascii_lowercase

# (family, length_constant, levels, item_fn) -- item_fn(level, i) -> (prompt, gold)
FAMILIES = {}


def _caesar(k, i):
    ch = L[(i * 7 + 2) % 26]
    return (f"Shift the letter {ch} forward by {k} in the alphabet. Answer with one letter.",
            L[(L.index(ch) + k) % 26])


def _addk(k, i):
    """v is chosen so v + k NEVER reaches 10.

    Otherwise the gold silently becomes two tokens exactly at the harder levels,
    which would make D89's multi-token defect correlate with the difficulty knob
    -- the readout would degrade with k for a tokenisation reason and look like a
    capability decline. Caught before running.
    """
    v = i % max(1, 10 - k)          # v + k <= 9 always, so gold stays single-token
    return (f"What is {v} + {k}? Answer with the number.", str(v + k))


def _nth(k, i):
    xs = [(i * 5 + j * 3) % 10 for j in range(8)]
    return (f"What is item number {k} in this list? Answer with the number.\n"
            f"List: {' '.join(map(str, xs))}", str(xs[k - 1]))


def _bits(n, i, salt):
    """Deterministic but genuinely varied bits.

    The first version used `(i*a + j*b) % 3 % 2`, which is CONSTANT in j whenever
    b is a multiple of 3 -- `track_n` came out with every operation identical, so
    its "difficulty" levels were all the same trivial task. Use an explicit seeded
    RNG instead of hand-rolled arithmetic.
    """
    import random
    rng = random.Random(1000 * salt + i)
    return [rng.randint(0, 1) for _ in range(n)]


def _count(n, i):
    bits = _bits(n, i, 1)
    return ("Count how many ones are in this sequence. Answer with the number.\n"
            f"Sequence: {' '.join(map(str, bits))}", str(sum(bits)))


def _parity(n, i):
    bits = _bits(n, i, 2)
    return ("Is the number of ones in this sequence even or odd? "
            "Answer 0 for even and 1 for odd.\n"
            f"Sequence: {' '.join(map(str, bits))}", str(sum(bits) % 2))


def _track(n, i):
    ops = [1 if b else -1 for b in _bits(n, i, 3)]
    body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1." for o in ops)
    return (body + " What is the final total? Answer with the number.", str(sum(ops)))


FAMILIES["addk"] = (True, list(range(1, 8)), _addk)   # v+k <= 9, single-token gold
# addk ONLY: D107 showed it is the ladder that survives honest fixed-depth
# scoring (41.7% -> 38.1%) while nth_item_k collapses to 4.2%.


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("grid.json")
    print(f"results -> {out_path}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))   # pip and python may differ; see REMOTE_RUNS.md

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def coda_head(h_state, freqs_cis):
        """Verbatim D71-validated tail: TWO ln_f calls, coda blocks, lm_head."""
        x = model.transformer.ln_f(h_state)
        block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            block_idx -= 1
            x = block(x, freqs_cis, block_idx, None, None)
        x = model.transformer.ln_f(x)
        return model.lm_head(x)

    def one(prompt, gold):
        """One forward -> the gold rank at EVERY depth. The whole design rests on
        this being a single continuous run rather than one run per depth."""
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        g_ids = tok(gold, add_special_tokens=False).input_ids
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        ranks = []
        mod = model.transformer.core_block[-1]
        mod._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                row = torch.log_softmax(
                    coda_head(o.detach(), freqs).float()[0, n_p - 1], dim=-1)
                ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)

        h = mod.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        return {"rank_curve": ranks, "n_tokens": int(n_p),
                "best_rank": int(min(ranks)), "best_depth": int(np.argmin(ranks)) + 1,
                "correct": bool(min(ranks) == 1),
                "multi_token_gold": bool(len(g_ids) > 1)}

    rows = []
    for fam, (len_const, levels, fn) in FAMILIES.items():
        print(f"\n=== {fam}  (length_constant={len_const}, "
              f"levels {levels[0]}..{levels[-1]}) ===", flush=True)
        for lv in levels:
            for i in range(N_ITEMS):
                prompt, gold = fn(lv, i)
                for _ in range(N_DRAWS):
                    try:
                        r = one(prompt, gold)
                    except Exception as exc:
                        r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
                    r.update({"family": fam, "level": lv, "item": i,
                              "length_constant": len_const, "gold": gold})
                    r.setdefault("ok", True)
                    rows.append(r)
            cell = [r for r in rows
                    if r["family"] == fam and r["level"] == lv and r.get("ok")]
            acc = float(np.mean([r["correct"] for r in cell]))
            bd = float(np.median([r["best_depth"] for r in cell]))
            ntok = sorted({r["n_tokens"] for r in cell})
            print(f"  level {lv:>3}: acc {acc:6.1%}  median best_depth {bd:5.1f}  "
                  f"n_tokens {ntok[0]}-{ntok[-1]}  (n={len(cell)})", flush=True)
        json.dump(rows, open(out_path, "w"))

    # THE TOKEN-COUNT GATE, checked rather than assumed (D87's discipline).
    print("\n=== TOKEN-COUNT GATE: is 'length_constant' actually true? ===", flush=True)
    for fam, (len_const, _lv, _f) in FAMILIES.items():
        ns = sorted({r["n_tokens"] for r in rows if r["family"] == fam and r.get("ok")})
        ok = (len(ns) == 1)
        verdict = "CONSTANT" if ok else f"VARIES {ns[0]}-{ns[-1]}"
        flag = "" if ok == len_const else "   <-- CLAIM DOES NOT MATCH MEASUREMENT"
        print(f"  {fam:>12}: claimed length_constant={len_const}, measured {verdict}{flag}",
              flush=True)

    # ---- P1-P6, implemented at the CORRECT UNIT --------------------------------
    from scipy.stats import spearmanr
    sys.path.insert(0, os.path.abspath("src"))
    from traj_geom.rigor import require_units

    ok = [r for r in rows if r.get("ok")]

    # P5 GATE: token count must be constant, CHECKED not asserted.
    ntok = sorted({r["n_tokens"] for r in ok})
    print("\n=== P5 TOKEN GATE ===", flush=True)
    if len(ntok) != 1:
        print(f"  VOID: n_tokens varies {ntok[0]}-{ntok[-1]} across levels, so any "
              f"best_depth correlation is confounded with prompt length. "
              f"Reporting nothing.", flush=True)
        json.dump({"verdict": "VOID_length_varies", "rows": rows},
                  open(out_path, "w"))
        return
    print(f"  PASSED: n_tokens == {ntok[0]} at every level", flush=True)

    # P1: one value per (level, item). Never per forward -- that was D101's error.
    cells = {}
    for r in ok:
        cells.setdefault((r["level"], r["item"]), []).append(r)
    keys = [f"L{lv}_i{it}" for (lv, it) in cells]
    require_units(keys, min_units=12, what="addk (level,item) cells")
    lv = np.array([k[0] for k in cells])
    bd = np.array([float(np.mean([r["best_depth"] for r in v])) for v in cells.values()])
    acc = np.array([float(np.mean([r["correct"] for r in v])) for v in cells.values()])
    print(f"\n=== P1: {len(cells)} independent (level,item) units, "
          f"{len(ok)} forwards ===", flush=True)

    # P2 + P3: one statistic, permutation null WITHIN level.
    rho = spearmanr(lv, bd).statistic
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(N_PERM):
        perm = bd.copy()
        for L in set(lv):
            m = lv == L
            perm[m] = rng.permutation(bd[m])
        if spearmanr(lv, perm).statistic >= rho:
            hits += 1
    p_perm = (hits + 1) / (N_PERM + 1)

    # P6: the accuracy control.
    rho_a, p_a = spearmanr(lv, acc)

    print("\n=== P2/P3/P4: THE PRE-REGISTERED TEST ===", flush=True)
    print(f"  Spearman(k, best_depth) at the (level,item) unit: rho = {rho:+.4f}", flush=True)
    print(f"  within-level permutation null, {N_PERM} draws: p = {p_perm:.4f}", flush=True)
    print(f"  alpha = 0.05 (one family, one statistic -- no correction owed)", flush=True)
    print(f"\n=== P6 ACCURACY CONTROL ===", flush=True)
    print(f"  Spearman(k, accuracy) = {rho_a:+.4f}, p = {p_a:.4f}", flush=True)
    print(f"  per-level accuracy: "
          f"{[f'{L}:{float(np.mean(acc[lv==L])):.2f}' for L in sorted(set(lv))]}",
          flush=True)

    print("\n=== VERDICT ===", flush=True)
    if p_perm < 0.05:
        print(f"  H2 SUPPORTED behaviourally on addk: best_depth rises with "
              f"difficulty at rho = {rho:+.3f}, p = {p_perm:.4f}, at the correct "
              f"unit and at a verified-constant token count.", flush=True)
        if p_a < 0.05 and rho_a < 0:
            print(f"  CAVEAT: accuracy also falls with k (rho = {rho_a:+.3f}, "
                  f"p = {p_a:.4f}), so the depth shift may be an artefact of harder "
                  f"items being answered wrongly rather than later. D101's ladder "
                  f"had FLAT accuracy, which is what made it interesting.", flush=True)
        else:
            print(f"  And accuracy does NOT fall significantly with k, so this is "
                  f"not an artefact of harder items simply failing.", flush=True)
    else:
        print(f"  NOT SUPPORTED: p = {p_perm:.4f} >= 0.05 with ~6x D101's draws on "
              f"its best surviving family. The behavioural arm of H2 joins the "
              f"geometric ones.", flush=True)

    json.dump({"rows": rows, "rho": float(rho), "p_perm": float(p_perm),
               "rho_acc": float(rho_a), "p_acc": float(p_a),
               "n_units": len(cells), "n_forwards": len(ok),
               "config": {"num_steps": NUM_STEPS, "n_items": N_ITEMS,
                          "n_draws": N_DRAWS, "n_perm": N_PERM}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}\nDONE", flush=True)


def _unused_legacy_analysis():
    print("\n=== PRE-REGISTERED CHECKS ===", flush=True)
    band = []
    for fam, (len_const, levels, _f) in FAMILIES.items():
        for lv in levels:
            cell = [r for r in rows
                    if r["family"] == fam and r["level"] == lv and r.get("ok")]
            if cell:
                a = float(np.mean([r["correct"] for r in cell]))
                if 0.2 <= a <= 0.8:
                    band.append((fam, lv, a, len_const))
    lc = [b for b in band if b[3]]
    print(f"  (1) cells in the 20-80% band: {len(band)} total, "
          f"{len(lc)} of them LENGTH-CONSTANT", flush=True)
    for b in band:
        print(f"        {b[0]:>12} level {b[1]:>3}: {b[2]:.1%}"
              f"{'  [length-constant]' if b[3] else ''}", flush=True)
    if not lc:
        print("      NO length-constant cell lands in the band -- the difficulty-knob "
              "route fails on this model and CLRS-Text becomes the fallback.", flush=True)

    from scipy.stats import spearmanr
    for fam, (len_const, levels, _f) in FAMILIES.items():
        pts = [(r["level"], r["best_depth"]) for r in rows
               if r["family"] == fam and r.get("ok")]
        accs = [(lv, float(np.mean([r["correct"] for r in rows
                                    if r["family"] == fam and r["level"] == lv
                                    and r.get("ok")]))) for lv in levels]
        if len(set(p[0] for p in pts)) > 2:
            rho_d, p_d = spearmanr([p[0] for p in pts], [p[1] for p in pts])
            rho_a, p_a = spearmanr([a[0] for a in accs], [a[1] for a in accs])
            print(f"  (2,3) {fam:>12}: acc-vs-level rho={rho_a:+.3f} (p={p_a:.3f}) | "
                  f"best_depth-vs-level rho={rho_d:+.3f} (p={p_d:.4f})", flush=True)

    json.dump({"rows": rows,
               "config": {"num_steps": NUM_STEPS, "n_items": N_ITEMS,
                          "n_draws": N_DRAWS, "model": MODEL_ID,
                          "revision": REVISION,
                          "families": {k: [v[0], v[1]] for k, v in FAMILIES.items()}}},
              open(out_path, "w"))
    print(f"\nwrote {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
