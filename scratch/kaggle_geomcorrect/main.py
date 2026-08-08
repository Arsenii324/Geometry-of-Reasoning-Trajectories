"""Does trajectory GEOMETRY differ between CORRECT and INCORRECT answers?

THE GAP THIS CLOSES (directions.md B6). Every geometric measurement in this
project has POOLED correct and incorrect trajectories -- and since exact-match
accuracy is ~0 nearly everywhere, the geometry on record describes FAILURE. The
project set out to ask whether trajectory shape encodes reasoning and has never
compared the shape of a trajectory that got the answer right against one that got
it wrong. D68 makes the comparison possible for the first time by locating cells
where the trained model demonstrably succeeds.

DESIGN: CORRECTNESS AS THE ONLY FREE VARIABLE. Within ONE task, at ONE difficulty,
with prompts of identical construction, split items by whether the model's answer
is rank 1 at the depth where that task peaks (D68: count4 peaks r=2, add1 and
count16 r=4). Task, length, format, difficulty and arm are all held fixed by
construction; only correctness varies. Tasks are chosen for BALANCE, not for
maximum accuracy -- count4 (33%), add1 (25%) and count16 (8%) give both classes,
whereas echo_digit at 96% would leave almost no failures to compare against.

GEOMETRY PER ITEM, on this project's established estimators
    rho_orbit   two orbits of the SAME prompt from different random inits, fitted
                on their separation d(r) -- the D52 methodology, per item
    rho_step    fitted on the step-norm decay of one orbit
    settle      first unroll where the step norm falls below 10% of its maximum
    cos_step    mean cosine between consecutive steps
Both rho estimators use fit_rho, which fits only the PRE-FLOOR regime -- D30
showed bf16 rounding makes this model look convergent ~4.6x too early, and a fit
that includes floored points inflates rho.

THE INSTRUMENT GETS A NULL, AND THE NULL GOT CHECKED. CLAUDE.md section 5 forbids
retiring or confirming anything with an instrument that has not passed a null;
here the null is a LABEL PERMUTATION -- shuffle correct/incorrect within a task
and re-run the same statistic. Validated on synthetic data BEFORE any GPU time,
which caught a hard defect in my own first draft:

    N_PERM = 200 makes the smallest attainable p-value 1/201 = 0.00498, which is
    ABOVE the Bonferroni alpha of 0.05/12 = 0.00417. Rejection was arithmetically
    IMPOSSIBLE -- the run would have returned "no metric significant" for every
    metric, a guaranteed null that says nothing about the science. Measured power
    at 200 permutations was 0.00 even for a 2 sd shift.

    N_PERM = 5000 puts the floor at 0.0002. Calibration is then correct (p<0.05
    fires on 5.5% of null trials) and measured power at n=64 with 30% positives
    is 0.09 at 0.5 sd, 0.64 at 1.0 sd, 0.93 at 1.5 sd, 1.00 at 2.0 sd.

SO A NULL RESULT HERE MEANS "no effect of 1.5 sd or larger", NOT "no effect".
That sensitivity is stated up front so the negative in P3 cannot be over-read --
D63 is the cautionary case, where an underpowered split printed a CONFIRMED
verdict on a ratio of two null slopes.

PRE-REGISTERED PREDICTIONS, written before the run (CLAUDE.md section 1)
  P1  If trajectory geometry tracks computation, correct and incorrect items must
      differ on AT LEAST ONE of the four metrics, at permutation p < 0.05 after
      Bonferroni over 4 metrics x 3 tasks (alpha = 0.05/12 = 0.00417, attainable
      because the p-floor is now 0.0002).
  P2  NO DIRECTION IS PREDICTED. I cannot justify a sign a priori -- "correct
      answers settle sooner" and "correct answers keep moving longer" are both
      tellable stories, and picking one after seeing the data is how D22 got a
      confirmation it later had to withdraw. Effect sizes are reported with sign;
      the hypothesis under test is DIFFERENCE, not direction.
  P3  THE CLEAN NEGATIVE, READ AT ITS MEASURED SENSITIVITY. If no metric
      separates the classes in any task, then trajectory geometry does not track
      correctness at matched difficulty AT AN EFFECT SIZE OF 1.5 sd OR MORE
      (power 0.93); effects below 0.5 sd would be missed (power 0.09). That
      is a real, falsifiable answer to the project's founding question and is
      only obtainable now that successes exist -- no amount of pooled measurement
      could have produced it.

Per B4.14 the per-unroll curves (separation, step norms, rank) are persisted, so
any follow-up question can be asked locally without another GPU run.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run fit_rho


def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


def fit_rho(curve, k=3.0, tail_frac=0.25):
    """Contraction rate from a decaying curve -> (rho, n_used, fit_r2).

    Fits ``log(curve)`` linear in t over the PRE-FLOOR regime only. Finite
    arithmetic floors any such curve; including floored points drags the slope
    toward zero and INFLATES rho, which is the single most common way a
    contraction estimate goes wrong here. Points are kept only while above
    ``k * floor``, floor = median of the last ``tail_frac`` of the run.

    Returns nan (not a number) when fewer than 4 points survive -- a fit on three
    points is not a measurement and must not be silently reported as one.
    """
    import numpy as np
    y = np.asarray(curve, dtype=np.float64)
    if len(y) < 6:
        return float("nan"), 0, float("nan")
    floor = float(np.median(y[-max(3, int(len(y) * tail_frac)):]))
    if not np.isfinite(floor) or floor <= 0:
        return float("nan"), 0, float("nan")
    n = 0
    for v in (y > k * floor):        # leading contiguous run; once floored it stays
        if not v:
            break
        n += 1
    if n < 4:
        return float("nan"), int(n), float("nan")
    t = np.arange(n, dtype=np.float64)
    ly = np.log(y[:n])
    slope, icpt = np.polyfit(t, ly, 1)
    pred = slope * t + icpt
    denom = ((ly - ly.mean()) ** 2).sum()
    r2 = float(1 - ((ly - pred) ** 2).sum() / denom) if denom > 0 else float("nan")
    return float(np.exp(slope)), int(n), r2

import json
import random
import zlib

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MAX_R = 64
N_ITEMS = 64
N_PERM = 5000
PEAK = {"count4": 2, "add1": 4, "count16": 4}      # D68's per-task accuracy peak
TASKS = ("count4", "add1", "count16")


def items(task, n=N_ITEMS):
    out = []
    for s in range(n):
        # zlib.crc32, NOT hash(): Python salts str hashes PER PROCESS, so
        # hash(task) made the item set differ on every run (measured: 544, 92,
        # 779 across three interpreters). geometry-graded-readout and
        # geometry-discourse each drew a DIFFERENT set, which is why the same
        # nominal cell read 96% in one and 83% in the other.
        rng = random.Random(s * 7919 + zlib.crc32(task.encode()) % 997)
        if task == "add1":
            v = rng.randint(0, 8)
            out.append((f"What is {v} + 1?", str(v + 1)))
        else:
            k = 4 if task == "count4" else 16
            b = [rng.randint(0, 1) for _ in range(k)]
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(sum(b))))
    return out


def coda_head(model, h, freqs):
    """ln_f -> coda -> ln_f -> lm_head. TWO ln_f calls (1.7-1.9 logit error if one)."""
    import torch
    x = model.transformer.ln_f(h)
    bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
    for block in model.transformer.coda:
        bi -= 1
        x = block(x, freqs, bi, None, None)
    return model.lm_head(model.transformer.ln_f(x))


def orbit(model, tok, text, gold, seed, want_rank):
    """One orbit: per-unroll hidden state (last position) and optionally gold rank."""
    import numpy as np
    import torch
    ids_p = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    g = [tok(v, add_special_tokens=False).input_ids[0] for v in (gold, " " + gold)]
    ids = torch.cat([ids_p, torch.tensor(
        [tok(gold, add_special_tokens=False).input_ids], device=model.device,
        dtype=ids_p.dtype)], dim=1)
    n_p = ids_p.shape[1]
    freqs = model.freqs_cis[:, : ids.shape[1]]
    hs, rank = [], []
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()

    def hook(_m, _i, o):
        with torch.no_grad():
            hs.append(o.detach()[0, -1, :].float().cpu().numpy())
            if want_rank:
                row = torch.log_softmax(coda_head(model, o.detach(), freqs).float()[0],
                                        dim=-1)[n_p - 1]
                rank.append(min(int((row > row[v]).sum().item()) + 1 for v in g))

    h = mod.register_forward_hook(hook)
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            model(input_ids=ids, num_steps=MAX_R)
    finally:
        h.remove()
    return np.stack(hs), rank


def geometry(a, b):
    """Four per-item statistics from two orbits of the same prompt."""
    import numpy as np
    d = np.linalg.norm(a - b, axis=1)                       # two-orbit separation
    s = np.linalg.norm(np.diff(a, axis=0), axis=1)          # step norms
    rho_o, n_o, r2_o = fit_rho(d)
    rho_s, n_s, r2_s = fit_rho(s)
    below = np.where(s < 0.1 * s.max())[0]
    settle = int(below[0]) if len(below) else len(s)
    v = np.diff(a, axis=0)
    nv = np.linalg.norm(v, axis=1) + 1e-12
    cos = float(np.mean(np.sum(v[1:] * v[:-1], axis=1) / (nv[1:] * nv[:-1])))
    return ({"rho_orbit": rho_o, "rho_step": rho_s, "settle": float(settle), "cos_step": cos,
             "r2_orbit": r2_o, "r2_step": r2_s, "d0": float(d[0]), "d_end": float(d[-1])},
            d.tolist(), s.tolist())


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True).to(torch.float32).to("cuda").eval()

    METRICS = ("rho_orbit", "rho_step", "settle", "cos_step")
    out = {"max_r": MAX_R, "n_items": N_ITEMS, "peak": PEAK, "tasks": {}}
    rng = np.random.default_rng(0)
    print(f"\n{'task':>9} {'n_ok':>5} {'n_bad':>6} " +
          " ".join(f"{m:>22}" for m in METRICS))
    for task in TASKS:
        recs = []
        for body, gold in items(task):
            text = tok.apply_chat_template([{"role": "user", "content": body}],
                                           tokenize=False, add_generation_prompt=True)
            a, rank = orbit(model, tok, text, gold, 1000, True)
            b, _ = orbit(model, tok, text, gold, 2000, False)
            g, dcur, scur = geometry(a, b)
            g["correct"] = bool(rank[PEAK[task] - 1] == 1)
            g["rank"] = rank
            g["sep_curve"] = dcur
            g["step_curve"] = scur
            recs.append(g)
        out["tasks"][task] = recs
        ok = np.array([r["correct"] for r in recs])
        cells = []
        for m in METRICS:
            v = np.array([r[m] for r in recs], float)
            fin = np.isfinite(v)
            if ok[fin].sum() < 3 or (~ok[fin]).sum() < 3:
                cells.append("      (too few)      ")
                continue
            obs = abs(np.median(v[fin & ok]) - np.median(v[fin & ~ok]))
            lab = ok[fin].copy()
            null = [abs(np.median(v[fin][p := rng.permutation(lab)]) -
                        np.median(v[fin][~p])) for _ in range(N_PERM)]
            p = float((np.sum(np.array(null) >= obs) + 1) / (N_PERM + 1))
            cells.append(f"d={obs:+.4f} p_perm={p:.3f}")
        print(f"{task:>9} {int(ok.sum()):>5} {int((~ok).sum()):>6} " +
              " ".join(f"{c:>22}" for c in cells), flush=True)
        with open("geomcorrect.json", "w") as f:
            json.dump(out, f)

    print("\n=== VERDICT (Bonferroni alpha = 0.05/12 = 0.0042) ===")
    print("  see the p_perm column; the null is a LABEL PERMUTATION, so a metric")
    print("  only counts if the real split beats its own shuffled labels.")
    with open("geomcorrect.json", "w") as f:
        json.dump(out, f)


main()
