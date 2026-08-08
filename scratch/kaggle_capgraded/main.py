"""UNDERSTANDING 6.1, with the capability axis measured by an instrument that works.

THE QUESTION. Is "training changes dynamics, not content" an artefact of only ever
testing content where the model cannot do the task? Every content-null in this
project (D40, D41, D48, D53) was measured on counting-family tasks the trained
model scores ~0% on, and if there is nothing to encode then finding random weights
encode it equally well is close to tautological.

WHY THIS KERNEL EXISTS ALONGSIDE `geometry-cap-content`. That kernel was built for
this question and measures its capability axis by GREEDY GENERATION plus exact
match. D68 showed that instrument is blind in precisely the regime that matters:
tasks reading 0.0% by exact match hold the gold token at median rank 22 against a
chance of 32768. Run as written its moderator would be ~0 across the whole ladder
for MEASUREMENT reasons, and it would "confirm" the headline by construction.
I have NOT edited that kernel -- it is someone else's experiment and CLAUDE.md
section 3 says to say what is wrong with it rather than rewrite it. This is a
separate kernel with the capability axis replaced.

THE DESIGN: CAPABILITY AS A CONTINUOUS MODERATOR.
    ladder      echo_digit -> add1 -> count4 -> count16, certain success to certain
                failure, every probe target a plain integer on a comparable scale
    capability  log10 rank of the gold token over the full 65536 vocabulary, at the
                unroll where it is best -- graded, non-zero even at 0% accuracy
    content     held-out ridge R^2 decoding the task integer from the final state,
                with a permutation null; the SAME probe as D41/D48/D53, so the
                comparison is like-for-like
    arms        trained and untrained (Huginn at step 0, its own init scheme)

    The test is whether the trained-minus-untrained CONTENT gap tracks the
    CAPABILITY, across four rungs.

PRE-REGISTERED PREDICTIONS, written before the run (CLAUDE.md section 1)
  P1  CONTROL. cv_r2's permutation null must land at or below 0 in every cell. An
      honest held-out null is negative; a positive one means d >> n has turned the
      probe into an interpolator and NOTHING else in the run may be read.
  P2  CAPABILITY REPLICATES D68 -- log10 rank ordered echo_digit < add1 ~ count4 <
      count16, and the untrained arm at chance (log10 ~ 4.5) everywhere. If the
      untrained arm is NOT at chance, the readout is broken, not informative.
  P3  THE TEST. If 6.1's confound is real, the trained-minus-untrained content gap
      is POSITIVE where capability is high and ~0 where capability is zero, giving
      a positive rank correlation between the two across the four rungs.
  P4  THE OTHER OUTCOME, equally publishable. If the content gap is ~0 at EVERY
      rung including the ones the model demonstrably does, then "content is
      architectural" survives its strongest challenge and D40/D41/D48/D53 stand as
      general rather than scoped.

  With four rungs a rank correlation cannot reach significance on its own (Spearman
  needs |rho| = 1.0 at N=4), so P3 is read as DIRECTION plus per-rung effect sizes,
  not as a p-value. That limit is stated here rather than discovered afterwards --
  D56 recorded exactly this trap, quoting rho with N=4 as if it were a test.

Per B4.14 the per-item ranks, states-derived probe inputs and null distributions
are persisted, so the analysis can be redone locally without another GPU run.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run load_arm free_arm cv_r2 preflight


def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


def load_arm(spec, cfg, revision=None):
    """Load one weight-set. `spec` is None for a fresh random init, else a repo id.

    Backfills config attributes ABSENT from an older checkpoint from the final
    model's config -- intermediate Huginn checkpoints predate fields the current
    modeling code reads (`test_time_noise`), and without this they raise
    AttributeError. Only missing keys are copied, and every backfill is logged.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    if spec is None:
        torch.manual_seed(revision if isinstance(revision, int) else 0)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    else:
        ck = AutoConfig.from_pretrained(spec, revision=revision, trust_remote_code=True)
        added = [k for k, v in vars(cfg).items()
                 if not hasattr(ck, k) and not k.startswith("_")]
        for k in added:
            setattr(ck, k, getattr(cfg, k))
        if added:
            print(f"  backfilled {len(added)} config attrs: {sorted(added)}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec, revision=revision, config=ck, trust_remote_code=True,
            low_cpu_mem_usage=True)
    return model.to(torch.float32).to("cuda").eval()


def free_arm(model, repo_id=None):
    """Release a weight-set and report what was actually reclaimed.

    Printing free memory is the point: a silent cleanup is how eight checkpoints
    were lost to OOM with 13.46 GiB still held (see `geometry-rho-direct`). Also
    purges the HF cache for `repo_id`, since ten 7GB checkpoints exhaust the disk.
    """
    import gc
    import os
    import shutil

    import torch
    try:
        model = model.to("cpu") if model is not None else None
    except Exception:                                          # noqa: BLE001, S110
        pass
    del model
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"  after cleanup: {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB",
          flush=True)
    if repo_id:
        d = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                         "models--" + repo_id.replace("/", "--"))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"  purged {d}", flush=True)


def cv_r2(x, y, groups=None, alpha=1e3, n_splits=5, n_null=0):
    """Held-out R^2 with an optional permutation null -> (r2, null_list).

    `groups` switches to GroupKFold, which is REQUIRED whenever rows come from the
    same prompt: a plain KFold leaks between positions of one sequence and inflates
    the score. The permutation null is the check that d >> n has not turned the
    probe into an interpolator -- an honest held-out null lands BELOW zero.
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    cv = (GroupKFold(n_splits=n_splits) if groups is not None
          else KFold(n_splits=n_splits, shuffle=True, random_state=0))

    def score(target):
        p = cross_val_predict(mdl, x, target, cv=cv, groups=groups)
        return float(1 - ((target - p) ** 2).sum() / ((target - target.mean()) ** 2).sum())

    r2 = score(y)
    rng = np.random.default_rng(0)
    null = [score(rng.permutation(y)) for _ in range(n_null)]
    return r2, null


def attainable(alpha, n_perm):
    """Can a permutation test with `n_perm` draws ever reach `alpha`?

    The smallest p a permutation test can report is 1/(n_perm+1). If that floor
    sits above the significance threshold, REJECTION IS ARITHMETICALLY
    IMPOSSIBLE and the run returns "not significant" for every cell no matter
    what the data say -- a guaranteed null that reads like a scientific result.

    Measured instance: `geometry-correctness` was drafted with n_perm=200 against
    a Bonferroni alpha of 0.05/12 = 0.00417. Floor = 1/201 = 0.00498 > alpha, and
    synthetic power was 0.00 even for a 2 sd shift. Same class as the winding
    null's p=0.024 floor at n=40.

    Returns (ok, floor).
    """
    floor = 1.0 / (n_perm + 1)
    return floor < alpha, floor


def degenerate(decoded, min_distinct=2):
    """Is an argmax population degenerate rather than informative?

    Two failure shapes, both seen on real runs:
      * COLLAPSE -- every item predicts the same token, so the measurement
        carries no per-item information.
      * UNPRINTABLE -- the argmax decodes to a partial UTF-8 byte fragment
        (U+FFFD after decode), which means the distribution is not on words at
        all. `geometry-discourse`'s prefill arm did BOTH: token ids 6704/7909/
        12894 ('ä¸') for 24/24 items on every task, and the kernel still printed
        a confident verdict from that arm.

    Returns (is_degenerate, reason).
    """
    uniq = set(decoded)
    bad = sum("�" in d for d in decoded)
    if bad > len(decoded) // 2:
        return True, f"{bad}/{len(decoded)} argmax tokens are unprintable byte fragments"
    if len(uniq) < min_distinct:
        return True, f"argmax collapsed to {len(uniq)} distinct token(s): {sorted(uniq)[:3]}"
    return False, ""


def gated_verdict(claim, passed, gates):
    """Print a conclusion ONLY if every precondition holds; else say why not.

    D62: a capability verdict printed over empty strings because the analysis
    excluded the control that would have caught it. `geometry-discourse` repeated
    it -- P3/P4 keyed on one arm and printed the OPPOSITE of the right answer
    without ever checking that arm's output was sane.

    `gates` is a list of (name, ok, detail). A verdict computed from the same
    variables as the run will agree with the run's mistakes, so the gates must
    test the INSTRUMENT, not the hypothesis.

    Returns the verdict string, and prints it.
    """
    failed = [(n, d) for n, ok, d in gates if not ok]
    if failed:
        msg = (f"  VERDICT WITHHELD -- {claim}\n"
               + "\n".join(f"    gate FAILED: {n} -- {d}" for n, d in failed)
               + "\n    the instrument did not qualify; this arm may not be read.")
    else:
        msg = f"  {claim}: {'CONFIRMED' if passed else 'REFUTED'}"
    print(msg, flush=True)
    return msg

import json
import random
import zlib

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MAX_R = 32                 # D68: capability peaks by r=4; 32 is the project default
N_ITEMS = 48
TASKS = ("echo_digit", "add1", "count4", "count16")
CHANCE_LOG10 = 4.515       # log10(65536/2)


def items(task, n=N_ITEMS):
    """zlib.crc32, not hash(): Python salts str hashes per process (D69)."""
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + zlib.crc32(task.encode()) % 997)
        if task == "echo_digit":
            v = rng.randint(0, 9)
            out.append((f"Repeat this number exactly.\nNumber: {v}", str(v), float(v)))
        elif task == "add1":
            v = rng.randint(0, 8)
            out.append((f"What is {v} + 1?", str(v + 1), float(v + 1)))
        else:
            k = 4 if task == "count4" else 16
            b = [rng.randint(0, 1) for _ in range(k)]
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(sum(b)), float(sum(b))))
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


def probe_and_rank(model, tok, text, gold):
    """Final-unroll state (for the content probe) and best-over-depth gold rank."""
    import numpy as np
    import torch
    ids_p = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    g = [tok(v, add_special_tokens=False).input_ids[0] for v in (gold, " " + gold)]
    ids = torch.cat([ids_p, torch.tensor(
        [tok(gold, add_special_tokens=False).input_ids], device=model.device,
        dtype=ids_p.dtype)], dim=1)
    n_p = ids_p.shape[1]
    freqs = model.freqs_cis[:, : ids.shape[1]]
    ranks, last = [], {}
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()

    def hook(_m, _i, o):
        with torch.no_grad():
            row = torch.log_softmax(coda_head(model, o.detach(), freqs).float()[0],
                                    dim=-1)[n_p - 1]
            ranks.append(min(int((row > row[v]).sum().item()) + 1 for v in g))
            last["h"] = o.detach()[0, n_p - 1, :].float().cpu().numpy()

    h = mod.register_forward_hook(hook)
    try:
        torch.manual_seed(1234)
        with torch.no_grad():
            model(input_ids=ids, num_steps=MAX_R)
    finally:
        h.remove()
    return last["h"], ranks


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    out = {"max_r": MAX_R, "n_items": N_ITEMS, "chance_log10": CHANCE_LOG10, "arms": {}}

    for arm, spec in (("trained", MODEL_ID), ("untrained", None)):
        print(f"\n=== {arm} ===", flush=True)
        model = load_arm(spec, cfg, REVISION if spec else 0)
        out["arms"][arm] = {}
        print(f"  {'task':>11} {'log10 rank':>11} {'acc@1':>7} {'content R2':>11} {'null':>8}")
        for task in TASKS:
            X, Y, R = [], [], []
            for body, gold, target in items(task):
                text = tok.apply_chat_template([{"role": "user", "content": body}],
                                               tokenize=False, add_generation_prompt=True)
                h, ranks = probe_and_rank(model, tok, text, gold)
                X.append(h); Y.append(target); R.append(min(ranks))
            X = np.stack(X); Y = np.array(Y, float); R = np.array(R, float)
            r2, null = cv_r2(X, Y, n_null=20)
            rec = {"log10_rank": float(np.median(np.log10(R))), "acc1": float((R == 1).mean()),
                   "r2": float(r2), "null_mean": float(np.mean(null)),
                   "ranks": R.tolist(), "targets": Y.tolist(), "null": list(map(float, null))}
            out["arms"][arm][task] = rec
            print(f"  {task:>11} {rec['log10_rank']:>11.2f} {rec['acc1']:>7.0%} "
                  f"{r2:>11.4f} {rec['null_mean']:>8.3f}", flush=True)
            with open("capgraded.json", "w") as f:
                json.dump(out, f)
        free_arm(model, spec)

    print("\n=== 6.1 MODERATION TEST ===")
    T, U = out["arms"]["trained"], out["arms"]["untrained"]
    cap = [-T[t]["log10_rank"] for t in TASKS]          # higher = more capable
    gap = [T[t]["r2"] - U[t]["r2"] for t in TASKS]
    print(f"  {'task':>11} {'capability(-log10 rank)':>24} {'content gap (T-U)':>19}")
    for t, c, g in zip(TASKS, cap, gap):
        print(f"  {t:>11} {c:>24.2f} {g:>19.4f}")
    from scipy.stats import spearmanr
    rho = float(spearmanr(cap, gap)[0])
    nulls_ok = all(v["null_mean"] <= 0.05 for a in out["arms"].values() for v in a.values())
    untr_chance = all(abs(U[t]["log10_rank"] - CHANCE_LOG10) < 0.9 for t in TASKS)
    gated_verdict(
        f"the content gap tracks capability (Spearman rho = {rho:+.2f} over {len(TASKS)} rungs)",
        rho > 0,
        [("cv_r2 null is honest (<=0.05)", nulls_ok, "a positive null means the probe interpolates"),
         ("untrained readout at chance", untr_chance,
          f"untrained log10 ranks {[round(U[t]['log10_rank'],2) for t in TASKS]} vs chance {CHANCE_LOG10}")])
    print("  N=4 rungs: read as DIRECTION plus per-rung effect sizes, never as a p-value.")
    out["moderation"] = {"capability": cap, "content_gap": gap, "spearman": rho}
    with open("capgraded.json", "w") as f:
        json.dump(out, f)


main()
