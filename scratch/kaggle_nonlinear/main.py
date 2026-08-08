"""A content probe whose target is NOT linearly present in the input.

WHY D70 FORCES THIS. Every content result in this project (D40, D41, D48, D53, and
now D70) probes a target that is a LINEAR FUNCTION OF THE INPUT TOKEN BAG -- a
count, a digit, a sum. Huginn re-injects the prompt embeddings at EVERY unroll
through the adapter on [h, e], so that bag is present in every state BY
ARCHITECTURE. A random-weight model preserves it, which is why D70 measured
untrained R2 = 1.0000 on count4 and count16 and why the trained-minus-untrained
gap was pinned into [-0.129, +0.039] with no headroom to move.

So those probes answer "is the INPUT linearly recoverable", not "did the model
COMPUTE anything". Both arms pass trivially and the comparison is near-vacuous.

THE FIX IS THE TARGET, NOT THE PROBE. Keep the probe fixed -- the same ridge
cv_r2 as D41/D48/D53 -- and vary how linearly available the target is from the
bag, all from ONE forward per item since every target is derived from the same
bit-string:

    count      sum(b)                    LINEAR in the bag  -> positive control
    parity     sum(b) % 2                NOT linear in the bag; the canonical case
    last       b[-1]                     needs position, not just the bag
    max_run    longest run of equal bits ORDER-dependent, nonlinear
    alt        count of adjacent flips   ORDER-dependent, pairwise

Parity is the sharp one and this project already carries the theory: Grazzi et al.
(2411.12537) -- finite-precision LRNNs with positive-eigenvalue state transitions
cannot solve parity; Merrill et al. (2404.08819) -- "the state in an SSM is an
illusion".

PRE-REGISTERED PREDICTIONS (CLAUDE.md section 1)
  P1  CONTROL. `count` reproduces D70: BOTH arms R2 > 0.9. If not, the pipeline
      changed and nothing else here may be read.
  P2  THE MECHANISM TEST. Untrained R2 on `parity` collapses toward 0 while its
      `count` stays high. That is what "the target rides in the re-injected input"
      predicts, and it is what gives this probe the headroom D70's lacked.
  P3  THE PAYOFF. If trained `parity` R2 exceeds untrained by a clear margin, that
      is the FIRST genuine content difference in this project -- a quantity the
      weights carry that random weights do not.
  P4  THE OTHER OUTCOME, equally informative. If BOTH arms fail parity (~0), then
      parity is simply absent from the final state in either model, and the
      content question must move to a different target or a different position --
      not to a nonlinear probe, which would only re-open D68's 6.7 worry.
  P5  If UNTRAINED decodes parity well, the re-injection story in D70(4) is wrong
      and D70 must be amended.

Gated on `has_dynamic_range`: if every target saturates in both arms, this repeats
D70's ceiling and the verdict is withheld rather than reported.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run load_arm free_arm cv_r2 preflight dynamic_range


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


def has_dynamic_range(values, ceiling=1.0, min_headroom=0.2, min_signal=0.05,
                      name="measure"):
    """Can a MODERATED variable move, or is it pinned against its own bound?

    A moderation test needs the moderated quantity to vary. For a bounded measure
    such as R^2 the question is not "are the values large" but "is there HEADROOM
    for a between-arm difference to appear". If every cell sits just under the
    ceiling, the gap is compressed into a sliver -- and a rank correlation over
    slivers still returns a confident-looking rho.

    Measured instance: `geometry-cap-graded` content R2 was 0.9691/0.9964/0.9945/
    0.8709 trained and 0.9301/0.9966/1.0000/1.0000 untrained. Headroom from the
    ceiling was only 1.0 - 0.8709 = 0.129, so the trained-minus-untrained gap could
    span at most [-0.129, +0.039]; Spearman over those four numbers printed +0.95
    CONFIRMED. That is D63's "a ratio of noise is not a confirmation" in a new
    costume, and it is why this guard exists.

    An earlier draft tested `min(values) > 0.95` and did NOT catch that case,
    because one cell sat at 0.8709. Headroom is the right quantity, not level.

    Returns (ok, detail).
    """
    v = [x for x in values if x == x]
    if not v:
        return False, f"{name}: no finite values"
    head = ceiling - min(v)
    if head < min_headroom:
        return False, (f"{name} is at CEILING: headroom {head:.4f} < {min_headroom} "
                       f"(min {min(v):.4f} against ceiling {ceiling}), so between-arm "
                       f"differences are bounded into a sliver")
    if max(v) < min_signal:
        return False, f"{name} is at FLOOR: max {max(v):.4f} < {min_signal}"
    return True, f"{name} spans [{min(v):.4f}, {max(v):.4f}], headroom {head:.4f}"

import json
import random
import zlib

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MAX_R = 32
N_ITEMS = 96          # more items: parity is a 2-class target and needs the n
M_BITS = 16
TARGETS = ("count", "parity", "last", "max_run", "alt")


def items(n=N_ITEMS):
    """One bit-string per item; EVERY target is derived from the same string, so
    all five share one forward and one state. crc32, not hash() (D69)."""
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + zlib.crc32(b"nonlin") % 997)
        b = [rng.randint(0, 1) for _ in range(M_BITS)]
        runs, cur = 1, 1
        for i in range(1, M_BITS):
            cur = cur + 1 if b[i] == b[i - 1] else 1
            runs = max(runs, cur)
        tgt = {"count": float(sum(b)), "parity": float(sum(b) % 2), "last": float(b[-1]),
               "max_run": float(runs),
               "alt": float(sum(b[i] != b[i - 1] for i in range(1, M_BITS)))}
        out.append(("Sequence: " + " ".join(map(str, b)) + ". How many ones?", tgt))
    return out


def final_state(model, tok, text):
    """State at the last prompt position after MAX_R unrolls -- the position whose
    distribution produces the answer, and the one prior project geometry used."""
    import torch
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    grab = {}
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()
    h = mod.register_forward_hook(lambda _m, _i, o: grab.__setitem__("h", o.detach()))
    try:
        torch.manual_seed(1234)
        with torch.no_grad():
            model(input_ids=ids, num_steps=MAX_R)
    finally:
        h.remove()
    return grab["h"][0, -1, :].float().cpu().numpy()


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    data = items()
    out = {"max_r": MAX_R, "n_items": N_ITEMS, "m_bits": M_BITS, "arms": {}}

    for arm, spec in (("trained", MODEL_ID), ("untrained", None)):
        print(f"\n=== {arm} ===", flush=True)
        model = load_arm(spec, cfg, REVISION if spec else 0)
        X = np.stack([final_state(model, tok, tok.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True))
            for body, _ in data])
        out["arms"][arm] = {}
        print(f"  {'target':>9} {'R2':>9} {'null':>8}   linear-in-bag?")
        for tname in TARGETS:
            y = np.array([t[tname] for _, t in data], float)
            r2, null = cv_r2(X, y, n_null=20)
            out["arms"][arm][tname] = {"r2": float(r2), "null_mean": float(np.mean(null)),
                                       "y": y.tolist()}
            tag = {"count": "YES (control)", "parity": "NO", "last": "position",
                   "max_run": "NO (order)", "alt": "NO (order)"}[tname]
            print(f"  {tname:>9} {r2:>9.4f} {np.mean(null):>8.3f}   {tag}", flush=True)
            with open("nonlin.json", "w") as f:
                json.dump(out, f)
        free_arm(model, spec)

    print("\n=== VERDICT ===")
    T, U = out["arms"]["trained"], out["arms"]["untrained"]
    print(f"  {'target':>9} {'trained':>9} {'untrained':>10} {'gap (T-U)':>11}")
    for tname in TARGETS:
        print(f"  {tname:>9} {T[tname]['r2']:>9.4f} {U[tname]['r2']:>10.4f} "
              f"{T[tname]['r2'] - U[tname]['r2']:>+11.4f}")
    allr2 = [v["r2"] for a in out["arms"].values() for v in a.values()]
    rng_ok, rng_why = has_dynamic_range(allr2, name="content R2 across targets")
    ctrl = T["count"]["r2"] > 0.9 and U["count"]["r2"] > 0.9
    nulls_ok = all(v["null_mean"] <= 0.05 for a in out["arms"].values() for v in a.values())
    gap = T["parity"]["r2"] - U["parity"]["r2"]
    gated_verdict(
        f"the weights carry parity that random weights do not (gap {gap:+.4f})",
        gap > 0.10,
        [("count control reproduces D70", ctrl,
          f"trained {T['count']['r2']:.4f} / untrained {U['count']['r2']:.4f}, both need >0.9"),
         ("cv_r2 nulls honest", nulls_ok, "a positive null means the probe interpolates"),
         ("targets have dynamic range", rng_ok, rng_why)])
    print(f"    untrained parity R2 = {U['parity']['r2']:.4f} "
          f"(P2 wants ~0; if high, D70(4)'s re-injection story is WRONG)")
    out["summary"] = {"parity_gap": gap, "range_ok": rng_ok, "control_ok": ctrl}
    with open("nonlin.json", "w") as f:
        json.dump(out, f)


main()
