"""Positive control for the readout D68 and D69 rest on. It never had one.

THE GAP, found by self-review. Every number in D68 and D69 comes from ranking the
gold token in `log_softmax(coda_head(state_r))`, where `coda_head` is MY
re-implementation of the model's tail: ln_f -> coda -> ln_f -> lm_head, applied to
a state hooked out of `core_block[-1]` at each unroll.

The controls I did have catch GROSS breakage only. The untrained arm sitting at
chance while the trained arm sits at rank 3 could not happen with a badly broken
head. But a SUBTLE bias would survive that check untouched -- and the specific
subtle bias is documented in this repo: using ONE ln_f instead of two feeds coda a
state it was never built to accept, measured at 1.7-1.9 max abs logit difference
(traj_geom/extraction/hook.py, 2026-07-23). A systematic logit shift of that size
would move ranks without ever looking wrong.

I built `preflight` guards for verdict logic and left the READOUT ITSELF
uncontrolled. This closes that.

THE IDENTITY. At the FINAL unroll the model's own forward applies exactly
ln_f -> coda -> ln_f -> lm_head to exactly that state. So

    coda_head(state at unroll R)  ==  model(input_ids, num_steps=R).logits

must hold to floating-point tolerance. It is exact, it costs one forward, and it
was never run.

PRE-REGISTERED PREDICTIONS (CLAUDE.md section 1)
  P1  POSITIVE CONTROL. The two-ln_f decode matches the model's own logits with
      max abs difference < 1e-2 in fp32, and gold rank agrees for 100% of items.
      If it does NOT, D68 and D69 rest on a biased readout and both rows must be
      amended -- which is the point of running this late rather than not at all.
  P2  NEGATIVE CONTROL, so the test is known to have teeth. The one-ln_f variant
      must DISAGREE, by order 1 or more in max abs logit difference. A test that
      passes both variants proves nothing.
  P3  RANK IS THE QUANTITY THAT MATTERS, so it is compared directly: the rank of
      the gold token computed from my decode must equal the rank computed from
      the model's own logits, item by item, not merely on average.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run preflight


def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


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
R_LIST = (4, 16, 32)
N_ITEMS = 12


def items(n=N_ITEMS):
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + zlib.crc32(b"headcheck") % 997)
        v = rng.randint(0, 9)
        out.append((f"Repeat this number exactly.\nNumber: {v}", str(v)))
    return out


def tail(model, h, freqs, two_ln_f=True):
    """The model's tail. `two_ln_f=False` is the KNOWN-WRONG variant, kept as the
    negative control -- a check that passes both variants would prove nothing."""
    import torch
    x = model.transformer.ln_f(h) if two_ln_f else h
    bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
    for block in model.transformer.coda:
        bi -= 1
        x = block(x, freqs, bi, None, None)
    return model.lm_head(model.transformer.ln_f(x))


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True).to(torch.float32).to("cuda").eval()

    out = {"r_list": list(R_LIST), "cells": []}
    print(f"\n{'r':>4} {'max|Δ| two_ln_f':>17} {'max|Δ| one_ln_f':>17} "
          f"{'rank agree':>11} {'gold rank (mine/model)':>24}")
    for R in R_LIST:
        d_ok, d_bad, agree, examples = [], [], [], []
        for body, gold in items():
            text = tok.apply_chat_template([{"role": "user", "content": body}],
                                           tokenize=False, add_generation_prompt=True)
            ids_p = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            g0 = tok(gold, add_special_tokens=False).input_ids[0]
            freqs = model.freqs_cis[:, : ids_p.shape[1]]
            grab = {}

            def hook(_m, _i, o):
                grab["h"] = o.detach()                    # overwritten each unroll -> final

            h = mod_hook(model, hook)
            try:
                with torch.no_grad():
                    res = model(input_ids=ids_p, num_steps=R)
            finally:
                h.remove()

            ref = res.logits[0, -1, :].float()
            mine = tail(model, grab["h"], freqs, True)[0, -1, :].float()
            wrong = tail(model, grab["h"], freqs, False)[0, -1, :].float()
            d_ok.append(float((mine - ref).abs().max()))
            d_bad.append(float((wrong - ref).abs().max()))
            r_mine = int((mine > mine[g0]).sum()) + 1
            r_ref = int((ref > ref[g0]).sum()) + 1
            agree.append(r_mine == r_ref)
            examples.append((r_mine, r_ref))
        cell = {"r": R, "max_delta_two_lnf": max(d_ok), "max_delta_one_lnf": max(d_bad),
                "rank_agreement": float(np.mean(agree)), "examples": examples[:4]}
        out["cells"].append(cell)
        print(f"{R:>4} {max(d_ok):>17.6f} {max(d_bad):>17.6f} "
              f"{np.mean(agree):>10.0%} {str(examples[:3]):>24}", flush=True)
        with open("headcheck.json", "w") as f:
            json.dump(out, f)

    print("\n=== VERDICT ===")
    worst_ok = max(c["max_delta_two_lnf"] for c in out["cells"])
    worst_bad = min(c["max_delta_one_lnf"] for c in out["cells"])
    worst_agree = min(c["rank_agreement"] for c in out["cells"])
    gated_verdict(
        "the per-unroll readout reproduces the model's own head",
        worst_ok < 1e-2 and worst_agree == 1.0,
        [("negative control discriminates", worst_bad > 1.0,
          f"one-ln_f max|delta| = {worst_bad:.4f}; a test passing both variants proves nothing"),
         ("rank agreement is exact", worst_agree == 1.0,
          f"gold-rank agreement {worst_agree:.0%} -- rank is the quantity D68/D69 report")])
    print(f"    two-ln_f  max|delta| = {worst_ok:.6f}  (P1 wants < 1e-2)")
    print(f"    one-ln_f  max|delta| = {worst_bad:.6f}  (P2 wants > 1)")
    with open("headcheck.json", "w") as f:
        json.dump(out, f)


def mod_hook(model, fn):
    m = model.transformer.core_block[-1]
    m._forward_hooks.clear()
    return m.register_forward_hook(fn)


main()
