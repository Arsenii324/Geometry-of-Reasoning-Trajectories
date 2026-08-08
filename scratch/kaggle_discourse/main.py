"""Is D68's depth collapse a DISCOURSE choice or a loss of COMPUTATION?

D68 established that the trained model holds gold far below chance on tasks that
exact-match scores 0%, and that acc@1 is non-monotone in depth: `echo_digit`
reaches 96% rank-1 at r=4 and 0% by r=8, while top-1 moves from the answer digit
to a prose opener ('The' 15/24, 'Number' 8/24 at r=64; 'The' 23/24 on rot13).

I INTERPRETED that as the model deciding to answer in a sentence rather than
failing to compute. That interpretation is not yet established, and it is my own,
so this kernel is built to REFUTE it. Two readings remain live:

  (A) DISCOURSE. The computation survives depth; only the token the model wants to
      emit FIRST changes, from the answer to a sentence opener. Gold falls to
      rank 2-3 because that is where it belongs after "The".
  (B) DEGRADATION. Deep unrolling genuinely destroys the answer, and the low rank
      at r=64 is a residue rather than a retrievable result.

THE DESIGN THAT SEPARATES THEM. Three prompt formats, same items, same depths:

    bare         D68's prompt, unchanged -- the replication control
    constrained  + "Reply with only the answer, nothing else."  (D60's instruction)
    prefill      the assistant turn is PRE-FILLED with "The answer is " so the
                 prose slot is ALREADY OCCUPIED and the next token can only be the
                 answer

`prefill` is the decisive arm and is why this kernel exists. Under (A) the
computation is intact at r=64 and merely mis-slotted, so filling the slot must
recover it. Under (B) nothing can recover it, because there is nothing left.

PRE-REGISTERED PREDICTIONS, written before the run (CLAUDE.md section 1)
  P1  REPLICATION CONTROL. bare reproduces D68: echo_digit acc@1 >= 80% at r=4,
      and at r=64 the top-1 is a prose opener for at least half the items. If this
      fails, nothing else in this kernel may be read -- the harness changed.
  P2  constrained IMPROVES median rank at r=64 over bare by >= 5x. D60 measured
      that instruction as worth 85 accuracy points, so it should bite here.
  P3  THE DECIDER. prefill gives HIGH acc@1 at r=64 (>= 50% on echo_digit).
      Confirms (A): depth kept the answer and changed only where it goes.
  P4  If instead prefill acc@1 at r=64 is LOW (< 20%) while bare acc@1 at r=4 was
      high, (B) holds, D68's interpretation is WRONG, and its row must be amended
      to say the computation degrades with depth.

P3 and P4 are mutually exclusive and jointly exhaustive at the stated thresholds,
so this run cannot come back uninformative.

Per B4.14 the full per-unroll curves are persisted, not fitted summaries.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run


def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)

import json
import random
import string

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MAX_R = 64
N_ITEMS = 24
CHANCE = 32768
FORMATS = ("bare", "constrained", "prefill")
PREFILL = "The answer is "
TASKS = ("echo_digit", "add1", "count4", "count16", "rot13_word")


def items(task, n=N_ITEMS):
    """Identical construction to geometry-graded-readout, so D68 is replicable."""
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + hash(task) % 997)
        if task == "echo_digit":
            v = rng.randint(0, 9)
            out.append((f"Repeat this number exactly.\nNumber: {v}", str(v)))
        elif task == "add1":
            v = rng.randint(0, 8)
            out.append((f"What is {v} + 1?", str(v + 1)))
        elif task == "count4":
            b = [rng.randint(0, 1) for _ in range(4)]
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(sum(b))))
        elif task == "count16":
            b = [rng.randint(0, 1) for _ in range(16)]
            out.append(("Count how many ones are in this sequence.\n"
                        f"Sequence: {' '.join(map(str, b))}", str(sum(b))))
        else:
            w = "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
            enc = "".join(chr((ord(c) - 97 + 13) % 26 + 97) for c in w)
            out.append((f"Decode this ROT13 text.\nText: {enc}", w))
    return out


def render(body, fmt, tok):
    if fmt == "constrained":
        body = body + "\nReply with only the answer, nothing else."
    text = tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True)
    return text + PREFILL if fmt == "prefill" else text


def coda_head(model, h_state, freqs_cis):
    """ln_f -> coda -> ln_f -> lm_head. TWO ln_f calls; skipping the first feeds
    coda a state it was never built to accept (1.7-1.9 logit error, measured)."""
    import torch
    x = model.transformer.ln_f(h_state)
    bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
    for block in model.transformer.coda:
        bi -= 1
        x = block(x, freqs_cis, bi, None, None)
    return model.lm_head(model.transformer.ln_f(x))


def curve(model, tok, text, gold):
    """Per-unroll (BEST-surface-form rank of gold, top-1) from ONE forward.

    SURFACE FORMS MATTER AND COULD HAVE DECIDED THIS RUN. After the `prefill`
    prefix "The answer is " the natural continuation is "1", while after `bare`'s
    "\n\n" it may be " 1". Scoring one fixed form would penalise whichever arm
    disagrees with it -- and since `prefill` is the decider between P3 and P4,
    a tokenisation artefact could have produced the verdict. Rank is therefore
    the MINIMUM over {gold, " " + gold}, which is what
    `traj_geom.eval_depth.answer_surface_forms` does for the same reason.
    """
    import torch
    variants = [v for v in (gold, " " + gold)
                if len(tok(v, add_special_tokens=False).input_ids) > 0]
    v_first = [tok(v, add_special_tokens=False).input_ids[0] for v in variants]
    ids_p = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    g_ids = tok(gold, add_special_tokens=False).input_ids
    ids = torch.cat([ids_p, torch.tensor([g_ids], device=model.device,
                                         dtype=ids_p.dtype)], dim=1)
    n_p = ids_p.shape[1]
    freqs = model.freqs_cis[:, : ids.shape[1]]
    rec = {"rank": [], "top1": [], "forms": variants}
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()

    def hook(_m, _i, o):
        with torch.no_grad():
            lp = torch.log_softmax(coda_head(model, o.detach(), freqs).float()[0], dim=-1)
            row = lp[n_p - 1]
            rec["rank"].append(min(int((row > row[v]).sum().item()) + 1 for v in v_first))
            rec["top1"].append(int(row.argmax().item()))

    h = mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids, num_steps=MAX_R)
    finally:
        h.remove()
    return rec


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True).to(torch.float32).to("cuda").eval()

    out = {"max_r": MAX_R, "n_items": N_ITEMS, "fmt": {}}
    for fmt in FORMATS:
        out["fmt"][fmt] = {}
        print(f"\n=== {fmt} ===", flush=True)
        print(f"  {'task':>11} {'acc@1 r4':>9} {'acc@1 r64':>10} {'med rank r4':>12} "
              f"{'med rank r64':>13}  top1@r64")
        for task in TASKS:
            rows = [curve(model, tok, render(b, fmt, tok), g) for b, g in items(task)]
            out["fmt"][fmt][task] = rows
            R = np.array([r["rank"] for r in rows])
            t1 = [tok.decode([r["top1"][-1]]) for r in rows]
            from collections import Counter
            top = Counter(t1).most_common(2)
            print(f"  {task:>11} {(R[:,3]==1).mean():>8.0%} {(R[:,-1]==1).mean():>9.0%} "
                  f"{np.median(R[:,3]):>12.0f} {np.median(R[:,-1]):>13.0f}  {top}", flush=True)
            with open("discourse.json", "w") as f:
                json.dump(out, f)

    print("\n=== VERDICT ===")
    import numpy as np
    def a1(fmt, task, r):
        R = np.array([x["rank"] for x in out["fmt"][fmt][task]])
        return float((R[:, r - 1] == 1).mean())
    def med(fmt, task, r):
        R = np.array([x["rank"] for x in out["fmt"][fmt][task]])
        return float(np.median(R[:, r - 1]))
    p1 = a1("bare", "echo_digit", 4) >= 0.80
    print(f"  P1 replication (bare echo_digit acc@1 r4 >= 80%): "
          f"{'OK' if p1 else 'FAILED -- read nothing else'}  ({a1('bare','echo_digit',4):.0%})")
    ratio = med("bare", "echo_digit", 64) / max(med("constrained", "echo_digit", 64), 1e-9)
    print(f"  P2 constrained improves median rank r64 >= 5x: {ratio:.2f}x")
    p3 = a1("prefill", "echo_digit", 64)
    print(f"  P3/P4 DECIDER -- prefill acc@1 at r64 = {p3:.0%}")
    if p3 >= 0.50:
        verdict = "(A) DISCOURSE: computation survives depth, D68 stands"
    elif p3 < 0.20:
        verdict = "(B) DEGRADATION: D68 interpretation WRONG, amend it"
    else:
        verdict = "INDETERMINATE between the pre-registered thresholds"
    print(f"     => {verdict}")
    with open("discourse.json", "w") as f:
        json.dump(out, f)


main()
