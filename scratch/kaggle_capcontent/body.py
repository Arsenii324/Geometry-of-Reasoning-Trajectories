"""Is "content is architectural" an artefact of only testing tasks the model fails?

THE BLIND SPOT (UNDERSTANDING.md 6.1)
    Every result behind this project's headline -- that training changes the
    model's DYNAMICS and not the CONTENT of its state -- was measured on
    counting-family tasks where the trained model scores ~0% (D56, D60, D61):

        D40  untrained reproduces every endpoint metric
        D41  untrained decodes the total 19x more precisely
        D48  untrained count representation is literally 1-dimensional (PR 1.0 vs 4.8)
        D53  untrained carries the register slightly better (0.7498 vs 0.7175)

    If the model cannot perform the task then there is NOTHING FOR TRAINING TO
    HAVE ENCODED, and discovering that random weights represent the input equally
    well is close to tautological -- both arms are carrying an unused input. The
    headline may be an artefact of never testing content where content exists.

THE DESIGN: CAPABILITY AS A MODERATOR
    Rather than argue about it, span capability and see whether the content gap
    tracks it. Four tasks chosen to run from certain success to certain failure:

        echo_num    repeat a 2-digit number   expected ~100%: copy is 100% (D60)
        add1        single-digit addition     expected high; a 3.5B LM does this
        count4      count ones in 4 bits      25% at n=4 (D56)
        count16     count ones in 16 bits     0% from n=8 onward (D56)

    EVERY PROBE TARGET IS AN INTEGER ON A COMPARABLE SCALE. An earlier draft used
    string `copy` with the word encoded base-26 as the latent; that encoding is
    arbitrary and essentially not linearly decodable, so its R2 would not have been
    comparable to the integer tasks and the moderation test would have been
    meaningless. `echo_num` keeps the high-capability cell while making its latent a
    plain integer. String `copy` is retained as the harness control only, scored for
    accuracy and never probed.

    For each task and each arm we measure TWO things on the SAME forward pass:
      CAPABILITY  greedy accuracy, chat-formatted
      CONTENT     cross-validated decodability of the task's latent variable from
                  the answer-position state (ridge, permutation null, grouped folds)

    PRE-REGISTERED PREDICTION, written before running:
      If 6.1 is a real confound, `content_gap = R2(trained) - R2(untrained)` is
      POSITIVE where capability is high and ~0 or negative where capability is 0 --
      i.e. corr(capability, content_gap) > 0. That would scope the headline to
      "training does not change content ON TASKS THE MODEL CANNOT DO", which is a
      much weaker and more honest claim.
      If instead the content gap is ~0 or negative EVERYWHERE, including on `copy`
      where the trained model is at 100% and the untrained at 0%, then the headline
      SURVIVES its most serious challenge and is considerably strengthened.

    Either outcome is decisive, which is why this is worth a kernel.

CONTROLS
    * `copy` doubles as the harness control: if trained accuracy on it is not high,
      the generator is broken and nothing may be read (the D62 lesson).
    * The permutation null must be NEGATIVE for every decodability figure, since
      d=5280 >> n (D41(6)).
    * Both arms see identical prompts in the same process.
"""
# @needs: run load_arm free_arm batched_generate assert_generation_works cv_r2 cv_r2_nonlinear

import json
import random

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 32
N_ITEMS = 80          # enough for a ridge probe with a permutation null
MAX_NEW = 8

TASKS = ("echo_num", "add1", "count4", "count16")
CONTROL = "copy"


def item(task, seed):
    """(question_body, gold_string, latent_value_for_the_probe)."""
    rng = random.Random(seed * 7919 + hash(task) % 997)
    if task == "copy":
        w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        return (f"Repeat this word exactly.\nWord: {w}", w, 0.0)   # control, not probed
    if task == "echo_num":
        v = rng.randint(10, 99)
        return (f"Repeat this number exactly.\nNumber: {v}", str(v), float(v))
    if task == "add1":
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        return (f"What is {a} + {b}?", str(a + b), float(a + b))
    n = 4 if task == "count4" else 16
    bits = [rng.randint(0, 1) for _ in range(n)]
    return (f"Count how many ones are in this sequence.\nSequence: "
            f"{' '.join(map(str, bits))}", str(sum(bits)), float(sum(bits)))


def render(body, tok):
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True)


def score(task, pred, gold):
    p = pred.lower().strip()
    if task == "copy":
        return float(gold in "".join(c for c in p if c.isalpha()))
    nums, cur = [], ""
    for c in p:
        if c.isdigit():
            cur += c
        else:
            if cur:
                nums.append(cur)
            cur = ""
    if cur:
        nums.append(cur)
    return float(bool(nums) and nums[-1] == gold)


def answer_states(model, tok, texts):
    """Final-unroll state at the answer position, one row per prompt."""
    import numpy as np
    import torch
    rows = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        cap = []
        mod = model.transformer.core_block[-1]
        mod._forward_hooks.clear()
        h = mod.register_forward_hook(
            lambda m, i, o, c=cap: c.append(o.detach()[0, -1, :].float().cpu().numpy()))
        try:
            torch.manual_seed(0)
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        rows.append(cap[-1])
        del ids
        torch.cuda.empty_cache()
    return np.stack(rows).astype(np.float64)


def main():
    run("pip install -q 'transformers>=4.50,<4.54' scikit-learn scipy")
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    data = {t: [item(t, s) for s in range(N_ITEMS)] for t in (*TASKS, CONTROL)}
    for t in (*TASKS, CONTROL):
        b, g, y = data[t][0]
        print(f"  {t:>8}: {b.replace(chr(10), ' | ')!r} -> {g!r} (latent {y})", flush=True)

    res = {}
    for arm in ("trained", "untrained"):
        print(f"\n=== {arm} ===", flush=True)
        model = None
        try:
            model = load_arm(None if arm == "untrained" else MODEL_ID, cfg,
                             0 if arm == "untrained" else REVISION)
            if arm == "trained":
                assert_generation_works(model, tok)
            res[arm] = {}
            for t in (*TASKS, CONTROL):
                texts = [render(b, tok) for b, _, _ in data[t]]
                preds = batched_generate(model, tok, texts, max_new=MAX_NEW,
                                         num_steps=NUM_STEPS, verbose=False)
                acc = float(np.mean([score(t, p, g)
                                     for p, (_, g, _) in zip(preds, data[t], strict=True)]))
                if t == CONTROL:                      # harness control: accuracy only
                    res[arm][t] = {"acc": acc, "r2": float("nan"),
                                   "null_max": float("nan"), "sample": preds[0][:40]}
                    print(f"  {t:>8}: acc {acc:>6.1%}   (control, not probed)", flush=True)
                    continue
                x = answer_states(model, tok, texts)
                y = np.array([v for _, _, v in data[t]], float)
                r2, null = cv_r2(x, y, n_null=12)
                r2n, nulln = cv_r2_nonlinear(x, y, n_null=6)
                res[arm][t] = {"acc": acc, "r2": r2, "null_max": float(max(null)),
                               "null_mean": float(np.mean(null)),
                               "r2_nonlin": r2n, "null_nonlin_max": float(max(nulln)),
                               "sample": preds[0][:40]}
                print(f"  {t:>8}: acc {acc:>6.1%}   linear R2 {r2:>+7.4f} (null "
                      f"{max(null):>+6.3f})   nonlinear R2 {r2n:>+7.4f} (null "
                      f"{max(nulln):>+6.3f})   e.g. {preds[0][:22]!r}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        finally:
            free_arm(model, None if arm == "untrained" else MODEL_ID)
            with open("cap_content.json", "w") as f:
                json.dump(res, f, indent=1)

    print("\n=== DOES THE CONTENT GAP TRACK CAPABILITY? (UNDERSTANDING 6.1) ===")
    if "trained" not in res or "untrained" not in res:
        print("  an arm failed; nothing may be concluded")
        return
    t_, u_ = res["trained"], res["untrained"]
    if t_.get(CONTROL, {}).get("acc", 0) < 0.5:
        print(f"  HARNESS CONTROL FAILED: trained {CONTROL} = "
              f"{t_.get(CONTROL, {}).get('acc')}")
        print("  no conclusion may be drawn (the D62 lesson).")
        return
    print(f"  {'task':>8} {'cap(trained)':>13} {'cap(untr)':>10} {'R2 trained':>11} "
          f"{'R2 untr':>9} {'content gap':>12}")
    caps, gaps = [], []
    for t in TASKS:
        ct, cu = t_[t]["acc"], u_[t]["acc"]
        gap = t_[t]["r2"] - u_[t]["r2"]
        caps.append(ct)
        gaps.append(gap)
        print(f"  {t:>8} {ct:>12.1%} {cu:>9.1%} {t_[t]['r2']:>+11.4f} "
              f"{u_[t]['r2']:>+9.4f} {gap:>+12.4f}")
    print("\n  === and is the content NONLINEARLY encoded in the trained arm? ===")
    print("  (every content measure in this project is a LINEAR probe; if training")
    print("   encodes the same content nonlinearly, a linear probe favours the")
    print("   untrained arm for reasons unrelated to information content)")
    print(f"  {'task':>8} {'lin gap':>9} {'nonlin gap':>11} {'trained lin->nonlin':>20}")
    for t in TASKS:
        lg = t_[t]["r2"] - u_[t]["r2"]
        ng = t_[t].get("r2_nonlin", float("nan")) - u_[t].get("r2_nonlin", float("nan"))
        lift = t_[t].get("r2_nonlin", float("nan")) - t_[t]["r2"]
        print(f"  {t:>8} {lg:>+9.4f} {ng:>+11.4f} {lift:>+20.4f}")
    lifts = [t_[t].get("r2_nonlin", float("nan")) - t_[t]["r2"] for t in TASKS]
    lifts_u = [u_[t].get("r2_nonlin", float("nan")) - u_[t]["r2"] for t in TASKS]
    print(f"  mean nonlinear lift: trained {np.nanmean(lifts):+.4f}, "
          f"untrained {np.nanmean(lifts_u):+.4f}")
    if np.nanmean(lifts) > np.nanmean(lifts_u) + 0.05:
        print("  => the TRAINED arm gains more from a nonlinear probe. The linear-only")
        print("     content measures underestimate it, and D41/D48/D53 must be re-read.")
    else:
        print("  => no differential nonlinear lift. The linear probes were not the")
        print("     reason the untrained arm looked better.")

    from scipy.stats import spearmanr
    rho, p = spearmanr(caps, gaps)
    print(f"\n  spearman(capability, content gap) = {rho:+.3f}, p={p:.3f}  (n={len(caps)})")
    hi = [g for c, g in zip(caps, gaps, strict=True) if c > 0.5]
    print(f"  mean content gap where capability > 50%: "
          f"{np.mean(hi) if hi else float('nan'):+.4f}")
    if hi and np.mean(hi) > 0.05:
        print("  => 6.1 IS A REAL CONFOUND. Training DOES change content where the")
        print("     model can do the task. The headline must be scoped to tasks the")
        print("     model cannot perform.")
    else:
        print("  => 6.1 IS NOT A CONFOUND. The content gap stays ~0 even where the")
        print("     trained model is at 100% and the untrained at 0%. The headline")
        print("     survives its most serious challenge.")


main()
