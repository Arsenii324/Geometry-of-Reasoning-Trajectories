"""Accuracy on every task family whose GEOMETRY this project measured but whose
correctness it never did.

THE GAP (claims_ledger D56(2))
    A census of `results/*.csv` found geometry measured on EIGHT task families and
    model correctness on ONE. `switch` (parity), `maxtask` (running max),
    `projection`, `three_scale`, `three_scale_modk`, `running_count`,
    `nesting_depth` and `count_ones` all have winding and steps-to-settle rows and
    no accuracy column anywhere. So for eight of ten tasks we do not know whether
    the model can do the thing whose "reasoning trajectory" was characterised --
    and if it cannot, those trajectories are the geometry of a model FAILING,
    which is a legitimate object but not what the project set out to describe.

    D56 called this the single highest-value gap and estimated it at one
    generation pass per task. It was not run then because generation cost ~1
    minute per completion (D57/C9); `batched_generate` removed that.

DESIGN
    * CHAT FORMAT, because D60 showed a trivial copy task scores 15% under raw
      prompting and 100% under Huginn's own chat template. Every earlier accuracy
      number here was measured raw and is a lower bound. Raw is kept as one arm so
      the size of that gap is measured per task rather than assumed from `copy`.
    * A DIFFICULTY LADDER per task, because D56 showed counting is 37.5% at n=2 and
      0% from n=8 onward. A single hard setting cannot distinguish "cannot do this"
      from "cannot do this at this size", and the geometry runs used the large
      sizes. Each task is asked at n = 2, 4, 8, 16, 32.
    * THE UNTRAINED ARM, so any non-zero cell is immediately checkable against
      chance -- the control that reshaped this project four times (D40/41/48/53).
    * A `copy` control at every ladder rung, so a row of zeros can be told apart
      from a broken harness.

WHAT WOULD CHANGE THE PICTURE
    If any family is well above zero at the sizes the geometry was measured at,
    that family's geometry describes a working computation and is worth revisiting.
    If all are at zero by n=8, then D56(2)'s worry is confirmed: the project
    characterised the trajectories of a model that could not do the tasks, and the
    honest scope of every geometric result narrows accordingly.
"""
# @needs: run load_arm free_arm batched_generate

import json
import random

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 32
MAX_NEW = 12
N_ITEMS = 16
LADDER = (2, 4, 8, 16, 32)


def make(task, n, seed):
    """(question_body, gold_answer_string) for one item."""
    rng = random.Random(seed * 7919 + n)
    if task == "count_ones":
        b = [rng.randint(0, 1) for _ in range(n)]
        return (f"Count how many ones are in this sequence.\nSequence: "
                f"{' '.join(map(str, b))}", str(sum(b)))
    if task == "parity":
        b = [rng.randint(0, 1) for _ in range(n)]
        return (f"Is the number of ones in this sequence even or odd?\nSequence: "
                f"{' '.join(map(str, b))}", "even" if sum(b) % 2 == 0 else "odd")
    if task == "running_max":
        v = [rng.randint(0, 99) for _ in range(n)]
        return (f"What is the largest number in this list?\nList: "
                f"{' '.join(map(str, v))}", str(max(v)))
    if task == "running_count":
        b = [rng.randint(0, 1) for _ in range(n)]
        return (f"Start at 0. For each 1 add one, for each 0 subtract one. "
                f"What is the final total?\nSequence: {' '.join(map(str, b))}",
                str(sum(1 if x else -1 for x in b)))
    if task == "nesting_depth":
        seq, d, mx = [], 0, 0
        for _ in range(n):
            if d == 0 or (rng.random() < 0.5 and d < 8):
                seq.append("(")
                d += 1
                mx = max(mx, d)
            else:
                seq.append(")")
                d -= 1
        seq += [")"] * d
        return (f"What is the maximum nesting depth of these brackets?\nBrackets: "
                f"{' '.join(seq)}", str(mx))
    if task == "modk":
        b = [rng.randint(0, 1) for _ in range(n)]
        return (f"Count the ones in this sequence, then give the remainder when "
                f"divided by 5.\nSequence: {' '.join(map(str, b))}", str(sum(b) % 5))
    if task == "copy":
        w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        return (f"Repeat this word exactly.\nWord: {w}", w)
    raise ValueError(task)


TASKS = ("count_ones", "parity", "running_max", "running_count",
         "nesting_depth", "modk", "copy")


def render(body, arm, tok):
    if arm == "raw":
        return body + "\nAnswer:"
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True)


def score(task, pred, gold):
    p = pred.lower().strip().strip(".").strip()
    if task == "parity":
        has_e, has_o = "even" in p, "odd" in p
        return float(has_e != has_o and (has_e if gold == "even" else has_o))
    if task == "copy":
        return float("".join(c for c in p if c.isalpha()) == gold)
    digits = "".join(c if (c.isdigit() or c == "-") else " " for c in p).split()
    return float(bool(digits) and digits[0] == gold)


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    print(f"{len(TASKS)} tasks x {len(LADDER)} sizes x {N_ITEMS} items x 2 formats "
          f"x 2 arms = {len(TASKS) * len(LADDER) * N_ITEMS * 4} completions", flush=True)
    ex = make("parity", 8, 0)
    print(f"  example: {ex[0]!r} -> {ex[1]!r}", flush=True)

    out, samples = {}, {}
    for arm in ("trained", "untrained"):
        print(f"\n=== {arm} ===", flush=True)
        model = None
        try:
            model = load_arm(None if arm == "untrained" else MODEL_ID, cfg,
                             0 if arm == "untrained" else REVISION)
            out[arm] = {}
            for fmt in ("chat", "raw"):
                out[arm][fmt] = {}
                for task in TASKS:
                    row = {}
                    for n in LADDER:
                        items = [make(task, n, s) for s in range(N_ITEMS)]
                        preds = batched_generate(
                            model, tok, [render(b, fmt, tok) for b, _ in items],
                            max_new=MAX_NEW, num_steps=NUM_STEPS, verbose=False)
                        sc = [score(task, p.split("\n")[0], g)
                              for p, (_, g) in zip(preds, items, strict=True)]
                        row[n] = float(np.mean(sc))
                        if n == LADDER[0]:
                            samples[f"{arm}/{fmt}/{task}"] = [
                                {"gold": items[i][1], "pred": preds[i].split("\n")[0][:40]}
                                for i in range(3)]
                    out[arm][fmt][task] = row
                    print(f"  {fmt:>4} {task:>14}: " + "  ".join(
                        f"n={n}:{row[n]:>5.0%}" for n in LADDER), flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        finally:
            free_arm(model, None if arm == "untrained" else MODEL_ID)
            with open("task_accuracy.json", "w") as f:
                json.dump({"acc": out, "samples": samples, "ladder": list(LADDER),
                           "n_items": N_ITEMS}, f, indent=1)

    print("\n=== DOES ANY FAMILY WORK AT THE SIZES THE GEOMETRY USED? ===")
    t = out.get("trained", {}).get("chat", {})
    u = out.get("untrained", {}).get("chat", {})
    print(f"  {'task':>14} " + " ".join(f"{'n=' + str(n):>7}" for n in LADDER)
          + "   untrained@n=8")
    for task in TASKS:
        if task in t:
            print(f"  {task:>14} " + " ".join(f"{t[task][n]:>6.0%} " for n in LADDER)
                  + f"   {u.get(task, {}).get(8, float('nan')):>6.0%}")
    live = [k for k, v in t.items() if k != "copy" and v.get(8, 0) > 0.25]
    print()
    if live:
        print(f"  FAMILIES ALIVE AT n=8: {live}. Their geometry describes a working")
        print("  computation and is worth revisiting.")
    else:
        print("  NO family except `copy` clears 25% at n=8. D56(2) is confirmed: the")
        print("  geometry in this project was measured on tasks the model cannot do,")
        print("  and every geometric result must be scoped accordingly.")


main()
