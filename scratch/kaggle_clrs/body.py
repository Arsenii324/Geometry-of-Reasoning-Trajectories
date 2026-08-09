"""Capability screen on CLRS-Text, which is IN Huginn's own training mixture.

THE GAP THIS ADDRESSES, NAMED BY THE SUPERVISOR. The project has never had a task
family with a difficulty PARAMETER and success in a readable band. Difficulty varies
across the 21-family battery, but the two parametric ladders inside it are broken:
`count4/count8/count16` reads 21% / 0% / 12% -- floored and non-monotone -- and
`caesar1_letter/caesar1_word/rot13_word` reads 8% / 0% / 0%. So "more reasoning
steps" has never been testable against a graded axis that the model can actually
climb.

WHY CLRS-TEXT AND NOT ANOTHER SYNTHETIC FAMILY.
  * It is in Huginn's pre-training data (`tomg-group-umd/CLRS-Text-train`, listed on
    the model card). Every task this project has used was invented here, so "the
    model looks incapable because the prompts are off-distribution" has been a live
    and untested explanation. This removes it.
  * Two independent difficulty knobs: the ALGORITHM (30 of them) and the problem
    SIZE n.
  * It can break the length/difficulty collinearity, which is the property that
    matters most here. D85 measured the geometry tracking prompt length and D88
    showed the shape reading a single token, so a difficulty axis collinear with
    token count is unusable. Array algorithms scale ~n in tokens and graph
    algorithms ~n^2, so (algorithm, n) cells exist with matched token counts and
    very different difficulty. This run measures the token counts so that design can
    be built rather than assumed.

SCORED ON THE DECODED STRING, NOT ON FIRST-TOKEN RANK. D89 found that `correct` --
the first token of gold reaching rank 1 -- measures the leading token for 8 of 21
battery families, and CLRS answers are long numeric strings where that instrument is
simply wrong. Every completion is banked verbatim so scoring can be redone offline
without another GPU hour.

PRE-REGISTERED (CLAUDE.md section 1).
P1 -- GATE ON THE INSTRUMENT. `assert_generation_works` must pass before any number
      is read. D62 printed a capability verdict over empty strings for want of it.
P2 -- THE SCREEN. Per (algorithm, size) cell, exact-match on the FINAL answer -- the
      field after the last "|" -- and a containment score, over 6 items. The output
      is the list of cells landing in 20-80%.
P3 -- THE LENGTH MAP, which is the part that makes the screen usable. Token count is
      recorded per item, so cells with matched token count and different difficulty
      can be identified. Reported whatever P2 returns.
P4 -- REFUSAL. If no cell lands in 20-80% at any depth, that is reported as the
      result and the next run is a different dataset, not a different window. The
      project has twice mistaken a floored axis for a measurement.

Two depths, because D86 found exact-match peaking shallow and collapsing by r=8
while containment rises to r=32: r=4 and r=32 bracket that.
"""
# @needs: run load_arm free_arm batched_generate assert_generation_works preflight

import json
import os
import re
import time
import traceback

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
DATASET = "tomg-group-umd/CLRS-Text-train"
OUTDIR = "/kaggle/working"

N_ITEMS = 6              # per (algorithm, size) cell
MAX_NEW = 24
DEPTHS = (4, 32)         # D86's bracket: exact peaks shallow, containment rises deep
N_ALGOS = 10
SIZE_BINS = (4, 8, 12, 16)   # problem sizes, read off the input list length


def final_answer(ans):
    """CLRS answers are `trace | final`; the screen scores the FINAL field.

    Splitting on the LAST separator, not the first: several algorithms emit `|`
    inside the trace, and taking the first would score a trace step as the answer.
    """
    return ans.rsplit("|", 1)[-1].strip() if "|" in ans else ans.strip()


def problem_size(question):
    """Number of elements in the first bracketed list -- CLRS's difficulty knob.

    Read from the question text rather than from a metadata column, because the
    dataset carries only `question`, `answer` and `algo_name`. Returns 0 when no
    list is found, and those items are dropped rather than binned as size 0.
    """
    m = re.search(r"\[([^\[\]]*)\]", question)
    if not m:
        return 0
    inner = m.group(1).strip()
    return len(inner.split()) if inner else 0


def normalise(text):
    """Same normalisation on both sides, as in `scripts/run_depth_accuracy.py`."""
    t = str(text).strip().lower()
    for ch in ".,!?;:'\"":
        t = t.replace(ch, " ")
    return " ".join(t.split())


def score(pred, gold):
    p, g = normalise(pred), normalise(gold)
    return (p == g), (g in p)


def main():
    run("pip install -q 'transformers>=4.50,<4.54' datasets")
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
          flush=True)
    ds = load_dataset(DATASET, split="train", streaming=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

    # Build the (algorithm, size) grid by streaming until each cell is full, so the
    # whole 430k-row parquet never has to be materialised on a Kaggle worker.
    want = N_ITEMS
    cells, seen_algos = {}, []
    for row in ds:
        algo = row["algo_name"]
        if algo not in seen_algos:
            if len(seen_algos) >= N_ALGOS:
                continue
            seen_algos.append(algo)
        n = problem_size(row["question"])
        if n == 0:
            continue
        size = min(SIZE_BINS, key=lambda b: abs(b - n))
        if abs(size - n) > 2:
            continue
        key = (algo, size)
        cur = cells.setdefault(key, [])
        if len(cur) >= want:
            continue
        cur.append({"algo": algo, "size": size, "n_raw": n,
                    "question": row["question"],
                    "gold": final_answer(row["answer"])})
        if len(cells) >= N_ALGOS * len(SIZE_BINS) and all(
                len(v) >= want for v in cells.values()):
            break
    items = [x for v in cells.values() for x in v if len(v) >= 3]
    print(f"{len(items)} items over {len({(i['algo'], i['size']) for i in items})} "
          f"(algorithm, size) cells, {len({i['algo'] for i in items})} algorithms",
          flush=True)

    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = load_arm(MODEL_ID, cfg, REVISION)
    assert_generation_works(model, tok, chat=True)
    print("P1: generation gate PASSED", flush=True)

    for it in items:
        text = tok.apply_chat_template([{"role": "user", "content": it["question"]}],
                                       tokenize=False, add_generation_prompt=True)
        it["text"] = text
        it["n_tokens"] = int(tok(text, add_special_tokens=False,
                                 return_tensors="pt").input_ids.shape[1])

    records = []
    t0 = time.time()
    for depth in DEPTHS:
        try:
            outs = batched_generate(model, tok, [i["text"] for i in items],
                                    max_new=MAX_NEW, num_steps=depth, verbose=False)
        except Exception as exc:
            print(f"  depth {depth}: FAILED {type(exc).__name__}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            continue
        deg = degenerate(outs)
        for it, o in zip(items, outs):
            ex, ct = score(o, it["gold"])
            records.append({k: it[k] for k in ("algo", "size", "n_raw", "gold",
                                               "n_tokens", "question")}
                           | {"depth": depth, "output": o, "exact": bool(ex),
                              "contains": bool(ct)})
        with open(os.path.join(OUTDIR, "clrs.json"), "w") as fh:
            json.dump(records, fh)
        here = [r for r in records if r["depth"] == depth]
        n = max(1, len(here))
        print(f"  depth {depth:>3}: exact {sum(r['exact'] for r in here) / n:.1%}, "
              f"contains {sum(r['contains'] for r in here) / n:.1%}, {len(here)} items, "
              f"degenerate={deg}", flush=True)

    print("\n=== P2: per-cell accuracy (best over depth) ===", flush=True)
    band = []
    for (algo, size) in sorted({(r["algo"], r["size"]) for r in records}):
        sub = [r for r in records if r["algo"] == algo and r["size"] == size]
        by_depth = {d: [r for r in sub if r["depth"] == d] for d in DEPTHS}
        best_ex = max((sum(r["exact"] for r in v) / max(1, len(v))
                       for v in by_depth.values()), default=0.0)
        best_ct = max((sum(r["contains"] for r in v) / max(1, len(v))
                       for v in by_depth.values()), default=0.0)
        ntok = sum(r["n_tokens"] for r in sub) / max(1, len(sub))
        flag = "  <- IN BAND" if 0.2 <= best_ex <= 0.8 else ""
        if flag:
            band.append((algo, size, best_ex))
        print(f"  {algo:>28} n={size:<3} exact {best_ex:>5.0%} contains "
              f"{best_ct:>5.0%}  {ntok:>5.0f} tok{flag}", flush=True)

    print(f"\n=== P2 RESULT: {len(band)} cells land in 20-80% exact ===", flush=True)
    if not band:
        print("  P4: NO cell is in band. The screen's answer is that this dataset,",
              flush=True)
        print("  at these sizes and depths, does not give a readable difficulty axis",
              flush=True)
        print("  either -- report it and change the dataset, not the window.", flush=True)

    print("\n=== P3: token count against size, per algorithm ===", flush=True)
    for algo in sorted({r["algo"] for r in records}):
        per = {}
        for r in records:
            if r["algo"] == algo:
                per.setdefault(r["size"], []).append(r["n_tokens"])
        print(f"  {algo:>28}: " + "  ".join(
            f"n={s}:{sum(v) / len(v):.0f}tok" for s, v in sorted(per.items())),
            flush=True)

    free_arm(model, MODEL_ID)
    print(f"\n=== {len(records)} generations in {(time.time() - t0) / 60:.1f} min ===",
          flush=True)
    print("scoring is redone offline from the banked strings", flush=True)
    print("DONE", flush=True)


main()
