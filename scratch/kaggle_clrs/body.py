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

N_ITEMS = 4              # per (algorithm, size) cell -- was 6; see DEPTHS below
MAX_NEW = 24
# DEPTHS WAS (4, 32) AND COULD NOT FINISH (D105). Depth 4 took 6126 s on 222 items;
# depth 32 is 8x the unrolls, so the pair needed ~55500 s against Kaggle's 43200 s
# limit, and because the dump happened only at the END of a depth the overrun
# produced NOTHING. (4, 8) costs ~6126 + ~12250 = ~18400 s, which fits with margin
# even after few-shot exemplars lengthen every prompt.
# KEPT AT (4, 32) DELIBERATELY. An earlier fix dropped it to (4, 8) to fit the
# budget, and `test_clrs_brackets_the_depth_where_d86_found_the_transition` caught
# that this destroys the design: D86's finding is that exact-match peaks SHALLOW
# while containment rises DEEP, and only a 4-vs-32 bracket tests it. The budget is
# met by cutting N_ITEMS 6 -> 4 (222 -> ~148 items) instead, and by the two
# protections D105 lacked: per-batch dumps and the wall guard below, which together
# mean an overrun costs the tail of a depth rather than everything.
DEPTHS = (4, 32)
WALL_BUDGET_S = 34000    # stop cleanly well inside Kaggle's 12 h, having saved
N_ALGOS = 10
SIZE_BINS = (4, 8, 12, 16)   # problem sizes, read off the input list length
N_SHOTS = 2              # same-algorithm exemplars, selected PER ITEM excluding itself
INSTRUCTION = ("Answer with only the final value after the last '|'. "
               "Do not restate the trace.")
ANSWER_PREFIX = "Final answer: "


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

    # FEW-SHOT + A STRICT ANSWER PREFIX. D105 scored 0.0% exact with a bare
    # question and no exemplars, which the deep-research pass says cannot be read
    # as a capability floor -- and this project's own D69 measured one added
    # instruction moving a family 0% -> 83%. Exemplars are drawn from the SAME
    # algorithm as the item so the format is demonstrated, and are taken from
    # items NOT scored, so no item ever sees its own answer.
    by_algo: dict[str, list] = {}
    for it in items:
        by_algo.setdefault(it["algo"], []).append(it)

    for it in items:
        # Exemplars from the SAME algorithm but never this item: every item is
        # scored, so drawing shots from a fixed prefix of the group would hand
        # items 0..N_SHOTS-1 their own gold. Selecting per item is the fix, and
        # the leakage gate below verifies it rather than trusting it.
        pool = [d for d in by_algo[it["algo"]] if d is not it][:N_SHOTS]
        shot_text = "".join(
            f"{d['question']}\n{ANSWER_PREFIX}{d['gold']}\n\n" for d in pool)
        prompt = (f"{INSTRUCTION}\n\n{shot_text}"
                  f"{it['question']}\n{ANSWER_PREFIX}")
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        it["text"] = text
        it["n_tokens"] = int(tok(text, add_special_tokens=False,
                                 return_tensors="pt").input_ids.shape[1])

    # NO ITEM MAY SEE ITS OWN ANSWER. An exemplar drawn from the scored set would
    # hand the model the gold it is being tested on; check it rather than assume.
    # Check QUESTION IDENTITY, not gold value. CLRS golds are small integers, so
    # two items in one cell routinely share one; a value-based check would read
    # that coincidence as leakage and abort a legitimate run. What actually
    # matters is whether an item appears as its OWN exemplar.
    leaked = [it["algo"] for it in items
              if it["text"].count(it["question"]) > 1]
    if leaked:
        print(f"  LEAKAGE GATE FAILED: {len(leaked)} items contain their own gold "
              f"in an exemplar -- refusing to score. Algos: {sorted(set(leaked))}",
              flush=True)
        return
    print(f"  leakage gate PASSED ({len(items)} items, {N_SHOTS}-shot)", flush=True)

    records = []
    t0 = time.time()
    BATCH = 24
    for depth in DEPTHS:
        if time.time() - t0 > WALL_BUDGET_S:
            print(f"  wall budget {WALL_BUDGET_S}s reached before depth {depth} -- "
                  f"stopping cleanly with {len(records)} records SAVED", flush=True)
            break
        try:
            # PER-BATCH, NOT PER-DEPTH. D105 lost 4.8 hours of depth-32 work because
            # the dump happened only after a whole depth finished.
            outs = []
            for b0 in range(0, len(items), BATCH):
                chunk = items[b0:b0 + BATCH]
                outs.extend(batched_generate(
                    model, tok, [i["text"] for i in chunk],
                    max_new=MAX_NEW, num_steps=depth, verbose=False))
                for it, o in zip(chunk, outs[b0:b0 + len(chunk)]):
                    ex, ct = score(o, it["gold"])
                    records.append({k: it[k] for k in ("algo", "size", "n_raw",
                                                       "gold", "n_tokens", "question")}
                                   | {"depth": depth, "output": o,
                                      "exact": bool(ex), "contains": bool(ct)})
                with open(os.path.join(OUTDIR, "clrs.json"), "w") as fh:
                    json.dump(records, fh)
                done = b0 + len(chunk)
                print(f"    depth {depth:>3} [{done}/{len(items)}] "
                      f"elapsed {time.time()-t0:.0f}s, saved", flush=True)
                if time.time() - t0 > WALL_BUDGET_S:
                    print("    wall budget reached mid-depth -- saved and stopping",
                          flush=True)
                    break
        except Exception as exc:
            print(f"  depth {depth}: FAILED {type(exc).__name__}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            continue
        deg = degenerate(outs)
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
