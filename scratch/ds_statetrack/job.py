"""A27: state tracking — the one difficulty axis DR1 argues recurrence MUST need.

WHY THIS TASK AND NOT ANOTHER. Every difficulty ladder this project has run is
shortcuttable in principle: `addk` (difficulty collinear with the answer), `fixedsum`
(VOID at 0% above one nonzero digit, D118), `binary_search` (depth ran OPPOSITE to
Huginn's difficulty, D128), `nth_item_k` (structurally clean, H2 null once list position 1
is removed, D143). A fixed-depth transformer can in principle collapse all of them.

DR1 names the exception. Merrill, Petty & Sabharwal show transformers and SSMs cannot
express computation outside TC0 and **cannot solve state-tracking problems like
permutation composition**; the A5 word problem is NC1-complete under TC0 != NC1. So
sequential state tracking is the one place where genuine recurrence is *required* rather
than merely available — and therefore the one place H2 gets a fair trial. This project has
never run it.

HONEST SCOPE, STATED BEFORE THE RESULT. **S3 is a SOLVABLE group, so the NC1-hardness
guarantee does NOT apply to this task.** Tracking one ball under swaps of three cups is a
state-tracking problem, not an NC1-complete one; a fixed-depth model could in principle
shortcut it. A5 would carry the guarantee and would also certainly read 0% on a model that
cannot add two nonzero digits (D118). The choice here is deliberate: **run the hardest
version that has any chance of leaving the floor**, measure it, and let the result decide
whether the harder version is worth GPU. Claiming the theoretical guarantee for S3 would
be false and this docstring exists so that no later row does.

THE DESIGN, and why the difficulty axis is clean:
  * The prompt always contains **exactly 8 swaps**, so token count is constant by
    construction and does not need to be gated after the fact — though it IS gated anyway.
  * A swap only moves the ball if it involves the cup the ball is currently under.
    **Difficulty m = the number of swaps that actually move the ball**, which varies
    0..8 at fixed length. Difficulty is ENACTED (the model must track through all eight to
    know which mattered) and ORTHOGONAL to prompt length by construction.
  * The answer is one of three cup labels: **chance floor 1/3**, single token, no
    multi-token gold (the defect D89 found in 8 of 21 families).

PREREGISTERED PREDICTIONS:

  P1  CAPABILITY GATE, and it decides whether anything else may be read. Following DR1's
      own Stage-1 advice and D130's lesson, this is a SCREEN first. **The registered bar:
      3-way accuracy at m = 1, in the better of the two formats, must exceed the
      MAJORITY-CLASS baseline of that level (0.357 at 14 items with a 5/5/4 gold split,
      not the 1/3 chance floor) at binomial p < 0.05 -- with n = 14 that needs >= 9 of
      14. If it does not, the task is dead for Huginn and NO geometry is read.**
      The baseline rather than chance, because golds are balanced but not perfectly:
      the first draft was A:31 C:35 B:21, where a constant "C" scored 40% and would
      have passed a chance-floor gate on no ability at all. Reported either way, because "Huginn cannot
      state-track at all" is itself the answer to whether H2 can ever be tested here.

  P2  DYNAMIC RANGE. `require_dynamic_range`'s condition, applied in advance: at least
      two m-levels must sit strictly inside (0.05, 0.95). B4b died on exactly this gate
      after the fact (D118); here it is checked before H2 is computed.

  P3  H2, only if P1 and P2 pass. Does `best_depth` rise with m? At the (m, item) unit,
      never the draw — D101's p-values were pseudoreplicated ~12x and D143's headline was
      an artefact of one trivial level. **m = 0 is excluded a priori**, exactly as D143's
      list-position-1 had to be excluded post hoc: with no ball-moving swap the answer is
      the start cup and no tracking is required.

  P4  FORMAT CONTROL. D103/D89 show the capability axis moves by up to 85 points with
      prompt format alone. Every item is run in TWO formats — bare and 2-shot — so a floor
      result cannot be blamed on elicitation without evidence.

  P5  A SHORTCUT PROBE, because S3 is solvable. If the model is tracking, accuracy should
      fall with m; if it is pattern-matching the final swap, accuracy should be flat in m
      and predicted by whether the LAST swap touches the ball. Both are recorded.
"""

import json
import os
import subprocess
import sys
import time

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 48
N_SWAPS = 8                 # constant -> token count constant by construction
CUPS = ("A", "B", "C")
M_LEVELS = (0, 1, 2, 3, 4, 5, 6)
N_ITEMS = 14                # per m-level
SEED = 20260810
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 10000

SHOTS = (
    ("Cups A, B, C. Ball under A. Swap A and B. Swap B and C. "
     "Swap A and B. Swap B and C. Swap A and C. Swap A and B. "
     "Swap B and C. Swap A and C.", "B"),
    ("Cups A, B, C. Ball under C. Swap B and C. Swap A and C. "
     "Swap A and B. Swap A and C. Swap B and C. Swap A and B. "
     "Swap A and C. Swap B and C.", "A"),
)


def _apply(pos, a, b):
    return b if pos == a else (a if pos == b else pos)


def build_items(rng=None):
    """Eight swaps always; difficulty m = how many of them move the ball."""
    import random as _r
    rng = rng or _r.Random(SEED)
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    out, seen = [], set()
    for m in M_LEVELS:
        # GOLD IS BALANCED WITHIN EACH LEVEL, by accepting only the next target answer
        # in rotation. Unbalanced golds would hand a constant responder more than the
        # 1/3 chance floor -- the first draft gave A:31 C:35 B:21, so always answering
        # "C" scored 40%, above the bar P1 gates on. That is how a capability screen
        # manufactures its own positive.
        tries = 0
        got = 0
        while got < N_ITEMS and tries < 400000:
            tries += 1
            want = CUPS[got % len(CUPS)]
            start = rng.choice(CUPS)
            swaps = [rng.choice(pairs) for _ in range(N_SWAPS)]
            pos = start
            moved = 0
            last_touches = False
            for i, (a, b) in enumerate(swaps):
                touches = pos in (a, b)
                if i == N_SWAPS - 1:
                    last_touches = touches
                if touches:
                    moved += 1
                pos = _apply(pos, a, b)
            if moved != m or pos != want:
                continue
            key = (start, tuple(swaps))
            if key in seen:
                continue
            seen.add(key)
            body = " ".join(f"Swap {a} and {b}." for a, b in swaps)
            out.append({
                "m": m, "item": got, "start": start, "gold": pos,
                "last_touches": last_touches,
                "body": f"Cups A, B, C. Ball under {start}. {body}",
            })
            got += 1
    return out


def prompt_for(it, fmt):
    q = (f"{it['body']} Which cup is the ball under? "
         f"Answer with one letter: A, B or C.")
    if fmt == "bare":
        return q
    ex = "\n\n".join(
        f"{b} Which cup is the ball under? Answer with one letter: A, B or C.\nAnswer: {g}"
        for b, g in SHOTS)
    return f"{ex}\n\n{q}\nAnswer:"


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("statetrack.json")
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"results -> {out_path}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    core_last = model.transformer.core_block[-1]
    # the three answer tokens, so rank can be scored against the 3-way choice as well as
    # the full vocabulary -- a full-vocab rank of 400 and a 3-way rank of 1 mean different
    # things, and D68 showed reporting only one of them hides the effect
    choice_ids = [tok(c, add_special_tokens=False).input_ids[0] for c in CUPS]
    print("choice token ids:", dict(zip(CUPS, choice_ids)), flush=True)

    def one(prompt, gold):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        g_ids = tok(gold, add_special_tokens=False).input_ids
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        ranks, choice_ok = [], []
        core_last._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                row = torch.log_softmax(
                    coda_head(o.detach(), freqs).float()[0, n_p - 1], dim=-1)
                ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)
                sub = [float(row[c]) for c in choice_ids]
                choice_ok.append(bool(int(np.argmax(sub)) == CUPS.index(gold)))

        h = core_last.register_forward_hook(hook)
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        return {"rank_curve": ranks, "n_tokens": int(n_p),
                "best_rank": int(min(ranks)), "best_depth": int(np.argmin(ranks)) + 1,
                "correct": bool(min(ranks) == 1),
                "choice_correct_final": bool(choice_ok[-1]),
                "choice_correct_any": bool(any(choice_ok)),
                "choice_best_depth": int(np.argmax(choice_ok) + 1) if any(choice_ok) else -1,
                "multi_token_gold": bool(len(g_ids) > 1)}

    items = build_items()
    print(f"{len(items)} items over m = {sorted({i['m'] for i in items})}", flush=True)
    t0, rows = time.time(), []
    for n, it in enumerate(items):
        for fmt in ("bare", "shots"):
            try:
                r = one(prompt_for(it, fmt), it["gold"])
            except Exception as exc:  # noqa: BLE001
                r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
            r.setdefault("ok", True)
            r.update({k: it[k] for k in ("m", "item", "start", "gold", "last_touches")})
            r["fmt"] = fmt
            rows.append(r)
        if n % 10 == 0:
            print(f"  {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok = [r for r in rows if r.get("ok")]
    print(f"\n{len(ok)}/{len(rows)} forwards ok", flush=True)
    print("\n=== P1 CAPABILITY GATE / P2 DYNAMIC RANGE ===", flush=True)
    for fmt in ("bare", "shots"):
        print(f"  format={fmt}", flush=True)
        for m in sorted({r["m"] for r in ok}):
            v = [r for r in ok if r["m"] == m and r["fmt"] == fmt]
            if not v:
                continue
            print(f"    m={m}  n={len(v):3d}  vocab-acc {np.mean([r['correct'] for r in v]):.3f}"
                  f"  3-way-final {np.mean([r['choice_correct_final'] for r in v]):.3f}"
                  f"  3-way-any {np.mean([r['choice_correct_any'] for r in v]):.3f}"
                  f"  median best_depth {np.median([r['best_depth'] for r in v]):.1f}",
                  flush=True)
    nt = sorted({r["n_tokens"] for r in ok if r["fmt"] == "bare"})
    print(f"\n  P3 token-count gate (bare): n_tokens {nt} -> "
          f"{'CONSTANT' if len(nt) == 1 else 'VARIES'}", flush=True)

    summary = {"rows": rows, "num_steps": NUM_STEPS, "n_swaps": N_SWAPS,
               "cups": list(CUPS), "chance": 1.0 / len(CUPS), "seed": SEED,
               "elapsed_s": time.time() - t0}
    with open(out_path, "w") as f:
        json.dump(summary, f)
    print(f"DONE {len(rows)} forwards, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
