"""A35: what DISPLACES the gold token? Bank the argmax, not just the gold's rank.

THE QUESTION TWO ROWS END ON. D158 found that 19 of 21 census families read exactly 0.000
when scored at the final unroll, with `echo_digit` going 0.997 -> 0.000 while `echo_word`
holds 0.833 -- a sharp dissociation by token type. D159 then showed the mechanism is not
decay but **displacement**: the gold reaches rank 1 at a median unroll of 4 and then
settles at a stable worse rank for forty-plus unrolls (`add1` at 19, `echo_digit` at 4).
Both rows end at the same sentence: **we do not know what wins instead**, because every
kernel in this project banks the GOLD's rank and nothing else.

D145 supplies a hypothesis and not an answer. The `kaggle_depthacc` generations end in a
role marker glued to the output -- `'4user\\n\\n'`, `'Clouduser\\n\\n'` -- so the model may
simply be moving on to the next TURN, and the turn boundary may outrank a short answer.
**That is untested. This tests it.**

The fix is trivial and should have been in every kernel from the start: record the **top-5
tokens and their log-probabilities at every unroll**, alongside the gold's rank.

PREREGISTERED PREDICTIONS:

  P1  REPLICATION GATE. The gold-rank trajectories must reproduce D159's banked shape on
      the same families -- `echo_digit` rank 1 early then ~4, `echo_word` rank 1 held,
      `add1` rank 1 early then ~19. Different prompts and a different seed, so exact
      equality is not expected; the ORDERING and the qualitative shape must hold or this
      run is measuring something else.

  P2  PRIMARY. At the final unroll, what is the argmax? The D145 hypothesis predicts a
      **role marker or turn/format token** (`user`, `Huginn`, a newline) for the families
      that lose the answer, and the **gold itself** for `echo_word`. Reported as the
      literal decoded strings, in full, not as a category count -- D110's printed verdict
      over empty strings is the reason this project reads raw output.

  P3  WHEN does the winner change? The unroll at which the argmax stops being the gold,
      compared against the unroll at which the gold's rank starts rising. If these coincide
      the displacement is a single event; if the argmax changes earlier the gold was never
      really 'held'.

  P4  THE DISSOCIATION CONTROL. `echo_digit` and `echo_word` are the same instruction over
      different content and they behave oppositely (D158). Both are run, so whatever the
      answer is, it must explain the difference between them and not merely describe one.

  P5  IS IT THE CHAT TEMPLATE? Each item is run BOTH through the chat template and as a
      bare prompt. If the displacing token is a role marker, removing the template should
      remove it -- and if the gold still gets displaced without a template, D145's
      hypothesis is refuted and the cause is internal.
"""

import json
import os
import subprocess
import sys
import time

DEPTH = 48
SEED = 20260811
TOPK = 5
N_ITEMS = 8
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 9000

# families chosen to span D158's dissociation: two that hold, two that lose it badly
FAMILIES = ("echo_word", "echo_digit", "add1", "sort_min")


def build_items(rng=None):
    """Small deterministic generators matching the census families by construction."""
    import random as _r
    rng = rng or _r.Random(SEED)
    words = ["cloud", "paper", "river", "stone", "candle", "garden", "window", "silver"]
    out = []
    for i in range(N_ITEMS):
        w = words[i % len(words)]
        d = rng.randrange(10)
        a, b = rng.randrange(10), rng.randrange(10)
        seq = [rng.randrange(10) for _ in range(6)]
        out += [
            {"family": "echo_word", "item": i, "gold": w,
             "prompt": f"Repeat this word exactly.\nWord: {w}"},
            {"family": "echo_digit", "item": i, "gold": str(d),
             "prompt": f"Repeat this number exactly.\nNumber: {d}"},
            {"family": "add1", "item": i, "gold": str((a + 1) % 10),
             "prompt": f"Add one to this number, modulo ten.\nNumber: {a}"},
            {"family": "sort_min", "item": i, "gold": str(min(seq)),
             "prompt": f"Report the smallest number in this list.\n"
                       f"List: {' '.join(map(str, seq))}"},
        ]
    return out


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("whatwins.json")
    mnt = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"results -> {out_path}; weights mount -> {mnt}", flush=True)

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

    # mounted weights (REMOTE_RUNS.md): no revision=, it is baked into the dataset
    tok = AutoTokenizer.from_pretrained(mnt)
    cfg = AutoConfig.from_pretrained(mnt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mnt, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    core_last = model.transformer.core_block[-1]

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def one(prompt, gold, templated):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True) \
            if templated else prompt
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]
        g_ids = tok(gold, add_special_tokens=False).input_ids
        g0 = g_ids[0]
        ranks, tops = [], []
        core_last._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                row = torch.log_softmax(
                    coda_head(o.detach(), freqs).float()[0, n_p - 1], dim=-1)
                ranks.append(int((row > row[g0]).sum().item()) + 1)
                lp, ix = torch.topk(row, TOPK)
                tops.append([[tok.decode([int(j)]), round(float(p), 3)]
                             for p, j in zip(lp.tolist(), ix.tolist())])

        h = core_last.register_forward_hook(hook)
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=DEPTH)
        finally:
            h.remove()
        # P3: first unroll at which the argmax stops being the gold, after it was the gold
        was = [i for i, t in enumerate(tops) if t[0][0].strip() == gold.strip()]
        lost = None
        if was:
            for i in range(was[0], len(tops)):
                if tops[i][0][0].strip() != gold.strip():
                    lost = i
                    break
        return {"rank_curve": ranks, "top5": tops, "n_tokens": int(n_p),
                "best_rank": int(min(ranks)), "best_depth": int(np.argmin(ranks)) + 1,
                "final_rank": int(ranks[-1]),
                "argmax_final": tops[-1][0][0], "argmax_final_lp": tops[-1][0][1],
                "argmax_ever_gold": bool(was), "first_gold_argmax": (was[0] if was else -1),
                "lost_at": (lost if lost is not None else -1),
                "multi_token_gold": bool(len(g_ids) > 1)}

    items = [it for it in build_items() if it["family"] in FAMILIES]
    print(f"{len(items)} items over {sorted(set(i['family'] for i in items))}", flush=True)
    t0, rows = time.time(), []
    for n, it in enumerate(items):
        for templated in (True, False):          # P5
            try:
                r = one(it["prompt"], it["gold"], templated)
            except Exception as exc:  # noqa: BLE001
                r = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
            r.setdefault("ok", True)
            r.update({k: it[k] for k in ("family", "item", "gold", "prompt")})
            r["templated"] = templated
            rows.append(r)
        if n % 6 == 0:
            print(f"  {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok = [r for r in rows if r.get("ok")]
    print("\n=== P2 WHAT WINS AT THE FINAL UNROLL (literal strings) ===", flush=True)
    import collections
    for tmpl in (True, False):
        print(f"  templated={tmpl}", flush=True)
        for f in FAMILIES:
            v = [r for r in ok if r["family"] == f and r["templated"] == tmpl]
            if not v:
                continue
            c = collections.Counter(repr(r["argmax_final"]) for r in v)
            gold_wins = sum(1 for r in v if r["argmax_final"].strip() == r["gold"].strip())
            print(f"    {f:11s} gold wins {gold_wins}/{len(v)}   argmax: "
                  f"{dict(c.most_common(4))}", flush=True)
    print("\n=== P1/P3 shape ===", flush=True)
    for f in FAMILIES:
        v = [r for r in ok if r["family"] == f and r["templated"]]
        if v:
            print(f"  {f:11s} median best_depth {np.median([r['best_depth'] for r in v]):.1f}"
                  f"  median final_rank {np.median([r['final_rank'] for r in v]):.1f}"
                  f"  lost_at {[r['lost_at'] for r in v][:6]}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "depth": DEPTH, "topk": TOPK, "seed": SEED,
                   "families": list(FAMILIES), "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} runs, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
