"""Block-resolved trajectories at a VERIFIED-IDENTICAL token count. A bank, not a test.

DATASPHERE PORT of `scratch/kaggle_blockbank/main.py` -- Kaggle's weekly 30 h GPU quota
is exhausted. Identical banking logic; the only changes are the pip preamble, a local
OUTDIR, and a tarball at the end because DataSphere `outputs:` collects declared FILES
and this run writes ~128 of them. It imports NOTHING from the repo, so the A4b
ImportError class cannot apply here.

WHY THIS RUN EXISTS. Three separate threads are blocked on the same missing data,
and none of them can be answered from anything currently banked:

  A3 (G2.2)  Decode task identity from the CYCLE RESIDUAL. D100 found every cycle
             summary statistic invariant across tasks; if the cycle carries anything,
             it lives in deviations from the mean cycle. Checked 2026-08-09:
             `ds_blockcycle` stores 6 summary records and `ds_cyclegeom` 63 feature
             rows -- **no per-block state vectors anywhere**, so the residual cannot
             be formed. The cheap substitute (decode family from banked cycle
             features) is unusable because D100's three families sit at 47 / 28 / 37
             tokens, making family and length collinear -- D80's clock again.

  D123       Persistent homology's signal is concentrated in the orbits that ROTATE
             (median normalised persistence 0.0241 vs 0.0000 settling, 17x its own
             manifold-matched null, 124/124). That was measured at `core_block[-1]`
             only. D98 found a period-4 cycle ACROSS the four blocks. Whether the
             H1 signature is a within-block or an across-block object is unanswerable
             without per-block states.

  D98        Its period-4 across-block cycle was measured on prompts that were not
             length-matched to each other. This banks the same object under the
             marker design, where the two arms differ in exactly ONE token.

WHAT IT BANKS. For each prompt, the state at the answer position after EVERY one of
the four core blocks, at every unroll: a [num_steps, 4, hidden] array. Plus the gold
rank curve read through the D71-validated coda head. **No analysis here by design** --
this run produces data, and every question above is then a local, zero-GPU script.
That separation is deliberate: D105's CLRS run died with nothing saved because its
analysis and its banking were in the same 12-hour job.

THE PROMPTS. The `lenmatch` marker design, whose two arms are identical except for a
single trailing `A`/`B` token, so task identity and token count are decoupled by
construction rather than by statistical control -- the thing D84 could not do and
D87 established. Token count is VERIFIED per pair in-kernel, not asserted (D101
carried a length claim that was false).

GATES, all before any state is written:
  P1  Every A/B pair must agree in token count exactly. A pair that does not is
      skipped and reported, never silently banked.
  P2  The two markers must select DIFFERENT golds, else there is no task contrast.
  P3  Behavioural context is printed per (pair, marker) -- what fraction reach rank
      1 -- so a later reader knows whether the model was doing the task at all
      before they interpret its geometry (D118: a ladder at 0% measures nothing).
  P4  Incremental save after every prompt, and a wall-clock budget well inside
      Kaggle's 12 h. The banking is the deliverable; it must survive a timeout.
"""

import json
import os
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 64
N_ITEMS = 22          # per pair-type; 3 types x 2 markers x 22 = 132 orbits
SEQ_LEN = 8
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 9000   # DataSphere port; bank, then stop cleanly


def build_items(rng):
    """The lenmatch marker design: identical prompt, one trailing token selects the task."""
    pairs = {
        "count_vs_last": ("for task A, report how many 1s are in the sequence. "
                          "For task B, report the last symbol of the sequence.",
                          lambda s: (str(sum(1 for x in s if x == 1)), str(s[-1]))),
        "first_vs_last": ("for task A, report the first element of the sequence. "
                          "For task B, report the last element of the sequence.",
                          lambda s: (str(s[0]), str(s[-1]))),
        "max_vs_min": ("for task A, report the largest element of the sequence. "
                       "For task B, report the smallest element of the sequence.",
                       lambda s: (str(max(s)), str(min(s)))),
    }
    items = []
    for pair, (rules, golds) in pairs.items():
        # DEDUPE THE SEQUENCE, not just the (pair, item) key. `count_vs_last` draws
        # 8 binary digits -- 256 possibilities -- so 22 draws collide by birthday,
        # and a repeated sequence is a repeated PROMPT. The dry run found 132 items
        # carrying 130 distinct prompts. Same failure as D110(b) and D118.
        seen_seq = set()
        for i in range(N_ITEMS):
            for _ in range(200):
                seq = ([rng.randrange(2) for _ in range(SEQ_LEN)] if pair == "count_vs_last"
                       else [rng.randrange(10) for _ in range(SEQ_LEN)])
                if tuple(seq) not in seen_seq:
                    break
            seen_seq.add(tuple(seq))
            ga, gb = golds(seq)
            for marker, gold in (("A", ga), ("B", gb)):
                items.append({
                    "pair": pair, "marker": marker, "item": i, "gold": gold,
                    "seq": " ".join(map(str, seq)),
                    "prompt": (f"Rules: {rules}\nSequence: {' '.join(map(str, seq))}"
                               f"\nTask: {marker}"),
                })
    return items


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


def main():
    out_tar = (os.path.abspath(sys.argv[1]) if len(sys.argv) > 1
               else os.path.abspath("blockbank.tgz"))
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    run("pip install torch==2.5.1")
    run("pip install transformers==4.53.3 accelerate safetensors")
    import random

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    items = build_items(random.Random(20260809))
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

    # P1 + P2, before the model is even loaded.
    def enc_len(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]

    keep, skipped = [], []
    by_key = {}
    for it in items:
        by_key.setdefault((it["pair"], it["item"]), []).append(it)
    for key, two in sorted(by_key.items()):
        if len(two) != 2:
            skipped.append((key, "missing arm"))
            continue
        la, lb = enc_len(two[0]["prompt"]), enc_len(two[1]["prompt"])
        if la != lb:
            skipped.append((key, f"token counts {la} != {lb}"))
            continue
        if two[0]["gold"] == two[1]["gold"]:
            skipped.append((key, f"both markers gold {two[0]['gold']}"))
            continue
        for it in two:
            it["n_tokens"] = int(la)
            keep.append(it)
    print(f"P1/P2: keeping {len(keep)} orbits from {len(by_key)} pairs; "
          f"skipped {len(skipped)}", flush=True)
    for k, why in skipped[:10]:
        print(f"    skipped {k}: {why}", flush=True)
    if not keep:
        print("NOTHING passes the gates -- VOID.", flush=True)
        return

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

    blocks = list(model.transformer.core_block)
    print(f"hooking all {len(blocks)} core blocks", flush=True)

    manifest = []
    for n, it in enumerate(keep):
        if time.time() - t0 > WALL_BUDGET_S:
            print(f"  wall budget reached at {n}/{len(keep)} -- stopping cleanly, "
                  f"everything banked so far is saved", flush=True)
            break
        try:
            text = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt",
                      add_special_tokens=False).input_ids.to(model.device)
            n_p = ids.shape[1]
            freqs = model.freqs_cis[:, :n_p]
            g_ids = tok(it["gold"], add_special_tokens=False).input_ids
            per_block = [[] for _ in blocks]
            ranks = []
            handles = []

            # Everything the hook needs is bound as a default, not captured. The
            # hooks only fire inside this iteration today, but a captured loop
            # variable is how a later edit silently reads the NEXT prompt's freqs.
            n_blocks = len(blocks)

            def mk(bi, *, per_block=per_block, ranks=ranks, n_p=n_p,
                   freqs=freqs, g_ids=g_ids, last=n_blocks - 1):
                def hook(_m, _i, o):
                    with torch.no_grad():
                        per_block[bi].append(
                            o.detach()[0, n_p - 1, :].float().cpu().numpy())
                        if bi == last:
                            row = torch.log_softmax(
                                coda_head(o.detach(), freqs).float()[0, n_p - 1], dim=-1)
                            ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)
                return hook

            for bi, b in enumerate(blocks):
                b._forward_hooks.clear()
                handles.append(b.register_forward_hook(mk(bi)))
            try:
                with torch.no_grad():
                    model(input_ids=ids, num_steps=NUM_STEPS)
            finally:
                for h in handles:
                    h.remove()

            arr = np.stack([np.stack(pb) for pb in per_block], axis=1)  # [T, 4, hidden]
            tag = f"{it['pair']}_{it['marker']}_i{it['item']:02d}"
            np.save(os.path.join(OUTDIR, tag + ".npy"), arr.astype(np.float16))
            rec = {**{k: it[k] for k in
                      ("pair", "marker", "item", "gold", "seq", "prompt", "n_tokens")},
                   "tag": tag, "shape": list(arr.shape), "num_steps": NUM_STEPS,
                   "rank_curve": ranks, "best_rank": int(min(ranks)),
                   "correct": bool(min(ranks) == 1), "ok": True}
        except Exception as exc:  # noqa: BLE001
            rec = {**{k: it[k] for k in ("pair", "marker", "item")},
                   "ok": False, "why": f"{type(exc).__name__}: {exc}"}
            print(traceback.format_exc(), flush=True)
        manifest.append(rec)
        with open(os.path.join(OUTDIR, "manifest.json"), "w") as fh:
            json.dump(manifest, fh)
        if n % 20 == 0:
            print(f"  {n}/{len(keep)}  [{time.time() - t0:.0f}s]", flush=True)

    ok = [r for r in manifest if r.get("ok")]
    print(f"\n=== banked {len(ok)}/{len(manifest)} in {(time.time() - t0) / 60:.1f} min ===",
          flush=True)

    # P3: behavioural context, so a later reader knows if the model did the task.
    for pair in sorted({r["pair"] for r in ok}):
        for mk_ in ("A", "B"):
            sub = [r for r in ok if r["pair"] == pair and r["marker"] == mk_]
            if not sub:
                continue
            acc = sum(r["correct"] for r in sub) / len(sub)
            print(f"  {pair:>16} {mk_}: {len(sub)} orbits, gold reaches rank 1 in "
                  f"{acc:.0%}, median best rank "
                  f"{float(np.median([r['best_rank'] for r in sub])):.0f}", flush=True)
    nt = sorted({r["n_tokens"] for r in ok})
    print(f"  token counts banked: {nt}", flush=True)
    print("\nno analysis here by design -- A3, per-block TDA and the D98 replication "
          "are local scripts over this bank.", flush=True)
    with tarfile.open(out_tar, "w:gz") as tf:
        tf.add(OUTDIR, arcname="out")
    print(f"wrote {out_tar} ({os.path.getsize(out_tar) / 1e6:.1f} MB)", flush=True)
    print("DONE", flush=True)



if __name__ == "__main__":
    main()
