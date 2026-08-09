"""A23: is the regime-selecting property linear in the TOKEN EMBEDDING?

D132 showed one noun flips the recurrent map between settling and a damped period-6
rotation, with computation, answer, token count and sequences all held fixed. D134 then
refuted every candidate explanation: `symbol`'s semantic neighbours rotate 0/144,
`element`'s rotate 96/120, and the FORM-matched controls rotate 72/72; token count
predicts nothing (50% within the 53-token group) and neither does subword count
(`cymbal` splits 3 ways and rotates, `glyph` splits 2 ways and does not).

By elimination the property lives in the token's learned EMBEDDING -- which is exactly
the object D111/D113 identified as the map's parameter, since `e` is the prelude's
output and the prelude's input is that embedding.

WHY THIS RUN IS CHEAP, AND IT IS D134 THAT MAKES IT SO. Rotation is DETERMINISTIC given
the noun: every noun tested so far is 24/24 or 0/24, never mixed, across different
sequences, both markers and unseeded h_0. So the label costs ~2 orbits per noun, not 24.
That converts the budget from "more orbits per noun" into "MORE NOUNS", which is exactly
what the first pass lacked -- D134(3b) got leave-one-out 0.250 against a 0.318 null on
16 nouns, which has almost no power at 8 per class.

WHAT THE FIRST PASS GOT WRONG, AND THIS FIXES. It tested the early STATE, which is the
prelude's NONLINEAR output of the embedding, so a direction linear in the embedding
need not be linear there. This run reads the **embedding matrix rows directly**
(`model.transformer.wte`) alongside the label, so the hypothesis is tested on its own
object.

PRE-REGISTERED.
  P1  PRIMARY. Leave-one-out accuracy of a linear classifier on the raw embedding
      beats a permuted-label null. With ~48 nouns and a balanced split this has real
      power, unlike the 16-noun first pass.
  P2  DIMENSIONALITY. If P1 holds, how many principal components of the embedding
      differences carry it? A 1-2 dimensional answer is a mechanism; a full-rank one is
      a restatement of "the tokens differ".
  P3  DETERMINISM GATE, and the run is VOID without it. Each noun gets 2 sequences x 2
      markers = 4 orbits. If ANY noun is mixed (some orbits rotate, some not), the
      "label per noun" premise fails and no separability claim may be read.
  P4  REPLICATION. `symbol`, `element`, `symptom`, `cymbal` and `token` are included;
      they must reproduce D134's labels (rotate, not, rotate, rotate, not).
  P5  The embedding is saved per noun so the analysis is re-runnable locally without
      re-reserving a GPU.
"""

import json
import os
import subprocess
import sys
import tarfile
import time
import traceback

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 64
WORDS = (
    # D134 replications (P4)
    "symbol", "element", "symptom", "cymbal", "token",
    # a wide net -- the point is MANY labels, not a curated hypothesis
    "character", "digit", "sign", "glyph", "numeral", "letter", "mark", "figure",
    "entry", "member", "value", "component", "term", "slot", "field", "cell",
    "number", "integer", "unit", "piece", "part", "atom", "node", "point",
    "word", "string", "code", "label", "tag", "key", "index", "position",
    "object", "thing", "chunk", "block", "group", "set", "list", "array",
    "signal", "pattern", "shape", "form", "image", "sample",
)
N_SEQ = 2                               # rotation is DETERMINISTIC per noun (D134), so
#                                         2 seqs x 2 markers = 4 orbits each is enough
SEQ_LEN = 8                      # fixed, so token count is constant
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 9000   # DataSphere port; bank, then stop cleanly


def build_items(rng):
    """The SAME sequences under three wordings. Only the noun changes."""
    items = []
    seen = set()
    seqs = []
    while len(seqs) < N_SEQ:
        s = [rng.randrange(10) for _ in range(SEQ_LEN)]
        if tuple(s) in seen:
            continue
        seen.add(tuple(s))
        seqs.append(s)
    for si, seq in enumerate(seqs):
        for word in WORDS:
            rules = (f"for task A, report the largest {word} of the sequence. "
                     f"For task B, report the smallest {word} of the sequence.")
            for mk in ("A", "B"):
                gold = str(max(seq)) if mk == "A" else str(min(seq))
                items.append({
                    "pair": word, "marker": mk, "item": si, "word": word,
                    "gold": gold, "seq": " ".join(map(str, seq)),
                    "prompt": (f"Rules: {rules}\nSequence: {' '.join(map(str, seq))}"
                               f"\nTask: {mk}"),
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

    # THIRD-PARTY IMPORTS MUST NOT BE AT MODULE LEVEL for a DataSphere job:
    # the CLI IMPORTS this file locally to analyse dependencies, and the
    # pipx venv has no numpy. This does not contradict D125's "imports at
    # the top" rule -- that is about position INSIDE main(), after the
    # install, where a missing REPO symbol fails in 30 s rather than hours.
    import random

    import numpy as np
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

    # P5: save the EMBEDDING ROW for each noun -- the object D134 leaves as the only
    # surviving candidate. The first pass (D134(3b)) tested the early STATE, which is
    # the prelude's NONLINEAR output of this, and failed on that proxy. Saved here so
    # the analysis re-runs locally without re-reserving a GPU.
    emb = model.transformer.wte.weight.detach()
    noun_emb = {}
    for w in WORDS:
        ids = tok(f" {w}", add_special_tokens=False).input_ids
        noun_emb[w] = {"ids": list(map(int, ids)),
                       "n_sub": len(ids),
                       "emb": emb[ids].mean(0).float().cpu().numpy().tolist()}
    with open(os.path.join(OUTDIR, "noun_embeddings.json"), "w") as fh:
        json.dump(noun_emb, fh)
    print(f"saved embeddings for {len(noun_emb)} nouns, dim {len(noun_emb[WORDS[0]]['emb'])}; "
          f"subword counts {sorted({v['n_sub'] for v in noun_emb.values()})}", flush=True)


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
