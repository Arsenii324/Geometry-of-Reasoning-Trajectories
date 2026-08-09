"""A22: WHICH nouns select the rotation? Turning D132's two-way switch into a mapping.

D132 established, fully controlled, that a single noun in the instruction switches the
recurrent map between settling and a damped period-6 rotation:

    "report the largest SYMBOL of the sequence"   -> rotates 36/36
    "report the largest ELEMENT of the sequence"  -> rotates  0/36
    "report the largest ITEM of the sequence"     -> rotates  0/36

at an identical 53 tokens, on the SAME 18 sequences, flipping in 36 of 36
within-sequence cells. `item` behaving like `element` is what rules out a two-word
coincidence: `symbol` is the special one.

But three nouns is a switch, not a mapping, and it cannot say WHY. This sweeps the same
slot over many nouns so the question becomes answerable: is the selector lexical
(which word), semantic (what the word denotes), or tokenisation (how it splits)?

THE NOUNS, chosen to separate those three, not at random:
  * near-synonyms of `symbol` in this context -- `character`, `digit`, `token`, `sign`,
    `glyph`, `letter`, `numeral`. If the effect is SEMANTIC, these follow `symbol`.
  * near-synonyms of `element` -- `entry`, `member`, `value`, `component`, `unit`,
    `term`, `slot`. If SEMANTIC, these follow `element`.
  * words matched on FORM rather than meaning -- `symptom`, `cymbal`, `symmetry` share
    a prefix or sound with `symbol` but denote nothing like it. If the effect tracks
    these, it is lexical/embedding-level rather than semantic.
  * `symbol` and `element` themselves, as internal replications of D132.

PRE-REGISTERED.
  P1  PRIMARY, descriptive: the rotation fraction per noun. This is a mapping, so the
      result is the partition itself, not a p-value.
  P2  THE DISCRIMINATOR. If the semantic neighbours of `symbol` rotate and the
      form-matched controls (`symptom`, `cymbal`, `symmetry`) do not, the selector is
      semantic. If the form-matched controls rotate too, it is lexical/embedding.
      If neither group is coherent, it is idiosyncratic to individual tokens -- which
      would itself be the finding, and the one that most constrains a mechanism.
  P3  TOKEN GATE. Nouns differ in token length, so n_tokens will NOT be constant here.
      It is recorded per orbit and reported per noun, and any partition that lines up
      with token count is reported as length-confounded rather than lexical. This is
      the one control D132 had by construction and this design cannot have.
  P4  REPLICATION. `symbol` and `element` are included; if they do not reproduce
      D132's 100% / 0% the run is VOID and nothing else is read.
  P5  UNIT is the (sequence, noun) cell; every noun sees the SAME 12 sequences.
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
    # D132 replications (P4)
    "symbol", "element",
    # semantic neighbours of `symbol`
    "character", "digit", "token", "sign", "glyph", "numeral",
    # semantic neighbours of `element`
    "entry", "member", "value", "component", "term",
    # FORM-matched controls: share form with `symbol`, denote nothing like it (P2)
    "symptom", "cymbal", "symmetry",
)
N_SEQ = 12                              # 12 seqs x 16 nouns x 2 markers = 384 orbits
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
