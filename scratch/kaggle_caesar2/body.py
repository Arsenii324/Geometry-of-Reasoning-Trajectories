"""Caesar, instrumented: per-character records with wrap, shift, prior and tokenisation.

WHAT THE SCREEN COULD NOT SEE
    `geometry-caesar-screen` reports one accuracy per cell. That is enough to decide
    whether to continue and nothing else. This run keeps **one record per output
    character**, so the covariates that plausibly drive the behaviour can be read
    off afterwards rather than guessed at. It is deliberately observational: the
    covariates below are free -- they are properties of data already being
    generated -- and this project has repeatedly been saved by a distribution it
    happened to have kept (D49) and burned by a number it had not measured (D41(3)).

FOUR AXES, each of which separates a different explanation

  1. WRAP-AROUND -- the sharp one.
     Decoding c -> p = (c - k) mod 26 requires the modular reduction exactly when
     the ciphertext letter index is < k. A model doing genuine modular arithmetic
     is indifferent to that; a model doing naive subtraction fails precisely there.
     The wrap fraction is k/26, so the shift sweep also sweeps how often it matters:
     k=1 -> 4% of letters wrap, k=13 -> 50%, k=25 -> 96%.
     **`char_acc | wrapped` vs `char_acc | not wrapped` is a direct test of whether
     the model computes the cipher or approximates it.**

  2. SHIFT VALUE (the "number of rotations").
     k=13 is ROT13, a *named* transform with abundant worked examples in any web
     corpus -- if accuracy spikes only there, the model is retrieving, not shifting.
     k=3 is the classical Caesar, also over-represented. k=1 and k=25 are adjacent
     to the identity in letter space but not in token space, and k=25 is a
     single-step *backward* shift disguised as a large forward one.

  3. MEMORISATION RISK -- three levels of plaintext.
     `classic` famous strings that plausibly appear verbatim in ROT13 examples;
     `fresh`   ordinary English unlikely to be a cipher exercise;
     `random`  random letter strings, which cannot be memorised or guessed from a
               language prior at all.
     If accuracy runs classic >> fresh >> random the model is leaning on prior and
     recall; if the three are comparable it is transforming the input. **Without the
     `random` arm, high accuracy on English is uninterpretable.**

  4. TOKENISATION.
     Space-separated letters are one token each, so the shift is a per-token map.
     Joined text puts several letters inside one token, so the same shift is a
     ragged many-to-many operation. This separates the cipher from the tokeniser.

ALSO RECORDED, because they cost nothing and have been needed before
    per-item: prompt token count, generated token count, whether generation stopped
    early; per-character: position in the word, the ciphertext character, both the
    gold and predicted character. Everything is written to `caesar_chars.json` so
    the analysis can be redone locally without a rerun.

The untrained arm runs identically. Its accuracy must be at chance or below; if any
cell exceeds that, the scorer is wrong, not the model.
"""
# @needs: run load_arm free_arm batched_generate

import json
import random
import string

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 32
MAX_NEW = 28
N_ITEMS = 10                       # per cell; 4 shifts x 3 priors x 2 tokenisations
SHIFTS = (1, 3, 13, 25)
ALPHA = string.ascii_lowercase

CLASSIC = ["the quick brown fox", "attack at dawn", "hello world",
           "to be or not to be", "all your base", "lorem ipsum dolor",
           "the cat sat on the mat", "mary had a little lamb",
           "four score and seven", "call me ishmael"]
FRESH = ["copper wire and glue", "the orange kettle hums", "eleven grey pebbles",
         "a ladder leans outside", "salt water in a jar", "the fence needs paint",
         "two clocks disagree", "warm bread on friday", "the gate swings wide",
         "counting sheep again"]


def rand_text(rng, n_words=4, wl=(4, 7)):
    return " ".join("".join(rng.choice(ALPHA) for _ in range(rng.randint(*wl)))
                    for _ in range(n_words))


def enc(text, k):
    return "".join(chr((ord(c) - 97 + k) % 26 + 97) if c in ALPHA else c for c in text)


def wraps(cipher_char, k):
    """True iff decoding this ciphertext letter needs the modular reduction."""
    return cipher_char in ALPHA and (ord(cipher_char) - 97) < k


def build():
    rng = random.Random(0)
    cells = []
    for k in SHIFTS:
        for prior, texts in (("classic", CLASSIC), ("fresh", FRESH),
                             ("random", [rand_text(rng) for _ in range(N_ITEMS)])):
            for tokmode in ("joined", "spaced"):
                items = []
                for t in texts[:N_ITEMS]:
                    ct = enc(t, k)
                    shown = " ".join(ct.replace(" ", "")) if tokmode == "spaced" else ct
                    gold = t.replace(" ", "") if tokmode == "spaced" else t
                    prompt = (f"Decode this Caesar cipher. Each letter was shifted "
                              f"forward by {k}. Shift each letter back by {k} to "
                              f"recover the original.\nCiphertext: {shown}\nPlaintext:")
                    items.append({"prompt": prompt, "gold": gold,
                                  "cipher": shown.replace(" ", "") if tokmode == "spaced" else ct})
                cells.append({"shift": k, "prior": prior, "tok": tokmode, "items": items})
    return cells


def norm(s):
    return "".join(c for c in s.lower() if c in ALPHA)


# NOTE: generation is delegated to `batched_generate`, which uses Huginn's own
# HuginnDynamicCache and generate_minimal. The hand-rolled loop that used to live
# here re-ran the full sequence through all 32 unrolls for every token at batch
# size 1: the screen run took 239 minutes, of which model load was 0.6. That is
# ~1 minute per 24-token completion, and it is why this file was rewritten.


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    cells = build()
    n_items = sum(len(c["items"]) for c in cells)
    print(f"{len(cells)} cells ({len(SHIFTS)} shifts x 3 priors x 2 tokenisations), "
          f"{n_items} items", flush=True)
    print(f"  wrap fraction by shift: "
          f"{ {k: round(k / 26, 3) for k in SHIFTS} }", flush=True)

    chars, items_out = [], []
    for arm in ("trained", "untrained"):
        print(f"\n=== {arm} ===", flush=True)
        model = None
        try:
            model = load_arm(None if arm == "untrained" else MODEL_ID, cfg,
                             0 if arm == "untrained" else REVISION)
            for ci, cell in enumerate(cells):
                ex, ca = [], []
                preds = batched_generate(model, tok, [it["prompt"] for it in cell["items"]],
                                         max_new=MAX_NEW, num_steps=NUM_STEPS,
                                         verbose=(ci == 0))
                for ii, it in enumerate(cell["items"]):
                    pred = preds[ii].split("\n")[0].strip()
                    n_prompt, n_gen, stopped = 0, 0, 0
                    g, p, c = norm(it["gold"]), norm(pred), norm(it["cipher"])
                    ex.append(float(p == g))
                    hits = 0
                    for pos in range(len(g)):
                        pc = p[pos] if pos < len(p) else ""
                        cc = c[pos] if pos < len(c) else ""
                        ok = int(pc == g[pos])
                        hits += ok
                        chars.append({"arm": arm, "shift": cell["shift"],
                                      "prior": cell["prior"], "tok": cell["tok"],
                                      "item": ii, "pos": pos, "gold": g[pos],
                                      "pred": pc, "cipher": cc, "correct": ok,
                                      "wrapped": int(wraps(cc, cell["shift"]))})
                    ca.append(hits / max(len(g), 1))
                    items_out.append({"arm": arm, "shift": cell["shift"],
                                      "prior": cell["prior"], "tok": cell["tok"],
                                      "item": ii, "exact": ex[-1], "char_acc": ca[-1],
                                      "n_prompt_tok": n_prompt, "n_gen_tok": n_gen,
                                      "stopped": int(stopped), "pred": pred,
                                      "gold": it["gold"]})
                print(f"  [{ci + 1:>2}/{len(cells)}] k={cell['shift']:>2} "
                      f"{cell['prior']:>7} {cell['tok']:>6}: exact {np.mean(ex):>6.1%} "
                      f"char {np.mean(ca):>6.1%}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        finally:
            free_arm(model, None if arm == "untrained" else MODEL_ID)
            with open("caesar_chars.json", "w") as f:
                json.dump({"chars": chars, "items": items_out}, f)

    # --- the observations this run exists to make ---
    import collections
    print("\n=== char accuracy by WRAP (does the model do modular arithmetic?) ===")
    for arm in ("trained", "untrained"):
        for k in SHIFTS:
            sel = [c for c in chars if c["arm"] == arm and c["shift"] == k]
            w = [c["correct"] for c in sel if c["wrapped"]]
            nw = [c["correct"] for c in sel if not c["wrapped"]]
            if w and nw:
                print(f"  {arm:>9} k={k:>2}: wrapped {np.mean(w):>6.1%} (n={len(w):>4})  "
                      f"not-wrapped {np.mean(nw):>6.1%} (n={len(nw):>4})  "
                      f"gap {np.mean(nw) - np.mean(w):>+7.1%}")
    print("\n=== char accuracy by PRIOR (memorisation vs computation) ===")
    for arm in ("trained", "untrained"):
        row = collections.defaultdict(list)
        for c in chars:
            if c["arm"] == arm:
                row[c["prior"]].append(c["correct"])
        print(f"  {arm:>9}: " + "  ".join(
            f"{p} {np.mean(v):>6.1%}" for p, v in sorted(row.items())))
    print("\n=== char accuracy by SHIFT x TOKENISATION ===")
    for arm in ("trained", "untrained"):
        for tm in ("joined", "spaced"):
            cells_acc = []
            for k in SHIFTS:
                v = [c["correct"] for c in chars
                     if c["arm"] == arm and c["shift"] == k and c["tok"] == tm]
                cells_acc.append(f"k={k}: {np.mean(v):>6.1%}" if v else f"k={k}:   n/a")
            print(f"  {arm:>9} {tm:>6}  " + "  ".join(cells_acc))
    print("\n  Full per-character records in caesar_chars.json; analyse locally.")


main()
