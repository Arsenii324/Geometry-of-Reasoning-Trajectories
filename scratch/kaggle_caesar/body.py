"""Caesar cipher SCREEN: does the trained model succeed anywhere, and the untrained nowhere?

WHY THIS RUNS FIRST, AND WHAT IT IS ALLOWED TO CONCLUDE
    Every content-level finding in this project was deflated by an untrained
    control (D40, D41, D48, D53). The one claim that would have made D41 sharp --
    "decodability and capability move in opposite directions" -- had to be
    RETRACTED because the counting task has no capability contrast to compare
    against: trained 1/60 and 0/120 at M=64, untrained 0/60, Fisher p=1.0
    (D41(3), D47). Half this project's claims are missing the same ingredient: a
    task the trained model can measurably do and random weights cannot.

    This kernel does NOT test any hypothesis about depth or geometry. It answers
    one screening question -- is there a cell where trained accuracy is above
    zero? -- because if there is not, the Caesar family is a counting-task repeat
    and should be dropped rather than elaborated (directions.md B1).

SIX CELLS, chosen so they differ in the KIND of work required, not just difficulty
    given_rot13_word    shift stated, ROT13, one word        -- easiest possible
    given_rot13_sent    shift stated, ROT13, a sentence      -- length effect
    given_s3_word       shift stated, shift=3, one word      -- is ROT13 special?
    infer_rot13_sent    shift NOT stated, must be found      -- search over 25
    named_only_s3       "Caesar cipher", no instruction      -- task identification
    spaced_s3_word      letters space-separated              -- tokenisation burden

TWO SCORES, because exact match alone is uninformative near zero (observability
D3: report the distribution, not the binary). A model recovering 80% of characters
is doing the task and failing to finish; exact-match would record that as 0 and the
family would be wrongly dropped.
    exact       normalised string equality
    char_acc    per-position character agreement over the shorter length

PRE-REGISTERED, so the screen can fail (directions.md B1):
    1. trained exact > 0 on at least given_rot13_word; untrained exact = 0 in every
       cell. If untrained is ever non-zero the scorer is broken, not the model.
    2. given_* >= infer_* -- stating the shift cannot make the task harder.
    3. char_acc for untrained ~ chance (~1/26 for letters), not 0, which is the
       check that the scorer is measuring anything at all.

Answers are generated greedily and compared to the known plaintext. `num_steps=32`
is Huginn's `mean_recurrence`; depth is NOT varied here -- that is the follow-up if
and only if this screen passes.
"""
# @needs: run load_arm free_arm

import json
import random
import string

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 32
MAX_NEW = 24
N_ITEMS = 20

WORDS = ["banana", "orange", "puzzle", "rocket", "silver", "meadow", "candle",
         "forest", "kitten", "planet", "guitar", "window", "yellow", "dragon",
         "pillow", "market", "summer", "castle", "bridge", "wizard"]
SENTS = ["the cat sat on the mat", "we will meet at dawn", "send help to the tower",
         "the red door is open", "bring water and bread", "she found the old map",
         "birds fly over the lake", "he lost his blue hat", "the ship sails today",
         "keep this message safe", "the key is under stone", "meet me by the river",
         "silence fell on the town", "they marched through snow", "a light in the window",
         "the road bends to left", "count the golden coins", "wind blows from north",
         "the door opens at nine", "follow the narrow path"]


def shift_text(text, k):
    out = []
    for ch in text:
        if ch in string.ascii_lowercase:
            out.append(chr((ord(ch) - 97 + k) % 26 + 97))
        else:
            out.append(ch)
    return "".join(out)


def cells():
    """(cell_name, [(prompt, expected_plaintext), ...])."""
    rng = random.Random(0)
    out = {}

    def mk(name, texts, k, template, spaced=False):
        items = []
        for t in texts[:N_ITEMS]:
            ct = shift_text(t, k)
            if spaced:
                ct = " ".join(ct.replace(" ", ""))
            items.append((template.format(ct=ct, k=k), t.replace(" ", "") if spaced else t))
        out[name] = items

    mk("given_rot13_word", WORDS, 13,
       "Decode this Caesar cipher. Each letter was shifted forward by {k}. "
       "Shift each letter back by {k} to recover the original.\nCiphertext: {ct}\nPlaintext:")
    mk("given_rot13_sent", SENTS, 13,
       "Decode this Caesar cipher. Each letter was shifted forward by {k}. "
       "Shift each letter back by {k} to recover the original.\nCiphertext: {ct}\nPlaintext:")
    mk("given_s3_word", WORDS, 3,
       "Decode this Caesar cipher. Each letter was shifted forward by {k}. "
       "Shift each letter back by {k} to recover the original.\nCiphertext: {ct}\nPlaintext:")
    mk("infer_rot13_sent", SENTS, 13,
       "The following text was encrypted with a Caesar cipher using an unknown "
       "shift. Work out the shift and decode it.\nCiphertext: {ct}\nPlaintext:")
    mk("named_only_s3", WORDS, 3, "Caesar cipher: {ct}\nPlaintext:")
    mk("spaced_s3_word", WORDS, 3,
       "Decode this Caesar cipher. Each letter was shifted forward by {k}. "
       "Shift each letter back by {k}.\nCiphertext: {ct}\nPlaintext:", spaced=True)
    _ = rng
    return out


def norm(s):
    return "".join(c for c in s.lower() if c in string.ascii_lowercase + " ").strip()


def char_acc(pred, gold):
    p, g = norm(pred).replace(" ", ""), norm(gold).replace(" ", "")
    if not g:
        return 0.0
    n = min(len(p), len(g))
    return sum(1 for i in range(n) if p[i] == g[i]) / len(g)


def generate(model, tok, prompt, max_new=MAX_NEW):
    import torch
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    gen = []
    for _ in range(max_new):
        with torch.no_grad():
            out = model(input_ids=ids, num_steps=NUM_STEPS)
        logits = out.logits if hasattr(out, "logits") else out[0]
        nxt = int(logits[0, -1].argmax())
        if nxt == getattr(tok, "eos_token_id", -1):
            break
        gen.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        if "\n" in tok.decode(gen):
            break
    del ids
    torch.cuda.empty_cache()
    return tok.decode(gen).split("\n")[0].strip()


def main():
    run("pip install -q 'transformers>=4.50,<4.54'")
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    data = cells()
    print(f"{len(data)} cells x {N_ITEMS} items", flush=True)
    print("  example:", repr(data["given_rot13_word"][0][0][-60:]), "->",
          repr(data["given_rot13_word"][0][1]), flush=True)

    results = {}
    for arm in ("trained", "untrained"):
        print(f"\n=== {arm} ===", flush=True)
        model = None
        try:
            model = load_arm(None if arm == "untrained" else MODEL_ID, cfg,
                             0 if arm == "untrained" else REVISION)
            results[arm] = {}
            for name, items in data.items():
                ex, ca, samples = [], [], []
                for prompt, gold in items:
                    pred = generate(model, tok, prompt)
                    ex.append(float(norm(pred) == norm(gold)))
                    ca.append(char_acc(pred, gold))
                    if len(samples) < 3:
                        samples.append({"gold": gold, "pred": pred})
                results[arm][name] = {"exact": float(np.mean(ex)),
                                      "char_acc": float(np.mean(ca)),
                                      "n": len(ex), "samples": samples}
                print(f"  {name:>18}: exact {np.mean(ex):>6.1%}  "
                      f"char_acc {np.mean(ca):>6.1%}   e.g. {samples[0]['gold']!r} -> "
                      f"{samples[0]['pred']!r}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            results[arm] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            free_arm(model, None if arm == "untrained" else MODEL_ID)
            with open("caesar_screen.json", "w") as f:
                json.dump(results, f, indent=1)

    print("\n=== SCREEN VERDICT ===")
    t = results.get("trained", {})
    u = results.get("untrained", {})
    print(f"  {'cell':>18} {'trained exact':>14} {'untr exact':>11} "
          f"{'trained char':>13} {'untr char':>10}")
    for name in data:
        te, ue = t.get(name, {}), u.get(name, {})
        print(f"  {name:>18} {te.get('exact', float('nan')):>13.1%} "
              f"{ue.get('exact', float('nan')):>10.1%} "
              f"{te.get('char_acc', float('nan')):>12.1%} "
              f"{ue.get('char_acc', float('nan')):>9.1%}")
    best = max((v.get("exact", 0) for v in t.values() if isinstance(v, dict)), default=0)
    worst_u = max((v.get("exact", 0) for v in u.values() if isinstance(v, dict)), default=0)
    print(f"\n  best trained exact-match: {best:.1%}")
    print(f"  best untrained exact-match: {worst_u:.1%}  (must be 0.0%)")
    if best > 0 and worst_u == 0:
        print("  => CAPABILITY CONTRAST EXISTS. This is the ingredient the counting")
        print("     task could not supply (D41(3)). Proceed to the depth variants.")
    elif best == 0:
        print("  => trained accuracy is zero everywhere. Per directions.md B1 this")
        print("     family is a counting-task repeat and should be DROPPED, not")
        print("     elaborated -- unless char_acc shows partial competence.")
    else:
        print("  => untrained scored above zero: the SCORER is broken, not the model.")


main()
