"""Given Huginn's actual generations, which EXTRACTION RULE recovers the answer best?

TWO DIFFERENT QUESTIONS, AND THIS IS THE ONE THAT NEEDS NO GPU. "How do we get Huginn to
produce a parseable answer?" is about the prompt, and A43 is on GPU testing five formats. "Given
the text it already produced, how should we read an answer out of it?" is about the scorer, and
every input it needs is already on disk: **~1420 banked generations with their golds** across
`kaggle_depthacc` (1260), `ds_genscore` (128) and `ds_regimeout` (36).

WHY IT MATTERS. D172 measured first-token exact-match at **0.125** where the answer is present
in **0.781** of generations. The whole gap is a reading problem, and this project has been
reading with `output.strip() == gold` -- the strictest rule available -- since the beginning.

WHY IT IS DANGEROUS, AND WHAT GUARDS IT. A looser rule scores higher on everything, including
noise. `'The answer is 4 - 1 = 3'` contains `3`, and so does a generation about a different
item whose gold is 3. **So every rule is scored twice**: against the item's own gold, and
against the OTHER items' golds in the same family. The second is the rule's false-positive
rate, and a rule's value is the **lift** between them, never its raw accuracy. D145 is the
cautionary case -- a reimplemented `contains` disagreed with the banked one by up to 13 points
and the difference was invisible until both were computed on identical inputs (RC4).

THE RULES. Ordered from strictest to loosest, so the accuracy/false-positive trade is visible
as a curve rather than a verdict:

    exact            output.strip() == gold                      -- what this project has used
    exact_stripped   after removing glued role markers (D145)
    first_token      the first generated token
    first_line       first line, trailing period stripped
    last_line        last line
    boxed            contents of \\boxed{...}
    after_marker     text after an "Answer:"/"Answer =" marker
    first_number     first number-like token in the output
    last_number      last number-like token -- the GSM8K convention
    last_word        final whitespace-delimited token
    contains         gold appears at a word boundary anywhere    -- the loosest

    uv run python scripts/parser_sweep.py
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# role markers Huginn glues to short answers; from D145's post-mortem of kaggle_depthacc
MARKERS = ("user", "Huginn", "assistant", "system")


def load():
    """(family, gold, output) triples from every local bank that stores generations."""
    out = []
    p = ROOT / "scratch/kaggle_depthacc/out/depthacc.json"
    if p.exists():
        d = json.loads(p.read_text())
        for r in (d["rows"] if isinstance(d, dict) else d):
            out.append((f"depthacc:{r['family']}", str(r["gold"]), r["output"]))
    p = ROOT / "scratch/ds_genscore/genscore.json"
    if p.exists():
        for r in json.loads(p.read_text())["rows"]:
            if r.get("ok"):
                out.append((f"genscore:{r['family']}", str(r["gold"]), r["gen"]))
    p = ROOT / "scratch/ds_regimeout/regimeout.json"
    if p.exists():
        for r in json.loads(p.read_text())["rows"]:
            if r.get("ok"):
                for side in ("set", "rot"):
                    out.append((f"regimeout:{r['chord']}", str(r["gold"]), r[side]["gen"]))
    return out


def strip_markers(text: str) -> str:
    t = text.strip()
    for m in MARKERS:
        if t.endswith(m):
            t = t[: -len(m)]
    return t.strip()


NUM = re.compile(r"-?\d+(?:\.\d+)?")


def rules(text: str) -> dict[str, str | None]:
    t = text.strip()
    lines = [x for x in t.split("\n") if x.strip()]
    nums = NUM.findall(t)
    boxed = re.search(r"\\boxed\{([^}]*)\}", t)
    marker = re.search(r"(?i)answer\s*[:=]\s*(.+?)(?:\n|$)", t)
    return {
        "exact": t,
        "exact_stripped": strip_markers(t),
        "first_line": lines[0].strip().rstrip(".").strip() if lines else None,
        "last_line": lines[-1].strip().rstrip(".").strip() if lines else None,
        "boxed": boxed.group(1).strip() if boxed else None,
        "after_marker": marker.group(1).strip().rstrip(".").strip() if marker else None,
        "first_number": nums[0] if nums else None,
        "last_number": nums[-1] if nums else None,
        "last_word": t.split()[-1].strip(".,\"'") if t.split() else None,
    }


def contains(text: str, gold: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(gold)}(?![A-Za-z0-9])", text,
                          re.IGNORECASE))


def main() -> int:
    data = load()
    if not data:
        print("no banked generations found")
        return 1
    fams = collections.defaultdict(set)
    for fam, gold, _ in data:
        fams[fam].add(gold)
    print(f"{len(data)} banked generations over {len(fams)} family-sources\n")

    names = list(rules("x").keys()) + ["contains"]
    hit = collections.Counter()
    fp_hit = collections.Counter()
    fp_n = collections.Counter()

    for fam, gold, text in data:
        ext = rules(text)
        others = fams[fam] - {gold}
        for nm in names:
            if nm == "contains":
                if contains(text, gold):
                    hit[nm] += 1
                for g in others:
                    fp_n[nm] += 1
                    if contains(text, g):
                        fp_hit[nm] += 1
            else:
                v = ext[nm]
                if v is not None and v.strip().lower() == gold.strip().lower():
                    hit[nm] += 1
                for g in others:
                    fp_n[nm] += 1
                    if v is not None and v.strip().lower() == g.strip().lower():
                        fp_hit[nm] += 1

    n = len(data)
    print(f"{'rule':16s} {'accuracy':>9s} {'false-pos':>10s} {'LIFT':>7s}   "
          f"(n={n}; false-pos = same rule scored against OTHER items' golds)")
    print("-" * 72)
    rows = []
    for nm in names:
        acc = hit[nm] / n
        fp = fp_hit[nm] / fp_n[nm] if fp_n[nm] else 0.0
        rows.append((nm, acc, fp, acc - fp))
    for nm, acc, fp, lift in sorted(rows, key=lambda r: -r[3]):
        print(f"{nm:16s} {acc:9.3f} {fp:10.3f} {lift:+7.3f}")

    print("\nThe project has been scoring with `exact`. Best lift:")
    best = max(rows, key=lambda r: r[3])
    base = next(r for r in rows if r[0] == "exact")
    print(f"  {best[0]} at accuracy {best[1]:.3f} (lift {best[3]:+.3f}) against "
          f"exact's {base[1]:.3f} (lift {base[3]:+.3f}) "
          f"-- {best[1] - base[1]:+.3f} accuracy for {best[2] - base[2]:+.3f} false positives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
