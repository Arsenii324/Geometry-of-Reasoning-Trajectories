"""Every ARC-Easy measurement this project has, in one table, for comparison against the paper.

WHY IT EXISTS. Huginn's published ARC-Easy figure is **0.699 at r = 32**, and this project spent
three runs and two ledger rows failing to reproduce it before D169 and D175 found two routes
that do. The numbers are spread across `ds_arcproto` (A30), `ds_arcshots` (A34) and
`ds_archarness` (A40) with different scoring rules, prompt formats, depths and shot counts, and
the comparison that matters -- ours against theirs -- needs all of them side by side.

It regenerates from the banked JSON rather than being a copied table, so it cannot go stale.

VOID ARM, EXCLUDED BY NAME: A40's `T_chat` reimplementation of D162 put the leading space in
the prompt instead of the continuation, which misaligns the scored span in BPE (D175). Its
numbers are not shown.

    uv run python scripts/arc_table.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLISHED = 0.699


def collect():
    rows = []
    p = ROOT / "scratch/ds_arcproto/arcproto.json"
    if p.exists():
        ok = [r for r in json.loads(p.read_text())["rows"] if r.get("ok", True)]
        for dep in sorted({r["depth"] for r in ok}):
            v = [r for r in ok if r["depth"] == dep]
            for k, lbl in (("letter_correct", "letter-argmax"),
                           ("text_raw_correct", "option-text raw"),
                           ("text_norm_correct", "option-text /token")):
                x = [r[k] for r in v if k in r]
                if x:
                    rows.append(("A30 arcproto", dep, 0, lbl, "chat + options listed",
                                 sum(x) / len(x), len(x)))
    p = ROOT / "scratch/ds_arcshots/arcshots.json"
    if p.exists():
        ok = [r for r in json.loads(p.read_text())["rows"] if r.get("ok", True)]
        for sh in sorted({r["shots"] for r in ok}):
            v = [r for r in ok if r["shots"] == sh]
            for k, lbl in (("letter_correct", "letter-argmax"),
                           ("text_raw_correct", "option-text raw"),
                           ("text_norm_correct", "option-text /token")):
                x = [r[k] for r in v if k in r]
                if x:
                    rows.append(("A34 arcshots", 32, sh, lbl, "chat + options listed",
                                 sum(x) / len(x), len(x)))
    p = ROOT / "scratch/ds_archarness/archarness.json"
    if p.exists():
        ok = [r for r in json.loads(p.read_text())["rows"] if r.get("ok")]
        names = {"L_chat": ("letter-argmax", "chat + options listed"),
                 "H_chat_raw": ("option-text raw", "chat + harness prompt"),
                 "H_chat_tok": ("option-text /token", "chat + harness prompt"),
                 "H_chat_chr": ("option-text /char", "chat + harness prompt"),
                 "H_bare_raw": ("option-text raw", "BARE harness prompt"),
                 "H_bare_tok": ("option-text /token", "BARE harness prompt"),
                 "H_bare_chr": ("option-text /char", "BARE harness prompt")}
        for a, (lbl, fmt) in names.items():
            x = [r[a] for r in ok if a in r]
            if x:
                rows.append(("A40 archarness", 32, 0, lbl, fmt, sum(x) / len(x), len(x)))
    return rows


def main() -> int:
    rows = collect()
    if not rows:
        print("no banked ARC runs found")
        return 1
    print(f"{'run':16s} {'r':>3s} {'shots':>5s} {'scoring':18s} {'prompt format':24s} "
          f"{'acc':>6s} {'n':>4s}   d(pub)")
    print("-" * 100)
    for run, dep, sh, lbl, fmt, acc, n in sorted(rows, key=lambda x: -x[5]):
        flag = "  <== within 0.10" if abs(acc - PUBLISHED) <= 0.10 else ""
        print(f"{run:16s} {dep:3d} {sh:5d} {lbl:18s} {fmt:24s} {acc:6.3f} {n:4d}  "
              f"{acc - PUBLISHED:+.3f}{flag}")

    zero_listed = [r[5] for r in rows if r[2] == 0 and r[4] == "chat + options listed"]
    zero_noopts = [r[5] for r in rows if r[2] == 0 and r[4] != "chat + options listed"]
    shots = [r[5] for r in rows if r[2] > 0 and r[3] == "letter-argmax"]
    print(f"\nPUBLISHED {PUBLISHED} at r = 32.")
    print(f"  zero-shot WITH the option list in the prompt : "
          f"{min(zero_listed):.3f}-{max(zero_listed):.3f}  (n={len(zero_listed)} arms)")
    print(f"  zero-shot WITHOUT it                         : "
          f"{min(zero_noopts):.3f}-{max(zero_noopts):.3f}  (n={len(zero_noopts)} arms)")
    if shots:
        print(f"  few-shot letter-argmax (options listed)      : "
              f"{min(shots):.3f}-{max(shots):.3f}  (n={len(shots)} arms)")
    print("\nVOID and excluded: A40's T_chat arm (BPE span misalignment, D175).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
