"""Validate `docs/claims_ledger.md`, and report the next free claim id.

Deliberately does NOT generate rows. Every row this project has added needed its
own insertion point, supersession wording and cross-references; a generator would
have produced worse prose than writing it by hand. What is worth automating is the
part a human gets wrong silently:

  * a duplicate id (two D41s would make every citation ambiguous)
  * a row missing one of the four columns, so the evidence or method cell is empty
  * a citation to a claim that does not exist -- typos in "D34" are invisible
  * a claim marked RETRACTED or SUPERSEDED with no pointer to what replaced it
  * a superseding row that does not exist

Run:  python -m scripts.ledger          validate, exit non-zero on error
      python -m scripts.ledger --next   print the next free id
      python -m scripts.ledger --list   id, first 70 chars of the headline
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "claims_ledger.md")
ROW = re.compile(r"^\| (D\d+) \|(.*)$", re.M)
FLAG = re.compile(r"\b(RETRACTED|SUPERSEDED|WITHDRAWN)\b", re.I)


def rows(path: str = LEDGER) -> list[tuple[str, str]]:
    text = open(path, encoding="utf-8").read()
    return [(m.group(1), m.group(2)) for m in ROW.finditer(text)]


def validate(path: str = LEDGER) -> list[str]:
    """Return a list of problems; empty means clean."""
    if not os.path.exists(path):
        return [f"{path} not found"]
    found = rows(path)
    problems: list[str] = []

    seen: dict[str, int] = {}
    for i, (cid, _) in enumerate(found):
        if cid in seen:
            problems.append(f"{cid}: duplicate id (rows {seen[cid]} and {i})")
        seen[cid] = i

    ids = set(seen)
    for cid, body in found:
        # a full row is | id | claim | evidence | verification |
        if body.count(" |") < 2 and body.count("|") < 3:
            problems.append(f"{cid}: fewer than 4 columns -- a cell is missing")
        for cell in [c for c in body.split(" | ")][:3]:
            if not cell.strip(" |"):
                problems.append(f"{cid}: an empty cell")
                break
        # A row-level retraction is declared in the HEADLINE cell. Prose elsewhere
        # in the row often withdraws a sub-claim, which is not the same thing and
        # must not be flagged (D7 does exactly this).
        headline = body.split(" | ")[0]
        if FLAG.search(headline) and not (
            re.search(r"\bD\d+\b", headline) or "docs/" in headline
            or "scripts/" in headline or "see (" in headline.lower()
        ):
            problems.append(f"{cid}: headline flagged "
                            f"{FLAG.search(headline).group(1)} but points to no "
                            "replacement claim, section or file")
        for ref in set(re.findall(r"\bD(\d+)\b", body)):
            rid = f"D{ref}"
            if rid not in ids and rid != cid:
                problems.append(f"{cid}: cites {rid}, which does not exist")
    return problems


def next_id(path: str = LEDGER) -> str:
    found = rows(path)
    return f"D{max((int(c[1:]) for c, _ in found), default=0) + 1}"


def main(argv: list[str]) -> int:
    if "--next" in argv:
        print(next_id())
        return 0
    if "--list" in argv:
        for cid, body in rows():
            head = body.split(" | ")[0].strip().lstrip("*").strip()
            print(f"  {cid:>5}  {head[:70]}")
        return 0
    problems = validate()
    if not problems:
        n = len(rows())
        print(f"claims_ledger.md: {n} claims, no problems (next id {next_id()})")
        return 0
    for p in problems:
        print(f"  {p}")
    print(f"\n{len(problems)} problem(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
