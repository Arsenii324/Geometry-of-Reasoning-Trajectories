"""Which A-number belongs to which experiment, and is any of them claimed twice?

WHY IT EXISTS. A subagent building a kernel in parallel named it **A46**, which the position
sweep already held -- two different experiments, one number, both launched. Caught by eye. The
numbers are the join key between kernels, ledger rows and job ids, so a collision corrupts the
record rather than just being untidy.

It scans both places numbers are assigned, because they live in two: GPU kernels under
`scratch/*/job.py` and zero-GPU analyses under `scripts/*.py` (A37 is `regime_onset.py`). It
reads the first eight lines rather than only line 1, since some headers open with a bare `\"\"\"`.

    uv run python scripts/experiment_registry.py
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAG = re.compile(r"\bA(\d+)\b\s*[:(]")


def scan():
    found = []
    for f in sorted(ROOT.glob("scratch/*/job.py")) + sorted(ROOT.glob("scripts/*.py")):
        head = "\n".join(f.read_text(errors="ignore").splitlines()[:8])
        m = TAG.search(head)
        if m:
            found.append((int(m.group(1)), f.relative_to(ROOT)))
    return sorted(found)


def main() -> int:
    found = scan()
    if not found:
        print("no A-numbers found")
        return 1
    counts = collections.Counter(n for n, _ in found)
    dupes = {n for n, c in counts.items() if c > 1}
    for n, path in found:
        flag = "   <== DUPLICATE" if n in dupes else ""
        print(f"  A{n:<4d} {path}{flag}")
    nums = sorted(counts)
    gaps = [n for n in range(min(nums), max(nums) + 1) if n not in counts]
    print(f"\n{len(found)} experiments, A{min(nums)}-A{max(nums)}")
    if gaps:
        print(f"  unused numbers: {', '.join('A' + str(g) for g in gaps)}")
    if dupes:
        print(f"  COLLISIONS: {', '.join('A' + str(d) for d in sorted(dupes))} "
              f"-- two experiments share a number; the ledger join key is broken")
        return 1
    print(f"  no collisions. Next free number: A{max(nums) + 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
