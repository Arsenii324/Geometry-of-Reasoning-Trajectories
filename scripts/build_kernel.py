"""Compose a self-contained Kaggle `main.py` from a body plus tested shared blocks.

Kaggle runs one file, which is why every bundle re-declared the same helpers:
4563 lines across 23 bundles, 28% of 8-line windows duplicated, `fit_rho` and
friends written 8-9 times each. That duplication re-imported a fixed bug twice
(`np.arange(1, n+1, float)`, which passes `float` as the step) and cost a GPU run.

A body declares its needs on one line and this inlines them, dependencies first:

    # @needs: fit_rho measure_rho summarise

so kernels start from code that `tests/test_kernel_common.py` has already checked
against known answers. Blocks live in `scratch/_lib/kernel_common.py` between
`# ---8<--- name` markers; `needs:` on a marker line declares that block's own
dependencies.

Run:  python -m scripts.build_kernel <bundle-dir> [...]   (writes <bundle>/main.py)
      python -m scripts.build_kernel --check <bundle-dir> (verify main.py is current)
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "scratch", "_lib", "kernel_common.py")
BANNER = ("# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.\n"
          "# ---- Edit body.py and rebuild; edits to this file are overwritten.\n")


def parse_lib() -> dict[str, tuple[str, list[str]]]:
    """name -> (source, dependencies)."""
    text = open(LIB, encoding="utf-8").read()
    blocks: dict[str, tuple[str, list[str]]] = {}
    pat = re.compile(r"^# ---8<--- (\w+)(?:\s+needs:\s*([\w ]+))?\s*$", re.M)
    marks = list(pat.finditer(text))
    for m in marks:
        end = text.find("\n# ---8<---", m.end())
        if end == -1:
            raise ValueError(f"block {m.group(1)} is not closed")
        blocks[m.group(1)] = (text[m.end():end].strip("\n"),
                              (m.group(2) or "").split())
    return blocks


def resolve(names: list[str], blocks: dict) -> list[str]:
    """Topologically order the requested blocks with their dependencies."""
    out: list[str] = []
    seen: set[str] = set()

    def visit(n: str, stack: tuple[str, ...] = ()) -> None:
        if n in out:
            return
        if n in stack:
            raise ValueError(f"circular dependency: {' -> '.join((*stack, n))}")
        if n not in blocks:
            raise KeyError(f"unknown block {n!r}; have {sorted(blocks)}")
        seen.add(n)
        for dep in blocks[n][1]:
            visit(dep, (*stack, n))
        out.append(n)

    for n in names:
        visit(n)
    return out


def build(bundle: str) -> str:
    body_path = os.path.join(bundle, "body.py")
    if not os.path.exists(body_path):
        raise FileNotFoundError(f"{body_path} not found")
    body = open(body_path, encoding="utf-8").read()
    m = re.search(r"^#\s*@needs:\s*(.+)$", body, re.M)
    needs = m.group(1).split() if m else []
    blocks = parse_lib()
    ordered = resolve(needs, blocks)

    # keep the body's module docstring at the top of the generated file
    doc_end = 0
    if body.lstrip().startswith('"""'):
        start = body.index('"""')
        doc_end = body.index('"""', start + 3) + 3
    head, rest = body[:doc_end], body[doc_end:]
    rest = re.sub(r"^#\s*@needs:.*$\n?", "", rest, flags=re.M)

    parts = [head, "\n", BANNER]
    if ordered:
        parts.append(f"# ---- inlined blocks: {' '.join(ordered)}\n")
    for n in ordered:
        parts.append("\n\n" + blocks[n][0] + "\n")
    parts.append("\n" + rest.lstrip("\n"))
    return "".join(parts)


def main(argv: list[str]) -> int:
    check = "--check" in argv
    bundles = [a for a in argv if not a.startswith("-")]
    if not bundles:
        print(__doc__)
        return 2
    bad = 0
    for b in bundles:
        b = b.rstrip("/")
        try:
            text = build(b)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"{b}: {type(e).__name__}: {e}")
            bad += 1
            continue
        out = os.path.join(b, "main.py")
        cur = open(out, encoding="utf-8").read() if os.path.exists(out) else None
        if check:
            if cur != text:
                print(f"{b}: main.py is STALE -- rebuild")
                bad += 1
            else:
                print(f"{b}: up to date")
        else:
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            n_inlined = text.count("# ---- inlined blocks:")
            print(f"{b}: wrote main.py ({len(text.splitlines())} lines, "
                  f"{'blocks inlined' if n_inlined else 'no blocks'})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
