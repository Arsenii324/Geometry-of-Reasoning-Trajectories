"""A module constant defined and never used is usually an unimplemented gate.

WHY THIS EXISTS, AND WHY IT IS NARROW. On 2026-08-09 `scratch/ds_implicit/job.py`
pre-registered five items and implemented three. P4 -- the brute-force cross-check that
would have validated a new attribution method against ground truth -- was never
written, and its constant `N_BRUTE` sat defined and unreferenced. P5's token-count
verification was likewise absent.

THE FIRST VERSION OF THIS TEST WAS WRONG AND IS RECORDED AS SUCH. It asked whether the
literal label `P<n>` appeared in the code body. That flagged 6 kernels, of which **4
were false positives**: the gate was implemented, just not labelled in the body (e.g.
`ds_paired` calls `require_null_can_move` and `require_units` without printing "P3").
A check with a 67% false-positive rate trains its reader to ignore it, which is worse
than no check -- the supervisor's warning that using code to enforce document
consistency "isn't easy" applies exactly here.

So this is the narrow version: an unused module-level constant. `ruff` catches unused
LOCALS but not module-level names, and in these kernels a constant exists to
parameterise a gate -- if nothing reads it, the gate is almost certainly missing. It
will not catch every unimplemented item, and it is not meant to; it catches the one
signature that is unambiguous.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Names that are legitimately declarative rather than used in-process.
ALLOWED = {"MODEL_ID", "REVISION", "OUTDIR", "OUT"}


def _kernels() -> list[pathlib.Path]:
    return sorted(p for p in (ROOT / "scratch").glob("ds_*/job.py") if p.is_file())


def test_no_module_constant_is_defined_and_never_used():
    offences: list[str] = []
    for path in _kernels():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        consts = {t.id: n.lineno
                  for n in tree.body if isinstance(n, ast.Assign)
                  for t in n.targets
                  if isinstance(t, ast.Name) and t.id.isupper() and t.id not in ALLOWED}
        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
        for name, line in sorted(consts.items()):
            if name not in used:
                offences.append(
                    f"\n  {path.relative_to(ROOT)}:{line}  {name} is defined and never "
                    f"read -- in these kernels a constant parameterises a gate, so an "
                    f"unread one usually means the gate was pre-registered and not "
                    f"written."
                )
    assert not offences, (
        "Unused kernel constants, which in this project mark unimplemented "
        "pre-registered gates." + "".join(offences)
    )
