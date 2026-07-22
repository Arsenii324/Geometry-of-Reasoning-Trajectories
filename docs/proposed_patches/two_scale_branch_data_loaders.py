"""Load PARARULE-Plus reasoning-depth examples from the Hugging Face Hub.

OWNER: Data+Analysis
STATUS: reconstructed 2026-07-17, BEST-EFFORT — this module was never
    committed to git on this branch (or any branch) either; see
    docs/guide.md sec 5 on `main`. Unlike `load_pararule`/`enrich` (main
    branch), which are a verbatim port of notebooks/01_mvp.ipynb, there is
    NO surviving notebook cell for `load_pararule_real`/`pararule_depth_of`
    on this branch -- this is a reconstruction from three sources: (1) the
    exact contract in tests/test_two_scale.py::
    test_pararule_depth_of_qdep_and_id_fallback, which this module is
    written to satisfy exactly; (2) docs/two_scale.md's one-line description
    ("depth from meta['QDep'], fallback '-D<d>-' tag in id"); (3) the real
    qbao775/PARARULE-Plus-Depth-{2..5} HF dataset schema, confirmed via the
    HF dataset viewer to actually have `id`, `context`, `question`, `label`,
    and `meta` (with `meta.QDep`, `meta.QCat`) fields -- matching both (1)
    and (2) exactly. What's UNVERIFIED: whether `load_pararule_real` in the
    original code sourced from these same per-depth-split repos (as
    reconstructed here) or from a different, combined PARARULE-Plus repo
    that `pararule_depth_of` was needed to filter by depth from scratch --
    the per-depth-repo version is the simpler, more likely reading given
    `main`'s `load_pararule` already establishes that pattern, but treat
    this as a hypothesis until someone with the original file confirms it.
TASK: Load a larger pool (`n` up to ~200) of PARARULE-Plus examples at a
    given depth, tagging each with its depth via `pararule_depth_of` (a
    self-checking derivation from the example's own metadata, rather than
    trusting the split parameter blindly).
I/O: load_pararule_real(depth, n) -> Iterator[dict]; pararule_depth_of(row) -> int.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

DEPTH_REPOS = {d: f"qbao775/PARARULE-Plus-Depth-{d}" for d in (2, 3, 4, 5)}
CTX, Q, LAB = "context", "question", "label"

_ID_DEPTH_RE = re.compile(r"-D(\d+)-")


def pararule_depth_of(row: dict[str, Any]) -> int:
    """Derive a PARARULE-Plus example's reasoning depth from its own metadata.

    Args:
        row: A raw HF-dataset example (or a dict with the same ``meta``/``id``
            shape), e.g. ``{"meta": {"QDep": "3"}, "id": "...-D3-12"}``.

    Returns:
        The integer depth, read from ``row["meta"]["QDep"]`` if present,
        else parsed from a ``-D<d>-`` tag in ``row["id"]``.

    Raises:
        ValueError: Neither source yields a depth.
    """
    qdep = row.get("meta", {}).get("QDep")
    if qdep is not None:
        return int(qdep)
    m = _ID_DEPTH_RE.search(row.get("id", ""))
    if m:
        return int(m.group(1))
    raise ValueError(f"cannot determine PARARULE-Plus depth for row: {row!r}")


def load_pararule_real(depth: int, n: int = 200) -> Iterator[dict[str, Any]]:
    """Yield up to ``n`` shuffled PARARULE-Plus examples at a fixed depth.

    Larger-pool sibling of ``main``'s ``load_pararule`` (n=200 vs ~15), for
    the length-matching band design, which needs a wide pool to find
    seq_len-matched pairs across adjacent depths. Depth is attached via
    :func:`pararule_depth_of` rather than trusted from the loop variable, so
    a mislabeled split would surface as a mismatch rather than silently pass
    through.

    Args:
        depth: Rule-chain depth (2-5).
        n: Pool size to draw (shuffled with a fixed seed for reproducibility).

    Yields:
        Dicts with ``prompt``, ``depth``, ``label``, and ``id``.
    """
    from datasets import load_dataset

    ds = load_dataset(DEPTH_REPOS[depth], split="train").shuffle(seed=0).select(range(n))
    for ex in ds:
        yield {
            "prompt": f"{ex[CTX]}\nQuestion: {ex[Q]}\nAnswer:",
            "depth": pararule_depth_of(ex),
            "label": int(ex[LAB]),
            "id": ex["id"],
        }
