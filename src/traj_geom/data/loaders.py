"""Load PARARULE-Plus reasoning-depth examples from the Hugging Face Hub.

OWNER: Data+Analysis
STATUS: reconstructed 2026-07-17 — the original module was never committed to
    git, on any branch, at any point in this project's history (silently
    swallowed by the .gitignore `data/` rule present since the initial
    scaffold commit; see docs/guide.md sec 5). `load_pararule`/`enrich` below
    are a faithful port of notebooks/01_mvp.ipynb cells 10-11 (the only
    surviving copy of this logic), adapted to take `tok` as an explicit
    argument instead of a notebook-global. Not independently re-verified
    against a fresh GPU run — if a re-run ever produces a `pararule.csv` that
    disagrees with the currently-committed one, trust the re-run and update
    this docstring.
TASK: Fetch examples from qbao775/PARARULE-Plus-Depth-{depth} (2-5) and shape
    them into the {prompt, depth, label, prompt_id} rows every PARARULE-based
    script expects; attach the tokenized prompt length (a confounder).
I/O: load_pararule(depth, n) -> Iterator[dict]; enrich(row, tok) -> dict.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

DEPTH_REPOS = {d: f"qbao775/PARARULE-Plus-Depth-{d}" for d in (2, 3, 4, 5)}
CTX, Q, LAB = "context", "question", "label"


def load_pararule(depth: int, n: int = 15) -> Iterator[dict[str, Any]]:
    """Yield ``n`` shuffled PARARULE-Plus examples at a fixed reasoning depth.

    Args:
        depth: Rule-chain depth (2-5); selects the
            ``qbao775/PARARULE-Plus-Depth-{depth}`` split.
        n: Number of examples to draw (shuffled with a fixed seed for
            reproducibility across calls).

    Yields:
        Dicts with ``prompt``, ``depth``, ``label`` (ground-truth 0/1), and
        ``prompt_id``.
    """
    from datasets import load_dataset

    ds = load_dataset(DEPTH_REPOS[depth], split="train").shuffle(seed=0).select(range(n))
    for i, ex in enumerate(ds):
        yield {
            "prompt": f"{ex[CTX]}\nQuestion: {ex[Q]}\nAnswer:",
            "depth": depth,
            "label": int(ex[LAB]),
            "prompt_id": f"d{depth}_{i}",
        }


def enrich(row: dict[str, Any], tok: Any) -> dict[str, Any]:
    """Add the tokenized prompt length (a confounder) and answer-token index.

    Args:
        row: A row yielded by :func:`load_pararule`.
        tok: The Huginn tokenizer.

    Returns:
        The same dict, with ``seq_len`` and ``answer_token_index`` added.
    """
    row["seq_len"] = int(tok(row["prompt"], return_tensors="pt").input_ids.shape[1])
    row["answer_token_index"] = -1
    return row
