"""Shared helpers for experiment scripts: results/ caching and lazy model loading.

Mirrors the notebook's `cached()` — compute once, reuse the CSV. When a result
CSV already exists (shipped from Kaggle in results/), the heavy compute is skipped
and no GPU / model extra is needed to re-run the analysis and figures.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def cached(name: str, compute_fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return ``results/<name>`` if it exists, else compute, save and return it.

    Args:
        name: CSV filename under results/.
        compute_fn: Zero-arg callable producing the DataFrame on a cache miss.

    Returns:
        The cached or freshly computed DataFrame.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    if path.exists():
        print(f"loaded {path}")
        return pd.read_csv(path)
    df = compute_fn()
    df.to_csv(path, index=False)
    print(f"computed+saved {path}")
    return df


def save_partial(rows: list[dict], name: str) -> None:
    """Checkpoint partial sweep progress after each iteration of a long GPU sweep.

    A crash/timeout/OOM on iteration N of a multi-hour Kaggle sweep otherwise
    loses everything computed so far -- ``cached()`` only ever writes once,
    at the very end, after ``compute_fn()`` returns. Call this after every
    ``rows.append(...)`` inside a sweep's inner loop instead.

    Written under a ``partial_`` prefix, NEVER the real cache filename
    ``cached()`` checks for -- a leftover partial file from a crashed run
    must not look like a complete cached result and silently short-circuit
    the next attempt before it actually finishes. This is a safety-net
    checkpoint, not a resume mechanism: on restart the sweep still recomputes
    from scratch, but nothing already-computed is lost to a crash.

    Args:
        rows: The sweep's accumulator list so far (may be empty -- a no-op).
        name: Final CSV filename this sweep will eventually save as via
            ``cached()`` (e.g. ``"three_scale.csv"``); the partial checkpoint
            is written as ``results/partial_<name>``.
    """
    if not rows:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"partial_{name}", index=False)


def load_model() -> tuple[Any, Any]:
    """Lazily import and load Huginn (needs the `model` extra and a GPU)."""
    from traj_geom.extraction.model import load_huginn

    return load_huginn()
