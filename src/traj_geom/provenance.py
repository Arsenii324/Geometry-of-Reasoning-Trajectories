"""Provenance records for saved artifacts (.npy trajectories, .csv results).

OWNER: Data+Analysis
STATUS: implemented 2026-07-25, in response to docs/rigor_audit.md sections 4
    and 5 -- two provenance failures that cost real analysis time and produced
    at least one wrong conclusion before being caught.
TASK: make every saved artifact self-describing, and make it mechanically
    checkable whether two artifacts came from the same run.

THE TWO FAILURES THIS PREVENTS
  (1) SILENT MISMATCH. `results/full_synthetic_experiments.csv` and
      `results/trajectories/*.npy` share task names, difficulty levels and
      seed indices, and their row/file counts are mutually consistent. They
      are nevertheless different data: 0 of 80 CSV winding values reproduce
      from the .npy files, and the CSV's `steps_settle` is 128 everywhere
      against a recomputed 9-19. Nothing in either artifact recorded which run
      produced it, so the mismatch was invisible for weeks and a null test run
      on one was applied to claims resting on the other.

  (2) SEMANTICS IN FILENAMES. `count_ones_n16_ns128_init3.npy` reads as
      "init seed 3". It is in fact the TASK seed -- it changes the prompt, not
      h_0. Proven by a collision: seeds 1 and 2 generate the same bit string
      for n_ops=2, and those two files are bit-identical. Any analysis that
      treated them as repeated initializations was measuring something else.

DESIGN, AND WHY THIS SHAPE
  Each artifact gets a JSON sidecar (`<file>.prov.json`) AND an append-only
  line in a per-directory `manifest.jsonl`. Both, deliberately:
    * the sidecar travels with the file, so a copied/moved artifact keeps its
      provenance and cannot be silently re-interpreted;
    * the manifest gives one place to scan a whole directory, and being
      append-only it preserves the history of what was written when.
  Filenames stay human-readable but carry NO load-bearing semantics -- the
  sidecar is the authority. This is the specific brittleness that failed.

  Every record carries a `run_id`, and artifacts written by the same script
  invocation share it. That single field is what makes failure (1)
  mechanically detectable: `verify_directory` reports cross-run mixtures, and
  a table row can name the exact trajectory file it was computed from.

  Content is hashed (sha256). A record whose hash no longer matches the file
  is reported as STALE rather than trusted -- the file changed after being
  described.

  Task seed and init seed are SEPARATE required fields for trajectories. They
  are not interchangeable and conflating them is failure (2).

I/O: save_array / save_table write artifact + sidecar + manifest line;
    verify_directory re-hashes and reports drift.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np

__all__ = [
    "new_run_id",
    "sha256_of",
    "record_artifact",
    "save_array",
    "save_table",
    "read_provenance",
    "verify_directory",
    "MANIFEST_NAME",
    "SIDECAR_SUFFIX",
]

MANIFEST_NAME = "manifest.jsonl"
SIDECAR_SUFFIX = ".prov.json"

_RUN_ID: str | None = None


def new_run_id() -> str:
    """Return this process's run id, creating it on first call.

    One id per interpreter process, so every artifact written by a single
    script invocation is linkable. Deliberately not derived from the clock
    alone -- two runs started in the same second must not collide.
    """
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    return _RUN_ID


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    """Stream a file through sha256 (chunked, so multi-GB arrays are fine)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str | None:
    """Current commit, or None outside a repo / if git is unavailable.

    Never raises: provenance capture must not be able to fail a run that has
    already spent GPU time producing the artifact.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes (None if unknown).

    A commit hash alone is not enough to reproduce an artifact if the tree was
    dirty when it was written; recording that fact is the honest minimum.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def _model_constants() -> dict[str, Any]:
    """Pinned model id/revision, if traj_geom.constants is importable."""
    try:
        from traj_geom.constants import MODEL_ID, MODEL_REVISION

        return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    except Exception:
        return {}


def record_artifact(path: str, kind: str, **meta: Any) -> dict[str, Any]:
    """Write a provenance sidecar for an existing file and append to the manifest.

    Args:
        path: Path to the artifact that has already been written to disk.
        kind: Free-form artifact type, e.g. ``"trajectory"`` or ``"table"``.
        **meta: Everything needed to regenerate or correctly interpret the
            artifact. For trajectories this SHOULD include ``prompt``,
            ``task``, ``task_seed``, ``init_seed``, ``num_steps``,
            ``token_index`` and ``compute_dtype``. ``task_seed`` and
            ``init_seed`` are separate on purpose -- see the module docstring.

    Returns:
        The record that was written.

    Raises:
        FileNotFoundError: if ``path`` does not exist. Provenance is recorded
            for real bytes only; there is no way to describe a file that was
            never written.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"cannot record provenance for missing file: {path}")

    rec: dict[str, Any] = {
        "file": os.path.basename(path),
        "kind": kind,
        "run_id": new_run_id(),
        "written_utc": datetime.now(UTC).isoformat(),
        "sha256": sha256_of(path),
        "bytes": os.path.getsize(path),
        "produced_by": " ".join(sys.argv) if sys.argv else None,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        **_model_constants(),
        **meta,
    }

    with open(path + SIDECAR_SUFFIX, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    manifest = os.path.join(os.path.dirname(os.path.abspath(path)), MANIFEST_NAME)
    with open(manifest, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    return rec


def save_array(path: str, arr: np.ndarray, kind: str = "trajectory", **meta: Any) -> dict[str, Any]:
    """``np.save`` the array, then record its provenance.

    Shape and dtype are captured automatically -- they are the two things most
    often needed and most often forgotten.
    """
    arr = np.asarray(arr)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, arr)
    return record_artifact(
        path, kind, shape=list(arr.shape), dtype=str(arr.dtype), **meta
    )


def save_table(path: str, df: Any, kind: str = "table", **meta: Any) -> dict[str, Any]:
    """``df.to_csv(index=False)``, then record provenance including the schema.

    Recording ``columns`` and ``n_rows`` makes a schema change visible in the
    manifest rather than only at the next read.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    return record_artifact(
        path, kind, n_rows=int(len(df)), columns=[str(c) for c in df.columns], **meta
    )


def read_provenance(path: str) -> dict[str, Any] | None:
    """Return an artifact's sidecar record, or None if it has none."""
    side = path + SIDECAR_SUFFIX
    if not os.path.exists(side):
        return None
    with open(side, encoding="utf-8") as fh:
        return json.load(fh)


def verify_directory(
    directory: str, patterns: tuple[str, ...] = (".npy", ".csv")
) -> dict[str, Any]:
    """Check every artifact in a directory against its recorded provenance.

    Args:
        directory: Directory to scan (non-recursive).
        patterns: File suffixes treated as artifacts.

    Returns:
        A dict with keys ``ok`` (hash matches), ``stale`` (file changed since
        it was described), ``undescribed`` (no sidecar at all) and ``run_ids``
        (every distinct run represented). More than one entry in ``run_ids``
        means the directory mixes runs -- legitimate for an append-only
        results directory, but the thing to check first when two artifacts
        that "should" correspond do not.
    """
    ok: list[str] = []
    stale: list[str] = []
    undescribed: list[str] = []
    run_ids: set[str] = set()

    for name in sorted(os.listdir(directory)):
        if not name.endswith(patterns) or name.endswith(SIDECAR_SUFFIX):
            continue
        full = os.path.join(directory, name)
        rec = read_provenance(full)
        if rec is None:
            undescribed.append(name)
            continue
        run_ids.add(rec.get("run_id", "?"))
        (ok if rec.get("sha256") == sha256_of(full) else stale).append(name)

    return {
        "directory": directory,
        "ok": ok,
        "stale": stale,
        "undescribed": undescribed,
        "run_ids": sorted(run_ids),
    }
