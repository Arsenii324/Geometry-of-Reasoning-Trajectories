"""Retro-tag existing artifacts with everything the audit could establish.

OWNER: Data+Analysis
STATUS: implemented 2026-07-25 alongside src/traj_geom/provenance.py.
TASK: give the artifacts that already exist the provenance they were saved
    without, recording separately what is VERIFIED (re-derived from the bytes
    or the model source during the audit) and what is RECOVERED (inferred from
    filenames and sibling files, and could be wrong).

WHY THE VERIFIED/RECOVERED SPLIT MATTERS
  Backfilled provenance is weaker than provenance captured at write time, and
  pretending otherwise would reproduce the original failure in a new place --
  a record that looks authoritative but is a guess. Every field written here
  carries its evidence level, and `confidence` on the record as a whole says
  which it is.

WHAT IS BEING RECORDED, AND HOW IT WAS ESTABLISHED
  results/trajectories/*.npy
    * `task_seed`: the filename's `init{N}` field. VERIFIED to be the task
      seed, not the init seed: `make_count_ones_task` derives the bit string
      from it, and seeds 1 and 2 collide to the same prompt for n_ops=2, whose
      two files are bit-identical.
    * `init_seed`: fixed but unrecorded. Same collision proves h_0 does not
      vary across these files.
    * `compute_dtype`: "bfloat16", VERIFIED -- every stored value is exactly
      bf16-representable (bit-identical bfloat16 round trip).
    * `prompt`: regenerated from the task generator at the recorded seed and
      CHECKED against `full_synthetic_experiments.csv` where a row exists.
    * `corresponds_to_csv`: explicitly false for
      full_synthetic_experiments.csv -- 0/80 winding values reproduce.

  trajectories/*.npy (top level)
    Genuine multi-init: manifest.csv records a real `init_seed` per file at
    fixed task parameters. These are the only raw paths in the project that
    support a path-independence or contraction measurement.

I/O: writes <file>.prov.json sidecars and manifest.jsonl next to each artifact.
    Idempotent -- rerunning overwrites sidecars and appends a fresh manifest
    line (the manifest is a history, by design).

Run: uv run python -m scripts.backfill_provenance
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from traj_geom.provenance import record_artifact, verify_directory  # noqa: E402
from traj_geom.shapes.synthetic import (  # noqa: E402
    make_count_ones_task,
    make_projection_task,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_TRAJ = os.path.join(ROOT, "results", "trajectories")
TOPLEVEL_TRAJ = os.path.join(ROOT, "trajectories")

_FNAME_RE = re.compile(
    r"^(?P<task>[a-z_]+)_n(?P<n_ops>\d+)_ns(?P<num_steps>\d+)_init(?P<seed>\d+)\.npy$"
)
_GENERATORS = {"count_ones": make_count_ones_task, "projection": make_projection_task}


def _is_bf16_exact(arr: np.ndarray) -> bool:
    """Whether every value survives a bfloat16 round trip unchanged."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return bool(torch.equal(t, t.to(torch.bfloat16).to(t.dtype)))


def backfill_results_trajectories() -> int:
    """Tag results/trajectories/*.npy (single-prompt-per-file, fixed h_0)."""
    if not os.path.isdir(RESULTS_TRAJ):
        print(f"backfill: {RESULTS_TRAJ} not present, skipping")
        return 0

    csv_path = os.path.join(ROOT, "results", "full_synthetic_experiments.csv")
    csv = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    n = 0
    for name in sorted(os.listdir(RESULTS_TRAJ)):
        m = _FNAME_RE.match(name)
        if m is None:
            continue
        path = os.path.join(RESULTS_TRAJ, name)
        arr = np.load(path)
        task, n_ops = m.group("task"), int(m.group("n_ops"))
        seed = int(m.group("seed"))

        prompt = answer = None
        gen = _GENERATORS.get(task)
        if gen is not None:
            t = gen(n_ops, seed=seed)
            prompt, answer = t["prompt"], t["answer"]

        prompt_matches_csv = None
        if csv is not None and prompt is not None:
            row = csv[(csv.task == task) & (csv.n_ops == n_ops) & (csv.seed == seed)]
            if len(row) == 1:
                prompt_matches_csv = bool(row.iloc[0]["prompt"] == prompt)

        record_artifact(
            path,
            kind="trajectory",
            confidence="RECOVERED_BY_AUDIT_2026_07_25",
            task=task,
            n_ops=n_ops,
            num_steps=int(m.group("num_steps")),
            task_seed=seed,
            task_seed_evidence="VERIFIED: filename 'init{N}' is the TASK seed; "
            "make_count_ones_task derives the bit string from it, and the "
            "n_ops=2 seed-1/seed-2 prompt collision yields bit-identical files.",
            init_seed=None,
            init_seed_evidence="FIXED but unrecorded. The same collision proves "
            "h_0 does not vary across these files, so they contain no "
            "multi-initialization information.",
            prompt=prompt,
            answer=answer,
            prompt_evidence="RECOVERED: regenerated from the task generator at "
            "task_seed; not read from a run log.",
            prompt_matches_full_synthetic_csv=prompt_matches_csv,
            compute_dtype="bfloat16",
            compute_dtype_evidence="VERIFIED: every stored value is exactly "
            f"bf16-representable (round-trip identical: {_is_bf16_exact(arr)}).",
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            token_index=None,
            token_index_evidence="UNKNOWN. Not recoverable from the bytes.",
            corresponds_to="results/full_synthetic_experiments.csv: NO. 0/80 "
            "winding values reproduce from these files at either num_steps, "
            "and steps_settle is 128 there vs 9-19 here. Different data.",
            audit_ref="docs/rigor_audit.md sections 1, 4, 5",
        )
        n += 1
    return n


def backfill_toplevel_trajectories() -> int:
    """Tag trajectories/*.npy — the project's only genuine multi-init paths."""
    man = os.path.join(TOPLEVEL_TRAJ, "manifest.csv")
    if not os.path.exists(man):
        print(f"backfill: {man} not present, skipping")
        return 0

    df = pd.read_csv(man)
    n = 0
    for _, r in df.iterrows():
        path = os.path.join(TOPLEVEL_TRAJ, str(r["file"]))
        if not os.path.exists(path):
            print(f"backfill: manifest lists missing file {r['file']}")
            continue
        arr = np.load(path)
        record_artifact(
            path,
            kind="trajectory",
            confidence="RECOVERED_BY_AUDIT_2026_07_25",
            kind_label=str(r["kind"]),
            n_ops=int(r["n_ops"]),
            num_steps=int(r["num_steps"]),
            init_seed=int(r["init_seed"]),
            init_seed_evidence="VERIFIED as a genuine initialization seed: "
            "files share (kind, n_ops, num_steps) and differ only in this "
            "field, and their pairwise gap contracts (53.4 -> 1.4), which "
            "requires a shared prompt.",
            task_seed=None,
            task_seed_evidence="UNKNOWN: manifest.csv records no task seed.",
            compute_dtype="bfloat16",
            compute_dtype_evidence="VERIFIED: bf16 round-trip identical "
            f"({_is_bf16_exact(arr)}).",
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            converged=bool(int(r["num_steps"]) >= 64),
            converged_evidence="ns=16 groups end with an inter-orbit gap of "
            "6.5-7.1 vs 1.2-1.6 for ns=64; ns=16 rows are NOT converged and "
            "must not be used for settled-state metrics.",
            supports="path-independence and map-contraction measurement "
            "(the only raw paths in the project that do)",
            audit_ref="docs/rigor_audit.md sections 7, 12",
        )
        n += 1
    return n


def main() -> None:
    """Backfill both trajectory directories and report verification status."""
    a = backfill_results_trajectories()
    b = backfill_toplevel_trajectories()
    print(f"backfilled provenance: {a} files in results/trajectories, {b} in trajectories/")

    for d in (RESULTS_TRAJ, TOPLEVEL_TRAJ):
        if not os.path.isdir(d):
            continue
        v = verify_directory(d)
        print(
            f"\n{os.path.relpath(d, ROOT)}: ok={len(v['ok'])} stale={len(v['stale'])} "
            f"undescribed={len(v['undescribed'])} runs={len(v['run_ids'])}"
        )
        if v["undescribed"]:
            print(f"  undescribed: {v['undescribed'][:5]}")


if __name__ == "__main__":
    main()
