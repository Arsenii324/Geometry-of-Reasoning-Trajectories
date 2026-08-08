"""Tests for artifact provenance recording.

The point of the module under test is that an artifact cannot be silently
re-interpreted later, so these tests focus on the failure modes that actually
occurred in this project: a file whose contents no longer match its record, an
artifact with no record at all, and two datasets that look related but came
from different runs.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from traj_geom.provenance import (
    MANIFEST_NAME,
    SIDECAR_SUFFIX,
    new_run_id,
    read_provenance,
    record_artifact,
    save_array,
    save_table,
    verify_directory,
)


def test_save_array_writes_sidecar_and_manifest(tmp_path) -> None:
    p = str(tmp_path / "traj.npy")
    rec = save_array(p, np.zeros((7, 3)), task="count_ones", task_seed=1, init_seed=0)
    assert os.path.exists(p + SIDECAR_SUFFIX)
    assert os.path.exists(tmp_path / MANIFEST_NAME)
    assert rec["shape"] == [7, 3]
    assert rec["task_seed"] == 1 and rec["init_seed"] == 0
    assert read_provenance(p)["sha256"] == rec["sha256"]


def test_task_seed_and_init_seed_are_recorded_separately(tmp_path) -> None:
    """The distinction whose absence caused audit finding #5."""
    p = str(tmp_path / "t.npy")
    save_array(p, np.zeros((4, 2)), task_seed=3, init_seed=0)
    rec = read_provenance(p)
    assert rec["task_seed"] == 3
    assert rec["init_seed"] == 0
    assert rec["task_seed"] != rec["init_seed"]


def test_verify_detects_a_file_changed_after_recording(tmp_path) -> None:
    """A silently-edited artifact must be reported STALE, never trusted."""
    p = str(tmp_path / "a.npy")
    save_array(p, np.zeros((5, 2)))
    v = verify_directory(str(tmp_path))
    assert v["ok"] == ["a.npy"] and v["stale"] == []

    np.save(p, np.ones((5, 2)))          # same shape, different content
    v = verify_directory(str(tmp_path))
    assert v["stale"] == ["a.npy"] and v["ok"] == []


def test_verify_reports_undescribed_artifacts(tmp_path) -> None:
    np.save(str(tmp_path / "orphan.npy"), np.zeros((3, 3)))
    v = verify_directory(str(tmp_path))
    assert v["undescribed"] == ["orphan.npy"]


def test_run_id_is_stable_within_a_process_and_links_artifacts(tmp_path) -> None:
    """Shared run_id is what makes a cross-run mixture detectable."""
    a, b = str(tmp_path / "a.npy"), str(tmp_path / "b.csv")
    save_array(a, np.zeros((3, 2)))
    save_table(b, pd.DataFrame({"x": [1, 2]}))
    assert read_provenance(a)["run_id"] == read_provenance(b)["run_id"] == new_run_id()
    assert verify_directory(str(tmp_path))["run_ids"] == [new_run_id()]


def test_save_table_records_schema(tmp_path) -> None:
    p = str(tmp_path / "r.csv")
    save_table(p, pd.DataFrame({"n_ops": [1], "winding": [0.5]}), source_traj="t.npy")
    rec = read_provenance(p)
    assert rec["columns"] == ["n_ops", "winding"]
    assert rec["n_rows"] == 1
    assert rec["source_traj"] == "t.npy"


def test_recording_a_missing_file_raises(tmp_path) -> None:
    """Provenance describes real bytes only."""
    with pytest.raises(FileNotFoundError):
        record_artifact(str(tmp_path / "nope.npy"), "trajectory")


def test_manifest_is_append_only_history(tmp_path) -> None:
    p = str(tmp_path / "a.npy")
    save_array(p, np.zeros((3, 2)))
    save_array(p, np.ones((3, 2)))
    lines = (tmp_path / MANIFEST_NAME).read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["sha256"] != json.loads(lines[1])["sha256"]


def test_save_table_refuses_to_destroy_data_with_an_empty_frame(tmp_path) -> None:
    """An empty result must not overwrite a committed CSV.

    Found 2026-08-07 by running every script in a fresh clone: the raw
    `trajectories/*.npy` that `run_answer_probe.py` reads are gitignored, so
    every task hit its `len(g) < 10: continue` guard, `compute()` returned a
    zero-row frame, and `save_table` wrote it straight over the committed
    11-row `results/answer_probe.csv`. The script then crashed on
    `df.groupby("task")` -- one line AFTER the evidence was already gone.
    """
    import pandas as pd
    import pytest

    p = str(tmp_path / "evidence.csv")
    save_table(p, pd.DataFrame({"task": ["count_ones"], "r2": [0.2]}))
    before = open(p, encoding="utf-8").read()

    with pytest.raises(ValueError, match="empty table"):
        save_table(p, pd.DataFrame())

    assert open(p, encoding="utf-8").read() == before, "the CSV was modified anyway"


def test_save_table_empty_frame_message_flags_the_destructive_case(tmp_path) -> None:
    """The message must distinguish 'would destroy data' from 'nothing there yet',
    because those need different responses from whoever reads the traceback."""
    import pandas as pd
    import pytest

    fresh = str(tmp_path / "new.csv")
    with pytest.raises(ValueError) as e:
        save_table(fresh, pd.DataFrame())
    assert "would be destroyed" not in str(e.value)

    existing = str(tmp_path / "held.csv")
    save_table(existing, pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError) as e:
        save_table(existing, pd.DataFrame())
    assert "would be destroyed" in str(e.value)
