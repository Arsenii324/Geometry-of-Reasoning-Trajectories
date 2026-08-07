"""Executable documentation: check docs/architecture_state.md's claims
against real repo state, instead of trusting hand-maintained prose.

OWNER: Data+Analysis
STATUS: implemented 2026-07-23.
TASK: catch documentation drift automatically -- (a) every results/*.csv
    and every scripts/run_*.py must be named in architecture_state.md, and
    (b) no results/<name>.csv may be committed *before* the last commit
    that touched the scripts/run_*.py documented as producing it (a script
    that changed more recently than its own output is a real, concrete
    "is this data still what the code currently does" question, not a
    hypothetical one -- this project has already been burned by exactly
    this: run_three_scale.py's real bug on 2026-07-23 was only caught by
    actually re-running it, and the two orphaned CSVs this test's own
    authoring pass found (dissociation_results.csv, h2_loops.csv, neither
    with a current producing script) had been sitting silently uncaught
    since the project's first commits).
I/O: no args -> pass/fail via pytest; failure messages name the exact gap.

WHY THIS EXISTS, NOT JUST architecture_state.md's PROSE: prose docs are
only as good as someone remembering to update them -- this project has
repeatedly found real drift (D10's length-confound methodology defect,
Gemini's coda-skip bug, tonight's spearman() tuple-format crash) that
static reading missed and only execution or a direct check caught. This
file is that direct check, run every time `pytest` runs, not something
someone has to remember to do separately.

WHAT THIS DELIBERATELY DOES NOT CATCH: whether a results/<name>.csv's
*numbers* still match what its producing script would compute today (that
needs re-running the script, not just comparing commit timestamps -- a
script edited only in a comment or docstring trips this check exactly the
same as a real logic change, a false-positive by design: a cheap nudge to
go check is worth more here than a missed real one). It also doesn't check
whether *analysis/interpretation* of a CSV (e.g. a claims_ledger.md row)
has gone stale relative to either the CSV or the script that reads it --
that needs a human/AI judgment call about whether a change was
substantive, not a timestamp comparison; see claims_ledger.md's own
practice of citing a commit hash next to load-bearing numbers (e.g. D11)
as the manual complement to this automated check.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
_arch_archive = ROOT / "docs" / "archive" / "architecture_state.md"
ARCH_DOC = _arch_archive if _arch_archive.exists() else ROOT / "docs" / "architecture_state.md"


def _git_last_commit_time(path: Path) -> int | None:
    """Unix timestamp of the last commit touching ``path``, or None if untracked."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", str(path.relative_to(ROOT))],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None
    return int(out) if out else None


# Explicit, justified sign-off for a (csv, script) pair whose script
# changed after its cached data without that changing the data's validity.
# NOT a silent suppression -- every entry names the commit and states why
# the change was resilience/style-only, not logic. Remove an entry (making
# the check fire again) the moment its script changes for a reason that
# might actually affect what gets computed.
_REVIEWED_NON_SUBSTANTIVE_CHANGES: dict[str, str] = {
    "counting_accuracy.csv": (
        "2026-07-23 (d009d4e): run_accuracy.py gained save_partial() "
        "checkpointing only, already had try/except -- no change to what's computed."
    ),
    "counting.csv": (
        "2026-07-23 (843505d/d009d4e): run_counting.py's partial_spearman call "
        "gained a try/except ValueError around the *print* step (the new "
        "collinearity guard), unrelated to compute() -- the cached CSV's values "
        "are from the unchanged compute() path. 2026-07-24: gained a STATUS "
        "docstring line only (documentation practice pass), no code change."
    ),
    "convergence.csv": (
        "2026-07-24: run_convergence.py gained a STATUS docstring line only "
        "(documentation practice pass, adding what every other script already "
        "had) -- no change to the convergence-descriptor computation itself."
    ),
    "dissociation.csv": (
        "2026-07-23 (d009d4e): run_dissociation.py's compute() gained "
        "try/except + save_partial() per iteration -- same extraction logic, "
        "just resilient to a crash mid-sweep now. 2026-07-24: gained a STATUS "
        "docstring line only, no code change."
    ),
    "dissoc_multiinit.csv": (
        "2026-07-23 (d009d4e): same checkpointing-only change as dissociation.csv. "
        "2026-07-24: gained a STATUS docstring line only, no code change."
    ),
    "forceloop.csv": (
        "2026-07-23 (d009d4e): same checkpointing-only change as dissociation.csv. "
        "2026-07-24: main() also gained the Fisher exact test reproducing "
        "claims_ledger.md B4 -- compute() untouched, verified the new numbers "
        "match the ledger's already-recorded values exactly against the "
        "unchanged CSV. Also gained a STATUS docstring line, no further code change."
    ),
    "homology.csv": (
        "2026-07-24: run_homology.py gained a STATUS docstring line only "
        "(documentation practice pass) -- no change to the H1-persistence "
        "computation itself."
    ),
    "maxtask.csv": (
        "2026-07-23 (843505d): same print-step-only guard as counting.csv, "
        "unrelated to compute(). 2026-07-24: gained a STATUS docstring line "
        "only, no code change."
    ),
    "pararule.csv": (
        "2026-07-23 (d009d4e): same checkpointing-only change as dissociation.csv."
    ),
    "phase.csv": (
        "2026-07-23 (d009d4e): same checkpointing-only change as dissociation.csv. "
        "2026-07-24: gained a STATUS docstring line only, no code change."
    ),
    "switch.csv": (
        "2026-07-23 (843505d): same print-step-only guard as counting.csv, "
        "unrelated to compute(). 2026-07-24: gained a STATUS docstring line "
        "only, no code change."
    ),
    "three_scale.csv": (
        "2026-07-24 (5a39e03): run_three_scale.py's main() gained the "
        "canonical per-level stat + multivariate_rank_control reporting -- "
        "compute() (the extraction loop) is byte-identical, so the cached "
        "rows still reflect what's computed; only how they're summarised "
        "changed. Verified by re-running main() against the unchanged CSV."
    ),
    "power_curve.csv": (
        "2026-07-24 (0accc95): run_power_analysis.py's BLAYNEY_RATE_OPTIMISTIC "
        "correction (2.81% -> 0.14%, see claims_ledger.md B9) only touches the "
        "loop-rate functions -- spearman_power_curve() (this CSV's producer) "
        "is byte-identical, confirmed by the file being byte-identical after "
        "re-running with the fixed script."
    ),
    "three_scale_modk.csv": (
        "2026-07-24 (83a81fe): run_three_scale_modk.py's compute()/grid were "
        "parameterized into a GRIDS dict (adding an --extended preset for a "
        "D15 follow-up) -- the 'default' grid's values (total_len=24, "
        "active_lens=(0,3,6,9,12,15,18), irrelevant_lens=(0,3,6), moduli=(2,5), "
        "n_seeds=3) are byte-identical to the prior hardcoded constants, "
        "confirmed by tests/test_three_scale_modk_results.py's pinned values "
        "still passing unchanged against this CSV."
    ),
}


def _parse_producer_map() -> dict[str, Path]:
    """Map ``results/<name>.csv`` -> the ``scripts/run_*.py`` documenting it.

    Parsed from each script's own module docstring (its existing ``I/O:``
    line already names its output by this project's own header convention)
    -- not a separate registry that itself could drift out of sync.
    """
    producer: dict[str, Path] = {}
    pattern = re.compile(r"results/([\w.]+\.csv)")
    for script in sorted(SCRIPTS_DIR.glob("run_*.py")):
        doc_match = re.match(r'^"""(.*?)"""', script.read_text(), re.DOTALL)
        if not doc_match:
            continue
        for csv_name in pattern.findall(doc_match.group(1)):
            producer.setdefault(csv_name, script)
    return producer


def test_every_results_csv_is_named_in_architecture_doc():
    """Every committed results/*.csv must appear (by filename) in
    architecture_state.md -- catches a new/renamed results file nobody
    documented, including orphaned legacy data with no current producer.
    """
    doc_text = ARCH_DOC.read_text()
    missing = [f.name for f in sorted(RESULTS_DIR.glob("*.csv")) if f.name not in doc_text]
    assert not missing, (
        f"results/*.csv not mentioned anywhere in {ARCH_DOC.relative_to(ROOT)}: {missing}. "
        "Add them (and note whether they have a current producing script) before this passes."
    )


def test_every_run_script_is_named_in_architecture_doc():
    """Every scripts/run_*.py must appear (by filename) in architecture_state.md.

    A glob-style reference (e.g. "run_dissociation*.py (3 variants)") does
    NOT satisfy this -- it must name each actual file, which is what
    caught run_dissociation.py/run_dissociation_multiinit.py being
    under-documented (and a *nonexistent* third variant implied) when this
    test was first written.
    """
    doc_text = ARCH_DOC.read_text()
    missing = [f.name for f in sorted(SCRIPTS_DIR.glob("run_*.py")) if f.name not in doc_text]
    assert not missing, (
        f"scripts/run_*.py not mentioned by exact filename in "
        f"{ARCH_DOC.relative_to(ROOT)}: {missing}."
    )


def test_no_results_csv_committed_before_its_producing_script_last_changed():
    """A results/<name>.csv must not predate the last commit to the
    scripts/run_*.py documented as producing it.

    If the script changed more recently than the data, the data may no
    longer reflect what the script currently computes -- this is a prompt
    to go check (re-run it, or read the diff to judge whether it matters),
    not an automatic verdict that the data is wrong. Silence here is not
    proof of freshness either: a script and its data committed together in
    the same commit (the common case when both are added at once) reads as
    consistent even if neither has been run since — this check catches
    drift *after* first creation, not correctness at creation time.
    """
    producer_map = _parse_producer_map()
    stale = []
    for csv_name, script in producer_map.items():
        if csv_name in _REVIEWED_NON_SUBSTANTIVE_CHANGES:
            continue
        csv_path = RESULTS_DIR / csv_name
        if not csv_path.exists():
            continue  # never run yet -- a different concern, not staleness
        csv_time = _git_last_commit_time(csv_path)
        script_time = _git_last_commit_time(script)
        if csv_time is None or script_time is None:
            continue  # uncommitted -- nothing to compare yet
        if script_time > csv_time:
            stale.append((csv_name, script.name))

    if stale:
        detail = "\n".join(
            f"  results/{csv} <- {script} (script changed more recently)" for csv, script in stale
        )
        pytest.fail(
            "Possibly stale results (producing script committed after its data):\n"
            f"{detail}\n"
            "Check whether the script change was substantive (re-run it or read the "
            "diff) before trusting these numbers, then note the finding wherever "
            "they're cited (e.g. claims_ledger.md)."
        )
