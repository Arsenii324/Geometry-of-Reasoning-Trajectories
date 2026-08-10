"""Generate `docs/EXPERIMENT_INDEX.md`: every claim, its code, its data, its run.

WHY THIS EXISTS. The supervisor's objection, verbatim: *"you really shouldn't keep
experiments unindexed and lost."* He is right, and this project has paid for it twice —
B3 was called "never done" while `ds_eigen` had already answered it (D119), and B4.8 was
called "the last untested topological claim" while D64 had closed it, after which the
existing instrument was OVERWRITTEN and its known-bad predecessor reinvented (D123). Both
were failures to find work that existed.

WHY IT IS GENERATED, NOT WRITTEN. `CLAUDE.md` §6 allows three hand-maintained documents
and calls anything that only accumulates "sediment"; this repo produced 23 such files and
they cost more than they returned. A generated index cannot drift from the ledger, cannot
be half-updated, and costs nothing to regenerate. **Do not hand-edit the output.**

WHAT IT CROSS-REFERENCES, and what each link buys:
  * the ledger row            -- the claim, its controls and its limits
  * `scratch/<kernel>/`       -- the code that produced it, at the commit that ran it
  * a DataSphere/Kaggle job   -- the run itself, recoverable via `job attach --id`
  * `scripts/run_*.py`        -- the re-derivable analysis, where one exists
  * banked `.npy` / `.json`   -- the raw states, for re-analysis with no GPU

It also reports what is NOT linked, which is the part worth reading: a claim with no code
reference is a claim nobody can re-derive, and this file names them rather than letting
them sit unnoticed.

Run:  uv run python scripts/build_index.py
"""

from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/claims_ledger.md"
OUT = ROOT / "docs/EXPERIMENT_INDEX.md"

JOB_RE = re.compile(r"\b(bt1[a-z0-9]{16,})\b")
KERNEL_RE = re.compile(r"scratch/([A-Za-z0-9_]+)")
SCRIPT_RE = re.compile(r"(scripts/[A-Za-z0-9_]+\.py|`(run_[A-Za-z0-9_]+\.py)`)")
SRC_RE = re.compile(r"\b(traj_geom(?:\.[A-Za-z0-9_]+)+)")


def rows() -> list[dict]:
    out = []
    for line in LEDGER.read_text().split("\n"):
        if not line.startswith("| D"):
            continue
        cells = line.split("|")
        if len(cells) < 4:
            continue
        rid = cells[1].strip()
        if not re.fullmatch(r"D\d+", rid):
            continue
        claim, evidence = cells[2], cells[3]
        blob = line
        headline = re.sub(r"[*_~`]", "", claim).strip()
        headline = re.split(r"(?<=[.!?])\s", headline)[0][:150]
        out.append({
            "id": rid,
            "n": int(rid[1:]),
            "headline": headline,
            "jobs": sorted(set(JOB_RE.findall(blob))),
            "kernels": sorted({k for k in KERNEL_RE.findall(blob) if k != "out"}),
            "scripts": sorted({(m[1] or m[0]).replace("scripts/", "")
                               for m in SCRIPT_RE.findall(blob)}),
            "modules": sorted(set(SRC_RE.findall(blob))),
            "retracted": bool(re.search(r"RETRACT|SUPERSEDED|WITHDRAWN|VOID", blob)),
            "evidence_raw": re.sub(r"[*_`]", "", evidence).strip()[:120],
        })
    return sorted(out, key=lambda r: r["n"])


def existing(paths: list[str], prefix: str, suffix: str = "") -> list[tuple[str, bool]]:
    return [(p, (ROOT / f"{prefix}{p}{suffix}").exists()) for p in paths]


def main() -> int:
    rs = rows()
    runs = {}
    rj = ROOT / "scratch/RUNS.json"
    if rj.exists():
        try:
            for r in json.loads(rj.read_text()):
                if isinstance(r, dict) and r.get("id"):
                    runs[r["id"]] = r.get("name", "")
        except json.JSONDecodeError:
            pass

    banked = {}
    for d in sorted((ROOT / "scratch").glob("*/")):
        n = len(list(d.rglob("*.npy"))) + len(list(d.rglob("*.json")))
        if n:
            banked[d.name] = n

    lines = [
        "# Experiment index — GENERATED, do not hand-edit",
        "",
        "Regenerate with `uv run python scripts/build_index.py`. Source of truth is",
        "`claims_ledger.md`; this file only cross-references it against the code, the runs",
        "and the banked data, so that no experiment is unfindable.",
        "",
        f"**{len(rs)} claims · {len({j for r in rs for j in r['jobs']})} distinct jobs referenced · "
        f"{len({k for r in rs for k in r['kernels']})} kernels · "
        f"{sum(banked.values())} banked artifacts across {len(banked)} directories**",
        "",
        "| row | claim (first sentence) | kernel | job | analysis | banked |",
        "|---|---|---|---|---|---|",
    ]
    for r in rs:
        ks = ", ".join(f"`{k}`{'' if (ROOT / 'scratch' / k).exists() else ' ⚠MISSING'}"
                       for k in r["kernels"]) or "—"
        js = ", ".join(f"`{j}`" + (f" ({runs[j]})" if j in runs else " ⚠unregistered")
                       for j in r["jobs"]) or "—"
        ss = ", ".join(f"`{s}`{'' if (ROOT / 'scripts' / s).exists() else ' ⚠MISSING'}"
                       for s in r["scripts"]) or "—"
        bk = ", ".join(f"{k}: {banked[k]}" for k in r["kernels"] if k in banked) or "—"
        flag = " 🔻" if r["retracted"] else ""
        head = r["headline"].replace("|", "/")
        lines.append(f"| **{r['id']}**{flag} | {head} | {ks} | {js} | {ss} | {bk} |")

    unlinked = [r for r in rs if not (r["kernels"] or r["jobs"] or r["scripts"] or r["modules"])]
    lines += [
        "",
        "## Claims with no code, job or script reference",
        "",
        "These cannot be re-derived by anyone but their author, and that is the defect",
        "`run_addk_arms.py` was written to fix for D110. Listing them is not an accusation —",
        "many are re-analyses of data banked elsewhere, or reasoning about the model source.",
        "But a load-bearing number in this list should get a script.",
        "",
    ]
    if unlinked:
        for r in unlinked:
            lines.append(f"- **{r['id']}** — {r['headline'][:110]} "
                         f"*(evidence column: {r['evidence_raw'][:70]})*")
    else:
        lines.append("- none")

    # A bank is only an orphan if NOTHING references it -- not merely if no ledger row
    # does. The first version checked ledger citations alone and reported
    # `kaggle_b6bank`, `ds_seeds` and `kaggle_register_correct` as orphans while
    # `run_b6.py`, `run_window_law.py`, `run_shape_decode.py`, `run_linearity.py`,
    # `recheck_register_window.py` and two test files all read them, and D50/D53/D79/D80
    # rest on them. An asset register that flags live assets as dead is worse than none.
    referenced = set()
    for f in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py")) \
            + list((ROOT / "src").rglob("*.py")):
        try:
            body = f.read_text()
        except OSError:
            continue
        for k in banked:
            if k in body:
                referenced.add(k)
    orphan = sorted(set(banked) - {k for r in rs for k in r["kernels"]} - referenced)
    lines += [
        "",
        "## Banked data no claim references",
        "",
        "Directories holding `.npy`/`.json` that NO ledger row cites AND no script,", "test or module reads. Each is either a dead",
        "run worth deleting or an unanalysed asset worth mining — and this project has",
        "already found one of the latter (three banks re-read with a new statistic became",
        "D138/D141 at zero GPU cost).",
        "",
    ]
    lines += [f"- `scratch/{k}` — {banked[k]} artifact(s)" for k in orphan] or ["- none"]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rs)} claims, "
          f"{len(unlinked)} unlinked, {len(orphan)} orphan banks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
