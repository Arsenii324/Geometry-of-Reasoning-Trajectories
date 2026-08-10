"""A recurring STRATEGIC audit: what is the shape of the bottleneck right now?

WHY THIS EXISTS, AND WHY IT IS NOT THE OTHER KIND OF AUDIT. The supervisor asked for a
periodic audit and I first built the wrong thing — a rule-compliance check (did I use
`-F`, did I reimplement a quantity). That kind is worth doing occasionally and it is
mechanical. What he actually meant is the kind that surfaces a *structural* move at the
moment it becomes relevant:

    "we need to push a battery of tasks if we want to find the good ones"
    "we need to speed up weights loading in the long term"

Both of those are real examples from this project's history, and in both cases the
observation was available for days before anyone acted on it. The first eventually became
the observability census (D130) after **three** experiments had already died on task
supply. The second was measured, written into `REMOTE_RUNS.md` with a verified mechanism,
and then not acted on for a day while ten more jobs each paid the 262-282 s it would have
removed.

Neither was a rule violation. Both were **repeated costs nobody had added up**, and the
signal was sitting in data this repo already had. So this script adds them up.

WHAT IT DOES. Assembles the state that a strategic question needs an answer to — what is
being paid repeatedly, what has been attempted and failed how many times, what is queued,
what the recent record is actually about — and prints it beside the questions. It does not
answer them; that is the point. A script cannot tell you the binding constraint, but it can
stop you from estimating one from memory.

USE. Run it at intervals, or whenever a run lands. Answer the five questions in the output
honestly. If the answers are the same as last time, that is itself information: either the
constraint is real and unaddressed, or the audit is not looking at the right thing.

    uv run python scripts/strategic_audit.py
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sh(*a: str) -> str:
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout


def repeated_costs() -> list[str]:
    """Fixed per-job costs, multiplied by how many jobs have paid them."""
    out = []
    kernels = sorted(ROOT.glob("scratch/*/job.py")) + sorted(ROOT.glob("scratch/*/main.py"))
    clones = sum("git clone" in k.read_text() for k in kernels)
    torch = sum("pip install torch" in k.read_text() for k in kernels)
    hf = sum(("from_pretrained" in k.read_text()) for k in kernels)
    out.append(f"{clones} kernels `git clone` the repo at launch")
    out.append(f"{torch} kernels `pip install torch` (~390 s measured)")
    out.append(f"{hf} kernels download model weights from HF (~262-282 s measured)")
    # KERNELS ARE NOT RUNS, and the first version of this script quoted the kernel
    # count as though they were -- RC5, committed in the audit tool's own first output.
    # The real denominator is jobs actually executed; count the ones that produced an
    # out/ directory, and note the platform total separately since it is not derivable
    # from the filesystem.
    ran = len(list(ROOT.glob("scratch/*/out")))
    out.append(f"{ran} kernels have an out/ dir, i.e. actually ran at least once")
    out.append(f"  -> weights alone: >= ~{ran * 272 / 3600:.1f} GPU-HOURS paid on runs "
               f"visible locally, for a phase REMOTE_RUNS.md records a verified way to "
               f"remove. `datasphere job list` shows the true job count is higher.")
    reg = ROOT / "scratch/RUNS.json"
    if reg.exists():
        import json as _j
        try:
            n = len(_j.loads(reg.read_text()))
            out.append(f"RUNS.json registers {n} runs -- the operational registry that "
                       f"`runs_status.py` reads. If that is far below the platform's job "
                       f"count, the registry is not a registry.")
        except Exception:
            pass
    return out


def repeated_attempts() -> list[str]:
    """Things tried more than twice — a strong hint the approach, not the run, is wrong."""
    led = (ROOT / "docs/claims_ledger.md").read_text()
    out = []
    for label, pat in (("difficulty ladders", r"ladder"),
                       ("capability/task screens", r"census|capability screen|screen"),
                       ("patching / causal designs", r"patch")):
        n = len(re.findall(pat, led, re.I))
        rows = {m for m in re.findall(r"\| (D\d+) \|", led)}
        out.append(f"{label}: {n} mentions across {len(rows)} rows")
    return out


def recent_record(n: int = 12) -> list[str]:
    """What the last n rows are ABOUT -- are we adding knowledge or repairing it?"""
    led = (ROOT / "docs/claims_ledger.md").read_text()
    rows = [l for l in led.split("\n") if l.startswith("| D")]
    rows = sorted(rows, key=lambda l: int(re.match(r"\| D(\d+) \|", l).group(1)))[-n:]
    kinds = collections.Counter()
    for l in rows:
        rid = re.match(r"\| (D\d+) \|", l).group(1)
        claim = l.split("|")[2]
        if re.search(r"RETRACT|WITHDRAWN|CORRECT|WRONG|ARTEFACT|DEFECT", claim, re.I):
            kinds["repair of our own record"] += 1
        elif re.search(r"\bNOT\b|NULL|CANNOT|FAILS|no .*consequence", claim, re.I):
            kinds["a bounded negative"] += 1
        else:
            kinds["a new positive"] += 1
    return [f"last {n} rows: " + ", ".join(f"{v} {k}" for k, v in kinds.most_common())]


def blocked() -> list[str]:
    ot = ROOT / "docs/OPEN_THREADS.md"
    if not ot.exists():
        return ["no OPEN_THREADS.md"]
    txt = ot.read_text()
    live = [l for l in txt.split("\n")
            if re.match(r"\| [A-Z]\d+ \|", l) and "LANDED" not in l]
    reasons = collections.Counter()
    for l in live:
        m = re.search(r"blocked on ([a-z ]+)", l, re.I)
        reasons[m.group(1).strip() if m else "not stated"] += 1
    return [f"{len(live)} threads not marked landed; blockers: {dict(reasons)}"]


QUESTIONS = [
    "1. WHAT IS THE BINDING CONSTRAINT RIGHT NOW? Not the last failure -- the thing that\n"
    "   would still bite if the last failure were fixed. (Past answers: task supply, then\n"
    "   the capability axis, then the model itself.)",
    "2. WHAT AM I PAYING REPEATEDLY THAT COULD BE PAID ONCE? Look at the cost table above\n"
    "   and multiply. A 4-minute phase across 15 jobs is an hour.",
    "3. WHAT CLASS OF RESULT HAVE I BEEN UNABLE TO PRODUCE -- and is that a MODEL limit,\n"
    "   an INSTRUMENT limit, or a DESIGN limit? These have different fixes and only one\n"
    "   of them is fixed by more GPU.",
    "4. WHAT AM I ABOUT TO DO THAT A PAST FAILURE PREDICTS WILL FAIL? Check the repeated\n"
    "   attempts above before queueing another of the same shape.",
    "5. WHAT WOULD I DO WITH 10x THE COMPUTE? WITH A TENTH? If the answers are the same,\n"
    "   compute is not the constraint and I should stop treating it as one.",
]


def main() -> int:
    print("=" * 74)
    print("STRATEGIC AUDIT -- state first, then the questions it exists to inform")
    print("=" * 74)
    for title, fn in (("REPEATED FIXED COSTS", repeated_costs),
                      ("REPEATED ATTEMPTS", repeated_attempts),
                      ("WHAT THE RECENT RECORD IS ABOUT", recent_record),
                      ("OPEN THREADS", blocked)):
        print(f"\n## {title}")
        for line in fn():
            print(f"  {line}")
    print("\n## GIT")
    print(f"  {len(sh('git', 'log', '--oneline', '--since=1 day ago').splitlines())} "
          f"commits in the last day")
    print("\n" + "=" * 74)
    print("QUESTIONS -- answer these, do not skim them")
    print("=" * 74)
    for q in QUESTIONS:
        print(f"\n{q}")
    print("\nIf the answers match the last run of this audit, either the constraint is real")
    print("and unaddressed, or this audit is measuring the wrong state. Both are actionable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
