"""One command that answers "what is running, what finished, what died, and what is salvageable".

WHY THIS EXISTS. On 2026-08-09 this project had five jobs in flight across two
platforms and lost time three separate times to the same shape of problem: a job
whose status said one thing while its raw log said another. The failures were
`bash: Killed` (host OOM) misdiagnosed as a network error, a `CUDA out of memory`
that still exited `SUCCESS` and printed a scientific verdict over zero data, and an
`ERROR` job whose results were in fact fully recoverable. Hand-polling each job with
a different CLI, and reading only the status field, is what made those expensive.

WHAT IT DOES.
  * one status line per job, both platforms, from a registry file;
  * for anything finished, it pulls the log and greps for the failure signatures
    that this project has actually hit -- not a generic "error" grep;
  * it reports whether a partial result is RECOVERABLE, because on DataSphere the
    declared `outputs:` file is collected even on ERROR if the job wrote it
    incrementally.

REGISTRY: `scratch/RUNS.json`, a list of {platform, id, what}. Keep it current; a
job missing from it is a job nobody is watching.

Run:  uv run python -m scripts.runs_status          # status only
      uv run python -m scripts.runs_status --pull   # also download finished jobs
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

REG = os.path.join("scratch", "RUNS.json")
PULL_DIR = os.path.join("scratch", "_pulled")
DS_ENV = {**os.environ, "GRPC_DNS_RESOLVER": "native"}   # required on this Mac

# Failure signatures this project has actually hit, most-specific first. The order
# matters: a `Killed` line and a `RemoteDisconnected` line can appear in the SAME
# log, and the network one is the red herring.
SIGNATURES = [
    (r"\bKilled\b", "HOST-RAM OOM (SIGKILL). Not a network error even if one is "
                    "also in the log. Build big models on the device, not on CPU."),
    (r"CUDA out of memory|OutOfMemoryError",
     "GPU OOM. Two 3.5B fp32 models do not fit in one T4 process -- "
     "split the arms into separate jobs."),
    (r"ModuleNotFoundError: No module named 'traj_geom'",
     "pip/python env mismatch. sys.path.insert(0, abspath('src')) after chdir."),
    (r"SELF-CHECK FAILED", "The instrument failed its own null and halted. This is "
                           "correct behaviour -- fix the extraction, do not lower the tolerance."),
    (r"VOID, NOT NULL|VOID, not null", "The run declared itself VOID. Its numbers may "
                                       "NOT be read as a null result."),
    (r"Traceback \(most recent call last\)", "Python exception."),
]

# Lines worth surfacing even on success -- the ones that carry a verdict or a gate.
VERDICT_PAT = re.compile(
    r"(VOID|NOT CONFIRMED|CONFIRMED|PREDICTION|GATE|banked \d+/\d+|SELF-CHECK|"
    r"statistics clearing|separating the arms|DONE)", re.I)


def sh(cmd: list[str], env=None) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                              env=env).stdout
    except Exception as exc:
        return f"<failed: {type(exc).__name__}: {exc}>"


def ds_status(job_id: str) -> str:
    out = sh(["datasphere", "project", "job", "get", "--id", job_id], env=DS_ENV)
    m = re.search(r"(EXECUTING|SUCCESS|ERROR|CANCELLED|PREPARING|CREATING)",
                  out.splitlines()[-1] if out.splitlines() else "")
    return m.group(1) if m else "?"


def kaggle_status(slug: str) -> str:
    env = {**os.environ, "KAGGLE_CONFIG_DIR": os.path.expanduser("~/.kaggle")}
    out = sh(["kaggle", "kernels", "status", slug], env=env)
    m = re.search(r"(RUNNING|COMPLETE|ERROR|QUEUED|CANCEL\w*)", out)
    return m.group(1) if m else ("?" + out.strip()[:40])


def ds_pull(job_id: str) -> str | None:
    d = os.path.join(PULL_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    sh(["datasphere", "project", "job", "download-files", "--id", job_id,
        "--with-logs", "--output-dir", d], env=DS_ENV)
    return d if os.path.isdir(d) and os.listdir(d) else None


def diagnose(d: str) -> list[str]:
    """Read the RAW logs, not the status. Returns human-readable notes."""
    notes = []
    text = ""
    for name in ("stdout.log", "stderr.log"):
        f = os.path.join(d, name)
        if os.path.exists(f):
            text += open(f, errors="replace").read()
    for pat, msg in SIGNATURES:
        if re.search(pat, text):
            notes.append(f"    ! {msg}")
    # recoverable partial output?
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            p = os.path.join(d, f)
            try:
                obj = json.load(open(p))
                rows = obj.get("rows", obj) if isinstance(obj, dict) else obj
                n = len(rows) if isinstance(rows, list) else "?"
                notes.append(f"    + RECOVERABLE: {f} holds {n} records "
                             f"({os.path.getsize(p)/1e6:.1f} MB)")
            except Exception:
                notes.append(f"    + {f} present but not JSON-parseable")
    verdicts = [ln.strip() for ln in text.splitlines() if VERDICT_PAT.search(ln)]
    for v in verdicts[-6:]:
        notes.append(f"    | {v[:150]}")
    return notes


def main() -> None:
    pull = "--pull" in sys.argv
    if not os.path.exists(REG):
        print(f"no registry at {REG}. Create it as a list of "
              f'{{"platform": "datasphere"|"kaggle", "id": "...", "what": "..."}}')
        return
    runs = json.load(open(REG))
    print(f"{'platform':>11} {'status':>10}  what")
    finished = []
    for r in runs:
        st = ds_status(r["id"]) if r["platform"] == "datasphere" else kaggle_status(r["id"])
        print(f"{r['platform']:>11} {st:>10}  {r['what']}  [{r['id']}]")
        if st in ("SUCCESS", "ERROR", "COMPLETE", "CANCELLED"):
            finished.append((r, st))
    # CAPACITY, NOT JUST OCCUPANCY. On 2026-08-09 this tool listed running jobs and
    # never said what was FREE, and the author consequently spent hours doing
    # "zero-GPU" work while DataSphere sat idle -- anchoring on Kaggle's 2-slot cap
    # because that was the constraint most recently hit. A monitor that shows only
    # what is busy invites exactly that error.
    kag_running = sum(1 for r, s in
                      [(r, ds_status(r["id"]) if r["platform"] == "datasphere"
                        else kaggle_status(r["id"])) for r in runs]
                      if r["platform"] == "kaggle" and s in ("RUNNING", "QUEUED"))
    ds_running = sum(1 for r, s in
                     [(r, ds_status(r["id"]) if r["platform"] == "datasphere"
                       else kaggle_status(r["id"])) for r in runs]
                     if r["platform"] == "datasphere" and s in ("EXECUTING", "PREPARING"))
    # SLOTS ARE NOT THE ONLY KAGGLE LIMIT. On 2026-08-09 a push was refused with
    # "Maximum weekly GPU quota of 30.00 hours reached" while a slot was free -- so
    # "1/2 used" read as available when nothing could be launched at all. Occupancy
    # and entitlement are different questions and this line used to conflate them.
    print("CAPACITY NOTE 2 (2026-08-09): DataSphere is NOT unlimited either. With one\n"
          "  gt4.1 job EXECUTING, a second launch is refused with 'Instance types gt4.1\n"
          "  are not available for your community' -- and g4i.1 is refused too, so the\n"
          "  constraint is CONCURRENT GPU INSTANCES, not the type. An earlier revision of\n"
          "  this line said 'NO SLOT CAP -- always launchable'; that was wrong.")
    print("CAPACITY NOTE: a free Kaggle SLOT does not mean Kaggle is usable -- the\n"
          "  weekly 30 h GPU quota is separate and, once spent, refuses every push.\n"
          "  Confirm with a real `kaggle kernels push` before planning around it.")
    print(f"\nCAPACITY  kaggle {kag_running}/2 used"
          f"{'  <-- FULL' if kag_running >= 2 else f'  ({2 - kag_running} FREE)'}"
          f"   |   datasphere {ds_running} running, GPU CONCURRENCY LIMITED -- see note "
          f"(personal account, budget confirmed)")
    if kag_running >= 2 and ds_running == 0:
        print("  ! Kaggle is full and DataSphere is idle. Do not call this "
              "'no GPU available'.")

    if pull and finished:
        print("\n=== pulling finished jobs and reading their RAW logs ===")
        for r, st in finished:
            if r["platform"] != "datasphere":
                print(f"  {r['what']}: kaggle -- pull with "
                      f"`kaggle kernels output {r['id']} -p <dir>` (slow, background it)")
                continue
            d = ds_pull(r["id"])
            print(f"  {r['what']} [{st}] -> {d}")
            if d:
                for n in diagnose(d):
                    print(n)


if __name__ == "__main__":
    main()
