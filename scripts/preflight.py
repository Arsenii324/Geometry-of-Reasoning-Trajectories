"""The two checks that would have prevented most of 2026-08-09's wasted work.

WHY THIS EXISTS. A cross-cutting review of that day's errors found they are not
twelve independent mistakes but **two recurring structures**, with different fixes:

  RC1 -- A PROXY SUBSTITUTED FOR THE THING (9 of 12 cases). The real check was
         available and cheap, and a cheaper signal was used instead:
           a local commit          for  the remote's contents   (2.5 h lost)
           `str.replace` not erroring for  the edit landing      (a thread reported
                                                                  closed for 2 h)
           distinct KEYS           for  distinct PROMPTS         (D110: 42 "units"
                                                                  that were 36)
           peak accuracy > floor   for  the ladder having range  (D118: VOID)
           `pairs[:N]`             for  a sample                 (D108: 75% -> 31%)
           a printed verdict       for  the result               (D110's p = 1.0000)
           reasoning about a file  for  opening it               (the CLRS output,
                                                                  which retracted
                                                                  half of D105)
  RC2 -- ACTED WITHOUT CONSULTING THE EXISTING RECORD (the rest, and the costlier
         ones). The knowledge was in the repo and was not looked up:
           B3 called "never done" while `ds_eigen` had answered it -> GPU spent
             re-obtaining corroboration that already existed (D119)
           B4.8 called "the last untested topological claim" while D64 closed it,
             and the existing instrument was OVERWRITTEN and its known-bad
             predecessor reinvented (D123)
           DR1 Stage 2 named the exact ladder the project needed; B4b built a
             different one instead and went VOID (D118, D124)
           D106's "20.6% radial component" carried as a measurement when it is
             |d|/(2R), an identity of the sphere (D122)

RC1 is fixed by asserting. RC2 is fixed by LOOKING FIRST, which is what this script
automates, because "remember to check" has demonstrably not worked.

USAGE
    python scripts/preflight.py --launch scratch/ds_foo    # before any remote job
    python scripts/preflight.py --prior "persistent homology" "surrogate"
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEARCH = ["docs/claims_ledger.md", "docs/directions.md", "docs/OPEN_THREADS.md",
          "docs/PLAN.md", "docs/UNDERSTANDING.md", "docs/related_work.md"]
DR_DIR = ROOT.parent / "files" / "deep_research_battery"


def _sh(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def check_launch(job_dir: str) -> int:
    """Everything that has actually broken a remote run, checked before spending one."""
    bad = 0
    print(f"PREFLIGHT for {job_dir}\n")

    # 1. Unpushed commits. A4b died on this after 2.5 h and 256 completed forwards:
    #    remote jobs `git clone` from GitHub, so local commits are invisible to them.
    ahead = _sh("git", "status", "-sb").splitlines()[:1]
    if ahead and "ahead" in ahead[0]:
        print(f"  FAIL  unpushed commits: {ahead[0]}")
        print("        remote jobs clone from GitHub -- push before launching")
        bad += 1
    else:
        print("  ok    no unpushed commits")

    # 2. Repo imports the kernel needs must exist in the tree that will be cloned.
    d = ROOT / job_dir
    src = next((p for p in (d / "job.py", d / "main.py") if p.exists()), None)
    if src is None:
        print(f"  FAIL  no job.py or main.py in {job_dir}")
        return bad + 1
    text = src.read_text()
    for mod, names in re.findall(r"from (traj_geom[\w.]*) import ([^\n]+)", text):
        path = ROOT / "src" / (mod.replace(".", "/") + ".py")
        if not path.exists():
            print(f"  FAIL  {mod} not found at {path.relative_to(ROOT)}")
            bad += 1
            continue
        body = path.read_text()
        for name in [n.strip() for n in names.split(",")]:
            if f"def {name}" not in body and f"{name} =" not in body:
                print(f"  FAIL  {mod}.{name} does not exist -- the remote clone will "
                      f"ImportError")
                bad += 1
            else:
                print(f"  ok    {mod}.{name} exists")

    # 3. Late repo imports. A4b's sat in the analysis block, so it failed after every
    #    forward had run instead of in the first 30 seconds.
    lines = text.split("\n")
    late = [i for i, ln in enumerate(lines)
            if "from traj_geom" in ln and i > len(lines) * 0.6]
    if late:
        print(f"  WARN  traj_geom import at line {late[0] + 1} of {len(lines)} -- late "
              f"imports turn a 30-second failure into an hours-long one")

    print(f"\n{'PREFLIGHT FAILED' if bad else 'PREFLIGHT PASSED'} ({bad} blocking)")
    return bad


def check_prior(terms: list[str]) -> int:
    """Has this already been done? RC2's automation.

    Searches the documents AND `scratch/*/out/`, because both times a thread was
    wrongly called unrun, the answer was banked under a RUN name while the index
    was keyed by THREAD name.
    """
    print(f"PRIOR WORK for {terms}\n")
    hits = 0
    for term in terms:
        pat = re.compile(re.escape(term), re.I)
        print(f"--- {term!r} ---")
        for rel in SEARCH:
            p = ROOT / rel
            if not p.exists():
                continue
            for i, ln in enumerate(p.read_text().split("\n"), 1):
                if pat.search(ln):
                    print(f"  {rel}:{i}  {ln.strip()[:150]}")
                    hits += 1
        if DR_DIR.exists():
            for p in sorted(DR_DIR.glob("*.md")):
                for i, ln in enumerate(p.read_text().split("\n"), 1):
                    if pat.search(ln):
                        print(f"  DR/{p.name}:{i}  {ln.strip()[:150]}")
                        hits += 1
        banked = [d.name for d in (ROOT / "scratch").iterdir()
                  if d.is_dir() and (d / "out").is_dir()
                  and any((d / "out").iterdir()) and pat.search(d.name)]
        for b in banked:
            print(f"  BANKED OUTPUT  scratch/{b}/out/  <- results already exist")
            hits += 1
            # AND SEARCH THE DOCS UNDER THE RUN NAME, NOT ONLY THE DIRECTORY NAME.
            # This is the blind spot that made D120 duplicate D52: the directory is
            # `kaggle_rho_ckpt_a`, the ledger cites `geometry-rho-ckpt-a`, and a
            # directory-name grep finds nothing. Strip the platform prefix and try
            # the run-name forms the ledger actually uses.
            stem = re.sub(r"^(kaggle|ds)_", "", b)
            for alias in {stem, stem.replace("_", "-"),
                          "geometry-" + stem.replace("_", "-"),
                          "geom-" + stem.replace("_", "-")}:
                apat = re.compile(re.escape(alias), re.I)
                for rel in SEARCH:
                    fp = ROOT / rel
                    if not fp.exists():
                        continue
                    for i, ln in enumerate(fp.read_text().split("\n"), 1):
                        if apat.search(ln):
                            print(f"    ALIAS {alias!r} -> {rel}:{i}  "
                                  f"{ln.strip()[:110]}")
                            hits += 1
    print(f"\n{hits} prior references. Read them BEFORE building "
          f"(D119, D123, D124 were all this).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--launch", metavar="JOB_DIR")
    ap.add_argument("--prior", nargs="+", metavar="TERM")
    a = ap.parse_args()
    if a.launch:
        return check_launch(a.launch)
    if a.prior:
        return check_prior(a.prior)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
