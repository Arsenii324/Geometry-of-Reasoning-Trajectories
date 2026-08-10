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
import ast
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEARCH = ["docs/claims_ledger.md", "docs/directions.md", "docs/OPEN_THREADS.md",
          "docs/PLAN.md", "docs/UNDERSTANDING.md", "docs/related_work.md"]
DR_DIR = ROOT.parent / "files" / "deep_research_battery"


def _sh(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _run_builder(src: pathlib.Path) -> set[str] | None:
    """Import a kernel and call its `build_items()`, returning the keys it really emits.

    Returns None if the kernel cannot be imported or has no zero-argument builder --
    a kernel that needs the GPU to build its items is not a failure of this check.

    The point of running it: BOTH of the launch failures on 2026-08-10 were in code
    that needs no GPU and executes in under a second locally, and neither was ever
    executed before being sent to a machine that charges for a 12-minute setup first.
    """
    import importlib.util

    # REFUSE TO IMPORT AN UNGUARDED MODULE. Importing executes module scope, so a
    # kernel whose `main()` is called at top level would RUN here -- cloning a repo,
    # pip-installing, and downloading 15 GB of weights onto the developer's machine,
    # from a command whose entire purpose is to be a cheap pre-launch check.
    #
    # The docstring above claimed module scope is pure "by the project's entrypoint
    # rule". That rule is a convention, not an invariant, and `scratch/kaggle_blockbank/
    # main.py` breaks it -- it defines `build_items` with no `if __name__` guard. The
    # earlier sweep missed it only because that sweep filtered on `job.py` and this
    # kernel is `main.py`. A convention is not a safety property; check it.
    try:
        tree = ast.parse(src.read_text())
    except SyntaxError:
        return None
    guarded = any(isinstance(n, ast.If) and "__main__" in ast.dump(n.test)
                  for n in tree.body)
    toplevel_calls = [n for n in tree.body
                      if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    if not guarded or toplevel_calls:
        return None

    spec = importlib.util.spec_from_file_location(f"_kernel_{src.parent.name}", src)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        fn = getattr(mod, "build_items", None)
        if fn is None:
            return None
        try:
            items = fn()
        except TypeError:
            import random as _r
            items = fn(_r.Random(0))          # some builders take an rng
        return {k for it in items for k in it} if items else set()
    except Exception:
        return None


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

    # 4. Third-party imports the kernel's OWN install line does not provide.
    #    A21 died 12 minutes in on `No module named 'scipy'`, after paying the full
    #    clone + torch + weights cost, because TWO install recipes coexist in this
    #    repo and they are not interchangeable:
    #      `pip install -e .[model]`                 -> repo deps, so scipy included
    #      `pip install transformers accelerate ...` -> leaner, and scipy is NOT there
    #    Both are legitimate; the lean one is faster. What is not legitimate is using
    #    the lean recipe and then importing a repo dependency, which is invisible
    #    until the line that needs it runs -- and this project puts third-party
    #    imports LATE on purpose, so that is the worst possible place to find out.
    # Comments are STRIPPED first. The first version of this check scanned the raw
    # file and was defeated by the explanatory comment added to `ds_gmres` next to
    # the fix -- that comment names `pip install -e .[model]`, so the check read a
    # comment as an install line, concluded the repo deps were present, and passed
    # the very file whose bug it was written for.
    code = "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("#"))
    installs = re.findall(r"pip install ([^\n]+)", code)
    editable = any("-e ." in s for s in installs)
    provided = {"torch", "numpy", "traj_geom"}       # torch is its own install line
    if editable:
        pyproject = (ROOT / "pyproject.toml").read_text()
        provided |= {m.replace("-", "_") for m in
                     re.findall(r'^\s*"([A-Za-z][\w-]*)', pyproject, re.M)}
        provided |= {"sklearn", "yaml"}              # scikit-learn, pyyaml import names
    for spec in installs:
        for tokenpart in spec.split():
            if tokenpart.startswith("-"):
                continue
            # strip the QUOTES too: several kernels write
            #   pip install -q 'transformers>=4.50,<4.54' scipy
            # and a capture that stopped at the first quote saw only "-q",
            # which flagged three already-successful kernels as broken.
            # strip trailing shell/Python punctuation as well as quotes: the line
            #   run("pip install ... scipy")
            # ends the last token as `scipy")`, and stripping quotes alone leaves
            # the paren, so `scipy` was still reported missing after being added.
            name = re.split(r"[=<>\[]", tokenpart.strip("\"')(,;"))[0]
            provided.add(name.replace("-", "_"))
    # Imports guarded by `try:` are OPTIONAL and must not be flagged. Three kernels
    # do `if os.environ.get("WANDB_API_KEY"): try: import wandb`, which is a
    # deliberate soft dependency -- reporting it would train the reader to ignore
    # this check, which is worse than not having it.
    code_lines = code.split("\n")
    guarded = {i for i, ln in enumerate(code_lines)
               if i and code_lines[i - 1].strip().rstrip(":") == "try"}
    code = "\n".join(ln for i, ln in enumerate(code_lines) if i not in guarded)

    imported = set(re.findall(r"^\s*import (\w+)", code, re.M))
    # `from X import` REQUIRES the literal ` import `: without it the pattern matches
    # ordinary docstring prose ("from the model's own init, ...") and reported a
    # missing package called `the` in two kernels.
    imported |= set(re.findall(r"^\s*from (\w+)[\w.]* import ", code, re.M))
    missing = sorted(m for m in imported - provided
                     if m not in sys.stdlib_module_names and not m.startswith("_"))
    if missing:
        for m in missing:
            print(f"  FAIL  imports `{m}` but no pip install line provides it"
                  f"{' (editable install present, so check pyproject)' if editable else ''}")
        print("        this fails only when the importing line RUNS -- A21 lost 12 min")
        bad += len(missing)
    else:
        print(f"  ok    every third-party import is covered by an install line")

    # 5. Item keys copied wholesale onto a result row that build_items() never sets.
    #    B4c died with `KeyError: 'array'` on its FIRST forward -- 12 minutes of clone,
    #    install and weight download before the first line that touched an item --
    #    from exactly this shape:
    #        r.update({k: it[k] for k in ("array", "pos", "depth", "key", ...)})
    #    while build_items() emitted `item` and `k`.
    #
    #    This targets that shape ONLY, and the narrowness is the result of a failed
    #    first attempt rather than caution in advance: a version that also checked
    #    direct `it["..."]` subscripts MISSED this bug (the keys are comprehension
    #    constants, not subscripts) while flagging `n_tokens` in five working kernels,
    #    where the subscripted variable was a result row and not an item at all.
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        print(f"  FAIL  job.py does not parse: {exc}")
        bad += 1
        tree = None
    builder = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == "build_items"), None) if tree else None
    if builder is not None:
        produced = {k.value for d in ast.walk(builder) if isinstance(d, ast.Dict)
                    for k in d.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}

        # RUN the builder rather than reading it, when that is possible. This is the
        # actual object; the AST walk above is a proxy for it, and substituting a
        # proxy for the thing is the single most common error in this project's
        # post-mortems. `_run_builder` imports ONLY files that carry an `if __name__`
        # guard and no top-level calls, and falls back to the static keys otherwise --
        # an earlier version of this comment asserted that the project's entrypoint
        # convention made every kernel safe to import, which is false: one kernel
        # breaks it and importing that one would have downloaded 15 GB of weights.
        real = _run_builder(src)
        if real is not None:
            extra = sorted(real - produced)
            print(f"  ok    build_items() RAN locally: {len(real)} keys"
                  + (f" (+{', '.join(extra)} not visible statically)" if extra else ""))
            produced |= real
        # keys ASSIGNED after construction count as produced: several kernels do
        #   it["n_tokens"] = int(la)
        # outside build_items() and then copy it, which is correct and must not flag.
        produced |= {n.slice.value for n in ast.walk(tree)
                     if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
                     and isinstance(n.slice, ast.Constant)
                     and isinstance(n.slice.value, str)}
        requested = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.DictComp):
                continue
            for gen in node.generators:
                # the value must actually SUBSCRIPT something by the loop variable,
                # which is what makes the constants item keys rather than plain data
                if not (isinstance(node.value, ast.Subscript)
                        and isinstance(node.value.slice, ast.Name)
                        and isinstance(gen.target, ast.Name)
                        and node.value.slice.id == gen.target.id):
                    continue
                if isinstance(gen.iter, (ast.Tuple, ast.List)):
                    requested |= {e.value for e in gen.iter.elts
                                  if isinstance(e, ast.Constant)
                                  and isinstance(e.value, str)}
        unknown = sorted(requested - produced)
        if unknown:
            for k in unknown:
                print(f"  FAIL  copies item key '{k}' that build_items() never sets "
                      f"(it sets: {', '.join(sorted(produced))})")
            print("        B4c lost a full setup cycle to exactly this KeyError")
            bad += len(unknown)
        elif requested:
            print(f"  ok    all {len(requested)} copied item keys exist in build_items()")

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
