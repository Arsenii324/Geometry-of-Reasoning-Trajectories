# Reliability practice for this project

*Derived from this repo's own failure record — 12+ retracted claims historically,
and ~10 errors caught in a single day (2026-08-09), each with a traceable cause.
Absorbs and replaces the standalone guidance retrospective. Organised by **where
errors enter**, because that is what determines which guard can catch them.*

**The one-line version.** Errors here never announce themselves: nothing crashes,
the number is plausible, and the plot looks fine. So the objective is not
correctness-by-review — it is **arranging for the wrong answer to be loud**. Every
practice below is judged by that, and a practice that produces noise instead of
signal is worse than nothing because it teaches you to ignore the alarm.

---

## Stage 1 — Writing a claim (most frequent, cheapest to prevent)

**The failure:** quoting a number from memory, from a printed summary, or from a
convenience slice, instead of re-deriving it from the artifact.

Instances: D98's residual range (one prompt's four per-block values quoted as a
six-prompt range — matched *no* prompt); D99's `sorted(glob)[:40]` (32 replicates
of one item + 8 of another, quoted as "40 independent orbits"; true median 2.6×
smaller); D94's corroborating interval (attributed to D52, actually the
single-orbit fit our own `regime.py` calls "measured by mistake"); a random-sphere
baseline simulated at R=76.0 instead of the measured 76.386.

**Practice.**
- Every number in a claim carries, in the same turn, the command that produced it.
  Not "I recall it was X" and not "the summary said X".
- **A range or an `n` is always reported with the count of independent units
  behind it.** `describe_range` in `traj_geom.rigor` formats it so the unit count
  cannot be omitted.
- **A convenience slice is never a sample.** `[:40]`, `head`, glob order — all
  produce something that looks like a sample and is not.
- This extends past research numbers to *operational* assertions — job status, what
  an agent returned, whether a branch is merged. Same failure, different object;
  it caused nearly every correction the supervisor had to make in one day.

## Stage 2 — Designing an instrument (rarest, most expensive when missed)

**The failure:** the instrument cannot answer the question, and says so only in
ways you have to look for.

Instances: a detection floor using 40 permutations against α = 0.00625, whose
minimum attainable p (1/41) *cannot* clear the threshold, so it reported 0%
detection at every effect size including 1.0 sd; a QK extraction spy capturing the
*prelude* block instead of `core_block[-1]` (error ~15, i.e. a different layer's
weights); a permutation p of 0.00015 below its own exact floor of 1/4096.

**Practice.**
- **The instrument passes its own null before it produces any number, and halts if
  it fails.** The QK probe did exactly this and its first submission stopped before
  emitting a single result. That is the discipline working, not a run wasted.
- **Check the test can reject before running it.** `require_resolvable_alpha`
  raises when `1/(n_perm+1) ≥ α`. When the permutation space is smaller than the
  number of draws, enumerate it exactly (`require_exact_test_if_small`).
- **Pre-register the prediction, the null, and the detection floor** in the job's
  docstring, before the data exists. This is what made D104 informative: it
  *refuted* what was written down, which a post-hoc reading would have absorbed.
- **Reproduce a bug locally before re-spending GPU.** A tiny structural fake using
  the real classes at tiny dimensions proved both the QK bug and its fix for zero
  compute.
- **Dry-run generators.** A task generator that was constant in its own loop index
  (`(i*5 + j*3) % 3`) made every item in a family identical; a gold answer's token
  count covaried with the difficulty knob; a family declared length-constant
  measured 28–29 tokens. All caught before submission by evaluating the functions.

## Stage 3 — Interpreting a result

**The failure:** the number is right and means something other than what is claimed.

Instances: pseudoreplication four times over (960 pairs from 16 prompts; 24
contrasts from 12 items; 3 items per level pooled); `correct = min(ranks)==1`,
an *oracle over depth* inflating accuracy 2.06× (22.7% → 11.0% at the best
selectable fixed depth, 0.7% at final depth); first-token-only gold scoring
affecting 8 of 21 families.

**Practice.**
- **`n` is a count of independent units, never of rows.** Contraction is a property
  of the prompt; correctness across h₀ draws is a property of the item.
  `require_units` raises when replicates are being counted as observations.
- **Every aggregate states what it maximises or pools over.** "Accuracy" that takes
  a minimum over 48 depths is an upper bound on capability, not capability.
- **A null needs its detection floor** or it is indistinguishable from an
  insensitive design. Report the effect size the design *could* have seen.
- **VOID is not NULL.** If the measurement did not happen, say the question is
  open. `require_evidence` refuses to enter a verdict branch on an empty result
  set — added after a script printed "the cycle is ARCHITECTURAL" over zero
  measurements.

## Stage 4 — Propagating a correction

**The failure:** the fix lands in `claims_ledger.md` and nowhere else. **Four of six
audit findings in one day were exactly this**, with `UNDERSTANDING.md`,
`RESULT.md`, `PLAN.md` and two research inquiries still asserting retracted claims
verbatim.

**Practice.** When a claim is retracted, its *phrasing* becomes forbidden
repo-wide: add it to `RETRACTED` in `tests/test_no_retracted_claims.py`, with why
it was retracted and what to write instead. The suite then fails until every
document is updated — in the same commit, which is the point. A correction that
reaches the ledger and not the synthesis documents is not a correction.

## Stage 5 — Running and monitoring compute

**The failure:** the platform's status field disagrees with what happened.

Instances in one day: an `ERROR` job whose results were fully recoverable; a
`SUCCESS` job that printed a scientific verdict over zero measurements; a
host-RAM OOM (`bash: Killed`) misdiagnosed as the `RemoteDisconnected` sitting
*above* it in the log, which had already retried and succeeded.

**Practice.** Detail in `REMOTE_RUNS.md`; the load-bearing parts:
- **Write results incrementally, after each arm.** It is the only reason every
  completed arm survived an ERROR-status job. There is no resume.
- **Diagnose bottom-up.** The proximate cause is the last thing printed, not the
  first alarming thing. Distinguish `Killed` (host OOM), `OutOfMemoryError` (GPU),
  and a traceback (real bug) — each has a different fix and guessing costs a run.
- **One status command, not per-job polling** (`scripts/runs_status.py`), which
  reads raw logs for the signatures this project has actually hit.

---

## What NOT to do, learned the same way

- **Do not build tests for things that are not numbers.** A first version of
  `test_docs_consistency.py` policed table format — statuses present, section
  letters resolvable — and failed twice on its author's own layout while revealing
  nothing about the research. A test that fires on formatting spends the suite's
  credibility. Guards belong on quantities and on retracted strings, not on shape.
- **Do not add a verifier layer on top of self-checking.** An uncapped fan-out of
  one verifier subagent per finding produced ~24 agents and zero findings, while
  every real error that day was caught by re-reading raw output directly. If a
  workflow is warranted, cap the fan-out structurally so agent count cannot scale
  with how much the finders find, and put mechanical checks on a cheaper model.
- **Do not treat a framework as the objective.** A methodology skill, a rule in
  CLAUDE.md, or a request is an *input* to judgement, not a substitute for it.
  Sections of an imported methodology that do not fit this project (verifying
  ports against reference numbers; exact-identity regression tests) should be
  dropped explicitly rather than satisfied ceremonially.
- **Do not read "archived" as "superseded".** `docs/archive/` and `h3_toy_model/`
  held work that directly bore on live claims — one of them inverted a conclusion
  written that morning. It means "not maintained", which is different.

## How the rules in CLAUDE.md behaved in practice

The rules mandating **checking** paid for themselves repeatedly: §4 (raw output,
not printed verdict) caught five wrong claims in one day, three of them ours; §1
(check the premise first) caught a degenerate generator and two confounds before
any GPU spend; §5 (an instrument needs a null) halted a probe reading the wrong
layer entirely.

The rules mandating **restraint** each cost something when read literally, because
each was written against a real past failure and reads, out of context, as broader
than intended:

| rule | literal reading | what it should mean |
|---|---|---|
| §7 no verification passes | don't re-check yourself | don't add verifier *subagent layers*; self-checking is the cheapest thing here |
| §2 code: be minimal | don't build tooling | governs *analysis* code; observability code is judged by errors prevented |
| §3 don't touch others' work | don't read their branch | don't *overwrite* or silently amend; read and cite and merge-with-renumbering |
| §6 three documents only | don't write anything down | the cap is right; but a correction must propagate, and archived ≠ superseded |

## A doc edit that prints success is not a doc edit that happened

**Occasion, 2026-08-09.** I closed thread B4 with a script whose body was
`t = t.replace(old, new)` followed by `print("OPEN_THREADS updated")`. The pattern
did not match, `replace` silently returned the string unchanged, the script printed
success, and the commit message said the thread was closed. **The stale open row
survived for two hours while I reported it as done** — and a second, redundant row
(`B4b-orig`) accumulated beside it. Found only by grepping the file to write a status
summary, not by any check at the time.

This is the same failure as `SELF_REVIEW.md` §1: reporting the state I last believed
rather than the state I last checked. It is worse here because the tool reported
success, so nothing looked wrong.

**The rule: every scripted edit to a document asserts its own precondition.**
`assert t.count(old) == 1` before replacing, or build the new file line-by-line with
an `assert found` at the end. Both cost one line. I had been doing this on some edits
and not others, which is the worst of both — the ones with the assert taught me to
trust the ones without.

## The two structures behind a day of errors

*Written 2026-08-09 after the supervisor suggested asking "why" repeatedly. I think a
linear why-chain is the wrong tool here — you can always manufacture a deeper why that
sounds profound and is untestable. What was actually missing is that each error had been
recorded individually and never compared across. Doing that, from the commit record rather
than memory, the twelve are **two structures**, and they have different fixes.*

### RC1 — a proxy substituted for the thing (9 of 12)

The real check was available and cheap, and a cheaper signal stood in for it.

| proxy trusted | the thing | cost |
|---|---|---|
| a local commit | the remote's contents | 2.5 h and 256 forwards (D125) |
| `str.replace` not erroring | the edit landing | a thread reported closed for 2 h |
| distinct keys | distinct prompts | D110: 42 "units" that were 36 |
| peak accuracy > floor | the ladder having range | D118: a VOID run |
| `pairs[:N]` | a sample | D108: 75% → 31% (D113) |
| a printed verdict | the result | D110's degenerate p = 1.0000 |
| reasoning about a file | opening it | retracted half of D105 (D124) |

**Fix: assert.** `assert t.count(old) == 1` before replacing; key units on content not
labels; gate on the property you need, not a weaker one that implies it; open the file.
These are one line each and every one of them is now either a `rigor.py` guard or a
`preflight.py` check.

### RC2 — acted without consulting the existing record (the costlier ones)

The knowledge was already in the repo. Nobody looked.

- **B3** was listed *"Never done — the only route to a second estimate of ρ"* while
  `ds_eigen` had measured the full operator a day earlier. **GPU was spent re-obtaining
  corroboration that already existed** (D119).
- **B4.8** was listed `open` in `directions.md` while `UNDERSTANDING.md` line 703 said
  **CLOSED (D64)** — the contradiction was sitting in the repo. Acting on the stale row,
  I overwrote the existing instrument and reinvented the isotropic surrogate the project
  had explicitly replaced in July (D123).
- **DR1 Stage 2** named the exact ladder the project needed and `PLAN.md` called it "the
  one escape"; B4b built a different one, which went VOID (D118, D124).
- **D106's "20.6% radial component"** was carried as a measurement for days when it is
  |d|/(2R), an identity of the sphere (D122).

**Why the index failed, specifically:** results are filed under RUN names
(`geom-eigenplane`, `ds_eigen`) and threads are indexed under THREAD names (B3, B4.8).
A grep for the thread name finds nothing and the row still reads `open`.

**Fix: `scripts/preflight.py --prior "<topic>"`**, which searches the ledger,
`directions.md`, `OPEN_THREADS`, `PLAN`, `UNDERSTANDING`, `related_work`, the DR reports
**and `scratch/*/out/`** — the last because that is where the answer actually was, both
times. Run it before building anything or calling any thread unrun. On "persistent
homology" it returns the D64 CLOSED line in one command.

### The asymmetry worth remembering

RC1 errors were caught within minutes, by me, and cost little. **RC2 errors cost GPU
money, destroyed a better instrument, and produced a VOID run — and every one of them was
invisible until something external forced a look** (the supervisor saying "use git", a
status summary needing a grep). Being careful does not catch RC2; only looking does.

### A small one, found the same day: backticks in a `-m` message

`git commit -m "... `correct` inherits ..."` ran `correct` as a command and silently
dropped the word from the message. The commit succeeded, so nothing looked wrong; the
only symptom was `(eval): command not found`. **Write non-trivial commit messages to a
file and use `-F`.** This is RC1 again in miniature: exit status 0 was trusted as a
proxy for "the message is what I wrote".

### The sharpest instance, made while writing the section above

Minutes after documenting RC1 ("a proxy substituted for the thing"), I hit
`INVALID_ARGUMENT: Instance types gt4.1 are not available for your community` on a
second concurrent launch, retried once on `g4i.1`, got the same message, and wrote
**"DataSphere runs ONE GPU job at a time"** into `runs_status.py`, `REMOTE_RUNS.md`
and a message to the supervisor.

The supervisor said he remembered parallel jobs. Checking the project's own job
timestamps: **6 overlapping pairs, including a three-way overlap at 12:26-13:08.**
Retrying the identical launch minutes later succeeded with the other job still
EXECUTING. The refusal was transient capacity.

**Two failed attempts felt like a pattern and were a sample of size two.** The rule
that generalises: *a failed attempt is evidence about that attempt.* Before writing
a platform limit into a tool other people will trust, either retry it or find the
record that already answers it — here `datasphere job get` on 13 jobs settled it in
one command, and that data existed the whole time.

Worth noting what did NOT go wrong: the claim was written into the tooling, so it was
correctable in one place. The failure mode to fear is the same inference made
silently and acted on for a week.
