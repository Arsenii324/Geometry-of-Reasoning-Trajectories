# Where the project's own guidance helped, and where reading it literally caused harm

*Written 2026-08-09 at the supervisor's request, from one long working day. Every
entry cites a specific occasion. This is a review of how the rules behaved in
practice — not a proposal to weaken them, and three of the four rules examined
earned their keep decisively.*

---

## Summary

| rule | verdict | evidence |
|---|---|---|
| §1 check the premise first | **strongly positive, no harm found** | caught 3 defects before GPU spend |
| §4 raw output, not printed verdict | **the single most valuable rule** | caught 5 wrong claims, 3 of them mine |
| §5 instrument needs a null | **strongly positive** | halted 2 runs that would have produced false numbers |
| §7 "no verification passes / re-checks" | **caused harm** | see below |
| §2 "code: be minimal" | **caused harm once** | see below |
| §3 "don't rewrite what isn't yours" | **caused harm once** | see below |
| §6 three documents only | **mixed — the cap is right, the omission is not** | see below |

---

## 1. §7 — "Do not add verification passes, re-checks, or verifier subagents"

**Harm, concrete.** I read this as a blanket prohibition and it made me slower and
less accurate, twice:

- **The `run_h0_within.py` detection floor.** Its planted-effect check used 40
  permutations against α = 0.00625, so its minimum reachable p-value (1/41 ≈
  0.024) could never clear the threshold — it reported 0% detection at *every*
  effect size including 1.0 sd. My first instinct was to accept the number and
  move on, because re-checking my own instrument felt like the "verification
  pass" §7 forbids. Only opening the raw output caught it. **Had I not, D93's
  null would have been reported as "underpowered" when the design in fact
  detects a planted 0.5 sd effect 100% of the time.**
- **The `ds_cyclenull` verdict.** A script I wrote printed *"D98's cycle is
  ARCHITECTURAL, not learned"* over **zero** untrained measurements. The guard
  that now prevents it — check the count of usable measurements before entering
  any verdict branch — is exactly the kind of self-check §7 reads as forbidding.

**But the rule is right about the thing it was written for.** The same day I
launched a workflow that spawned **one verifier subagent per finding with no cap**
(~24 agents, zero findings returned before it was paused). Meanwhile *all* four
errors caught that day came from me re-reading raw output myself.

**Practical reading, now recorded in memory:** the prohibition is on **verifier
subagent layers**, not on self-checking. Self-checks, instrument nulls, and
recomputing a number before quoting it are not "verification passes" in the sense
that costs tokens without improving results — they are the cheapest thing in the
project. When a workflow *is* right, cap the fan-out structurally
(`MAX_VERIFIERS`) so agent count cannot scale with how much the auditors find, and
put mechanical checking on a cheaper model.

## 2. §2 — "Code: be minimal"

**Harm, once, and it was expensive.** I did not build a shared run-monitoring tool
until late in the day, because a status-checking script felt like exactly the
"configurability for a one-off" the rule warns against. In the interim I
hand-polled five jobs across two CLIs and **misdiagnosed three failures in a row**:
a host-RAM OOM read as a network error, a `SUCCESS` job that had printed a verdict
over zero data, and an `ERROR` job whose results were in fact fully recoverable.
`scripts/runs_status.py` — about 120 lines — would have caught all three by reading
the raw logs for known signatures. **The rule optimised for lines of code and cost
hours of wall-clock.**

**Practical reading:** "minimal" governs *analysis* code, where an abstraction
layer over a kernel that runs three times is genuine waste. It should not govern
**observability** code, where the return is measured in errors not made. A tool
that turns "read five logs by hand and guess" into "one command that flags the
three failure modes we have actually hit" is not over-engineering.

## 3. §3 — "Do not rewrite a claim you did not measure / don't touch others' work"

**Harm, once, and it is still unresolved.** A parallel session's branch
(`claude/geometry-remote-continued`) contains a real result we do not have —
*"depth does not compute the answer, it relocates it"*, with a regeneration script
and 11 tests — plus the identification of a defect (`correct` maximised over
depth) that I independently rediscovered hours later and recorded as D103.
**I read §3 as a reason not to touch that branch, and the cost was rediscovering
its finding from scratch.** The branch is still unmerged and now carries a ledger
numbering collision (its D92–D94 are different claims from ours).

**Practical reading:** §3 protects against *overwriting* a teammate's work or
amending a claim without a measurement. It should not prevent *reading* that work,
*citing* it, or *merging* it with renumbering. The version of the rule that would
have helped: "do not overwrite or silently amend another session's claims — but
you must read them before duplicating their work."

## 4. §6 — "The three documents. These only."

**The cap is right and I would not relax it.** This repo produced 23 accumulating
files before the rule existed. But two costs are real:

- **A correction that reaches the ledger and not the synthesis documents is not a
  correction.** Four of six audit findings today were exactly this: D94's
  retracted "convergent validation" and its n = 960 pseudoreplication were fixed
  in the ledger row while `UNDERSTANDING.md`, `RESULT.md`, `PLAN.md` and both
  research inquiries still asserted them verbatim. **Guard now in place: when a
  ledger row is corrected, grep every other document for the retracted phrasing in
  the same commit.**
- **The rule discouraged me from opening `docs/archive/`,** where several
  documents were directly load-bearing on live work — `h3_toy_model/`
  (an H3 result that inverted my H3 conclusion), `power_and_preregistration.md`,
  `observability.md`, `exit_criteria.md`. "Archived" was read as "superseded". It
  means "not maintained", which is different.

## 5. What the rules got right, and it dominates

Stated plainly so this review is not mistaken for an argument against the guidance:

- **§4 (raw output, not printed verdict) is the most valuable rule in the file.**
  It caught, in one day: D95's kernel declaring itself VOID in a log while its
  manifest looked like clean data; a job printing a scientific verdict over zero
  measurements; a `RemoteDisconnected` that had already retried and succeeded,
  sitting above the real `Killed`; D98's residual range that matched no prompt;
  and D96's confound correlation that dissolved under a clustered permutation.
  **Three of those five were my own claims.**
- **§1 (check the premise before building on it)** caught a fully degenerate task
  generator, a gold answer whose token count covaried with the difficulty knob,
  and a `caesar_k` family declared length-constant that measured 28–29 tokens —
  all before any GPU time.
- **§5 (an instrument needs a null)** halted the QK probe before it produced a
  single number, on an error of ~15 that turned out to be the wrong layer
  entirely. Without it, D96 would have been published from a spy reading the
  prelude block.

## 6. The error I made twice, which no rule caused

Worth recording separately, because it is not a rule-reading failure but a habit
one. **I quoted one item's spread as though it were a range, twice in one day.**
First in D98 ("per-block residual 0.09–0.11", which was `echo_digit`'s four
per-block values and matched no prompt's across-prompt range). Then — *after*
identifying and fixing that — again in D99, where "40 independent orbits, median
1.30e-04" turned out to be `sorted(glob)[:40]`, i.e. **32 replicates of
`add1_i08` plus 8 of `add1_i10`: 1.25 items.** The true population of 512 orbits
gives a median 2.6× smaller.

Both survived my own review and were caught by an auditor recomputing from the
raw files. The generalisable guard, now applied: **whenever a range or an n is
quoted, print the number of independent units behind it in the same breath** —
`n_prompts`, not `n_files`. A convenience slice (`[:40]`, `head`, a glob order)
is never a sample.

**The pattern across today:** the rules that mandate *checking* paid for
themselves many times over. The rules that mandate *restraint* (§2, §3, §7) each
cost something when read literally, because each was written against a real past
failure and reads, out of context, as broader than intended.
