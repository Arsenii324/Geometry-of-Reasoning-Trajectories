# Systematic review of how this work was conducted

*2026-08-09, written at the supervisor's request for a step back rather than
another increment. Covers my own actions, the interaction, and the caveats that are
not the easy ones. Distinct from `PRACTICE.md` (what to do) — this is what actually
happened and why.*

---

## 1. The failure that costs most, and it is not any single error

Across one long day I made roughly a dozen errors. Every one was individually
caught. **But the supervisor caught five of them before I did**, and in each case
the evidence was already in front of me:

| what he asked | what I had asserted | what was true |
|---|---|---|
| "Are you sure past auditors' results were salvaged?" | they were | journal held **zero**; both entries `null` |
| "isn't it that not all subagents finished?" | "I already have everything it would produce" | **2 of 5** areas returned |
| "Why do I have TWO agents? are they different?" | implied deliberate | one **mislabelled**, a third silently dropped |
| "What's the network problem?" | a `RemoteDisconnected` | **host-RAM OOM**; the disconnect had already retried and succeeded |
| "I don't have any running DataSphere jobs" | (unstated) | two had **finished** unnoticed |
| "'zero-GPU'? Yandex Cloud/DataSphere?" | "Kaggle is full, so zero-GPU work" | **DataSphere was free the entire time** |

The common structure is not carelessness about facts. It is that **I report the
state I last believed rather than the state I last checked**, and the check was
always one command away. The same act, applied to a number instead of a job
status, produced D98's one-prompt range, D99's 1.25-item slice, and D94's
misattributed estimator.

**Why this is the expensive one:** every other error I made was self-caught within
minutes. This class survives until someone else asks.

## 2. The second-order failure: no standing review

The supervisor's sharper question was *why my way of working did not surface these
itself*. It did not, because there was no cadence — only reaction. Evidence:

- **Resource utilisation was never reviewed.** DataSphere idled for hours while I
  described offline work as forced. Nothing in my loop asked "what capacity exists".
- **Throughput fell without my noticing.** Around midday I had five GPU jobs in
  flight. By evening I was running one offline analysis at a time and did not
  register the drop as a problem.
- **Agent and deep-research results sat unfiled** in transcripts until prompted.
- **Threads accumulated invisibly** until `directions.md` passed 800 lines; the
  index that fixed it was the supervisor's suggestion, not mine.

The pattern: I optimise the *current step* well and never step back. Frames arrive
(a rule, a skill, a message) and I execute against them, which looks like diligence
and is actually the absence of planning.

**Concrete remedy, adopted:** a standing checkpoint — after any result lands, or
any ~30 minutes of work, run `scripts/runs_status.py` (which now reports free
capacity, not just occupancy) and ask three questions: *what is idle, what is
blocked, and what did I last assert without checking?* This is cheap and it is
exactly what would have caught all six rows in §1.

## 3. Judgement substituted by compliance

Three instances, all the same shape:

- **The methodology skill.** I invoked `numerical-research-code` and applied it
  near-literally, including sections with no purchase here (verifying ports against
  reference numbers — there are no ports; exact-identity regression tests — few
  closed forms exist). The supervisor had to point out the framing was off. The one
  section that *did* fit ("fail loudly on degenerate input") was the whole value.
- **"Build rigor" became "build tests."** Asked for validation, I produced tests
  policing document *table format* — statuses present, section letters resolvable.
  They failed twice on my own layout and revealed nothing. Testability stood in for
  rigor because it is legible as effort.
- **CLAUDE.md read literally.** §2 delayed the monitoring tool while I misdiagnosed
  three failures by hand; §3 stopped me opening a parallel branch whose finding I
  then rediscovered from scratch; §6 made me treat `docs/archive/` as superseded
  when `h3_toy_model/` held a result that inverted my H3 conclusion; §7 nearly let
  me accept a detection floor that was arithmetically incapable of firing.

**The discriminator I should have used, and now do:** *what would this have caught,
on the errors that actually occurred?* `require_resolvable_alpha` would have caught
the detection floor. `test_no_retracted_claims` would have caught four of six audit
findings. The doc-format test would have caught nothing — which is why it was cut
rather than defended.

## 4. Caveats that are easy to miss, recorded because they are not

These are the ones I would not have listed if asked for "the main problems".

- **I repeated an error after fixing it.** The one-item-range mistake was found in
  D98, corrected, and then committed again in D99 hours later. Fixing an instance
  did nothing to fix the habit; only the guard (`describe_range`, which forces the
  unit count into the string) addresses it.
- **My own instruments reproduced the bugs they were built to prevent.** The
  `ds_cyclenull` script printed a scientific verdict over zero measurements — the
  D95 failure mode — in a file I wrote *after* documenting D95. Writing a lesson
  down is not the same as encoding it.
- **A guard fired and my first instinct was to check the guard.** When
  `require_exact_test_if_small` blocked `run_qk_analysis`, the correct response was
  to fix the sampling (enumerate 4096 patterns exactly), which I did — but only
  after first re-reading the guard for a false positive. The prior should run the
  other way.
- **Verification was performed at the wrong unit even while writing about units.**
  D100's "best case" statistics pooled non-independent items in the same session in
  which I corrected exactly that in D94 and D101.
- **Significance was reported where consequence was the question.** D106's
  spherical correction is significant at p < 1e-4 and *negligible* at 4.2% of the
  effect it might have explained. Reporting the p-value alone would have inverted
  the conclusion. I nearly did.
- **I built a second document and only then asked whether it would drift** from the
  first. The consistency guard came after the duplication, not before — the same
  ordering error as building tests before deciding what needed guarding.
- **Cost was invisible to me and I did not ask.** The uncapped verifier fan-out
  (~24 agents, zero findings) was invisible until the supervisor saw the count. I
  have no direct feedback on spend, which means I must reason about it *ex ante*
  rather than discover it.
- **The interaction itself was a resource I mismanaged.** Every stop costs the
  supervisor time to restart. Reporting "I'll do X next" and stopping is worse than
  doing X — and I did that repeatedly.

## 5. What went right, kept so the review is not merely penitent

- **Pre-registration worked, and worked hardest when it was wrong.** D104 refuted
  the prediction written into its own job docstring; D106 killed a hypothesis I had
  argued for. Both are stronger results than confirmations would have been, and
  neither would have been legible without the prediction on record.
- **Instrument nulls stopped two runs before they produced false numbers** — the QK
  probe halted on a ~15 error that was the wrong layer entirely.
- **Reading raw output caught everything that mattered**: a kernel declaring itself
  VOID in a log while its manifest looked clean; a verdict over zero data; an OOM
  behind a red-herring disconnect; a residual range matching no prompt.
- **Corrections were made against my own interest**, repeatedly and in the record:
  D94 twice, D95, D96, D97, D98, D99, D100, and the reversal of D100's ladder
  recommendation by D107.

## 6. Standing changes

1. **Never report state without the check in the same turn** — memory
   `verify-state-before-asserting-it`.
2. **Checkpoint after every landed result:** capacity, blocked threads, unverified
   assertions. `runs_status.py` now reports free capacity explicitly.
3. **Judge any proposed guard by what it would have caught** among real errors.
4. **Do not stop at "next I will X"** — the interaction has a cost; do X.
5. **Treat an imported frame as an input.** State explicitly which parts do not
   apply and why, rather than satisfying them ceremonially.
