# Open question: did Huginn actually learn difficulty-calibrated recurrent compute?

STATUS: **not implemented** — proposed eval-only research direction only, as of
2026-07-22. Zero code written, zero data collected. Nothing below has been
run or verified empirically; it is a reasoned hypothesis plus a proposed
cheap test, written down so the reasoning isn't lost, not a result.

SCOPE: **eval-only, by design and by necessity.** No training, fine-tuning,
or RL is proposed or planned here — explicitly ruled out on cost grounds
(no spare GPU budget/time for training runs). Whatever this line of inquiry
produces has to come from probing a fixed, already-trained Huginn checkpoint,
same as everything else in this project.

OWNER: unassigned.

STALENESS WARNING: this reflects one conversational exchange on 2026-07-22,
reasoning from general recollection of Huginn's published training design
(recurrent-depth transformer, depth-recurrence architecture,
`tomg-group-umd/huginn-0125`), not a fresh re-read of the paper or its repo
in this session. Specific numeric claims below (sampling distribution shape,
mean depth, truncated-backprop window) are stated with hedges deliberately —
**re-verify against the actual paper/model card before citing any of them as
fact**, especially before writing them into anything paper-facing.

---

## The question

Huginn's training samples the number of recurrent unrolls `r` per training
step from a distribution, largely decoupled from how hard that specific
example is — training doesn't explicitly say "this is a hard problem, use
more steps." The worry: if easy examples sometimes get trained with many
steps and hard examples sometimes get trained with few, does the model ever
learn a coherent, difficulty-appropriate "spend more compute when it's
hard" pathway at all — or does it learn something that merely tolerates
variable depth without actually calibrating to difficulty?

This is not a settled question the Huginn paper answers by construction.
The actual design bet (as best recalled here) is weaker than "learns
depth-difficulty calibration": train the recurrent block to behave like a
stable, roughly-monotonic refinement operator — more iterations, no worse
an answer, usually better — regardless of what `r` happened to be during
any given training step. If that generalizes, the model doesn't need to
internally know "the right" depth per input; a *stopping rule applied
externally at test time* can do that job instead. That this project's
target model ships with six native exit criteria (`entropy-diff`,
`latent-diff`, `cosine`, `kl`, `minp-kl`, `argmax-stability` — see
`exit_criteria.md`) is itself a tacit admission by Huginn's own authors that
the model isn't trusted to self-report "I'm done" without external
instrumentation — supporting the concern that depth-difficulty matching
isn't automatically/fully learned end-to-end.

Whether that instrumentation actually recovers a real signal — i.e.
whether the model's internal state, even without training-time difficulty
conditioning, still carries something legible enough for an external
criterion to detect — is exactly what's untested, and exactly what this
project already has most of the pieces to check.

## Why a clean negative result is still a good outcome

Raised explicitly in conversation and worth preserving: this is a
genuinely well-posed, falsifiable eval-only question. A **100% negative
result** — exit-criterion stop-time shows no correlation with actual
correctness across depth/difficulty — is not a failed experiment. It would
be a real, citable finding: evidence that Huginn's depth-randomized
training does *not* yield a self-calibrating stopping signal, that the
model's adaptive-compute story rests entirely on the external heuristics
layered on top rather than something the model itself learned. That's a
legitimate, interesting negative result for a methodology/audit-flavored
paper angle (an angle this project's own `START_HERE.md` already floats as
defensible), not a wasted analysis. No training required either way; the
question is falsifiable purely from inference-time probing.

## Proposed minimal test (not implemented)

1. Wire one native exit criterion — start with `argmax-stability`, simplest
   to reason about — into the (already fixed, not yet GPU-rerun)
   `run_v6_correctness_probe.py` path, so each unroll step records both
   (a) whether the criterion would stop here, and (b) whether the argmax at
   that step matches the correct answer's first token (the probe already
   computes (b); (a) is new).
2. Across the depth/difficulty sweep, check whether the criterion's
   stop-step correlates with the step at which the argmax first becomes
   correct — same `partial_spearman`/`spearman_by_level` convention already
   used elsewhere in this project, to avoid the pseudoreplication mistake
   documented in `claims_ledger.md` D10.
3. Interpretation: correlated → real evidence the model learned *something*
   depth-calibrated despite random-`r` training. Uncorrelated → real
   evidence it didn't, and the exit criteria are just heuristics riding on
   top of an uncalibrated process.

Blocked on: the same GPU rerun already queued for `three_scale.csv`/
`v6_correctness_probe.csv` (see `architecture_state.md`, open items) — this
should ride along with that rerun rather than trigger a separate one.

## Explicitly out of scope (not planned, not this project, not now)

- **RL fine-tuning** to teach depth-appropriate stopping. Plausible
  mechanism in principle (outcome-based reward conditioned on steps used is
  the same idea behind reasoning-model RL on chain-of-thought length,
  applied to latent iterations instead of tokens) — but this is a training
  intervention, ruled out on cost/time grounds, and would be a materially
  different research program (modifying the checkpoint) than what this
  project does (analyzing a fixed one).
- **Deeper mechanistic interpretability** — activation patching per
  recurrent step, linear probes for an internal "steps remaining" counter,
  sparse autoencoders on `core_block` hidden state. Would give a causal
  rather than correlational answer. Not proposed now: bigger scope, no
  concrete plan, listed here only so it isn't reinvented from scratch later
  if the minimal test above comes back interesting enough to justify it.

## Relationship to existing project artifacts

- `docs/exit_criteria.md` — documents all six native criteria; none has
  ever been wired into an actual experiment in this repo before this
  proposal.
- `src/traj_geom/extraction/hook.py`'s `return_logits=True` path — the
  per-step logit access this test needs already exists (fixed this session,
  self-validating, unconfirmed on real hardware — see that module's
  GOTCHAS before trusting its output).
- H3 (contraction bottleneck) — a related but distinct question (is the
  recurrent map a contraction at all) tested so far only on a toy model
  outside this repo (`src/h3_results/`), never on real Huginn. Relevant
  background, not a prerequisite for the test above.
