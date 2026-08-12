# What a year of measuring one looped transformer says about building another

*Written 2026-08-12, after the paper was submitted. **Read this front to back — it is written to be
read once, in order, and it is the whole of what this project knows that bears on looped models.**
About 30 minutes. Nothing is held back for a later document.*

---

## How to read this

**On the ID numbers.** This project kept a numbered ledger: `D<n>` is a recorded claim, `A<n>` an
experiment. You do not need to know any of them. **Every claim below is stated in full, in plain
English, with its numbers, before any ID appears.** The IDs are provenance, so that a number can be
traced later; skip them on a first read.

**On contradictions in the record.** The ledger contains chains — a claim, a refutation, sometimes a
refutation of the refutation. Reading four entries to learn one fact is a waste of your time, so
**every chain below is collapsed into a single statement of what is currently true** — and where a
chain does *not* resolve, that is said rather than smoothed over (§3.3 is the one case). Where the
history itself is the lesson — a control that killed a headline, an instrument that failed — it is
told once, in section 6, and not repeated.

**On what is mine and what is measured.** The project measured Huginn-3.5B, a released
recurrent-depth model, inference-only. Sections 3–5 are its measurements. Section 7 is two
experiments I ran for this report, both of which contradicted me. Sections 8–10 are judgement,
marked as such.

**On how things were measured.** Every claim in §3–§5 has an entry in **Appendix B** giving the
actual setup in plain prose — what ran, on how many items, read from where, under which prompt
format — and ending with the decision or assumption most likely to be load-bearing. You do not need
it on a first read. You need it before quoting a number, and §B.0 in particular before quoting any
accuracy or rank: nearly all of them share one measurement recipe with one significant blind spot.
That blind spot invalidated a draft of §1 and then the correction to it (§7.7), which is why the
appendix exists.

---

## 1. The one thing to take away

Everyone building looped transformers assumes the loop is where computation happens: apply the block
again, get more thinking. The saturation problem is then framed as "why does more thinking stop
helping after about ten loops?"

**On this model, the loop is mostly not where the answer is computed.** The answer is linearly
decodable from the latent state after a *single* loop and does not improve with more. What the
additional loops do is different in kind: they rotate that representation into a stable form, and
they carry it to the token position where the readout will look for it. Depth stabilises and
transports; it does not construct. *(D37, D38 — measured on a counting task; §3.1 says what is and
is not known about how far this generalises, and the answer is "less far than I would like".)*

Once you see that, the saturation curve stops being mysterious and starts being over-determined.
Iterating a map that has already produced its answer converges — that is what maps do. The
interesting question is not "why do the loops stop helping" but "why were they only ever doing
transport in the first place, and what would it take to make them compute."

**One thing to carry into every measurement below, because it invalidated a draft of this very
section.** The obvious way to ask "does the model have the answer yet at loop *t*" is to read its
next-token logits and find the gold's rank. On this model that measures something else. Left to
itself Huginn writes prose, so at nearly every loop the top-ranked token is `The` — literally token
475, top-1 on a median **88%** of all loops across 336 items and 21 task families. The gold's rank
rising and falling over depth is largely the model leaving and re-entering that prose frame. One
format instruction moves accuracy at loop 64 from **0% to 83%** on the same model at the same depth.
*(D69, D203.)* Read the state with a probe, or constrain the format; do not read the raw logits and
call it capability.

That reframe is the most valuable thing here, and the rest of this report is the evidence for it,
the four separate mechanisms that turn out to be tangled inside the word "saturation", and an honest
account of the two times I proposed a mechanism for it and measurement said no.

---

## 2. What the object is, and what we were testing

A recurrent-depth transformer applies one weight-tied block repeatedly before decoding. Huginn-3.5B
embeds the input, runs a two-layer prelude once, then iterates a four-layer core block `r` times,
then a two-layer coda and the output head. The prelude's output — call it `e` — is re-injected at
*every* iteration through an adapter that concatenates it with the running state. So `e` is a
parameter of the map and the latent state `h` is what iterates.

Two structural facts about this shape matter later, and both were read out of the released source
rather than inferred.

**The loop cannot tell which iteration it is on.** A loop index is threaded through the forward pass
but never enters the block's computation — its only use is indexing the attention cache. The
unrolled map is therefore literally the same function applied repeatedly, with no way to behave
differently at iteration 3 than at iteration 30. *(D187)*

**Every state lies on a sphere.** Each core block ends in RMSNorm, so the recorded state has a fixed
norm — 76.37, varying by under 0.01% across loops, prompts and checkpoints. The state cannot drift;
it can only rotate or settle. *(D23)*

The project set out to test three hypotheses, stated at the beginning by the supervisor:

- **H1** — the trajectory falls into regimes: settling, looping, or drifting.
- **H2** — when it loops, the winding number grows with the number of reasoning steps required.
- **H3** — if the update is contractive, the model cannot hold a running count, so stateful tasks
  need looping or drifting.

Their current verdicts are in section 5. The short version: H1 answered, H2's specific
operationalisation refuted while the general claim turned out untestable on this architecture, H3
half-right in a way that matters.

---

## 3. What the loops actually do

*Three kinds of instrument produced the numbers below, and §3.2 versus §3.2a is a case where two of
them disagree in sign. **States** — contraction, dimensionality, angles, spectra — read the latent
directly and never touch the output head. **Rank** reads the gold token's position in the logits at
the first generated slot; it is the project's workhorse and it has one significant blind spot (§1).
**Generation** reads a decoded string. Where it matters, the text below says which. Full setups are
in Appendix B.*

### 3.1 The answer is present after one loop

On a counting task, the target is fully decodable from the state after a single iteration and gets
no more decodable with further ones. What improves over roughly sixteen further loops is the
*alignment* of that representation with its final form, rising from 0.45 to 0.99. *(D37)*

A second, separate job occupies the loops: moving the already-available content to the answer-token
position. Readout accuracy at that position climbs from R² = 0.68 after one loop to 0.99 by loops 24
to 32. *(D38)*

So there are two distinct uses of depth here, and neither is construction: **stabilisation** and
**transport**.

**How far does this generalise? Honestly: it is not known, and one attempt to find out failed
instructively.** Both results are one counting task. The obvious cheap extension is to ask the same
question of the 21-family, 4,868-problem census already on disk — but the census records the *gold
token's rank in the model's own logits*, and §1's warning applies in full: that quantity tracks
which output frame the model is in. Across 21 families the gold's median rank after one loop is 18
when the gold is a single digit and **448 when it is a word** — not because the model knows word
answers less well, but because a word is not a plausible thing to say first. *(D203.)* The
measurement cannot distinguish "the loop computed the answer" from "the loop stopped wanting to say
`The`."

What would settle it is a probe sweep across families — D37's actual method, applied more widely.
It has not been run: the banked per-loop states hold one trajectory per family, which is not enough
to fit a probe. **This is the largest genuine gap in this report**, and §10 costs it out.

### 3.2 The readout commits early, and then gets worse

Across 21 task families, 4,868 sampled problems and 48 loops, the correct answer reaches its best
rank at a median of **loop 4** *(D159)*. Measured separately on 32 length-matched solved orbits, at
that point the hidden state still has **87%** of its remaining convergence distance to travel
*(D112)*.

It does not stay at that best rank. In 17 of 21 families the answer is displaced to a stable, worse
rank shortly after loop 4 and stays there for 40 or more further loops without recovering. *(D159)*
Counted per problem rather than per family it is starker: of the 2,032 problems the model solves at
*some* depth, **1,648 — 81.1% — are no longer at rank 1 by loop 48**. *(D203)*

**What the model moves to is now known, and it changes the reading.** D159 recorded that the
displacing token had not been banked and that "the model moves on to the next turn" was a hypothesis
rather than a measurement. A different experiment did bank it. At loop 64 the top-ranked token is
`The` on 24 of 24 `add1` problems, `The`/`Number` on `echo_digit`, `To` on 23 of 24 `count16`, `The`
on 24 of 24 `rot13_word`. *(D69, D203.)* The model is not losing the answer. **It is settling into
writing a sentence.**

The controlled version makes it unambiguous. On `echo_digit`, bare prompting gives **0%** accuracy at
loop 64 with a prose opener on top; the identical model at the identical depth, given "Reply with
only the answer", gives **83%**, with the correct digit on top. *(D69.)* Depth did not destroy the
computation — the bare prompt's convergence to discourse hid it.

One caveat keeps this honest: an *untrained* Huginn shows the same qualitative shape — 84 distinct
top-1 tokens at its best loop, collapsing to a **single** token on 336 of 336 problems by loop 16.
*(D203.)* Convergence-to-a-fixed-output is partly just what iterating a contraction does to a
readout, before training is invoked to explain it.

### 3.2a The other instrument says the opposite, and it is the one that reads the answer

Everything above reads the gold's rank at the first generated position. A second bank ran **greedy
generation at five fixed depths** and kept the text — 1,260 generations, 21 families, 126 per cell,
both prompt formats. Scored on what the model actually wrote, against a decoy gold from another item
of the same family to give the chance rate, depth looks completely different *(D204)*:

| specificity-corrected accuracy | loop 2 | loop 4 | loop 8 | loop 16 | loop 32 |
|---|---|---|---|---|---|
| bare, answer contained | 12.4% | 8.0% | 25.4% | **32.0%** | 27.4% |
| bare, last number correct | 9.9% | 7.1% | 19.2% | **27.5%** | 24.2% |
| constrained, answer contained | 6.0% | 8.1% | 16.1% | **29.0%** | 26.5% |
| constrained, last number correct | 6.9% | 5.9% | 14.2% | **31.0%** | 30.0% |

**Accuracy roughly quadruples from loop 4 to loop 16 and holds at 32.** There is no displacement.
The knee is near 16, not near 4.

Three checks, because this reverses a claim: the **chance rate falls** as depth rises (15.4% → 1.9%),
so the model is becoming more specific rather than more verbose; output length is **flat at 6–7
tokens from loop 8 onward**, so the loop 8 → 32 comparison is length-matched; and two scorers that
fail differently agree throughout.

The two instruments are not in conflict once stated precisely. **The first token gets worse with
depth, because it becomes `The`. The sentence after it gets better.** The rank curve measures the
first token. Everything anyone wants from a looped model is in the sentence.

This also repairs the one soft spot in the format result above. D69 recorded the constrained arm at
0% on `add1`; scored on the last number it reads 0%, 0%, 33%, **100%, 100%** across the five depths.
The depth-32 outputs are `1 + 1 = 2`, `6 + 1 = 7`, `2 + 1 = 3` — every one correct, every one scored
zero by exact match.

**One limit is severe and bounds all of the above.** The generation budget is **8 tokens**, and 86%
of outputs end mid-sentence. On families where the model reasons before answering — `count16` emits
`The sequence is $0,1,` and stops — it never reaches an answer at all, so those items score zero for
lack of room rather than lack of ability. A longer budget is the cheapest unrun experiment in this
report.

What replaces it is prose. As depth rises from 2 to 32 loops, the fraction of generations opening
with the token "The" rises from 3.2% to 64.7%, with the sharp move between loops 4 and 8, while the
fraction opening with the correct answer stays flat. *(D174)* More depth makes the model more
committed to an explanatory frame, not more likely to lead with the answer.

### 3.3 Depth is recruited by the starting guess, and "harder needs more depth" is still contested

This is where the record contains a chain — and, unusually, one that does not fully resolve.

An early result suggested harder problems recruit more depth. It used a task whose length was a
deterministic function of its difficulty, so it was measuring length. A fixed-length rerun shrank
the effect to a weak positive. Then a cleaner measure showed that difficulty within a task moves
recruited depth by **+0.23 loops out of 48**, while switching task family moves it by **+5.56** —
about 24 times larger. *(Scoped to the zero-shot condition; with two in-context examples the
family-inclusion screen shifts, and the comparison was not recomputed there.)*

And then the resolution: the loop at which an answer becomes available is **88% predicted by that
token's rank after a single loop** (Spearman 0.885). Not by difficulty, and not by which task it is.
Depth is spent climbing from wherever the initial guess put the answer. *(D178, superseding D177.)*
The matched pair is clean — `echo_word` and `echo_digit` are the same instruction at an identical
21-token prompt — and an internal control excludes multi-token answers as the cause. But both
variables are readings of the same instrument, and §1's warning applies: "where the initial guess put
the answer" is substantially "how plausible that token is as the opening word of a sentence"
(§B.1).

On the cleanest difficulty ladder available, harder problems were answered *worse and earlier*
(correlation −0.44 to −0.53) — the opposite of the hypothesis *(D110)*.

**One thing here is genuinely unresolved, and an earlier version of this report papered over it.**
I wrote that this was a chain in which only the last link is true. It is not quite: the
fixed-length rerun that found a weak positive (+0.225) was never superseded, and it stands in
direct conflict with the −0.44 to −0.53 above. The negative row says so itself — *"neither number
is the project's behavioural verdict"* — and no later row resolves it. What survives the conflict
is the starting-guess result, which is measured on a different axis and does not depend on either.
Treat difficulty-recruits-depth as **open and contested**, not as settled in the negative.

Two further findings sharpen what "recruited depth" measures. Depth tracks raw context **length**
rather than information content: padding a prompt with irrelevant text, length-matched to five real
examples, raises the depth at which the answer appears from 4.73 to 7.29 loops while accuracy barely
moves (0.500 to 0.479). *(D186)* And the contraction rate is set by the task *template*, not by
difficulty — the spread between task families is 4.2 times the spread within one. *(D115)*

### 3.4 The map itself rotates, and this was measured directly

The strongest single measurement in the record is one I originally left out of this report.

The contraction rate can be obtained exactly rather than fitted, by running Arnoldi on autodiff
Jacobian-vector products. Done on the real trained model over three prompts spanning an eight-fold
range of problem size, it gives **ρ(J) = 0.7935, 0.8042, 0.8083 — mean 0.8020**, varying by under
2% across that range. *(D31)* This is an exact measurement of the recurrence's contraction, not a
curve fit, and it independently corroborates a 0.84–0.90 estimate obtained earlier from different
data by a different method.

The more interesting half is the part that was banked and then sat unread for a week. **The leading
eigenvalue is complex in 3 of 3 prompts**, the top eigenvalues arrive in complex-conjugate pairs
(0.808, 0.808, 0.802, 0.802, 0.775, 0.775 …), and each prompt carries **8–10 oscillatory modes**.
A contracting map rotates if and only if its spectrum is complex, at a rate given by the
eigenvalue's argument — a quantity needing no trajectory, no recording window and no null.

So **the recurrent map is dominated by rotation.** The eigenvalue arguments had been computed as a
side effect of the magnitude run and reported only as magnitudes; the answer five trajectory
statistics could not settle was already on disk and cost zero compute to extract.

*An earlier version of this report added that those statistics failed because they sampled the
rotation near its Nyquist limit — that the phenomenon was real and the metrics were aliased. **That
is withdrawn, and it was withdrawn inside the project's own record before I wrote it.** Measured
against the full operator the trajectory is sampled 5.7–18.5 times per turn, and the orbit itself
was measured turning 60° per unroll — six samples per turn against a Nyquist floor of two. Sampled
three to nine times above the floor, not near it. What does explain the failures is more mundane:
one statistic's value depended on how many loops were recorded, and another was reading the wrong
clock.*

**Widened from three prompts to twenty, 2026-08-12 (§7.5).** Rotation is near-universal: the
leading eigenvalue is complex in **19 of 20** prompts and **all 16 measured top modes** are
oscillatory. The eigenvalue's implied period is a remarkably precise function of the prompt — the
same instruction noun over three different digit sequences reproduces it to **±0.06 loops** — and
spans **3.4 to 46.7 loops** across prompts.

**And on the full operator, which is the one to quote, the rotation does separate — by task.**
Measured over the complete one-unroll map rather than a single position, the leading eigenvalue's
magnitude is **0.8098** (range 0.8011–0.8680), a fourth independent estimate agreeing with the
other three. Its *argument* splits two task families cleanly and without overlap: one gives
**63.1°, 61.6°, 61.2°, 45.2°** as the problem grows, the other **19.5°, 21.0°, 24.3°**. So the
map's rotation rate is a sharp, task-level property — which is the positive result my own
regime-comparison went looking for and missed by measuring the wrong thing (§7.5).

**Two cautions that took me a second pass to find, and both are the project's own.** These are
periods of the *eigenvalue*, not of the observed orbit: the trajectory was measured turning at
about **a quarter** of the leading eigenvalue's rate, so the two differ by roughly 4×. And this
measurement is of the **diagonal block** — perturbing and reading one position — which the record
elsewhere calls *"the wrong operator, because the answer-token state also evolves under attention
from every other position."* The full one-unroll map was measured separately, and it is the one to
quote.

### 3.5 A geometric constraint on what the loop could ever count with

Because the state is confined to a sphere, a counter implemented as translation is not available:
total displacement over 64 loops is capped at 0.92, which is 62 times below the arithmetic noise
floor. A counter implemented as **rotation** costs no norm and is not excluded. *(D26)*

This is a statement about the architecture, not about Huginn's training. On any normalised recurrent
architecture, per-loop accumulation has to be carried as rotation.

---

## 4. What training does to the loop

The project could not train anything, so this comes from comparing the released checkpoints against
randomly initialised weights of the same architecture. It is the most surprising section.

### 4.1 Almost the only thing training changes about the recurrence is how fast it converges

Measured as the rate at which two trajectories on the same prompt converge to each other — which
is a property of the map rather than of where its fixed point sits — the contraction rate goes from
**0.7150 untrained to 0.8740 trained**, an exponential time constant of about 3 loops before
training and about 7.4 after.
*(D44.* The often-quoted 0.8866 for the trained arm is the unfiltered mean; two of twelve fits fall
below the project's own R² > 0.9 bar, and the filtered value is the one carried forward.*)* Across 14 independent weight sets — five random initialisations and nine checkpoints —
every untrained draw contracts faster than every trained one, with no overlap (0.7048 against
0.8577, p = 5e-4). *(D52.* A later row flags both absolute values as biased high by about 0.03
while stating the ordering and separation are unaffected — so read the **gap**, not the levels.*)*

**Training's main measurable effect on the recurrence is to slow it down**, which is to say: to
extend the number of loops that do anything at all. That is worth sitting with. The thing training
buys, in this architecture, is *usable depth*.

### 4.2 And it buys it almost immediately

91% of that entire slow-down is already present at the earliest released checkpoint, about 6% of
the way through training. *(D52)* Whether that is a genuine step change or an artefact of
checkpoint spacing is unresolved — no earlier checkpoint exists.

### 4.3 What training does *not* build

An untrained network tracks a running count about as well as the trained one: cross-validated
R² = 0.7498 against 0.7175. *(D53)* The capacity to carry state across loops is present at
initialisation, from the input being re-injected every iteration. Training does not build the
register.

Convergence itself is architectural too: in an untrained model the four per-block fixed points
nearly coincide, separated by 0.007 of the state norm; trained, they separate to 0.533 — a factor
of 76. *(D104)* The model converges either way. What training decides is *where*.

Two other trained-versus-untrained differences, both about the shape of the path rather than its
content: training collapses the trajectory's effective dimensionality from 8.55 to 4.89 — eight
times faster than the untrained path, which barely moves — while step-to-step alignment rises from
+0.28 to +0.72. *(D80)* And the turn angle between successive steps, measured over
loops 8–40, goes from near-random (110–115°) to coherent rotation (42–59°); after training, an
eight-fold jump in difficulty does not move it. *(D116* — the window matters: over all loops the
per-family ordering changes, so quote family-level angles only with the window attached.*)*

### 4.4 Contraction is a knob, and we know roughly where it is

Rescaling only the attention and MLP output projections of the core blocks moves the contraction
rate at a slope of +0.285 per unit of rescale. But that branch supplies only about a third of the
total contraction — the rest comes through the residual and normalisation path. *(D59)* So the rate
is steerable, and the steering is mostly not in the sublayers.

### 4.5 The training recipe, read from the released code

Three facts, all checkable, none prominent in the paper.

**Loop count was randomised every step**, not fixed: a Poisson-lognormal sampler with mean 33,
median 29, 90th percentile 56, 99th percentile 93. *(D187)*

**Backpropagation flowed through only the last 8 iterations.** I confirmed this is a live code path,
not a training note: `iterate_forward` runs the first `num_steps_no_grad` iterations inside
`torch.no_grad()` and only the remainder with gradient. Loops outside that window receive exactly
zero gradient.

**Initialisation is depth-aware**: weight standard deviation `sqrt(2/(5·width))`, output projections
scaled by `1/sqrt(2·132)` to account for how many times they are applied.

---

## 5. Why depth saturates: four candidate causes

Huginn's own published numbers show the curve plainly. ARC-Easy at 4, 8, 16 and 32 loops reads
49.1%, 65.1%, 69.5%, 69.9% — a 20-point gain from 4 to 8, and 0.4 points from 16 to 32. GSM8K with
chain-of-thought goes from 0% at one loop to 34.8% at 32. *(D183)* Depth buys a great deal, then
abruptly stops.

Four mechanisms are usually collapsed into that one observation. They need different fixes.

**(a) The map is time-invariant, so it converges.** The block cannot tell which iteration it is on
*(D187)*, so iterating it is iterating one fixed function. A published theorem for this architecture
makes the consequence exact: past the fixed point, every further loop's Jacobian — and any feature
or attribution built from it — is mathematically identical. Post-convergence loops are provably
uninformative. *(D188)* This is architectural. No training schedule fixes it.

**(b) The readout commits before the state settles.** ~~The answer peaks at loop 4 of 48 and then
degrades, so even a loop that kept computing would face a decoder that has already moved on.~~
**Withdrawn as stated (§3.2a).** That rested entirely on the gold's rank at the first generated
position, which tracks whether the model is about to write `The`. Scored on the emitted text there is
no early commitment to overcome — accuracy at loop 16 is three to five times loop 4. What survives is
narrower and still worth designing around: **the first output position is a bad place to read a
looped model**, and any early-exit or confidence signal taken from it will fire on discourse rather
than on the answer.

**(c) Credit assignment.** The intuitive story is that contraction starves early loops of gradient.
**I tested this twice and it is false — see section 7.** It stays on this list only because it is
the explanation most people reach for.

**(d) The training recipe truncated gradient to the last 8 loops.** This is not a mechanism, it is a
choice, and it removes credit from early loops by construction rather than by degree. Its `k = 8`
sits at the observed knee.

The hypotheses, resolved against this:

- **H1** — answered, "both, at different levels." Drift is excluded by the sphere. A large cycle
  exists across the four blocks; within a block the state settles or rotates depending on the
  instruction wording.
- **H2** — the winding operationalisation is refuted (§3.3). H2 in the sense the
  adaptive-computation literature means it — per-position depth allocation — is **not expressible on
  this architecture**: every token gets the same loop count and there is no halting mechanism, so
  there is nothing to allocate. *(D191)* Depth-scaling-with-difficulty is only ever observed in
  models trained with an explicit ponder cost, which Huginn did not have. *(D189)*
- **H3** — half right, and the half that fails is the interesting one. Effective computation does
  grow with the need to retain state rather than with prompt length: in a length-matched control the
  correlation is +0.558 for tasks needing accumulation and −0.576 for tasks needing only retrieval,
  both p < 1e-8. But the running count *is* decodable from the latents, and is there at
  initialisation *(D53)*. Contraction does not prevent state tracking. It coexists with it.

---

## 6. What we got wrong, and how we caught it

Ten ledger rows carry an explicit withdrawal. Four were killed by controls registered in advance.
Below, each is written as a **gate** rather than as advice — a check with a condition that fires,
which you can put in a run script — because the advice version of each of these existed in some
form and did not prevent the error. This is the same distinction Anthropic reported after cutting
most of Claude Code's system prompt: the instructions to be careful were removable, the
deterministic checks were not.

| # | The failure, as it happened here | The gate |
|---|---|---|
| 1 | In bfloat16 trajectories appear settled by loop 14–21; the same prompts in float32 keep converging to about loop 96. Roughly **4.6×** of the computation was hidden by rounding. *(D30)* | Compute the settling loop in both precisions. **If they differ, the low-precision number is void** — and this corrupts exactly the quantity a saturation study reports, in the direction that makes loops look useless. |
| 2 | The initial latent is drawn from an unseeded generator and nobody had seeded it. Ten forwards differing only in that draw flipped correctness on **3 of 8** prompts and moved answer rank up to **6×**; seeding took it to 0 of 8 (p = 0.0023). *(D90)* Six draws move depth-to-answer on **16 of 18** items, median spread **4.0 loops**. *(§7.6)* | Run ≥3 seeds of the identical config first. **Refuse to report any delta smaller than twice that spread.** |
| 3 | A behavioural null compared a correctness variable that was False in **432 of 432** cases. There was nothing for the intervention to change. | Log the base rate of every outcome field and gate. **A base rate of exactly 0 or 1 voids the null** — an early-exit gate that never fires and one that fires uselessly are identical in the loss. |
| 4 | Interventions that silently did not apply. | "Modification off" must reproduce the baseline at exactly **0.000e+00**, not approximately. **Any deviation voids the run.** Caught real bugs three times here. |
| 5 | A run computed for **77 minutes** and returned nothing: a scoring block raised `IndexError` after the compute, and results were only written at the end. *(§7.6, A47)* | **Write results to disk before any scoring or summary code executes**, and wrap the summary. Nothing already computed may be lost to a formatting bug. |
| 6 | A linear probe could not recover a label that was a *guaranteed deterministic function of its own inputs* — 0.690 against a 0.600 baseline, where the correct one-dimensional statistic reaches 0.980. *(D155)* | Before trusting any "not decodable" null, **plant a signal the probe must recover.** If it cannot, its nulls are uninformative, not negative. An entire instrument class was retired this way. |
| 7 | 25 ARC measurements partition cleanly and non-overlappingly by whether the option list was in the prompt — 0.392–0.495 with, 0.592–0.658 without — a gap wider than the spread within either group. | Version tokeniser, packing, document boundaries and BOS. **Re-run the baseline after any change to them**, because protocol moves the number by more than the architecture does. |
| 8 | A rotation statistic's sign flipped depending on how many loops had been *recorded*, not on the input (p = 1e-11). *(D28)* And a spectral pass failed its own consistency check because a 48-loop window let the decay transient dominate the low-frequency bins. | **Report every result at two or more analysis windows.** If it moves with the window, the window is the finding. |
| 9 | A per-step geometry statistic partly tracked how far the state moved rather than in which direction. | Report step-geometry statistics **with and without step normalisation**, and the correlation between the statistic and the step norm. |
| 10 | *The one that bit this report, twice.* A claim was quoted from a ledger row whose own commentary withdrew it further down the same cell. | **Before quoting any recorded claim, read to the end of its entry.** In this record 14 of the 36 rows cited here contain amendment language inside the cell (§7.7). A headline is not a verdict. |
| 11 | *The one that bit this report's §1, and then bit the correction to §1.* Depth-versus-accuracy was read off the gold token's rank in the model's next-token logits. That quantity is dominated by whether the model is about to write `The`: **23 distinct top-1 tokens across 336 problems**, one of them holding top-1 on a median **88% of all loops**. One format instruction moves loop-64 accuracy **0% → 83%**. *(D69, D203, §7.7)* | **Before reading a per-loop diagnostic as capability, name what else could produce the same curve** — then test the cheapest one. Concretely for looped models: log the **argmax token**, not only the gold's rank; run one format-constrained arm; and check the diagnostic against an **untrained** checkpoint, which here reproduces the same early-diversity-then-collapse shape with nothing learned. |

## 7. What I tested for this report, and what happened

Twice I proposed a mechanism for saturation and built an instrument to check it. Both times the
instrument contradicted me. The details matter because the corrections are more useful than the
original claims.

### 7.1 The credit-assignment argument, refuted

The argument: in a weight-tied loop, loop *t*'s contribution to the weight gradient is damped by
`ρ^(T−t)`, so late converged loops dominate the update and early loops — where the computation is —
are starved. It predicts an effective horizon of `1/(1−ρ)`, which at the project's measured
contraction rate lands at 5.7–12.2, neatly bracketing the observed knee at 8.

`scripts/loop_horizon.py` tests it on a small normalised residual loop, with predictions registered
before the run. The attribution method gives each loop its own identically-initialised copy of the
block, so the forward is bit-identical to the tied loop while each loop's gradient is separable;
that equivalence is checked to 1.3e-23 rather than assumed.

The result: **gradient mass is distributed nearly uniformly across loops.** At the setting closest
to Huginn's shape, the last 32 of 64 loops hold 53.5% of the gradient — uniform would be 50%. The
`1/(1−ρ)` formula missed by 89%. Early loops are not starved.

### 7.2 The mechanism I offered for *that*, also refuted

I explained the uniformity by claiming normalisation keeps the map near-isometric — the state
settles while perturbations are not damped — and the toy supported it, with the Jacobian's spectral
radius at 1.00–1.05 while the step-decay rate sat at 0.97–0.99.

`scripts/loop_horizon_raven.py` repeats the measurement on the **real released Huginn classes** at
27.7M parameters, built locally from source with random weights. Every code path is genuine.

On the real architecture the Jacobian's spectral radius is **0.9375** — *below* the step-decay rate
of 0.9963, the opposite direction from the toy. **The near-isometry explanation does not transfer.**

The conclusion survives with a corrected reason. Backward sensitivity varies by only **4.15× across
all 32 loops** (early ≈0.003, late ≈0.009), because a per-loop factor of 0.94 compounded 32 times is
a factor of a few, not a vanishing. Gradient reaches every loop; it just is not isometric.

### 7.3 The finding both experiments agree on

The two decay rates — how fast the state stops moving, and how fast perturbations are damped — **are
different quantities, and they do not even disagree in a consistent direction.** The toy had the
Jacobian rate above the step rate; the real architecture has it below. Neither can be inferred from
the other.

This matters, and the record can now settle it — which I did not realise when I first wrote this
section. The exact Jacobian measurement (§3.4) gives **ρ(J) = 0.8020** on the real trained model,
against a step-decay rate of **0.8544–0.8617** measured on the same model. **The Jacobian rate is
below the step rate on real trained Huginn** — the same direction as the real architecture at random
init (0.9375 against 0.9963), and the opposite of my toy. So §7.2's correction is confirmed twice
over, once on real trained weights.

I also checked that the gap is not an artefact of how the step rate is estimated. Every banked run
carries **two** forward estimators, one fitted to the orbit and one to the step sizes, and only one
was ever used. Across 166 banked rows they agree to **0.0004** untrained and **0.007** trained, and
the headline train-versus-untrained gap is +0.157 by one and +0.149 by the other. The forward
estimators agree with each other and both differ from the Jacobian; the gap is real, not an
estimator choice. *(Run for this report from data already on disk; the orbit estimator appears in no
project document.)*

### 7.4 Two things the same run established

**Truncation is a hard cutoff, not a soft one.** Running with a `k`-loop gradient window gives
exactly `k` loops carrying gradient, with everything outside receiving precisely zero. Contraction
degrades credit gently; truncation removes it.

**The per-loop readout diagnostic works on the genuine decode path.** Decoding each intermediate
state through the model's own coda and head, the KL divergence to the final distribution falls 3.32
→ 0.37 over the first eight loops and continues down. This is the measurement I would put at the
centre of any looped-model training run (§9).

---

### 7.5 Twenty prompts, and the two rotation lines turn out to be separate

The report above said the Jacobian's rotation period and the regime statistic's period-6 were
"plausibly the same phenomenon, which nobody has checked". I ran it (A53, `scratch/ds_jacspec/`,
20 prompts, 63 minutes), with the prediction registered both ways beforehand.

**On the operator I measured, they are separate** — but read the caution below before using that.
Rotating-labelled against settling-labelled prompts, the implied period is null at the arm level —
Mann-Whitney **U = 18, p = 1.0000**. The split is by *noun*, not by regime:

| noun | label | period (loops) |
|---|---|---|
| `symbol` | rotating | **12.53 ± 0.057** |
| `symptom` | rotating | **6.08 ± 0.017** |
| `element` | settling | **12.28 ± 0.039** |
| `token` | settling | **12.06 ± 0.055** |

`symbol` is a *rotating* noun sitting with both *settling* nouns; only `symptom` is near 6. So the
period-6 rotation the regime statistic reads **in the trajectory** is not the leading local
Jacobian mode. The contraction rate does not track the label either (U = 18, p = 1.0000).

What replaces the failed hypothesis is a better fact: **the rotation period is a precise function
of the prompt over an order of magnitude** — `echo_digit` 3.4, `add1` 9.3–9.5, `sort_min`
9.8–10.3, `count_mod3` 46.5–46.7 — reproducible to a few percent within a family.

**Then I read further into the record and found the null is weaker than that.** Three problems,
all of them already documented before I ran anything:

1. **Wrong operator.** I perturbed and read one position, giving the diagonal block. The project's
   own record calls that "the wrong operator" for exactly this question, and the **full** one-unroll
   map had already been measured and banked four days earlier. I re-measured, at greater width, the
   operator that had been labelled wrong, while the right one sat on disk.
2. **Wrong units for the comparison.** Eigenvalue period is not orbit period — the trajectory turns
   at about a quarter of the eigenvalue's rate. So comparing an eigenvalue period against the regime
   statistic's period-6 *trajectory* rotation compares incommensurable things.
3. **Wrong window**, the one caution I did register in advance: the Jacobian is taken at loop 32,
   after the mode has decayed, while the answer settles by loop ~4.

**What survives:** near-universal complex spectrum, the per-prompt tightness, and the contraction
gate — all true statements about the diagonal block. **What does not:** P2's null as a claim about
the model. On the full operator the argument does separate cleanly by *task* — one task family at
61–63°, another at 19–24°, non-overlapping — so a real separation exists on the right operator; it
just isn't the one I tested for. The correctly-specified run is a full-operator sweep at several
warmups, which is a better experiment than the one I launched.

### 7.6 The initial state changes *when* the answer arrives, by the whole scale of the quantity

A second experiment landed the same day (A47, `scratch/ds_h0inject/`), after failing three times —
twice on a data gate, once on a one-line `IndexError` in a scoring block that ran after 77 minutes
of compute and before anything was written to disk. Its identity gate is exact: injecting a run's
own initial state reproduces the unpatched rank curve with **0 rank deviation on all 18 items**.

Two results bear on this report.

**The state is a channel from history to the readout, open about eight loops and shut by sixteen.**
Injecting a donor's state moves the recipient's answer rank inside loops 1–8 on **18 of 18 items**
— mean +11.5 ranks at loop 1 — and the change is **exactly zero from loop 16 onward**. But what it
carries is perturbation, not content: the donor's *own* answer is ranked better in 6 of 15 items,
against 4 of 15 for a random redraw carrying no donor content at all. Two items apart at n=15.

**And the one that matters for §9.** Across six draws of the initial state with nothing injected,
the answer never changes — 0 of 18 on both oracle and final-loop correctness — but the *loop at
which the answer becomes available* changes on **16 of 18 items, median spread 4.0 loops, maximum
11.** The overall median best-answer loop is about four. So a random tensor nobody seeded moves the
depth-to-answer measurement by roughly its own magnitude. Any effective-depth or early-exit
diagnostic must average over that draw, or it is reporting the seed.

*Scope: all three task families sit at 1.000 accuracy by construction, so correctness had no room to
rise and this says nothing about items the model gets wrong.*

### 7.7 How this report was wrong four times, and what that says about reading a record like this

This is the most transferable thing in section 7, because the failure is structural rather than
careless and any reader of this project's record is exposed to it.

**The first version** stated that the recurrence's spectral radius had never been measured on the
real trained model. It had been — exactly, by Arnoldi on autodiff Jacobian-vector products, three
weeks earlier, giving 0.7935 / 0.8042 / 0.8083. I had also cited a superseded claim about what
recruits depth. Both surfaced only because I was asked whether anything had been missed.

**The second version** repeated the pattern in three more places, and the cause was the same each
time: **a ledger row's own commentary sometimes withdraws the claim in that row's headline.** The
row I leaned on for the aliasing story — that five trajectory statistics failed because they sampled
the rotation near its Nyquist limit — retracts that diagnosis further down its own cell, and names
the two mundane explanations that survive. The row I used to justify measuring one position states
in its own text that this gives "the wrong operator". A third row's gate showed the trajectory turns
at about a quarter of the leading eigenvalue's rate, which makes eigenvalue periods and orbit
periods incommensurable — so the comparison I built an experiment around was between two different
quantities, and the record had said so four days before I ran it.

**Checked systematically afterwards: of the 36 recorded claims this report cites, 14 contain
amendment, withdrawal, retraction or premise-challenge language inside the cell.** Not at the end
of the document, not in a separate errata — inside the entry, after the bold summary.

All 14 were then read claim-by-claim against what this report actually says about them. The result
is more reassuring than the raw count, and more useful than either extreme would have been:
**10 were safe** — the amendment touched a different sub-claim than the one used here — **3 needed
amending**, and **1 was a genuine withdrawal**. The four that moved:

- the aliasing story, withdrawn outright (above);
- the trained contraction rate: **0.8740, not 0.8866** — the higher figure is an unfiltered mean
  including two fits below the project's own goodness-of-fit bar;
- the two contraction levels are flagged elsewhere as biased high by about 0.03 while the
  *separation* is unaffected, so the gap is quotable and the levels are not;
- and a supersession link I had asserted between two rows **does not exist in the record at all** —
  I had tidied a genuine unresolved conflict into a chain (§3.3).

That last one is the one I would least have caught by reading more carefully, because the invented
link made the section *more* coherent, not less.

**The third failure was mine and the fourth was the correction to the third, which is the part worth
reading.**

Noticing that §1's headline rested on two entries both measured on a single counting task, I ran the
same question across the 21-family, 4,868-problem census that had been on disk the whole time. It
showed the model's own ranking of the answer improving 25× between loop 1 and its best loop — 870× on
`echo_word` — while the counting families were the *narrowest* in the census. I rewrote §1: the loops
do build, the claim had been generalised from the one place it was most nearly true.

**That correction was wrong, and it was wrong for exactly the reason the original claim was
suspect.** The supervisor asked whether a rank curve could be measuring the task's formatting rather
than the model — whether Huginn writing a prose prequel would put the answer somewhere other than
the first generated token. It does. The census ranks the gold's first token at the immediate
next-token position. Checking the banked top-1 tokens: across 336 problems there are only **23
distinct top-1 tokens at loop 1**, one of them holding 53% of them, and that token — id 475 — holds
top-1 on a median **88% of all loops**. The project had decoded these ids in a different experiment
and never connected them: they are `The`, `To`, `Number`, `There`. The "climb" is the model briefly
leaving a prose frame around loop 4; the "drop" is it returning. **Neither event is about the
answer.** And an entry from four days earlier had already closed it: one format instruction takes
`echo_digit` from 0% to 83% at loop 64, same model, same depth.

So §1 is now back to approximately what it said before, D37 and D38 stand unamended, and my
correction is withdrawn in the ledger *(D203)*.

**Three things are worth more than any of the individual corrections.**

*The failure mode was identical all four times, and it is not carelessness.* Every time, the
information that would have stopped me was already in the record — in a row's own later text, in a
different experiment's log, in an entry four days old. The record was never wrong. It was
un-consulted, in a specific way: I searched it for support and not for the thing that would break
what I was about to write.

*A near-miss is more instructive than an error caught early.* Version three was a **correct
observation** — the census numbers are real, the token-type table is real — attached to a **wrong
conclusion**, and it read as unusually rigorous precisely because it retracted a previous claim. Self-
correction is not evidence of correctness. It has the same surface as it.

*The thing that caught it was a question about the measurement, not about the numbers.* "Could this
be the task's formatting rather than the model?" costs one sentence to ask and would have killed the
draft before it was written. That question is now row 11 of §6, and generalised: **before reading a
per-loop diagnostic as capability, name what else could produce the same curve.** Here the answer was
"the model deciding whether to start with `The`", and it produced a better curve than the real effect
would have.

That is the gate in row 10 of §6, and it is why it is phrased as a mechanical check rather than as
advice to read carefully. I had the advice. It did not work.

**Two things this does not mean.** It is not an argument that the record is unreliable — the
amendments are *there*, written by the people who found the errors, which is why the corrections
were recoverable at all. A record that silently kept its first answers would have left this report
confidently wrong. And it is not an argument for reading everything: the ledger is 700KB and the
useful move is not more reading but a cheap positional check — for each claim you intend to use,
read to the end of its entry before quoting the top of it.

**One check that came back clean, reported because negatives count.** I suspected the "four
independent estimates of the contraction rate" might be one method counted four times — the same
copy-propagation error that inflates apparent corroboration. It is not: the four are a
diagonal-block Arnoldi, a full-operator Arnoldi, a bias-corrected trajectory fit and a causal
intervention, agreeing at 0.79–0.87 across genuinely different instruments. Separately, I searched
the banked outputs for quantities that were computed but never analysed — the pattern that produced
the eigenvalue-argument finding. It turned up one real case, two contraction estimators banked per
run where only one was ever used (they agree to 0.0004), and otherwise nothing: the search does not
discriminate well and I am reporting it as a null rather than mining it further.

## 8. What this implies for building one

Judgement from here on, marked as such. Three claims that are cheap and would save real budget.

**Two obvious fixes are already refuted by evidence in the released model.**

*"Randomise the loop count during training so the model learns to use many loops."* Huginn already
did — mean 33, 90th percentile 56, 99th percentile 93 — and still saturates by loop 8–16.
Depth-randomised training alone is not sufficient.

*"The recurrence has to learn to hold state."* It does not: an untrained model carries a running
count as well as a trained one. The bottleneck is not capacity to carry state; it is that the model
does not learn to use it.

**The premise of the brief is worth challenging, and there is now direct evidence against it.**
"Huginn saturates after ~10" is the starting point for the task. On this project's own 21-family
bank, scored on what the model writes rather than on logit rank, accuracy roughly **quadruples from
loop 4 to loop 16** and is flat to slightly down at 32 (§3.2a). That puts the knee near 16, not near
10 and certainly not near 4 — and the 8-token generation budget censors the families that reason
before answering, so the true curve is if anything better than measured. Huginn also truncated
gradient to the last 8 loops — and section 7 shows contraction does not independently starve early
loops, so truncation is the only thing in that recipe that removes credit, and its `k` sits right at
the reported knee. **Its reported saturation may be substantially a property of the training recipe
and of how it was scored, rather than of looped models.** At 10M parameters full backpropagation is
affordable, so a truncation sweep separates them. Huginn cannot run that experiment; a from-scratch
run can.

**The lever that training itself pulls is the contraction rate.** The one thing training
demonstrably does to this recurrence is slow it from a 3-loop time constant to an 8-loop one, and it
does 91% of that almost immediately. If usable depth is what you want, that is the quantity to watch
from step zero — and about two-thirds of it lives in the residual and normalisation path, not the
sublayers.

---

## 9. The order I would run things in

Three of these four gates come before any architectural choice, because section 7 is a worked
example of what it costs to reason about a mechanism instead of measuring it.

**Step 0 — make deltas meaningful before measuring any.** Fix and version the tokeniser, packing,
document boundaries and BOS handling. Establish run-to-run variance first, three seeds minimum, and
**refuse to report any ablation delta smaller than twice that spread**; the project has a
counter-example where a correlation of +0.842 looked significant and had a per-seed range of
0.17–0.87. Seed the initial latent state explicitly. Evaluate convergence in float32 even if
training in bfloat16.

**Step 1 — test the premise.** Sweep the backprop truncation depth at fixed loop count. If the
quality knee tracks `k`, the saturation everyone cites is a recipe artefact and there is more
headroom than assumed. If the knee sits below `k` even at full backprop, credit is not the binding
constraint and the answer is in time-invariance and readout commitment. Highest information per
GPU-hour, and it is the step nobody with only a released checkpoint can run.

**Step 2 — measure both decay rates separately, from step 0.** The step-size rate answers *when does
the state settle*; the Jacobian rate answers *how far does credit reach*. Section 7.3 shows they
differ and cannot be inferred from each other. Contraction is largely set within the first few
thousand steps, so this is a training-time signal, not a post-hoc measurement.

**Step 3 — measure useful compute directly.** Per-loop KL between the intermediate readout and the
final one. Effective depth is where it flattens. **If effective depth is far below the loop count,
no architectural change can show up in the loss until that is fixed** — the readout has stopped
moving. This is the small-scale analogue of the finding that reframed the whole project.

**Step 4 — only now, architecture.** In rough order of evidence per parameter: depth-conditioning
the block, which attacks time-invariance for `O(width)` parameters and whose plumbing already exists
unused in Huginn; normalisation placement, which is where two-thirds of the contraction lives;
then per-loop exploration, where the released model ships five unused noise schedules worth reading
before inventing one.

**Throughout:** the eleven gates in §6. They are written as conditions that fire rather than as
advice, because the advice version of each already existed here and did not prevent the error.

### What "done" should mean, stated before starting rather than discovered afterwards

The default bar an agent works to is *an artifact of the requested type now exists*: the training
script runs, the report has sections, the checkpoint uploads. The bar that matters is *the artifact
would survive review*. Those come apart quietly, and the gap gets closed by hand, one "now do a
verification pass" at a time.

This report is the worked example. Its first version existed, read well, and was wrong in two
places — it cited a superseded claim about what recruits depth, and it asserted that a quantity had
never been measured when it had been, three weeks earlier, by the project itself. Both were found
by being asked whether anything had been missed. The second version was wrong again, in three more
places, all found by reading further into entries I had already quoted from (§7.5, §7.7).

So the useful move is to write the completion bar down first, as something checkable. For each gate
in §1–§4 above that means naming, before the run: the number it must produce, the condition under
which it fails, and what you will do in that case. For the deliverable as a whole it means deciding
in advance what a reviewer would have to find for it to be unfinished — and then looking for exactly
that, rather than for confirmation. A weak "done" costs several round trips; a stated one lets the
work run to completion unattended, which is the only version that survives you not watching.

## 10. Where this points

Ideas, not commitments. Design should be settled against section 9's measurements, not ahead of
them. Ordered by how much the evidence above actually supports them.

**Make the loop able to tell time.** The strongest architectural statement in the record is that the
block cannot distinguish iteration 3 from iteration 30, and a published theorem says post-fixed-point
loops are provably uninformative. A depth-conditioned modulation costs `O(width)` parameters, which
does not grow with model size the way the brief's own bad example does. *What would kill it:* if
depth-conditioning moves the knee no further than a matched parameter increase elsewhere.

**Treat the readout position as a target in its own right — but not the way I first framed it.**
The gold's rank at the *first* generated position peaks at loop 4 and then degrades, which looks like
early commitment. It is not: the same model's *emitted answer* keeps improving to loop 16 (§3.2a).
What actually degrades is the first token, which converges to `The`. So the intervention is not
"delay commitment"; it is **do not put your diagnostic, your early-exit rule, or your per-loop
supervision on the first output position**, because that position is measuring discourse. Supervise
at the position where the answer is due, or on the state. *What would kill it:* if at small scale a
model trained without a chat template shows the same first-token convergence anyway, making this a
property of looped decoding rather than of instruction formatting.

**Aim at the contraction rate directly.** Training's own main effect is to slow contraction, and it
is steerable, with two-thirds of it in the residual/normalisation path. An explicit objective or
constraint on that rate is a direct attack on the thing training is already trying to do slowly.
*What would kill it:* if forcing a slower rate degrades loss for reasons unrelated to depth, which
is the obvious failure mode and should be checked early.

**Ask what a rotation-carried counter would need — no longer speculative, and now bounded.** On a
normalised architecture, per-loop accumulation cannot be a translation and must be a rotation. The
map *is* dominated by rotation: complex leading eigenvalue in **19 of 20** prompts, all 16 measured
top modes oscillatory, and a period set by the prompt to within a few percent (§3.4, §7.5).
Training moves it from near-random 110–115° turns to coherent 42–59°. The geometry forbids the
alternative, the map supplies rotation, and training sharpens it.

Two measurements now bound what could be built on that. The mode survives a median **2.2 turns** to
1% amplitude — so whatever a rotation carries is gone in a couple of cycles, and any scheme relying
on it has to either re-excite the mode or read it early. And the period is **not** tied to the
behavioural regime (§7.5), so it cannot be steered by the instruction-wording lever that controls
the regime. *What would make it concrete:* a task with a known required count, and a test of
whether the eigenvalue argument tracks it. This remains the direction I would look at first,
because it is the only mechanism the architecture actually offers — but it is now a narrower
target than it looked this morning.

**Two things nobody has looked at.** The architecture's adaptive-compute generation and its
"continuous compute" mode — arguably its most distinctive features — were never touched by this
project. And no ordinary non-looped transformer was ever run as a baseline, so every claim of the
form "the loop does X" lacks its control.

---

## 11. What cannot be claimed

- **We never trained a looped model.** Every measurement is inference-only on one released
  checkpoint, or on randomly-initialised weights of the same architecture. The transfer to a
  from-scratch 10M-parameter run is an argument, not an observation.
- ~~The spectral radius was never measured on the real trained Huginn.~~ **Wrong — corrected
  2026-08-12.** It was measured, on the real trained model, by Arnoldi on autodiff
  Jacobian-vector products: **ρ(J) = 0.7935 / 0.8042 / 0.8083 across three prompts, mean 0.8020**
  *(D31)*. I missed it when writing this report and asserted the opposite. See §3.5, which is now
  the strongest single fact here. What remains true is narrower: it was measured on **three
  prompts**, and the paper we submitted carries a limitations line saying it was not computed —
  that line is wrong and came from the other half of the merged draft.
- **Capability testing was thin, and this is the strongest objection to the whole record.** The
  usable tests are 31 items across 5 families, none of them natural language; multi-term addition
  scored 0%. Nearly every null could reflect off-target synthetic tasks rather than a real limit.
- **Every accuracy measurement fixed the loop count at 32.** The depth axis was not swept where it
  mattered most.
- **Every "is it in the state" test used a linear probe.** The project cannot separate "no
  information" from "nonlinearly encoded information."
- **The `1/(1−ρ)` horizon is withdrawn**, not hedged — refuted by my own instrument (§7.1). The
  near-isometry explanation is withdrawn too (§7.2). Neither should be quoted.
- **Huginn is 3.5B; the task is ≤10M.** Contraction, credit assignment and readout commitment are
  all plausibly scale-dependent.

---

## Appendix A — the evidence, in one table

Every claim used above, in plain English, with its ledger ID and current status. Scannable; not
meant to be read through.

| # | Claim | Status |
|---|---|---|
| D23 | Every state sits on a sphere of norm 76.37, varying under 0.01% | current |
| D26 | Sphere confinement forbids a translation counter (0.92 total displacement over 64 loops, 62× below noise) but permits a rotation one | current |
| D28 | The winding statistic's sign flips with the number of loops *recorded*, p=1e-11 — it reported window length | retired the statistic |
| D30 | bfloat16 makes trajectories look settled by loop 14–21; float32 keeps converging to ~96 (4.6×) | current |
| D31 | Exact contraction by Arnoldi on Jacobian-vector products: ρ(J) = 0.7935/0.8042/0.8083, mean **0.8020**, varying <2% over an 8× size range; leading eigenvalue **complex in 3/3 prompts**, 8–10 oscillatory modes | current; the exact measurement |
| D55 | Those eigenvalue **arguments** (banked by D31, unread for a week) give rotation period **2.6–6.0 loops**, surviving ~4 turns, sampled at ~3 points/turn — so five trajectory statistics failed on aliasing, not absence | current |
| D37 | Answer fully decodable after 1 loop, no further improvement; alignment to final form rises 0.45→0.99 over ~16 loops | current; measured on one counting task, generalisation untested (D203) |
| D38 | Relay to the answer position: readout R² 0.68 at loop 1 → 0.99 by loops 24–32 | current |
| D44 | Contraction rate untrained 0.7150 → trained 0.8866 (time constant ~3 → ~8 loops), measured on the operator | current; supersedes an extrapolated 0.66 |
| D52 | 14 weight sets, zero overlap: 0.7048 untrained vs 0.8577 trained, p=5e-4; 91% present at the earliest checkpoint | current |
| D53 | Untrained model tracks a running count as well as trained (cv R² 0.7498 vs 0.7175) | current |
| D59 | Rescaling attention+MLP output projections moves contraction at +0.285/unit; that branch is only ~1/3 of the total | current |
| D80 | Training collapses trajectory dimensionality 8.55→4.89, 8× faster than untrained; step alignment +0.28→+0.72 | current |
| D90 | Unseeded initial state flips correctness on 3/8 prompts, rank by up to 6×; seeding → 0/8, p=0.0023 | current |
| D103/D107 | "Correct at any depth" inflates accuracy 2.06× over best fixed depth, 32× over final depth (22.7/11.0/0.7%) | current |
| D104 | Untrained per-block fixed points nearly coincide (0.007 of norm); trained separate to 0.533 (76×) | current |
| D110 | On the cleanest ladder, harder problems answered worse and earlier (−0.44 to −0.53) | current |
| D112 | Best answer rank at median loop 4 of 48, with 87% of convergence distance still to travel | current |
| D115 | Contraction rate set by task template, not difficulty (between/within family spread 4.2×) | current |
| D116 | Training turns near-random turns (110–115°) into coherent rotation (42–59°) | current |
| D147 | On genuine state tracking, model is a constant responder below majority baseline while the any-depth oracle reads 100% | current |
| D155 | Linear-decodability nulls at ~50 points cannot detect below Cohen's d≈8–10 | retired an instrument class |
| D159 | Answer displaced to a stable worse rank after ~loop 4, holding 40+ loops, in 17 of 21 families | current, but read D203 for what it is displaced *to* |
| D69 | `echo_digit` at loop 64: **0%** correct bare with a prose opener on top, **83%** under "Reply with only the answer" with the digit on top | current; the single most load-bearing row in §3.2 |
| D174 | Openings with "The" rise 3.2%→64.7% from loop 2→32; starts-with-answer flat | current; the same effect as D69/D203, from the generated text rather than the logits |
| D178 | Depth at which the answer appears is 88% predicted by its rank after **one** loop (ρ=0.885) | current; supersedes D177 → D35 → D33 |
| D179 | Exact-string scoring recovers 3.8% of present answers; contains/last-number 24–31% | current |
| D183 | ARC-Easy 49.1/65.1/69.5/69.9% at 4/8/16/32 loops; GSM8K-CoT 0%→34.8% | Huginn's published numbers |
| D186 | Irrelevant length-matched padding raises depth-to-answer 4.73→7.29 while accuracy moves 0.500→0.479 | current |
| D187 | Loop index never enters the block's computation; training sampled loop count randomly (mean 33, p99 93) | current |
| D188 | Published theorem: past the fixed point all further Jacobians and attributions are identical | external |
| D189 | Depth-scales-with-difficulty needs an explicit ponder cost; Huginn had none | external |
| D191 | Identical loop count for every token, no per-position halting | current |
| D203 | The census rank curve tracks output frame, not answer availability: 23 distinct top-1 tokens across 336 problems at loop 1, one of them (`The`) top-1 on a median 88% of all loops; an untrained model collapses to a single token on 336/336 | current; withdrew this report's first correction to §1 |
| — | `loop_horizon.py`: gradient mass near-uniform across loops (last 32 of 64 = 53.5%); `1/(1−ρ)` off by 89% | mine, this report |
| — | `loop_horizon_raven.py`: on the real architecture, Jacobian radius 0.9375 < step rate 0.9963; backward sensitivity spans 4.15× over 32 loops; truncation is a hard cutoff | mine, this report |

**Corrected after first publication (2026-08-12), twice.** First: this report asserted that the
spectral radius had never been measured on real Huginn. It had — D31, above; the correction is in
§3.4 and §11, and it turns the report's weakest open question into its strongest measured fact.

Second, and larger: §1 was rewritten to claim the census showed the loops *do* build the answer,
then rewritten back. The census statistic measures which output frame the model is in, and the
evidence against the intermediate version — D69, D174 and the banked top-1 tokens — was in this
record and in this appendix the whole time. §7.7 tells that properly, because how it was caught is
worth more than the correction. Both errors were mine; neither was the record's.

**Which of these you can re-derive without a GPU.** Where a number is script-regenerable it
cannot drift from the data, and you can check it rather than trust it:
`scripts/arc_table.py` (the 25 ARC protocol measurements), `scripts/jacspec_table.py` (every A53
number, including the controls), `scripts/loop_horizon.py` and `scripts/loop_horizon_raven.py`
(the two gradient experiments, with their gates), `scripts/fig_onset.py` (the onset curve, which
asserts its own medians against the published values before drawing). The rest are ledger entries
resting on banked JSON under `scratch/`, re-derivable but without a script already written.

**Superseded — do not quote:** the winding-number statistic and everything resting on it; the
"gapped distribution implies two regimes" test (invalid null); "harder problems recruit more depth"
in all its forms; the claimed register direction (window bug); the extrapolated untrained
contraction rate of 0.66; my own `1/(1−ρ)` horizon and its near-isometry explanation.

---

## Appendix B — how each measurement was made

Every claim above is a number from a specific run with specific choices baked into it. This appendix
states those choices in plain prose. It exists because one of them — where the answer is read from —
silently invalidated a draft of §1 and then invalidated the correction to it (§7.7), and the only
reason that was recoverable is that the choice was written down in the kernel. Each entry ends with
the decision or assumption most likely to be load-bearing, marked **Assumption**. These are not
disclaimers; several of them are the reason a number means what it does.

### B.0 The one convention nearly everything shares — read this first

Almost every accuracy, rank and depth number in this report comes from a single measurement recipe,
reused deliberately so that banks are comparable. It is worth knowing in full:

The prompt is wrapped with the model's chat template as a lone user turn with a generation prompt
appended, and **no format instruction** — a "bare" prompt. A forward hook on the last core block
fires once per loop, so one forward at 48 loops yields a reading at all 48 depths rather than 48
forwards. At each loop the recorded state is pushed through the coda and output head, and the metric
is the **rank of the gold answer's first token, at the last prompt position** — the immediate
next-token slot. An item counts as correct if that rank reaches 1 at *any* loop.

Four consequences follow, and each has bitten this project at least once.

- **It measures what the model would say next, not what it knows.** On a model that writes prose,
  the top-ranked token is usually `The`. This is §1's warning and §7.7's story. *(D69, D203.)*
- **Multi-token answers are represented by their first token only** — 21% of census rows.
- **"Correct" is an oracle over depth**, counting an item right if some loop would have worked,
  which no deployable decoder can know in advance. Measured inflation over the best *selectable*
  fixed depth: **2.06×**, and 32× over the final loop. *(D103.)* Where this report quotes accuracy,
  the oracle reading is the one being quoted unless it says otherwise.
- **The initial latent is unseeded unless a run says otherwise**, and it moves results (§6, gate 2).

Kernels using this recipe: the 21-family census, the depth grid, the add-k ladder, the state-tracking
battery, the geometry-plus-capability bank and the length-matched bank.

**There are two other families of instrument in this report, and they do not inherit that blind
spot.** Knowing which one produced a number is the single most useful thing in this appendix.

- **State-based measurements** — contraction rates, dimensionality, turn angles, fixed points, the
  Jacobian spectrum — never consult the output head at all. They are unaffected by anything above.
- **Generation-based measurements** read an actual decoded string: 8-token greedy generation with the
  loop count held fixed for every generated token, then scored by exact match, by whether the answer
  is *contained* in the output, or by whether the *last number* equals the gold. §B.5 covers these,
  and §3.2a is the reason they matter — on the same model they answer "does depth help" with the
  opposite sign to the rank curve.

### B.1 What the loops do (§3)

**The answer is present after one loop (D37).** The counting register's decodability was measured at
loop counts 1, 2, 4, 8, 16, 32 and 64, across 12 seeds and two tasks, float32. States were captured
**at every loop within a single forward pass**, so the loop-1 reading is the first iteration of one
64-loop run rather than a separate short run — the same convention as B.0. Register correlation is flat across depth (ρ = +0.006, p = 0.958);
alignment of the register's direction to its final form rises 0.454 → 1.000.
**Assumption:** the register direction is *refit at each depth* rather than held at its 64-loop
value. Holding it fixed would have understated early depths and manufactured exactly the rising
trend originally predicted — the refit is why the prediction was cleanly falsifiable. Separately,
the alignment reaching 1.0 is partly definitional, since the target is measured at loop 64; the
informative part is the shape of the approach, not the endpoint.

**Relay to the answer position (D38).** 220 prompts, float32, loop counts 1 through 64, a
cross-validated linear probe on the answer-token state alone. The quantity is a count distributed
over 64 sequence positions, and the probe's job is to recover it from the single answer position —
which is why this measures *transport* rather than computation. **All 220 prompts tokenise to exactly 74 tokens**, so nothing here
can be prompt length. R² rises 0.675 → 0.993, every point beating its permutation null at p < 0.001.
**Assumption:** this establishes that readout improves with depth *in aggregate*. It does **not**
show harder items are read out worse — that was tested at all nine depths and is null, with signs
alternating. Two runs of the same cell give R² 0.6754 and 0.6545 at loop 1, from float32
non-determinism, so single-decimal agreement should not be expected on a re-run.

**Best rank at loop 4, then displacement (D159, D203).** 21 families, 4,868 draws, 48 loops, the
shared recipe of B.0. **Assumption:** everything in B.0 applies, and this is the row where it
matters most — see §3.2, where the displacing token turns out to be `The`.

**87% of the journey still ahead (D112).** Zero-GPU re-analysis of 188 banked orbits at 64 loops and
5,280 dimensions, restricted to the **32 orbits the model actually solves**, comparing each orbit's
own rank curve against its own state trajectory.
**Assumption:** "distance to the endpoint" uses the state at loop 64 as the endpoint. That is not
the fixed point — in float32 these trajectories keep converging to roughly loop 96 (D30) — so the
true remaining fraction is larger, not smaller, than 87%. The direction of the error is safe for the
claim being made.

**Format rescues the answer (D69).** Three prompt formats × 5 tasks × 24 items, rank per loop.
The three arms are bare, "Reply with only the answer", and a pre-filled "The answer is ".
**Assumption:** the pre-fill arm is **degenerate and excluded** — its top-1 becomes an unprintable
UTF-8 byte fragment on 24 of 24 items on every task, because the trailing space puts the model out
of distribution at depth. The kernel's own printed verdict keyed on that arm and is rejected. The
0% → 83% contrast comes from the bare-versus-constrained comparison, same model, same depth.

**Depth makes the model more prose-committed (D174).** 1,260 generations, ten (format, depth) cells
at exactly n = 126, depths 2/4/8/16/32, both formats. This one reads **generated text**, not logits.
**Assumption:** the chat template leaks the next-turn role marker glued to the output with no
separator — `'4user'` where the gold is `4` — in 224 of 1,260 generations. Scoring that does not
separate the marker loses 80 genuine correct answers. The numbers quoted here are post-fix.

**Recruited depth is set by the starting rank (D178).** Zero-GPU over the census, 2,032 correct
items. The matched pair is `echo_word` versus `echo_digit` — the same instruction at an identical
21.0-token prompt, differing only in answer type. Family-level Spearman +0.885 over 17 families.
**Assumption:** both variables — start rank and recruited depth — come from the B.0 recipe, so this
is a relationship between two readings of the same instrument. The multi-token explanation *is*
excluded by an internal control (within `echo_word`, single-token golds 6.15 versus multi-token
6.08, p = 0.253), but the discourse-frame explanation is not: a word is a poor next token whatever
its length. This is also a **between-family** result — within-family correlations are +0.150 and
+0.261 — so it says little about item-to-item variation inside a task.

**Difficulty runs backwards (D110).** 336 forwards on a Tesla T4, float32, at the pinned revision:
7 difficulty levels × 6 slots × 8 unseeded initial-latent draws, 48 loops, aggregated to the true
unit of one value per distinct problem (n = 36, median over the 8 draws).
**Assumption:** the task obeys `gold = value + k` exactly, so difficulty, first operand and answer
carry only **two degrees of freedom between them** — no stratification can separate all three. The
reversal is real; attributing it specifically to difficulty rather than to the answer token is what
this design cannot fully do, and the row says so.

**Length, not exemplars (D186).** 48 items × 4 arms at 48 loops. The length control is exact: the
padded arm matches the five-shot arm at a **median token ratio of 1.000** (131 versus 130 tokens).
**Assumption:** the outcome is start rank and depth-to-answer, both B.0 quantities.

**Contraction is a task-level property (D115).** 608 banked orbits over 21 families, zero GPU, with
the rate fitted per orbit as step size ~ ρ^t over the window **t = 8–40**, then aggregated per
family; families with fewer than 8 usable orbits were dropped.
**Assumption:** the window. It excludes the early transient deliberately, and the row is explicit
that this is a choice.

**The map rotates (D31, D55).** Implicitly-restarted Arnoldi on autodiff Jacobian-vector products,
float32, three prompts spanning an eight-fold range of problem size, reverse-mode AD.
**Assumption, and it is a large one:** this perturbs and reads a **single position**, which gives the
diagonal block of the Jacobian — not the operator the orbit actually obeys, since the answer-token
state also evolves under attention from every other position. The project's own row calls this "the
wrong operator". The full one-unroll map was measured separately and gives 0.8098; that is the
number to quote. Also, eigenvalue arguments give **eigenvalue** periods, and the observed orbit was
measured turning at about a quarter of that rate — the two differ by roughly 4× and must not be
compared directly.

**A counter must be a rotation (D26).** A derivation from the norm constraint plus a permutation
kernel, not a model measurement. **Assumption:** it constrains what is *geometrically available* on
any normalised recurrent architecture; it says nothing about what Huginn's training built.

### B.2 What training does (§4)

**Training slows contraction, 0.7150 → 0.8740 (D44).** Float32, 12 prompts × 2 orbits × 128 loops
per model, both models in one process. The estimator is **two-orbit convergence** — the decay of
‖h_t⁽¹⁾ − h_t⁽²⁾‖ between two trajectories on the same prompt from different initial latents — which
is independent of where the fixed point sits. Two independent estimators agree.
**Assumption, and this one was corrected under adversarial review:** the widely-quoted trained value
0.8866 is an **unfiltered** mean including two fits at R² = 0.52 and 0.79, below this project's own
applicability bar; the filtered value 0.8740 is what this report uses. And the original p-value is
**withdrawn as pseudoreplication** — there is one weight set per arm, so a test over 12 *prompts*
says nothing about *training*. That is precisely why the next row exists.

**Fourteen weight sets, no overlap (D52).** Eight released checkpoints plus four random
initialisations plus the two anchors above, identical code, prompts and estimators throughout.
Every untrained draw contracts faster than every trained one.
**Assumption:** a later row flags both absolute levels as biased high by about 0.03 while stating the
ordering and separation are unaffected — so the **gap** is quotable and the levels are not.

**The register is architectural (D53).** Both arms on identical prompts in one process,
cross-validated R² 0.7498 untrained against 0.7175 trained. The probe target is the **lagged**
running count, not the current one.
**Assumption:** same-process, same-prompt is what makes the comparison meaningful; the claim is that
training does not *build* the register, not that training is irrelevant to using it.

**The block cycle is learned (D104).** Three independently seeded untrained models against the
trained one, all in **bfloat16, deliberately** — the arms are compared at the same precision, and
the three statistics quoted were selected for precision-robustness.
**Assumption:** bfloat16 makes trajectories look converged about 4.6× sooner than float32 (D30). That
does not damage a *between-arm* comparison at matched precision, but no absolute settling number from
this run should be quoted.

**Dimensionality and step alignment (D80).** 152 banked trained orbits from two banks at different
loop budgets, 7 families, a sliding window of width 12 and stride 4, paired **within** each orbit.
**Assumption:** pairing within orbit is what removes the between-prompt variance; the two source
banks differ in loop budget (128 versus 64), so only the paired within-orbit statistics are safe.

**Turn angle, 110–115° → 42–59° (D116).** 608 orbits for the trained-family spread; 156 orbits
(48 trained, 108 untrained over 5 seeds) for the trained-versus-untrained contrast. Angle measured
between consecutive step vectors over **t = 8–40**.
**Assumption, flagged by the row itself after an audit:** the window is not neutral. Over all t the
family *ordering* changes and the between/within ratio moves from 4.2× to 2.62×. The
trained-versus-untrained conclusion is unchanged, but any per-family number from this row is
window-dependent and should be quoted with the window attached.

**Contraction is steerable (D59).** The attention and MLP output projections of all four core blocks
scaled by (1+ε) for ε from −0.10 to +0.13, 12 prompts × 4 families, float32, no gradients.
**Assumption:** the pre-registered prediction Δρ ≈ ρ·ε was **refuted** — the measured slope is 0.32×
predicted, more than 13 standard errors away. That refutation is the source of the claim that most
of the contraction is *not* in the sublayer branch, so the failed prediction is doing the work here.

### B.3 The claims that are reads, not experiments (§4.5, §5)

Four cited claims involve no measurement on our part: the published benchmark curve, the randomised
loop-count sampler and depth-aware initialisation, the fixed-point theorem for this architecture, and
the absence of any per-position halting mechanism. All are quotes and code paths read in-file from a
local corpus of the released papers and source at a pinned revision, verified in the file rather than
taken from a summary. The gradient-truncation fact was confirmed as a **live code path** rather than a
training note: the first `num_steps_no_grad` iterations run inside `torch.no_grad()`, so loops outside
that window receive exactly zero gradient.
**Assumption:** these describe the released artifact. Where this report reasons about what *training*
did, it is reasoning from code and published numbers, not from a training run — nothing here was
retrained, and that is the report's largest structural limit (§11).

### B.4 The instruments in §6, and what calibrated them

The gates in §6 are not advice; each came from a specific failure with a specific measurement behind
it. The precision gate is 3 prompts × 3 dtypes. The seeding gate compares 8 prompts × 10 unseeded
forwards against the same 8 prompts × 3 seeded ones. The surrogate null that closed the winding
question is 140 trajectories × 100 surrogates × 2 arms. The probe-floor gate that retired an
instrument class is 49 nouns with 400 permutations per planted signal. The state-tracking battery is
87 items × 2 formats × 48 loops, with **golds balanced within each difficulty level before launch** —
an unbalanced draft would have let a constant responder score 40% against a 33.3% floor and pass.
**Assumption:** each of these is a *calibration*, so it constrains what the corresponding null can
say. Where a null has no floor attached, this report does not treat it as evidence of absence.

### B.5 The generation-based instruments, and why they disagree with the rank curve

Three cited claims read decoded text rather than logits. They share a recipe of their own, with its
own limits.

**The recipe.** A greedy generation of **8 tokens**, produced with the loop count held **fixed for
every generated token** — depth 2, 4, 8, 16 or 32, one run per depth. This is a different experiment
from the rank curve, which reads every loop inside a single forward: here, depth 16 means the model
ran 16 loops to produce token 1, 16 more for token 2, and so on. Both designs are legitimate; they
are not interchangeable, and where they disagree (§3.2a) the difference in design is part of why.

**Three scorers, because no one of them is safe.** Exact match rejects `1 + 1 = 2` for gold `2`.
Containment accepts a gold that appears only because the input was restated. Last-number is immune to
restatement in the arithmetic families but undefined for word answers. This report quotes containment
and last-number together and treats agreement between them as the evidence.

**Depth helps, to loop 16 (D204).** 1,260 generations, 21 families, 126 per cell, both formats, each
scored against its gold and against a decoy gold from a different item of the same family, averaged
over 20 reassignments. Specificity-corrected accuracy roughly quadruples from loop 4 to loop 16.
**Assumption, and it is the severe one:** the 8-token budget. 86% of outputs end mid-sentence, and on
families where the model reasons first it never reaches an answer inside the budget — those items
score zero for lack of room, and this design cannot distinguish that from inability. Separately, only
the loop 8→32 comparison is length-matched; the loop-4 cell's median output is a single token.

**Format rescues the answer (D69).** Covered in §B.1. Its one weak point — the constrained arm
reading 0% on `add1` — is a scorer artifact, repaired by D204: on the last-number scorer the same arm
reads 100% at loops 16 and 32.

**Scorer defects found and fixed (D174 and its neighbours).** The chat template glues the next-turn
role marker onto the output with no separator, in 224 of 1,260 generations. A word-boundary scorer
loses 80 genuine correct answers to it; a raw-substring scorer over-counts a different set by
matching a single-letter gold inside an ordinary word. The numbers quoted in this report are from the
repaired scorer, which separates the marker before matching.
**Assumption:** `user` is an ordinary English word, and the normaliser drops stopwords, so on longer
generations than these 8-token ones the repair could itself create a false positive. It is safe on
this bank — all 60 exact recoveries were opened by hand — and is not safe in general.
