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
**every chain below is collapsed into a single statement of what is currently true.** Where the
history itself is the lesson — a control that killed a headline, an instrument that failed — it is
told once, in section 6, and not repeated.

**On what is mine and what is measured.** The project measured Huginn-3.5B, a released
recurrent-depth model, inference-only. Sections 3–5 are its measurements. Section 7 is two
experiments I ran for this report, both of which contradicted me. Sections 8–10 are judgement,
marked as such.

---

## 1. The one thing to take away

Everyone building looped transformers assumes the loop is where computation happens: apply the block
again, get more thinking. The saturation problem is then framed as "why does more thinking stop
helping after about ten loops?"

**On this model, the loop is mostly not where the answer is computed.** The answer is linearly
decodable from the latent state after a *single* loop and does not improve with more. What the
additional loops do is different in kind: they rotate that representation into a stable form, and
they carry it to the token position where the readout will look for it. Depth stabilises and
transports; it does not construct.

Once you see that, the saturation curve stops being mysterious and starts being over-determined.
Iterating a map that has already produced its answer converges — that is what maps do. The
interesting question is not "why do the loops stop helping" but "why were they only ever doing
transport in the first place, and what would it take to make them compute."

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

### 3.1 The answer is present after one loop

On a counting task, the target is fully decodable from the state after a single iteration and gets
no more decodable with further ones. What improves over roughly sixteen further loops is the
*alignment* of that representation with its final form, rising from 0.45 to 0.99. *(D37)*

A second, separate job occupies the loops: moving the already-available content to the answer-token
position. Readout accuracy at that position climbs from R² = 0.68 after one loop to 0.99 by loops 24
to 32. *(D38)*

So there are two distinct uses of depth here, and neither is construction: **stabilisation** and
**transport**.

### 3.2 The readout commits early, and then gets worse

Across 21 task families, 4,868 sampled problems and 48 loops, the correct answer reaches its best
rank at a median of **loop 4**. At that point the hidden state still has 87% of its remaining
convergence distance to travel. *(D112)*

It does not stay at that best rank. In 17 of 21 families the answer is displaced to a stable, worse
rank shortly after loop 4 and stays there for 40 or more further loops without recovering. *(D159)*
Convergence does not merely stop adding information — it removes some.

What replaces it is prose. As depth rises from 2 to 32 loops, the fraction of generations opening
with the token "The" rises from 3.2% to 64.7%, with the sharp move between loops 4 and 8, while the
fraction opening with the correct answer stays flat. *(D174)* More depth makes the model more
committed to an explanatory frame, not more likely to lead with the answer.

### 3.3 Depth is recruited by the starting guess, not by difficulty

This is where the record contains a chain, and only the end of it is true.

An early result suggested harder problems recruit more depth. It used a task whose length was a
deterministic function of its difficulty, so it was measuring length. A fixed-length rerun shrank
the effect to a weak positive. Then a cleaner measure showed that difficulty within a task moves
recruited depth by **+0.23 loops out of 48**, while switching task family moves it by **+5.56** —
about 24 times larger.

And then the resolution: the loop at which an answer becomes available is **88% predicted by that
token's rank after a single loop** (Spearman 0.885). Not by difficulty, and not by which task it is.
Depth is spent climbing from wherever the initial guess put the answer. *(D178, superseding D177,
which superseded D35, which superseded D33.)*

On the cleanest difficulty ladder available, harder problems were answered *worse and earlier*
(correlation −0.44 to −0.53) — the opposite of the hypothesis. *(D110)*

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

Measured directly on the operator, the contraction rate goes from **0.7150 untrained to 0.8866
trained** — an exponential time constant of about 3 loops before training and about 8 after.
*(D44)* Across 14 independent weight sets — five random initialisations and nine checkpoints —
every untrained draw contracts faster than every trained one, with no overlap (0.7048 against
0.8577, p = 5e-4). *(D52)*

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
+0.28 to +0.72. *(D80)* And the turn angle between successive steps goes from near-random
(110–115°) to coherent rotation (42–59°); after training, an eight-fold jump in difficulty does not
move it. *(D116)*

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

**(b) The readout commits before the state settles.** The answer peaks at loop 4 of 48 and then
degrades (§3.2). Even a loop that kept computing would have to overcome a decoder that has already
moved on.

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
This section is the part I would most want carried into any new project, because every item is a
way a looped-model experiment can produce a confident wrong number.

**A metric that reported its own window length.** The project's main rotation statistic flipped sign
depending on how many loops had been *recorded* — not on the input — with p = 1e-11 under a
calibrated null. *(D28)* Any statistic read off a fixed window can be reporting the window.

**Low precision faked convergence.** In bfloat16, trajectories appear to settle by loop 14–21. The
same prompts in float32 keep converging to about loop 96. Roughly 4.6× of the computation was hidden
by rounding. *(D30)* This is the single most dangerous item on the list for a saturation study,
because it corrupts exactly the quantity being measured, in the direction that makes loops look
useless.

**The initial state was random and unseeded.** Huginn draws its initial latent from an unseeded
generator, and nobody had seeded it. Ten forward passes differing only in that draw flipped
correctness entirely on 3 of 8 prompts and moved the answer's rank by up to 6×. Seeding it took the
variation to 0 of 8 (p = 0.0023). *(D90)* Any "loops help" curve must average over, or control, that
draw.

**An outcome variable with no variance read as a clean null.** One behavioural null compared a
correctness variable that was false in 432 of 432 cases. There was no variance for the intervention
to change. *(Re-run on an outcome that varied, the original conclusion came back — but the evidence
had been vacuous.)*

**A scoring rule that measured the scorer.** Exact-string matching recovered 3.8% of answers that
were actually present in the generated text; containment and last-number rules recovered 24–31% at
lower false-positive cost. *(D179)* Separately, first-token scoring read 12.5% where full-text
reading found the answer in 78.1%.

**An oracle metric that read 100% on a model that was not answering.** Scoring "correct if the gold
reaches rank 1 at *any* of 48 depths" inflates accuracy 2.06× over the best fixed depth and 32× over
the final-depth output — 22.7% against 11.0% against 0.7%. *(D103/D107)* On a genuine state-tracking
task, that oracle metric read 100% at every difficulty level while the model was a constant
responder scoring below the majority-class baseline. *(D147)*

**A null from an instrument with no power.** "Not linearly decodable" conclusions drawn from about
50 points in thousands of dimensions cannot detect anything below Cohen's d ≈ 8–10, where d = 0.8 is
already called large. Confirmed both by planting a signal and by failing to recover a label that was
guaranteed present. *(D155)* An entire instrument class was retired.

**A one-token window misalignment.** A claimed "register direction" shared across inputs turned out
to be 97% the current-token embedding contrast — a window bug, not a counter. The real register is
nearly orthogonal to it.

---

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

### 7.4 Two things the same run established

**Truncation is a hard cutoff, not a soft one.** Running with a `k`-loop gradient window gives
exactly `k` loops carrying gradient, with everything outside receiving precisely zero. Contraction
degrades credit gently; truncation removes it.

**The per-loop readout diagnostic works on the genuine decode path.** Decoding each intermediate
state through the model's own coda and head, the KL divergence to the final distribution falls 3.32
→ 0.37 over the first eight loops and continues down. This is the measurement I would put at the
centre of any looped-model training run (§9).

---

## 8. What this implies for building one

Judgement from here on, marked as such. Three claims that are cheap and would save real budget.

**Two obvious fixes are already refuted by evidence in the released model.**

*"Randomise the loop count during training so the model learns to use many loops."* Huginn already
did — mean 33, 90th percentile 56, 99th percentile 93 — and still saturates by loop 8–16.
Depth-randomised training alone is not sufficient.

*"The recurrence has to learn to hold state."* It does not: an untrained model carries a running
count as well as a trained one. The bottleneck is not capacity to carry state; it is that the model
does not learn to use it.

**The premise of the brief is worth challenging.** "Huginn saturates after ~10" is the starting
point for the task. Huginn also truncated gradient to the last 8 loops — and section 7 shows
contraction does not independently starve early loops, so truncation is the only thing in that
recipe that removes credit, and its `k` sits at the knee. **Its reported saturation may be
substantially a property of the training recipe rather than of looped models.** At 10M parameters
full backpropagation is affordable, so a truncation sweep separates them. Huginn cannot run that
experiment; a from-scratch run can.

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

**Throughout:** log the base rate of every gate and outcome; make every "modification off" condition
reproduce the baseline exactly rather than approximately; bank per-loop logits and states rather
than final loss only; and keep a positive control that is not the optimised metric — group
composition over `S₃`–`S₅` is provably outside `TC⁰`, so depth must help there if the loop works at
all.

---

## 10. Where this points

Ideas, not commitments. Design should be settled against section 9's measurements, not ahead of
them. Ordered by how much the evidence above actually supports them.

**Make the loop able to tell time.** The strongest architectural statement in the record is that the
block cannot distinguish iteration 3 from iteration 30, and a published theorem says post-fixed-point
loops are provably uninformative. A depth-conditioned modulation costs `O(width)` parameters, which
does not grow with model size the way the brief's own bad example does. *What would kill it:* if
depth-conditioning moves the knee no further than a matched parameter increase elsewhere.

**Treat readout commitment as a target in its own right.** The answer peaks at loop 4 of 48 and then
gets worse, and depth buys prose rather than answers. If that reproduces at small scale, the useful
intervention is delaying commitment — per-loop supervision, or a decoder that cannot settle early —
rather than adding loops behind a decoder that has stopped listening. *What would kill it:* if the
per-loop KL curve at small scale flattens only at the very end, meaning there is no early commitment
to delay.

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

## Appendix — the evidence, in one table

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
| D37 | Answer fully decodable after 1 loop, no further improvement; alignment to final form rises 0.45→0.99 over ~16 loops | current |
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
| D159 | Answer displaced to a stable worse rank after ~loop 4, holding 40+ loops, in 17 of 21 families | current |
| D174 | Openings with "The" rise 3.2%→64.7% from loop 2→32; starts-with-answer flat | current |
| D178 | Depth at which the answer appears is 88% predicted by its rank after **one** loop (ρ=0.885) | current; supersedes D177 → D35 → D33 |
| D179 | Exact-string scoring recovers 3.8% of present answers; contains/last-number 24–31% | current |
| D183 | ARC-Easy 49.1/65.1/69.5/69.9% at 4/8/16/32 loops; GSM8K-CoT 0%→34.8% | Huginn's published numbers |
| D186 | Irrelevant length-matched padding raises depth-to-answer 4.73→7.29 while accuracy moves 0.500→0.479 | current |
| D187 | Loop index never enters the block's computation; training sampled loop count randomly (mean 33, p99 93) | current |
| D188 | Published theorem: past the fixed point all further Jacobians and attributions are identical | external |
| D189 | Depth-scales-with-difficulty needs an explicit ponder cost; Huginn had none | external |
| D191 | Identical loop count for every token, no per-position halting | current |
| — | `loop_horizon.py`: gradient mass near-uniform across loops (last 32 of 64 = 53.5%); `1/(1−ρ)` off by 89% | mine, this report |
| — | `loop_horizon_raven.py`: on the real architecture, Jacobian radius 0.9375 < step rate 0.9963; backward sensitivity spans 4.15× over 32 loops; truncation is a hard cutoff | mine, this report |

**Corrected after first publication (2026-08-12):** this report originally asserted that the
spectral radius had never been measured on real Huginn. It had — D31, above. The error was mine, not
the record's; the correction is in §3.4 and §11, and it turns the report's weakest open question
into its strongest measured fact.

**Superseded — do not quote:** the winding-number statistic and everything resting on it; the
"gapped distribution implies two regimes" test (invalid null); "harder problems recruit more depth"
in all its forms; the claimed register direction (window bug); the extrapolated untrained
contraction rate of 0.66; my own `1/(1−ρ)` horizon and its near-isometry explanation.
