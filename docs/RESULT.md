# The latent trajectory encodes the input, not the outcome

*Huginn-3.5B, a recurrent-depth transformer. Inference only. 2026-08-09.*

Every number here cites the row in `claims_ledger.md` that carries its evidence,
its controls and its limits. This document is the argument; the ledger is the
record.

---

## The claim

A recurrent-depth transformer reuses one weight-tied block many times, and the
sequence of latent states it passes through has been treated across this literature
as a trace of reasoning — something whose *shape* should reflect the computation
being performed. We measure that directly and find a dissociation:

> **The shape of the latent trajectory identifies the input with near-perfect
> fidelity — enough to separate two prompts differing in a single token, whether or
> not that token changes what the model computes — and carries essentially no
> information about whether the computation succeeded.**

Both halves are measured on the same orbits with the same classifier, and both
halves are bounded: the positive at ceiling, the negative against a measured
detection floor.

---

## 1. What depth is doing

**It is a contraction, and the contraction is linear where we measure it.**
Held-out one-step prediction of the step directions recovers a median **91%** of
what the fitted subspace can express, and a general operator beats a
subspace-matched scalar rate in **148–152 of 152 orbits** (D82). Where the absolute
score is low the cause is a gapless spectrum — later steps enter directions earlier
steps never spanned — not a failure of linearity, which is what DMD independently
found when its snapshot spectrum showed no gap and failed its stability gate on
80/80 orbits (D74(4)).

**Its geometry is dominated by how far it has run.** In a fixed-width window slid
along the unroll axis, participation ratio falls 8.55 → 4.55 between unroll 0 and 28
— **down in 147 of 147 orbits** — while the step cosine rises +0.276 → +0.710, up in
134/147. Each shift is ~3× the entire prompt-to-prompt spread of the same statistic,
and the contraction *rate* does not move over that range (p = 0.32), which is the
control that makes the other two specific rather than a generic drift (D80).

This dissolves an apparent contradiction in our own record: "the orbit spans 13.3
effective dimensions" (D74) and "the orbit spans 2.3" are the same orbits at a
~21-unroll and a ~90-unroll window. **A shape statistic quoted without its window is
uninterpretable**, which is also why winding's sign flipped with the recording
budget (D28).

**And the collapse is something training builds.** Over 84 untrained orbits from
**six independent weight draws**, on identical prompts and at matched depths, the
trained orbit sheds directions **8× faster** (Δ −3.62 against −0.45) and the
untrained step cosine does not move at all (−0.013, p = 0.063). Random weights
diffuse; the trained map collapses onto a dominant mode (D80(7)).

---

## 2. The shape encodes the input, down to one token — and only the input

Rather than hand-picking statistics, we hand a classifier the **complete
rotation-, translation- and scale-invariant description of the path**: the Gram
matrix of unit step directions, verified invariant on real orbits to
max|Δ| = 6.2e-16. Raw states are run at exactly the same feature count as the
positive control — necessary, because Huginn re-injects the prompt embeddings at
every unroll (D70(4)) and random weights decode a count from the raw states at
R² = 0.99999 (D73), so a classifier fed states decodes anything.

- **Task family: 100% balanced accuracy** within every bank, against permutation
  nulls of 25–33% (D84).
- **At an identical token count: 96.9–100%.** Three task pairs whose two forms are
  byte-identical apart from one trailing marker character, gate-verified in-kernel
  to tokenise to the same length — 188 of 188 items kept, 0 dropped, token multisets
  identical per pair. Pooled **98.9%**, p = 0.0025, the floor (D87).

The second is what makes the first mean anything. Across every previously banked
orbit, **no two families share a token count** (0 of 6 pairs in one bank, 0 of 3 in
another), so "the shape encodes the task" and "the shape encodes the length" made
identical predictions and no statistical control could separate them — the same
situation where `partial_spearman` refused at ρ = −1.000 (D74(6)). It needed a
design, and the design says length was not the explanation.

**But it is the token, not the computation, and the control that shows this is one
we ran expecting the opposite.** With three markers whose rule block defines **A and
C to do exactly the same thing** and B something different, the shape separates
A from C — *same computation, one token apart* — at **98.0–100%**, against A-vs-B's
**96.0–100%**. Pooled, 100.0% against 97.9%. **No gap** (D88). The behavioural gate
passes, so the control is real: the model reaches rank 1 in 0% of `count_vs_last` A,
**58%** of B and 0% of C.

So the trajectory's shape responds to a changed input token whether or not that token
changes what the model does. The pre-registered prediction — a gap would mean
computation — failed, and this is the result rather than a reframing of it.

---

## 3. The shape does not encode the outcome

The same orbits, the same classifier, and three independent ways of asking:

- **Across 21 task families spanning 0% to 100% accuracy**, no geometric statistic
  tracks capability at any of five windows — **0 of 18 usable cells** — while 6 of 18
  track prompt length. The run measures its own reliability at **0.95–1.00**, so a
  *perfect* relation would have shown |ρ| in **[0.80, 1.00]** against an observed
  |ρ| ≤ 0.33 (D85).
- **Within a task at a matched answer value**, correct and incorrect trajectories do
  not differ in effective dimensionality, step cosine, contraction rate or settling
  time — in the one family D79 could test, and in two more from the new bank (0 of 8
  cells at α = 5.95e-04).
- **By the full shape classifier**, correctness is not decodable where family
  identity has been conditioned out, against a measured detection floor.

The reliability ceiling is what makes this a result rather than a shrug. This
project has twice reported a null from an instrument that could not have moved:
D70 read a content gap of ~0 off a probe pinned at R² ≥ 0.87 in both arms, and D72's
design admitted 300 label arrangements so no data could reach α. Each null above
carries the number it could have shown.

---

## 4. Where the computation does live

The trajectory's *shape* is blind to the outcome; the *readout* is not.

- Capability spans **0% to 100%** across the 21 families (D75, D85), so there is a
  computation to track.
- **Depth does not destroy the answer — it wraps it in prose.** On decoded strings
  across 21 families, exact-match accuracy peaks shallow and reaches **0.0% by r=8**,
  while containment rises **13.5% → 33.3%** from r=4 to r=32, up in 9 of 10 families
  that moved (p = 0.0215) (D86).
- **Rank predicts the emitted answer only where the output is still short.** At r=4
  under a constrained prompt, being top-1 raises P(exact) from 3.7% to **58.8%**
  (φ = +0.600); by r=32 nothing is emitted exactly at all and the link is gone (D86).

So the answer is available early and stays available; what deepening changes is the
*form* of the output, not the presence of the content. That is the same object the
shape statistics are blind to.

---

## 5. What this means for reading latent trajectories

The natural inference from a converging latent path is that its geometry traces the
computation. On this model it traces the input and the clock. Concretely:

1. **Shape statistics are largely a depth readout.** Report the window or report
   nothing: the same orbits move ~3× more along depth than across prompts.
2. **A geometric difference between conditions is a difference between INPUTS until
   a same-computation control says otherwise.** Task identity was perfectly
   collinear with prompt length in every design we had; a purpose-built
   length-matched pair separated those, and then a same-computation control showed
   even that separation was the token. This project had no such control before
   today, and the first two versions of its own synthesis were wrong in opposite
   directions without one.
3. **A null needs its ceiling.** Reliability, combinatorial floor, detection
   power — whichever applies. Every null here carries one.

---

## What is not established

- **Trained arm, one token position, synthetic tasks.** D80(7) supplies the untrained
  contrast for the *geometry* but not for the capability axis, and no
  natural-language reasoning benchmark has been run end to end with this
  instrumentation.
- **The marker sits at the end of the prompt**, adjacent to the read position. A
  marker placed early might behave differently, and that is not tested (D88(6)).
- **The correctness null is bounded, not absolute.** It rules out differences above
  the measured detection floor and says nothing below it, and it is a linear
  classifier on one shape code.
- **One model.** Every number is Huginn-3.5B at a pinned revision.
