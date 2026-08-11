# The latent trajectory encodes the input, not the outcome

*Huginn-3.5B, a recurrent-depth transformer. Inference only. 2026-08-09, extended 2026-08-10 with the causal result in §6.*

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
> not that token changes what the model computes — and carries, at most, a weak and
> not-yet-confirmed trace of whether the computation succeeded.**

Both halves are measured on the same orbits with the same classifier, and both
halves are bounded: the positive at ceiling (D84, D87, D88), the negative side
across every hand-picked-statistic and between-family test (D79, D85) against a
measured reliability ceiling. The single most sensitive test run, a within-stratum
full-shape classifier, finds a small, reproducible, family-concentrated signal
(D92) that converged with an independent lead from a second design (D91) —
**and the powered, pre-registered replication of that lead has now returned NOT
CONFIRMED (D93), under an instrument whose own detection floor was found broken,
fixed, and then shown to catch a planted 0.5 sd effect 100% of the time.** So D91's
specific transient-window mechanism is ruled out rather than merely unconfirmed;
D92's between-item result is a different test and stands, still unreplicated.

**AND THE SAME NOW HOLDS OF THE FIXED POINT, WHICH IS THE STRONGER VERSION OF THE
CLAIM.** D109 found that the converged state h\* decodes the specific answer (0.647
dimension-matched, 5 of 5 cells) while the path's shape sits at chance — a clean
dissociation between the two objects. But in that design the answer is a *function of
the input*, so "decodes the answer" and "encodes fine input detail" predict the same
result. A paired-marker experiment separates them: the **same** sequence under two
task markers requires **different** answers. Across 421 cross-marker pairs at a
verified-constant token count, sharing the required ANSWER raises h\* similarity by
**+0.00016 (z = +0.40, p = 0.70)** — nothing — while sharing the INPUT raises the same
statistic on the same orbits by **+0.0034**. The answer effect is **4.3% of the input
effect** (D125). So the title of this document is true of the trajectory's shape *and*
of its endpoint: **what the latent carries is the input.**

*(Limit, stated because it is the one that could overturn this: capability is low in
that experiment — 35 of 225 orbits reach rank 1, one marker sits at 4.7% — so it
cannot separate "the fixed point does not encode answers" from "no answer was
computed to encode". The input effect is immune to that, since the input is present
either way.)*

**A second claim, established the same day and independent of the first:**

> **The iteration is a uniform contraction at a fixed block, and the
> non-contracting kind of loop is absent — though two other things the literature
> calls a "loop" are present.** The per-step contraction factor measured the honest
> way — between two orbits of the *same* prompt from *different* random
> initialisations — is ρ = 0.855 raw — **≈0.82 once the +0.033 bias our own D58 diagnosed in this
> log-linear fit is applied** — with **0.0% of orbit pairs at ρ ≥ 1** across
> **16 prompts** (n_eff is the prompt, not the ~960 pairs, which are replicates;
> by the rule of three that excludes a per-prompt rate above ~19%, no better).
> The independent Arnoldi/JVP route (D31) gives 0.79–0.81, so the two agree to
> about 0.02 rather than coinciding. Independently,
> classifying every token position's own path with a settle/loop/drift
> classifier gives **179 of 179 positions `settle`** (D97).

That second claim carried an explicit caveat, we tested it, and **the caveat won.**
Both instruments read the state at the *same* block of the recurrent stack. Hooking
all four core blocks instead (D98): each converges tightly to its **own** fixed
point (per-block residual 58–71 → ~0.1), and those four fixed points are separated
by **~52% of the state norm** — a separation-to-residual ratio of **95–730** — with
the cycle's perimeter **exceeding the entire distance travelled from initialisation
to convergence by 60%**.

> **There is a large, stable, period-4 cycle across the blocks, and reading one
> block samples a single point of it forever.**

The cycle is real structure rather than an artefact of having four points, and the
nulls that establish that come from a third measurement: **the state is confined to
a sphere.** RMSNorm terminates every block, and on our banked data ‖h‖ = 76.386
with per-orbit relative sd of median 5.0×10⁻⁵ over all 512 orbits (D99). That fixes
the random baseline exactly — two random points on that sphere lie 108.03 ± 0.75
apart — so the cycle's vertex separation of ~38 is **0.35× chance**, far closer
than random, while its planarity of 0.933 sits **78 standard deviations above** the
0.676 ± 0.003 expected of four random points in 5280 dimensions (D100).

It also has a consequence for the hypothesis being tested: on a compact manifold,
**unbounded drift is not an available asymptotic regime at all**, so the
settle/loop/drift trichotomy is exhaustive rather than three guesses, and reduces
to the sign of the top Lyapunov exponent.

Both statements are true of different objects: no loop *in the
iteration-to-iteration map at a fixed block*, and a large loop *across blocks
within an iteration*. The second claim is therefore scoped, not withdrawn — no
number in D94 or D97 changes — and this independently reproduces on Huginn a
published claim we previously held only at abstract-level confidence.

---

## 0. Why the nulls happen — a chain, not a list

The results above are mostly negatives, and a run of negatives invites the reading
that the instruments are weak. Three measurements made on 2026-08-09 close that off
by supplying the mechanism, and each is a *rate* or a *link* rather than another
absence.

**(a) Difficulty does not reach the dynamics.** Fitting the contraction rate per
task family over 608 banked orbits across 21 families, ρ spans **0.8239 to 0.9181**
with a between-family spread **4.2×** the within-family spread — and it survives
length matching, which it had to, since the marginal length correlation is real
(+0.56, p = 0.011): at an identical 29 tokens `nth_item` gives 0.8370 and `sort_min`
0.8910, a gap **7.5×** the within-family sd. But on the one genuine difficulty
ladder in the bank, `count4 → count8 → count16`, quadrupling the count moves ρ by
**0.0033** — under half the within-family sd (D115). The rotation angle behaves the
same way and is independent of the modulus (ρ = +0.11): task identity moves it
38–83° across families, difficulty moves it nowhere (pooled ρ = −0.073, p = 0.62,
with an *untrained control*) (D116). **This is an H2 test on a rate rather than a
duration, so D80's clock critique cannot apply to it — and H2 still returns
nothing, while task identity moves the same quantity by an order of magnitude
more.**

**(b) The dynamics do not reach the readout.** Joining the causal per-prompt
contraction rate to the per-prompt depth at which the answer is best available gives
ρ = **+0.118, p = 0.65**, against a floor that detects a relationship three times
smaller than the variation actually present (D114).

**(c) Because the readout never waits for the fixed point.** On the orbits Huginn
actually solves, the answer is top-ranked at **median unroll 4**, when **86.9% of
the state's journey still lies ahead**; those same states do not reach a residual
below 0.01 until median unroll **44**. Replicated on an independent bank: 202 solved
orbits across 21 families give unroll 4 and 82.9% remaining (D112). *This also
reconciles our median-34 convergence depth with the published median-6 without
needing the published number — the two are reading different objects at roughly the
4–5× ratio the literature reports.*

Read together: **difficulty does not move the dynamics, the dynamics do not move the
readout, and the readout is finished long before the dynamics are.** Every H2
instrument this project built measured the trajectory; the answer was already
decided in the first few unrolls.

One caution about our own numbers, since the same day retired two of them: the state
lives on a sphere of radius 76.386, and that alone fixes quantities that read as
measurements. The difference of two same-prompt orbits is orthogonal to their
midpoint *by identity*, and D106's "20.6% radial component" is |d|/(2R) — a
restatement of step length that would hold for a random walk (D122). D106's
conclusion stands; "is this a coordinate artefact?" can no longer be tested by
decomposing steps radially.

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

## 3. The shape mostly does not encode the outcome — with one loose thread

Every HAND-PICKED-statistic and BETWEEN-family test is a clean null:

- **Across 21 task families spanning 0% to 100% accuracy**, no geometric statistic
  tracks capability at any of five windows — **0 of 18 usable cells** — while 6 of 18
  track prompt length. The run measures its own reliability at **0.95–1.00**, so a
  *perfect* relation would have shown |ρ| in **[0.80, 1.00]** against an observed
  |ρ| ≤ 0.33 (D85).
- **Within a task at a matched answer value**, correct and incorrect trajectories do
  not differ in effective dimensionality, step cosine, contraction rate or settling
  time — in the one family D79 could test, and in two more from the new bank (0 of 8
  cells at α = 5.95e-04).

The reliability ceiling is what makes these results rather than a shrug. This
project has twice reported a null from an instrument that could not have moved:
D70 read a content gap of ~0 off a probe pinned at R² ≥ 0.87 in both arms, and D72's
design admitted 300 label arrangements so no data could reach α. Each null above
carries the number it could have shown.

**The one test that is NOT a clean null is the most sensitive one.** The full
shape-code classifier, stratified by (family, gold value) with its centring fitted
inside each cross-validation fold, decodes correctness at **61.0% balanced accuracy
against a within-stratum null of 50.2%, p = 0.030** — stable 0.010–0.062 across six
CV seeds — pooled over 86 orbits in 5 strata from 4 banks. It is driven by 2 of 3
well-populated strata (`nth_item` 93.8%, p = 0.005; `local_last` 74.6%, p = 0.020)
with the third flat (`parity8` 43.8%, n.s.), and the design's own detection floor
shows it reliably catches only effects at or above ~1 standard deviation (D92).
That is a small, fragile result — not powered, not pre-registered, carried by a
minority of strata — but it converges with an independently designed sliding-window
test that found correctness decodable specifically at unrolls 6–24 and nowhere else
(D91), in the location an unrelated theoretical paper predicts the algorithm should
live if it lives anywhere (arXiv:2607.20594). Two different designs landing on the
same qualitative answer is worth more than either alone, and neither is confirmatory
by itself. The powered, pre-registered test (`geometry-h0bank`,
`scripts/run_h0_within.py`) was launched specifically to settle it.

**It has now returned, and D91's lead does not replicate (D93).** 512 orbits from
16 boundary prompts × 32 unseeded initialisations, 10 of 16 prompts splitting on
correctness so the pre-registered gate passed. Best window (unrolls 6–18): **52.0%
against a 46.8% null, p = 0.0125** — missing the corrected α = 0.00625 by about
2×, with no far-tail window significant either. The decision rule, the window grid,
the statistic and the null were all fixed in a file committed before the data
existed.

**The reason that null is worth believing is that the instrument was caught failing
first.** The run's planted-effect detection floor reported 0% detection at every
effect size *including 1.0 sd*, which reads as a hopelessly underpowered design and
is in fact an arithmetic impossibility: it used 40 permutations against α = 0.00625,
and the smallest p-value reachable with 40 permutations is 1/41 ≈ 0.024. The check
could not have fired whatever the data said. Re-run at 400 permutations it detects
a planted **0.5 sd** effect **100%** of the time. So the design would have seen an
effect several times smaller than what D91 reported, and saw nothing.

Its timing control adds one more qualification: `best_depth` alone — *when* the
answer peaks, which varies 7–25 within a single prompt — decodes correctness about
as well as the best shape window does. So even the near-miss is not cleanly about
shape.

One loose thread is left honestly open: the *position* comparison arm (raw state
features, not shape-invariant) reaches p = 0.0050 at that same 6–18 window, which
clears the per-window threshold. It was never part of the pre-registered gate and
would not survive correction across the full 24-row grid actually tested, so it is
reported as an uncorrected lead, not a finding.

---

## 4. Where the computation does live

The trajectory's hand-picked statistics are blind to the outcome, and its full
shape mostly is too (§3); the *readout* carries it clearly.

- Capability spans **0% to 100%** across the 21 families (D75, D85), so there is a
  computation to track. **Caveat carried from D89, and it applies to the central
  null's independent variable, not to a side result:** that axis is scored on the
  FIRST TOKEN of the gold, and **8 of the 21 families have multi-token golds — 4 of
  them for every item**, so `compare`'s 92% is first-*digit* accuracy. The null was
  re-tested against a tokenisation-free axis (decoded strings from
  `geometry-depthacc`) and **survived**: no metric tracks capability at any window
  after the correction. So the defect does not overturn the null — but every number
  quoted from the first-token axis is a first-token number and should be read as one.
- **Depth does not destroy the answer — it wraps it in prose.** On decoded strings
  across 21 families, exact-match accuracy peaks shallow and reaches **0.0% by r=8**,
  while containment rises **13.5% → 33.3%** from r=4 to r=32, up in 9 of 10 families
  that moved (p = 0.0215) (D86).
- **Rank predicts the emitted answer only where the output is still short.** At r=4
  under a constrained prompt, being top-1 raises P(exact) from 3.7% to **58.8%**
  (φ = +0.600); by r=32 nothing is emitted exactly at all and the link is gone (D86).

So the answer is available early and stays available; what deepening changes is the
*form* of the output, not the presence of the content. The hand-picked statistics
and between-family tests are blind to that content (§3); whether the full shape
classifier's weak signal (D92) is a trace of it is exactly what remains open.

---

## 5. What this means for reading latent trajectories

The natural inference from a converging latent path is that its geometry traces the
computation. On this model it traces the input and the clock. Concretely:

0. **The iteration is a contraction, and that governs where an intervention can
   act — but not in the direction we first stated.** ρ ≈ 0.855 (raw; ~0.82 after
   the bias correction our own D58 diagnosed) between orbits of the same prompt
   from different initialisations (D94); 179 of 179 token positions classify as
   `settle` (D97). A perturbation injected at iteration *r* is attenuated by
   ρ^(R−r), so the exponent *shrinks* as *r* grows: ~1843× at r=16 but only ~12×
   at r=48 for R=64. **Late interventions survive; early ones decay** — the
   opposite of what an earlier version of this document said.
   The practical lesson is different and sharper: perturbing the **state** *h*
   perturbs an initial condition of a contraction, while the re-injected prelude
   output *e* is the map's **parameter**, so perturbing *e* moves the fixed point
   *h\*(e)* itself and persists. And patching the same prompt from a different
   random initialisation is **inert at both ends by construction** — the two runs
   share an attractor, so at late *r* there is nothing left to import. Anyone
   planning activation patching on a recurrent-depth model should patch the
   injected stream across *different prompts*, and should build a potency-vs-*r*
   calibration curve before interpreting any null.
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

## 6. The causal version of the claim — one word, and the map itself

Everything above is measured on the trajectory's *shape*. The same dissociation holds one
level down, in the *dynamics*, and there it is causal rather than correlational.

**Eighteen sequences under three instructions differing in one word** — *"report the
largest **symbol** / **element** / **item** of the sequence"* — both markers, token count
verified in-kernel at 53 for all three, and the banked gold answers identical in 36 of 36
cells. `symbol` rotates **36/36**; `element` and `item` rotate **0/36**. The two regimes
differ on every statistic measured, not only the period: late-step ratio 0.0217 vs
0.0004/0.0014, H₁ persistence 0.02005 vs 0.00000/0.00225, fitted modulus of the 6-mode
0.9351 vs 0.8811/0.8940 (D132). `item` is what makes it a result rather than a
coincidence — it behaves like `element`, so `symbol` is the special one.

**Seven candidate causes were tested and eliminated**: meaning (semantic neighbours of
`symbol` rotate 0/144 while neighbours of `element` rotate 96/120 — an inversion), surface
form, subword structure, symbol diversity (0 of 132 orbits cross the threshold), sequence,
marker, and the learned embedding both linearly and nonlinearly (best-of-six leave-one-out
0.551 against a null-of-maximum 0.589, p = 0.808) (D129, D134, D135, D136, D137, D139).

**Then the manipulation.** `e` is the map's *parameter*; it can be set to values no token
produces. Interpolating `e` along straight lines: all **8** cross-regime paths cross the
regime threshold **exactly once**, each crossing inside a single 0.05 grid step; **5**
rotating→rotating paths never leave (21/21 points inside); and at t = 1 the orbit matches
the rotating noun's own statistic *even though h₀ came from the settling prompt* — the
attractor follows the parameter (D140, D144).

**Direction, not distance.** Travelling the *same euclidean distance* from a carrier's `e`
in a random bearing: **none of 8 random bearings rotates**, and on a rotating carrier a
random bearing of that magnitude **destroys the rotation in 3 of 3 cases** (0.845 → 0.579,
0.977 → 0.271, 0.974 → 0.131), while the chord toward another rotating noun keeps all 21
points inside. The rotating set is a structured region of parameter space, not a
neighbourhood any large perturbation escapes (D144).

Scoring throughout uses a threshold validated before use: one cut at R > 0.6677 reproduces
the independently computed period-6 label on **691 of 696 orbits** across three banks
(D141). The patch itself is proven exact: with h₀ seeded, a no-op patch reproduces the
unpatched orbit at **0.000e+00** on all 8 chords (D144).

**And it has no behavioural consequence — now bounded, not merely unrefuted.** D132's
observational comparison found nothing (16.0 vs 14.5, p = 0.156; 6.5 vs 7.0, p = 0.880)
but computed no detection floor, so it excluded nothing. A regression discontinuity across
the boundary — the *same* prompt at two parameter values 0.05 apart in `e`, 48 of 48 units
with exactly one crossing, the no-op patch verified to reproduce the rank curve identically
48/48, and the readout verified to respond to `e` — bounds it in part: `best_depth` is null against a floor
that catches a planted 0.5 sd (4.7 unrolls) 99.3% of the time and 0.25 sd never; and the only
signal is a **one-rank shift** in 16 of 48 pairs, every difference exactly ±1, 13 in the same
direction, exact sign test p = 0.0213 — which does *not* survive correction over the three
measures tested (D148). Depth effects below ~4.7 unrolls remain untested.

**The correctness arm of that test is withdrawn, and it was the load-bearing one.** `correct`
is `False` in **432 of 432** orbits — the gold never reaches rank 1 anywhere in the run, in
either regime, at any grid point, best rank 3. "Correctness changes in 0 of 48 units" is a
statement about a variable that is identically zero on both sides of the boundary. Huginn
cannot do the task the geometry was scored on, and that was never checked. So the sharpest
form of this document's thesis — *a single word restructures the entire latent dynamics
without changing what the model answers* — **is not established.** Half of it is measured: the
restructuring is real, sharp, replicated and causally controlled. The other half has never
been tested on a task this model can do, and that is now the strongest open question here.

---

## 7. The capability axis, measured against itself and against the world

This is the strongest objection to everything above, and it is now measured rather than
asserted. Three findings, in order of how much they cost the record.

**The axis behind most numbers here is an oracle over depth, and on the census it is
inflated 5.29×.** Almost every accuracy figure in this ledger is `min(rank) == 1` over the
unroll budget — the gold reaching rank 1 at *any* depth. Rescoring the census's 4868 banked
draws at the **final** unroll instead: mean per-item accuracy falls **0.336 → 0.063**, and
its 41 items in the 20–80% "dynamic range" band become **0** (D157). Set against D103's
2.06× on our own tasks and A29's 2.00× on ARC-Easy, the inflation is a property of task and
budget, **not a constant — no single factor should be quoted.**

**Scored at the final unroll, 19 of 21 task families read exactly 0.000.** Including
`echo_digit` — "repeat this number", the easiest task in the suite — at **0.997 → 0.000**.
The model puts the gold digit at rank 1 by unroll 2 and, at the end of its own 48-unroll
computation, has it at rank 4. Only two families hold their answer: `echo_word`
(**0.861 → 0.833**) and `compare` (0.899 → 0.500). **Huginn holds a copied word to the end
of its computation and cannot hold a copied digit** (D158).

**And the answer is displaced, not decayed.** Median argmin unroll **4**; 17 of 21 families
reach their best rank by unroll 8; then the rank **plateaus at a stable worse value** for
forty-plus unrolls — `add1` at 19, `echo_digit` at 4 (D159). The map converges to a state
whose readout does not prefer the gold.

**What it prefers instead is `The`.** Banking the top-5 tokens at every unroll: the final
argmax is a prose sentence-opener — `The` ×12 of 32, `One` ×3, `Number` ×2, `To` ×2 — taking
`sort_min` **8 of 8** at ~92% probability while the gold sits at rank 2. The gold's final rank
is median **2.5** and it is still in the top 5 for **24 of 32** items (D166). So **the
final-unroll axis measures answer formatting**: it asks whether the model opens with the bare
gold token, and Huginn opens a sentence. The numbers above stand; the reading *"Huginn cannot
hold a copied digit"* does not — it holds the digit at rank 2–4, behind `The`. `echo_word`
survives for the reason that explains the whole dissociation: the natural prose answer to
*"repeat this word"* **begins with the word**, and its argmax is the gold's own first token in
8 of 8. Strip the chat template and the answer genuinely goes — median final rank **10**, one
item at **509**, with `\n` winning 23 of 32.

**Seen in generated text rather than in ranks, on an independent bank, the same token wins.**
`kaggle_depthacc` holds 1260 generations over 21 families at five depths, banked weeks earlier
for another purpose. **`The` opens 496 of them — 39.4%** — then `To` (7.4%), `Text`, `There`,
`Letter`, `Sequence`. Exact-match reads **0.020** where the answer is actually present in
**0.255**, a **12.75×** gap, and only 8.3% of generations begin with the gold. The literal text
is the argument (D170):

| family | gold | what Huginn actually wrote | scored |
|---|---|---|---|
| `sub1` | `3` | *"The answer is 4 - 1 = 3"* | **wrong** |
| `sort_min` | `18` | *"The smallest of these numbers is 18"* | **wrong** |
| `echo_digit` | `4` | *"The number 4 is repeated exactly."* | **wrong** |
| `local_last` | `Subtract` | *"The last instruction is \"Subtract 1."* | **wrong** |
| `compare` | `83` | *"83 is larger than 23."* | right |

**And more depth buys containment but never exactness**: across r = 2 → 32 containment rises
0.222 → 0.333 bare and 0.159 → 0.389 constrained, while exact-match stays at ~0.000 throughout.
Depth is spent on the completion, not on the decision — the behavioural counterpart of the rank
being settled by unroll ~4.

**The collapse has two causes, not one, and only one of them is formatting.** Restricting the
census to the 10 families with oracle accuracy ≥ 0.35 — the only ones where a final-unroll zero
can mean anything — gives 2720 draws in which **the gold's final rank never exceeds 35 and none
sit beyond rank 100**, in a 65k vocabulary (`final_rank` is uncapped in that bank; it reaches
8971 elsewhere). But the pool splits: copy/compare/select tasks sit at median final rank **2.0**
(top-5 in 0.896), while `add1`/`sub1`/`add_2d` sit at **16.0** (top-5 in 0.000). A prose opener
costs one or two positions — that is the copy group exactly. It cannot explain arithmetic at 16,
and A35's literal top-5 for `add1` shows what does: `['One', 'The', 'To', '1']` — prose openers
**and the input digit**, gold outside the top five. **For arithmetic the model is uncertain
about the answer itself, which is capability, not formatting** (D168). So D157's 5.29× is
decomposed rather than restated, and one factor should never have been quoted across both.

**The size of the gap, with the control that makes it mean something.** Generating 24 tokens
on four families where the model is competent: **first-token exact-match reads 0.125 while the
answer is actually produced in 0.781** of generations (0.797 at r = 48) — a **6.3×** gap. The
obvious objection is that containment is cheap for single-character golds, so the cross-item
false-positive rate was measured rather than assumed: how often does *some other item's* gold
appear in a generation? **`echo_word` 0.000, `echo_digit` 0.042, `sort_min` 0.062, `add1`
0.152** — against observed containments of 1.000 / 0.750 / 0.625 / 0.781, lifts of **+0.562 to
+1.000** (D172). Containment on these families is informative, not chance.

That also corrects something I had over-read. D168's arithmetic group sits at median final rank
16 and I called it *uncertainty about the answer*. It is not: `add1` writes **"One more than 1
is 2"** — gold `2`, correct, in the first clause — at 0.781 containment against 0.152 chance.
**The model is computing in prose; the answer simply is not at position 1.** The failure mode
when it comes is visible and is not silence: *"One more than 1 is 2. When we add one to 2, we
get 3. When we take 3 modulo 1"* — right, then continuing past the answer and derailing.

**And the whole thing has one mechanism, with a timetable.** The gold's rank is settled by
unroll ~4 (D159). **By unroll ~8 the model has committed to a prose frame**, and the transition
is sharp: the rate at which a generation opens with `The` runs **0.032 → 0.099 → 0.631 → 0.560
→ 0.647** across r = 2/4/8/16/32, a 20× rise, while the median output jumps from 7 to 30.5
characters between r = 4 and r = 8. **Meanwhile the rate at which the generation starts with the
gold does not move at all: 23/252 → 18/252, p = 0.515** (D174).

That is the uncomfortable form of the claim. Containment rises with depth, starts-with-gold is
flat, prose-opening more than doubles — so **on this model, spending more compute makes the
standard accuracy metric worse while making the underlying output better.** Every
depth-versus-accuracy result in this project was read through that metric.

**And the external calibration, which failed all day, now works — because the block was
elicitation.** Huginn's paper reports **69.9%** on ARC-Easy at r = 32. Zero-shot we measure
**0.407** by option-letter argmax and **0.433 / 0.413** by option-text likelihood; the three
protocols agree within ~5 points at every depth, so changing the *scoring rule* does not
explain the 29-point gap (D153, D162). **Adding in-context examples does.** Letter-argmax
accuracy runs **0.416 → 0.713 (2-shot) → 0.723 (5-shot)** against the published 0.699 — two
shots already clear it — and the effect is paired: over 101 items, 5-shot fixes **35** that
0-shot got wrong and breaks **4**, exact McNemar **p = 3.35e-07** (D169).

**The gain is protocol-specific, and that is the mechanism rather than a caveat.** Over the
same shots the option-text arms move ~2 points (raw 0.495 → 0.545, normalised 0.465 → 0.465)
while the letter arm moves **30.7**. A likelihood score never requires the model to *emit*
anything. The letter arm requires a bare option letter at the first generated position — and
zero-shot Huginn opens with prose there instead (`The` ×12 of 32, D166). **The answer was
available at zero-shot and could not get out**, which is the same thing D168 measures when it
finds the gold never leaves the top 35 on tasks the model can do. Few-shot exemplars
demonstrate the output format and the block clears.

**And there is a second, independent route to the same number that needs no examples at all.**
Scored exactly as `lm-evaluation-harness` scores ARC-Easy — bare `"Question: …\nAnswer:"`, no
option list in the prompt, continuation likelihood normalised by characters — Huginn reads
**0.658** at **zero** shots, within 0.041 of the published 0.699. Decomposing which part of our
prompt was costing us: the chat template is worth **+0.033** and character-vs-token
normalisation **+0.025**, while **removing the option list from the prompt is worth ~+0.21**
(D175). That last one was not what I expected — I expected the template, since stripping it
moves the gold's median rank from 2.5 to 10 (D166). Listing the options in the prompt disturbs
a likelihood comparison between those same option texts, which is a scoring-design point rather
than a fact about the model.

So this project now reproduces a published number about the model it studies by **two
independent routes** — five in-context examples (0.723) or the harness's own prompt format at
zero shots (0.658), with 0.699 between them. It took until row 169 because the obstruction was
in how we asked, not in what we measured.

*(One arm of that run failed its own replication gate and the numbers from it are void: my
`T_chat` reimplementation of D162 put the leading space in the prompt rather than the
continuation, and in BPE that misaligns the scored span. D162's own numbers stand; the
option-list figure above is quoted against D162's banked 0.413, not against the broken arm.)*

**One thing this does not license.** The tempting corollary — that because the answer lives
in the transient, transient geometry predicts correctness where the fixed point does not —
is checked and **unsupported**: D93's powered window sweep already covers the argmin and
finds no trend across eight windows, with the argmin window itself at chance (D163).
*Where the answer is available* and *where the geometry predicts success* are different
questions, and only the first is established.

---

## What is not established

- **Trained arm, synthetic tasks.** D80(7) supplies the untrained contrast for the
  *geometry* but not for the capability axis, and no natural-language reasoning
  benchmark has been run end to end with this instrumentation. *(The "one token
  position" limitation that stood here was CLOSED by D97: all 179 positions were
  measured, they converge in a tight 31–36 interquartile band, and the answer
  position sits at a mean 30th percentile — so single-position reading was not
  sampling an unrepresentative phase.)*
- **Everything is read at ONE BLOCK of the recurrent stack**, and this is now the
  load-bearing caveat. If the literature's "loop" is a cycle *across* the four core
  blocks — each converging to its own fixed point — then D94 and D97 are consistent
  with a cycle rather than evidence against one. The test is running; until it
  returns, the contraction claim is scoped to the iteration-to-iteration map at a
  fixed block.
- **The marker sits at the end of the prompt**, adjacent to the read position. A
  marker placed early might behave differently, and that is not tested (D88(6)).
- **The correctness picture is narrowed but not closed.** The clean nulls (D79,
  D85) rule out differences above their measured detection floors and say nothing
  below them. D91's sliding-window lead is now **ruled out** by a powered
  pre-registered replication under a validated instrument (D93). D92's
  between-item stratified result (61.0% vs 50.2%, p = 0.030) is untouched by that
  test and remains small, not pre-registered, carried by 2 of 3 strata, and
  unreplicated.
- **Causal evidence is no longer absent — the instrument works (D108).** Patching
  the *re-injected prelude output* `e` from a different prompt **flips which answer
  the model prefers in **35 of 112 measurements (31.2%)**, at a mean potency of
  **+1.598 nats** (D113). *This paragraph previously quoted D108's 42 of 56 (75%) and
  +3.97 nats; the ledger records those as measured into a SINGLE recipient, and at 16
  distinct recipients the rate and potency fall by ~2.4×. The flatness in r and the
  0/112 state-arm contrast both survive, so the conclusion below is unchanged.* The earlier patching null
  (D95) was uninformative rather than negative: it patched the **state**,
  same-prompt with a different random initialisation, which is inert at both ends
  because donor and recipient share an attractor. `e` is the map's *parameter*, and
  perturbing it moves the fixed point itself. The effect is **flat in the unroll at
  which patching begins** — 87% of it is present with only 8 unrolls remaining —
  so the map reaches its new fixed point almost immediately. What is *not* yet
  shown: that the donor's answer becomes the global argmax, and any claim of
  selectivity rather than potency.
- **Task supply is no longer the binding constraint, but the ceiling is low.** The
  observability census (D130) yields 31 usable items across 5 families, and a difficulty
  ladder satisfying every structural criterion at once now exists (D143: 37 tokens at all
  eight levels, all eight in the live band). H2 is null on it once list position 1 — which
  needs no counting — is removed: +0.2825 (p = 0.0051) becomes −0.0286 (p = 0.7986).
  **The remaining constraint is that the model's accuracy on these tasks runs 6–40%.**
  Two earlier experiments died on task supply outright. Tasks with clean difficulty knobs are synthetic and
  score near zero; tasks the model genuinely does (GSM8K 32.6%, ARC-E 69.9% at
  r=32, from the model's own paper) are natural-language with uncontrolled
  difficulty and multi-token answers. Runs aimed squarely at this are in flight.
- **One model.** Every number is Huginn-3.5B at a pinned revision.
