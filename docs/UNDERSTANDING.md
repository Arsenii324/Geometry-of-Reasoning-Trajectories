# State of understanding — 2026-08-09

Supersedes the 2026-08-08 revision, which predates D73–D83. Written to be read on
its own; every number cites the ledger row that carries its evidence. §1 is the
current synthesis; **§1-old below is the 2026-08-08 text, kept unedited** because it
is what the D68/D69 work concluded at the time and the difference is informative.

---

## 1. The synthesis, as of 2026-08-09

**One sentence: the geometry of Huginn's latent trajectory is set by the weights,
by how far the contraction has run, and by the input — down to a single token,
whether or not that token changes what the model computes — and carries, at most, a
weak and not-yet-confirmed trace of whether the computation succeeded.**

**And as of the 2026-08-09 evening pass, the contraction is no longer an assumption
but a measurement, under two independent instruments: the map contracts at
ρ ≈ 0.855 raw / ≈0.82 bias-corrected, with 0 orbit pairs at ρ ≥ 1 across 16 prompts (D94; n_eff is the prompt, not the ~960 pairs), and 179 of 179 token
positions classify as `settle` with zero loops or drift (D97).**

**A third instrument has now measured ρ *causally*, and it agrees.** Injecting a
donor's state at unroll r and reading how much of the kick survives to the output
gives a potency that decays as ρ^(R−r) across nearly three orders of magnitude
(D111). Across **16 distinct recipient prompts** this fits **ρ = 0.8335, sd 0.028,
with 14 of 16 inside the 0.79–0.86 bracket** the passive instruments had already
set (D113). Every earlier estimate was a passive fit to how fast an unperturbed
orbit stops moving; this one perturbs the system and watches the perturbation die.
**D94's row should no longer be read as uncorroborated.**

**But ρ is a property of the task family, not of the architecture.** Over 608
banked orbits across **21 families**, the rate spans **0.8239 to 0.9181**, with
between-family spread **4.2×** the within-family spread — and it survives length
matching, which it had to, since the marginal length correlation is real
(+0.56, p = 0.011): at an identical 29 tokens `nth_item` gives 0.8370 and
`sort_min` 0.8910, a gap **7.5×** the within-family sd (D115). So D94's 0.855,
D31's 0.79–0.81, D113's 0.8335 and D112's 0.911 are not competing estimates of one
number; each should be read with its prompt family attached.

**Three results now explain the whole run of H2 nulls, and they form a chain.**

1. **Difficulty does not reach the dynamics.** Quadrupling the count in
   `count4→count8→count16` moves ρ by 0.0033, under half the within-family sd,
   while *task identity* moves the same quantity by 15× that at matched length
   (D115). This is an H2 test on a *rate* rather than a duration, so D80's
   clock-reading critique does not apply to it — and H2 still returns nothing.
2. **The dynamics do not reach the readout.** Joining per-prompt ρ to per-prompt
   `best_depth` gives ρ = +0.118, p = 0.65, against a detection floor that would
   have caught a relationship three times smaller than the variation present
   (D114).
3. **Because the readout never waits for the fixed point.** On the orbits Huginn
   actually solves, the answer is top-ranked at **median unroll 4**, when
   **86.9% of the state's journey still lies ahead**; those same states do not
   reach a residual below 0.01 until median unroll 44 (D112). The readout and the
   latent are two clocks, and every H2 instrument built here has been reading the
   one that difficulty does not drive.

D112 also pins the slowest mode: on `lenmatch` prompts the late motion rotates
exactly 60° per unroll, |h_t − h_{t+6}| < |h_t − h_{t+1}| in 124 of 188 orbits,
decaying with **|λ| = 0.911 in 124 of 124** — a damped rotation, which leaves D97's
`settle` verdict untouched since a damped rotation still settles. *(D119: do not
read that turning rate as arg λ. The full-operator Arnoldi's own gate puts the
orbit's rotation at a median **0.241×** |arg λ|; the two coincide only when a single
pair dominates.)*

**Where the eigenvalue itself has been measured, it agrees and it is task-shaped.**
`ds_eigen` runs Arnoldi on the complete unroll map (adapter + all four blocks):
\|λ\| median **0.8098**, and **arg λ separates tasks cleanly** — `track` 61–63°
across every difficulty, `local` 19.5–24.3° (D119). Four instruments of three
unrelated kinds — passive orbit decay, diagonal-block Arnoldi, full-operator
Arnoldi, and causal state injection — now put the contraction rate in **0.79–0.87**.

**And the whole structure is learned, early.** Across 8 public training checkpoints
and 4 random inits: untrained ρ = **0.7042** (sd 0.0091), trained ρ = **0.8582**
(sd 0.0389), with **total separation** — the minimum trained value exceeds the
maximum untrained value, so all 142 orbits are classified by ρ alone. Training
pushes the map toward the edge of stability, **and it is done by step 6144**, the
earliest checkpoint that exists (the next 35,584 steps move it only within noise,
p = 0.194). Training also multiplies ρ's across-prompt spread **4.3×**, so the
task-dependence above is something training *built* (D120). This is the quantitative
partner to D116's finding that training replaces a near-orthogonal walk
(110–115°/step) with coherent rotation (42–59°).

**One caution about this project's own numbers.** The state lives on a sphere of
radius 76.386 (D99), and that alone fixes several quantities that read as
measurements: the difference of two same-prompt orbits is orthogonal to their
midpoint *by identity*, so radial-vs-tangential comparisons of it are vacuous; and
D106's "20.6% radial component" is |d|/(2R), a restatement of step length that would
hold for a random walk (D122). D106's *conclusion* stands — the artefact's
consequence is 4.2% of the effect it might have explained — but "is this a
coordinate artefact?" can no longer be tested by decomposing steps radially.

**The same experiment turns the DEQ framing from an architectural argument into a
measured fact.** `e` (the re-injected prelude output) is the map's PARAMETER and `h`
its STATE: patching `e` flips which answer the model prefers 6 of 8 times **at every
injection depth, including r = 40 with 8 unrolls left**, and its potency is flat in
r — because moving `e` moves the fixed point h\*(e) itself. Patching `h` never flips
the answer (0 of 56) and decays at exactly the contraction rate, because it displaces
an initial condition the contraction erases (D111).

**H2's behavioural arm is now closed, and it closes against H2.** On `addk`, harder
problems reach their best readout depth **earlier**, not later: ρ(k, best_depth) =
−0.440 at the true problem unit, and the two stratifications that move the first
operand in *opposite* directions both give ≈−0.5 while holding difficulty fixed
gives +0.07 — so the driver is difficulty, not the answer token (D110). In H2's
pre-registered positive direction p = 0.995. This also reconciles D101's +0.362,
which came from a design where the answer was 93% collinear with difficulty. The
effect is ~1 unroll in 48 and post-hoc on one family, so it is a direction, not yet a
law — but it converges with the two results above: the fixed point arrives fast, and
on harder problems the model settles sooner onto a worse answer.

**But both instruments read ONE block of the recurrent stack, and D98 shows that
choice was hiding the loop.** Hooking all four core blocks: each converges tightly
to its *own* fixed point (residual 58-71 → ~0.1), and those four fixed points are
separated by ~52% of the state norm — a separation/residual ratio of 95–730, with
the cycle's perimeter exceeding the entire distance travelled from h₀ to
convergence by 60%. **So there IS a large, stable, period-4 cycle in block-space,
and a `core_block[-1]`-only read samples one point of it forever.** Both readings
are true of different objects: no loop *in the iteration-to-iteration map at a
fixed block*; a large loop *across blocks within an iteration*. D94 and D97 are
scope-corrected, not retracted — no number in them changes. The binding constraint
on further progress is task supply, not instrumentation.

*(This sentence has been rewritten three times in one day, and the history is the
evidence. It began as "…not of the input or of the computation". D84 refuted the
first half — the shape identifies the input at ceiling — but could not separate task
from prompt length, since no two families in any bank share a token count. D87 then
settled that: at verified-identical token counts, with prompts differing in exactly
one marker character, the shape separates two computations at 96.9–100%. The second
half — "not by whether the computation succeeded" — held through D79/D84/D85's
hand-picked-statistic and between-family tests, but D92 found that the SAME
full-shape classifier, run within (family, gold) strata with its centring fitted
inside each CV fold rather than on the pooled data, decodes correctness at 61.0%
against a 50.2% null (p = 0.030, stable across CV seeds) — small, driven by 2 of 3
strata, not pre-registered, but converging with an independently designed test
(D91) that found the same qualitative thing at unrolls 6–24 specifically. `geometry-h0bank`,
launched before D91/D92 existed, was the powered, pre-registered test of D91's
specific lead — and it came back NOT CONFIRMED (D93): best window 6–18 gives
52.0% against a 46.8% null, p = 0.0125, missing the corrected 0.00625 by about 2x,
with no far-tail window significant either. This is a trustworthy null, not an
underpowered one — the test's own detection floor was found broken on first run
(mathematically incapable of registering any effect at the corrected alpha, a
40-permutation floor against a threshold that needs 400+), fixed, and re-run: it
now catches a planted half-SD effect with 100% certainty at this n, several times
smaller than D91's original signal. D91's specific transient-window mechanism is
therefore ruled out, not merely unconfirmed. D92's result is a different test —
between-item, stratified, no unroll sweep — and stands untouched by this: still
small, still suggestive, still not independently confirmed.)*

That is a claim with three legs, and each is measured rather than argued.

**(a) The path's shape is governed by how far the contraction has run.** Sliding a
fixed-width window along the unroll axis over all 152 banked trained orbits,
participation ratio falls 8.55 → 4.55 between unroll 0 and 28 — **down in 147 of
147 orbits** — while the consecutive-step cosine rises +0.276 → +0.710, up in
134/147. Each shift is about **3× the entire prompt-to-prompt spread** of the same
statistic. The contraction *rate* does not move over that range (p = 0.32), which
is the control that makes the other two specific rather than a generic drift
(D80). So the single largest determinant of any shape statistic is *where in the
convergence it was measured*. This also dissolves an apparent conflict in our own
record: D74's "13.3 effective dimensions" and the asymptotic "2.3" are the same
orbit at a ~21-unroll and a ~90-unroll window.

**And that depth law is something training BUILDS.** Extended to the untrained arm
— 84 orbits over **six independent weight draws**, on identical prompts, at the
depths every orbit in both arms reaches — participation ratio falls 8.553 → 4.889
trained (149/149 orbits) against 9.553 → 9.104 untrained: **the trained orbit sheds
directions 8× faster**. The step cosine moves +0.438 trained and −0.013 untrained
(p = 0.063, not significant); the untrained path does not straighten at all
(D80(7)). Random weights *diffuse* at roughly constant effective dimension; the
trained orbit *collapses* onto a dominant mode. Training installs the spectral
structure that makes the recurrent orbit converge, and the depth profile is the
clearest expression of it.

**(b) The dynamics are linear enough for that to mean what it sounds like.** Held-out
one-step prediction of the step directions recovers a median **91%** of what the
fitted subspace can express, with a general operator beating a subspace-matched
scalar rate in 148–152 of 152 orbits (D82). So D52's ρ and every Jacobian quantity
stand, and the orbit really is a contraction collapsing onto its dominant mode
rather than something a linear vocabulary merely approximates. Where the absolute
prediction score is low, the cause is a **gapless spectrum** — later steps enter
directions earlier steps never spanned — which is what D74(4) independently found
when DMD failed its stability gate on 80/80 orbits. Gapless is not nonlinear.

**(c) And the shape does not track the computation.** Across **21 task families
spanning 0% to 100% accuracy**, no geometric statistic tracks capability at any of
five windows — 0 of 18 usable cells — while 6 of 18 track prompt length on the same
orbits. *(The independent variable here carries a defect worth stating where the
claim is made, not only in the ledger: `correct` scores the FIRST TOKEN of the gold,
and 8 of these 21 families have multi-token golds, 4 of them for every item — so
`compare`'s 92% is first-digit accuracy. D89 re-ran the null against a
tokenisation-free axis built from decoded strings and **the null survived**. The
defect therefore weakens no conclusion here, but it does mean every capability
number quoted from this axis is a first-token number.)* The run measures its own reliability at 0.95–1.00, so a perfect relation
would have shown |ρ| in [0.80, 1.00] against an observed |ρ| ≤ 0.33: the null is
readable, not merely observed (D85). At matched answer value, correct and incorrect
trajectories do not differ in effective dimensionality, step cosine, contraction
rate or settling time — in the one family D79 could test, and now in two more from
the new bank (0 of 8 cells at α = 5.95e-04). H2 — "more reasoning steps, more turning" — returns nothing on the one
design that escapes the length confound: the paired `track − local` contrast gives
ρ = +0.543 at an exact p of 0.328 over 2048 arrangements (D83). With D28 (winding)
and D74(6) (dimensionality), **every instrument this project has built for H2 now
returns nothing**, and D80 explains why: they were all reading a clock.

**(c′) — CORRECTED 2026-08-09 by D84, and the correction sharpens the claim.** An
earlier draft of this section said the geometry is "the input it cannot see". That
is wrong. Handing a classifier the *whole* shape — the rotation-, translation- and
scale-invariant Gram matrix of unit step directions, a complete invariant of the
path — decodes the **task family at 100% balanced accuracy within every bank**
against permutation nulls of 25–33%. The shape identifies which prompt is being
processed, at ceiling. What the same classifier on the same orbits cannot decode is
**correctness: 55.3% balanced, p = 0.065** (D84). So the dissociation is not between
weights and input. It is between **input and outcome**: the path tells you what the
model is reading and not whether it got the answer right.

Two caveats travel with that, both measured. The family result **cannot be
attributed to the task**: no two families share a prompt length anywhere in the
banked data — 0 of 6 pairs in one bank, 0 of 3 in the other — so task identity and
sequence length are perfectly collinear, the same situation where D74(6)'s
`partial_spearman` refused at ρ = −1.000. And the correctness null is "not linearly
decodable at n = 128", not "absent".

**What training does, by contrast, is unmissable.** ρ rises 0.7048 → 0.8577 across
14 weight-sets with no overlap (D52); the step cosine *flips sign*, −0.379 →
+0.541, with completely disjoint distributions across five independent untrained
draws (D76, D76(8)); and rotation per unroll read off the operator falls from 1.892
to 1.024 (D83(4)).

**The most telling single number** is in D83(6): across `track`/`local` and n_ops
4–32, the *measured* step cosine spans just **0.44–0.57**, while the Jacobian's
leading oscillatory argument over the same prompts spans **0.37–1.10 rad**. The
trajectory's turning is more uniform than the local spectrum that supposedly
generates it.

### What this predicts, and what is still open

The reading is not merely compatible with the nulls — it *predicts* them, and it
predicts more that is not yet in. Two runs are in flight to test exactly that:

- `geometry-geomcap` — **landed, D85.** Both gates passed: capability correlates
  with D75's at ρ = +0.936, the fixed-seed replicates came back bit-identical
  (h₀ is the only stochastic input) and the unseeded ones all differed (the ceiling
  is measured, not manufactured). It also puts a number on the mechanism: **h₀
  accounts for 0.39–1.42 of the within-family variance** in these statistics, so
  what a single orbit's geometry shows is largely where it started.
- `geometry-depthacc` — **landed, D86.** Rank is capability only where the output is
  still short. Exact-match accuracy hits 0% by r=8 while CONTAINMENT rises 13.5% →
  33.3%, up in 9 of 10 families that moved (p = 0.0215): **depth does not destroy
  the answer, it wraps it in prose.** Being top-1 raises P(exact) from 3.7% to
  58.8% at r=4, and the link is gone by r=32. It also amends D69 — that row's
  0% → 83% was one trivial task; over 21 families the same instruction is worth
  +9.5 points at r=4 and ~0 elsewhere.
- `geometry-marker` — **landed, D88, and it refuted the reading I expected.** A-vs-C
  (same computation, one token apart) decodes at 98.0–100% against A-vs-B's
  96.0–100%. No gap: the shape reads the **token**, not the computation. D87 stands
  as measured but narrows to single-token input sensitivity — which makes the main
  dissociation cleaner, since the thing the shape tracks is now demonstrably the
  input and the experiment that could have shown otherwise came back negative.
- **A stale-CSV bug surfaced a real signal, D92.** `run_correctness_decode.py` was
  fixed (D88 commit) to stratify by (family, gold) and fit its centring inside each
  CV fold, closing the D72-shaped confound its first draft had — but the committed
  CSV was never re-generated, so it sat showing the pre-fix numbers until a machine
  switch's staleness check caught it. The corrected run: shape decodes correctness
  at **61.0% against a 50.2% null, p = 0.030**, stable 0.010–0.062 across six CV
  seeds, driven by `nth_item` (93.8%, p = 0.005) and `local_last` (74.6%, p = 0.020)
  with `parity8` flat. Position does WORSE (57.8%, n.s.) — evidence against pure
  residual family leakage, since both features are centred identically. Converges
  with D91 (below) from an unrelated design.
- `geometry-h0bank` — **landed, D93, and D91's lead did NOT replicate.** The
  powered pre-registered test (16 boundary prompts × 32 unseeded h₀ draws, 512
  orbits, 10/16 prompts splitting so the P1 gate passed) returns **NOT
  CONFIRMED**: best window (unrolls 6–18) gives 52.0% against a 46.8% null,
  p = 0.0125, missing the corrected α = 0.00625 by about 2×, with no far-tail
  window significant either. **The null is trustworthy because the instrument's
  own detection floor was found broken and fixed first** — it used 40
  permutations against α = 0.00625, whose minimum reachable p-value (1/41 ≈
  0.024) can never clear that threshold, so it reported 0% detection at every
  planted effect size including 1.0 sd. Re-run at 400 permutations it detects a
  planted 0.5 sd effect **100%** of the time. So D91's transient-window mechanism
  is *ruled out*, not merely unconfirmed. Its P6 timing control also shows
  `best_depth` alone decodes correctness about as well as the best shape window,
  so even the near-miss is confounded with *when* the answer peaks rather than
  with shape.
- `geometry-clrs` — **running.** Capability screen on CLRS-Text, which is IN
  Huginn's training mixture and has a clean integer difficulty knob — the
  parametric task ladder this project has never had (the battery's two ladders,
  `count4/8/16` and the Caesar family, are both floored or non-monotone, D89).

### H3 has a sharper, mechanistic form — and our own data already answers it

A sourced interpretability pass (2026-08-09) made the point that "is the spectral
radius < 1" is the *weak* form of H3. The strong form, standard in the RNN
fixed-point literature (Sussillo & Barak 2013; Maheswaranathan et al. 2019), is:

> **How many Jacobian eigenvalues sit near 1?** A *line attractor* — a manifold of
> near-unity eigendirections — is the mechanism by which a recurrent network
> integrates or holds a running count. A strict contraction has none, and then
> counting is impossible.

That is a sharp, falsifiable prediction, and **D31 already measured the quantity
it needs.** Implicitly-restarted Arnoldi on autodiff Jacobian-vector products
gives top eigenvalues in complex-conjugate pairs at **0.808, 0.808, 0.802, 0.802,
0.775, 0.775 …** across 3 prompts. Arnoldi returns the *largest* eigenvalues, so
if the top modulus is 0.808 then **nothing is near 1: there is no line attractor
and no integrator eigendirection.**

So **the state h carries no near-unity eigendirection** — no unbounded register
lives in h. **Three caveats travel with that, and the third is the one an internal
audit found missing (2026-08-09):**

1. Three prompts only.
2. The Jacobian is evaluated at particular points along particular orbits, so it
   is a local statement about the region those orbits visit.
3. **D31 measured the DIAGONAL BLOCK, not the operator that governs the orbit.**
   Its own row ends: *"NEXT: measure the FULL operator (all positions perturbed
   and read) rather than the diagonal block, which is the version whose spectrum
   should match the observed orbit decay."* The full operator has never been
   measured. So the "no line attractor" claim rests on the spectrum of a
   sub-operator our own ledger says is not the governing one — and by the same
   token, D94's comparison of the pair-based ρ against D31's 0.79–0.81 is not
   quite a like-for-like comparison of the same object.

**And our own data already contains a positive counterweight, which I had not
connected to H3.** D32 probed Barannikov's Task a directly: after removing the
linear dependence on position — the confound that makes a raw running-count probe
worthless, since position alone explains 98.5% of a running count — the latent
still predicts the count at **R² = +0.601 on residuals of sd 1.06 counts**, i.e.
it tracks deviations of about ±1 from what position predicts. D32's own words:
*"evidence that a running count is maintained, not merely that the model knows
where it is in the string."* Task b (nesting depth) is cleaner still at R² = +0.590
balanced / +0.718 unbalanced, where position explains only 0.160.

**Taken together the H3 picture is now the opposite of a clean confirmation:** the
scope argument says contraction need not forbid counting when the input is
re-injected; the architecture says Huginn re-injects; and D32 says a running count
*is* linearly decodable from the latents. The narrow surviving statement is only
that no unbounded register is carried in h across iterations — and even that rests
on a sub-operator's spectrum.

**But it does NOT follow that Huginn cannot count, and the earlier version of this
section said it did. Corrected 2026-08-09.** The Contraction Bottleneck Theorem's
proof is about **forgetting h₀**: iterated contraction destroys dependence on the
arbitrary starting point. It says nothing about dependence on an input that is
**re-injected at every step** — and a contraction toward a fixed point still lets
that fixed point *h\*(e)* depend arbitrarily on *e*. Huginn re-injects the prelude
output every unroll, by deliberate design, precisely to buy path-independence. So
the count need never survive *in h at all*; it can live in how *h\** depends on *e*,
which the contraction does not touch.

**What the correction rests on, in order of weight.** It is deliberately *not*
built on the toy model, which is far too unlike Huginn to carry it:

1. **A scope argument about the theorem, which needs no experiment.** The proof
   (Banach fixed point, `T_max ≤ log(D/ε)/log(1/c)`) is correct and is about h₀.
   §5's applied claim — "therefore N-state counting must fail under contraction" —
   requires the N states to be encoded *as different h₀ values*. Nothing in the
   proof establishes that. If the information instead enters as *e*, the fixed
   point *h\*(e)* may depend on it arbitrarily and contraction never touches it.
2. **A verified fact about Huginn's architecture**, read from
   `raven_modeling_minimal.py` at the pinned revision: the prelude output is
   re-injected through the adapter at **every** unroll, and `block_idx` reaches
   only the KV-cache slot. So Huginn is exactly the re-injection case, by
   deliberate design — Geiping et al. built it that way to buy path-independence.
3. **Independent corroboration of the framing** from a 2026-08-09 interpretability
   pass, which arrived at *e* = the map's **parameter** and *h* = its **state**
   from the DEQ literature rather than from this project's notes.

`h3_toy_model/` (imported 2026-07-22) is *suggestive only* and is cited last for a
reason. It reports probe R² ≥ 0.996 under forced contraction in a depth-recurrent
model against R² ≈ 0 in a time-recurrent RNN — but it has **no attention, a
mean-pooled context that discards token order, a binary vocabulary, forced
contraction via a loss term, and h₀ ≡ 0**. That last one matters most: with no h₀
variation, it cannot test h₀-forgetting at all, so it demonstrates the *mechanism*
(information arriving via re-injection is untouched by contraction) in a setting
where re-injection is the only channel. **It argues the theorem's scope; it is not
evidence about Huginn**, and its own README says as much.

**So the defensible position is narrow:** *no unbounded register is maintained in
the recurrent state h across iterations* (D31's spectrum, D94's rate). Whether
Huginn can count is a question about *h\*(e)*, which none of our contraction
measurements addresses. **And D90 is the reason this is not purely academic:**
correctness on a fixed prompt varies with h₀ alone, which a fully path-independent
contraction to a unique *h\*(e)* forbids — so at the depths we run, the transient
still carries h₀ information and the attractor has not taken over.

**And the bounded-counter caveat is ours to state, because nobody else has.** The
same pass found no published source drawing the distinction, so it is uncontested
only because unstated: contraction to a limit *set* of small but nonzero diameter
still permits a **bounded** counter. "Contraction ⇒ no running state" is sound
only for *unbounded* registers, and any claim we make must be scoped that way.

### The hypotheses and goals, as of the 2026-08-09 evening pass

Four results landed in one afternoon and they change the standing of H1, H3, G1
and G2. Stated plainly, with what each rests on:

- **H3 (contraction) — MEASURED, for the first time, with the estimator H3 is
  actually about (D94).** `contraction_from_pair` needs two orbits of the *same*
  prompt from *different* h₀; no bank in this project satisfied that at depth
  until `geometry-h0bank` produced 16 prompts × 32 unseeded draws as a by-product.
  Over ~960 pairs: **median ρ = 0.8550 raw (≈0.82 after D58's +0.033 bias correction for this fit), and 0.0% of pairs at ρ ≥ 1 across 16 prompts.** Per-prompt
  medians span 0.832–0.901 — a range that sits *inside* D52's independently
  measured [0.808, 0.920]. Two unrelated instruments agreeing on the same physical
  quantity is the strongest convergent evidence this project has. The residual gap
  between orbits is bounded away from zero everywhere (median 2e-2), so different
  initialisations reach *nearby but distinct* endpoints — which is what makes
  correctness h₀-dependent at all (D90).
- **G3 (spectral radius on Huginn) — substantially closed by the same result.**
  It was long flagged as never measured on the real model; the pair-based ρ is
  that quantity, measured dynamically rather than by autograd.
- **H1 (settle / loop / drift) — the empirical premise is now measured rather
  than assumed (D94 + D97).** The concern on record was that the argument
  "loop/drift cannot persist under a contraction" risks circularity by *assuming*
  the contraction. Both conjuncts are now measured: the autonomous-map fact from
  source (the iteration index reaches only the KV-cache slot, so there is no
  timestep conditioning), and ρ ≈ 0.855 from data. **G1's per-token census (D97)
  adds a second, independent instrument on a different object: 179 of 179 token
  positions across 12 prompts classify as `settle`. Zero loop, zero drift.**
- **G1 (each token's path at every depth) — DONE, and it is a negative control
  that passed (D97).** This was the largest structural blind spot: every
  instrument in this ledger reads *one* token position, and the live worry was
  that per-token convergence is a wide mixture, making every null a claim about
  an arbitrary phase of the clock D80 identified. It is not. Excluding the
  trivial BOS position (which converges at unroll 4 in all 12 prompts), the 167
  content positions converge with **median 34 and an interquartile range of just
  31–36**; only 4.8% converge before unroll 24. Positions are *synchronised*, and
  the answer position sits at a mean **30th percentile** of its own prompt's
  distribution rather than at an extreme. **Single-position reading was not
  misleading, which retroactively strengthens D79, D84, D85 and D93.** My own
  pre-registered prediction of a wide spread was wrong and is recorded as wrong;
  the cited contrary paper uses a different convergence criterion, so this is not
  a refutation of it.
- **G2 (query–key alignment) — measured at last, and INCONCLUSIVE (D96).** It
  was 0% done. The probe's self-check earned its place immediately: the first
  submission compared `core_block[-1]`'s reconstructed (q, k) against the
  *prelude* block's ground truth, failed at an error of ~15, and halted before
  producing a single number; after the fix it matches at exactly 0.0. The result
  then **rejects its own pre-registered prediction** — A-vs-B (different
  computation) separates 2.15× more than A-vs-C (same computation), significant
  at all four windows. But B is simply an *easier* task (median best_rank 7
  against 33.5 and 21), the rank-gap ratio is 2.88 against the QK-gap ratio 2.15,
  and |Δlog rank| predicts |ΔQK| at Spearman **+0.750**, p = 2.4e-05. So a third
  reading neither branch pre-registered — *QK tracks how well the computation is
  going, not which computation it is* — explains the same numbers, and this design
  cannot separate them. The control's own premise is also only partly met: in
  4 of 12 items C behaves closer to B than to A.
- **H2 — every GEOMETRIC instrument still returns nothing** (D28 winding, D74(6)
  dimensionality, D83 the Jacobian argument). The *behavioural* version has never
  been run and is now in flight: does the depth at which the answer first becomes
  available rise with problem difficulty? The pre-registered prediction is **no**.

**The binding constraint is no longer instrumentation — it is task supply.** Two
experiments have now died on it rather than on method: D47, and D95, whose gate
wanted 4 of 9 prompts to split on correctness and got 2 (the patching instrument
itself validated cleanly — a no-op replay reproduced the original rank curve
exactly). Tasks with clean difficulty knobs are synthetic and score ~0; tasks the
model genuinely does well (GSM8K 32.6%, ARC-E 69.9%) are natural-language with
uncontrolled difficulty and multi-token answers. `geometry-census` and the
difficulty×depth grid are the two runs aimed squarely at this, and it is the
subject of the first of three deep-research inquiries in `docs/`.

---

## 1-old. The 2026-08-08 synthesis, kept unedited

### The one-paragraph version (as of 2026-08-08)

The project asked whether latent-trajectory *geometry* in Huginn-3.5B encodes
reasoning depth. It does not, in the form originally proposed. The 2026-08-05
revision then held that **training changes the model's dynamics, not the content
of its state** — ρ rising 0.7048 → 0.8577 across 14 weight-sets with complete
separation (D52), while every content measurement was matched or beaten by random
weights (D40, D41, D48, D53). **That second half is now in serious doubt, and the
reason is that the readout was the confound.** Every content-null result was
measured on tasks scored 0% by greedy generation plus exact match. Scored instead
by rank of the gold token over the full vocabulary, the trained model holds the
answer *far* below chance on those same tasks — `count16` at median rank 22
against a chance of 32768 — and the trained-minus-untrained gap is **ordered by
capability**: +4.05, +3.43, +2.96, +2.81, +1.37 in log₁₀ rank from `echo_digit`
down to `rot13_word` (D68). Where the model can do the task, its weights carry
four orders of magnitude more answer-information than random ones. That is a
content difference, and a large one.

## 1b. What changed on 2026-08-08, and what it costs

Three findings, in the order they landed.

- **The instrument was wrong, not the model** (D68). Teacher-forced rank per
  unroll, from one forward per item — an instrument `eval_depth.py` already
  contained and that had been used for D35 and never for capability.
- **Depth makes a discourse choice** (D68, amended). Accuracy is non-monotone in
  `r`: `echo_digit` reaches 96% rank-1 at r=4 and 0% by r=8, while top-1 moves
  from the answer digit to a prose opener — `'The'` for 96–100% of items at r=64
  across every task. Every accuracy kernel in this project ran at `num_steps=32`,
  past that transition. **Amended the same day**: the "answer merely drops to rank
  2–3 behind *The*" reading holds only for the trivial task; on `count4`/`count16`/
  `add1` gold sits 10–25 places back, so genuine uncertainty rides on top of the
  discourse choice. It is a decomposition, not one mechanism.
- **The computation survives depth** (D69). With *"Reply with only the answer"*,
  `echo_digit` at r=64 goes 0% → **83%**, median rank 1, top-1 the actual digit.
  Same model, same depth, one instruction. Depth did not destroy the computation;
  the bare prompt's discourse convergence hid it.

The cost: D69's kernel **printed the opposite conclusion** and had to be rejected
on its own raw output, because the arm I nominated as decider was degenerate
(argmax = partial-UTF-8 byte fragments for 24/24 items on every task) and the
verdict logic never checked. That is D62 repeated in new code, and it is why the
`preflight` guards now exist.

## 1c. The strongest defensible result, and the strongest objection

**Best replicated:** D52 — ρ separates trained from untrained across 14 independent
weight-sets with no overlap (U=0, p=5e-04, d=13.2). Unchanged by any of this.

**Most consequential, least replicated:** D68's capability-ordered rank gap. It has
a working control — the untrained arm sits *at* chance (14668–46119), which a
broken decode could not produce alongside rank 3 in the other arm — but it is one
run, n=24 per task, one seed.

**The strongest objection I cannot answer:** rank is not capability. A low rank
means the answer is available to the readout, not that the model would emit it, and
the whole point of D68 is that those differ. Everything in §1 therefore rests on a
proxy whose relationship to behaviour is exactly what D69 began measuring and has
established for one trivial task only.

**The decisive missing experiment follows directly:** D40/D41/D48/D53 — the
content-null results — have *not* been redone with a graded readout. Until they
are, "content is architectural" is neither established nor refuted; it is measured
with an instrument now known to be blind in that regime. Note that `capcontent`,
the queued kernel for exactly this question, still scores capability by generation
and would repeat the error as written.

## 2. The architecture, verified against source

Facts below are read from `raven_modeling_minimal.py` at the pinned revision, not
from the paper's prose.

- `embed → prelude(2) → [core_block(4)]×r → coda(2) → ln_f → lm_head`, `n_embd=5280`.
- **The core block is weight-tied** and applied `r` times; `r` is sampled during
  training (log-normal-Poisson, mean 33) and chosen freely at inference.
- **Prompt embeddings are re-injected every unroll** via an adapter on the
  *concatenation* `[h, e]`. This makes the core an autonomous map `h ← F_e(h)`.
- **RMSNorm terminates every block**, so the state lives on a sphere.
  **Measured on our own banked data (D99):** `‖h‖ = 76.386`, with per-orbit
  relative sd **median 5.0e-05, max 1.6e-04 across all 512 h₀bank orbits**, and
  1.3e-05 to 8.9e-05 per token across unrolls on the full per-token grids. So the
  state is confined to S⁵²⁷⁹ of radius 76.386 to four significant figures. All
  dynamics are angular; there is no radial mode (independently consistent with
  D58(3), where the orbit difference is 100% tangential).
  **Two consequences that are not decoration.** (a) **Unbounded DRIFT is not an
  available asymptotic regime**, so H1's settle/loop/drift trichotomy is
  *exhaustive* rather than three guesses, and reduces to the sign of the top
  Lyapunov exponent. (b) It fixes the correct random baseline for any distance
  between states: two random points on that sphere are **108.03 ± 0.75** apart
  (= R·√2 exactly), which is what makes D98's cycle vertices at ~38 meaningful —
  0.35× random, i.e. far *closer* than chance.
- Init is Huginn's own, not an HF default: `std = √(2/5d)`, out-projections scaled
  by `1/√(2·d_eff)` with `d_eff = 132` — **depth-scaled for the unrolled depth**.
  So "random weights" here means Huginn at step 0, not a generic init.
- Training used **truncated backprop through the last k=8 unrolls**.
- The model ships a **chat template** (`<|begin_header|>`/`<|end_header|>`/
  `<|end_turn|>`, assistant role `Huginn`) and four `generate_*` methods.

## 3. What is established

**Tier 1 — measured across many weight-sets, survives controls.**

- **Training slows the contraction.** ρ 0.7048 → 0.8577, 5 random inits vs 9 trained
  checkpoints, Mann-Whitney U=0 (no overlap), p=5e-04, d=13.2 (D52). **But 91% of
  the shift is present at the earliest published checkpoint** (step 6144), and the
  within-training trend flips significance under a fit-quality filter — so it is a
  step change before anything saved, not a gradual property.
- **Content is architectural.** Untrained models match or beat trained ones on:
  the counting register (cv R² 0.7498 vs 0.7175 on identical prompts, D53), total
  decoding precision (0.046 vs 0.934 counts, D41), and effective dimensionality
  (PR 1.0 vs 4.8 — the untrained count representation is *literally
  one-dimensional*, PC1 correlating +0.999973 with the count, D48).
- **The trajectory rotates.** Jacobian leading eigenvalue complex in 3/3 prompts,
  27/30 top modes complex, period 2.6–6.0 unrolls (D55).

**Tier 2 — measured once, well-powered, not yet replicated.**

- **ρ is steerable.** Scaling `attn.proj` and `mlp.proj` by (1+ε): paired slope
  **+0.2853 ± 0.0450**, p=2.3e-04, 9/12 prompts monotone (D59). The pre-registered
  prediction `dρ/dε = ρ = 0.887` **failed** at 0.32× — so the branches carry only
  about a third of the contraction and the skip/RMSNorm structure carries the rest.
- **Prompt format is worth ~85 points** on a task the model can do (copy: 15% raw →
  100% chat, D60).
- **Failure is structured, not noisy.** Seven distinct retrieval modes; none is an
  attempted computation (D61).

**Tier 3 — real but scoped.**

- bfloat16 rounding makes the model appear to converge ~4.6× sooner than it does
  (D30).
- Required depth scales with difficulty (D35) — but this lives in the *answer*
  process, not in when the count becomes decodable (D54), and is structurally
  immune to the untrained control that deflated everything else, because random
  weights have no answer process to time.

## 4. What was retracted, and the pattern

Twelve claims retracted or superseded **by later work in this same project**. The
pattern is worth more than any individual retraction:

| what failed | why | row |
|---|---|---|
| winding as evidence | window-governed; sign flips with `num_steps` | D28 |
| the counting register's *direction* | one-token window misalignment; `v` is 97% the current-token contrast | D50 |
| the ρ→usable-depth bound | flipped under estimator and threshold choice | D43 |
| ρ constant across task families | ANOVA never run; family is significant | D45 |
| "decodability vs capability" | accuracy figures came from a different config; neither kernel measured accuracy | D41(3) |
| causal patching verdict | outcome had zero dynamic range before intervention | D47 |
| H1's settle/loop/drift | classes closer together than their own scatter, p=0.48 | D56 |

**Three of these were caught only by reading code or verbatim outputs, not
summary numbers.** Two were caught by a control that had been built and then
excluded from the analysis that depended on it (D47, D62).

## 5. Methodological state

**What works.** Tests that encode *retractions* executably — the only mechanism
here that survived context compaction with zero re-reading. Pre-registered
predictions inside kernel docstrings, which is why D59's failure was informative
rather than rationalised. `ledger.py` validation (found two real gaps first run).

**Known-broken or unresolved.**

- **The 0.058 estimator gap.** Two-orbit gives ρ=0.887, Arnoldi ~0.80. All three
  candidate explanations were tested and **refuted**: tangent projection changes
  nothing (radial fraction 0.0000), the envelope fit works on 3/12 real prompts and
  implies a period contradicting Arnoldi, and the orbits do converge (d_end/d0 =
  4e-05, so not a limit cycle). **Unexplained.** The paired design in D59 sidesteps
  it without resolving it.
- **The KV cache is unused and the reason is unresolved** (D62). `generate_minimal`
  returned empty output; both proposed mechanisms were refuted against the source.
  Sidestepped, not diagnosed.
- **Between-prompt sd of ρ is 0.056**, larger than the 0.03 effect we call
  detectable — so unpaired ρ comparisons are uninterpretable.

---

## 6. Critical re-analysis: blind spots

### 6.1 The deepest one — the untrained controls ran on tasks the model cannot do

Every "content is architectural" result (D40, D41, D48, D53) was measured on
counting-family tasks where the trained model scores **~0%** (D56, D60, D61). If
the model cannot perform the task, there is *nothing for training to have encoded*,
and finding that random weights represent the input equally well is close to
tautological — both models are carrying an unused input.

**This is the single largest threat to the project's headline claim.** "Training
changes dynamics, not content" may be an artefact of only ever testing content on
tasks with no content to change. The claim needs at least one task the trained
model demonstrably performs. Today the only such task in the entire project is
`copy` (100% under chat format, D60), which is trivial.

*Status: PARTLY ANSWERED, 2026-08-08, and the answer runs AGAINST the headline.
D68 measured the trained-minus-untrained gap with a graded readout across a
capability ladder and found it **ordered by capability**: +4.05 log₁₀ rank on
`echo_digit` down to +1.37 on `rot13_word`. So the content gap is real and large
where the model can do the task, and shrinks where it cannot — which is what 6.1
predicted would happen if the confound were real. The counting-family nulls
(D40, D41, D48, D53) sit at the far, capability-zero end of that ladder.*

*NOT closed, for two reasons. (a) D68 is one run, n=24 per task, one seed, and
rank is not capability. (b) The content-null results themselves have not been
REDONE with a graded readout — only new tasks have been measured with it. Until
D40/D41/D48/D53 are re-measured, "content is architectural" is neither
established nor refuted.*

*A WARNING ABOUT THE QUEUED KERNEL. `geometry-cap-content` was built to settle
this and still scores its capability axis by greedy generation plus exact match —
the instrument D68 shows is blind in exactly the regime that matters. Run as
written, its moderator variable would be near-zero across the ladder for
measurement reasons, and it would "confirm" the headline by construction. It
needs the rank readout wired into its capability axis before it is launched.*

### 6.2 ρ has never been connected to behaviour

D43 tried and was retracted. So ρ — the project's central quantity, the one thing
training changes, now steerable — has **no demonstrated behavioural consequence**.
D59 moves it; nothing measures what moving it does to anything the model outputs.
The ε-sweep should have carried an accuracy arm and did not.

### 6.3 Depth was never varied against accuracy

Audited 2026-08-05: every accuracy kernel fixed `num_steps=32`. On a recurrent-depth
model whose authors report GSM8K 9-10% at r=4 → 28-38% at r=32. `geometry-prompt-depth`
addresses this; it has not yet produced a result.

### 6.4 Huginn's own inference modes are untouched

`generate_with_adaptive_compute`: 0 kernels. `continuous_compute`: 0 kernels until
today. The latter is the architecture's distinctive feature.

### 6.5 Untested but assumed

- ρ is measured at the **final token position only**, on 12 prompts, one seed.
- The untrained arm samples **one init family**; whether ρ≈0.705 is a property of
  random weights or of *Huginn's chosen init scale* is untested (vary `std` ±2×).
- ~~Persistent homology: no null ever computed~~ — **CLOSED 2026-08-05 (D64).** It
  now has one, and it *passed*: real trajectories exceed all 40 manifold-matched
  surrogates in 9/9 cases with loops, 8.7 vs 1.9 mean count. Heavily qualified
  though — the excess is in the NUMBER of cycles, not their prominence (max
  persistence ratio 1.003), every cycle is at the arithmetic-noise scale
  (0.0000–0.0090 of the diameter), and the metric is still budget-governed: 0
  cycles at `num_steps=16` in all 6 runs, 7–26 at 64 in all 9, p=6e-04. **The first
  topological positive in the project, and it is a positive about counts of
  noise-scale cycles that only exist past a recording threshold.**
- `attn` vs `mlp` split is underpowered and its own gate says don't interpret (D63).

### 6.7 Every content measure in this project is LINEAR — found on the second pass

`cv_r2` is ridge. D41, D48, D50 and D53 are all linear decodability. That licenses
one reading of the headline and forbids another that **no measurement here can
currently distinguish**: training may encode the same content *nonlinearly*. If the
trained model represents the quantity in a curved or distributed way while random
weights preserve it linearly — which prompt re-injection would do — then a linear
probe favours the **untrained** arm for reasons unrelated to information content.

That is precisely the observed pattern (D41: untrained 19× more precise; D48:
untrained representation literally 1-D). So it is a live alternative explanation of
the headline, not a hypothetical.

*Status: NOT UNDER TEST — corrected 2026-08-09. The probe has never touched a
real state. `cv_r2_nonlinear` is inlined in exactly one bundle, `kaggle_capcontent`,
and that kernel has never been launched (§6.1 warns it would repeat the instrument
error). `geometry-nonlinear-content`, the one that did run, used the LINEAR probe
deliberately — its own docstring says "not to a nonlinear probe, which would only
re-open D68's 6.7 worry". So 6.7 is fully open, and it now bears directly on D73:
if a nonlinear probe lifts the untrained arm on `max_run`/`alt`, D73's
capability-ordered gap is a LINEARITY artifact rather than a content difference.
See directions.md B13, which is the experiment that settles it. Original status
text, retained because the probe design work it records is real:*

*UNDER TEST in the same kernel, via a nonlinear probe. Getting that probe
right took four designs — MLP(64) reached only 0.42 on a **clean linear** target,
RBF kernel ridge 0.02 with a positive null, PCA-24+poly2 0.36 — because a weak
nonlinear probe reports "no lift" and falsely confirms the headline it exists to
challenge. The accepted design (PCA-8 + degree-2 ridge) is asserted in tests to
recover a linear latent (0.73), beat the linear probe by 0.76 on a curved one, and
give a negative null. Two of the four rejections came from tuning on **isotropic**
synthetics before noticing the real states are low-rank (PR 1.0–4.8, D48).*

### 6.8 Not yet examined at all

- **No non-recurrent baseline.** Everything is Huginn. "Training changes dynamics,
  not content" may be true of any LM of this size — there is no architecture
  contrast, so nothing attributes it to recurrent depth.
- **No natural-language task.** Every task is a synthetic bit-string or word. The
  model's *reported* competence (GSM8K, ARC-C, HellaSwag) is on natural benchmarks,
  so "the model cannot do the task" may partly indict our task construction.

### 6.9 ρ is characterised entirely OUTSIDE the training regime — found on the third pass

Huginn was trained with `r` sampled log-normal-Poisson, **mean 33, median 29, mode
24**. Every ρ in this project is fitted over the pre-floor window of a 128-unroll
run, and that window is **65–98 points long — all 12/12 prompts beyond 2× the mean
training depth**.

So the project's central quantity, the one thing training demonstrably changes and
the only quantity it has managed to steer, describes the map's behaviour in a
regime the model was never trained to operate in. Whether ρ measured within r≤32
equals ρ fitted over r≤98 is **unknown and untested**, and it bears directly on
every claim built on it: D42, D44, D52, D59, and the retracted D43.

*Status: open. It is a rerun, not an analysis — see 6.10.*

### 6.10 The sweeps saved conclusions, not data

6.9 cannot be answered from what is on disk. `eps_sweep.json` stores the fitted ρ
and the number of points used, **not the per-unroll separation curves** — so
refitting on a restricted window requires re-running the GPU job. The same is true
of the checkpoint sweeps.

This is a scaffolding failure with a clear rule attached: **a kernel should persist
the curve it fitted, not just the fit.** The marginal cost is a few MB; the cost of
not doing it is a GPU rerun for any question the original analysis did not
anticipate — and this project has generated such questions at a rate of roughly one
per experiment.

### 6.6 What a reader should not conclude

- Not "Huginn cannot count" — only "does not, at these sizes, under these prompts."
- Not "geometry is irrelevant" — winding failed; the rotation it was trying to
  measure is real (D55).
- Not "training does nothing to representations" — see 6.1; the test was weak.
