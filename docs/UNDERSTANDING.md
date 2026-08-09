# State of understanding — 2026-08-09

Supersedes the 2026-08-08 revision, which predates D73–D83. Written to be read on
its own; every number cites the ledger row that carries its evidence. §1 is the
current synthesis; **§1-old below is the 2026-08-08 text, kept unedited** because it
is what the D68/D69 work concluded at the time and the difference is informative.

---

## 1. The synthesis, as of 2026-08-09

**One sentence: the geometry of Huginn's latent trajectory is set by the weights,
by how far the contraction has run, and by the input — down to a single token,
whether or not that token changes what the model computes — but not by whether the
computation succeeded.**

*(This sentence has been rewritten twice in one day, and the history is the
evidence. It began as "…not of the input or of the computation". D84 refuted the
first half — the shape identifies the input at ceiling — but could not separate task
from prompt length, since no two families in any bank share a token count. D87 then
settled that: at verified-identical token counts, with prompts differing in exactly
one marker character, the shape separates two computations at 96.9–100%. What
survives every revision is the second half: the outcome is what the path does not
carry.)*

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
orbits. The run measures its own reliability at 0.95–1.00, so a perfect relation
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

**The strongest objection I cannot yet answer.** Everything above is about
statistics of the *path*. It does not show that no geometric description could
encode the computation — only that the four this project can justify do not, at
the scales and windows measured, on this model. A representation living in a
direction none of these functionals is sensitive to would be invisible here, and
D79(5) says so explicitly.

**The second-strongest.** All of (a)–(c) is the trained arm at one token position on
synthetic tasks. D76 supplies the untrained contrast for the geometry but not for
the capability axis, and no natural-language reasoning benchmark has been run
end-to-end with this instrumentation.

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
- **RMSNorm terminates every block**, so the state lives on a sphere,
  `‖h‖ = 76.37 ± 0.007`. All dynamics are angular; there is no radial mode
  (confirmed empirically — the orbit difference is 100% tangential, D58(3)).
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
