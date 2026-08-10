# State of understanding — 2026-08-10

Rewritten, not appended (CLAUDE.md §6). §1 below is the current synthesis; the
2026-08-09 text it replaces is kept from §1-old onward because the difference is
itself evidence. Every figure cites the ledger row carrying its controls and limits.

---

## 1. The synthesis

**One sentence.** *The latent trajectory encodes the input — its shape and its endpoint
alike — while difficulty never reaches the dynamics, the dynamics never reach the
readout, and the readout is finished long before the dynamics are.*

### 1.1 What the latent carries

The trajectory's **shape** identifies the input at ceiling, separating two prompts one
token apart whether or not that token changes the computation (D84, D87, D88), and
carries at most a weak, unreplicated trace of whether the computation succeeded
(D92 positive, D93 not-confirmed under a floor that detects a planted 0.5 sd effect
100% of the time).

**The same is now true of the endpoint, which is the harder case.** D109 found the
fixed point h\* decoding the specific answer while the shape sits at chance — but in
that design the answer is a *function* of the input, so both readings predict it. A
paired-marker experiment separates them: sharing the required **answer** moves h\*
similarity by **+0.00016 (p = 0.70)**; sharing the **input** moves the same statistic
on the same orbits by **+0.0034**. The answer effect is **4.3%** of the input effect
(D125). The h\*-versus-shape dissociation stands; its interpretation narrows to
*h\* encodes the input, from which the answer is recoverable.*

And the topological signal behaves the same way: real, well-nulled, and outcome-blind
(D123 positive, D126 null at matched answer value).

### 1.2 Why H2 returns nothing — a chain, not a list

1. **Difficulty does not reach the dynamics.** The contraction rate spans 0.824–0.918
   across 21 families (between/within 4.2×, surviving length matching at 7.5×), yet
   quadrupling the count in `count4→count8→count16` moves it **0.0033** (D115). The
   rotation angle behaves identically, independently of the modulus, with an untrained
   control (D116). Both are **rates, not durations**, so D80's clock-reading critique —
   which explains away every earlier H2 instrument — cannot apply.
2. **The dynamics do not reach the readout.** ρ(contraction rate, best_depth) =
   **+0.118, p = 0.65**, against a floor detecting effects three times smaller than the
   variation present (D114).
3. **A causally manipulated regime change produces no behavioural difference.** In
   D132's paired design — same sequence, same marker, same required answer, only the
   instruction's noun differing — the rotating arm and its settling counterparts differ
   in neither the depth at which the answer is best available (16.0 vs 14.5, p = 0.156)
   nor the best rank reached (6.5 vs 7.0, p = 0.880). This is the strongest of the three
   because it manipulates rather than correlates.
4. **Because the readout never waits.** On solved orbits the answer is top-ranked at
   **median unroll 4**, with **87% of the state's journey still ahead**; the state
   reaches a 1% residual only at unroll 32–36 (D112, replicated across banks; threshold
   dependence measured in D97's correction).

Every H2 instrument this project built measured the trajectory. The answer was already
decided in the first few unrolls.

### 1.3 The map itself is well characterised, and it is a task-level object

Four instruments of three kinds put the contraction rate in **0.79–0.87**: causal
state-injection **0.8335 ± 0.028** over 16 recipients (D113), full-operator Arnoldi
**0.8098** (D119), diagonal-block Arnoldi 0.79–0.81 (D31), passive orbit decay ≈0.82
bias-corrected (D94). Training moves it from **0.7042 → 0.8582** with *total*
separation, and is done by the earliest public checkpoint (D52, rediscovered and
retracted as D120).

**And a single word selects the qualitative regime.** *"largest **symbol**"* rotates
36/36; *"largest **element**"* and *"largest **item**"* rotate 0/36 — same task, same
sequences, **identical 53-token count**, flipping in **36 of 36** within-sequence cells
(D132). Symbol diversity is refuted as the cause (D129). This has a mechanism already
in the ledger: `e` is the map's **parameter** (D111, D113), and the spectrum at h\*(e)
is task-dependent (D115, D116, D119) — D132 localises that dependence to one token.

**And the three nouns request the same computation and the same answer.** `symbol`,
`element` and `item` all mean *take the maximum of this sequence*, and the banked golds
are identical across all three in **36 of 36** cells. Computation fixed, answer fixed,
token count fixed, sequences shared — **and the dynamics change completely.** This is
D88 one level down: D88 found the trajectory's *shape* reads the input token rather than
the computation; D132 finds the same of the *dynamics*. **The latent geometry tracks the
surface form of the instruction, not what the instruction asks for** — which is the
sharpest statement of §1.1 available, and it is causal rather than correlational.

**What selects the regime is still unknown, and one round of guessing has been spent.**
Meaning, surface form and subword structure are all refuted (D134). D134 then inferred
*by elimination* that the property must live in the token's learned embedding; A23 tested
that survivor on 51 nouns and it **fails** — leave-one-out from the embedding is 0.551
against a permuted null of 0.530, below the 0.592 majority baseline, and the separating
direction is not low-dimensional (D135). Six classifiers, linear and nonlinear, then give
a **best-of-six of 0.551 against a null-of-max of 0.589, p = 0.808** (D136). So the
property is a function of the token that no simple rule reads off its embedding at n = 49.

**The regime is genuinely bimodal, and D132's determinism survives at larger scale.** A
continuous rotation statistic separates the two populations (median **0.896** vs
**0.336**, p = 0.0002) and the distribution is **gapped** (p = 0.0002) — two regimes, not
a continuum with an arbitrary cut. Of the two nouns that failed D135's determinism gate,
`word` is a **detector false positive** (its "rotating" orbit carries the *lowest*
period-6 power of its four) and only `set` is genuinely intermediate, sitting in the gap
with all four orbits between the modes. **50 of 51 nouns are deterministic**, with no
marker or sequence confound (D137) — which retires D135's own reading that the regime
depends on the token *and* the sequence.

**And a single threshold on that statistic reproduces the regime label across three
banks.** R > 0.6677 agrees with the independently computed period-6 label on **384/384**
(`nounsweep`, 16 nouns × 12 sequences), **108/108** (`wordswap`), and **199/204**
(`embsep`) — **691 of 696 orbits**, across banks with different nouns, sequence counts
and templates (D141). *An earlier version of this paragraph claimed a distributional
**gap** instead; that test used a uniform null the data beats for reasons unrelated to
the hypothesis, and the gap it found sits inside the settling population in two of the
three banks. It is retracted and replaced by the threshold result, which is what the
claim needed in the first place.*

**The regime is causally controlled by `e`, and the boundary is a surface rather than a
split.** Interpolating `e` along straight lines between nouns, all eight cross-regime
paths cross the threshold **exactly once**, every crossing inside a single 0.05 grid
step — sharper than the design resolves — and at t = 1 the orbit matches the rotating
noun's own R to within the h₀ noise even though h₀ came from the settling prompt, which
confirms D111/D113's parameter claim causally. But the rotating set is **not convex**: a
rotating→rotating path never leaves the regime (0 crossings in 21 points), while a
**settling→settling** path crosses **twice**, rising to R = 0.74 before falling to 0.05.
t\* varies 0.28–0.74 by pair (D140). **That non-convexity also explains D135/D136** —
a linear rule or a nearest-neighbour vote on 49 points cannot represent such a region.

The live candidate is now `e` itself, the prelude's nonlinear image of the embedding,
which is the map's parameter (D111, D113) and is where a threshold would sit. A24 tests
it causally rather than by fitting — interpolating `e` from a settling noun's to a
rotating noun's and locating the flip — because at 51 nouns power, not the hypothesis, is
what binds.

### 1.4 What the corrections did to the older record

An independent audit on 2026-08-10 found that **three older positives all rest on one
instrument**: `classify_shape`, whose loop branch D25(3) proved fires on straight lines,
random walks and decaying spirals alike. B4 ("the project's cleanest positive result" —
loops under a starved budget) is retracted: its settle test is the last step over the
*largest* step, which truncation inflates by construction. D14's claim to validate the
pipeline against Blayney's rate is withdrawn — a detector with an unknown false-positive
rate cannot be validated by matching a published number. D100's "working difficulty
ladder" is withdrawn (3 forwards per level; only 0/33/67/100 observable).

Two more dissolved rather than resolved: D97 vs D30 is a **threshold**, not a
disagreement (11→47 unrolls for 30%→0.1%); D100 vs D101 is two underpowered estimates.
And D55's near-aliasing diagnosis is withdrawn — its samples-per-turn came from the
diagonal-block operator D31(3) calls "the wrong operator"; the trajectory is sampled
3–9× above Nyquist, not near it. **D55's third prompt, at 60.1°, was already reading the
mode D112/D132 later characterised.**

### 1.5 The strongest objection to all of it

Capability. The census yields **31 usable items across 5 families** (D130), none of them
natural-language reasoning; multi-term addition is at 0% (D118); binary-search depth runs
*opposite* to Huginn's difficulty (D128); CLRS was never tested in the form its
deep-research pass specified (D124). So every null here is a null *on synthetic,
largely-OOD tasks*, and the D124 ambiguity — genuine OOD difficulty versus scoring
artefact — is unresolved. The one experiment that would most sharpen the picture,
D125's answer-versus-input test, is limited by exactly this: at 4.7% accuracy on one
marker it cannot separate "no answer encoded" from "no answer computed."

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
