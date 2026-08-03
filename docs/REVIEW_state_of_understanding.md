# Review: state of understanding

STATUS: written 2026-08-04 as a bird's-eye critical review, not project work.
Grounded in the actual record (39 ledger entries, 205 tests, 9 commits, 8 GPU
kernels), not in memory. Pass 1 of an intended cycle: write, attack, research
what the attack finds, re-attack.

PURPOSE: establish what I actually understand about (a) this project, (b) my own
research practice, and (c) the field this sits in — and find where that
understanding is thin.

---

## Executive summary — the state of understanding after four passes

**On the project.** The three original operationalisations were each wrong in a
way an audit was needed to see: H1's classifier had one reachable output, H2's
winding tracked the recording budget, H3's premise was true but its applied
claim overreached. What survives is a set of decoding results — a counting
register, a readout curve, a ~3-of-10-dimensional count subspace — plus one
finding independent of the framing: **bfloat16 makes the model appear to
converge 4.6× sooner than it does.**

**On my own reliability.** Nine retractions, six of them mine, none an
arithmetic slip. Every one was structural: the number was computed correctly and
meant something other than what I said. The taxonomy (II.2) has six classes, and
Part XI turns each into a checklist item that would have caught it before
reporting.

**On the field.** The project's second- and third-most-cited sources are
unlocatable and misattributed. It engages none of the probing-methodology
literature despite every positive result being a probe, and it re-derived the
tuned lens' motivating observation as if new. Honest novelty, finding by
finding, is in III.3 and is lower than I had been treating it — including for
the "law" I was one step from calling a spotlight result (III.4).

**On method choice.** Part X is the artifact that was missing: thirteen
interpretability methods, of which this project used essentially one family
(linear decoding) plus two it used without naming. Four fifths of the effort
went to refining a measurement family already shown to fail. **The correction is
to enumerate the method space when a measurement fails, rather than refine the
failing method.**

**What is running.** Two controls chosen precisely because they are the most
likely to overturn the positive results: an untrained-architecture baseline, and
the causal patching test that decides whether the register is *used* or only
decodable. Neither had been attempted in six weeks of work.

**Confidence trajectory.** Each pass has reduced it. Pass 1 downgraded the law;
pass 2 killed the monotone-accumulator reading; pass 3 retracted the bandwidth
cutoff; pass 4 caught a multiplicity artifact I was about to report as
reassurance, and found a bug I had already fixed once. That is the procedure
working as intended, and it is the honest description of where things stand.

---

## POST-REVIEW UPDATE (2026-08-04): the untrained baseline lands, and it invalidates part of this review

The control this review identified as blind spot O1 was run while the review was
being written. It is the most deflationary result in the project.

```
                          trained    untrained (RANDOM WEIGHTS)
  v_pairwise_cos          +0.7996    +0.9991
  register_r              +0.2822    +0.2608
  register_sign_frac      +1.0000    +1.0000
  readout_r2_count        +0.9773    +1.0000
  positional_k1_resid_r2  -0.2325    +0.5272
```

**A network that has never been trained reproduces every positive finding, most
of them more strongly.** It decodes the total count from its answer token at
R² = 1.0000 and produces a *more* consistent "register direction" than the
trained model.

The mechanism is a random-features property: the count is a linear function of
the input tokens, and random mixing plus a residual stream preserves linear
functions in decodable form. So D34, D36, D39 and the *endpoint* of D38 describe
a normalised residual stream, not a trained counting mechanism.

**This directly overturns two things in the review below.** Part VI's
evidence-versus-belief table rated "count linearly available at answer token" as
*strong evidence, high belief* — it is strong evidence for an architectural
fact, not for anything about Huginn. And Part V's skeptic objection #1 ("attention
computes sums, this is expected") is not merely *fair*, it is **correct**, and I
graded it too gently.

**What the control does not touch, stated so the retraction is not over-broad.**
It measured the endpoint at r=32 only. D37 and D38 are claims about the *depth
trajectory*, and that comparison was not made; a follow-up is running. Also
untouched: the bf16 result (a precision claim), the Jacobian spectrum (a
measurement), the depth-scaling result (a claim about the model's own logit
margin, which an untrained model has no version of), and every negative result.

### The constructive reading, which is stronger than what it destroys

D40 kills the register story, but it hands the project a cleaner result in
exchange, and one that is *about* the field rather than about Huginn.

```
                       decodability of the count    ability to do the task
  UNTRAINED            R^2 = 1.0000                 none (random logits)
  TRAINED              R^2 = 0.9773                 ~10% accuracy
```

**An untrained network decodes the count perfectly and cannot count at all.**
That is a crisp existence proof, on a real architecture and a real task, of the
central critique of the probing paradigm: *linear decodability of a task
variable is nearly free architecturally and says almost nothing about whether
the model can perform the task.*

Most statements of that critique are theoretical, or rest on synthetic control
tasks with random labels (Hewitt & Liang's design). This is a naturally-occurring
instance with a *meaningful* target: the quantity is exactly the one the task
asks for, the probe is honest, the controls (signed sums, position, token
identity) all pass — and the representation is still worth nothing behaviourally.

It also sharpens the project's own earlier observation. The 0.993-decodable /
10%-accurate gap was noted as curious; the baseline shows the decodable side of
that gap is architectural, so **the entire content of "training" here is on the
readout side, not the representation side.** That is a specific, testable claim
about what recurrent-depth training does, and it is more interesting than the
register was.

**The practice lesson supersedes everything else in Part XI.** This control
costs one GPU hour, was available from day one, and was run in week six after
twelve refinements of a geometric metric and four probing experiments. An
untrained baseline should be the first control for any representational claim.
Its absence invalidated more work than every other omission in this review
combined — including all six of my retractions put together.

---

## Part I — What the project is, and its honest scientific standing

### I.1 The question and its three operationalisations

Huginn-3.5B is a recurrent-depth transformer: context re-injection makes the
core, for a fixed prompt, an autonomous map `h_{t+1} = F_e(h_t)`. The project
asked whether the geometry of that iteration encodes reasoning depth, via:

| | hypothesis | operationalisation | verdict |
|---|---|---|---|
| H1 | trajectories fall into shapes | settle/loop/drift classifier | **never tested** — the classifier has one reachable output (140/140 "settle") |
| H2 | winding ∝ required depth | winding number at the answer token | **false as operationalised**; the metric tracks the recording window |
| H3 | contraction forbids counting | ρ < 1 ⟹ no unbounded register | **premise true** (ρ ≈ 0.80–0.90), applied claim needed rescoping to *unbounded* |

Every operationalisation was wrong in a way that took an audit to see. That is
the project's central fact and, I now think, its most transferable output.

### I.2 What actually survived

**Positives.** Required depth scales with difficulty at *fixed prompt length*
(rho = +0.225, p = 1.3e-3, n = 200, all prompts exactly 77 tokens). A counting
register exists, is linearly decodable, and passes a ℤ-action test. Readout into
the answer token improves R² 0.65 → 0.993 over ~24 unrolls while the register
itself is flat. The count occupies a ~3-dimensional subspace of the answer-token
state.

**The one finding independent of the project's framing.** bfloat16 rounding
makes the model appear to converge ~4.6× sooner than it does (regime ends at
20.7 in bf16, 96.3 in fp32). This is a claim about the published architecture.

**Negatives with mechanism.** Winding's comparison to any null is governed by
`num_steps`; four metrics failed identically by averaging signal and arithmetic
noise; 16 of 18 historical results files have no raw data and cannot be
corrected.

### I.3 Honest standing

This is a good two-week output for a school project and it is **not** a
spotlight paper. Section III argues that finding by finding. The most
publishable single fact is the bf16 result, and it is currently a
one-architecture observation.

---

## Part II — Epistemic audit of my own work

### II.1 The retraction record

Nine retractions are on the books; **six are mine**, made and corrected inside
this work:

| | claim | what was wrong | what caught it |
|---|---|---|---|
| R6 | noise floor matches bf16 to 1.36× | wrong formula — isotropic noise adds in **quadrature**, not linearly | an external survey; then I verified by simulation rather than accepting |
| R7 | the winding null "reverses" | underpowered (47×40, unstratified) | my own follow-up on a different stratum gave the opposite sign |
| R8 | a ~14-step oscillatory mode exists | DMD fitted to isotropic rounding noise | the isotropy diagnostic (participation ratio 18–19/20) |
| — | ρ = 0.871 (contraction) | distance to a trajectory's *own* endpoint — near-tautological | asking what operator the number describes |
| — | ρ ≈ 1.000 (no contraction) | measured across *different prompts* | the prompt-collision test |
| — | symmetry ratio 0.9688 | an algebraic identity equal to `n_open/n_close` | sd = 1.9e-16 across 16 random strings |
| — | +4.17° answer clustering | driven by prompt collisions (identical prompts → 0.00°) | inspecting which pairs drove it |
| — | "no evidence" of depth scaling | correlated 6 level-means, one with n=1, instead of 23 instances | the project's own critical-value table |
| — | register R² = 0.892 | decodes *position*, which explains 98.5% of a running count | asking what else predicts the target |

### II.2 The error taxonomy — and it is not random

Sorting by *kind* rather than by claim:

- **Wrong window** (4×): a statistic averaged over signal + noise. R2, R3, R8, and the whole `dlr`/`conv_rate` family.
- **Wrong operator/object** (3×): measuring a quantity that is not the one the claim is about. ρ-to-own-endpoint; ρ-across-prompts; `J_TT` vs the full coupled operator.
- **Underpowered or mis-powered** (2×): R7; the level-means depth test.
- **Confound not checked** (3×): position (register R²), prompt collisions (clustering), token identity (would have been, if not caught).
- **Vacuous statistic** (2×): the symmetry identity; `classify_shape` with one reachable output.
- **Leakage** (1×): whole-dataset residualisation making held-out points determined.

**Not one was an arithmetic slip.** Every error was structural: the number was
computed correctly and meant something other than what I said. That is worth
naming precisely, because "be more careful with the maths" is the wrong lesson
and "ask what this number is a property of" is the right one.

### II.3 The generating mechanism

The cheap version of any measurement is available immediately; the expensive
version is not; and the cheap version is *usually* directionally right, so the
habit of trusting it is reinforced far more often than punished. I ran an audit
whose entire subject was that habit, and reproduced it six times while running
it.

The practical implication is an ordering rule, not an exhortation: **run the
confound check before reporting the number, not after.** Every one of my
retractions was caught by a check I was capable of running first.

### II.4 What actually caught the errors

- **External challenge**: 3 (the reversing-rotation question, the step-size
  question, the tokenizer question — each redirected a whole line of work).
- **A hook or test refusing my claim**: 2 (both times I had asserted a blocker
  without verifying it).
- **My own follow-up check**: 4.
- **A diagnostic I built for another purpose**: 2 (isotropy; prompt collision).

The asymmetry matters: the highest-yield interventions were short questions from
someone not invested in the answer. That is a structural advantage I cannot
fully replicate from the inside, and the closest substitute is a pre-committed
checklist run before reporting.

### II.5 What I did well, stated so it is repeatable

- **Calibration arms.** Feeding a surrogate back in as data is what made D28
  citable; without it the real arm is uninterpretable. No previous null in this
  project had one.
- **Retracting fast and in writing.** Six retractions with diagnoses beat six
  quiet fixes, because the diagnoses are what stopped repeats.
- **Pre-registering before the run.** The depth-threshold kernel's criteria were
  in the docstring before it ran, which is why the level-means correction was a
  power fix rather than a fishing expedition.
- **Encoding a retracted test AS a test.** `test_ratio_symmetry_is_an_algebraic_identity`
  makes that specific error unreintroducible.
- **Writing the not-done list.** It made the unexplored surface visible, which is
  invisible by default in any results write-up.

---

## Part III — The field, and an honest novelty assessment

### III.1 Literature grounding: measured, not assumed

Citation counts across the project's docs and code:

```
Barannikov 42   Movahedi 29   Pappone 28   Geiping 27   Tu 17   Lu 12
Miyato 7   Merrill 6   Grazzi 6   Perea 4   Harer 4   Elsayed 4   Cunningham 4
Trefethen 1   Karcher 1   Jung 1   Hemati 1   Gavish 1   Fasy 1   Churchland 1
```

**The second and third most-cited non-curator sources are broken.**
"Movahedi et al." (29 mentions) could not be located at all. "Pappone et al."
(28 mentions) is real but studies a GPT-2-scale model rather than Huginn,
defines the drift-to-loop ratio differently from the implementation, and reports
a consecutive-step cosine of **+0.5 to +0.65** — the opposite sign to the −0.276
attributed to it, and in fact agreeing with this project's own corrected
converging-regime value of +0.084.

That is a serious hygiene failure, and it is upstream of a real scientific
error: R2's "the path zig-zags inward" reading was justified by a citation that
says the reverse.

### III.2 The blind spot that matters most

```
tuned lens      0 docs        control task    0 docs
MDL probe       0 docs        Hewitt          0 docs
selectivity     1 doc         Belrose         0 docs
```

**The project's strongest positive results are all probing results, and it
engages none of the probing-methodology literature.** Specifically:

- **Hewitt & Liang (2019), control tasks and selectivity.** A probe that decodes
  a property may simply be an expressive probe. The standard remedy is a control
  task — the same probe on randomised labels of matched structure — and reporting
  *selectivity*, the gap. I ran permutation nulls, which is related but weaker:
  a permutation null tests whether the mapping is real, a control task tests
  whether the *probe* is doing the work. **No control task was ever run here.**
- **Voita & Titov (2020), MDL probes.** Accuracy conflates information with
  accessibility; description length is the better measure. Every R² in this
  project has that conflation.
- **Belrose et al. (2023), tuned lens; nostalgebraist's logit lens.** The tuned
  lens exists *precisely because* a fixed decoder is biased at intermediate
  depths and must be re-fit per layer. That is extremely close to D37's finding
  that the increment direction rotates and must be refit per unroll.

This is the review's most important discovery about my own work: **I derived, as
if new, a phenomenon the probing literature has institutionalised as a tool.**

### III.3 Novelty, finding by finding — sceptically

| finding | closest prior art | honest novelty |
|---|---|---|
| bf16 masks 4.6× of computation | DEQ solver-tolerance discussions (Bai et al.); numerical shadowing (Hammel–Yorke–Grebogi) | **Moderate–high.** I have not seen "low precision makes an iterate *look* converged, by a measurable factor" stated. Generalises to any iterative model. Best candidate here. |
| register found by mean-difference; ℤ-action | difference-of-means probing is the classical linear discriminant; counting features are well-trodden | **Low** as method. The matched-pair ℤ-action test is a nice touch, not a contribution. |
| register/readout dissociation (D37/D38) | tuned lens: intermediate representations are not in the final basis | **Moderate.** It is tuned-lens along the *recurrence* axis rather than the layer axis, plus a rate law. A delta, not a discovery. |
| observable-convergence law | one-line Taylor expansion; the tuned lens already implies the qualitative claim | **Low.** See III.4. |
| winding tracks the recording window | Elsayed & Cunningham (2017), rotation as an epiphenomenon under an inadequate null | **Low as a result** (it is a null), **moderate as a methods case study** on a new architecture. |
| sphere forbids a translation register | normalisation geometry is standard; my "theorem" needed an isometry assumption the real update does not satisfy | **Low.** The assumption-free version (bounded linear readout on a sphere) is trivial. |

### III.4 On the "law" I was about to call spotlight-level

I was heading toward presenting the observable-convergence law as the A\*
result. Reviewing it cold, that was wrong, and the way it was wrong is
instructive.

**What is true:** `1 − |cos(v_r, v_∞)| ∝ ρ^{2r}` fits with R² = 0.997; the
implied ρ = 0.861–0.868 lands in the independently measured 0.85–0.90; synthetic
validation recovers a known ρ to 1.3%; and a second, unrelated observable (the
readout curve) gives 0.905–0.910.

**Why it is not a spotlight result:**

1. The derivation is a first-order Taylor expansion. Any numerical analyst would
   call the statement obvious once posed.
2. The qualitative consequence — that a fixed decoder is biased at intermediate
   depth — is exactly why the tuned lens exists. I did not know that when I
   derived it, which is a literature failure, not a discovery.
3. The empirical confirmation is on **one model**, and the "independent"
   estimates all come from the same 200-odd prompts on the same checkpoint.
4. Its sharpest form — *rising decodability is not evidence of computation over
   depth* — is a real methodological warning, but it is close to folklore in the
   probing community and I cannot claim it without engaging III.2's literature.

**The salvageable contribution** is narrow and should be stated as such: a
*quantitative* rate law connecting probe-curve convergence to the model's
contraction rate, giving a cheap estimator for ρ that needs no Jacobian — plus
the observation that ρ estimated four ways from observables agrees with ρ
measured directly. That is a workshop-paper-sized contribution, honestly
described.

Catching this is, I think, the most valuable single output of this review: I was
one step from over-claiming in exactly the way I spent the whole project
correcting others for.

---

## Part IV — Practice review

### IV.1 Research practice

**Worked.**
- *Check the premise before acting on it.* Three assertions I made or inherited
  were false: "no GPU access" (Kaggle was configured, and had been used ten
  times), "raw paths were never saved" (140 were on disk), "convergence.csv does
  not reproduce" (my tolerance was too tight for 4-decimal storage).
- *Do the intuitive version first.* The best result in the project — the T₁
  direction at 66.8σ — is a mean difference and a cosine. It is stronger
  evidence than the ridge probe it replaced *because* it has no fitted
  parameters. I had been treating "simple first" as pedagogy; it is
  epistemology.
- *Stratify.* Pooling produced spurious results three times (winding reversal,
  cross-task agreement, depth level-means).

**Did not work.**
- *Summary-first data handling.* Four GPU kernels returned printed numbers
  before one returned raw states. Every new question then cost new GPU time and
  left conclusions whose underlying data did not exist locally — the exact
  failure the audit was documenting. The states dump should have been kernel #1.
- *Analysis in throwaway snippets.* The strongest results lived only in terminal
  history until I scripted them retroactively. Same failure, different medium.
- *No global multiplicity accounting.* An early FDR pass covered 46 tests; the
  ~15 tests added later were never folded in. With effect sizes this modest that
  is not a formality.

### IV.2 Code and script practice

**Good.** Provenance sidecars with sha256 and separate `task_seed`/`init_seed`;
205 tests including regression tests that pin real-data findings; a
consistency test that fails if prose and CSV drift apart; encoding a retracted
error as a test; additive staleness banners rather than rewriting teammates'
prose.

**Weak.** The seven kernels duplicate ~60% of their code (task generators,
extraction hooks, metric functions) because each had to be self-contained; a
shared payload uploaded once would have been cleaner and would have removed the
copy-drift risk. No CI. No environment lock recorded per result. Kernels do not
seed `h_0`, so re-running reproduces prompts but not exact states — which is why
55 MB of arrays had to be committed.

### IV.3 Where the good ideas actually came from

Of the ideas that moved the project: **three came from short external questions**
("what if it rotates one way then the other", "you missed the step size",
"aren't there tokens for both '2' and ' 2'"), **two from a systematic sweep**
(the audit's file inventory; the generator census), **two from taking a
constraint seriously** (RMSNorm ⟹ sphere ⟹ rotation not translation), and
**one from a hook refusing my claim** (checking whether GPU was actually
unavailable).

Almost none came from unprompted generation. The productive pattern was:
someone points at an anomaly, I take it literally and chase it to a mechanism.

---

## Part V — Blind spots this pass identifies

Ranked by how much they threaten current conclusions.

1. **No control task or selectivity on any probe** (Hewitt & Liang). Every
   positive result is a probe; none has the standard control. *Threat: high.*
2. **No untrained/architecture baseline.** Nothing establishes that any finding
   is Huginn-specific rather than generic to a normalised residual stream.
   *Threat: high.*
3. **The register direction's identity is unexamined.** `v = mean(Δ|1) −
   mean(Δ|0)` is never compared to the token-embedding difference for `1` vs
   `0`. The *correlation* was controlled for token identity; the *direction* was
   not. *Threat: high, and cheap to check.*
4. **Task b target mismatch.** The model is asked for **max** depth; the probe
   decodes running depth `d_i`. Those are different functions. *Threat: moderate.*
5. **No global multiplicity correction** across the ~15 newer tests. *Threat:
   moderate*, given rho ≈ 0.23 effects.
6. **MDL/description-length never considered**; every claim uses R² or
   correlation. *Threat: moderate.*
7. **Single checkpoint, single model, single revision.** *Threat: moderate.*
8. **The b_bal last-third puzzle** is open after three refuted explanations.
   *Threat: low, but unexplained.*
9. **`h_0` unseeded in the newer kernels.** *Threat: low, reproducibility only.*

---

## Part VI — What a strong version of this project looks like

Not "more experiments" — a different shape:

1. **Lead with the bf16 result**, and make it architectural rather than
   anecdotal: three iterative models × three precisions, showing that apparent
   convergence depth scales with mantissa bits. That is a paper.
2. **Frame the metric failures as a methods contribution**, Elsayed–Cunningham
   style: four metrics, one shared defect, a diagnosis, and a fix validated by
   showing the corrected metric recovers a *planted* signal.
3. **Do the probing properly**: control tasks, selectivity, MDL, and a tuned-lens
   comparison — which would also situate D37 correctly instead of
   re-deriving it.
4. **Keep the negatives**, which are unusually well-mechanised, and stop trying
   to promote them into positives.

---

---

## Pass 2 — researching the blind spots, and what it changed

### VII.1 Blind spot 1 (no control task): resolved, and it strengthens the result

The serious alternative to "the model counts" is "the model RETAINS the 64 bits
and the probe does the arithmetic." Never tested. Testing it — decoding
differently-weighted sums of the *same* bits from the answer-token state:

```
the COUNT           sum(b_i)          R2 = +0.9928
RANDOM SIGNED sum   sum(eps_i b_i)    R2 = +0.1328        bits NOT retained
a SINGLE bit        b_i               R2 = +0.2111
```

A signed sum requires individual bits and fails; the count succeeds. **The
retention alternative is dead**, and this is a stronger control than the
permutation null I had, because it holds the *input* fixed and varies only the
*function* being asked for.

### VII.2 A claim of mine that pass 2 killed

I then reported "any non-negative weighting is decodable — a monotone
accumulator". That was **wrong, and wrong in the project's signature way**: any
non-negative weighting correlates ~0.99 with the count, so those successes were
the count leaking through. Residualising each target on the count:

```
random POSITIVE w~U(0,1)   corr w/ count 0.988   residual R2  -0.012
random SPARSE 8/64         corr 0.826            residual R2  +0.013
linear in position w=i/M   corr 0.990            residual R2  +0.858
```

Random weightings carry **nothing** beyond the count. Smooth *positional*
weightings carry a great deal. That is a different claim from the one I made.

### VII.3 What is actually there: a low-pass positional summary

Decoding positional Fourier modes of the bit sequence, `sum_i cos(2πki/M) b_i`,
residualised on the count:

```
k=1  +0.585    k=3  +0.123    k=8   +0.125    k=24  -0.249
k=2  +0.220    k=4  +0.077    k=16  +0.006    k=32  -0.075   (Nyquist)
```

**The answer token carries the count plus roughly the first two or three Fourier
modes of the positional bit profile, with a cutoff near k ≈ 3 out of a possible
32.** The count is the DC component of a low-pass positional summary.

This is a sharper and more informative description than "a counting register",
and it is a genuine characterisation result: the model summarises a 64-symbol
sequence by a handful of low-order positional moments rather than by retaining
it or by reducing it to a scalar.

### VII.4 Blind spot 3 (is v just the token-embedding difference?): argued, not fully closed

Two arguments, neither requiring the embedding matrix. (a) The `y_{i-1}` control
already targets it: an embedding-difference direction encodes only the CURRENT
symbol, so its projection should not track the value *before* that symbol; it
tracks it at r = +0.269, 16/16 same sign. (b) An embedding difference is a fixed
vector, independent of unroll count — but `|cos(v_1, v_64)| = 0.454`, so the
estimated direction moves substantially with depth and cannot be a fixed
embedding contrast.

**Still not closed:** neither argument bounds what *fraction* of v is
embedding-contrast. Since the register correlation is 0.27, most of v's variance
is something else, plausibly the current symbol. The honest statement is that a
component of v carries the accumulated value, not that v *is* the register axis.
Closing it needs the embedding matrix — one cheap GPU call.

### VII.5 Blind spot 2 (architecture-specificity): open, and it is the big one

Nothing here establishes that any finding is about Huginn rather than about
normalised residual streams generally. No untrained baseline, no second
architecture, no non-recurrent control. Every claim in this project is
single-checkpoint. This is the largest remaining threat to external validity and
it cannot be addressed offline.

### VII.6 Revised assessment of what is novel

Pass 2 changes the ranking in III.3. The low-pass positional summary (VII.3) is
now the most interesting *characterisation* result — it is quantitative, it has
a clean control, and it says something specific about how a recurrent-depth
model compresses a sequence. It sits near the "what do models represent about
sequences" literature, which I have also not engaged (see below).

---

## Pass 2 stopping condition: NOT met

Researching three blind spots produced one dead alternative (VII.1), one killed
claim of mine (VII.2), one new result (VII.3) — and exposed **new** thin areas:

- **N1. Sequence-summary literature not engaged.** VII.3 is a claim about how a
  model compresses a sequence into a fixed-width state. There is relevant work
  on positional summarisation, on what RNN/SSM states retain, and on
  compression-based views of representation, none of which this project cites.
- **N2. The k=1 mode is only R² = 0.585, and I have not asked what limits it.**
  Is the cutoff set by the recurrence, by attention, or by the noise floor?
  Testing across dtypes would separate them and connects to the bf16 result.
- **N3. VII.3 rests on one dump (220 prompts, r=64, one task).** The bandwidth
  claim needs the same sweep at other depths and on task b.
- **N4. Blind spot 3 remains open** and is one GPU call from closed.
- **N5. Blind spot 2 (architecture-specificity) remains fully open.**

A third pass is required. The cycle is doing exactly what it should: each pass
has killed at least one of my own claims and produced at least one result that
was invisible from the previous vantage point.

---

## Pass 3 — which kills the pass-2 headline

### VIII.1 The bandwidth claim does not survive a power check

VII.3 concluded that the positional code is low-pass with a cutoff near k ≈ 3.
Subsampling the 220 prompts:

```
n prompts     k=1      k=2      k=4      k=8
      220   +0.585   +0.220   +0.077   +0.125
      160   +0.511   +0.026   +0.057   +0.004
      110   +0.410   +0.041   -0.274   +0.023
       70   +0.011   +0.073   +0.047   +0.083
```

**k=1 collapses with sample size** — 0.585 → 0.011 — so nothing has saturated at
n = 220. Ridge R² from cross-validation is downward-biased at small n, so a
curve still rising in n means the estimate is power-limited. The apparent cutoff
at k ≈ 3 may therefore be **where my statistical power runs out, not where the
model's positional code ends.**

**VII.3 is retracted as a bandwidth characterisation.** What survives is weaker
and still worth having:

- positional information beyond the count **exists** — a linear-in-position
  ramp has residual R² = 0.858, far above any single Fourier mode and far above
  the random-weighting controls at ≈ 0;
- k = 1 carries more than k ≥ 4 *at this sample size*, which is a lower bound on
  the positional content, not a description of its spectrum.

Settling the spectrum needs a prompt count where R² saturates in n. That is a
sample-size calculation I have not done and a run I have not made.

### VIII.2 Task b is not answered

With n = 16 per condition, every residual R² is negative — ridge on 5280
features with 16 samples is vacuous, as flagged before running it. N3 stands
open. The generalisation test needs a task-b dump at the scale of the task-a one
(220 prompts, not 16).

### VIII.3 A practice failure this pass exposed

Both in pass 2 and here, **I wrote the interpretation into the script's own
print statement before seeing the numbers** — "=> the cutoff is a property of
the representation, not of the sample" was printed by code that had not yet
compared anything to anything. The numbers then contradicted it, and the
contradiction was visible only because I read the table rather than the summary
line I had authored.

This is a specific, correctable defect, and it is a variant of the project's
signature error: a statement whose truth was assumed by its own container. The
rule that follows: **a script may print numbers and the criterion, never the
verdict.** The verdict is written after reading the numbers, in prose, by hand.
`run_precision_check.py` and `run_manifold_null.py` do this correctly (they
branch on the measured value); the ad-hoc analyses in passes 2 and 3 did not.

### VIII.4 Where the cycle now stands

Three passes, three of my own claims killed (the monotone-accumulator reading,
the bandwidth cutoff, and — from pass 1 — the framing of the
observable-convergence law as a spotlight result). One control genuinely
established (bits are not retained: signed-sum R² = 0.13 against count 0.99).
One real but weaker finding standing (positional information beyond the count,
residual R² = 0.858).

Remaining open, ranked:

- **O1. Architecture-specificity.** No baseline, no second model. Every claim is
  single-checkpoint. *Largest threat to external validity.*
- **O2. Positional-code spectrum.** Needs a power analysis and a larger dump.
- **O3. Task-b generalisation.** Needs 220-prompt scale.
- **O4. Is `v` partly the token-embedding contrast?** One GPU call.
- **O5. Probing-methodology literature** (control tasks, MDL, tuned lens) still
  unengaged in the write-ups, though VII.1 has now done the *substance* of a
  control task without citing the frame.
- **O6. Sequence-summary literature** unengaged.
- **O7. No global multiplicity correction** over the ~20 newer tests.

---

## Pass 4 — literature, multiplicity, and the gap both of them expose

### IX.1 Where this work actually sits, by literature

Written from knowledge rather than from a search, and flagged as such: specific
claims below should be checked against primary sources before any of this is
cited. That caveat is not boilerplate — this project's second- and third-most-cited
sources turned out to be unlocatable and misattributed respectively (III.1).

**Probing methodology.** The relevant frame is Hewitt & Liang (2019) on control
tasks and *selectivity*, Voita & Titov (2020) on MDL/description-length probes,
and Pimentel et al. (2020) on information-theoretic probing. The shared point is
that decodability conflates what the representation contains with what the probe
can compute. This project ran permutation nulls throughout, which tests whether
a mapping exists but not whether the *probe* is doing the work — and only in
pass 2 did it run the substance of a control task (VII.1: holding the input
fixed and varying the target function). That control should have been the first
thing, not the twelfth.

**Lens methods.** nostalgebraist's logit lens and Belrose et al.'s *tuned lens*
exist because a fixed decoder is biased at intermediate depths and must be
re-fit per layer. D37 rediscovered that along the *recurrence* axis and I
presented it as new. The genuine delta is the quantitative rate — that the
re-fitting requirement decays at the model's contraction rate — not the
qualitative fact.

**Counting and length generalisation.** There is an active literature on whether
transformers can count and why they fail beyond training length. This project's
most interesting datum for it is the **gap between decodable and used**: the
count is recoverable from the answer token at R² = 0.993 while the model's own
output accuracy is ~10%. That is an "information present, readout fails" result,
and it is the kind of thing that literature argues about.

**Recurrent-depth and iterative models.** Geiping et al. (Huginn) is the model;
Deep Equilibrium Models (Bai, Kolter & Koltun), Universal Transformers
(Dehghani et al.) and looped transformers are the family. The bf16 result
(apparent convergence is a precision artifact) is a claim about this whole
family, and is the finding with the best claim to generality.

**Population-dynamics nulls.** Elsayed & Cunningham (2017) is the direct
precedent for the winding failure and *was* engaged, correctly.

### IX.2 The gap that engaging the literature exposes: nothing here is causal

Every positive result in this project is **decoding**. Not one is
**interventional**. The modern bar — activation patching, causal mediation,
causal abstraction (Geiger et al.), and the broad critique that a probe can
read a feature the model does not use (Ravichander et al.) — is not met
anywhere.

The specific missing experiment is cheap and obvious in hindsight: **patch the
state along `v` and check whether the emitted count moves.** If adding `k·v`
shifts the model's answer by `k`, the register is causally used; if the answer
is unchanged, `v` is a correlate the model ignores, and the whole "counting
register" framing weakens to "count-correlated variance exists."

This is now the largest single methodological gap in the project, larger than
any of O1–O7, and it was invisible until the literature frame was applied.
Recorded as **O8**.

### IX.3 Multiplicity, and an error I nearly made

Pooling all 20 confirmatory tests from D24–D39 into one BH family gives "14/20
survive", which I was about to report as reassurance. **That is an artifact of
family construction.** BH gains power from strong positives in the family, so
mixing a p = 5.5e-72 result with marginal H2 strata *resurrects* claims
(D26's permutation strata at p = 0.027; the answer-clustering min-p) that were
correctly reported as null under their own within-family correction.

Done correctly, within scientifically distinct families:

```
H2 permutation (4 strata)        0/4 survive BH     Bonferroni 0.107
answer probe (2 tasks)           1/2                Bonferroni 0.010
register direction (3 conds)     3/3                Bonferroni 2.8e-10
depth scaling (2 designs)        2/2                Bonferroni 0.0027
readout (2)                      2/2                Bonferroni 0.0002
```

And the robustness floor — Bonferroni against the **entire** 20-test count, the
most conservative correction available:

```
register task a      p x 20 = 1.8e-08     depth at fixed length  p x 20 = 0.027
register b_unbal     p x 20 = 1.9e-09     readout curve          p x 20 = 0.002
register b_bal       p x 20 = 4.7e-05     count vs PC1           p x 20 = 2.0e-08
bits-not-retained    p x 20 = 2.0e-08
```

Every project-critical positive survives. **Multiplicity was never the threat to
these results.** Power, confounds, single-checkpoint scope and the absence of
causal evidence are.

---

## Part IX-bis — The hypothesis register, as it stands now

The project still nominally runs on H1/H2/H3, which is misleading: those are
mostly settled, and the live questions are ones nobody wrote down. Stating the
actual current set, with status and the test that would move each.

### The original three

| | claim | status | note |
|---|---|---|---|
| **H1** | trajectories fall into settle/loop/drift | **dead as posed** | the instrument had one reachable output; a working classifier exists now (`classify_shape_regime`) but the hypothesis was never interesting once the trajectory turned out to be transient-then-noise |
| **H2** | winding ∝ required depth | **false as operationalised, TRUE in a different index** | winding at the answer token tracks the recording budget. But required *depth* does scale with difficulty (rho +0.225 at fixed length), so the phenomenon is real and the geometric readout was wrong |
| **H3** | contraction forbids counting | **premise true, claim rescoped** | ρ = 0.802 measured exactly. Contraction forbids an *unbounded* register; a bounded one fits comfortably, and one is there |

### The live hypotheses, which are not in any project document

| | claim | status | what would settle it |
|---|---|---|---|
| **HA** | The count is computed in ~1 unroll; recurrence performs **readout**, not computation | supported (D37 flat register + D38 rising readout) | the causal test now running; and whether attention heads that aggregate digits are active at r=1 |
| **HB** | The answer token holds a **low-dimensional summary** — count plus positional structure — not the input and not a scalar | supported but under-powered (ID ≈ 10, count in ~3; positional residual R² 0.858; spectrum unresolved) | n ≈ 450 prompts, which both the power analysis and the TwoNN validity bound independently call for |
| **HC** | Low precision **truncates visible computation**, not merely adds noise | strongly supported on one model (regime 20.7 → 96.3 bf16 → fp32) | the same sweep on a second iterative architecture; this is the most generalisable claim in the project |
| **HD** | On a normalised stream a register must be realised as a **rotation**, not a translation | derived, **untested** | measure the angle swept per increment and check for the aliasing a rotation implies |
| **HE** | The difficulty-dependence of required depth lives in **logit formation**, not information availability | open — D38 showed readout quality is difficulty-independent at every depth, so the r\* effect is unexplained | compare the probe's decodability curve against the model's own margin curve, per instance |
| **HF** | The register is **causally used**, not merely decodable | **being tested now** | activation patching along the readout direction, with two controls |

### What this register makes obvious

Three of the six live hypotheses (HA, HE, HF) are about the **gap between what
the state contains and what the model uses**. That gap was not a research
question when the project started; it emerged from the finding that the count is
decodable at R² = 0.993 while the model's own accuracy is ~10%.

That is the project's real subject now, and it has a literature (the probing
critique, causal abstraction) that the project has never cited. **The framing
drifted from geometry to mechanism without anyone updating the framing
documents**, which is why the hypothesis register had to be reconstructed here
rather than read off.

---

## Part X — The interpretability method landscape, and where we sit in it

This is the planning artifact the review was missing. Column "used?" is factual;
"what it would buy" is specific to *this* project's open questions, not generic.

### X.1 The matrix

| # | method | used? | what it bought / would buy here | cost | priority |
|---|---|---|---|---|---|
| 1 | **Linear probes** | **heavily** | the register, the readout curve, the positional finding. Everything positive. | free (CPU, banked states) | done, but see #4 |
| 2 | **Logit lens** | **yes, unnamed** | `_replicate_coda_head` *is* a logit lens — it applies coda+head to intermediate unrolls. Used in `eval_depth` and the v6 probe. We never called it that, so we never inherited its known caveats. | free | recognise it as such |
| 3 | **Tuned lens** | **no** | the principled fix for D37: fit an affine correction per unroll instead of noting that the direction rotates. Would turn "the frame rotates" into a corrected readout and situate the finding in existing work. | 1 GPU hr | **high** |
| 4 | **Activation patching / causal tracing** | **running now** | closes O8 — whether the register is *used* or merely decodable. Without it every positive is correlational. | 1 GPU hr | **highest** |
| 5 | **Interchange interventions** (causal abstraction) | **no** | strictly stronger than #4: swap the register component between a count-20 and a count-40 prompt and see whether the *answers* swap. Tests the register as a causal variable, not just a steerable axis. | 1–2 GPU hr | **high** |
| 6 | **Steering vectors** | **partially** (#4 is a steering experiment) | if patching works, the natural follow-up is whether the same axis steers behaviour at other counts/tasks. | cheap once #4 lands | medium |
| 7 | **Attention/circuit analysis** | **no — total gap** | the obvious mechanistic question is *which heads aggregate the digits into the answer token*. Huginn's core block has attention; head-level patterns from the answer position to digit positions are cheap and would give a mechanism rather than a correlate. | 1 GPU hr | **high** |
| 8 | **Sparse autoencoders** | **no** | the modern standard for unsupervised feature discovery. Would test whether the register is a natural feature of the state or an artifact of asking for it. Expensive; needs a corpus of states and a training run. | days | low (scope) |
| 9 | **Intrinsic dimensionality** | **partially** | participation ratio 3.88 and the ~3-D count subspace are ID-flavoured. Proper estimators (TwoNN, MLE) would make it a real ID claim and connect to the compression literature. | free | medium |
| 10 | **Jacobian / "J-space"** | **yes** (D31, Arnoldi on JVPs) | gave ρ = 0.802 and the complex spectrum — the project's only exact measurement. **Extension not done:** the Jacobian w.r.t. the *input embeddings* rather than the state, which is input attribution and would show which positions the answer depends on. | 1 GPU hr | **high** |
| 11 | **Gradient attribution** (IG, attribution patching) | **no** | cheaper approximation to #4/#10 across all positions at once; good for triage before expensive patching. | cheap | medium |
| 12 | **RSA / CKA** | **no** | comparing trained vs untrained representations — the baseline kernel currently running does a bespoke version; CKA would be the standard one. | free once states are banked | low |
| 13 | **Topological (PH, RTD)** | **attempted, failed** | PH failed for a *good* reason (T ≪ n makes a 128-point cloud a generic simplex). **Barannikov's own RTD was never applied**, which is notable given he is the curator. | cheap | medium, and politically relevant |

### X.2 What the matrix says about the project's shape

Three things stand out.

**We used exactly one family.** Everything positive came from linear decoding.
That is one axis of evidence supporting a mechanistic-sounding claim, and it is
why O8 (causality) was invisible for so long — within the decoding paradigm
there is no question it fails to answer.

**We used two methods without naming them**, and paid for it. The logit lens
(#2) came with a literature of caveats we never inherited. The Jacobian work
(#10) was framed as bespoke rather than as spectral analysis, so I did not think
to also differentiate w.r.t. the *input*.

**The cheapest unexplored methods are the most diagnostic.** #7 (attention
patterns) and #10-extended (input Jacobian) each cost about one GPU hour and
each would convert a correlate into a mechanism. Neither was attempted in six
weeks, because the project's frame was *geometric* and these are *mechanistic*.

### X.3 Concrete plan, in dependency order

1. **#4 causal patching** — running. Gates everything: if the register is not
   causal, the framing changes and #5–#7 become less interesting.
2. **#7 attention analysis** — which positions the answer token reads. Cheapest
   route to a mechanism.
3. **#10 input Jacobian** — same question by a different method; agreement
   between #7 and #10 would be strong.
4. **#5 interchange intervention** — the strongest causal claim available.
5. **#3 tuned lens** — reframes D37 correctly and connects it to prior work.
6. **#9 proper ID estimation** — cheap, sharpens the ~3-D claim.
7. **#13 RTD** — worth doing for the curator's own method, on the answer-token
   state cloud across prompts (where it is well-posed) rather than on a single
   trajectory (where it is not).

### X.4 The practice lesson, generalised

The review asked what I should improve beyond following instructions. This is
the clearest answer: **I chose methods by inheritance rather than by
enumeration.** The project began with a geometric frame, so I refined geometric
tools — nine winding variants, surrogate nulls, regime separation — long past
the point where the frame had been shown to fail. A method matrix like X.1
written at week one would have shown that the geometric family was one column of
thirteen, and that the two cheapest unexplored columns answered questions the
geometric one structurally could not.

The rule I would extract: **when a measurement fails, enumerate the method space
before refining the failing method.** Refinement is the default because it is
locally cheaper; enumeration is what actually moves the work.

---

## Part XI — Practice: the concrete artifacts, not the exhortations

The review is only useful if it produces things I can *run*, not resolutions.
Each item below is derived from a specific failure in II.2, not from general
good advice.

### XI.1 The pre-report checklist

Six questions, one per error class in the taxonomy. Every one of my nine
retractions would have been caught by exactly one of them, before reporting.

1. **What object is this a property of?** Name it out loud. Is it the object in
   my claim? *(Caught nothing at the time; would have caught ρ-to-own-endpoint,
   ρ-across-prompts, and `J_TT` vs the coupled operator — 3 errors.)*
2. **What interval is this averaged over, and is all of it signal?** *(Would have
   caught the cosine, `steps_to_settle`, `dlr`, `conv_rate`, the DMD mode — 5.)*
3. **What else predicts this target?** Position, length, token identity, the
   trivial answer, the prompt itself. Run the top candidate *before* reporting.
   *(Would have caught the register R²=0.892 and the +4.17° clustering — 2.)*
4. **Can this statistic take a different value on this data?** Compute its range
   before interpreting it. *(Would have caught `classify_shape`'s single
   reachable output and the `n_open/n_close` identity — 2.)*
5. **Does any preprocessing touch held-out data?** *(Would have caught the
   level-mean residualisation leakage — 1.)*
6. **What is the critical value at this N, and am I aggregating away power?**
   *(Would have caught the level-means depth test — 1.)*

The checklist is cheap: at most a few minutes each, versus the hours each
retraction actually cost.

### XI.2 Script discipline, from two specific failures

- **A script prints numbers and the criterion. Never the verdict.** In passes 2
  and 3 I authored `=> the cutoff is a property of the representation` into a
  print statement *before* any comparison existed, and the numbers then
  contradicted it. `run_precision_check.py` does this right — it branches on the
  measured value — the ad-hoc analyses did not. The verdict is written by hand
  after reading the table.
- **Raw data off the expensive machine first.** Four GPU kernels returned
  summary numbers before one returned states. Every later question then cost new
  GPU time and left conclusions whose data did not exist locally — the same
  failure the audit was documenting. The states dump should have been kernel #1.
- **Name the method you are using.** `_replicate_coda_head` is a logit lens;
  the Arnoldi work is spectral analysis. Naming them would have inherited their
  literatures — and, for the Jacobian, would have prompted the obvious
  extension to input-space differentiation that I never considered.

### XI.3 The working loop that actually produced results

Stated so it is repeatable, in the order that worked:

1. Take the claim and try to make it trivial. Most of the good findings came
   from asking "what would make this vacuous?" and then discovering it was.
2. Do the **weak** version first. The strongest result in the project — the
   register direction at 66.8σ — is a mean difference and a cosine. It beats the
   ridge probe it replaced *because* it has no fitted parameters.
3. Build a calibration arm before believing any null.
4. Stratify before pooling. Pooling produced three spurious results here.
5. Retract in writing, with the diagnosis. Six retractions with mechanisms beat
   six quiet fixes, because the mechanisms are what prevented repeats.

### XI.4 What I would change about how I chose problems

The honest failure is not any single error; it is that I spent roughly four
fifths of the effort refining a measurement family that had already been shown
to fail, and one fifth on the methods that produced every positive result. The
geometric tools got nine winding variants, three surrogate nulls, a regime
framework and a manifold-respecting null. The decoding tools got a probe. The
decoding tools produced the findings.

That allocation was never chosen; it was inherited from the project's framing
and never re-examined. **The correction is Part X's matrix, written early
rather than late.**

### XI.3b Idea generation — the honest audit, and what substitutes for inspiration

IV.3 counted where the productive ideas came from. Reading that count again, the
pattern is sharper and less flattering than I first wrote.

**Almost nothing came from unprompted generation.** Three of the pivotal ideas
were short external questions. Two came from a *systematic sweep* (the audit's
file inventory; the generator census). Two came from *taking a constraint
literally* (RMSNorm ⟹ sphere ⟹ rotation not translation). One came from a hook
refusing my claim.

The two categories that were genuinely mine — sweeps and constraint-taking — are
**mechanical procedures, not inspiration**. That is the useful finding: my idea
generation works when it is enumeration and fails when it is intuition. So the
correct investment is in enumeration machinery, not in trying to be more
creative.

Concretely, the procedures that produced ideas:

- **Enumerate the space, then look at what is empty.** Part X's method matrix
  produced four immediate experiments; it should have existed in week one. The
  file inventory found two directories I had never accounted for, one of which
  held the only multi-init data in the project.
- **Take an architectural constraint literally and follow it to a consequence.**
  "RMSNorm means the state is on a sphere" is a one-line observation that
  invalidated a rotation centre, an entire null model, and a register design.
  Nobody had followed it because it reads like a technicality.
- **Ask what the data shows that the hypothesis does not predict.** The bf16
  finding — the best result here — came from noticing that "settles at t≈14" was
  a claim about the *dtype*, not the model. That is an anomaly the framing
  actively hid.

And the anti-pattern, stated so I can recognise it: **refining a failing
measurement feels like progress and is not.** Nine winding variants were nine
attempts to fix an estimator of an object (winding number of an open arc) that
does not exist. Each felt like a step. None was.

### XI.4b A natural experiment on the duplication problem, run by accident

IV.2 criticised the seven kernels for duplicating ~60% of their code because
each must be self-contained. Pass 4 produced direct evidence.

The eighth kernel failed with `TypeError: unsupported operand type(s) for /:
'int' and 'type'` — from `np.arange(1, n + 1, float)`, which passes `float` as
the *step* rather than the dtype. **This is a bug I had already found and fixed
in a local analysis days earlier.** Checking the other kernels:

```
kaggle_baseline/main.py:99     np.arange(1, n + 1, float)        <- the bug
kaggle_register/main.py:151    np.arange(1, n + 1, dtype=float)  <- correct
kaggle_register_depth/main.py  np.arange(1, n + 1, dtype=float)  <- correct
```

The correct form existed in two kernels and I still reintroduced the broken one
when writing a third from scratch. A fix applied to duplicated code does not
propagate, and human memory is not a propagation mechanism.

The concrete correction: **upload a shared payload once as a Kaggle dataset and
have every kernel import from it**, instead of inlining. That was rejected early
for expedience — inlining avoided pushing to the user's GitHub — but a Kaggle
dataset achieves the same isolation without the duplication. Cost of the wrong
choice, measured: one failed GPU run and one debugging cycle, plus whatever
silent divergence exists between the seven copies that I have not checked.

### XI.5 Matrix item #9 executed: proper intrinsic dimensionality

Participation ratio was a proxy; TwoNN (Facco et al. 2017) is the estimator.
On the 220 answer-token states:

```
TwoNN intrinsic dimension        10.23
participation ratio               4.64
PCA dims for 80 / 90 / 95 / 99% variance    5 / 11 / 21 / 41
count-carrying subspace: 1PC 0.529   3PC 0.912   10PC 0.977
```

The sharper statement this licenses: **the answer-token cloud has intrinsic
dimension ≈ 10, and the count occupies roughly 3 of those directions.** The
earlier phrasing ("the count is ~3-dimensional") was right but incomplete — it
omitted that the count subspace is a *part* of a larger low-dimensional
structure, and said nothing about what the other ~7 dimensions carry. Given
VII.2's finding that smooth positional weightings decode beyond the count, the
natural hypothesis is that some of them are positional — testable, untested.

Caveat, stated because it matters: TwoNN with n = 220 in 5280 ambient dimensions
is at the edge of validity (it assumes locally uniform density and enough
neighbours). It needs the n ≈ 450 the power analysis independently called for —
the same bound, arrived at from a different direction, which is mild
corroboration that 220 is the binding constraint on several of these questions
at once.

---

## Part XII — Where the field is, and the program that follows

### XII.1 The state of the sub-field this sits in

Recurrent-depth and looped architectures (Huginn, Universal Transformers, DEQs,
looped transformers) are having a moment because they promise test-time compute
scaling without longer chains of text. The interpretability of that family is
much thinner than the interpretability of standard transformers, for a
structural reason: **the standard toolkit is indexed by layer, and these models
have one layer applied many times.** A "feature at layer 12" has no analogue; a
feature at unroll 12 is the same weights in a different state.

That is exactly where this project stumbled and, I think, where the opportunity
is. Three concrete gaps in the family's tooling that this work bumped into:

1. **Lens methods have no recurrence-axis version.** The tuned lens fits an
   affine correction per layer. The analogue here — per-unroll — does not exist
   as a named tool, and D37 shows it is needed: the readout basis rotates and
   locks only after ~16 unrolls.
2. **Convergence is assumed, not measured.** Every claim of the form "the model
   settles" in this family is a claim about the numerics as much as the model.
   The bf16 result says apparent convergence depth scales with mantissa bits,
   which nobody appears to check.
3. **Nulls are inherited from the layer world and are wrong here.** A surrogate
   that ignores the state manifold, or a statistic averaged over a fixed unroll
   budget, produces artifacts specific to iterated maps. This project produced
   four of them.

### XII.2 The program, in priority order

**Tier 1 — closes the current story.**
1. Causal patching (running) and interchange interventions. Decides whether
   "register" survives as a description.
2. Attention/head analysis and the input-space Jacobian: which positions the
   answer token reads. Two cheap routes to the same mechanism; agreement would
   be strong.
3. n ≈ 450 prompts. Three separate analyses (spectrum, TwoNN validity, effect
   precision) are all bounded by n = 220.

**Tier 2 — makes it generalisable.**
4. The bf16 sweep on a second and third iterative architecture. This is the
   paper-shaped result and it is currently one checkpoint.
5. The untrained baseline (running) extended: is any of this about *learning*?

**Tier 3 — connects it to the field.**
6. A per-unroll tuned lens, named as such, compared against the fixed decoder.
7. RTD on the answer-token cloud across prompts — well-posed there, unlike on a
   single trajectory — which is also the curator's own method, never applied.

### XII.3 What I would tell someone starting this project again

- Write the method matrix (Part X) in week one. It costs an hour and it would
  have redirected four fifths of the effort.
- Dump raw states from the first GPU run, not the fifth.
- Run one control task before running twelve refinements of a metric.
- Check what a statistic's *range* is on your data before interpreting its
  value. Two of the nine retractions were statistics that could not vary.
- When a citation is load-bearing, open it. Two of the three most-cited sources
  here are broken, and one of them justified a conclusion that was backwards.

## Pass 3 stopping condition: NOT met

O1–O4 are load-bearing and unaddressed; O2 and O3 are the direct consequence of
this pass. The honest summary of the cycle so far is that **my confidence has
been correctly reduced at every pass**, which is the intended behaviour of the
procedure and an uncomfortable but accurate description of where the
understanding stands.
