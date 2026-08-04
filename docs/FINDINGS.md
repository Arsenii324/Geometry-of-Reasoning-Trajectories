# Findings

Huginn-3.5B (`tomg-group-umd/huginn-0125`, revision `bb6621b6…`), 2026-07-26 → 08-04.
Every number below is reproducible from a script in `scripts/` or a kernel bundle
in `scratch/kaggle_*/`; the ledger row is cited for each. Evidence levels are
stated, including for the negatives.

**Read `docs/state_of_knowledge.md` for the full picture including retractions,
and `docs/backlog_not_done.md` for what was never attempted.**

---

## The one-paragraph version

The original hypothesis — that latent-trajectory *geometry* encodes reasoning
depth — is not supported, and the metrics that appeared to support it were
measuring the recording window and the arithmetic precision rather than the
model. The counting register that replaced it is real but **architectural**: a
randomly-initialised Huginn reproduces it (D40). What survives that control, and
is the project's actual result, is an inversion of the assumption probing rests
on. **The untrained model decodes the count 20× more precisely than the trained
one** — held-out R² 1.0000 (error 0.046 counts) against 0.9928 (0.934 counts) — at
a sequence length where the trained model's *measured* accuracy is 0%, so the more
precisely decoded model is not the more capable one (D41). And **one parameter explains it: training slows the
contraction, ρ = 0.715 → 0.887 measured directly on the operator**, lengthening
the depth time-constant from 3.0 to 8.3 unrolls (D42, D44). A fast contraction is finished after ~8 unrolls however many
you give it; slowing it is what makes test-time depth do anything at all, and it
is paid for in linear decodability of the input. Two independent facts about the
architecture stand apart from all of this: **required recurrent depth scales with
difficulty** at fixed prompt length (D35), and **bfloat16 rounding makes the
model appear to converge ~4.6× sooner than it does** (D30).

---

## 0. The headline: decodability and capability move in opposite directions

**Claim: D41, D42. Evidence: verified-live on GPU; the ρ-across-training sweep is
queued and could still falsify D42.**

Identical protocol on trained and randomly-initialised Huginn — 220 prompts, all
exactly 74 tokens, float32, states captured at every unroll in one forward pass,
readout scored by 5-fold `cross_val_predict` with `Ridge(alpha=1e3)`:

| unroll r | 1 | 2 | 4 | 8 | 16 | 32 | 64 | mean abs error at r=64 |
|---|---|---|---|---|---|---|---|---|
| **trained** | 0.675 | 0.904 | 0.972 | 0.967 | 0.986 | 0.992 | 0.993 | **0.934 counts** |
| **untrained** | 0.200 | 0.415 | 0.851 | 0.997 | 1.000 | 1.000 | 1.000 | **0.046 counts** |
| measured accuracy at this length | | | | | | | | trained 0%; untrained cannot exceed chance |

Three things make this a result rather than an artefact:

1. **It is not an interpolating probe**, the obvious objection at d=5280 ≫ n=220.
   Every prediction is out-of-fold, and the label-permutation null sits at −0.05
   to −0.19 — reliably *negative*, which is the signature of an honest held-out
   score. A probe with capacity to interpolate would push the permuted null
   toward zero.
2. **It is not decoding the generative parameter.** Prompts come from five
   bit-rates, so a probe could score high on `p_one` alone — but that caps at
   R²=0.9237 / error 3.851 counts. Trained errs 0.934 (4.1× finer), untrained
   0.046 (**83.7× finer**). Both exceed the ceiling, so both resolve count
   *within* level.
3. **The ceiling dates "earlier" properly.** Crossing R²=0.9237 is the point where
   a readout provably carries more than the task's generative parameter. Trained
   crosses at r≈2.4, untrained at r≈5.7 — training makes the count available
   **2.3× earlier**. (The raw r=1 gap, 0.675 vs 0.200, sits *below* the ceiling
   for both, so it is a coarse-resolution difference only.)

**The mechanism is a single number, and it is not the one I first proposed.** I
asserted that re-injection makes the untrained net a coherent linear accumulator;
that has a closed-form consequence, and fitted to the data it fails — both curves
pin ρ at the 0.999 boundary with systematically S-shaped residuals. Retracted.
What fits is this project's own derived law (`observable_convergence.py` eq. 3),
`R²_∞ − R²_r ~ C·ρ^{2r}`: untrained ρ̂ = 0.6618, trained ρ̂ = 0.9155.

**Then ρ was measured directly on the operator, and both numbers moved (D44).**
Two-orbit convergence over 12 prompts × 128 unrolls, two independent estimators,
both models in one process: **untrained 0.7150 ± 0.0069, trained 0.8866 ± 0.0362**
(Welch t=16.1, p=2e-09, 4.7 sd), with the step-norm estimator agreeing at 0.7164
vs 0.8646. **The direction is confirmed. The magnitude was inflated 1.5×** — the
curve-based estimator is biased *outward* at both ends (7.4% low on the untrained,
3.3% high on the trained), so it exaggerated the very gap it was used to measure.
Direct gap **+0.1716**, not +0.2537. Time constants: **2.98 and 8.31 unrolls**.

**ρ is a property of the operator, not the prompt (D45).** Across counting,
nesting-depth, arithmetic word-problems and commonsense continuations — token
lengths spanning 15 to 74 — ρ varies by at most **0.047**, against a training
effect of 0.172. That is 3.6× smaller than the effect, and it closes the largest
open assumption in the backlog. The estimator is calibrated on this exact model — its
trained output agrees with three unrelated direct measurements (orbit convergence
0.85–0.90, Arnoldi 0.79–0.81, feature rotation 0.861/0.868).

So neither curve shows a feature being *built*. Both are pure convergence, exactly
as `observable_convergence.py` warns, and the untrained model looks both "later"
and "more precise" because it is a **faster contraction onto a better-conditioned
fixed point**.

**Why it matters.** This is a naturally-occurring counterexample to "high probe R²
⇒ the model represents the quantity usably", on a real architecture with a
*meaningful* target — the usual form of that critique relies on synthetic
random-label control tasks. Both models carry the count essentially perfectly
(R² 1.0000 and 0.9928) at a length where `results/counting_accuracy.csv` measures
the trained model at **0%**.

**Correction, and the limit of the claim.** An earlier version of this section read
"R²=1.0000 at 0% accuracy against R²=0.9928 at ~10%" and concluded that
decodability and capability move in opposite directions. **Neither kernel measured
accuracy.** The ~10% was the average over a different configuration (n_ops 2–48),
and it is 37.5% at n_ops=2 falling to 0% from n_ops=8 onward. At M=64 both models
are at zero. So what is established is that **probe R² is uninformative about
capability here**, not that training trades precision for usability — the latter
needs a task where the trained model is measurably better. Scoring accuracy across
the ten training checkpoints alongside ρ, on this same task, is queued and settles it.

**What the same run cost elsewhere.** The bound in D43 — that ρ predicts how deep
Huginn can usefully think — **does not survive the direct ρ and is retracted.** At
0.8866 the state is 95% settled by r=24.9, *below* the r≈32 where ARC-C with
few-shot saturates; the claim held only on the inflated inferred value, and flips
with both the estimator used and the arbitrary choice of convergence threshold. A
quantity that changes sign under two free choices is not a bound. What survives is
an observation: at r=32 the trained state is ~98% converged, the right order of
magnitude for the published 8–32 range, and nothing more.

**RESOLVED (D52).** Across **14 independent weight-sets** — 5 random inits and 9
trained checkpoints — **every untrained model contracts faster than every trained
one**: 0.7048 ± 0.0087 vs 0.8577 ± 0.0139, Mann-Whitney **U=0** (complete
separation), p=5.0e-04, Cohen d=13.2. This is the comparison across *weights* that
D44's withdrawn p-value could not make. But **91% of the effect is already present
at the earliest published checkpoint** (step 6144), and the within-training trend
flips significance under a fit-quality filter (p=0.036 → 0.19), so *"ρ keeps rising
through training"* is not claimed. It is a step change occurring before any saved
checkpoint.

**Old note, superseded.** The eight intermediate checkpoints of Huginn's own training run
would turn the two-point contrast into a curve. The sweep reached only the two
anchors: one checkpoint has an older config missing a field the current modeling
code reads (`test_time_noise`), and the other seven hit CUDA OOM — a 3.5B model in
float32 is 14 GB against the T4's 14.56 GB, so the third load fragments. Both are
fixable; neither affects D44, which is measured.

---

## 1. Positive results

### 1.1 Required depth scales with difficulty — confound-free (D35)

At Barannikov's design (m = 64 fixed, difficulty varied by the number of ones),
**all 200 prompts tokenise to exactly 77 tokens**, so difficulty and prompt
length are decoupled by construction rather than by statistical control.

Measure: teacher-forced `log P(gold) − log P(distractor)` at every unroll,
marginalised over surface forms, with the threshold r\* defined as the first
unroll where the margin reaches 90% of its final level and stays there.

```
pre-registered test, complete cases   rho = +0.175, p = 0.044, n = 133
censoring-aware, all instances        rho = +0.225, p = 1.3e-03, n = 200
                                      bootstrap 95% CI [+0.086, +0.355]
```

67/200 instances never resolve within 64 unrolls, and those have **higher**
counts (36.5 vs 30.0, p = 0.014) — so dropping them removes the largest r\*
values and biases the estimate *downward*. Treating them as right-censored
strengthens the result, as the bias direction predicts. The censoring rate is
itself a second, independent signature: harder instances are both slower to
resolve and likelier never to.

Effect size is modest: mean r\* rises from 28.9 (counts 2–15) to 50.2 (38–50).

A hypothesis of mine — that difficulty is really about *balance*, since a count
of 58/64 is as uniform as 6/64 — was tested and **rejected** twice: entropy
gives rho = +0.139 against count's +0.225 and adds +0.020 R² in a joint rank
regression.

### 1.2 A counting register exists, and it is a ℤ-action (D34, D36)

> **QUALIFIED BY D40 — read this before §1.2, §1.2b, §1.2c and §1.2d.** A
> randomly-initialised Huginn reproduces every endpoint measurement in these four
> subsections: the register, its sign consistency, the ℤ-action, and the count
> subspace. So the *existence* of the register is a property of the architecture
> — 5280 dimensions, RMSNorm, and prompt re-injection — not of training. What
> below is genuinely trained is the **depth trajectory** (§0): the trained model
> reaches within-level count resolution 2.3× earlier and holds a 20× coarser
> representation. Every "the model has learned…" reading of §1.2–§1.2d is
> withdrawn; the measurements themselves stand exactly as reported.

Found by subtraction, not by a probe. For per-position states, form
`Δ_i = h_{i+1} − h_i` and take

$$v = \text{mean}(\Delta \mid \text{increment}) - \text{mean}(\Delta \mid \text{decrement})$$

No fitted parameters, so nothing to overfit.

**The direction is shared across independent strings:**

| condition | pairwise cosine | vs random (sd 1/√5280 = 0.0138) |
|---|---|---|
| Task a (running count) | **+0.9199** | 66.8 σ |
| Task b (balanced parens) | **+0.9383** | 68.2 σ |
| Task b (unbalanced control) | **+0.8847** | 64.3 σ |

All 16 strings per condition select the same digit-region offset, so the search
found the real region rather than per-string noise.

**It carries the accumulated value, not position or token identity.** The target
is `y_{i−1}` — the register value *before* the current symbol, which token
identity at position *i* cannot explain — with position and current symbol both
partialled out:

```
Task a       r = +0.2686   16/16 same sign   p = 9.2e-10
Task b bal   r = +0.2248   16/16             p = 2.4e-06
Task b unbal r = +0.2890   16/16             p = 9.5e-11
```

Task b is the harder test: depth is a *signed* bridge, so `R²(depth ~ position)`
is only 0.160 and a direction tracking "how far along am I" cannot fake it. For
Task a, position alone explains 98.5% of a running count, which is why the raw
uncontrolled probe R² of 0.892 was worthless.

**It behaves as a group action.** Two-step displacement along v, by symbol pair:

```
                 ((        ()        )(        ))     matched vs 0
balanced      +31.61    +16.39    −17.26    −29.67    −0.44 (p=0.891)
unbalanced    +30.33    +20.39    −17.29    −31.84    +1.55 (p=0.666)
```

`(` and `)` displace symmetrically, and **a matched pair returns the state to
where it started** — that is `T₋₁ = T₊₁⁻¹`. Nothing in the construction of v
forces this. Deviation from strict additivity: `()` reads +16.4 rather than 0,
so the *earlier* symbol of a pair dominates the two-step difference; additive in
the mean, not term by term.

**The two tasks use separate axes** — cosine between Task a's `+1` direction and
Task b's `(` direction is **+0.058**. No reuse of a single increment axis.

### 1.2b Recurrence rotates the register, it does not build it (D37)

The two positives above stood unconnected. If unrolling *constructs* the
register, its decodability should rise with unroll count. Measured in one
forward pass per string, states captured at every unroll:

```
r         1      2      4      8     16     32     64
register_r   +0.248 +0.257 +0.242 +0.255 +0.251 +0.250 +0.250     rho=+0.006  p=0.958
|cos(v_r,v_final)|  0.454  0.562  0.727  0.906  0.992  1.000  1.000   rho=+0.990  p=5.5e-72
```

**The content is fully present after a single core-block application and does
not grow. The direction rotates, reaching 99% alignment by r≈16.** All 12 seeds
follow the same rotation curve (across-seed sd falls 0.014 → 0.000), so this is
not an averaging artifact.

I had pre-registered the opposite prediction. The consequence matters: since the
register is complete at r=1, the extra depth that harder instances require in
§1.1 is **not** being spent building it. What it is spent on is open.

Caveat: `|cos| → 1` is partly definitional as r → 64. The informative part is
the shape of the approach, not the endpoint.

### 1.2c What the extra depth is FOR: readout (D38)

§1.2b left a gap. If the register is complete at r=1, the depth that harder
instances demand in §1.1 is not building it. The hypothesis: the register is
distributed across 64 positions, the answer is one token, and *aggregating* it
is a different job from computing it.

Decoding the total from the **answer-token state alone**, 220 prompts all at
exactly 74 tokens:

```
r          1      2      4      8     16     24     32     48     64
readout R2  0.675  0.904  0.972  0.967  0.986  0.992  0.992  0.993  0.993
```

`rho(unroll, R²) = +0.983`; every point beats its permutation null at p<0.001
with null means −0.10 to −0.27. **Contrast with §1.2b on the same model at the
same depths: the per-position register is flat (rho = +0.006) while the
answer-token readout rises (rho = +0.983).** The two differ only in *where* the
state is read, so the difference is about location, not method.

**The mechanism, assembled:** the count is computed immediately — attention can
sum bits in one pass; its representational frame rotates into a stable basis
over ~16 unrolls; and it is progressively transferred into the answer token
over ~24. The depth requirement in §1.1 is about the last of these.

**Two honest limits, one of which splits the claim.** R² = 0.675 at r=1 is
already high, so the depth-dependent part is the final +0.318, not the whole
thing.

More importantly: per-instance readout error was tested at **every** depth and
does **not** track difficulty anywhere — rho ranges −0.09 to +0.16 across
r = 1…64, the one nominal hit (r=8, p=0.016) dies under Bonferroni over nine
tests, and the signs alternate. Mean error does fall steeply (6.94 → 0.93
counts).

So depth buys readout *in aggregate*, but that does **not** explain why harder
instances need more of it. The two measurements ask different questions: a
linear probe asks whether the count is PRESENT in the answer-token state (it is,
R²=0.65 at r=1), while §1.1's r\* asks when the model's own logit margin
saturates. Information availability and the model's use of it are different
things, and §1.1's difficulty-dependence lives in the latter — which the probe
bypasses. That is the open question.

### 1.2d The original thesis is true — at the right index (D39)

This project asked whether latent *geometry* encodes reasoning depth, and
measured it as the shape of a trajectory over unrolls, where it is false
(§2.1, §2.2, §2.4). Asked of the **answer-token state across prompts**, it is
true:

```
PCA of 220 answer-token states   36.1% / 22.7% / 13.7% / 5.0% / 3.7%
participation ratio              3.88 of 10
count vs PC1                     pearson +0.727   (spearman +0.755)
cumulative R2 for the count      0.529 (1 PC)  0.912 (3 PCs)  0.977 (10 PCs)
```

**The count occupies a roughly 3-dimensional subspace of the answer-token
state.** PC1 is the dominant axis but carries only about half of it alone.
All 220 prompts are exactly 74 tokens, so length cannot explain it. State norms
are 76.37 ± 0.0007 — a 0.0010% variation, so the RMSNorm sphere holds across
prompts as tightly as across depth.

This is the same correction made three times now: the instinct was right and
the index was wrong. Winding belongs across token positions, not unrolls
(§3). Depth belongs to readout, not register construction (§1.2c). And
geometry belongs to the answer representation across instances, not to
trajectory shape.

### 1.3 H3's premise, measured exactly for the first time (D31)

Implicitly-restarted Arnoldi on autodiff Jacobian-vector products at the
converged state, float32:

```
rho = 0.7935 / 0.8042 / 0.8083   at n_ops 64 / 32 / 8   (mean 0.8020)
```

Contraction confirmed, varying under 2% across an 8× difficulty range —
independently corroborating a two-orbit estimate of 0.84–0.90 obtained from
different data by a different method. The leading eigenvalue is **complex in
3/3 prompts**, in conjugate pairs, with 8–10 oscillatory modes: **the recurrent
map rotates**, even though winding cannot detect it (§2.1).

Scoping correction: contraction forbids an *unbounded* register, not a bounded
one. The converged neighbourhood spans ~1.4 units across 5279 tangent
dimensions — ample to separate 64 states, which is what §1.2 finds.

### 1.4 bfloat16 masks ~4.6× of the computation (D30)

Same prompts, same seed, only the compute dtype varying:

```
dtype        residual radius    regime ends at
bfloat16          1.1895            20.7
float16           0.1548            32.3
float32           0.000290          96.3      (of 127)
```

bf16 → fp16 gives 7.7× against 8× predicted, slope −0.981 against −1. So the
post-convergence residual is **arithmetic**, not a limit set.

The larger implication is in the third column. "Huginn settles at t ≈ 14" — the
fact the entire project was built on — holds **only in bfloat16**. In float32 the
same prompts keep converging to t ≈ 96. Low precision does not merely add a
floor; it *truncates visible computation*. This is a claim about the published
architecture, and it generalises to any convergence claim made about an
iterative system in low precision.

---

## 2. Negative results

### 2.1 Winding does not measure rotation (D28)

Under an on-manifold null (calibration arm 6.4% against 5% nominal, so the
construction is unbiased), the effect **flips sign with `num_steps`**:

```
ns=64    obs−null = +0.0127   mean z = +10.97   (n=60)
ns=128   obs−null = −0.0091   mean z =  −5.58   (n=80)
```

In both tasks independently, strata differing at p = 1e-11. `num_steps` is a
recording budget: the same prompts and weights, recorded longer, reverse the
comparison. **|winding| tracks how much arithmetic noise was recorded, not the
trajectory's content.**

### 2.2 Winding does not track difficulty (D26)

Permutation test on real trajectories only — no surrogate model, so the manifold,
convergence profile, noise floor and anisotropy are all preserved by
construction. No stratum survives BH or Bonferroni; signs disagree across strata;
one nominal hit in four is what chance gives (P = 0.185).

### 2.3 No quantisation of winding on balanced strings (D32)

`docs/register_geometry.md` derived that a balanced string closes the curve,
making position-indexed winding a genuine integer invariant, and predicted
near-integer values. Measured distance to nearest integer: balanced 0.281,
unbalanced 0.353. The pre-registered comparison passes (p = 0.0005) **but is
misleading** — against the actual no-quantisation reference (uniform → 0.250),
balanced strings are *not* closer (p = 0.9954). The gap is driven by unbalanced
strings being anomalously far, not balanced ones being close.

### 2.4 H1 was never tested (rigor_audit §18)

`shapes.gate.classify_shape` returns "settle" for **140/140** trajectories
because it compares the last step to the *largest* step, and `s[-1]/s.max()`
maxes at 0.0188 against a 0.10 threshold. "loop" and "drift" are unreachable. An
instrument with one attainable output cannot test a three-way hypothesis.

---

## 3. What this means for the curator's tasks

The register content is **present and linearly decodable** in both Task a and
Task b, and behaves as the ℤ-action the minimal-algorithm framing specifies.
Three design points from this work:

1. **Probe in tangent coordinates, not ambient space.** RMSNorm confines the
   state to a shell of relative thickness 7.3e-5, which bounds a straight-line
   register to 0.0144 per increment over m=64 — 62× below the arithmetic noise
   floor, and the bound survives a 1000× error in its one input. A translation
   register is architecturally impossible; a rotation (equivalently a tangent
   translation, which agrees at small angle) is not.
2. **Fixed m is doing real work.** Every other synthetic generator in this
   project has `rank-corr(difficulty, seq_len) = exactly 1.000`, which makes a
   length control mathematically degenerate. Fixing m breaks it at the source,
   and §1.1 is the first result here that cannot be prompt length.
3. **Probe the latents, not the output.** Model accuracy on counting is ~10%,
   and the apparent successes are a zero-prior artifact — correctness is
   perfectly separated by whether the answer happens to be 0 (4/5 vs 0/8, all at
   unroll 1).

Do **not** read the register off a winding number: §2.1 and §2.3 show winding
sees neither the rotation that is present in the map (§1.3) nor the register
that is present in the states (§1.2).

---

## 4. Honesty notes

- Six claims of mine were retracted during this work, including a wrong
  noise-floor formula, an underpowered "reversal", and a symmetry test that was
  an algebraic identity (`n_open/n_close`, giving 31/32 = 0.96875 with sd
  1.9e-16 — no model property can be that constant). All are recorded in the
  ledger with their diagnoses rather than deleted.
- The register correlation carries an ~18.5% leakage estimate from residualising
  position over all 64 points. Real, but far from the exact determination that
  invalidated an earlier probe.
- 16 of 18 historical results files have **no raw data on disk**, so their
  conclusions cannot be corrected without GPU re-runs. That is the concrete cost
  of storing derived scalars and discarding what produced them.
- Effect sizes are modest throughout: rho ≈ 0.23 for depth scaling, r ≈ 0.27 for
  the register. These are real and pre-registered, not large.

Verification: 193 tests, `ruff check src scripts tests` clean, 155 artifacts
hash-verified, all GPU results from Kaggle T4 kernels whose logs are in
`scratch/kaggle_*/out/`.
