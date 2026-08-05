# State of understanding — 2026-08-05

Supersedes `REVIEW_state_of_understanding.md` (last touched 2026-08-04, predates
D59–D63). Written to be read on its own; every number cites the ledger row that
carries its evidence.

---

## 1. The one-paragraph version

The project asked whether latent-trajectory *geometry* in Huginn-3.5B encodes
reasoning depth. It does not, in the form originally proposed. What the work
actually established is narrower and, I think, more interesting: **training changes
the model's dynamics, not the content of its state.** The contraction rate rises
from ρ = 0.7048 ± 0.0087 to 0.8577 ± 0.0139 across 14 independent weight-sets with
*complete separation* (D52), while every content-level measurement — linear
decodability of the task variable, effective dimensionality, endpoint geometry, the
counting register itself — is matched or **beaten** by a randomly initialised model
(D40, D41, D48, D53). And ρ is now the first quantity in the project to be moved
rather than merely observed: a gradient-free rescale of the core block's output
projections shifts it with paired slope +0.285 ± 0.045 (D59).

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

*Status: UNDER TEST. `geometry-cap-content` (queued) spans capability from ~100%
(`echo_num`) through `add1` and `count4` to 0% (`count16`) and measures the
trained-minus-untrained content gap at each, with a pre-registered prediction: if
6.1 is a real confound the gap is positive where capability is high and ~0 where it
is zero. Either outcome is decisive — it either scopes the headline to "tasks the
model cannot do" or strengthens it considerably.*

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

*Status: UNDER TEST in the same kernel, via a nonlinear probe. Getting that probe
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

### 6.6 What a reader should not conclude

- Not "Huginn cannot count" — only "does not, at these sizes, under these prompts."
- Not "geometry is irrelevant" — winding failed; the rotation it was trying to
  measure is real (D55).
- Not "training does nothing to representations" — see 6.1; the test was weak.
