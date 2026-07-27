# Results report — Geometry of Reasoning Trajectories in Huginn-3.5B

> **⚠ PARTIALLY SUPERSEDED — read `docs/state_of_knowledge.md` first.**
> This file predates the two-pass rigor audit of 2026-07-25/26
> (`docs/rigor_audit.md`, `claims_ledger.md` D24–D26) and contains at least one
> claim the audit overturned. Known overturns that may appear below:
> `steps_settle` is a proxy for the contraction rate, **not** for effective
> compute; the "82.1% wind less than their null" result rests on an
> off-manifold null and is deleted; `cos = −0.276` is a noise-regime artifact
> (the computing-regime value is **+0.084**); "homology max persistence 0.009"
> is a normalisation artifact; and the "Movahedi et al." citation could not be
> located. Nothing here has been deleted — this notice is additive, and the
> text is left as its authors wrote it.


STATUS: written 2026-07-24. An exhaustive, grounded pass over every experiment
and metric currently in `results/`. Every number below was recomputed directly
from the raw CSVs (not copied from memory or older docs) and cross-checked by an
independent adversarial-verification pass; corrections that pass found are folded
in and noted. Where a claim is uncertain, underpowered, or confounded, that is
stated inline, not omitted. This report supersedes scattered result summaries;
for the per-claim evidence ledger see `claims_ledger.md`, for the forward plan
see `project_plan.md`.

---

## 0. What was measured, and how (the shared setup)

**Model.** `tomg-group-umd/huginn-0125`, pinned revision `bb6621b…`. A
recurrent-depth transformer (~3.5B params): an embedding → 2 prelude layers →
a **4-layer core block iterated N times** ("unrolls") → 2 coda layers → output
head. The same core-block weights are reused every unroll; "reasoning depth" at
test time = number of unrolls.

**What a trajectory is.** For one input prompt, we run the forward pass and, at
each unroll step, record the hidden state of one token position via a forward
hook on the last core-block layer. Stacking those gives a path
`[num_steps, hidden_dim=5280]` through latent space — the "trajectory." Unless
noted, `num_steps=64`, the random hidden-state initialization uses `seed=0`
(fixed — a real limitation, see §7), and only the **answer token** (last
position) is traced. Extractions ran on **Kaggle T4 GPU**.

**The four scalar metrics** (all defined before use):
- **winding** = `|winding_of(traj, burn=4)|`: project the trajectory to 2D via
  per-trajectory PCA (dropping the first 4 steps as an init transient), measure
  the total signed angle swept, take the absolute value (PCA sign is arbitrary).
  Large winding ≈ the path loops/orbits; ~0 ≈ it moves in one direction and stops.
- **steps_settle** = first unroll at which the step-to-step displacement drops
  below 10% of its maximum. A proxy for "effective compute": how many unrolls the
  path keeps moving before going quiet.
- **shape** = `classify_shape(traj)` ∈ {`settle`, `loop`, `drift`}: settle = pulls
  to a fixed point; loop = orbits; drift = keeps moving without returning.
- **contraction / lyap** = convergence-rate proxies (mean log step-norm ratio).
  Note: this is a convergence rate, **not** a true finite-time Lyapunov exponent
  (it cannot tell a slow-settling path from a neutral loop).

**The canonical statistic.** For "does metric M rise with difficulty D," the
trusted statistic is the **per-level Spearman**: collapse all repeated
measurements at each difficulty level to one group-mean, then rank-correlate the
N level-means. This avoids the inflated sample size (and false significance) of
correlating all individual rows, which treats many correlated within-level
measurements as independent. At **N=6 levels**, the critical |ρ| for p<0.05 is
**0.886**; at **N=4**, no |ρ| can reach p<0.05 at all (the smallest achievable
two-tailed p is 0.083) — significance is structurally unreachable.

**The three hypotheses under test** (from the project proposal):
- **H1** — a token's latent path is one of a few shapes (settle / loop / drift),
  distinguishable by geometry.
- **H2** — *when the path loops*, the winding number grows with the number of
  reasoning steps required.
- **H3** — forced strict contraction (spectral radius ρ<1 of the recurrent map)
  destroys the ability to hold a running count, so state-tracking tasks must
  loop or drift rather than settle.

---

## 1. H1 — the shapes (settle / loop / drift)

**Finding: at full compute budget, on this project's tasks, essentially
everything settles. Genuine loops exist but are very rare and require either an
artificially starved budget or a specific enriching prompt.**

- **Full-budget synthetic + PARARULE**: every counting trajectory, all 40
  PARARULE trajectories, and all 15 banked convergence trajectories classify as
  `settle`. Directly confirmed: `conv_rate < 0` for **15/15** banked trajectories
  (strict exponential pull to a fixed point); in the `num_steps=64` subgroup the
  step size shrinks **~52×** end-to-start with adjacent-step cosine ≈ −0.28
  (settles by *inward oscillation* — zig-zags inward), while the starved
  `num_steps=16` subgroup shrinks only ~10× with cosine ≈ +0.26 (a monotone
  glide — qualitatively different dynamics). [convergence.csv, 15 rows, recomputed
  exact]

- **Forced loops by starving compute** (the cleanest H1 result). `forceloop.csv`
  (96 rows): at `num_steps=16`, the unsettled (loop+drift) fraction rises with
  count length — **1/8 at n_ops=8 → 7/8 at n_ops=24 → 6/8 at n_ops=48**; a Fisher
  exact test on the 8-vs-24 settled/unsettled 2×2 gives **p=0.0101**. At
  `num_steps≥24`, **100%** settle. So loops are a symptom of too little compute,
  not of depth per se. The broader `phase.csv` (216 rows over num_steps × n_ops)
  gives the regime mix **133 settle / 61 drift / 22 loop** — i.e. the "everything
  settles" statement is specific to full-budget runs; across starved budgets,
  drift and loops are common.

- **First real (non-starved) loops** (`blayney_repro.csv`, 7,122 token-trajectories,
  the largest dataset here). Reproducing Blayney et al.'s (arXiv:2604.11791)
  "Long Persona" system prompt on 24 GSM8K questions, all token positions: the
  per-token loop rate is **0.1286% (7/5445)** under Long Persona vs **0.0596%
  (1/1677)** with no system prompt — the former within ~8% of Blayney's own
  independently-measured 0.14%, a passing external positive-control. Every
  non-settle here is a `loop` (zero `drift`). **Novel position finding
  [verified]:** all 7 Long-Persona loops sit at relative position **0.819–0.897**
  (the last ~fifth of the prompt); **zero** loops occur in the first two-thirds;
  per-third loop counts are **[0, 0, 7]**. Five of the seven come from a single
  GSM8K example (idx 18); the strongest (winding=1.22, the first of that cluster)
  lands on the token **" makes"** — one of the two tokens Geiping et al.
  (arXiv:2502.05171) name by name as orbit-prone. Loop windings span 0.30–7.18
  (several are multiple full turns, unlike the ~0.65 sub-turn loops seen under
  starved budget).

**H1 verdict.** The three-way shape taxonomy is real and the pipeline detects it
correctly (validated against Blayney's independent method at a matching rate).
But at full budget on this project's own tasks, only "settle" occurs; loops
require starving compute or an enriching prompt. **Caveats:** (a) the
persistent-homology H1 metric intended as a third discriminator is degenerate —
max normalized H1 persistence is ≤0.009 across all 15 trajectories, because a
single 1-D curve has no independent 1-cycles by construction; this is a
mathematical artifact, not a finding about Huginn, and the sound version
(population/delay-embedding homology) is not implemented. (b) The Lyapunov
exponent named in H1 is unmeasured — only a convergence-rate proxy exists.

---

## 2. H2 — does winding grow with reasoning depth?

**Finding: no. Across every task where the length confound is genuinely
controlled, winding shows no consistent relationship with difficulty. The only
"significant" positive hits are the tasks where difficulty is perfectly
confounded with prompt length.**

The cross-task table of per-level winding~difficulty Spearman ρ (all recomputed
exactly from the raw CSVs, verified):

| Task | winding~difficulty ρ | N levels | length-confounded? |
|---|---|---|---|
| counting | **+0.943** (p=0.005, sig) | 6 | **yes** (rank-corr n_ops~seq_len = 1.0) |
| maxtask | **+0.943** (p=0.005, sig) | 6 | **yes** (rank-corr = 1.0) |
| dissociation-track (5-seed) | +0.771 | 6 | matched-length control |
| pararule | +0.200 | 4 | partly (depth~seq_len = 0.816) |
| switch | −0.086 | 6 | yes (rank-corr = 1.0) |
| three_scale (active_len) | −0.400 | 5 | **no** (but prefix confound, §2b) |
| three_scale_modk (N=7) | +0.107 | 7 | **no** (length constant) |
| three_scale_modk (N=15) | −0.125 | 15 | **no** (length constant) |

**The two significant hits (both +0.943) are length artifacts.** For counting,
switch, and maxtask, the difficulty variable `n_ops` is a *strictly monotone
function of* `seq_len` (each n_ops maps to exactly one token count — every added
operation is exactly one added token, e.g. "Add 1." ≡ "Subtract 1." in tokens).
The rank correlation of n_ops with seq_len is **exactly 1.0**. So
"winding~n_ops" is literally the same measurement as "winding~seq_len" — depth
and length are inseparable by construction. The length control cannot even be
run: `partial_spearman` correctly **raises** on this data (residualizing n_ops
against seq_len leaves only ~1e-14 of floating-point noise) rather than
returning a noise-driven number. [verified live: raises for all three tasks]

**The clean tests are null.** `three_scale_modk` is the first task designed so
that total token count and answer-token position are held *constant by
construction* (verified: seq_len std = 0.0, at 42 tokens for N=7 and 36 tokens
for N=15). There, winding~active_len is null and stays null under a
**properly-powered replication**: Fisher-combined p across the two moduli =
0.266 at N=15 (with real power at N=15 — see §5 — so this reads as a genuine
absence, not merely underpowered).

### 2b. Two confounds this project's own "controlled" designs still had

- **three_scale prefix confound (D11).** The original three_scale placed its
  "irrelevant" filler as a *prefix block*, so it lengthened total context AND
  pushed the answer token later — reintroducing exactly the length/position
  confound it was built to remove. Its headline ("winding tracks irrelevant
  padding, β=−0.488, p=7e-11") is therefore consistent with winding tracking
  total length / answer position, not reasoning content. Not a clean H2 test.
- **three_scale_modk residual confound [newly found, verified].** In the modk
  design, `neutral_len = total_len − active_len − irrelevant_len` with total_len
  fixed, so at fixed active_len, `irrelevant_len` and `neutral_len` are **exactly
  perfectly negatively dependent** (correlation −1.0 within every active_len
  group). A joint 3-way regression finds significant `winding~irrelevant_len`
  (β=−0.21, p=4e-5) and a huge `winding~modulus` (β=+0.69, p=6e-29) — but the
  irrelevant_len coefficient is mathematically inseparable from −(the neutral_len
  coefficient), and the modulus effect almost certainly reflects the question
  text literally changing ("modulo 2" vs "modulo 5"), not difficulty. Milder than
  D11, but real: even the "clean" task has a residual structural confound.

**H2 verdict.** H2 is **not supported**. On the one design with the confound
genuinely removed, winding does not track difficulty, at properly-powered N. The
apparent positive results elsewhere are length effects. Additionally, H2's own
antecedent ("*when the path loops*") is essentially never satisfied on the traced
answer token at full budget (§1), so most winding numbers reported are measured
on settling paths — not the object H2 is about.

---

## 3. H3 — does contraction destroy state-tracking?

**Finding: the theorem is correct; its *applicability to Huginn* is architecture-
dependent and probably does NOT bind. The spectral radius has never been measured
on real Huginn.**

- **The proof** (`contraction_proof.md`): strict contraction (ρ<1) ⇒ a unique
  fixed point and exponential decay of *initial-state (h₀) memory*, with a
  maximum distinguishable-history bound `T_max ≤ log(D/ε)/log(1/c)`. The algebra
  (Banach fixed point + the log manipulation) is independently re-verified as
  correct.
- **The scope subtlety [derived from the proof itself].** The theorem's map is
  `h_{t+1} = R(h_t; e)` with the context `e` held *fixed* throughout — it only
  proves that differences in the *initial state* decay, and says nothing about
  `e`-dependence (the fixed point `h*(e)` is itself an arbitrary function of `e`).
  Huginn re-injects the full prompt `e` every unroll, and for this project's
  counting tasks the count is fully readable from `e` at step 0. So contraction
  erasing h₀-memory does not forbid the count — the theorem only bites if the
  count is encoded the *streamed* way (accumulated over steps, no re-injection).
- **The toy test (A6), recomputed exactly from the sweep CSVs.** A
  recurrent-over-**depth** model (full context re-injected each step, matching
  Huginn): probe R² for decoding the count stays **≥0.996 at every one of 21
  runs**, *including* the strongly-contracting β=20 regime (ρ≈0.25) — R² is even
  slightly *higher* under contraction. A recurrent-over-**time** model (Elman
  RNN, streaming, no re-injection): a sharp phase transition — probe R² **cliffs
  toward ~0.00 and in-accuracy to ~25%** as soon as β≥0.5 forces ρ<0.14. Same
  contraction, opposite outcome, decided entirely by the re-injection topology.

**H3 verdict.** The proven theorem stands, but its practical claim ("contraction
⇒ can't count ⇒ must loop") is a property of *streaming* recurrence and, per the
project's own toy result, likely does **not** transfer to Huginn's
context-re-injecting architecture. **Critical open gap:** the spectral radius
ρ(∂ₕR) has been measured only on the toy model, **never on real Huginn** — so
H3 on Huginn is neither confirmed nor refuted, only scoped. (Note: the toy
estimator computes σ_max via power iteration on JᵀJ — the Miyato 2018 recipe —
which is a distinct quantity from the true spectral radius ρ that Yang et al.'s
direct-J iteration computes; σ_max ≥ ρ always, so it upper-bounds the contraction
factor but is not literally "ρ".)

---

## 4. Correctness — the result that reframes everything (the honesty headline)

**Finding [verified, and the single most important caveat in the whole project]:
Huginn essentially fails the counting task at exactly the depths where the
"geometry tracks depth" signals appear. The geometry correlations are measured on
trajectories that produce wrong answers.**

- **Accuracy collapses with depth** (`counting_accuracy.csv`, 56 rows,
  generation + regex check): n_ops=2 → **37.5%**, 4 → 25%, **8 → 0%, 16 → 0%,
  24 → 12.5%, 32 → 0%, 48 → 0%**. At every level where winding and steps_settle
  "rise with reasoning depth" (n_ops≥8), the model is not holding the counter at
  all.
- **Geometry does not separate right from wrong.** Within this file, winding does
  **not** distinguish correct from incorrect trajectories (point-biserial
  r=−0.087, p=0.53), and steps_settle is *higher* for **wrong** answers
  (r=−0.295, p=0.027) — the opposite of a "more effective compute → better
  reasoning" reading.
- **Per-unroll logit lens** (`v6_correctness_probe.csv`, 24 rows). On the 13
  honestly-verifiable single-digit-answer rows, the correct answer token is in the
  model's **top-5 for 13/13 (100%)** — almost always by unroll step 1 — but is the
  strict #1 argmax for only **4/13 (31%)**, and those four are all `target=0`. So
  the model reliably *shortlists* the right count as a live candidate, but a
  generic small-number bias (favoring 0/1/2) usually wins the final pick. (The
  other 11/24 rows have negative targets whose leading token is the shared "−"
  sign, so a first-token check can't verify them at all.)

**Consequence.** Any presentation of the counting/maxtask winding results as
evidence for "geometry tracks reasoning depth" **must** foreground that the model
fails the task at those depths. The honest reading of the counting "signal" is: a
**length effect (§2), measured on failed reasoning (§4), at N=6 with limited
power (§5)** — not the isolated ρ=+0.943.

---

## 5. The rigor / accountability layer (how much to trust the above)

- **Multiple comparisons (BH-FDR).** Across 46 project-wide correlation tests, 22
  are raw-significant (p<0.05); **20 survive** Benjamini-Hochberg FDR correction
  (identical under project-wide and per-experiment family definitions — the family
  choice doesn't change the answer here). The two "prime target" anomalies
  (maxtask winding +0.943; dissociation-local winding −0.943) both **survive** FDR
  — but are independently invalidated anyway: maxtask by the perfect n_ops~seq_len
  confound (§2), dissociation by non-replication (below). **Lesson recorded:
  FDR-survival was the smaller problem; the confounds and non-replication are what
  actually kill these.**
- **Non-replication.** The 5-seed dissociation "local winding~n_ops = −0.943
  (significant)" drops to **−0.543 (p=0.27, n.s.)** at 15 seeds, and even flips
  sign in the per-row statistic. A small-N artifact that FDR alone would not have
  caught.
- **Formal power [quantified, damning].** At N=6 levels — the modal case for this
  project's synthetic tasks — a per-level Spearman has only **~0.66 power to
  detect even a true ρ=0.9**, and ~0.44 at ρ=0.8 (`power_curve.csv`,
  Monte-Carlo). So the headline statistic misses a real strong effect roughly a
  third of the time. This is the quantitative backing for treating N=6 "hits"
  cautiously and for reading N=15 nulls (§2) as genuine.
- **Loop-rate power.** At Blayney's 0.02% baseline per-token rate, ~**25,000**
  (token,trajectory) draws are needed to expect ≥5 genuine loops; this project's
  full answer-token pool is ~1,294 (expected ≈0.26 loops) — so the near-total
  absence of loops at full budget is the *correctly-powered prediction*, not
  evidence against H2. A pre-registered test family is frozen in
  `power_and_preregistration.md`.

---

## 6. A cross-task pattern worth naming: steps_settle and answer structure

**Finding [verified, previously unnoticed]:** steps_settle~difficulty is
**positive on every task whose answer monotonically accumulates** (counting +0.93,
switch +0.78, count_ones, pararule +0.80, dissociation-track +0.81, three_scale
+1.00) but **negative on every task whose answer saturates or wraps** — running
max (maxtask −0.77) and modular count (modk −0.66 to −0.74, individually
significant at N=15). More content to represent cumulatively → the path keeps
moving longer; a saturating/cyclic target → it settles faster. The modk
steps_settle effect replicates and *strengthens* with N (Fisher-combined p=0.015
at N=7 → **p=0.0006 at N=15**), which is what a real effect does.

Two important qualifiers: (a) this is `steps_settle`, an internal-dynamics proxy,
not accuracy — and §4 shows steps_settle does not track correctness; (b) a
cleaner-looking signal exists but is confounded: `count_ones` shows
`mean_normed_accel~n_ops = −0.976` (p=3e-5, N=8) — the tightest monotone
relationship anywhere in the project — but count_ones has the same D10 length
confound (rank-corr n_ops~seq_len = 1.0), so it too is inseparable from a length
effect. The saturating/accumulating contrast is a genuine, mechanistically
unexplained observation, offered as a lead, not a confirmed claim.

---

## 7. Limitations and uncertainty (what these results cannot say)

1. **Answer-token-only.** Almost every scalar traces only the final answer token
   — architecturally where settling is strongest and where the literature says
   loops *don't* live (they live on question/content tokens). No full-trajectory
   result characterizes "Huginn's geometry," only the answer token's.
2. **Single init seed.** Headline numbers fix `seed=0`. The one robustness check
   (`dissoc_multiinit.csv`, 5 init seeds) shows winding is comparatively
   init-stable (within/between-config std ratio 0.50) but `contraction` is
   **init-noise-dominated** (ratio 1.72 — noise exceeds signal), so any
   contraction~difficulty correlation should be distrusted more than a winding one.
3. **Spectral radius never measured on Huginn** — H3's central quantity is toy-only.
4. **QK probe 0% done** — the curator's own method (Tulchinskii et al., exact
   statistic now known: a query·key dot product with per-setup head calibration)
   is not implemented; it never targeted weight-tied recurrence, so the port is
   genuinely novel and curator-gated.
5. **Geometry vs. activations, untested** — a positive linear probe on the raw
   5280-d state (§4's top-5 result) shows the *state* holds information; it does
   NOT show the *geometric summary* (winding/shape/λ) is what carries it. The
   deepest conceptual gap.
6. **Path-independence** — Geiping's paper states the same shapes emerge
   regardless of init; if true here, winding's init-stability (limitation 2) could
   mean the geometry is an architecture property, not a computation signal.
7. **Power** — N=4 (pararule) is structurally unable to reach significance; N=6 is
   underpowered (§5). Only the N=15 modk run is adequately powered, and only there.
8. **PARARULE is the only non-synthetic dataset used** (d2–d5, N=4). Three of the
   four proposal datasets (ProntoQA-OOD, MultiLogicEval, GSM8K-as-reasoning) and
   all four named baselines are not run.
9. **Lyapunov / true FTLE, RTD, native exit-criteria** — named metrics not
   implemented.

---

## 8. Future work, in rough priority

1. **The keystone extraction** (`project_plan.md` §5, Phase 1.1): one instrumented
   pass capturing *all* token positions + per-layer Q/K + per-step logits, saved
   once, unblocking the null-model test, the all-token shape census, the linear
   probes, and the QK probe. Design is done (6 substeps, ready-to-build vs
   curator-gated split); most of it needs no curator input.
2. **Measure ρ on real Huginn** — port the toy σ_max estimator (or the direct-J ρ
   estimator) to `core_block_forward`; closes the H3 gap.
3. **Geometry-vs-activations discriminator** — test whether geometric features
   *alone* predict task/correctness above a raw-state probe. Resolves limitation 5.
4. **QK-alignment probe** — curator-gated on the (layer, head, unroll-step) search
   protocol; datasets are public.
5. **Redesign the clean task once more** to remove the modk residual confound
   (§2b): vary active/neutral ratio without making irrelevant/neutral perfectly
   anti-dependent.
6. **Explain the saturating-vs-accumulating steps_settle contrast** (§6) — extend
   to more task families; check whether it survives an accuracy control.
7. **Native exit-criteria vs. correctness**, RTD, population-homology — the
   curator-aligned methods not yet touched.

---

## 9. One-paragraph honest summary

Across ~14 experiments on real Huginn-3.5B, the project has produced mostly
**rigorous negatives**. H1's shape taxonomy is real and the pipeline is validated
(loops reproduced at the literature's own rate), but at full budget on these
tasks only "settle" occurs. H2 (winding grows with depth) is **not supported**:
every apparent positive is a perfect length confound, and the one genuinely
length-controlled test is null at properly-powered N. H3's theorem is correct but
its applied claim likely does not bind on Huginn's re-injecting architecture, and
its central quantity (ρ) was never measured on the real model. Crucially, the
model **fails** the counting task at the depths where the geometry "signals"
appear, and the geometry does not separate correct from incorrect trajectories —
so those signals cannot be read as geometry-of-reasoning. The most defensible
contribution is the **audit itself**: the confound-catching, non-replication
checks, power analysis, and pre-registration that show *why* prior positive-looking
claims don't hold — plus one clean, replicated, still-unexplained observation
(settling speed tracks whether a task's answer accumulates or saturates).
