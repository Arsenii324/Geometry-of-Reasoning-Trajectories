# Methods and findings, from zero

STATUS: written 2026-07-24. A single self-contained read for someone who knows
the relevant papers by name but not this project's concrete setups. It states
what is actually computed (implementations, not filenames), every experiment as
an explicit test with its numbers and pass/fail criterion, and — called out
throughout — the assumptions each statistical test makes and where they bite.
Every number is recomputed from the raw data and cross-checked; where a result
is uncertain, underpowered, or confounded, that is said, not omitted.

VERIFICATION PROVENANCE (2026-07-25): every quantitative and factual claim in
this document was re-derived from source in a dedicated max-depth pass —
experimental numbers recomputed from the raw `results/*.csv`; model facts
(revision `bb6621b…`, `n_embd=5280`, prelude/recurrent/coda = 2/4/2,
`mean_recurrence=32`) read from the pinned `raven_config_minimal.py`/`config.json`;
code thresholds (collinearity 0.95, condition-number 1e10, settle/return frac
0.1/0.25, the `>3` loop window, burn=4) read from the actual functions; the
Spearman critical-value table (N6=0.886, N5=1.000, N4 absent) from
`_SPEARMAN_CRIT_P05`; row counts checked as levels×seeds; attributions
(Miyato 2018, Yang, Tulchinskii, Geiping, Blayney) against the primary sources
read this session. Rounding is consistent throughout (e.g. steps-Fisher
0.00059→0.0006, dissociation p 0.2657→0.27, contraction ratio 1.715→1.72). No
discrepancy was found; the load-bearing numbers are additionally pinned by
regression tests (`tests/test_*`).

Two companion docs: `results_report.md` (the same findings, more compact, assumes
domain fluency) and `claims_ledger.md` (every claim tagged with its evidence and
verification status). References below of the form D10, D15, B4, A6, E1, E3 are
that ledger's row IDs — D-rows are cross-cutting/meta claims, B-rows are the
original experiment results, A-rows are theory, C-rows are code claims, E-labels
are the proposal's experiment names — look any of them up there for the full
evidence trail.

**One notation warning to avoid a real ambiguity:** the symbol **ρ** is used for
*two different things* here, because both are standard. In every experiment/
statistics context (Parts III, V, VI) **ρ is Spearman's rank-correlation
coefficient**. In the H3 discussion only (0.1 and Part VII) **ρ is the spectral
radius** — and there it is always written "spectral radius ρ" in full. If you see a
bare ρ, it is the Spearman one. (Part VII also introduces σ_max, the largest
singular value, a distinct quantity from ρ.)

---

## Part 0 — Orientation: the whole thing on one page

Read this before the details; everything after is an expansion of it. Terms in
**bold** here are defined precisely where they are first used later.

### 0.1 The three hypotheses, stated precisely

Every experiment in this document is an attempt to test one of these. The project
proposal states them; here they are exactly, with their antecedents (which matter
more than the conclusions).

- **H1 — a few shapes.** Every token's latent trajectory (Part I.2) is one of
  three geometric types — **settle** (converges to a point), **loop** (orbits),
  **drift** (moves without returning) — and the three are *distinguishable by
  geometry*. The proposal names three discriminating instruments; here is how each
  maps to what is actually implemented, which is itself part of the finding:
  a **Lyapunov exponent λ** → in practice the convergence-rate proxy
  `contraction`/`lyap` (II.4), and it is init-noise-dominated (VI.3); a
  **self-return** measure → in practice `classify_shape`'s loop test, "did the path
  come back near a point it visited *more than 3 steps* earlier" (II.3, the window
  is a strict `> 3`, so a point exactly 3 steps back does not count); **persistent
  homology** → implemented but degenerate on a single curve (VIII.10). So H1's
  taxonomy is real (validated in V.8) but only one of its three instruments
  (self-return, via `classify_shape`) is on solid footing.
- **H2 — loops encode depth.** ***Conditional on the trajectory looping***, the
  **winding number** grows with the number of reasoning steps the task requires.
  The antecedent is load-bearing: a winding number measured on a *settling* path
  is not a measurement of the thing H2 is about, and at full budget on our tasks
  the answer token settles in **every** extraction we have — zero loops on the
  answer token were ever observed except under an artificially starved compute
  budget (V.1) or on non-answer positions of a persona-prompted GSM8K run (V.8).
- **H3 — contraction forbids counting.** If the recurrent update is forced to
  *strictly contract* — **spectral radius** ρ < 1, where ρ is the largest absolute
  eigenvalue of the one-step **Jacobian** ∂hₜ₊₁/∂hₜ (the matrix of partial
  derivatives of the next hidden state with respect to the current one) — then it
  cannot hold a running count, so state-tracking tasks *must* loop or drift rather
  than settle.

### 0.2 Which metric tests which hypothesis

| Metric (Part II) | Hypothesis it serves | Status |
|---|---|---|
| **shape** (settle/loop/drift) | H1 (the taxonomy itself) | measured, validated (V.8) |
| **winding** | H2 (the "loops encode depth" quantity) | measured; never null-tested (II.1) |
| **spectral radius ρ** | H3 (the contraction quantity) | **never measured on Huginn** (VII) |
| **steps_settle** | auxiliary ("effective compute" proxy; in no hypothesis directly) | measured |
| **contraction/lyap** | auxiliary (convergence-rate stand-in) | measured; init-noise-dominated (VI.3) |
| persistent homology H1 | H1's third instrument | degenerate on single curves (VIII.10) |

So: two of the three hypotheses' *primary* quantities are in trouble before any
result — H3's ρ is unmeasured, and H1's homology instrument is degenerate. H2's
winding is measured but un-adjudicated against chance.

### 0.3 The codebase in seven layers (where any piece lives)

An abstract map, so you can place any function without hunting filenames:

1. **Model / extraction** — load Huginn at the pinned revision, run the forward
   pass, hook the core block, return the trajectory (+ optionally reconstruct
   per-unroll logits). *One model, one extraction path.*
2. **Task generators** — pure functions `prompt-string ← (difficulty, seed)`:
   counting, switch, maxtask, variants (track/local), three_scale, three_scale_modk,
   count_ones, projection. No model here — just text.
3. **Metrics** — `trajectory → scalar or label`: winding, steps_settle (scalars),
   shape (a label in {settle,loop,drift}), contraction/lyap, homology. No
   statistics here — just per-trajectory geometry.
4. **Statistics / analysis** — the tests (Part III): per-level Spearman + the
   critical-value table, partial_spearman, multivariate_rank_control,
   benjamini_hochberg, Fisher-combine. No model, no trajectories — just numbers.
5. **Experiment runners** — glue one generator × a difficulty sweep × extraction ×
   metrics → one results CSV. Each is a thin `run_*.py`.
6. **Rigor / meta** — the FDR sweep, the power analysis, the pre-registration, and
   an executable test that checks the docs match the real repo state.
7. **Theory / toy** — the H3 contraction proof (paper math) plus a *separate*
   toy-model codebase that tests H3 empirically on models runnable without a GPU.

The dependency structure: **the layer-2 and layer-4 code imports no GPU/torch**
(generators are text, statistics read CSVs); **layer 1 needs the GPU**; **layer 3
(metrics) is numpy-only but needs a trajectory to run on**. Most raw trajectories
were *not* saved — the experiment runners compute the metrics inline during
extraction and persist only the scalar outputs to CSVs (the exception is the 15
banked `.npy` trajectories used by convergence/homology). This is why most of the
analysis in this doc is 0-GPU (it re-reads the cached CSVs, layer 4), and also why
re-deriving a metric or running the winding null test needs a *fresh* extraction
(layer 1) rather than an offline replay.

### 0.4 The experiments, by type

Grouping the individual experiments (Part V) into kinds, so the *shape* of the
evidence is visible:

- **Synthetic, length-confounded** (nominally H2/H3): counting, switch, maxtask,
  count_ones — all have difficulty ≡ length (Part IV), so none can isolate depth.
- **Length-controlled *by intent*** (the actual H2 tests): three_scale —
  *attempted* the control but **failed** it (its filler was a prefix, so it still
  moved length/position, V.6), and three_scale_modk — the one that *succeeds* at
  holding length constant (clean, V.5, at N=7 and N=15), modulo one residual
  confound (V.5b). Only modk belongs in "length-controlled" without an asterisk.
- **Length-matched dissociation** (H3): track vs local, at 5 and 15 seeds (V.3).
- **Compute-budget / regime** (H1): forceloop, phase (V.1).
- **External validation of the detector** (H1): Blayney persona reproduction (V.8).
- **Correctness / interpretability**: counting_accuracy (real generation), the
  per-unroll logit-lens probe (V.4).
- **Robustness / meta**: dissoc_multiinit (init seeds, VI.3), FDR + power (VI.4).
- **Real (non-synthetic) reasoning data**: PARARULE, N=4 (V.7).
- **Theory**: contraction proof + toy sweep (VII).

### 0.5 Experiment ↔ analysis-method grid

Which post-analysis each experiment feeds (compact map; PL-Sp = per-level
Spearman, part = partial_spearman, MV = multivariate rank control, Fish-c =
Fisher-combine p-values, Fish-x = Fisher exact 2×2, PB = point-biserial, shape =
classify_shape counts, all correlation tests fold into the project-wide BH-FDR).
The `shape` column is only marked where that CSV actually stores a shape/regime
classification — `switch`, `maxtask`, `three_scale`, `modk`, `dissociation`, and
`counting_accuracy` compute winding/steps but not a stored shape:

| Experiment | PL-Sp | part | MV | Fish-c | Fish-x | PB | shape |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| counting | ● | ✗(raises) | | | | | ● |
| switch / maxtask | ● | ✗(raises) | | | | | |
| three_scale | ● | | ● | | | | |
| three_scale_modk (N7, N15) | ● | | ● | ● | | | |
| dissociation (5s, 15s) | ● | | | | | | |
| dissoc_multiinit | ●¹ | | | | | | |
| forceloop | | | | | ● | | ● |
| phase | | | | | | | ●² |
| blayney_repro | | | | | | | ● |
| counting_accuracy | | | | | | ● | |
| pararule | ● | ● | | | | | ● |

¹ `dissoc_multiinit`'s headline analysis is not a Spearman at all but a
within-vs-between-config variance *ratio* (VI.3); it also reports a per-level rho.
² `phase` stores the taxonomy under the column name `regime`, not `shape` — same
three labels, and it runs no Fisher test (it is a raw phase map, not a
significance test).

("✗(raises)" = the length control is *attempted but refuses to run* because the
confounder is rank-collinear — Part III.2; that refusal is itself a result, D10.)

### 0.6 What matters most (if you read nothing else)

In priority order, because the build-up in Parts I–V can bury the punchline:

1. **On the one clean, adequately-powered test, H2 is not supported** — winding
   does not track depth once length is genuinely controlled (V.5, VI.1).
2. **The model *fails* the counting task at the depths where the winding "signal"
   appears** — accuracy is ~0% (0–12.5%) for `n_ops ≥ 8` — **and the geometry does
   not separate correct from incorrect answers** (V.4). This reframes the
   significant winding "positives": they are measured on wrong answers.
3. **The only two *statistically significant* winding "positives" are length
   confounds** (counting and maxtask, both +0.943 with difficulty rank-collinear
   with length; Part IV, VI.1). (Other winding correlations exist but are not
   significant — e.g. the length-*matched* dissociation-track +0.771 is not a
   length confound, just below the N=6 significance bar and non-replicating.)
4. **The one live, replicated effect is not about winding**: settling *speed*
   (steps_settle) tracks whether a task's answer accumulates vs. saturates (VI.2).
5. **The defensible contribution is the audit itself** — the confound catalogue,
   the non-replication check, the power analysis — plus finding 4.

---

## Part I — The object of study

### I.1 The model, concretely

`tomg-group-umd/huginn-0125`, pinned revision `bb6621b…`, loaded in bfloat16
with `trust_remote_code=True` (the architecture is custom code shipped with the
weights). ~3.5B parameters. Its forward pass is:

```
embed → prelude (2 transformer layers)
      → [ core block (4 transformer layers) ] repeated N times   ← "the unrolls"
      → coda (2 transformer layers) → final layernorm → output head
```

The **core block's 4 layers share one set of weights across all N repetitions**
(weight-tied recurrence in depth). "Reasoning depth" at test time = **N, the
number of unrolls** — you can dial it at inference without retraining. We use
`N = num_steps = 64` unless a specific experiment starves it. The recurrence is
*context-re-injecting*: the prompt's full input embeddings (all token positions)
are concatenated back in at every unroll, not just carried implicitly in the
hidden state (this is the crux of the H3 analysis, Part VII). Each hidden state we
look at has dimension **5280**.

One implementation detail with downstream consequences: the recurrent loop is
started from a **random** initial hidden state `h₀` (Gaussian noise), seeded by
`torch.manual_seed(seed)`. Almost all results below fix `seed=0`; only one
experiment varies it (init-robustness, Part VI.3).

### I.2 A trajectory, and how it is extracted

For one prompt, we register a PyTorch **forward hook on the last core-block
layer** (`model.transformer.core_block[-1]`). The hook fires once per unroll and
records that layer's output hidden state. Stacking the N captured states gives a
path

```
trajectory ∈ ℝ^[num_steps, 5280]
```

— one point per unroll, in unroll order. That path through latent space is the
"trajectory" every metric below is computed on. We run `model(input_ids, num_steps)`
(a plain forward pass, **not** `.generate`), and by default keep only **one token
position** — the **last input position** (`token_index = -1`), whose next-token
prediction is the model's answer (the prompts end in `"A:"`, so this is the `:`
position and its prediction is the answer). We call it "the answer token" below as
shorthand, but note it is the position that *produces* the answer, not a token of
the answer itself. Capturing all positions is a one-line change (the hook already
sees the full `[1, n_tokens, 5280]` tensor; we currently slice it) but has not
been run at scale — see Limitations. The random `h₀` makes the first state an
outlier; only the **winding** metric drops a burn-in for this (`burn = 4`), the
other metrics use the full path.

For the correctness probe (Part V.4) we also apply a **logit lens** — decode an
*intermediate* hidden state through the model's output head to read off "what
token would the model predict if it stopped unrolling here." Concretely we run
the intermediate state through the model's own tail `final-LN → coda → final-LN →
output-head`. Getting this exactly right required reading the model source
directly — the recurrent loop already applies one layernorm before returning its
state, so the naive `coda → LN → head` is wrong and a built-in check
(`validate_logits`, which compares the reconstruction against a genuine forward
pass and raises on mismatch) caught it (D12). This is the one place we reconstruct
logits; everywhere else we use only the hidden-state geometry.

---

## Part II — The four per-trajectory metrics, exactly as implemented

These are the four quantities computed directly from a single trajectory — three
scalars (**winding**, **steps_settle**, **contraction/lyap**) and one categorical
label (**shape** ∈ {settle, loop, drift}). Two of them (**winding**,
**steps_settle**) feed rank correlations against a difficulty variable; **shape**
feeds category counts and a Fisher exact test, not a correlation; **contraction/
lyap** is mostly unused as a correlate and shows up in the init-robustness variance
analysis (VI.3). (The other two
quantities named in Part 0.2 — the spectral radius ρ and persistent homology —
are *not* in this list: ρ is never measured on Huginn, VII; homology is degenerate
on single curves and covered in VIII.10.) The implementation *and the fragility*
of each matters.

### II.1 winding — "does the path loop, and how much?"

Implementation (`winding_of`):
1. Drop the first `burn = 4` states (the `h₀` transient).
2. **Project the remaining path to 2D with PCA** fit on *that trajectory's own
   points* (`PCA(n_components=2).fit_transform`).
3. Compute the **signed winding number** about the 2D centroid: take the angle
   `atan2(y, x)` of each point relative to the centroid, difference consecutive
   angles, wrap each difference into (−π, π], sum them, divide by 2π. Result ≈
   +1.0 for one counter-clockwise loop, ≈0 for a path that goes out and stops.
4. Experiments use the **absolute value** `|winding|`.

**Assumptions / fragilities you must keep in mind:**
- **The PCA basis is refit per trajectory.** Windings are therefore *not* in a
  common coordinate frame — a magnitude of 0.6 on one trajectory and 0.6 on
  another are not guaranteed comparable in a strict sense. We compare them anyway;
  that is an assumption.
- **PCA always finds *some* 2D plane maximizing spread.** A path that is
  essentially a straight drift with noise in high dimensions can look like it
  curves in its own top-2 plane. So a nonzero winding is *not* by itself proof of
  a real orbit. The intended guard is a **matched-random-walk null test**
  (`matched_random_walk`: same per-step displacement *sizes*, directions
  randomized on the hypersphere — preserves how far each step moved, destroys
  directional correlation) to ask "is this winding bigger than chance?" **That
  null test is implemented but has never been run on real trajectories**, because
  the summary CSVs store only the scalar `winding`, not the raw path (a full
  trajectory is `[64, 5280]` floats; the CSVs keep one number per trajectory). So
  *every* winding number reported below is currently un-null-tested. This is a
  real, known gap, flagged in ledger row D11 (which notes the `winding_null_test`
  could not be run on `three_scale.csv` for exactly this reason).
- **The sign is arbitrary** (PCA components are sign-ambiguous), which is why we
  take `|winding|`. That in turn discards the *directional* content H2 is
  literally about ("winding grows" is a signed statement).

### II.2 steps_settle — "effective compute / how long the path keeps moving"

Implementation (`steps_to_settle`, `frac = 0.1`): compute the per-step
displacement norms `‖hₜ₊₁ − hₜ‖`; return the **index of the first step whose
displacement is below 10% of the maximum displacement** over the trajectory (or
the full length if it never drops that low). Interpretation: how many unrolls the
model "spends" before the path goes quiet.

**Assumptions:** the 0.1 threshold is a fixed, un-tuned choice; the metric is a
proxy for "effective compute," not a measurement of it. A second-order variant
(`steps_to_settle_by_acceleration`, thresholds on how much the step
direction+size is changing) exists as a separate column and is *not*
interchangeable with this one.

### II.3 shape ∈ {settle, loop, drift} — the H1 taxonomy

Implementation (`classify_shape`, `settle_frac = 0.1`, `return_frac = 0.25`):
1. If the **last** step's displacement is below 10% of the max → **settle**.
2. Otherwise, form the full pairwise-distance matrix of the trajectory points;
   look only at pairs **more than 3 steps apart** in time (strict `> 3`, so a pair
   exactly 3 steps apart is excluded); if the *minimum* such far-in-time distance
   is below 25% of the overall maximum distance → **loop** (the path came back near
   a point it visited more than 3 steps earlier).
3. Otherwise → **drift**.

**Assumptions:** three hard-coded thresholds (0.1, 0.25, the ">3 steps" window),
none tuned or validated against a ground truth; "loop" is a return-to-near-a-past-
point heuristic, not a topological cycle detector. This is an independent method
from the winding number (Part II.1) and from Blayney et al.'s FFT-based orbit
detector — agreement across methods (Part V.8, where our detector reproduces
Blayney's loop rate) is evidence the thresholds aren't pathological, but they
remain choices.

### II.4 contraction / lyap — convergence rate (NOT a Lyapunov exponent)

The implemented quantity is a mean log step-norm ratio — a **convergence-rate
proxy**. It is explicitly **not** a finite-time Lyapunov exponent: it cannot tell
a slowly-settling path from a marginally-stable loop. The true FTLE (top singular
value growth of the composed Jacobian) is not computed. Treat "lyap" columns as
"how fast the step size shrinks," nothing more.

---

## Part III — The statistical toolkit and every assumption it makes

This is the section that decides whether the numbers mean anything. The tests are
deliberately simple, but each interacts nontrivially with the sample sizes, so the
assumptions are load-bearing.

### III.1 The core statistic: per-level Spearman

**Why not just correlate all rows.** Each difficulty level (e.g. `n_ops = 8`) is
measured with multiple random seeds (5 to 15, depending on the task), producing
that many correlated rows per level. Treating those rows as independent inflates
the effective sample size and manufactures significance ("pseudoreplication" —
counting correlated repeated measurements as if they were independent samples). So
the **canonical statistic** first **collapses each level to its group-mean**, then
computes Spearman's rank correlation over the N *level-means* (`spearman_by_level`).

**Spearman itself** measures *monotone* (not linear) association via ranks — this
is the property you already know. The subtleties are in the **p-value and the
sample size**:

- The number of levels is **small** — N = 4, 5, 6, 7, 8, or (once) 15 across the
  various tasks. At such N, scipy's default Spearman p-value (a t-distribution
  approximation) is only asymptotically valid. To avoid trusting that
  approximation, the project uses an **exact-permutation critical-value table**
  `_SPEARMAN_CRIT_P05`: the smallest |ρ| that reaches two-tailed p<0.05, computed
  by enumerating all N! rank orderings. At **N=6 the bar is |ρ|≥0.886**; at **N=5
  it is 1.000** (only a perfect correlation is significant); at **N=4 it is
  unreachable** — the smallest achievable two-tailed p is 2/24≈**0.083**, so *no*
  result at N=4 can ever be significant at 0.05, however clean it looks (this is
  why PARARULE, Part V.7, is formally inconclusive by construction).
- **Critical assumption of that table: no ties among the level-means.** The table
  is exact-permutation over *distinct* ranks. Real means can tie — found in the
  modk data (Part V.5): two active_len levels averaged to the *exact* same
  steps_settle, and at that point the table over-states the bar while scipy's own
  tie-aware p-value gives a different (lower) number. When means can tie, prefer
  the p-value scipy returns over the table lookup. This is a documented gap in the
  crit-table's applicability (D15).

### III.2 Controlling for length: partial_spearman

The recurring confound is that "difficulty" and "prompt token length" move
together (Part IV). `partial_spearman(x, y, z)` rank-transforms all three,
**linearly residualizes** the x- and y-ranks against the z-rank (fits and subtracts
a straight line in rank space), and correlates the residuals.

**Assumptions and the guard:** linear-in-rank residualization assumes the
confounder's rank-effect is approximately affine. More importantly, if the
confounder `z` is (near-)perfectly rank-collinear with `x`, residualizing leaves
essentially *zero* real variance — the returned correlation is then computed from
floating-point rounding noise, not signal. The implementation **raises an error
above |rank-corr| = 0.95** rather than return a noise number. This fires on every
synthetic counting-type task, where difficulty and length are rank-correlated
**exactly 1.0** (Part IV) — the length control is not merely weak there, it is
*impossible*, and the code refuses to fake it (D10).

### III.3 Multiple simultaneous confounds: multivariate_rank_control

For a task varying three length scales at once — active, neutral, irrelevant
(Part V.5) — `multivariate_rank_control` rank-transforms the response and all three
predictors, fits **one ordinary least-squares regression** with an intercept, and
reports each predictor's coefficient and a two-sided t-test p-value.

**Assumptions:** (1) the t-tests assume approximately normal residuals — on
rank-transformed data this is an approximation, acceptable at these N but not
exact; (2) a **condition-number guard** (raise above 1e10) catches near-singular
designs — the condition number is the ratio of the largest to smallest singular
value of the design matrix, and a huge ratio means the predictors are nearly
linearly dependent, so their individual coefficients become numerically
meaningless. Note a distinct, subtler trap this guard does *not* catch, found by
hand: a coefficient can be un-interpretable even when the design is well-
conditioned, if the predictors are constrained by construction. In the modk task
(V.5b) the symbol counts satisfy `active + neutral + irrelevant = total` (a fixed
constant), so raising `irrelevant` necessarily lowers `neutral` — the `irrelevant`
coefficient then measures "swap a neutral token for a filler token," not a pure
filler effect, and no amount of good conditioning fixes that. It is an
identifiability problem coming from the *task design*, not the numerics.

### III.4 Combining independent replications: Fisher's method

When the same directional hypothesis is tested in two independent conditions
(e.g. two moduli, different task text, different seeds), their p-values are
combined with **Fisher's method** (`combine_pvalues(method='fisher')`): `−2 Σ ln
pᵢ` is χ² with 2k degrees of freedom. **Assumption: the combined p-values are
independent.** The two moduli use different prompts and seeds, so independence is
defensible, but they run on the *same model* — a shared-model dependence cannot be
fully ruled out, so read a Fisher-combined p as "strong evidence under a
reasonable independence assumption," not as an exact figure (used in Part V.5).

### III.5 Many tests at once: Benjamini-Hochberg FDR

`benjamini_hochberg` applies the standard step-up procedure to control the
false-discovery rate across a family of tests. **Assumptions:** BH controls FDR
under independence or *positive* dependence of the tests; and — the load-bearing
judgment — **what counts as "one family" is a choice that changes the answer**. We
report both a project-wide family and a per-experiment family; here they happen to
agree (Part VI), but that agreement is a fact about this data, not a general
guarantee.

### III.6 The other tests, briefly, with their assumptions

- **Fisher exact test** (forceloop, Part V.1): exact test on a 2×2 count table,
  **no distributional assumption**, but it *conditions on both margins* being
  fixed (hypergeometric) — the standard reading of "exact" here.
- **Point-biserial correlation** (correctness, Part V.4): Pearson correlation
  between a binary (correct/incorrect) and a continuous variable; its p-value
  assumes the continuous variable is approximately normal within each group. At N=56
  with a lopsided binary (mostly-incorrect) it is approximate, not exact.
- **Monte-Carlo power curve** (Part VI): simulates data from a **bivariate-normal**
  generating process (where Spearman ≈ Pearson) to estimate detection power. Real
  trajectory data need not match that generator, so the power numbers are
  *indicative* — good enough to say "N=6 is badly underpowered," not to quote a
  power to three digits.
- **Binomial / Wilson interval** (Blayney per-example rate, Part V.8, D14):
  treats each example as an independent Bernoulli trial; if examples are not
  exchangeable the interval is optimistic.

---

## Part IV — The confound behind the significant winding *positives*: length ≡ difficulty (D10)

In the **naive counting-family** synthetic tasks — counting, switch, maxtask,
count_ones — making the problem harder means adding tokens. (This is *not* true of
three_scale / three_scale_modk, which were built specifically to break it by
holding total length fixed, Part V.5–V.6; the confound below is exactly why those
had to exist.) For `make_counting_task` the prompt is literally

```
Start at 0. Add 1. Subtract 1. Add 1. ... Final total? A:
```

and each additional operation (`n_ops`) is exactly one more `"Add 1."` /
`"Subtract 1."` clause. Crucially, `"Add 1."` and `"Subtract 1."` tokenize to the
**same** length — 3 tokens each (`Add`/`Subtract` are each a single token) — so
*which* operations a given seed picks never changes the token count, only *how
many* operations there are does. That is why, at a fixed `n_ops`, `seq_len` is
**exactly** constant across seeds (verified: one distinct `seq_len` value per
level), and therefore **difficulty `n_ops` and token length `seq_len` are a
strictly increasing function of each other with rank correlation exactly 1.0** —
not approximately (the raw values differ — `seq_len = 3·n_ops + 10` for counting —
but rank-for-rank they are identical). The same holds for switch
(`"Flip."`/`"Wait."`, 2 tokens each) and maxtask (single-digit numbers, 1 token
each).

Consequence: any **rank** correlation of a metric with `n_ops` (Spearman — what
this project uses) **is numerically identical** to its rank correlation with
`seq_len`, because the two share an identical rank order (a Pearson correlation
would *not* be identical, since the raw values differ — but we never use Pearson
for this). Depth and length cannot be separated, the length control cannot be run
(III.2 raises), and this holds for counting, switch, maxtask, and count_ones
alike. This is not a bug to fix with a better statistic; it is a property of the
task designs, and it is why the "clean" tasks (Part V.5) had to be built.

---

## Part V — The experiments, each as an explicit test

For each: the generator (what the prompt literally is), what varies, sample size,
the exact statistic + criterion + result, and the honest verdict.

### V.1 Forced loops by starving compute (the cleanest H1 result, B4)

- **Setup.** Task = counting variants (`make_variants`). Grid: `num_steps ∈
  {16, 24, 32, 64}` × `n_ops ∈ {8, 24, 48}` × 8 seeds = 96 trajectories. Classify
  each with `classify_shape`.
- **Test + criterion.** At `num_steps = 16`, is the unsettled (loop+drift)
  fraction higher for longer counts? Fisher exact test on the `n_ops = 8` vs
  `n_ops = 24` settled/unsettled 2×2.
- **Result.** Unsettled fraction 1/8 → 7/8 → 6/8 for `n_ops` 8/24/48; Fisher
  **p = 0.0101**. At `num_steps ≥ 24`, **100%** settle.
- **Verdict.** Loops are induced by *insufficient compute budget*, not by depth
  per se, and vanish at full budget. This is the project's cleanest positive
  significance result — but it is about the budget, not H2.

### V.2 The synthetic state-holding tasks (counting / switch / maxtask)

- **Setup.** counting (running ±1 sum, answer = the total), switch (light
  on/off, answer = parity of flips: `"Light is off. Flip. Wait. … Is the light on?
  A:"`), maxtask (running maximum digit). Each sweeps `n_ops` over 6 levels with
  5–10 seeds (30 / 60 / 48 rows). Metrics: `|winding|`, `steps_settle`.
- **Test + criterion.** Per-level Spearman vs `n_ops`; significant iff |ρ| ≥ 0.886
  (N=6).
- **Result.** counting: winding ρ=**+0.943** (sig), steps ρ=+0.928 (sig).
  maxtask: winding ρ=**+0.943** (sig), steps ρ=−0.771 (n.s.). switch: winding
  ρ=−0.086 (n.s.), steps ρ=+0.783 (n.s.).
- **Verdict.** The two significant winding hits are **entirely inside the D10
  confound** (III.2 refuses the length control because rank-corr = 1.0), so they
  cannot be read as depth rather than length. And the "effect" isn't even
  consistent: switch's winding is flat, maxtask's steps_settle *decreases* with
  length.
- **Two more tasks in the same family, documented here so the taxonomy has no
  dangling members: `count_ones`** (count the 1s in a 0/1 string) and
  **`projection`** (successive shifts along basis vectors), both in
  `full_synthetic_experiments.csv`, both **D10-length-confounded** (rank-corr 1.0),
  swept over 8 `n_ops` levels. Their CSVs carry an additional geometry metric not
  stored for the other tasks, `mean_normed_accel` (how much the step direction and
  size keep changing from step to step). count_ones's
  `mean_normed_accel~n_ops = −0.976` (p=3e-5, N=8) is the *tightest* monotone
  relationship in the whole project — but, being length-confounded, it is
  uninterpretable as depth (it reappears in VI.2 only as a confounded lead).
  projection's is weak (−0.19, n.s.). Neither gets its own subsection because
  neither adds a length-clean datapoint.

### V.3 The length-matched dissociation (E3) and its non-replication (D13)

- **Setup.** `make_variants` produces two prompts with an **identical body** and
  only the *question* differing: `track` ("Final total?" — needs accumulation) vs
  `local` ("What was the last instruction?" — needs only the last step). Same
  token length by construction. Swept over 6 `n_ops` levels; run at **5 seeds**
  (60 rows) and again at **15 seeds** (180 rows).
- **Test + criterion.** Per-level Spearman vs `n_ops`, N=6, bar 0.886.
- **Result.** At 5 seeds, `local` winding~n_ops = **−0.943 (significant)**. At
  **15 seeds it drops to −0.543 (p=0.27, n.s.)** and flips sign in the per-row
  statistic.
- **Verdict.** A **small-N artifact**: a "significant" N=6 result that does not
  survive tripling the seeds. This is the clearest single demonstration of why N=6
  per-level results need replication, not just FDR (Part VI).

### V.4 Correctness — the result that reframes the winding "signals" (D17)

- **Setup.** `counting_accuracy` — 56 runs; the model's answer is obtained by
  **actual generation + a regex parse** of the output (not the logit lens), scored
  against the true sum, across `n_ops` levels 2…48.
- **Tests + results.**
  - Accuracy by level: **37.5% (n_ops=2), 25% (4), 0% (8), 0% (16), 12.5% (24),
    0% (32), 0% (48).** The model **fails** counting at exactly n_ops≥8 — the
    levels where V.2's winding "rises."
  - Does geometry separate correct from incorrect? Point-biserial correlations:
    `winding~correct` **r=−0.087, p=0.53 (n.s.)**; `steps_settle~correct`
    **r=−0.295, p=0.027 (significant, and negative)** — steps_settle is *higher*
    for **wrong** answers.
- **Verdict — the honest headline.** The V.2 winding correlations are a **length
  effect measured on trajectories that produce wrong answers**, and the geometry
  does not distinguish success from failure (if anything, more "effective compute"
  goes with *worse* answers). They cannot be presented as "geometry reflects
  reasoning."
- (A finer probe, `v6_correctness_probe`, per-unroll logit lens: on the 13
  single-digit-answer rows — the only ones a first-token check can honestly verify
  — the correct token is in the model's **top-5 for 13/13**, but is the strict #1
  for only 4/13, all of them the answer "0". So the model *shortlists* the count
  but a small-number bias usually wins the final pick; D12.)

### V.5 The "clean" H2 test: modular counting with length held constant (D15)

- **The design.** To break the D10 confound, `make_three_scale_modk_task` fixes
  the total token count by construction: it lays down `active_len` ones,
  `irrelevant_len` filler symbols `x`, and `neutral_len = total_len − active_len −
  irrelevant_len` zeros, **shuffled together into one sequence** (no prefix
  block), and asks for `active_len % modulus` — a **single-token** answer whenever
  `modulus ≤ 10` (the answer is then a single digit 0–9; the generator enforces
  `2 ≤ modulus ≤ 10`, and the runs use moduli {2, 5}), sidestepping the
  multi-digit-answer problem. Verified: tokenized `seq_len` is **exactly constant**
  (std = 0) across the whole sweep.
- **Setup + statistic.** Two runs: N=7 active_len levels (126 rows) and **N=15**
  (270 rows), each × 3 irrelevant lengths × 2 moduli × 3 seeds, `num_steps=64`.
  Per-modulus per-level Spearman, then **Fisher-combine** the two moduli (III.4).
- **Result — winding: null, and now powered.** N=15: mod=2 ρ=+0.043, mod=5
  ρ=−0.461; **Fisher-combined p = 0.266**. At N=15 the power curve (Part VI) gives
  real power to detect a moderate effect, so this is a **genuine absence**, not
  underpowered.
- **Result — steps_settle: a real, replicated, unexplained effect.** N=15: mod=2
  ρ=−0.549 (p=0.034), mod=5 ρ=−0.739 (p=0.0016); **Fisher-combined p = 0.0006** —
  an order of magnitude stronger than the N=7 run's p=0.015 (strengthening with N
  is what a real effect does). Direction: **more content to count → the path
  settles *faster***.
- **V.5b — a residual confound even here.** The three symbol counts always sum to
  the fixed total: `active + neutral + irrelevant = total_len`. So you cannot raise
  one while holding the other two fixed — adding one `irrelevant` (`x`) token at a
  fixed `active_len` necessarily *removes* one `neutral` (`0`) token. The 3-way
  regression's predictors are `active`, `irrelevant`, `modulus` (neutral is left
  out precisely because including all three plus an intercept would be perfectly
  collinear), so the estimated `irrelevant` coefficient (β=−0.21, p=4e-5) actually
  measures "swap a 0 for an x" — it conflates any filler effect with a
  neutral-token effect and cannot be read as a pure filler effect. Separately, the
  large `modulus` coefficient (β=+0.69, p=6e-29) almost surely reflects the
  question text literally changing ("modulo 2" vs "modulo 5"), not a difficulty
  effect. Milder than V.6 below, but real — even the clean task has one residual
  confound (D20).
- **Verdict.** On the best-controlled test, **H2 is not supported** (winding null
  at proper power). The one live finding is elsewhere: settling *speed* depends on
  task structure (VI.2).

### V.6 The original three-scale task and its prefix confound (D11)

The earlier version of V.5 placed the filler as a **prefix block**, so it
lengthened the context *and* pushed the answer token to a later absolute position.
Its headline ("winding tracks the irrelevant padding, β=−0.49, p=7e-11") is
therefore consistent with winding tracking **total length / answer position**, not
content — exactly the confound it was meant to remove. Not a clean H2 test; the
cautionary tale that motivated V.5.

### V.7 PARARULE-Plus — the only non-synthetic reasoning data (E1)

- **Setup.** Real logical-reasoning sentences with a controlled proof **depth
  d2–d5** (N=4 levels). Metrics vs depth. (There is a known data subtlety the team
  raised: "people" vs "animals" sentences differ in length within a depth, so
  length windows must be chosen carefully.)
- **Test + result.** Per-level Spearman: winding~depth = +0.200, steps~depth =
  **+0.800**. depth is itself correlated with length (rank-corr = 0.816).
- **Verdict.** **Structurally inconclusive.** At N=4 significance is *unreachable*
  (III.1: min possible p = 0.083), so steps~depth = +0.80 looks strong but proves
  nothing formally; the window concern is real but secondary to the N=4 wall.
  Status: "cannot be confirmed or refuted here," not "no signal."

### V.8 First real loops — external positive control (D14)

- **Setup.** Reproduce Blayney et al.'s "Long Persona" system prompt (a specific
  verbatim persona text known to induce orbits) on 24 **GSM8K** questions (a
  standard benchmark of grade-school math word problems), capturing
  **all token positions** (7,122 trajectories), vs a no-system-prompt baseline.
  Classify each with `classify_shape`.
- **Result.** Per-token loop rate **0.1286% (7/5445)** under Long Persona vs
  **0.0596% (1/1677)** baseline. The Long-Persona rate is close to Blayney's own
  independently-measured 0.14% — but it rests on only 7 events, so this is
  order-of-magnitude agreement, not a precise match; the baseline (1 event) is too
  small to compare at all. **Novel:** all 7 loops sit at relative position
  **0.819–0.897** (the last fifth of the prompt); zero in the first two-thirds
  (D20); the strongest lands on the token `" makes"` — a token Geiping et al. name
  as orbit-prone.
- **Verdict.** The pipeline's own shape detector finds real, non-starved loops at
  the same rate as an independent method — a **passing validation** that our
  `classify_shape` measures the same phenomenon the literature does. (Caveat: the
  per-*example* rate, 3/24 = 12.5%, runs ~5× above Blayney's 2.5%; a binomial test
  says this is not comfortably explained by N=24 noise (p=0.021), but
  distinguishing "different detector" from "real difference" needs the raw paths,
  which weren't saved. D14.)

---

## Part VI — Cross-cutting results and the rigor layer

### VI.1 Winding has no consistent relationship with difficulty (D16)

Assembling the per-level winding~difficulty ρ across all tasks:

| Task | ρ | N | length-clean? |
|---|---|---|---|
| counting | +0.943 (sig) | 6 | no (rank-corr 1.0) |
| maxtask | +0.943 (sig) | 6 | no |
| dissociation-track (5s) | +0.771 | 6 | matched control |
| pararule | +0.200 | 4 | partly |
| switch | −0.086 | 6 | no |
| three_scale | −0.400 | 5 | prefix-confounded |
| modk N=7 | +0.107 | 7 | **yes** |
| modk N=15 | −0.125 | 15 | **yes** |

No consistent sign; the only two significant *positive* hits are the
length-confounded pair (counting, maxtask); every genuinely length-clean test is
null-to-negative. (The table shows the accumulation-relevant `track` variant of
dissociation. Its counterpart `local` gives a significant *negative*
winding~n_ops = −0.943 at 5 seeds — but that is the length-*matched*,
non-replicating hit from V.3, not a positive and not a length confound; at 15
seeds it decays to −0.543, n.s. Either way it does not support H2.) Assembled this
way, the table is the **clearest cross-task summary of the case against H2** — it
shows the incoherence at a glance; the single strongest *controlled* test is
separately V.5 (the length-clean, adequately-powered null).

### VI.2 Steps_settle tracks answer structure, not difficulty per se (D18)

steps_settle~difficulty is **positive on every task whose answer accumulates**
(counting +0.93, switch +0.78, pararule +0.80, dissociation-track +0.81,
three_scale +1.00) and **negative on every task whose answer saturates or wraps**
(maxtask −0.77, modk both moduli, Fisher p=0.0006). **Why this is not merely
another length effect:** counting (accumulate) and maxtask (saturate) are *both*
D10-length-confounded — difficulty rises with length in both — yet their
steps_settle signs are *opposite* (+0.93 vs −0.77). A pure length effect would
give them the *same* sign, so the accumulate-vs-saturate distinction adds
something beyond length. And the cleanest single datapoint (modk, V.5) is at
*constant* length, so its negative sign cannot be length at all. A real,
systematic phenomenological pattern; mechanism unknown. Preconditions worth
stating: this is `steps_settle` (an internal-dynamics proxy, II.2), **not**
accuracy — V.4 shows steps_settle does not track correctness; and the individual
per-task ρ values are mostly at N≤7 (only modk-N15 is well-powered), so the
*pattern* is the claim, not any one task's significance. (A confounded but even
tighter-looking lead: count_ones `mean_normed_accel~n_ops = −0.976`, p=3e-5 — but
same D10 length confound, so uninterpretable on its own.)

### VI.3 Not all metrics are init-robust (D19)

The one dataset varying the random `h₀` seed (`dissoc_multiinit`, 5 init seeds ×
120 configs) lets us compare **within-config init-noise** against
**between-config signal** for each metric. Std ratios: **winding 0.50**
(most stable), **steps_settle 0.84** (borderline — noise nearly matches signal),
**contraction 1.72** (noise *exceeds* signal — init-dominated). Implication: trust
any contraction~difficulty correlation *less* than a winding one; and winding's
init-stability is double-edged (it could mean the geometry is an architecture
property, per Geiping's path-independence, not a computation signal — an open
concern).

### VI.4 Multiple-comparisons and power (D13, honesty gate)

- **BH-FDR** over all 46 correlation tests: 22 raw-significant, **20 survive**
  correction (identical under both family definitions). The two "prime target"
  anomalies (maxtask, dissociation-local) both survive FDR — but are invalidated
  anyway by the D10 confound and by non-replication (V.3). Lesson recorded:
  FDR-survival was the *smaller* problem here.
- **Power.** A Monte-Carlo (bivariate-normal generator; III.6) gives per-level
  Spearman at N=6 only **~0.66 power to detect a true ρ=0.9** and ~0.44 at ρ=0.8 —
  so N=6 misses a real strong effect ~⅓ of the time. This is the quantitative
  backing for treating N=6 hits cautiously and reading the N=15 modk null (V.5) as
  genuine. A pre-registered test family is frozen (`power_and_preregistration.md`).

---

## Part VII — H3 (contraction), separately

- **The theorem** (Banach fixed point): a strictly contracting recurrent map has a
  unique fixed point and **erases memory of its initial state `h₀`** exponentially,
  with a maximum distinguishable-history bound. The algebra is independently
  re-verified as correct. **Precision note on "contracting":** the theorem's
  contraction condition is a *Lipschitz* constant `c < 1` (‖R(x)−R(y)‖ ≤ c‖x−y‖),
  which for the linearized map is the largest **singular value** σ_max < 1 — this
  is *not* the same as the **spectral radius** ρ < 1 that H3 is stated with (0.1).
  Since ρ ≤ σ_max always, ρ < 1 is *necessary but not sufficient* for the theorem
  to apply; the quantity that actually controls contraction is σ_max (which is what
  the toy estimator, and any future real-Huginn measurement, should compute — see
  the code note at the end of this section).
- **The scope subtlety.** The theorem only proves *initial-state* memory decays,
  with the context held fixed. Huginn **re-injects the full prompt every unroll**,
  and for the counting task the count is fully readable from that re-injected
  context at step 0 — so contraction erasing `h₀`-memory need not forbid the count.
  It only bites if the count is encoded the *streamed* way (accumulated over steps,
  no re-injection).
- **The toy test (A6), on models we can run without a GPU.** Two small recurrent
  models trained on a synthetic count, swept from non-contracting to strongly
  contracting (a regularizer β drives the contraction factor down). The contraction
  axis reported below is the toy's own measured value: its code column is named
  `rho`, but it is populated by the same σ_max estimator described in the code note
  (JᵀJ power iteration), so these are really **σ_max** values, not true spectral
  radii. A recurrent-over-**depth** toy (re-injects the full input each step, like
  Huginn): a linear probe decodes the count with **R² ≥ 0.996 at every contraction
  strength tested**, *including* σ_max≈0.25. A recurrent-over-**time** toy (a
  streaming vanilla RNN — an Elman RNN, one input token per step, no re-injection):
  probe R² **cliffs to ~0** as soon as contraction forces σ_max<0.14. Same
  contraction, opposite outcome, decided entirely by the re-injection topology.
  (This mirrors the two ways a "recurrent net" can be built — iterating *depth* on
  a fixed input vs. stepping *through a sequence* — and only the latter matches the
  theorem's streaming assumption.)
- **Verdict.** The proof stands; its *applied* claim ("contraction ⇒ can't count ⇒
  must loop") is a property of streaming recurrence and **likely does not bind on
  Huginn's architecture.** Crucially, **no contraction quantity (neither σ_max nor
  ρ) has ever been measured on real Huginn** (only on the toy), so H3-on-Huginn is
  *scoped*, not tested. (A code
  note on *which* quantity the toy estimator actually computes, because two are
  easily confused: it estimates **σ_max**, the largest *singular value* of the
  Jacobian J, via **power iteration** — repeatedly applying JᵀJ to a vector and
  renormalizing, which converges to the top eigenvector of JᵀJ; this is the Miyato
  et al. 2018 spectral-normalization recipe. σ_max is *not* the same as the true
  **spectral radius ρ** (largest absolute *eigenvalue* of J), which Yang et al.'s
  direct power-iteration on J itself would compute. They coincide only for normal
  matrices; in general **σ_max ≥ ρ**, so σ_max *upper-bounds* the contraction factor
  — good enough to certify "contracts if σ_max<1," but it is not literally ρ. This
  distinction was a real error in the project's own README, corrected this session.)

---

## Part VIII — Consolidated limitations, assumptions, and uncertainty

1. **Answer-token-only tracing** — almost every scalar is measured at the final
   answer token, where settling is strongest and where the literature says loops
   *don't* live. No result characterizes "Huginn's geometry," only the answer
   token's. The all-token capture is a one-line change, unrun at scale.
2. **Winding is un-null-tested** — the matched-random-walk surrogate that would
   distinguish a real orbit from a PCA artifact is implemented but never run,
   because raw paths aren't saved. Every winding number is currently un-adjudicated
   against chance.
3. **Single init seed** — headline numbers fix `seed=0`; the one robustness check
   shows contraction is init-noise-dominated (VI.3).
4. **No contraction quantity measured on real Huginn** — neither σ_max nor the
   spectral radius ρ (H3's central quantity) has been computed on Huginn; both are
   toy-only.
5. **Small N everywhere but once** — N=4 is provably inconclusive, N=6 is
   underpowered (~0.66 power at a true Spearman ρ=0.9); only the N=15 modk run is
   adequately powered.
6. **Statistical-assumption caveats that actually bite:** the crit-value table
   breaks on ties (III.1, real case in V.5); Fisher's method assumes independence
   the shared model doesn't fully guarantee (III.4); BH's family boundary is a
   judgment call (III.5); the power numbers assume a normal generator (III.6);
   point-biserial's p is approximate at this N (III.6).
7. **Geometry vs. activations, untested** — the evidence that Huginn's *state*
   holds the answer information is the V.4 logit lens (the model's own pretrained
   output head — a *fixed* linear readout — puts the correct count in its top-5).
   Note this is a logit lens, **not** a *trained* linear probe on the activations
   (no such probe has been fit on real Huginn; the trained-probe result is on the
   toy model, VII). Either way, showing the *state* linearly decodes to the answer
   does **not** show the *geometric summary* (winding / shape / λ) is what carries
   it. The deepest conceptual gap: the project measures things about activations
   (or their pretrained readout) while claiming things about geometry — and it has
   never tested whether the geometric features *alone* predict anything above the
   raw state.
8. **One residual confound survives even in the clean task** (V.5b).
9. **Only one non-synthetic dataset used *as a reasoning-depth test*** (PARARULE,
   N=4). Of the proposal's four named datasets, only PARARULE is used that way;
   GSM8K appears only inside the loop-detection control (V.8), and ProntoQA-OOD and
   MultiLogicEval are not run at all. None of the four named baselines are run. The
   curator's own QK-alignment probe (Tulchinskii et al. — a query·key dot product
   with per-setup head calibration) is 0% implemented.
10. **Homology is degenerate** — persistent-homology H1 on a single 1-D curve is
    ~0 by construction (no independent 1-cycles); the sound population/delay-
    embedding version is not implemented, so H1's third named discriminator is
    effectively missing.

---

## Part IX — Honest bottom line

Across ~14 experiments on real Huginn-3.5B, the project has produced mostly
**rigorous negatives**. H1's shape taxonomy is real and the detector is validated
against an independent method, but at full budget on these tasks only "settle"
occurs. **H2 (winding grows with depth) is not supported**: every *statistically
significant* winding-vs-difficulty positive is a perfect length confound (the
non-confounded positives are all non-significant), and the one properly
length-controlled, adequately-powered test is null. H3's theorem is correct but its applied claim
likely does not bind on Huginn's re-injecting architecture, and its central
quantity was never measured on the real model. And critically, the model **fails**
the counting task at the depths where the geometry "signals" appear, with the
geometry not separating right from wrong answers — so those signals are not
geometry-of-reasoning. The most defensible contribution is the **audit itself**
(the confound catalogue, the non-replication check, the power analysis, the
pre-registration) plus one clean, replicated, still-unexplained observation:
settling *speed* tracks whether a task's answer accumulates or saturates.
