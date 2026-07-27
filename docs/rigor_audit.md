# Rigor audit — full project re-verification (2026-07-25)

STATUS: findings below are each reproduced from raw data, model source at the
pinned revision, or the code itself. Every number in this document was
recomputed during the audit; none is quoted from an earlier document.

SCOPE: the whole project — metrics and their hyperparameters, arbitrary
pre/post-processing choices, extraction settings, task text, tokenization and
correctness checking, statistical assumptions, file inventory, and the
*suppositions* connecting a measured quantity to the thing it is meant to
stand for.

The audit is organised by **severity of consequence**, not by file.

---

## 0. Summary table

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Compute dtype is bf16; the "limit set" is arithmetic noise | **Invalidates** | confirmed |
| 2 | `consecutive_step_cosine` is measured on the noise regime | **Invalidates** | confirmed |
| 3 | `steps_to_settle` measures ρ, not effective compute | **Invalidates** | confirmed |
| 4 | `full_synthetic_experiments.csv` ≠ `results/trajectories/*.npy` | **Invalidates** | confirmed |
| 5 | `init{N}` in trajectory filenames is the *task* seed | **Misleads** | confirmed |
| 6 | `path_independence` / `convergence_rate` fit through the floor | **Biases** | confirmed |
| 7 | H3's contraction premise holds: ρ ≈ 0.84 | **Corrects** | confirmed |
| 8 | Recurrent state is confined to a sphere (RMSNorm) | **Reframes** | confirmed |
| 9 | PCA is nondeterministic (no `random_state`) | Hygiene | confirmed |
| 10 | Accuracy uses a different compute budget than geometry | **Non-comparable** | confirmed |
| 11 | `configs/default.yaml` is stale and disagrees with every run | Misleads | confirmed |
| 12 | `ns=16` trajectory groups are not converged | Scoping | confirmed |
| 13 | Spearman ties: census done; cluster entirely on `steps_settle` | Scoping | confirmed |
| 14 | `h3_toy_model/` was unlinted, untested-in-CI, undocumented | **Process** | fixed |
| 15 | The v6 probe's apparent successes are a zero-prior | **Invalidates** | confirmed |
| 16 | No injected noise; `burn=4` discards 21% of the signal | Scoping | confirmed |
| 17 | **No rotational signal in winding.** Under an unbiased on-manifold null the effect flips sign with `num_steps` (+10.97 vs −5.58) in both tasks — it tracks the recording window, not the content | **Invalidates** | **resolved** |
| 18 | `gate.classify_shape` is degenerate — H1 was never tested | **Invalidates** | confirmed |
| 19 | `h1_persistence` normalises features by the transient extent | **Invalidates** | confirmed |
| 20 | Two citations misattributed (Pappone) or unlocatable (Movahedi) | **Attribution** | confirmed |
| 21 | The length confound is UNIVERSAL: rank-corr = 1.000 in all 5 generators | **Invalidates** | confirmed |

---

## 1. The compute dtype is bf16, and the "limit set" is arithmetic noise

`load_huginn(dtype=torch.bfloat16)` is the default (`extraction/model.py`), and
this is verifiable in the saved data: every stored value in
`results/trajectories/*.npy` is **exactly** bf16-representable — casting to
bfloat16 and back is bit-identical, max deviation `0.000e+00`.

Consequence. bf16 has 8 mantissa bits, so relative precision is `2^-8 ≈ 3.9e-3`,
not float32's `1.2e-7`. With per-coordinate RMS `1.0510`, one rounding event has
norm `≈ 0.086`.

**ARITHMETIC CORRECTED (second audit pass).** This section first reported a floor
of `η/(1−ρ) = 0.668` against an observed 0.911 — "within 1.4×", called a
parameter-free prediction. **That formula is wrong.** It sums *aligned*
displacements; isotropic rounding adds in **quadrature**. The variance recursion
is `E‖e‖² = ρ²E‖e‖² + η²`, giving a stationary RMS of `η/√(1−ρ²)`:

```
correct prediction  = 0.086 / sqrt(1 - 0.871^2) = 0.175
observed limit-set radius                         0.911     ratio 5.2
```

Verified by direct simulation of `e_{t+1} = ρ e_t + ξ_t` with isotropic ξ in 5280
dimensions: at ρ=0.871 the simulated stationary norm is **0.171** (formula 0.175;
the aligned-sum formula 0.667). **The 1.36× match is retracted** — it was a
coincidence of the wrong formula, and should never have been presented as a
prediction.

**The 5.2× residual is then explained, quantitatively (2026-07-26).** The
missing factor is that `η` is a *per-unroll* injection, not a single rounding
event. One unroll is the adapter plus four `SandwichBlock`s, each of which is
`norm_1 → attention (QKV proj, scores, softmax, out proj) → norm_2 → norm_3 →
GatedMLP (two projections, gate, out proj) → norm_4` — roughly ten rounded
operations per block, ~40 per unroll. Independent errors add in quadrature, so
`k` sites give `η ≈ √k · η₁`:

```
k sites    4      16      36      40      64
eta      0.172   0.344   0.517   0.545   0.689
floor    0.35    0.70    1.05    1.11    1.40      (at rho=0.871)
```

Inverting the observed floor of 0.911 gives **k ≈ 27** independent rounding
sites per unroll — squarely in the range the architecture implies. So the
corrected account is:

> floor = √k · η₁ / √(1−ρ²), with k ≈ 27–40 arithmetic sites per unroll

which reproduces the observation without free parameters beyond a site count
that can be read off the model definition. **This is a stronger result than the
original claim**, which got a plausible number from the wrong formula.

A secondary contribution, measured and real but not dominant: ρ is
mode-dependent. Fitting per-mode decay of the inter-orbit gap on its top eight
principal directions gives ρ ranging **0.73 to 1.02** across modes. Slow modes
do raise the floor, but the slowest reliably-estimated one implies only ~0.25,
so accumulated rounding is the main term.

**The reading "this is arithmetic" therefore rests on isotropy AND on a
now-quantitative magnitude match.** Two further checks:

- **Isotropy.** Tail PCA gives top-2 variance 8.8–10.7% with participation ratio
  18–19 out of 20 — variance spread nearly uniformly across components. Genuine
  low-dimensional dynamics would concentrate in 2 components.
- **Amplitude.** The multi-init convergence floor (§7) is 1.41 at ns=64, the same
  order.

**What this invalidates.** Any statistic computed on the post-settling segment.
That includes the "≈14-step oscillatory mode" found by DMD on the tail (it was
fitting a linear operator to isotropic rounding noise), the tail-restricted
windowed-winding numbers, and D22's characterisation of "residual jitter at the
fixed point" as a property of the model.

**Falsifiable prediction.** A single float32 rerun of any one prompt should shrink
the floor by `2^-16 ≈ 1.5e-5`, to radius `≈ 1.4e-5`. This is the cheapest
decisive experiment in the project and has not been run.

## 2. `consecutive_step_cosine` reports a property of rounding noise

`convergence.consecutive_step_cosine` averages the adjacent-step cosine over the
**second half** of the trajectory. Paths settle at t ≈ 14; for ns=128 the second
half is t = 64…127, which is entirely below the bf16 floor. Split by regime over
30 banked ns=128 trajectories:

```
converging regime  t = 0..14    mean cos = +0.0836
post-floor regime  t = 64..127  mean cos = -0.3313   <- what the metric reports
```

White-noise increments have lag-1 autocorrelation exactly −0.5, which is where
the post-floor number is heading.

**What this invalidates.** The project-wide reading "cos < 0 ⇒ the path zig-zags
inward" (sourced to Pappone et al. / Movahedi et al.). In the regime where the
model is actually computing, the cosine is **positive**: the path glides. The
`winding_variants` module's W7/W8/W9 were introduced specifically to handle
direction reversal, citing `cos = -0.276` as evidence that reversal is real in
this data. That evidence does not survive: it is arithmetic.

## 3. `steps_to_settle` measures the contraction rate, not effective compute

`dynamics.steps_to_settle(traj, frac=0.1)` returns the first step whose
displacement falls below `frac` of the **maximum** displacement. The maximum is
always the initial transient (the model forgetting the random `h_0`). For a path
decaying at rate ρ this threshold is crossed at a value fixed by ρ alone:

```
t* = log(frac)/log(rho) = log(0.1)/log(0.8407) = 13.27
observed mean over 140 banked trajectories   = 13.84  (std 1.98)
```

The prediction uses no task information and matches to within 0.6 steps.

**What this invalidates.** The supposition, stated in the function's own
docstring, that this is "our proxy for effective compute". It is a proxy for ρ.
This explains the metric's persistently tiny variance and why correlations
against `n_ops` are weak — there is almost nothing there to correlate. Any
claim resting on `steps_settle ~ difficulty` needs re-derivation.

This is the clearest instance of the failure mode the audit was asked to look
for: *X resembles class Y, so property Y2 is used to estimate Y1 — without
checking that the Y1↔Y2 relation holds for the actual X.*

## 4. The headline synthetic CSV does not correspond to the banked trajectories

`results/full_synthetic_experiments.csv` (80 rows) and
`results/trajectories/*.npy` (140 files) share task names, `n_ops` levels and
seed indices, and the row/file counts are consistent (2 tasks × 8 levels × 5
seeds; 2 × (8+6) × 5). They are nevertheless **different data**:

```
recomputed |winding| from the .npy files, matched against the CSV:
    vs ns=64 files :  0/80 rows match
    vs ns=128 files:  0/80 rows match

CSV    |winding|: mean 0.990  std 1.303  range [0.359, 8.442]
banked |winding|: mean 0.625  std 0.039  range [0.382, 0.675]

CSV steps_settle: 128 for every row (i.e. "never settled")
recomputed on banked ns=128: min 9, max 19, mean 13.7
```

Not one row reproduces, the distributions differ by a factor of 33 in spread,
and the settle column is categorically inconsistent.

**What this invalidates.** D22's null test was run on the `.npy` files, and its
conclusions were then applied to the project's winding claims — which rest on
the CSVs. That inference is not supported: the two are different datasets. D22's
caveat (a) called the banked trajectories' provenance "undocumented"; it is
stronger than that — they are demonstrably *not* the data behind the CSV.

## 5. `init{N}` in trajectory filenames is the task seed, not the init seed

`make_count_ones_task(n_ops, seed)` uses `seed` to generate the bit string, so
the seed changes the **prompt**. `full_synthetic_experiments.csv` confirms it by
storing prompts directly:

```
seed 0: "Sequence: 1 1. ..."  answer 2
seed 1: "Sequence: 0 0. ..."  answer 0
seed 2: "Sequence: 0 0. ..."  answer 0
```

Seeds 1 and 2 collide to the same prompt, which gives a direct test:
`count_ones_n2_ns128_init1.npy` and `..._init2.npy` are **bit-identical**
(max abs diff `0.000e+00`). Same prompt ⇒ same trajectory ⇒ `h_0` is *fixed*.

**Consequence.** `results/trajectories/` contains **no** multi-initialization
data — every file is a distinct prompt (one collision pair aside). It cannot be
used to measure path-independence, Lyapunov exponents, or map contractivity.
Genuine multi-init raw paths do exist, in the top-level `trajectories/` directory
with `manifest.csv` (§7).

The name is actively misleading and cost this audit one wrong conclusion before
the collision test caught it.

## 6. Two convergence metrics fit through the noise floor

`convergence.path_independence` and `convergence.convergence_rate` both fit a
single log-linear slope over the **entire** trajectory. Because the gap decays
exponentially and then plateaus at the bf16 floor, the plateau drags the fitted
slope toward zero:

```
                      whole-range fit   early-region fit (first 30%)
mean over 5 groups        x0.9244            x0.8407
systematic bias        +0.0837 per step understated contraction
```

`convergence_rate` additionally references `traj[-1]` — a single state that is
itself one noise sample — rather than a tail mean.

**Consequence.** All 600 `lyap` values in `results/dissoc_multiinit.csv` and all
`conv_rate` values in `results/convergence.csv` are biased toward "less
contracting" by roughly 0.08/step.

## 7. H3's contraction premise holds — ρ ≈ 0.84

Measured correctly, from the convergence of two orbits with the **same prompt and
different `h_0`**, using the raw multi-init paths in top-level `trajectories/`:

```
 kind  n_ops  ns    gap0    gapT   whole   early   floor
local     48  64  53.393  1.434  0.9499  0.8960  1.377
track      4  64  50.744  1.179  0.9533  0.8487  1.259
track     24  16  52.170  6.512  0.8762  0.7884  9.083   <- not converged
track     48  16  50.554  7.072  0.8900  0.7749  9.990   <- not converged
track     48  64  50.554  1.376  0.9527  0.8957  1.602
```

Converged (ns=64) groups give ρ ≈ 0.85–0.90; the residual gap floor is 1.41. Against the CORRECTED bf16 prediction (0.175 at
this ρ, see §1) that is a factor of 8, consistent with the floor being set by
the slowest-contracting modes rather than the aggregate rate.

**Conclusions.** (a) The map contracts: ρ < 1, so H3's premise is satisfied on
real Huginn. (b) Different initializations converge to the same state up to
arithmetic precision, which **confirms** Geiping et al.'s path-independence
claim. (c) Contraction shows no relationship with difficulty
(rank-corr(n_ops, ρ) = +0.224, N=5 groups — far below any critical value).

Three earlier numbers must be retired: `0.871` (measured as distance to a
trajectory's *own* endpoint — near-tautological, measures sequence convergence
not map contractivity); `1.000` (measured across *different prompts* — the wrong
data, see §5); and `0.9244` (whole-range fit through the floor, §6). Only the
early-region two-orbit figure is valid.

**Scoping note for H3's applied claim.** Contraction forbids an *unbounded*
register. It does not forbid a *bounded* counter: the converged state retains a
neighbourhood of diameter ~1.4 across 5279 tangent dimensions, ample room to
separate 64 states. This matters directly for Barannikov's Task a/b, which ask
about counters bounded by m = 64.

## 8. The recurrent state is confined to a sphere

`SandwichBlock.forward` (pinned revision, line 535) ends with
`x = self.norm_4(...)`, an RMSNorm, and the extraction hook attaches to
`core_block[-1]`. Every recorded state is therefore an RMSNorm output:

```
||h_t|| = 76.37 +- 0.007   (0.0073% within-trajectory variation, 20x128 states)
||h_t|| / sqrt(n_embd) = 1.0511
```

Consequences: (a) the radial coordinate carries no information, so no quantity
can be encoded in the state's magnitude; (b) the Euclidean centroid used as
`winding_of`'s rotation center lies **off the manifold** (‖centroid‖ = 74.20 vs
radius 76.37, 2.8% inside); (c) `metrics.winding.matched_random_walk` — the null
model — leaves the manifold entirely, diffusing to ‖h‖ = 154.76, a 5.86%
deviation against the real data's 0.0073%. Every null test in the project has
compared an on-manifold path against off-manifold surrogates.

In geodesic terms the transient sweeps 69.7° of arc while the converged
neighbourhood spans 0.63°, so tangent-space (locally Euclidean) analysis is sound
near convergence and unsound across the transient.

## 9. PCA is nondeterministic — hygiene, not a correctness bug

No call site passes `random_state` (`winding.py`, `winding_variants.py` ×2,
`projection.py`, `plot_trajectories.py`). For a [128, 5280] array sklearn's
`svd_solver='auto'` selects randomized SVD, so results vary run to run.

Measured severity: relative spread over 15 repeated calls on identical input is
**≤ 7.4e-7**, and the PC2/PC3 relative eigenvalue gap is ≥ 0.33 for **all 140**
trajectories (none below 5%), so the projection plane never flips. This explains
the three distinct winding values recorded for bit-identical inputs in
`full_synthetic_experiments.csv` (5.588866578 / 5.588866390 / 5.588865752).

`svd_solver="full"` is exact, deterministic, and costs 45 ms on this shape.

**Sign handling is correct**: every call site wraps `abs(winding_of(...))`, with
an explicit comment that the PCA sign is a convention. No bug there.

## 10. Accuracy and geometry are measured under different compute budgets

`eval.counting_correct(..., num_steps=32)` calls
`model.generate_with_adaptive_compute`, i.e. accuracy is measured at a cap of 32
unrolls **under Huginn's adaptive exit criterion**. Geometry is measured at fixed
`num_steps` ∈ {16, 64, 128} with no early exit.

**Consequence.** The correctness figures (10.7% overall; 0% at n_ops ∈
{8,16,32,48}, 12.5% at 24) and the geometry figures are not directly comparable,
and any statement of the form "the model fails yet the geometry shows X" is
comparing two different compute regimes. This is separate from, and additional
to, the already-documented latent-vs-readout distinction.

Answer parsing (`re.findall(r"-?\d+", txt)`, first match, `max_new_tokens=8`) has
no output-format priming and takes the first integer anywhere in the
continuation.

## 11. `configs/default.yaml` is stale and contradicts every actual run

The file is referenced only as an argparse default in `scripts/extract.py` and
`scripts/run_mvp.py`, both of which are unimplemented stubs (`extract.py` body is
a `TODO(...)` string). Nothing parses it. It nevertheless specifies
`num_steps: 32`, `dataset: pararule-plus`, `max_examples: 200` and
`token_selection.strategy: answer_token` — values that match no experiment in
`results/`. A reader would reasonably take it as the run configuration.

## 12. `ns=16` trajectory groups are not converged

Two of the five multi-init groups (`track_n24_ns16`, `track_n48_ns16`) still have
an inter-orbit gap of 6.5–7.1 at the final step, versus 1.2–1.6 for the ns=64
groups. Any converged-state metric computed on them is measuring a transient.
`results/convergence.csv` includes these rows.

## 13. Spearman ties — already documented; now censused

`correlate.py:48–59` already carries a thorough caveat that
`_SPEARMAN_CRIT_P05` assumes no ties, complete with a real counterexample
(N=7, |rho|=0.775 reads "n.s." against the table while scipy's own p is 0.041).
This audit adds only the missing empirical part: **where** ties actually occur.

Checking every (x, y) pair across `results/*.csv`, ties appear in exactly six
places and **all six are `steps_settle`** — `counting`, `counting_accuracy`,
`dissoc_multiinit`, `switch` (1 tied level each), `forceloop` (2 of 4), and
`full_synthetic_experiments` (7 of 8). No `winding`, `contraction`, `lyap` or
`conv_rate` correlation is affected.

The `full_synthetic_experiments` case is not a tie but a degeneracy:
`steps_settle` is the constant 128 at every level, so rho is undefined (scipy
returns NaN with `ConstantInputWarning`). No correlation should be reported
from it at all.

That ties cluster entirely on `steps_settle` is exactly what §3 predicts: a
metric fixed by the contraction rate has almost no spread between levels.

## 14. `h3_toy_model/` was outside every quality gate

The directory holds the **A6** finding, which is cited twice in
`claims_ledger.md` as evidence bearing on H3. It was in neither
`pyproject.toml`'s `testpaths` nor ruff's `src`, so its 5 tests were never
collected by CI and its code was never linted — while a load-bearing claim
rested on it. It was also absent from `docs/architecture_state.md`, which is
the one document a `test_architecture_consistency` test enforces completeness
against (that test covers `scripts/run_*.py` and `results/*.csv`, not
top-level packages).

**Fixed**: `testpaths` now includes it and all 5 tests pass; it is documented
in `architecture_state.md`. It remains **linted but not clean** (~60 style
errors against 0 in `src/`) — scoped, known debt rather than a silent gap.

## 15. The correctness probe's apparent successes are a zero-prior

`results/v6_correctness_probe.csv` is the project's only accuracy-versus-depth
data. Its own docstring correctly restricts valid scoring to the 13
single-token-answer rows. Within those rows:

```
                 ever correct
target == 0        4 / 5
target != 0        0 / 8
```

and all four successes occur at unroll **1**. Correctness is perfectly
separated by whether the answer happens to be zero, and it appears
immediately rather than after any amount of computation. That is a prior
toward emitting "0", not a counting result — so the project's reported
accuracy figures overstate even the ~10% they claim.

This does not refute the depth-threshold hypothesis; it shows the existing
instrument cannot test it. Binary first-token argmax has no resolution for
locating a threshold, and no control against answer-string priors.

**Addressed** by `src/traj_geom/eval_depth.py` and
`scripts/run_depth_threshold.py` (implemented, NOT yet run — needs GPU):
teacher-forced `log P(gold) − log P(distractor)` per unroll, which is
continuous, exact for multi-token and negative answers, cancels answer-string
priors via the distractor, and excludes `answer == 0` outright.

## 16. Noise sources and the burn-in, checked

Two remaining "arbitrary choice" questions, both resolved:

**Injected noise: none.** `RavenConfig` sets `test_time_noise = 0` by default
and nothing in this project overrides it, so no stochastic perturbation is
added at inference. Combined with §5 (h_0 fixed across the banked files) and
§1 (bf16 confirmed), **floating-point rounding is the only noise source in the
system.** That is what makes the arithmetic-floor prediction in §1 a genuine
prediction rather than one term among several.

**`burn=4` costs more than it looks.** `winding_of` drops the first 4 states
to skip the h_0 transient. But the converging regime is only 19.2 steps long
on average (min 12, max 35), so the burn discards **21% of the signal** — and
what remains is ~15 informative steps embedded in the ~109 post-floor steps
that the metric also averages over. Stated plainly:

> the project's central metric computes a rotation statistic from about 15
> informative steps, diluted by roughly 109 steps of rounding noise.

The burn-in is not wrong in principle — the h_0 arc really is an outlier — but
at 4 steps out of 19 it is a large fraction of a short signal, and it was
chosen when the trajectories were believed to be informative throughout.

## 17. The winding null REVERSES under a manifold-respecting surrogate

§8 established that `matched_random_walk` leaves the state manifold. Replacing
it (`metrics/surrogate.py`) with a surrogate that preserves the sphere, the
radial convergence profile and every consecutive angular step — randomising
only the rotational degrees of freedom — reverses the project's headline
winding result.

The construction is exact rather than approximate. Writing each state as
`h_t = R(p cos φ_t + q_t sin φ_t)` about the converged pole `p` splits the
trajectory into a radial part (`φ_t`) and a rotational part (`q_t`). Holding
`φ_t` fixed preserves the convergence profile, and the spherical law of cosines

```
cos ψ_t = cos φ_t cos φ_{t+1} + sin φ_t sin φ_{t+1} (q_t · q_{t+1})
```

then fixes `q_t · q_{t+1}` exactly, leaving only the orthogonal direction free —
which is precisely the degree of freedom under test. Invariants verified to
machine precision and **uniformly across dimension 8 → 5280** (no scaling
pathology): norms to 1e-14, radial profile to 7e-6, angular steps to 1e-8.

```
47 real trajectories x 40 surrogates, |winding|

OLD null (ambient random walk, leaves the sphere):
   mean z = -9.64   median z = -5.78   beats null  4/47 =  8.5%
NEW null (on-manifold, radial profile + step sizes preserved):
   mean z = +5.86   median z = +3.61   beats null 31/47 = 66.0%
```

**Calibration.** A tighter null inflates z, so the procedure was calibrated by
feeding a surrogate back in as though it were data. An unbiased procedure must
beat its own null about 5% of the time:

```
REAL trajectories beat their null : 11/20 = 55%   mean effect +0.00311
SURROGATES beat their own null    :  0/20 =  0%   mean effect +0.00049
```

0/20 is consistent with 5% (P = 0.95²⁰ = 0.36). The excess is therefore not an
artifact of the construction.

### RESOLVED at full power (2026-07-26) — see below. The retraction stood.

`scripts/run_manifold_null.py`, 140 trajectories × 100 surrogates, with a
calibration arm and stratification:

```
CALIBRATION arm: beats null  6.4%   mean effect +0.00008   mean z +0.03
REAL arm:        beats null 62.1%   mean effect +0.00020   mean z +1.51

stratified by num_steps:
  ns=64    obs-null = +0.0127    mean z = +10.97   (n=60)
  ns=128   obs-null = -0.0091    mean z =  -5.58   (n=80)

by (task, num_steps):
  count_ones  ns=64  +0.0149 (z +13.35)   ns=128  -0.0150 (z -8.92)
  projection  ns=64  +0.0104 (z  +8.59)   ns=128  -0.0032 (z -2.24)
```

**The construction is unbiased** — the calibration arm beats its own null at
6.4% against a nominal 5%, with mean z +0.03 and mean effect +7.6e-5. So the
real arm can be interpreted.

**And the effect has no consistent sign.** It is positive at num_steps=64 and
negative at num_steps=128, in *both* tasks, with Mann-Whitney p = 1e-11 between
the strata. The pooled "62.1% beat the null" is an average over two strata that
cancel, which is why the pooled effect is +0.0002 — essentially zero.

`num_steps` is a **recording-budget parameter**. The same prompts and the same
model, recorded for longer, flip the sign of the comparison. What differs
between the strata is how much of the arithmetic-noise regime is included:
ns=64 carries ~50 post-floor steps, ns=128 carries ~110.

> **VERDICT. There is no rotational signal in winding. Its comparison to a null
> is governed by how much noise regime was recorded, not by the trajectory's
> content.** D22's result is deleted (its null was invalid), and nothing
> replaces it. This is the cleanest available demonstration of the audit's
> central theme: a statistic averaged over a window that mixes signal with
> arithmetic reports the window.

The retracted intermediate claims are kept below for the record.

**!! THE FOLLOWING NUMBERS ARE UNDERPOWERED AND THEIR CONCLUSION IS RETRACTED.**
A follow-up check on 8 trajectories restricted to num_steps=128 gave a mean
excess of **−0.0087** — the OPPOSITE SIGN to the +0.0031 above. The two samples
(47×40 and 20×40, neither stratified by `num_steps`; then 8×12) are all too
small, and the earlier run pooled ns=64 and ns=128, which §12 already showed
behave differently. **The sign of the effect is currently unknown.**

`scripts/run_manifold_null.py` runs all 140 trajectories × 100 surrogates with
a built-in calibration arm and stratification by `num_steps` and task. Nothing
about the direction of the winding effect should be cited until it completes.

**What DOES survive from this section**, independent of the sign:

- The *method* claim: `matched_random_walk` is off-manifold (‖h‖ 76.37 →
  154.76) and cannot support any conclusion, so **D22's headline result rests
  on an invalid null regardless of what replaces it.** That is a deletion of
  evidence, not a reversal of it.
- The construction and its verification: invariants exact to 1e-14 / 7e-6 /
  1e-8, uniformly across dimension 8 → 5280.
- The anisotropy sensitivity (below), which is a property of the estimator and
  not of any particular sample.

**Anisotropy, quantified.** Winding depends on the effective dimensionality of
the motion, and the real azimuths are more concentrated than isotropic
surrogates (participation ratio ~40 against ~46). Sweeping the surrogate's
subspace dimension shows |winding| rises steeply with PR at low PR and then
**saturates**:

```
subspace_dim     2     4     8    16    32    64   128   none
mean PR        2.0   3.8   7.1  12.3  19.6  27.6  34.8   46.3
mean |winding| .560  .567  .595  .631  .633  .645  .647  .646
```

A global linear fit through this gives d|winding|/dPR = +0.0026, but that is an
artifact of extrapolating through a saturating curve — the honest local slope
above PR≈15 is **+0.0006**. This matters methodologically: the real data sits
in the saturated regime, so moderate PR differences move winding very little,
and any future anisotropy correction must use the local slope.

**What it does NOT yet mean, and the control that must come next.** The effect
is small in absolute terms (+0.003 against a base of ~0.64, i.e. ~0.5%), and
this null draws the azimuth directions isotropically. It therefore does not
control for **tangent-space anisotropy**: if the recurrent Jacobian confines
motion to a low-dimensional subspace, that concentration alone can produce
apparent rotation with no rotational generator. This is exactly the failure
mode Elsayed & Cunningham (*Nat. Neurosci.* 20:1310, 2017) demonstrated for
jPCA rotations in neural population data, using Tensor Maximum Entropy
surrogates that preserve the mean and covariance across each tensor mode. Their
result was that apparent rotation was fully reproduced by surrogates matching
only those marginal covariances. **Until a covariance-preserving null is run,
"real trajectories wind more than chance" should be read as "more than an
isotropic on-manifold null", not as evidence of a rotational generator.**

Also worth noting: the strongest available null needs no synthesis. The 140
banked real trajectories already lie on the manifold with every structure
intact, so a permutation test across prompts tests "does winding vary with
difficulty" assumption-free.

## 18. H1's instrument is degenerate — the hypothesis was never tested

`shapes.gate.classify_shape` decides "settle" by asking whether the LAST step is
below 10% of the LARGEST step. The largest is always the h_0 transient (~72) and
the last is the bf16 floor (~1), so the ratio is structurally tiny:

```
s[-1]/s.max() over 140 banked trajectories: mean 0.0155, max 0.0188  (threshold 0.10)
classify_shape output over those 140:       {'settle': 140}
```

The maximum across all 140 is five times below the threshold. **"loop" and
"drift" are unreachable.** An instrument with one attainable output cannot test a
three-way hypothesis, so the project's H1 conclusion — "all trajectories settle,
no loops" — is a property of the metric, not of Huginn.

Its fallback branch is no better. It calls a path a loop when the smallest
far-pair distance is under 25% of the cloud diameter; the noise ball supplies
distances at ~0.012 of the diameter, so any non-settling path would be labelled
"loop" for the same structural reason.

And that branch is wrong independently of the noise ball: the minimum far-pair
distance scales like `min_lag × spacing`, which shrinks as T grows, so a long
enough path trips the threshold whatever its shape. Verified — `classify_shape`
labels a straight 90-point line, a random walk, and a decaying spiral all
**"loop"**.

**Fixed** by `regime.classify_shape_regime`, which (a) decides settling against
the noise floor rather than the maximum, and (b) uses the **chord-to-arc ratio**
`‖x_i − x_j‖ / arclength(i..j)`, which is exactly 1.0 for a straight line, tends
to 0 for a closed orbit, and is invariant to sampling density. Validated on
constructed cases with known answers:

```                      NEW      OLD
closed circle          loop     loop
straight line         drift     loop   <- old is wrong
random walk           drift     loop   <- old is wrong
decaying spiral      settle     loop   <- old is wrong
decay to a point     settle   settle
```

On the 140 banked trajectories the new classifier also returns "settle" — but
now because they demonstrably reach their floor within budget, a statement that
could have come out otherwise.

## 19. `h1_persistence` normalises features by the transient extent

`metrics.homology.h1_persistence` runs Vietoris–Rips on the whole trajectory and
divides the longest bar by the whole cloud's diameter. On real trajectories:

```
signal points 19-21   noise-ball points 107-109  (84-85% of the cloud)
cloud diameter ~93 (set by the transient)   noise-ball diameter 9-15
```

Any H1 feature lives in the noise ball while the normaliser is the transient
extent, so the score is approximately (noise-ball scale)/(transient extent) —
the right order for the project's reported max persistence of **0.009**, and not
a topological property of the computation. An open arc has trivial H1 by
construction in any case, so that number was never evidence of anything.

**Partially addressed** by `regime.h1_persistence_signal`, which restricts to the
converging regime so the normalisation refers to the points the features come
from. This does not rescue the method: 19–21 points in 5280 dimensions is a
generic simplex. The literature's answer (Perea & Harer, *Found. Comput. Math.*
15:799, 2015) is to delay-embed a **scalar observable** and score H1 on that —
delay-embedding the full state reuses the same points and creates no new
information. Not implemented.

## 20. Two literature attributions do not hold

Flagged by an external primary-source survey and consistent with this audit's
own measurements:

- **Pappone et al. (arXiv:2509.23314)** is a real paper but studies a
  **GPT-2-scale model, not Huginn**; defines the drift-to-loop ratio as
  cross-block-drift ÷ within-loop-refinement, **not** `‖h_T − h_0‖ / mean step`
  as implemented in `convergence.drift_to_loop_ratio`; and reports a
  consecutive-step cosine settling at **+0.5 to +0.65 — positive**. That is the
  opposite sign to the −0.276 this project attributed to it, and it independently
  corroborates §2's finding that the real converging-regime cosine is positive.
- **"Movahedi et al."**, cited alongside Pappone in
  `convergence.consecutive_step_cosine`, **could not be located**. It should be
  treated as an unverified citation until someone produces the reference.

Both are cited in code docstrings as justification for the "path zig-zags
inward" reading, which §2 already showed to be a measurement artifact. The
attributions should be corrected or removed before anything paper-facing.

## 21. The length confound is universal, and three generators have answer-format defects

`claims_ledger` D10/D21 document rank-corr(n_ops, seq_len) = 1.0 for the
counting and maxtask designs. Tokenizing every generator's prompts at the pinned
tokenizer shows it is not specific to those two — it holds for **every**
synthetic task in the project:

```
generator                  rank-corr(n_ops, seq_len)   answers at n_ops=2,4,8,16,32
make_count_ones_task              +1.000               [2, 3, 7, 10, 19]
make_counting_task                +1.000               [-2, -2, -6, -4, -6]
make_max_task                     +1.000               [7, 7, 9, 9, 9]
make_projection_task              +1.000               [0, 0, 6, 4, 13]
make_switch_task                  +1.000               ['off','on','on','off','on']
```

Exactly 1.000 in all five. **Partial correlation on length is therefore
mathematically degenerate for every task in the project**, not just for the two
where it was documented. No existing synthetic result can separate a
difficulty effect from a prompt-length effect.

Three further defects, each specific to a generator:

- **`make_max_task` saturates.** A running maximum over a larger set converges
  to the alphabet's top value: answers are 7, 7, 9, 9, 9. Beyond n_ops≈8 the
  answer stops varying with difficulty, so the target itself carries almost no
  signal — a ceiling effect independent of the length confound.
- **`make_counting_task` produces negative, multi-token answers throughout**
  (5/5 negative, 5/5 multi-token here). This is precisely the case where a
  first-token argmax check inspects only the "−" sign, shared by every negative
  value — the defect `run_v6_correctness_probe.py` documents. For this
  generator it is systematic, not occasional.
- **`make_projection_task` yields answer 0 in 2 of 5 sampled levels.** §15
  showed correctness is perfectly separated by target==0 (a zero-prior), so a
  substantial fraction of this task's instances are uninformative about
  computation.

**Bearing on Barannikov's Task a/b.** His fixed m=64 removes the length
confound at its source — with the string length fixed, difficulty and token
count are decoupled by construction — and his per-position target `y_i` removes
the answer-token degeneracy. The evidence that this is the right design is now
universal across the project's task set rather than anecdotal from one task.

## 22. Sensitivity of the audit's OWN thresholds

The audit criticised the project's arbitrary constants, so its own were put
through the same check. `regime.py` introduces `k` (floor multiple defining
signal) and `tail_frac` (fraction of the path used to estimate the floor);
`classify_shape_regime` adds `loop_frac`.

**R2 — the signal-regime cosine is positive** (reported +0.084 at k=3):

```
k       1.5     2.0     3.0     5.0    10.0
cos   +0.146  +0.149  +0.124  +0.082  +0.006      sign holds at every k
```

`tail_frac` ∈ {0.15, 0.25, 0.40} changes the result by less than 1e-4 — the
floor estimate is robust because the tail is long and flat. The sign never
flips. The magnitude does decay toward zero as `k` grows, which is expected and
worth stating: larger `k` restricts to the earliest, steepest part of the
transient, and the positive alignment is strongest mid-transient.

**E4 — ρ from two-orbit convergence** (reported 0.84–0.90):

```
k         1.5     2.0     3.0     5.0    10.0
mean rho  0.894   0.886   0.884   0.866   0.844
```

Stable across a factor of ~7 in the threshold.

Both conclusions are therefore threshold-stable in the sense the audit demanded
of the original metrics — unlike `steps_to_settle`, whose value is *determined*
by its threshold (§3), or `classify_shape`, whose output cannot change at all
(§18).

---

## Fixes shipped in this audit

| Finding | Fix | Verification |
|---|---|---|
| 1, 2, 3, 6 | `src/traj_geom/metrics/regime.py` — empirical floor detection plus floor-aware `contraction_from_pair` and `step_cosine_converging` | 9 synthetic tests; on real data cosine **−0.3225 → +0.1390**, contraction bias **−0.068/step** removed |
| 4, 5 | `src/traj_geom/provenance.py` + `scripts/backfill_provenance.py` — sha256 sidecars, append-only `manifest.jsonl`, shared `run_id`, `task_seed`/`init_seed` as separate required fields | 8 tests; **155/155** existing artifacts backfilled, all hashes verify |
| 9 | `svd_solver="full"` at all 4 PCA sites | `winding_of` now bit-identical across repeated calls |
| 14 | `h3_toy_model` added to `testpaths`; documented in `architecture_state.md` | 5 previously-uncollected tests now run |
| 15 | `eval_depth.py` + `run_depth_threshold.py`; answer probability MARGINALISED over surface forms (both `' 2'`=[402] and `'2'`=[50] exist as distinct tokens, so P(answer) is a sum over disjoint continuations) | 23 tests incl. against the real cached tokenizer; **experiment not yet run** |
| 17 | `metrics/surrogate.py` — on-manifold surrogate preserving sphere, radial profile and every angular step | 14 tests; invariants exact to 1e-14/7e-6/1e-8 across dim 8→5280; **self-calibration 0/20 vs 5% expected** |
| 18 | `regime.classify_shape_regime` — floor-relative settling + chord-to-arc loop test | discriminates all 3 labels; fixes 3 wrong 'loop' calls the original makes |
| 19 | `regime.h1_persistence_signal` — restricted to the signal regime | partial: T≪n remains fatal for full-state PH |
| 20 | Corrected the Pappone/Movahedi attribution in `convergence.py` | — |
| 2, 3, 6, 11 | Corrected docstrings in `dynamics.py`, `convergence.py` (×3), `winding_variants.py`, `configs/default.yaml` | — |

Existing metrics were left **bit-compatible** rather than silently corrected,
so every cached CSV stays reproducible; their bias is documented in place and
the corrected estimator offered alongside.

Suite: **115 passed**. Lint clean on every file this audit created or edited.

## Not fixed — deliberately left open

- **The fp32 rerun** (§1). One float32 run of a single prompt would confirm or
  refute the arithmetic-floor account outright. Needs GPU. Highest
  value-per-cost item in the project.
- **Re-running the nulls with a sphere-constrained surrogate** (§8). Every
  existing null compares on-manifold against off-manifold paths, so all nine
  winding variants' z-scores need redoing before any is cited.
- **`run_depth_threshold.py`** (§15). Implemented, needs GPU.
- **The `full_synthetic_experiments.csv` provenance question** (§4). Now
  pinned by a test, but which run produced it is still unknown.
- **`h3_toy_model` lint debt** (§14) and `analysis/plots.py`'s 9 errors.

---

## Cross-cutting: what remains valid

Not everything is affected. These survive the audit unchanged:

- **The transient regime (t ≲ 15) is real signal**, far above the noise floor,
  and is where all genuine dynamics live. It has been systematically
  under-analysed because the deployed metrics average over the full path.
- **Prompt information persists.** Trajectories from *different prompts* remain
  43.1 units apart (33° of arc), flat to four decimal places over 120 steps. The
  state does not collapse to a prompt-independent point. Combined with §7, the
  correct picture is: same prompt + different `h_0` → converge; different
  prompts → stay separated.
- **Sign handling, the N=4 Spearman fix, and the length-confound documentation**
  are all correct as written.
- **H3's premise** (§7) and **path-independence** are confirmed, not refuted.

## Cross-cutting: the recurring failure mode

Findings 1, 2, 3 and 6 are the same error in four places: **a statistic averaged
over a window that mixes the signal regime with the noise regime.** The
trajectories have a ~15-step informative segment followed by 50–115 steps of
arithmetic noise, and metrics that average over "the whole path" or "the second
half" are dominated by the noise. The fix is uniform — restrict every
trajectory-level statistic to the converging regime, and report the floor
separately rather than fitting through it.
