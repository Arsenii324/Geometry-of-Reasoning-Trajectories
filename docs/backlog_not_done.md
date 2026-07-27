# Things we could do but didn't

STATUS: living document, started 2026-07-25 during the rigor audit.
PURPOSE: make the *unexplored* surface of this project explicit, so that a
reader can tell the difference between "we checked and it's fine", "we checked
and it's broken", and "nobody looked". The third category is invisible in a
results write-up unless it is written down deliberately.

CONVENTIONS
- **Cost**: `free` (CPU, minutes) / `cheap` (CPU, hours) / `GPU` / `hard`
  (needs new derivation or design work).
- **Value**: how much a claim in `claims_ledger.md` would move.
- Items are NOT ordered by priority within a section; the priority shortlist is
  at the end.
- Deliberately includes low-value and fiddly items. Omitting them would defeat
  the purpose — the point is coverage, not a curated to-do list.

---

## 1. Experiments that need a GPU

| # | Item | Cost | Value | Note |
|---|---|---|---|---|
| 1.1 | **float32 rerun of one prompt.** Predicts the noise floor shrinks by 2⁻¹⁶ to ≈1.4e-5. | GPU, minutes | **Very high** | Single decisive test of the whole bf16-floor account (`rigor_audit` §1). Cheapest high-value item in the project. |
| 1.2 | **`run_depth_threshold.py`** — teacher-forced log P(gold)−log P(distractor) per unroll. | GPU, ~1h | **Very high** | H2 restated behaviourally. Implemented and tested; never run. |
| 1.3 | **Arnoldi on Jacobian-vector products** at the converged state: exact eigenvalues of `DF_e(h*)`. Gives ρ (H3) and arg λ (H2 rotation) from one object. | GPU, ~1 min/prompt | **Very high** | ~30 matvecs. `scipy.sparse.linalg.eigs` on a `LinearOperator` wrapping a JVP. Never attempted. |
| 1.4 | Multi-init raw paths for the *current* task set. Only 15 such files exist, from an older vintage. | GPU, ~1h | High | Everything about contraction/path-independence rests on those 15. |
| 1.5 | Re-run the synthetic sweeps saving raw paths **with provenance**, so CSV and `.npy` are linked by `run_id`. | GPU, ~2h | High | Closes `rigor_audit` §4 at the source. |
| 1.6 | Trace **non-answer tokens**. Everything is measured at the answer token, which always settles. | GPU | High | Blayney loops occur in the prompt's final third, not at the answer. |
| 1.7 | Sweep `num_steps` far past 128 (256, 512) to check the floor is a floor and not slow drift. | GPU | Medium | |
| 1.8 | Vary `init_scale` (Huginn exposes it) to test whether the transient length is init-driven. | GPU | Medium | Would confirm §3 causally rather than by formula. |
| 1.9 | Turn on `test_time_noise` deliberately and check the floor scales with it. | GPU | Medium | A second, independent handle on the floor mechanism. |
| 1.10 | Measure accuracy at the *same* fixed `num_steps` used for geometry, without adaptive exit. | GPU | High | Closes `rigor_audit` §10's non-comparability. |
| 1.11 | Per-position probing for Barannikov Task a (running count) — linear probe for `s_i` at each position. | GPU | **Very high** | This is the curator's actual request. |
| 1.12 | Same for Task b (nesting depth, balanced parentheses). | GPU | **Very high** | |
| 1.13 | Check whether a single direction `v` exists such that seeing a `1` translates the state by ≈`v` (the ℤ-action-by-translation hypothesis). | GPU | **Very high** | The precise formalisation of Barannikov's "register updated by T₁". |
| 1.14 | Larger models / other recurrent-depth checkpoints, to test whether findings are Huginn-specific. | GPU, high | Medium | |

## 2. Analyses runnable right now (no GPU)

| # | Item | Cost | Value |
|---|---|---|---|
| 2.1 | **Permutation test across prompts** for `winding ~ n_ops`. Needs no surrogate at all — real trajectories are already on-manifold. | free | **High** |
| 2.2 | Covariance/TME-style null (Elsayed & Cunningham) preserving marginal covariances, not just dimensionality. | cheap | High |
| 2.3 | Re-run **all nine** `winding_variants` against the on-manifold null. Only W1 has been redone. | cheap | High |
| 2.4 | Null-test the never-tested metrics: `mean_normed_accel`, `conv_rate`, `dlr`, `shrink`, homology. | cheap | High |
| 2.5 | Re-derive every `results/*.csv` conclusion using regime-restricted metrics; tabulate which survive. | cheap | **High** |
| 2.6 | Bootstrap confidence intervals on every reported correlation (none currently have them). | free | Medium |
| 2.7 | jPCA (Churchland et al. 2012): fit ẋ = Mx with M skew-symmetric, extract the maximal-rotation plane instead of the maximal-variance plane. | cheap | High |
| 2.8 | Log-polar diagnostic: plot θ against log r. Straight ⇒ single log-spiral mode, and the slope is arg λ / log|λ|. | free | High |
| 2.9 | DMD **restricted to the converging regime** with Gavish–Donoho rank truncation (4/√3 threshold). | cheap | Medium |
| 2.10 | Total-least-squares DMD to de-bias eigenvalues (Hemati et al.). | cheap | Low |
| 2.11 | SW1PerS on a **scalar** observable (e.g. ‖h_t−h*‖ or a leading-eigenvector projection), not the full state. | cheap | Medium |
| 2.12 | Fasy et al. subsampling confidence bands for any persistence claim. | cheap | Medium |
| 2.13 | Fréchet/Karcher mean on the sphere instead of the Euclidean centroid, then tangent-space (log-map) analysis. | cheap | Medium |
| 2.14 | Principal Nested Spheres (Jung, Dryden & Marron 2012) as the manifold-correct PCA analogue. | hard | Medium |
| 2.15 | Estimate ρ per *mode* rather than as one aggregate, to test the "slowest modes set the floor" explanation quantitatively. | cheap | High |
| 2.16 | Check whether the 43.1-unit inter-prompt separation is *decodable* — can a linear probe recover which prompt from a converged state? | cheap | High |
| 2.17 | Cross-task comparison of converged states: do same-answer prompts converge closer than different-answer ones? | free | **High** |
| 2.18 | Sensitivity of every threshold: `burn`, `frac=0.1`, `return_frac`, `k=3.0`, `tail_frac`, `loop_frac`. Report which conclusions are threshold-stable. | cheap | High |
| 2.19 | Re-run `winding_variants` null with per-variant calibration arms (only W1 has one). | cheap | Medium |
| 2.20 | Audit `normed_acceleration` and `drift_to_loop_ratio` for the regime error (both still unaudited). | free | Medium |
| 2.21 | `make_three_scale_task` / `make_three_scale_modk_task` were unreachable by the generator sweep (different signatures) — audit them for the universal length confound. | free | Medium |
| 2.22 | Effective rank of the trajectory over time — does dimensionality collapse with the radius? | free | Medium |
| 2.23 | Check whether `winding` correlates with `seq_len` *across tasks* (within-task it is rank-1 degenerate). | free | Medium |

## 3. Maths to work out properly

| # | Item | Note |
|---|---|---|
| 3.1 | Expected |winding| of a geodesic random walk on S^{n−1} with a prescribed radial profile — a closed form would replace the Monte-Carlo null entirely. | Genuinely tractable; the azimuth process is a spherical random walk with fixed increments. |
| 3.2 | Distribution of chord-to-arc ratio for a diffusive path, to place `loop_frac` on a principled footing rather than the current ~1/√L heuristic. | |
| 3.3 | Non-normality of `DF_e`: ρ vs σ_max, pseudospectra, Kreiss constant. The project's σ_max was implicitly treated as informative about ρ; for non-normal operators it is not. | Trefethen & Embree. |
| 3.4 | Exact bias of a log-linear fit through an exponential-plus-floor, as a function of floor level and fit length — would let old `lyap`/`conv_rate` values be corrected analytically rather than recomputed. | |
| 3.5 | Information capacity of a ball of radius ~1.4 in 5279 dims under bf16 quantisation: how many states are *actually* distinguishable? Bears directly on whether a bounded m=64 counter fits. | The scoping claim in D24(7) is currently qualitative. |
| 3.6 | Whether Barannikov's ℤ-action-by-translation is compatible with the RMSNorm sphere constraint at all — translations do not preserve norm, so the register must be realised as a *rotation* or in a quotient. | **Potentially important**: could reframe Task a/b before any compute is spent. |
| 3.7 | Expected participation ratio of the azimuth process under the on-manifold null, analytically — currently only measured. | |
| 3.8 | Why winding saturates in PR above ~15 (observed empirically, unexplained). | |
| 3.9 | Whether teacher-forced prefix probability can be corrected to exact-string probability with a boundary term, and how much it changes. | |

## 4. Stronger or alternative tests of existing claims

| # | Item |
|---|---|
| 4.1 | Every ledger claim should carry an explicit **falsifier**: what observation would overturn it. Currently only some do. |
| 4.2 | Pre-register the depth-threshold analysis (thresholds, exclusions, stopping rule) **before** running 1.2. |
| 4.3 | Adversarial re-derivation of D24/D25 by an independent pass, as was done for D21. |
| 4.4 | Hold-out: split trajectories, develop metrics on one half, confirm on the other. Nothing in the project does this. |
| 4.5 | Negative controls: shuffled-prompt trajectories, prompts with no computation to do. |
| 4.6 | Positive control: a synthetic task where the answer is *given* in the prompt — geometry should differ from a task requiring computation. |
| 4.7 | Test-retest: does re-running the identical prompt on different hardware give the same trajectory? (bf16 kernels may not be deterministic across GPUs.) |

## 5. Sources to check

| # | Item |
|---|---|
| 5.1 | **Verify "Movahedi et al." exists.** Cited in `convergence.py`; could not be located. |
| 5.2 | Read Pappone et al. (arXiv:2509.23314) directly — the attribution is already known to be wrong in three respects. |
| 5.3 | Geiping et al.'s own test-time-scaling numbers, to compare against our steps-to-settle. |
| 5.4 | Elsayed & Cunningham 2017 + the TME code, before implementing 2.2. |
| 5.5 | Perea & Harer 2015 before implementing 2.11. |
| 5.6 | Barannikov's own RTD papers — the curator's method, never applied here. |
| 5.7 | Lu et al. (arXiv:2507.02199) on latent-CoT critiques: does it target recurrent-depth specifically? |
| 5.8 | The second file in `files/ai_docs/` ("pre-defense memo") — flagged by the user as possibly stale, not yet read. |
| 5.9 | Whether anyone has published on numerical-precision floors in DEQ/recurrent-depth fixed points specifically. |

## 6. Verifiability practices — implemented, and still missing

**Implemented during the audit**: provenance sidecars + `manifest.jsonl` with
sha256 and separate `task_seed`/`init_seed`; regression tests pinning every
audit finding; `test_architecture_consistency` (pre-existing) enforcing that
every `run_*.py` and `results/*.csv` is documented; deterministic PCA.

| # | Still missing |
|---|---|
| 6.1 | CI: nothing runs the test suite automatically. All checks are manual. |
| 6.2 | `results/*.csv` are not provenance-tagged (only the trajectories are). |
| 6.3 | No environment lock recorded per result (numpy/sklearn versions affect PCA). |
| 6.4 | No test asserts that a *ledger claim's number* matches the CSV it cites; drift is possible. |
| 6.5 | No "unverified" marker convention in docs — verified and speculative text look alike. |
| 6.6 | `scratch/` (1787 files) is undifferentiated; some contains full repo copies that could be mistaken for source. |
| 6.7 | The ~20 files in `docs/` have never been checked for staleness. Several predate findings that contradict them. |
| 6.8 | No single "current best estimates" table; numbers must be assembled from the ledger's prose. |
| 6.9 | Random seeds are passed ad hoc; no project-wide seeding convention. |
| 6.10 | `h3_toy_model` is linted but not clean (~60 style errors). |

## 7. Task design

| # | Item |
|---|---|
| 7.1 | **Fixed-length prompts** (Barannikov's m=64) to break the universal rank-1.0 length confound. The single most valuable design change available. |
| 7.2 | Balanced answer distributions — exclude/rebalance answer 0 (currently a pure prior) and avoid `make_max_task`'s saturation. |
| 7.3 | Single-token answers by construction (0–9, or a fixed answer vocabulary), removing the multi-token scoring problem entirely. |
| 7.4 | A task where difficulty varies *without* changing the prompt at all (e.g. same string, different question). |
| 7.5 | Graded difficulty with a known minimal algorithm and known required depth, so r* has a ground truth to be compared against. |
| 7.6 | Distractor-controlled prompts: identical surface form, different required computation. |

## 8. Deliberate non-goals

Recorded so they are not mistaken for oversights: no training or fine-tuning
(explicitly ruled out on cost); no attempt to improve Huginn's accuracy; no
new architecture; no claim about models other than Huginn at the pinned
revision.

---

## 8b. GPU experiments — DONE 2026-07-26 (Kaggle T4, free tier)

The "needs GPU" section below was written when I believed no compute was
available. That was an unverified assumption: the Kaggle CLI was configured all
along and this project had used it ten times before. Checking beats assuming.

| Item | Outcome | Ledger |
|---|---|---|
| 1.1 float32 rerun (THE GATE) | **PASSED.** residual 1.190 (bf16) / 0.155 (fp16) / 0.000290 (fp32); bf16→fp16 slope −0.981 against −1 predicted. Strengthened from a single ratio to a 3-dtype regression. **Bonus finding: bf16 makes Huginn appear to converge 4.6× sooner than it does** (regime_end 20.7 → 96.3). | D30 |
| 1.3 Arnoldi on JVPs | **DONE.** ρ = 0.802 exactly (first exact measurement in the project); leading eigenvalue complex in 3/3; 8–10 oscillatory modes. Forward-mode AD unusable (no SDPA rule; MATH backend OOMs) → reverse mode, exact since eig(Jᵀ)=eig(J). | D31 |
| 1.11 / 1.12 / 1.13 Task a/b probes | **DONE.** Task a R² = +0.601 position-controlled; Task b R² = +0.590 where position explains only 0.160. **No winding quantisation** (p=0.9954 against the null) — the rotation-register readout fails. | D32 |
| 1.2 depth threshold | pushed and running at time of writing | — |

Still not done from that section: 1.4 (multi-init paths for the current task
set), 1.5 (re-run sweeps with provenance), 1.6 (non-answer tokens), 1.7–1.10,
1.14. All remain worth doing; none is blocked on anything but time.

## 8c. Open puzzle found 2026-07-27 — unexplained, deliberately not forced

**The register correlation drops in the last third of BALANCED parenthesis
strings, and only there.**

```
                first third   middle third   last third
Task a (count)     +0.3565      +0.3522       +0.3267    flat  (paired p=0.60)
b_unbal            +0.3903      +0.4177       +0.3540    flat
b_bal              +0.3306      +0.3647       +0.1762    DROPS (paired p=0.025)
```

The robust part: **the register does not decay with accumulated count** in
either unconstrained condition. Only the closure-constrained design shows it.

Two explanations were proposed and both **refuted by measurement**:

1. *Range restriction* — a bridge returns to 0, so the target's spread should
   collapse at the end. Refuted: b_bal's residual target sd **increases** 1.71×
   toward the end (1.273 → 2.174), the opposite of the prediction.
2. *Position absorbing the signal* — depth should become near-deterministic in
   position once closes are forced. Refuted: `R²(target ~ position)` is flat
   across thirds (0.350 / 0.400 / 0.343).

What is left unexplained but observed: b_bal's last third has markedly fewer
increments (0.378 against 0.586 in the first third) — forced by balancedness —
while b_unbal stays near 0.5 throughout. That asymmetry is real but no mechanism
connecting it to the correlation drop has been tested.

**UPDATE 2026-07-28 — a third explanation tested and also refuted.** The
increment-rate-matched control proposed here was run: b_unbal's thirds were
subsampled to match b_bal's measured increment fractions (0.586 / 0.542 /
0.378). If the rate caused the drop, the matched control should reproduce it.

```
b_bal    raw            +0.3306 / +0.3647 / +0.1762     drop -0.154
b_unbal  raw            +0.3903 / +0.4177 / +0.3540     flat
b_unbal  rate-MATCHED   +0.3776 / +0.3928 / +0.3833     flat, last-first +0.006 (p=0.907)
```

Matching the increment rate does **not** reproduce the drop. Three explanations
now refuted: range restriction, position absorption, increment rate.

Still worth trying: a signed rather than symmetric estimator for v; mutual
information instead of correlation, since correlation is fragile where the
conditional symbol distribution shifts; and a same-length control generated
with a *different* balancedness constraint (e.g. Dyck paths conditioned to stay
above a positive floor) to separate "returns to zero" from "never goes
negative". **Still recorded as open — a fourth hypothesis fitted after three
failures would be exactly the behaviour this project's audit exists to
prevent.**

## 9. Completed since this list was written (2026-07-26)

Kept here rather than deleted, so the list records what was tried as well as
what remains.

| Item | Outcome |
|---|---|
| 2.1 permutation test | **Done.** H2 null; no stratum survives correction (D26). |
| 2.8 log-polar diagnostic | **Done.** θ vs log r is straight, mean R²=0.984 — a single dominant log-spiral mode. Yields a falsifiable prediction for the Arnoldi step: \|arg λ\| ≲ 0.78 rad. |
| 2.15 per-mode ρ | **Done.** ρ ranges 0.73–1.02 across the gap's top 8 modes. Real but not the main term in the floor. |
| 2.16 probe converged states | **Done and positive-but-fragile** — see D27. Superseded the weaker geometric test. |
| 2.18 threshold sensitivity | **Done** for the audit's own constants; both key findings threshold-stable (§22). |
| 2.20 audit accel / DLR | **Done.** Both inherit the regime error; DLR inflated **5.32×** (D26(3)). |
| 2.21 three_scale generators | **Done.** They are the designed length-control exception, not an oversight. |
| 3.6 ℤ-action vs the sphere | **Done** — `docs/register_geometry.md`. Translation register impossible; rotation feasible. |
| 6.4 ledger-vs-CSV drift test | **Done** — `tests/test_winding_permutation.py`. |
| 6.5 / 6.7 doc staleness | **Done.** Ten superseded docs carry an additive banner; `state_of_knowledge.md` is the single source. |
| 2.2 covariance/TME null | **Partially done.** Anisotropy addressed by a `subspace_dim` sweep showing winding saturates in PR above ~15, so the real-vs-null PR gap moves winding very little. A full TME null is still not implemented. |

## Priority shortlist

If only five things happen next, these:

1. **1.1** float32 rerun — settles the noise-floor account for minutes of GPU.
2. **1.3** Arnoldi on JVPs — gives H2 and H3 from one exact object.
3. **2.5** re-derive all existing conclusions with regime-restricted metrics — tells us how much of the project survives.
4. **1.2 / 1.11–1.13** the depth-threshold and Barannikov probes — the only routes to a *positive* result.
5. **3.6** whether a ℤ-translation register is even compatible with the sphere — cheap thinking that could redirect items 1.11–1.13 before compute is spent.
