> **FROZEN 2026-08-05.** Archived under CLAUDE.md §6 (three live documents only).
> Not maintained. Superseded by `docs/UNDERSTANDING.md`, `claims_ledger.md`, `directions.md`.
> Statements here may be contradicted by later work — several assert things since disproven
> (e.g. "H1 was never tested": it was, and the test itself was invalid — see D56, D65).

# State of knowledge

STATUS: consolidated 2026-07-26, after the two-pass rigor audit
(`rigor_audit.md`, ledger D24–D26). This is the single document to read to know
where the project stands. Every number here is reproducible from a script in
`scripts/` or from the raw trajectories; the ledger row is cited for each.

**Read this before any other results document.** Several earlier documents
(`results_report.md`, `methods_and_findings.md`, `narrative.md`) predate the
audit and state conclusions it overturned. Where they disagree with this file,
this file is correct.

---

## 1. What is established

| # | Finding | Evidence | Ledger |
|---|---|---|---|
| E1 | **The recurrent state is confined to a sphere.** Every `SandwichBlock` ends in RMSNorm, so `‖h‖ = 76.37 ± 0.007` (0.0073% variation). The radial coordinate carries no information. | model source L535 + 20×128 states | D24(8) |
| E2 | **Huginn runs in bfloat16.** Every stored value is exactly bf16-representable (round-trip deviation 0.000e+00). Unit roundoff 2⁻⁸, not float32's 1.2e-7. | banked trajectories | D24(1) |
| E3 | **Trajectories have two regimes**: ~15 informative steps, then 50–115 steps at an arithmetic noise floor. The floor is **isotropic** (tail top-2 PCA 8.8–10.7%, participation ratio 18–19/20, where 2-D dynamics would give ~2). | banked trajectories | D24(1), D25(1) |
| E4 | **The map contracts: ρ ≈ 0.84–0.90.** Measured correctly, from two orbits of the *same prompt with different h₀*, fitting only above the floor. | `trajectories/` multi-init | D24(7) |
| E5 | **Path-independence holds** up to arithmetic precision — different h₀ converge to the same state (residual gap 1.2–1.6 ≈ the floor). Confirms Geiping et al. | same | D24(7) |
| E6 | **Prompt information persists indefinitely.** Different prompts stay 43.1 units apart (33° of arc), flat to four decimals over 120 steps. | banked trajectories | D24 |
| E7 | **The length confound is universal**: rank-corr(n_ops, seq_len) = **exactly 1.000** for every synthetic generator, so partial correlation on length is degenerate project-wide. | tokenizer + all generators | D25/§21 |
| E8 | **A translation register is architecturally impossible; a rotation register is not.** Bound `D < R√(2δ) = 0.923` total, i.e. 0.0144 per increment over m=64 — 62× below the noise floor, robust to a 1000× shell error. | derivation + measurement | D26(5) |

## 1a. GPU results (2026-07-26) — the core hypotheses, tested

| # | Finding | Ledger |
|---|---|---|
| G1 | **The residual IS arithmetic.** Residual radius by dtype: bf16 1.190 / fp16 0.155 / fp32 0.000290. bf16→fp16 gives 7.7× against 8× predicted (slope −0.981). The regime framework is confirmed. | D30 |
| G2 | **bf16 makes Huginn appear to converge ~4.6× sooner than it does.** `regime_end` 20.7 (bf16) → 32.3 (fp16) → **96.3 (fp32)**. "Settling at t≈14" is a precision artifact; in fp32 there is ~5× more signal. | D30 |
| G3 | **ρ(∂ₕR) measured exactly for the first time: 0.7935–0.8083** (mean 0.802), stable across an 8× difficulty range. H3's premise confirmed; independently corroborates D24(7)'s 0.84–0.90 from different data by a different method. | D31 |
| G4 | **The recurrent map is dominated by rotating modes** — leading eigenvalue complex in 3/3 prompts, top eigenvalues in conjugate pairs, 8–10 oscillatory modes each. | D31 |
| G5 | **Task a POSITIVE (position-controlled):** running count decodable at R² = +0.601 on position-residuals (sd 1.06 counts). Raw R² = 0.892 is worthless — position alone explains 98.5% of `y_i`. | D32 |
| G6 | **Task b POSITIVE and cleaner:** nesting depth decodable at R² = +0.590 where `R²(depth ~ position) = 0.160`. | D32 |
| G12 | **Depth buys readout — in aggregate only.** Decoding the total from the ANSWER TOKEN rises R²=0.65→0.993 over ~24 unrolls (rho=+0.97…+0.98, all p<0.001) while the per-position register is flat. **But readout quality does NOT track difficulty at any depth** (rho −0.09…+0.16, the one nominal hit dies under Bonferroni), so this does *not* explain why harder instances need more depth. A probe asks whether the count is PRESENT; r\* asks when the model's own margin saturates. | D38 |
| G13 | **OPEN: why do harder instances need more depth?** The count is present from r=1, readout improves with depth but not differentially by difficulty, and the register is complete immediately. D35's difficulty-dependence lives in the model's own logit formation, which every probe here bypasses. | D38(4) |
| G11 | **Recurrence rotates the register, it does not build it.** Decodability is flat from r=1 (rho=+0.006, p=0.958) while the increment direction rotates monotonically to 99% alignment by r≈16 (rho=+0.990, p=5.5e-72), identically across all 12 seeds. Dissociates the depth-scaling result from register construction. My pre-registered prediction was the opposite. | D37 |
| G10 | **The register is a genuine ℤ-action.** `(`/`)` displace by ±v symmetrically (+31.6 / −29.7) and a matched pair returns the state to its start (p=0.891). Confirmed on Task b too: r = +0.225/+0.289, 16/16 sign, p≤2.4e-06. Task a and Task b use **orthogonal** axes (cosine +0.058). | D36 |
| G8 | **Barannikov's `T_1` found directly.** `v = mean(Δ\|bit=1) − mean(Δ\|bit=0)` is shared across 16 independent strings at cosine **+0.920** (66.8σ), and its projection tracks the count beyond position AND current-token identity: r = **+0.269**, 16/16 same sign, p=9.2e-10. No probe, no CV. | D34 |
| G9 | **H2 CONFIRMED, confound-free.** At m=64 fixed (all 200 prompts exactly 77 tokens), required depth scales with the count: rho=+0.225, p=1.3e-3, CI [+0.086,+0.355]. Censoring biased it down; the censoring rate is itself a second signature. Balance/entropy rejected as the driver. | D35 |
| G7 | **No quantisation of winding on balanced strings** (0.281 vs the 0.250 no-quantisation reference, p=0.9954). The rotation-register readout is not supported. | D32 |

## 1b. What is suggestive but NOT established

| # | Finding | Why not established | Ledger |
|---|---|---|---|
| S1 | **The answer is linearly decodable from the converged state at fixed prompt length** — `count_ones` within-level LOO R² = +0.205…+0.219 across ridge α ∈ [1e1, 1e3], permutation p = 0.020/0.005/0.030, Bonferroni over 2 tasks = 0.040. This is the project's first positive signal about *computation* rather than about settling or length. | Fragile: leave-one-level-out drops R² to −0.009 when `n_ops=24` alone is removed. N=38. One task of two (`projection` is null). Five α values swept, so the minimum p is selection-biased. Needs a properly powered replication. | D27 |

## 2. What is null (tested, no effect found)

| # | Finding | Strength | Ledger |
|---|---|---|---|
| N0 | **|winding| does not measure rotation at all.** Under an unbiased on-manifold null (calibration 6.4% vs 5% nominal) the effect flips sign with `num_steps`: +0.0127 (z +10.97) at ns=64, −0.0091 (z −5.58) at ns=128, in both tasks, strata differing at p=1e-11. It tracks the recording window, not the content. | **Definitive** | D28 |
| N1 | **H2 — |winding| does not track difficulty.** Permutation test on real trajectories only, no surrogate: no stratum survives BH or Bonferroni; signs disagree across strata; one nominal hit in four is what chance gives (P=0.185). | **Strongest available** — assumption-free | D26(1) |
| N2 | **Converged states do not cluster by answer.** Mean effect −0.12°, 1/13 strata nominal vs 0.7 expected. *Near-zero power* (1–3 same-answer pairs per stratum) — "no evidence", not "evidence of absence". | Weak (underpowered) | D26(2) |
| N3 | **Contraction does not vary with difficulty.** ρ vs n_ops rank-corr +0.224, N=5 groups. | Weak (N=5) | D24(7) |
| N5 | **`consecutive_step_cosine`'s sign is set by the recording length, not the model.** +0.258 at ns=16 (no floor to contaminate it), −0.324/−0.368 at ns=64/128; 90.3% of trajectories flip sign under regime restriction. Same structural result as N0. | **Definitive** | D29 |
| N4 | **Model output fails the counting task**, and the apparent successes are a zero-prior: correctness is perfectly separated by whether the answer is 0 (4/5 vs **0/8**), all at unroll 1. | Moderate | D25, §15 |

## 3. What was believed and is now retracted

| # | Retracted claim | Why |
|---|---|---|
| R1 | "82.1% of trajectories wind LESS than their null (median z=−5.61)" — D22's headline | The null (`matched_random_walk`) leaves the state manifold (‖h‖ 76.37→154.76). Evidence **deleted**, not reversed. |
| R2 | "cos = −0.276 ⇒ the path zig-zags inward" | Measured on the second half, which is entirely noise regime. In the computing regime cos = **+0.084** — the path glides. |
| R3 | "`steps_to_settle` is a proxy for effective compute" | It is a proxy for ρ: `log(0.1)/log(0.8407) = 13.27` predicted vs 13.84 observed, using no task information. |
| R4 | "All trajectories settle, no loops" (H1) | `classify_shape` returns "settle" for **140/140** by construction; `s[-1]/s.max()` maxes at 0.0188 against a 0.10 threshold. **H1 was never tested.** |
| R5 | "Homology max persistence 0.009 shows no loops" | 84–85% of the cloud is the noise ball; the score is ≈ (noise-ball scale)/(transient extent). An open arc has trivial H₁ anyway. |
| R6 | "The noise floor matches a parameter-free bf16 prediction to 1.36×" (mine, D24) | Wrong formula — isotropic noise adds in **quadrature**. Correct: `η/√(1−ρ²)` = 0.175, ratio **5.2×**. Conclusion survives on isotropy, not magnitude. |
| R7 | "The winding null reverses under an on-manifold surrogate" (mine, D25) | Underpowered. **Resolved at full power (D28): there is no consistent sign** — it flips with `num_steps`. The retraction was correct. |
| R8 | "A ≈14-step oscillatory mode exists" (mine) | DMD was fitting a linear operator to isotropic rounding noise. |
| R9 | Pappone et al. / "Movahedi et al." support the zig-zag reading | Pappone is GPT-2-scale not Huginn, defines DLR differently, and reports cos **+0.5 to +0.65 (positive)**. "Movahedi et al." could not be located. |

## 4. What is unknown

- Whether the noise floor is really arithmetic — **one float32 rerun settles it**; predicted to shrink by 2⁻¹⁶ to ≈1.4e-5.
- ρ and arg λ of the true Jacobian `DF_e(h*)` — never measured; Arnoldi on JVPs would give both exactly.
- Whether required depth scales with difficulty (`run_depth_threshold.py`, implemented, unrun).
- Whether Huginn holds a bounded counting register — E8 says it would have to be a rotation; untested.
- Which run produced `full_synthetic_experiments.csv` — it demonstrably is **not** the banked trajectories (0/80 rows reproduce).

## 4b. The honest denominator: how much of the existing data is affected

Triaging every `results/*.csv` against the metrics the audit invalidated
(`winding` D28, `steps_settle` D24(3), `cos` D24(2), `conv_rate`/`lyap`/
`contraction` D24(6), `dlr`/`mean_normed_accel` D26(3), homology D25(4)):

> **18 of 29 results files carry at least one invalidated metric.**

Affected (row counts): `blayney_repro` 7122, `dissoc_multiinit` 600,
`three_scale_modk_extended` 270, `three_scale` 180, `dissociation_15seed` 180,
`pararule` 120, `three_scale_modk` 126, `forceloop` 96, `full_synthetic_experiments` 80,
`counting_accuracy` 56, `h2_loops`/`switch`/`dissociation`/`dissociation_results` 60 each,
`maxtask` 48, `counting` 30, `convergence` 15, `smoke_new_tasks` 8.
`homology.csv` is also affected via D25(4) (its persistence score is normalised
by the transient extent) even though its column names do not match the triage
keys.

Unaffected and usable as-is: `answer_probe`, `manifold_null`,
`winding_permutation`, `winding_null`, `winding_variants_null`,
`v6_correctness_probe`, `fdr_correction`, `phase`, `power_curve`,
`power_loop_rate`.

This does **not** mean 18 files are worthless — the raw quantities are still
correctly computed, and several (e.g. the `dissoc_multiinit` `lyap` column) can
be recomputed floor-aware from data already on disk. It means no *conclusion*
drawn from those columns should be carried forward without recomputation. That
recomputation is the largest remaining CPU task (`plan_forward.md` Step 5).

## 5. The one methodological lesson

Four separate metrics failed the same way: **a statistic averaged over a window
that mixes the signal regime with the arithmetic-noise regime.** Trajectories
carry ~15 informative steps followed by 50–115 steps of rounding, and metrics
that average "the whole path" or "the second half" report arithmetic.
`metrics/regime.py` provides floor-aware replacements; the originals are kept
bit-compatible so cached results stay reproducible, with the bias documented in
place.

A second lesson, from my own errors (R6–R8): **an underpowered check reported
confidently is worse than no check**, because it enters the record and has to
be chased down. Both R7 and R6 were caught within hours, but only because the
numbers were written down precisely enough to be re-derived.

## 6. Bearing on the curator's tasks

Barannikov's Task a (running count, m=64 fixed, per-position `y_i`) and Task b
(nesting depth, balanced parentheses) address the project's two worst problems
directly:

- **m = 64 fixed** breaks the universal length confound (E7) at its source.
- **Per-position `y_i`** escapes the answer-token degeneracy, where the state
  merely converges and every geometric metric is dominated by that convergence.

Two things this project can contribute to the design:

1. **Probe for a rotation, not a translation** (E8) — or equivalently a
   translation in tangent/log-map coordinates, which agrees at small angle.
   This *rescues* the register framing rather than contradicting it.
2. **Task b's balancedness closes the curve.** Depth returns to 0, so if the
   register is a rotation the state returns to its start, and the winding
   number becomes a genuine integer invariant in π₁(ℝ²∖{p}) ≅ ℤ. That predicts
   **quantization** — a far stronger test than any correlation, and it cures
   the open-arc defect that made every winding number in this project a
   non-invariant real number.

Also load-bearing: probe the **latents, not the output** (N4), since the output
fails and the toy model shows the count can stay linearly decodable under
contraction.

## 7. Where to look next

See `docs/backlog_not_done.md` for the full surface. The priority shortlist:
float32 rerun (E2/E3); Arnoldi on JVPs (ρ and arg λ from one object);
re-derivation of all existing conclusions with regime-restricted metrics; the
depth-threshold experiment; and the register probes for Task a/b.
