> **FROZEN 2026-08-05.** Archived under CLAUDE.md §6 (three live documents only).
> Not maintained. Superseded by `docs/UNDERSTANDING.md`, `claims_ledger.md`, `directions.md`.
> Statements here may be contradicted by later work — several assert things since disproven
> (e.g. "H1 was never tested": it was, and the test itself was invalid — see D56, D65).

# Forward plan

STATUS: written 2026-07-26, after the two-pass rigor audit. Supersedes
`project_plan.md` for anything about *what to do next*; that file's Phase 0/1
structure predates findings that removed several of its planned steps.

READ FIRST: `docs/state_of_knowledge.md` (what is established, null, retracted,
unknown) and `docs/backlog_not_done.md` (the full unexplored surface).

---

## 1. Where the project actually is

The audit converted this from "a set of geometric findings" into "a set of
mostly-retracted geometric findings plus a working method". That sounds bad and
is actually the useful state: the instruments now measure what they claim, and
the negative results are trustworthy.

- **H1 was never tested** — its instrument has one reachable output (§18).
- **H2 is null** under the strongest available test, which needs no surrogate
  (D26). It also *cannot be confirmed* by any current synthetic task, because
  difficulty and prompt length are rank-correlated at exactly 1.000 (§21).
- **H3's premise holds** — the map contracts, ρ ≈ 0.84–0.90 (D24(7)) — but the
  applied claim needs rescoping: contraction forbids an *unbounded* register,
  not a bounded one.

The single most valuable thing the project now owns is not a result but a
constraint: **the state lives on a sphere, so a counting register must be a
rotation** (D26(5)). That reframes the curator's tasks and makes the winding
number relevant again, in a different index.

## 2. The critical path, in order

Each step is gated on the previous only where stated.

### Step 1 — settle the noise-floor account (GPU, minutes)
Run one prompt in float32. Prediction: the floor drops by 2⁻¹⁶, from ≈0.91 to
≈1.4e-5. **Gate**: if it does not, the entire regime-separation framework
(`metrics/regime.py`, findings §1–§3, §6) needs rederivation, and everything
downstream should stop until it is.

### Step 2 — measure the Jacobian spectrum (GPU, ~1 min/prompt)
Arnoldi on Jacobian-vector products of the core block at the converged state.
Returns ρ (H3, exactly, vs the current orbit-fit estimate) and arg λ (rotation
per step) from one object. Report pseudospectra alongside: the Jacobian is
expected to be strongly non-normal, and for non-normal operators ρ ≪ σ_max, so
the project's existing σ_max is not informative about ρ.

**This step now has a falsifiable prediction.** A log-polar diagnostic on the
signal regime (backlog 2.8, run 2026-07-26) finds θ vs log r is a straight line
with **mean R² = 0.984** over 10 trajectories, slope dθ/d log r ≈ **−4.5**,
consistent across prompts (−3.8 to −5.1). That is the log-spiral signature: a
*single dominant* complex mode, not a mixture. Since the slope is
`arg λ / log|λ|`, with |λ| = ρ ≈ 0.84 it implies

> ~~**|arg λ| ≲ 0.78 rad ≈ 45°/step, period ≳ 8 steps.**~~ **FALSIFIED (D31): measured 1.945 rad, 111°/step, period 3.2 steps.** Two causes: I measured the diagonal block `J_TT` rather than the full coupled operator, and the log-polar θ was cumulative *unsigned* azimuth, which is not the upper bound I claimed it was.

The bound is one-sided because the diagnostic accumulates *unsigned* azimuth
(the sum of `arccos(q_t · q_{t+1})`), not net signed rotation in a fixed plane.

**If Arnoldi returns a purely real leading eigenvalue, the log-spiral reading is
wrong** and the angular motion is not a single rotating mode — which would also
retire the last route by which a rotation register (Step 4) could be read off
the recurrence.

### Step 3 — the depth-threshold experiment (GPU, ~1h)
`scripts/run_depth_threshold.py`, already implemented and tested (23 tests).
Pre-register the analysis before running: thresholds, exclusions, stopping rule.
This is the only route to a *positive* result from the existing task set, and
it is behavioural, so it touches none of the geometry the audit dismantled.

### Step 4 — the register probes for Task a/b (GPU)
Gated on `docs/register_geometry.md`'s conclusion: probe for a **rotation** (or
a tangent-space translation, equivalent at small angle), not an ambient-space
translation. Two experiments:
- **Task a**: per-position linear probe for the running count `s_i` in tangent
  coordinates; test whether the fitted angle is linear in `y_i`.
- **Task b**: balanced parentheses. Because depth returns to 0, the curve
  closes and the winding number becomes a genuine integer invariant. Test for
  **quantization**, not correlation — a far stronger prediction.

### Step 5 — re-derive what survives (CPU) — **DONE 2026-07-26**
`scripts/run_regime_rederivation.py`, 155 trajectories, ledger D29. Headline:
the sign of `consecutive_step_cosine` follows the **recording length**, not the
model — **+0.258 at ns=16** (too short to reach the floor), **−0.324/−0.368 at
ns=64/128**. 90.3% of trajectories flip sign under regime restriction.
`conv_rate` understated **3.2×**, `dlr` inflated **4.04×**. The signal fraction
of a recording is 52% / 30% / 15% at ns=16/64/128 — recording longer adds
arithmetic, not information.

**The limit of Step 5, which is itself a finding.** Only `convergence.csv` and
`homology.csv` have surviving raw paths. **16 of the 18 invalidated results
files have no raw data on disk**, so their conclusions cannot be corrected
without a GPU re-run. The project stored derived scalars and discarded what
they were derived from — the concrete cost of the provenance gap in D24(4)/(5).

---

## 2b. Status: every step is now code-complete

| Step | Script | State |
|---|---|---|
| 1 gate | `run_precision_check.py` / `scratch/kaggle_precision_gate` | **DONE, PASSED** — D30 |
| 2 spectrum | `run_jacobian_spectrum.py` / `scratch/kaggle_jacobian` | **DONE** — ρ=0.802 exact, rotation present — D31 |
| 3 depth threshold | `run_depth_threshold.py` / `scratch/kaggle_depth` | pushed, running |
| 4 register probe | `run_register_probe.py` + Task a/b generators | **DONE** — both tasks positive; no quantisation — D32 |
| 5 re-derivation | `run_regime_rederivation.py` | **DONE** — D29 |

All four GPU steps were run on Kaggle's free T4 tier, which this project had
already used ten times. I had asserted "no GPU access" without checking — the
CLI was configured throughout.

Step 1 passed, so the regime framework underlying Steps 2–5 is confirmed rather
than needing re-derivation.

## 3. Completed since this plan was written (2026-07-26)

- **`run_manifold_null.py` — DONE, and it closes the winding question.** The
  null is unbiased (calibration arm 6.4% vs 5% nominal) and the effect has **no
  consistent sign**: +0.0127 (z +10.97) at ns=64 against −0.0091 (z −5.58) at
  ns=128, in both tasks, strata differing at p=1e-11. `num_steps` is a
  recording budget, so |winding| tracks the recording window, not the content.
  D22 is deleted and nothing replaces it (D28).
- **`run_winding_permutation.py` — DONE.** H2 null under the assumption-free
  test; no stratum survives BH or Bonferroni (D26).
- **`run_answer_probe.py` — DONE.** First positive signal, suggestive only: the
  answer is decodable from the converged state at fixed prompt length for
  `count_ones` (R² +0.21, p 0.005–0.030 across α), fragile to
  leave-one-level-out (D27).
- **Log-polar diagnostic — DONE.** Feeds Step 2's prediction above.

## 4. What to tell the curator

`docs/state_of_knowledge.md` §6 is written for this. The substantive points:

1. His design fixes our two worst problems — fixed m=64 breaks a length
   confound that is *universal* across our task set, and per-position `y_i`
   escapes the answer-token degeneracy.
2. We have a derivation that the register must be a rotation, not a
   translation, with the bound robust to a 1000× error in its one input.
3. Task b's balancedness closes the curve and makes the winding number a true
   topological invariant — which is exactly the defect that invalidated our own
   use of it.
4. Probe the latents, not the output: the output fails, and the failures we
   have are a zero-prior artifact.

## 5. Explicit non-goals

No training or fine-tuning (ruled out on cost). No claim about models other
than Huginn at the pinned revision. No attempt to rewrite teammates' authored
prose — superseded documents carry an additive banner, not edits.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Step 1 fails and the regime framework collapses | It is the first step precisely so this is found early and cheaply. |
| No GPU access before the deadline | Steps 1–4 all need GPU. Steps 5 and the CPU items in `backlog_not_done.md` §2 do not, and are enough to produce an honest write-up on their own. |
| A positive Step 3/4 result gets over-read | Pre-register; report effect sizes beside p-values; state the length-confound ceiling explicitly. |
| Retracted claims leak into the write-up from stale docs | Ten superseded docs now carry banners; `state_of_knowledge.md` is the single source. |
