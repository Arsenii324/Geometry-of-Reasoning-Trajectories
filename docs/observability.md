# Observability: what to measure so the unexpected is visible

This is not generic advice. Every item below is derived from a failure that
actually occurred in this project and is traceable to a ledger row.

---

## 1. The central lesson: many metrics is not the same as diverse metrics

The project measured rotation five ways — **winding number**, **chord-to-arc
ratio**, **H₁ persistence**, **participation ratio**, **subspace dimension** — and
took the agreement between them as corroboration.

**They all failed together** (D22, D26, D28, D32). Not because the statistics were
wrong but because every one of them shared a single unexamined assumption: *that
the recorded window contains signal.* It did not — past ~19 unrolls the trajectory
is arithmetic noise (D24, D30), and each statistic dutifully summarised the noise.
The winding effect even **reverses sign with `num_steps`**, a recording-budget
parameter that is not a property of the model or the prompt (D28).

> **Diversity in *statistic* is worthless. Diversity in *assumption* is
> everything.** Five metrics that all need a well-chosen window are one metric.

**The concrete miss.** The rotation question has an answer that needs no
trajectory, no window and no null: the **imaginary parts of the Jacobian's
eigenvalues**. A contracting map with complex eigenvalues rotates at a rate given
by their argument; one with real spectrum does not rotate at all. D31 ran Arnoldi
on Jacobian-vector products and reported only the *magnitude* (ρ ≈ 0.79–0.81). The
arguments were in hand and never looked at. **That single number would have settled
in one run what five trajectory statistics could not settle in twenty.**
**It did.** Applying this principle to the existing log, with no new compute: the
leading eigenvalue is complex in 3/3 prompts, rotation period ≈2.6–6.0 unrolls —
**the trajectory really does rotate** — but the mode survives only ~4 turns and is
sampled at ~3 points per turn against a Nyquist floor of 2, while winding was
computed over windows 5–9× longer than the rotation lives. That is a complete
mechanistic account of D28's sign instability, and it had been sitting in a log for
over a week (D55). Cost: zero GPU.

**Rule.** Before adding a metric, ask what would have to be true for it to be
wrong. If the answer matches an existing metric's, the new one adds nothing.
Prefer a metric that depends on the *operator* over one that depends on a
*sampled trajectory*, because the operator has no recording window.

---

## 2. Standing diagnostics

Cheap, and each one is here because its absence caused a specific error.

| # | diagnostic | the failure it would have caught |
|---|---|---|
| D1 | **Report the noise floor next to every trajectory statistic**, and the fraction of the window above it | D24, D28 — four metrics summarised arithmetic noise |
| D2 | **Report fit quality with every fitted parameter**, and flag fits below the applicability bar | D44 — 2 of 12 trained fits (R²=0.52, 0.79) produced both the published mean and its sd |
| D3 | **Report the per-instance distribution, never only the mean** | D49 — R²=0.9928 reads as "almost perfect" for a population where two thirds never resolve the integer and 99/220 sit on the decision boundary |
| D4 | **Report effective dimensionality (participation ratio) beside every R²** | D48 — R² 1.0000 vs 0.9928 looks like rounding; PR 1.0 vs 4.8 is the same fact as a factor of five |
| D5 | **Report the outcome's dynamic range BEFORE any intervention** | D47 — a causal patch on an output that emits `1.00` for all 120 prompts; the experiment had no power and was misdiagnosed as "broken" |
| D6 | **Declare the unit of analysis: what varies across replicates?** | D44 — a t-test over 12 prompts with n=1 weight-set per arm, presented as evidence about *training* |
| D7 | **Sweep the analyst's free choices and report the sensitivity** | D43 — the verdict flipped with the ρ estimator and the convergence threshold; D52(2) — a trend moved p=0.036 → 0.19 on a fit filter |
| D8 | **Run a permutation null and require it to be NEGATIVE** | the anti-interpolation defence at d=5280 ≫ n=220 (D41(6)); a null at or above zero means the probe can interpolate |
| D9 | **Include a calibration arm: feed a surrogate back as if it were data** | D28 — every earlier winding null was uncalibrated and therefore uninterpretable; the calibrated one beat its null at 6.4% against 5% nominal |
| D10 | **Derive index windows from recorded metadata; assert, never assume** | D34/D36/D50 — a one-token offset the ‖v‖ search preferred by 9%, while `meta.json` recorded the true token positions all along |
| D11 | **Every quoted number names the file it came from** | D41(3) — "trained ~10%" was quoted from a different task configuration; neither kernel had measured accuracy at all |
| D12 | **Run the untrained control as a companion, not as a special study** | D40, D41, D48, D53 — four content-level findings turned out to be architectural, each discovered late |

---

## 3. Metrics to add

**Not more shape statistics.** The three below each answer a question no current
metric answers, and each fails differently from what exists.

1. **Jacobian eigenvalue arguments** — rotation rate, from the operator, no window,
   no null. See §1. (B4.6)
2. **Effective rank of the state trajectory over depth** — distinguishes "the
   state moves along one direction" from "the state explores a subspace", which
   winding cannot. Companion to D4.
3. **Outcome-variance-before-intervention**, reported automatically by any causal
   experiment. Companion to D5; would have made D47 self-diagnosing.

## 4. Metrics to retire

- **Winding number** as evidence about the model. It is governed by the recording
  budget (D28). Keep it only as a *worked example* of a window-dependent statistic.
- **`steps_to_settle`** as a difficulty measure. It thresholds at 10% of the
  maximum step, which is always the `h₀` transient, making it a function of ρ alone
  (D24(3)).
- **`register_r`** as published. Off-by-one window; superseded by a grouped,
  in-fold-controlled cv R² (D50).

## 5. What must not be cut for terseness

Compressing routine output is good. These are not routine:

- **Full logs when a result is surprising.** Two of this project's most important
  corrections (D47's misdiagnosis, D50's window bug) came from reading *code and
  logs*, not summary numbers.
- **Per-instance data.** D49 exists only because the distribution was plotted.
- **Second, unrelated sources for a load-bearing number.** ρ has four independent
  estimates; that is why its magnitude error was caught (D44(2)).
- **Cross-checks that are not strictly necessary.** The trained-vs-untrained
  comparison was not required by any hypothesis and it reshaped the whole project.

> Terseness is for output you expect. The moment output is unexpected, verbosity
> is the entire point.
