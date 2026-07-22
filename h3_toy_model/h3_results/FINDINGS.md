# H3 toy-model validation — findings (2026-07-18)

Built because no GPU/Huginn access exists on this machine (18GB free disk
ruled out even a careful Huginn-3.5B download; see reasoning in the session
log). This validates H3 (Contraction Bottleneck Theorem) on the one model in
the whole project that's actually CPU-runnable: `TinyRecursiveModel` in
`train_tiny_recursive.py`. Code: `spectral.py` (power-iteration operator-norm
estimator, self-tested against known-answer linear/nonlinear maps in
`test_spectral.py`, 5/5 pass), `h3_validation.py` (the sweep), raw data at
`h3_sweep.csv`, figure at `h3_sweep.png`.

## What was actually measured (nobody had before)

The original presentation.md claim (100% vs 90% in-distribution accuracy,
0% extrapolation, beta=0 vs beta=5.0) never measured the model's real local
contraction factor ρ(∂ₕR) — `compute_jacobian_reg_loss` is a single-sample
finite-difference proxy used only as a *training loss term*. This sweep:
7 beta values (0, 0.5, 1, 2, 5, 10, 20) × 3 seeds, each scored on:
in/extrapolation accuracy, a linear-probe R² (count decodability from the
frozen final hidden state, fit by closed-form ridge regression — see below
for why this matters), Spearman(prediction, true count), and ρ(∂ₕR)
estimated via power iteration at 12 sampled points per model along real
trajectories (not just at h_0).

## Finding 1 (confirms + quantifies): beta_regularization does control ρ

Mean ρ drops from ~1.4-1.9 (beta≤2, not contracting) to ~0.2-0.4 (beta≥5,
genuinely contracting, c<1). Nobody had verified this before — the training
loss term was never checked against an actual post-hoc measurement.

## Finding 2 (a real confound in the original claim, quantified)

The original "0% extrapolation, both conditions" result is confounded by
**output label-space extrapolation**, not (only) internal state failure:
with train_len=15, 71.5-74.5% of test_len=45's true count labels *never
appear in the training set at all* (train labels: 0-11 or 0-12; test true
counts: 3-25). A perfect internal counter would still score ~0% exact-match
here because the classifier head was never trained on those classes. The
original presentation.md already qualitatively flagged this ("classification
label shifts") — this quantifies it precisely for the first time.

## Finding 3 (the interesting one): count information survives strong contraction almost perfectly

Controlling for finding 2 via a linear probe (ridge regression, h_T → true
count, fit/evaluated on held-out splits, independent of the untrained-for
classifier head): **probe R² ≥ 0.996 at every single beta value tested,
including beta=20 (ρ≈0.25-0.33, unambiguously contracting)**. If anything,
R² is *higher* under strong contraction (0.9997-0.9999) than under weak/no
contraction (0.998-0.999) — Spearman(ρ, R²) = −0.86, p=5e-7 across the 21
runs, i.e. statistically real but practically tiny (R² barely moves at all).
The raw classifier's predictions also become dramatically *more stable*
(not less informative) under contraction: Spearman(prediction, true count)
jumps from noisy 0.53-0.76 (std up to 0.42) at beta≤2 to a tight 0.963
(std~0.002) at beta≥5.

**This is the toy-model version of H3's mechanism not showing up where the
theorem's own worked example (contraction_proof.md §5) says it should.**

## Why, probably (a genuine gap in the theorem's applied argument, not just this implementation)

The proof (§§2-4, checked correct — Banach fixed point, `T_max ≤
log(D/ε)/log(1/c)`) is about **forgetting the initial state h_0** under
iterated contraction. But `TinyRecursiveModel.forward()` sets `h_0 = 0`
identically on every rollout — there is no h_0 variation to forget in the
first place. Information here is injected via `context` (pooled input
embedding), re-supplied at *every* recurrent step, not just at t=0. A
contraction toward a fixed point still lets that fixed point h*(context)
depend arbitrarily on context — contraction only kills dependence on the
arbitrary starting point, not on an input that keeps getting re-injected.
Section 5 of the proof ("Architectural Implications for Counting and
State-Tracking") leaps from "the theorem shows h_0-dependence decays" to
"therefore N-state counting/FSA tasks must fail under contraction" by
implicitly treating "the N historical states to distinguish" as if they were
encoded via different h_0 values — but nothing in the proof itself
establishes that a counting task's information has to be represented that
way rather than via context-dependence of the fixed point, which contraction
doesn't touch.

**Caveat, stated plainly:** Huginn's actual architecture *also* re-injects a
fixed context/prompt embedding `e` at every recurrent step (same pattern),
which is exactly why Geiping et al. built it to promote path-independence
(different random h_0 → same h*(e)) as a *feature*. So this isn't just a toy-
model quirk to fix — if the reasoning above is right, it suggests the
Contraction Bottleneck Theorem's proof (which is correct as stated) may not
actually predict what §5's applied claim says it predicts, for *this class
of architecture generally*, not only for the toy baseline. This is a
substantive claim worth the team's own scrutiny before it goes in the paper
either way — it was derived by reasoning, not proven with the same rigor as
the Banach argument itself, and someone should check it independently.

## Finding 4 (2026-07-18, follow-up): the mechanism DOES bite on the architecture the theorem actually assumes

Built `SequentialCountingRNN` (`sequential_rnn.py`): a classic Elman RNN,
`h_{t+1} = tanh(W_h h_t + W_x x_t)`, one input token consumed per recurrent
step — no re-injected context, the running count only exists if `h` carries
it forward. This is the architecture Section 5 of `contraction_proof.md`
implicitly assumes (history distinguishable only via evolving `h`), unlike
`TinyRecursiveModel`/Huginn (recurrent-over-*depth*, full context re-supplied
every step, per Finding 3 above). Same task, same train/test lengths (15/45),
same beta grid, same seeds, same jacobian-reg-loss mechanism, same rho
measurement and linear-probe methodology — the only variable changed is the
architecture class. Full sweep: `sequential_h3_validation.py` →
`h3_results/sequential_sweep.csv`.

**Result: a sharp phase transition, not a gradual decline.** At β=0 (mean
ρ≈1.46, mildly expanding, matching the depth-recurrent model's β=0 regime):
in-dist accuracy 92-93%, linear-probe R²=0.79-0.86 — already imperfect
(this architecture is a harder learning problem than the depth-recurrent one
even unconstrained, which matters for interpretation, see caveat below). At
**any** β≥0.5 (mean ρ collapses immediately to 0.01-0.14, strongly
contracting): in-dist accuracy craters to 22-27% (barely above a
majority-class-ish floor), **probe R² craters to ≈0.00** (mean −0.0007 to
+0.0021 across β=0.5-20 — statistically indistinguishable from "no
information," not a partial degradation), and stays there through β=20. The
pooled Spearman(ρ, R²) across the whole sweep is ≈0 (p=0.92) — not because
there's no relationship, but because it's a step function (cliff at some
c well below 1), not a smooth correlation the rank statistic is built to
catch; the by-β breakdown makes the cliff obvious where the pooled number
doesn't.

**Direct architecture-matched comparison, same ρ range [0.15, 0.6] on both
models:** the depth-recurrent model has 9 runs in that band, mean probe
R²=0.9998 (count info still ~perfectly recoverable). The sequential model has
**zero runs** landing in that band at all — its ρ jumps straight past it
between β=0 and β=0.5, which is itself informative: this architecture can't
sustain a *moderately* contracting regime, it's either non-contracting
(β=0) or crashed (any β>0), consistent with the theorem's own `T_max ∝
1/log(1/c)` shape predicting a cliff once `c` drops meaningfully below 1,
not a gentle slope.

**Caveat, stated plainly, so this isn't overclaimed either:** this
architecture is a harder learner to begin with (β=0 accuracy 92% and R²=0.83,
vs. the depth-recurrent model's clean 100%/0.9996) and reaches much smaller ρ
for the same nominal β coefficient than the depth-recurrent model does — the
two models' Jacobian-regularization losses aren't calibrated to hit the same
ρ at the same β, so this is a comparison of *architecture class under
contraction*, not a controlled "same β" comparison. The ρ-matched-band
comparison above is the fairer read, and it still shows the same qualitative
result (depth-recurrent: perfect recovery at ρ~0.15-0.6; sequential: no data
survives to that band without already being crashed) — but a cleaner version
of this experiment would tune each architecture's β range independently to
get real overlapping ρ coverage in the [0.15,0.6] band for both, rather than
observing that the sequential model skips over it.

## Bottom line across all 4 findings

The Contraction Bottleneck Theorem's proof (A1) is correct. Its applied claim
that contraction destroys counting/FSA state-tracking (§5) is **architecture-
dependent, not universal**, and this toy-scale evidence suggests it applies
to recurrent-over-*time* architectures (classic RNNs — where it's already
well-established in the literature this project itself cites, Merrill et
al. 2404.08819, Grazzi et al. 2411.12537) but does **not** straightforwardly
transfer to recurrent-over-*depth* architectures like Huginn, where the
theorem's own worked example implicitly assumes a different information-flow
pattern than the one these models actually have. If real, this is a
meaningful, citable clarification for the paper's H3 section — sharpening
"forcing contraction destroys state-tracking" to "...for architectures where
state-tracking depends on surviving h_0, which recurrent-depth transformers
with full-context re-injection are not obviously an instance of." Someone on
the team should scrutinize this reasoning independently before it goes in
the paper; it was derived and tested in one session, not peer-reviewed.

## Status

Both sweeps complete and saved (`h3_sweep.csv`, `sequential_sweep.csv`,
`h3_sweep.png`). Natural next steps, not done here: (a) a genuinely random
h_0 variant of the depth-recurrent model, to directly test whether adding
h_0-dependence (without changing the re-injected-context structure) is
sufficient to make the mechanism bite there too, isolating "random h_0" from
"recurrent-over-time" as the operative difference; (b) recalibrate each
architecture's β grid to get overlapping ρ coverage for a cleaner matched
comparison than the one above.
