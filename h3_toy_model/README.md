# H3 toy-model validation

Imported into this repo 2026-07-22. Previously lived outside git entirely
(`~/build-projs/barannikov-work/src/`, a sibling directory to this repo) —
invisible to any clone or push, at real risk of being lost with no backup.
Brought in as-is because it's a genuine, self-tested, substantive finding
(see `h3_results/FINDINGS.md`), not because it's finished or fully reviewed.

**What this is**: a from-scratch empirical test of H3 (the Contraction
Bottleneck Theorem) on toy models this project can actually run without a
GPU or Huginn access — a standalone side investigation, separate from the
main `src/traj_geom/` package, sharing no code with it.

**Headline finding** (full detail + caveats in `h3_results/FINDINGS.md`):
a recurrent-over-**depth** toy model (same information-flow topology as
Huginn — full context re-injected every step) preserves count information
almost perfectly even under strong contraction (linear-probe R² ≥ 0.996
across the whole tested range), while a recurrent-over-**time** counterpart
(classic RNN, one token per step — the architecture the theorem's own
worked example assumes) shows a sharp collapse (probe R² craters to ≈0.00
once contraction crosses a threshold). Suggests the theorem's applied claim
about state-tracking may be architecture-dependent rather than universal —
a real, citable point for this project's H3 section, **not yet
independently reviewed by the team**, as `FINDINGS.md` itself says.

**How much this actually tells you about real Huginn — read before citing.**
`TinyRecursiveModel` (`train_tiny_recursive.py`) shares Huginn's abstract
information-flow *topology* (context re-injected at every recurrent step)
and nothing else. Concretely, it diverges from real Huginn in every way
that could matter empirically:
- No attention — `core_layer` is a plain `Linear→LayerNorm→SiLU→Linear`
  residual MLP, not a transformer block. Different function class.
- `context` is the *mean* of token embeddings — token order is discarded
  before the recurrence starts. Fine for "count the 1s"; not a stand-in for
  language reasoning.
- Binary vocab (`vocab_size=2`), not a real tokenizer.
- `h_0 = torch.zeros(...)`, always — not random. Huginn's actual random-`h_0`
  init (meant to induce path-independence) is untested here; noted as an
  open gap in `FINDINGS.md`'s own "Status" section.
- `num_steps` is **fixed at 16 during training**, not randomized. This does
  not replicate Huginn's actual random-depth training scheme at all — this
  toy model is not evidence one way or the other on whether that scheme
  yields difficulty-calibrated stopping (see
  `docs/open_question_depth_calibration.md`).
- Contraction is force-fed via an explicit added loss term
  (`compute_jacobian_reg_loss`, weighted by `beta_regularization`). Nothing
  indicates Huginn's real training has any such term — so "does state
  survive as beta rises" tests *"if you force a depth-recurrent model to
  contract, does state survive,"* not "is Huginn actually in a contracting
  regime."
- Scale: `hidden_dim=64` vs Huginn's 5280.

Net: treat this as a minimal-counterexample-style check on whether the
theorem's §5 argument is architecture-universal in principle — useful for
that, and only that. It is not empirical evidence about real Huginn's
actual dynamics, training regime, or scale. Only running `spectral.py`'s
estimator against real Huginn hidden states would be that evidence, and
that hasn't been done.

**Status of the code**: `spectral.py`'s power-iteration estimator is
self-tested (`test_spectral.py`, 5/5 passing, reverified from this location
before commit). `h3_validation.py` / `sequential_h3_validation.py` are the
actual sweeps that produced `h3_results/h3_sweep.csv` /
`h3_results/sequential_sweep.csv`. `instrument_huginn.py` is unused by the
sweeps — a separate, apparently-unfinished attempt at extracting real
Huginn trajectories (predates and duplicates what `src/traj_geom/extraction/hook.py`
now does properly); kept for reference, not imported by anything else here.

**Why it matters for this project specifically**: `spectral.py`'s
operator-norm estimator is the one piece of infrastructure that could
actually run H3's contraction check against *real* Huginn hidden states —
currently done only on the toy model. That's a real, flagged, still-open
next step (see `docs/architecture_state.md` / `docs/open_question_depth_calibration.md`),
not attempted here.
