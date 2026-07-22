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
a recurrent-over-**depth** toy model (same information-flow pattern as
Huginn — full context re-injected every step) preserves count information
almost perfectly even under strong contraction (linear-probe R² ≥ 0.996
across the whole tested range), while a recurrent-over-**time** counterpart
(classic RNN, one token per step — the architecture the theorem's own
worked example assumes) shows a sharp collapse (probe R² craters to ≈0.00
once contraction crosses a threshold). Suggests the theorem's applied claim
about state-tracking may be architecture-dependent rather than universal —
a real, citable point for this project's H3 section, **not yet
independently reviewed by the team**, as `FINDINGS.md` itself says.

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
