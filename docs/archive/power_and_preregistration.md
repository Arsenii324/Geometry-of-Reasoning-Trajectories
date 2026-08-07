> **FROZEN 2026-08-05.** Archived under CLAUDE.md §6 (three live documents only).
> Not maintained. Superseded by `docs/UNDERSTANDING.md`, `claims_ledger.md`, `directions.md`.
> Statements here may be contradicted by later work — several assert things since disproven
> (e.g. "H1 was never tested": it was, and the test itself was invalid — see D56, D65).

# Power analysis + pre-registration (project_plan.md §0.1/§7)

STATUS: written 2026-07-24, Phase 0's last item (§15 item 4). This is the
"honesty gate" the plan requires before any further GPU spend on H2: a
power calc computed *before* the keystone pass, and a frozen family of
tests it will be scored against, so results can't be cherry-picked after
the fact.

**CORRECTED same day** after downloading and reading Blayney et al.'s actual
PDF (the first version of this doc relied on a WebFetch HTML summary of
the paper that conflated two different quantities). The original "2.81%
optimistic ceiling" was Table 4's per-EXAMPLE rate ("does *any* question
token in this example show the behavior"), not a per-token rate —
mechanically much larger than any single token's real probability because
it aggregates over every token in the example. The correct per-token
ceiling, from Table 3 directly, is **0.14%** (Huginn, "Long Persona"
system prompt). This changes the "optimistic" N-needed-for-bar from 178 to
**~3,571** and every downstream keystone-scale recommendation below.
See `claims_ledger.md` B9 for the full table.

Script: `scripts/run_power_analysis.py` (0-GPU). Data: `results/power_loop_rate.csv`,
`results/power_curve.csv`. Verified live 2026-07-24 (post-correction); re-run to reproduce.

---

## 1. The two power problems, separately

This project has two different, easily-conflated power questions. Answering
one does not answer the other.

**(A) Can we expect to observe a genuine winding LOOP at all**, at the rates
the literature reports for this architecture class? This is a Poisson-style
expectation over (token, trajectory) draws, gated by Blayney et al.
(arXiv:2604.11791 Table 3, `claims_ledger.md` B9): only **0.02%** of tokens
are non-fixed-point with no system prompt, rising to **0.14%** under their
"Long Persona" system-prompt condition — both PER-TOKEN rates, from the same
table, directly comparable to each other (unlike the paper's separately-
reported per-example "2.81%" figure, which answers a different question and
must not be used here — see the correction note above).

**(B) Given N task-difficulty "levels" (4-10, every synthetic task in this
project), can a Spearman test detect a real winding-vs-depth correlation** if
one exists? This is ordinary small-sample correlation power, independent of
whether any single trajectory ever loops.

## 2. (A) Loop-rate power

Real numbers, computed live from the project's own cached data (not
estimates): **1,294** real Huginn extractions currently carry a
winding/shape classification, project-wide. Excluding the artificially
starved `forceloop.csv` sweep (`num_steps=16`, the only in-project condition
that ever induces a loop by construction), the remaining **1,198** real,
full-compute-budget extractions show **zero** loop or drift trajectories —
every single one settles.

| Rate | N needed for E[loops] >= 5 |
|---|---|
| Blayney baseline (0.02%) | **25,000** draws |
| Blayney Long Persona (0.14%) | **~3,571** draws |

At the *current* pool (1,294), E[loops] = 0.26 under the baseline rate and
E[loops] = 1.81 under the Long Persona rate — both correctly predict "expect
to see zero or very few," which is exactly what was observed (P(zero events)
under Poisson(1.81) is ~16%, unremarkable). **The project's near-total
absence of loops in real data is consistent with a true, undetected H2
effect under either Blayney rate — it is not evidence against H2.**

**What would it take to clear the bar?** A keystone all-token extraction
(project_plan.md §5) turns every prompt's *entire* token sequence into
(token, trajectory) draws, not just the answer token — multiplying the pool
by ~S (tokens/prompt) at zero extra GPU cost (the unrolls are already
computed for every token; only the return statement currently discards them).

| Scenario | Draws | E[loops] @ baseline | E[loops] @ Long Persona | Clears bar of 5 @ baseline? |
|---|---|---|---|---|
| Current answer-token pool (real) | 1,294 | 0.26 | 1.81 | No |
| Keystone, 500 prompts x 30 tok | 15,000 | 3.0 | 21.0 | No |
| Keystone, 500 prompts x 100 tok | 50,000 | 10.0 | 70.0 | **Yes** |
| Keystone, 1000 prompts x 50 tok | 50,000 | 10.0 | 70.0 | **Yes** |
| Keystone, 1000 prompts x 100 tok | 100,000 | 20.0 | 140.0 | **Yes** |

**Recommendation for Phase 1 scale** [C, see §4]: the keystone pass needs
**at least ~500 prompts averaging >=100 tokens each** (or equivalent draws)
to have a reasonable expectation of clearing the baseline-rate bar. Shorter
synthetic prompts (the ~30-token end of this project's current range) do not
clear it even at 500 prompts under the pessimistic rate (though they would
under the Long Persona rate: 21 expected) — either lengthen the task
templates, extract more prompts, or accept the more optimistic assumption
knowingly [C].

## 3. (B) Depth-correlation detection power

Monte Carlo, 5,000 trials per (N, true rho) cell, bivariate-normal data,
`scipy.stats.spearmanr` on each simulated sample (the exact call every
`run_*.py` script makes — not a closed-form approximation).

| N levels | rho=0.3 | rho=0.5 | rho=0.7 | rho=0.9 |
|---|---|---|---|---|
| 4 (structurally unreachable, D7) | 0.10 | 0.14 | 0.21 | 0.41 |
| 6 (counting/maxtask/switch/dissociation) | 0.09 | 0.16 | 0.30 | 0.66 |
| 8 | 0.10 | 0.22 | 0.45 | 0.85 |
| 10 | 0.13 | 0.29 | 0.59 | 0.94 |
| 20 | 0.22 | 0.57 | 0.91 | 1.00 |
| 30 | 0.32 | 0.77 | 0.99 | 1.00 |

Full grid (11 N values x 9 rho values) in `results/power_curve.csv`.

**Reading it plainly:** at N=6 (the modal case — every current synthetic
task except pararule uses 6 levels), even a *strong* true effect (rho=0.7)
is detected only 30% of the time; conventional 80% power requires N>=20
levels at that effect size, or a near-perfect rho>=0.9 at N=6. **No synthetic
task in this project currently has enough levels to reliably detect anything
short of a very strong effect.** This is a second, independent reason (beyond
the loop-rate problem in §2) that a null H2 result on the current task
designs cannot be read as evidence against H2 — it is underpowered by
construction, not just by luck.

## 4. Frozen pre-registered test family for Phase 1/2

The point of pre-registration is that this list is fixed *before* the
keystone pass's data exists, so nothing below can be added post-hoc because
it happened to come back significant. Additions after Phase 1 data lands
must be labeled exploratory, analyzed separately, and not pooled into this
family's FDR correction.

**H1 (settle/loop/drift, three-instrument agreement):**
1. `winding_null_test` on every real winding from Phase 1 (incl. force-loop's
   existing 13 loops) vs matched-random-walk null (§Phase 2.1).
2. All-token shape census: shape-fraction by position class x budget (§2.3).
3. Native geometric exit-criteria (latent-diff, cosine) vs correctness (§2.5).

**H2 (loops track depth, conditional on looping):**
4. Winding-vs-depth on the redesigned three_scale (constant length, constant
   answer position, §9) — the first clean H2 test this project will have run.
5. Winding-vs-depth on PARARULE extended to d<=6, if the loader extension
   lands [C].
6. QK-alignment-vs-depth, once the statistic is locked with the curator [C]
   (§8.2) — a separate family from 1-5 if it lands after them, per the
   "don't pool the ~7,000 QK tests with the ~50" rule (§9).

**H3 (contraction forces looping):**
7. Joint spectral radius sigma_max on real Huginn (§3.1) — a measurement, not
   a significance test; reported, not FDR-corrected.
8. Linear count/state probe (ridge, held-out R2) vs a shuffled-dynamics
   control and a length-only baseline (§2.4).

**Cross-cutting validity checks (not hypothesis tests, but gate
interpretation of 1-8):**
9. Geometry-vs-activations discriminator (§2.6) — do geometric features
   *alone* predict task identity/correctness, above the raw-state probe.
10. Path-independence null across init seeds (§2.7).

Items 1-6 (excluding 6 if QK lands separately) form ONE BH-FDR family when
Phase 1 data lands, following the project-wide precedent set in
`claims_ledger.md` D13. Items 7-10 are not p-value tests and are not part of
that family.

## 5. Hard rules this pre-registration establishes

- **An underpowered null on H2 must be reported as "untestable at this N,"
  never as "H2 refuted."** Both §2 and §3 independently show the current
  task designs cannot detect a moderate true effect even if one exists.
- **Blayney's per-example "2.81%" figure (Table 4) must never be used as a
  per-token/per-draw rate** — it answers a different question ("does any
  token in this example show the behavior") and is not comparable to the
  0.02%/0.14% per-token rates (Table 3) this document's arithmetic depends
  on. This mistake was made once already in this document's first version
  and corrected same-day; see the top-of-file changelog note.
- **The "≥5 genuine loops" bar** (project_plan.md §12 point 5) is used
  throughout this document as the working definition of "enough to analyze
  at all" — proposed here, not yet confirmed by the curator. [C]
- Any test added to the H1/H2/H3 analysis after Phase 1 data exists that is
  not in the §4 list is exploratory and must be reported as such, with its
  own separate (and honestly labeled, likely uncorrected-for-multiplicity)
  significance claim — not folded into the pre-registered family's q-values.
