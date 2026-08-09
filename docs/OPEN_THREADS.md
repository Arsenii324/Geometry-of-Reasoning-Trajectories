# Open threads — the scannable index

**What this is.** One line per followable thread, prioritised, with cost and what it
would settle. **It is an INDEX, not a store**: detail lives in `directions.md`
(sections A–G), evidence in `claims_ledger.md`. This file exists because
`directions.md` reached 800+ lines, and the failure it prevents is real — when many
threads are open and you follow one, the rest become invisible rather than
deprioritised.

**Rule for keeping it honest:** a thread leaves this file only by being *done*
(with the ledger row that closed it) or *dropped* (with why). Never by being
forgotten. Add a line the moment a thread is recognised, even one sentence.

*Updated 2026-08-09 evening. Status key: `open` · `running` · `blocked` · `done` · `dropped`.*

---

## A. Zero-GPU, actionable immediately

These need no compute. GPU is the bottleneck, so these are the highest
value-per-minute available.

| # | thread | cost | settles | status |
|---|---|---|---|---|
| A1 | **Reconcile our median-34 convergence depth against arXiv:2607.14427's median-6** using the readout-blind-spot mechanism (arXiv:2606.24898): a scale-invariant RMSNorm readout hides radial scale from the loss (radial-gradient fraction ~1e-8 vs ~1e-2 raw), so KL-on-output converges ~4–5× sooner than the latent. Neither paper is cited anywhere in the repo. | ~15 min, doc only | why D97's number differs from a published one without either being wrong; also a candidate mechanism for D30's unexplained bf16 4.6× | `open` |
| ~~A2~~ | **CLOSED by D106 — the H2 nulls are NOT a coordinate artefact.** The radial (chord) component is 20.6% of step length early, but removing it shifts the step cosine by only 5.0% relative and the participation ratio by 1.4% — 4.2% the size of D80's depth effect. Killed as pre-registered, though not for the predicted reason: the artefact is large, its consequence is not. | — | — | `done` |
| A3 | **G2.2 — decode task identity from the cycle RESIDUAL.** D100 found every cycle statistic invariant across tasks; if the cycle is a carrier, signal lives in deviations from the mean cycle. | ~1 h; block-resolved banks (6 + 12 prompts) | whether the learned cycle (D104) carries any task information at all | `open` |
| A4 | **G2.1 redesign — decode the gold VALUE across items at matched token count** from *h\** versus shape, on the lenmatch banks. The within-prompt version was mis-specified (*e* is fixed within a prompt, so *h\** is constant). | ~1 h on banked data | whether the computation lives in the fixed point rather than the path | `open` |
| A5 | **Re-score the capability axis at a FIXED depth** rather than the oracle `min(ranks)==1` (D103: 2.06× inflation, 22.7% → 11.0%). Pick the depth on held-out items. | ~1 h; rank curves are banked | what every accuracy number in the ledger actually means | `open` |
| A6 | **F2.8 — re-run the census pooling simulation under a BIMODAL prior.** My justification for pooling used Uniform(0,1); this repo's items are mostly pinned near 0/1, where the selection is far more informative and the curse may be larger. | ~20 min | whether `run_census_analysis.py`'s primary estimator is the right one | `open` |
| A7 | **F2.9 — `split_stages` mis-assigns stage-2 draws to stage 1 when a screening draw failed.** Matters only if the census had failed forwards; check `ok:false` rows first. | ~15 min | correctness of the winner's-curse block | `open` |
| A8 | **F2.12 — `related_work.md` gives one paper two verification statuses** and stretches `[V]` to cover a scouting read, which its own key calls `[U]`. | ~15 min | the verification ladder still discriminating (it is what prevents another Movahedi/Du) | `open` |
| A9 | **F2.13 — apply the D89 wording amendment** to `RESULT.md`/`UNDERSTANDING.md`: the capability axis behind the central null was first-token-scored for 8 of 21 families. | ~15 min | honest statement of the null's independent variable | `open` |
| A10 | **F2.10 — `run_h0_within.py`'s alpha is Bonferroni over 8 windows while 25 tests run.** The pre-registered P5 verdict stands on its own terms; D93(5)'s *secondary* multiplicity argument is vacuous and should be restated. | ~20 min | precision of a stated caveat, not a conclusion | `open` |

## B. GPU-light, high value

| # | thread | cost | settles | status |
|---|---|---|---|---|
| B1 | **Cross-prompt e-stream patching + a potency-vs-r calibration curve.** Reuses the validated hook mechanism in `scratch/kaggle_patch/`; only donor selection changes. **D95(4b) and two independent deep-research passes all name this as the next step.** The calibration curve is an instrument null: do not interpret any patching result before it. | ~1 Kaggle run | whether this project has ANY working causal instrument — currently `RESULT.md` says causal evidence is absent | `open` |
| B2 | **Jacobian eigenspectrum vs difficulty on the working ladder.** D31 swept 3 prompts and never against difficulty; `nth_item_k`/`addk` (D101) are the first length-constant ladders that make "does an eigenvalue approach 1 as difficulty grows" answerable. Reuses `scratch/kaggle_jacobian/`'s Arnoldi. | ~1 run, JVPs only | H3's line-attractor form, currently argued from 3 points | `open` |
| B3 | **Measure the FULL operator, not the diagonal block.** D31's own row ends: *"NEXT: measure the FULL operator … which is the version whose spectrum should match the observed orbit decay."* Never done. | ~1 run | the only route to a second, valid estimate of ρ — D94 currently has NO independent corroboration | `open` |
| B4 | **Powered re-run of D101's best_depth** on `nth_item_k` and `addk` only, pre-registered, single statistic, length gate enforced, correct unit of independence built in. | ~1 run | the project's only positive H2 direction (ρ=+0.437, p=0.033, not clearing correction) | `open` |
| B5 | **Implicit-differentiation attribution** via (I−J)⁻¹∂F/∂e as a Neumann series (~30–40 JVPs, geometric convergence from ρ≈0.85). Unclaimed in the literature per the DR pass. | ~1 run | exact input attribution without BPTT; cross-validates B1 | `open` |
| B6 | **Permutation / group-composition task family (S₃–S₅, A₅).** Provably requires recurrence under TC⁰≠NC¹ — the one task class theory says *should* show H2 if anything will. Never built. | build + 1 run | H2 on the theoretically privileged task | `open` |

## C. Running or blocked

| # | thread | status |
|---|---|---|
| C1 | `geometry-census` — the observability census (which items have dynamic range) | `running` |
| C2 | `geometry-clrs` — CLRS-Text screen. **Caveat when it lands:** its `body.py` has no few-shot exemplars and no strict answer prefix, which DR1 says to add before concluding CLRS is a floor for Huginn | `running` |
| C3 | **Merge `claude/geometry-remote-continued`.** Holds a result we lack ("depth relocates the answer", with a no-GPU regeneration script and 11 tests) and the `'4Huginn'` role-marker parsing hazard. **Ledger numbering collision: its D92–D94 differ from ours — a merge must RENUMBER, not overwrite.** | `open` |
| C4 | `scratch/kaggle_answerpos/` — built, linted, tested by the remote session, never pushed | `open` |

## D. Closed today (kept briefly so the index shows motion)

| # | thread | closed by |
|---|---|---|
| D1 | Is D98's block cycle learned or architectural? | **D104** — learned; untrained fixed points are coincident (0.007 vs 0.533 of the state norm), arms disjoint on 3/3 robust statistics |
| D2 | Does D91's transient-window correctness lead replicate? | **D93** — NOT CONFIRMED, under a floor shown to detect a planted 0.5 sd effect 100% of the time |
| D3 | Does H2 appear on the newly-found cycle? | **D100** — no; 0 of 15 cells |
| D4 | Is the state confined to a sphere? | **D99** — yes, ‖h‖ = 76.386 to 4 s.f.; makes the settle/loop/drift trichotomy exhaustive |
| D5 | Is our accuracy definition an oracle over depth? | **D103** — yes, 2.06× inflation; independently replicated by the remote session |
