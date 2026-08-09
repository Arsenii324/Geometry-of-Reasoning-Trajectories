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

**And before calling a thread unrun, SWEEP `directions.md`'s run table and
`scratch/*/out/` for it.** D119: B3 sat here as *"Never done"* for a day while the
run answering it was banked in `scratch/ds_eigen/out/`, filed in `directions.md`
under its run name (`geom-eigenplane`) rather than its thread name — and GPU was
spent partly to obtain corroboration that already existed. Indexing by thread name
alone is what made it invisible.

*Updated 2026-08-09 evening. Status key: `open` · `running` · `blocked` · `done` · `dropped`.*

---

## A. Zero-GPU, actionable immediately

These need no compute. GPU is the bottleneck, so these are the highest
value-per-minute available.

| # | thread | cost | settles | status |
|---|---|---|---|---|
| ~~A1~~ | **CLOSED by D112, and without needing the published number.** The gap is measured INSIDE single orbits: on solved orbits the answer is top-ranked at median unroll 4 with 86.9% of the state's journey still ahead, while the state reaches residual<0.01 only at unroll 44. An output-read metric and a latent-read metric are both right about different objects, at roughly the 4–5× ratio the literature reports. | — | — | `done` |
| ~~A2~~ | **CLOSED by D106 — the H2 nulls are NOT a coordinate artefact.** The radial (chord) component is 20.6% of step length early, but removing it shifts the step cosine by only 5.0% relative and the participation ratio by 1.4% — 4.2% the size of D80's depth effect. Killed as pre-registered, though not for the predicted reason: the artefact is large, its consequence is not. | — | — | `done` |
| A3 | **G2.2 — decode task identity from the cycle RESIDUAL.** D100 found every cycle statistic invariant across tasks; if the cycle is a carrier, signal lives in deviations from the mean cycle. | ~1 h; block-resolved banks (6 + 12 prompts) | whether the learned cycle (D104) carries any task information at all | `open` |
| ~~A4~~ | **CLOSED by D109 — the fixed point decodes the answer, the shape does not.** Dimension-matched: h\* 0.647 vs shape 0.300 (chance 0.367), 5/5 cells. Explains why every shape-based correctness null failed — wrong object. Converges with D108 causally. | — | — | `done` |
| A4b | **Bank paired items where BOTH markers span the SAME gold range.** D121: 92 paired items are already banked but only `first_vs_last` has overlapping gold ranges, giving 84 usable pairs and z = 1.45 (two-sided p = 0.150); pooling the other templates produces a 5.0-sd ARTEFACT pointing the wrong way. 64 items per marker with golds balanced 0–9 takes z to ~3.2. | ~1 GPU run | whether h\* encodes the ANSWER or merely fine input detail | `queued` |
| ~~A5~~ | **CLOSED by D107.** Honest per-family held-out depth gives 22.7% → 13.7% (1.66×); inflation is family-dependent (1.1× `addk` … 4.8× `nth_item_k`) and **reverses D100's ladder recommendation: build on `addk`, not `nth_item_k`.** All selected depths are shallow (0–4). | — | — | `done` |
| ~~A6~~ | **CLOSED — pooling survives a bimodal prior.** Re-simulated at 40k trials under Uniform, Beta(0.3,0.3) and the census's own empirical accuracy distribution: pooled RMSE 0.0897 / 0.0859 / 0.0869 against stage-2-only 0.1000 / 0.0944 / 0.0966 — **pooled wins on error under all three**. The worry was partly right: pooled bias grows ~20× under the empirical prior (+0.0061 vs −0.0002), still negligible against the band width. Recorded in the script's docstring where the choice lives. | — | — | `done` |
| A7 | **F2.9 — `split_stages` mis-assigns stage-2 draws to stage 1 when a screening draw failed.** Matters only if the census had failed forwards; check `ok:false` rows first. | ~15 min | correctness of the winner's-curse block | `open` |
| A8 | **F2.12 — `related_work.md` gives one paper two verification statuses** and stretches `[V]` to cover a scouting read, which its own key calls `[U]`. | ~15 min | the verification ladder still discriminating (it is what prevents another Movahedi/Du) | `open` |
| A9 | **F2.13 — apply the D89 wording amendment** to `RESULT.md`/`UNDERSTANDING.md`: the capability axis behind the central null was first-token-scored for 8 of 21 families. | ~15 min | honest statement of the null's independent variable | `open` |
| A10 | **F2.10 — `run_h0_within.py`'s alpha is Bonferroni over 8 windows while 25 tests run.** The pre-registered P5 verdict stands on its own terms; D93(5)'s *secondary* multiplicity argument is vacuous and should be restated. | ~20 min | precision of a stated caveat, not a conclusion | `open` |

| ~~A11~~ | **CLOSED by D116.** The argument IS family-level (38.15°–82.70° over 21 families, between/within 1.84×), independent of the modulus (ρ=+0.114), and nearly free of the length confound (+0.165 vs the modulus's +0.559). Trained rotates 42–59°/unroll, untrained 110–115°. Difficulty moves neither component. | — | — | `done` |
| A13 | **Dequeued unrun-again: does the readout lock in earlier for lookup tasks than computed ones?** Run once (ρ = −0.328, p = 0.266, n = 13 families with ≥5 solved orbits) and NOT supported, but the floor at n=13 is weaker than D117's already-modest one. Re-running on the same bank cannot settle it; it needs more families with high solve rates, which is what `geometry-census` is for. | blocked on census | whether early lock-in marks retrieval vs computation | `dropped for now` |
| A12 | **D114's second arm at p = 0.080:** harder `addk` problems may contract FASTER (ρ = −0.453, n = 16). Underpowered, not a claim. B4b's fixed-sum ladder plus one more e-stream run would settle it. | ~1 GPU run | whether difficulty touches the dynamics at all, on a second ladder | `open` |

| ~~A14~~ | **CLOSED by D122 — not a bug, a geometric identity.** The difference of two same-radius states is orthogonal to their midpoint by construction (measured |cos| = 1.9e-4), so the tangent projection is a no-op and the two columns must agree. The same identity shows D106's '20.6% radial component' is \|d\|/(2R) — a restatement of step length, ratio 1.000 to prediction. | — | — | `done` |

## B. GPU-light, high value

| # | thread | cost | settles | status |
|---|---|---|---|---|
| ~~B1~~ | **CLOSED by D108 — the instrument WORKS.** e-stream patching flips the preferred answer in 42/56 measurements (75%), mean potency +3.97 nats, and is flat in r (87% of the effect with only 8 unrolls left). My pre-registered "potency falls with r" is refuted: the fixed point is reached almost immediately. **The contrasting STATE arm is void by my own design flaw** (persistent overwrite + donor state captured at unroll 0) — see B1b. | — | — | `done` |
| ~~B1b~~ | **CLOSED by D111 — the dissociation is total AND yields a causal rho.** e-arm flat in r (4.04 nats, 6/8 flips at every r incl. r=40); state arm 0/56 flips but |potency| decays as rho^(R-r) over three orders of magnitude, giving **rho = 0.822** — matching D94's bias-corrected ~0.82 and D31's Arnoldi 0.79-0.81 from a different instrument class. | — | — | `done` |
| ~~B1c~~ | **CLOSED by D113 — CONFIRMED, and it corrected my own headline.** Median rho = 0.8335 over 16 distinct recipients, sd 0.028, 14/16 inside the passive bracket: D94 has its corroboration. But the e-arm flips 35/112 = 31.2% at +1.598 nats, not D108's 75% at +3.97 — that was a single-recipient artefact, inflated 2.4×. | — | — | `done` |
| ~~B1c-orig~~ | **Replicate that rho across RECIPIENTS.** B1b's 8 pairs all shared one recipient (`pairs[:N]` off a loop whose outer index is the recipient), so rho described one fixed point. 16 stratified recipients, varied donors, VOID below 12, fit on \|potency\| not the signed mean that cancelled. | running | whether rho=0.822 is a property of Huginn or of one prompt — and whether D94 finally has corroboration | `running` |
| ~~B4b~~ | **VOID by D118 — the design worked, the task did not.** Orthogonality achieved (ρ(k,gold)=+0.027, n_tokens==31 everywhere) but accuracy was 54.2% at k=1 and **0.0% at k=2–5**; the gold never reached rank 1 in 348 forwards. Capability bound recorded: Huginn cannot add two or more nonzero digits. D110 is now un-replicated, not contradicted. | — | — | `done (VOID)` |
| B4c | **The same orthogonal-ladder construction over a task Huginn CAN do at ≥2 difficulties.** Excluded so far by measurement: multi-term addition (D118, 0% at k≥2), `caesar_k` (~0% everywhere), `nth_item_k` (identical prompts). `geometry-census` is the instrument that will name the survivors. | blocked on census | whether D110's reversal is Huginn or `addk` | `blocked` |
| B4b-orig | ~~a decollinearised difficulty ladder~~. D110's reversal is post-hoc on one family, and `addk`'s `gold = v + k` leaves only 2 d.o.f. `caesar_k` is clean but sits at ~0% accuracy (checked); `nth_item_k`'s generator makes items 0,2,4 the SAME prompt. A ladder with difficulty genuinely independent of the answer token does not yet exist. | build + 1 run | whether harder-answered-earlier is Huginn or `addk` | `open` |
| ~~B1-orig~~ | **Cross-prompt e-stream patching + a potency-vs-r calibration curve.** Reuses the validated hook mechanism in `scratch/kaggle_patch/`; only donor selection changes. **D95(4b) and two independent deep-research passes all name this as the next step.** The calibration curve is an instrument null: do not interpret any patching result before it. | ~1 Kaggle run | whether this project has ANY working causal instrument — currently `RESULT.md` says causal evidence is absent | `open` |
| B2 | **Jacobian eigenspectrum vs difficulty on the working ladder.** D31 swept 3 prompts and never against difficulty; `nth_item_k`/`addk` (D101) are the first length-constant ladders that make "does an eigenvalue approach 1 as difficulty grows" answerable. Reuses `scratch/kaggle_jacobian/`'s Arnoldi. | ~1 run, JVPs only | H3's line-attractor form, currently argued from 3 points | `open` |
| ~~B3~~ | **CLOSED by D119 — and it was already done before this file called it unrun.** `ds_eigen` (job `bt1mvf3juvrcjpl6kl1i`, banked 2026-08-08) runs Arnoldi on the FULL unroll map (adapter + all four core blocks): **\|λ\| median 0.8098 (0.8011–0.8680)**, a fourth agreeing estimate, and **arg λ separates tasks** (`track` 61–63°, `local` 19.5–24.3°). It also retracts D116's naming of the step angle as arg λ — the orbit turns at a median **0.241×** the eigenvalue's rate. | — | — | `done` |
| B4 | **Powered re-run of D101's best_depth** on `addk` FIRST (D107: it is the ladder that survives honest scoring at 38.1%; `nth_item_k` collapses to 4.2%), pre-registered, single statistic, length gate enforced, correct unit of independence built in. | ~1 run | the project's only positive H2 direction (ρ=+0.437, p=0.033, not clearing correction) | `open` |
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
