> **FROZEN 2026-08-05.** Archived under CLAUDE.md §6 (three live documents only).
> Not maintained. Superseded by `docs/UNDERSTANDING.md`, `claims_ledger.md`, `directions.md`.
> Statements here may be contradicted by later work — several assert things since disproven
> (e.g. "H1 was never tested": it was, and the test itself was invalid — see D56, D65).

# Architecture & experiment state — living doc

> **⚠ PARTIALLY SUPERSEDED — read `docs/state_of_knowledge.md` first.**
> This file predates the two-pass rigor audit of 2026-07-25/26
> (`docs/rigor_audit.md`, `claims_ledger.md` D24–D26) and contains at least one
> claim the audit overturned. Known overturns that may appear below:
> `steps_settle` is a proxy for the contraction rate, **not** for effective
> compute; the "82.1% wind less than their null" result rests on an
> off-manifold null and is deleted; `cos = −0.276` is a noise-regime artifact
> (the computing-regime value is **+0.084**); "homology max persistence 0.009"
> is a normalisation artifact; and the "Movahedi et al." citation could not be
> located. Nothing here has been deleted — this notice is additive, and the
> text is left as its authors wrote it.


Started 2026-07-22. Distinct from the other docs in this folder:
`narrative.md`/`research_log.md`/`claims_ledger.md` explain and evidence
*findings*; `infra.md` is the original (2026-07-18) architecture reference.
This doc tracks **current inventory** — what file does what, what experiment
is in what state, what decisions were made and why — kept up to date as the
project moves, not a one-time writeup. Update it in place; don't fork a v2.

Not a replacement for `docs/START_HERE.md` — that's still the reading-order
index. This is the "where is everything / what's the state of everything"
reference to check before touching a file or asking "has this been run."

## Verification practice — this doc is checked, not just trusted

Prose documentation only stays true if someone remembers to update it every
time something changes. This project has repeatedly found real drift that
remembering-to-update didn't catch: the length-confound methodology defect
(D10), Gemini's coda-skip bug, `run_three_scale.py`'s `spearman()`
tuple-format crash that only surfaced by actually re-running it. So the
inventory claims in this doc are backed by an automated check, not just
this prose: `tests/test_architecture_consistency.py`, run every time
`pytest` runs, asserts —
1. every `results/*.csv` is named in this file (catches undocumented or
   orphaned data — this is exactly how `dissociation_results.csv` and
   `h2_loops.csv` were found, both committed in the project's first
   commits with no current producing script, sitting undetected until
   this check existed),
2. every `scripts/run_*.py` is named in this file by exact filename (a
   glob reference like "run_dissociation*.py" doesn't satisfy it),
3. no `results/<name>.csv` is committed *before* the last commit to the
   `scripts/run_*.py` documented as producing it — a script that changed
   more recently than its own cached output is a real "does this data
   still reflect what the code does" question.

**What it deliberately can't automate**: whether a flagged script change
was *substantive* (changed the actual computed values) or not (comments,
resilience, unrelated refactor) — that needs a human/AI judgment call, not
a timestamp diff. Reviewed-and-judged-safe cases are recorded explicitly
in `tests/test_architecture_consistency.py`'s `_REVIEWED_NON_SUBSTANTIVE_CHANGES`
dict, each with the commit and a stated reason — an audit trail, not a
silent suppression. Remove an entry (making the check fire again) the
moment its script changes for a reason that might actually matter.

It also can't automate whether an *interpretation* of a CSV (a
`claims_ledger.md` row, a narrative.md paragraph) has itself gone stale
relative to the data or the analysis code that reads it. The manual
complement: load-bearing numeric claims should cite the exact commit they
were computed at (see `claims_ledger.md` D11's `results/three_scale.csv`
citation) — not because that's automatically checked, but so a future
reader can `git log <hash>..HEAD -- <script>` themselves and judge whether
anything relevant changed since, rather than trusting the number on faith.

## Repo layout

```
Geometry-of-Reasoning-Trajectories/
├── src/traj_geom/
│   ├── extraction/    hook.py (extract_trajectory), model.py (load_huginn)
│   ├── metrics/       winding.py, dynamics.py, homology.py, convergence.py, projection.py
│   ├── shapes/        synthetic.py (task generators), gate.py (classify_shape)
│   ├── analysis/      correlate.py (spearman/partial_spearman), plots.py
│   ├── data/          loaders.py (PARARULE-Plus loader/enrich)
│   ├── constants.py   MODEL_ID/REVISION, HIDDEN_DIM, DEFAULT_NUM_STEPS
│   └── eval.py        generate_with_adaptive_compute-based accuracy eval
├── scripts/           one run_*.py per experiment (see table below); _common.py has cached()/load_model()
├── tests/             pytest, one file roughly per src module
├── results/           cached CSVs — see "results/ inventory" below, not everything referenced elsewhere lives here
├── figures/           generated plots
├── notebooks/         01_mvp_h2.ipynb (source of most extraction/metrics code), 02_two_scale.ipynb (two-scale branch only)
├── docs/              this file + narrative/research_log/claims_ledger/infra/guide/end_to_end_guide/START_HERE
├── scratch/            untracked, throwaway — Kaggle harness (kaggle_v6_probe/), pulled Kaggle outputs (kaggle_output_N/)
└── archive/ (parent dir, not in this repo) — pre-defense task.md, slide decks
```

`code_env_info/` — **sibling directory** to this repo
(`~/build-projs/barannikov-work/code_env_info/`), **not tracked by this
repo's git**. Anyone cloning only this repo does not get it. Contents:
- `kaggle-cli-guide.md` — full Kaggle CLI reference (push/status/logs/output,
  GPU selection gotchas, T4 vs P100 sm_60 CUDA break, submission shapes).
- `yandex-cloud-smiles-access.md` — Yandex Cloud DataSphere access notes for
  the team's shared grant. Re-checked live 2026-07-24: original project
  `Online_project24_2` (id `bt14qn4u9t3n09nfjoqu`) shows unit balance
  **5,000,000** — same round number as before, even though 2 real jobs ran
  on it (1 SUCCESS, 1 ERROR, `bt1csk570bg4jaq6tdme`/`bt1n02qvocurm6dsbvme`,
  58s/58s on `c1.4` CPU config, ~232 units estimated cost each). **Don't
  read "5,000,000" as "literally untouched"** — real compute did run, the
  balance display may just round/lag below ~0.01% of the total budget; no
  way found yet to query exact spend to more precision. Also found (same
  day): a SECOND community (`smiles2026`, id `bt1qioruc9l0ev470t53`,
  created 2026-07-22) with a new project `Online_Project19_3`
  (`bt1730riliibkg6u7fhb`) not previously visible — its `:unitBalance`
  endpoint returns empty `{}` (zero/unallocated, not "more credits yet").
  Worth re-checking later. Deployment mechanism is DataSphere Jobs
  (`pip install datasphere`, `datasphere project job execute -p <id> -c
  config.yaml`) or JupyterLab in-browser. **Open gap**: actual per-hour/
  per-job unit burn rate for this project's GPU config was never queried
  (pricing formula is on the docs page, not looked up) — don't launch a real
  job against the shared quota without checking that first.
- `local_envs_venvs.md` — stub, still just "TODO".

## src/traj_geom/shapes/synthetic.py — task generators

| Function | seq_len vs n_ops | Notes |
|---|---|---|
| `make_counting_task`, `make_switch_task`, `make_max_task`, `make_variants` | deterministic (rho=1.0) | original (notebook-derived); length-partial control is mathematically degenerate for all of these — see `claims_ledger.md` D10 |
| `make_count_ones_task`, `make_projection_task` | deterministic (rho=1.0) | added 2026-07-19, same confound as above, now documented in-docstring |
| `make_three_scale_task` | **independent** (irrelevant_len / neutral_len / active_len vary separately) | added 2026-07-19; decouples length from difficulty, but has its own known prefix confound — `claims_ledger.md` D11 |
| `make_three_scale_modk_task` | **independent, AND total_len constant by construction** (neutral_len is derived) | added 2026-07-24; fixes D11's prefix confound properly (no distinguished prefix region, answer position provably constant across a sweep); answer is `active_len % modulus`, single-token for any active_len; generalises `make_switch_task`'s mod-2 to modulus 2..10. Not yet run on real Huginn — feeds Phase 1 (`project_plan.md` §5/§9). |

## scripts/ inventory (one experiment each unless noted)

| Script | Experiment | Uses `cached()`? | results/ output |
|---|---|---|---|
| `run_pararule.py` | E1 — PARARULE-Plus depth vs winding/steps | yes | `pararule.csv` |
| `run_counting.py` | E2 — counting task, length-partial control | yes | `counting.csv` |
| `run_switch.py`, `run_maxtask.py` | switch/maxtask analogues of E2 | yes | `switch.csv`, `maxtask.csv` |
| `run_accuracy.py` | correctness via `generate_with_adaptive_compute` + regex parse | yes | `counting_accuracy.csv` |
| `run_dissociation.py` | E3, the length-matched track-vs-local dissociation | yes | `dissociation.csv` (--seeds 5) or `dissociation_15seed.csv` (--seeds 15) |
| `run_dissociation_multiinit.py` | multi-init-seed robustness check on the same dissociation | yes | `dissoc_multiinit.csv` |
| `run_convergence.py` | convergence metric sweep | yes | `convergence.csv` |
| `run_b6.py` | **D79 -- geometry conditioned on correctness WITHIN a fixed answer value, 0-GPU.** Stratified permutation with two gates: P1 (a gold value must carry both classes -- D72's lesson) and P1b (`combinatorial_floor`: the design must admit enough within-stratum arrangements to reach alpha at all, which the sampling-floor check structurally cannot see). *(Added 2026-08-09; doc otherwise FROZEN, inventory still enforced.)* | yes | `b6.csv` |
| `run_arm_geometry.py` | **D76 -- what training does to the SHAPE of the orbit, 0-GPU over the banked two-arm states.** Consecutive-step cosine (= cos(phi) for a linear map, so a rotation estimator with no window, plane or centre) and effective dimensionality, both at MATCHED window length and swept across windows, because participation ratio grows with sample count and the arms' natural windows differ 2.4x. *(Added 2026-08-09; doc otherwise FROZEN, inventory still enforced.)* | yes | `arm_geometry.csv` |
| `run_effective_dim.py` | **D74 -- how many directions the orbit actually moves in, 0-GPU over the banked trajectories.** Step DIRECTIONS (unit-normalised, so the geometric decay envelope is removed) over a data-chosen pre-floor window, participation ratio against an isotropic null at matched sample count. Reports the same quantity at `num_steps` 64 and 128 and across three window thresholds, because D28's failure mode was a statistic that reported the recording budget. *(Added 2026-08-09; doc otherwise FROZEN, inventory still enforced by `tests/test_architecture_consistency.py`.)* | yes | `effective_dim.csv` |
| `run_battery.py` | **the 21-family capability screen, graded by full-vocab rank, 0-GPU analysis of the `geometry-battery` kernel.** Reports capability at the BEST depth rather than a fixed one, because D68 showed accuracy is non-monotone in `r`. Its load-bearing output is `b6_usable`: whether any family has a gold value carrying BOTH a success and a failure, which is the ingredient D72 showed B6 is missing. *(Added 2026-08-09; doc otherwise FROZEN, inventory still enforced by `tests/test_architecture_consistency.py`.)* | yes | `battery.csv` |
| `run_h2_rotation.py` | **B4.15 — H2 on the Jacobian eigenvalue ARGUMENT, length-matched, 0-GPU analysis of the `geometry-h2-rotation` kernel's output.** Winding was retired as an instrument (D28: sign flips with the recording budget; D65: retiring a hypothesis with a broken ruler is circular); `arg(lambda)` reads rotation per unroll off the operator itself, with no window, no PCA plane and no surrogate. H2's statistic is the PAIRED `track - local` difference against `n_ops`, since the two share a byte-identical body. Exact paired sign-flip null: 12 pairs give 2^12 = 4096 arrangements, floor 2.4e-4. *(Added 2026-08-09; this doc is otherwise FROZEN, but its script inventory is still enforced by `tests/test_architecture_consistency.py`, so a new script must be entered here.)* | yes | `h2_rotation.csv` |
| `run_contrast.py`, `run_phase.py`, `run_forceloop.py` | contrast/phase/forced-loop-budget experiments | yes | `contrast.csv` (n/a locally), `phase.csv`, `forceloop.csv` |
| `run_homology.py` | persistent homology shape metric | yes | `homology.csv` |
| `run_three_scale.py` | V6 — three-scale length ablation (the decoupled task) | yes | `three_scale.csv` — **real, post-fix data as of 2026-07-23** (Kaggle T4, all 180 configs succeeded). Result leans against H2: `winding`/`steps_settle` track `irrelevant_len` far more strongly than `active_len` — see `claims_ledger.md` D11. |
| `run_three_scale_modk.py` | modulus-counting length-decoupled task, D11's prefix confound fixed by construction | yes | `three_scale_modk.csv` (N=7 default) — real data, all 126 configs succeeded, seq_len constant (42, std=0.0), clean null on winding~active_len. `three_scale_modk_extended.csv` (N=15, `--extended`) — **real data, 2026-07-24, Kaggle T4, all 270 configs succeeded**, seq_len constant (54, std=0.0), winding null replicates more decisively (Fisher p=0.266), steps_settle~active_len replicates and strengthens (Fisher p=0.0006). See `claims_ledger.md` D15. |
| `diag_blayney_repro.py` | Phase 1.2 positive control — reproduce Blayney's known loop-inducing condition | no (one-off diagnostic) | `blayney_repro.csv` — **real data, 2026-07-24, Kaggle T4.** First real loops observed outside forceloop.csv: 7/5445 (0.1286%) long_persona, matching Blayney's 0.14%. See `claims_ledger.md` D14. |
| `run_v6_correctness_probe.py` | V6 — per-unroll logit lens, correctness timing (strict argmax AND top-5) | yes | `v6_correctness_probe.csv` — real data, re-run 2026-07-24 with the top-k/single-token-answer fix. Top-5 hit rate on the honestly-verifiable single-token subset: **13/13 (100%)**, almost always by step 1; strict argmax only 4/13. Substantially revises the old "small-number prior, doesn't count" reading. See `claims_ledger.md` D12. |
| `run_smoke_new_tasks.py` | smoke test for count_ones/projection + normed acceleration | no (overwrites) | `smoke_new_tasks.csv` |
| `run_fdr_correction.py` | Phase 0 — BH-FDR across every project correlation test, 0-GPU | no (reads other CSVs directly) | `fdr_correction.csv` — 46 tests, 20/46 survive. See `claims_ledger.md` D13. |
| `run_power_analysis.py` | Phase 0 — loop-rate + Spearman-detection power analysis, 0-GPU | no (reads other CSVs + pure simulation) | `power_loop_rate.csv`, `power_curve.csv`. See `docs/power_and_preregistration.md`. |
| `run_winding_null.py` | matched-random-walk null test on the 140 banked raw trajectories — adjudicates the winding metric itself, 0-GPU | yes | `winding_null.csv` — **real result 2026-07-25: 82.1% of trajectories wind LESS than their step-size-matched null (median z=−5.61); only 9/140 beat it after FDR. See `claims_ledger.md` D22.** |
| `run_precision_check.py` | **plan Step 1, the GATE. NEEDS GPU, minutes.** Same prompt in bf16 vs fp32; predicts the post-convergence residual drops by 2⁻¹⁶ (0.91 → 1.4e-5) if the floor is arithmetic | yes | `precision_check.csv` (not yet produced). Branches explicitly: arithmetic / not arithmetic / partial. |
| `run_jacobian_spectrum.py` | **plan Step 2. NEEDS GPU.** Arnoldi (ARPACK) on autodiff Jacobian-vector products at h*; returns ρ (H3) and arg λ (H2 rotation) as modulus and argument of the same eigenvalue | yes | `jacobian_spectrum.csv` (not yet produced). Tests the log-polar prediction \|arg λ\| ≲ 0.78 rad. |
| `run_register_probe.py` | **plan Step 4. NEEDS GPU.** Barannikov Task a/b probed in TANGENT coordinates (ambient translation is forbidden by the norm constraint); Task b tests **quantisation** of position-indexed winding on balanced vs unbalanced strings | yes | `register_probe.csv` (not yet produced). Pre-registered reading in the docstring. |
| `run_register_analysis.py` | **Barannikov's register, offline** (CPU) from the banked per-position latents: increment direction v, register correlation with position+token-identity controls, and the ℤ-action matched-pair test | yes | `register_analysis.csv` — cosine +0.92/+0.94 (66–68σ), r=+0.22…+0.29 all 16/16 sign, matched pairs return to start (p≥0.66). See `claims_ledger.md` D34, D36. |
| `run_regime_rederivation.py` | **plan Step 5**: recomputes every invalidated metric legacy-vs-floor-aware on all 155 trajectories with raw paths, and reports which conclusions flip, 0-GPU | yes | `regime_rederivation.csv` — `cos` sign follows the RECORDING LENGTH (+0.258 at ns=16, −0.324/−0.368 at ns=64/128); 90.3% flip sign; `conv_rate` understated 3.2×, `dlr` inflated 4.04×. See `claims_ledger.md` D29. |
| `run_answer_probe.py` | is the ANSWER linearly decodable from the converged state, **within** a difficulty level (so `seq_len` is fixed)? Ridge + LOO with level means fitted inside each fold, within-level permutation null, ridge-penalty sweep, 0-GPU | yes | `answer_probe.csv` — **SUGGESTIVE, NOT ESTABLISHED**: count_ones R²=+0.219 p=0.020 (Bonferroni 0.040) but fragile to leave-one-level-out; projection null. See `claims_ledger.md` D27. |
| `run_winding_permutation.py` | H2 with **no surrogate model at all**: permutes `n_ops` labels among real trajectories, stratified by (task, num_steps), 20k permutations, BH+Bonferroni corrected, 0-GPU | yes | `winding_permutation.csv` — **no stratum survives correction; H2 is NULL under the assumption-free test.** See `claims_ledger.md` D26. |
| `run_homology_null.py` | the null persistent homology never had: H1 on 15 saved trajectories against 40 manifold-matched surrogates each (sphere, radial profile and angular step sizes preserved; only rotational freedom randomised). 0-GPU | yes | `homology_null.csv`. Basis of D64 — the first topological positive in the project, and a heavily qualified one. |
| `recheck_register_window.py` | recomputes the register statistics on the CORRECT token window (derived from `meta.json`'s `token_ids`, not hardcoded), redoes the contrast control in the residual stream where `v` actually lives, and runs the constructive test — is the lagged count decodable at all after in-fold controls, grouped by prompt. 0-GPU | yes | `register_window_recheck.csv`. **Supersedes D34/D36's `register_r`** — see D50. |
| `plot_regimes.py` | four figures from saved states: per-instance decodability vs depth with the integer decision boundary, the answer-state manifold trained vs untrained, state motion while reading, log-distance-from-endpoint with the arithmetic floor. 0-GPU | yes | `figures/readout_regimes.png`, `answer_manifold_pca.png`, `register_trajectory_pca.png`, `log_distance_from_end.png`. Basis of D48/D49. |
| `run_manifold_null.py` | winding vs an ON-MANIFOLD null (sphere + radial profile + angular steps preserved), 140 traj × 100 surrogates, with a built-in calibration arm and stratification by `num_steps`/task, 0-GPU | yes | `manifold_null.csv`. Supersedes the underpowered subsample numbers in an earlier draft of `rigor_audit.md` §17 — **the sign of the winding effect is unresolved until this runs**. |
| `run_depth_threshold.py` | **NEEDS GPU, not yet run.** Teacher-forced log P(gold) − log P(distractor) at every unroll r; locates r\* per instance and tests whether required depth scales with difficulty — H2 restated behaviourally, with no geometry | yes | `depth_threshold.csv` (not yet produced). Excludes answer==0 instances: in `v6_correctness_probe.csv` correctness is perfectly separated by target==0 (4/5 vs **0/8**), all at unroll 1 — a zero-prior, not computation. |
| `run_winding_variants_null.py` | puts 9 alternative rotation estimators through the *identical* matched-random-walk null, so the metric can be chosen by evidence rather than inheritance, 0-GPU | yes | `winding_variants_null.csv` — **2026-07-25: the deployed metric (W1) beats its null on 10% of trajectories vs 97–100% for the direction-agnostic variants. Read with `docs/rigor_audit.md` §8: the surrogate leaves the state manifold, so all of these z-scores partly measure on- vs off-manifold.** |
| `backfill_provenance.py` | one-off: retro-tag every banked `.npy` with a provenance sidecar + `manifest.jsonl`, splitting VERIFIED from RECOVERED fields | n/a | writes `*.prov.json` + `manifest.jsonl` beside each artifact. See `docs/rigor_audit.md` §§4–5. |
| `backfill_seq_len.py` | one-off: backfill `seq_len` onto switch/maxtask CSVs without GPU | n/a | mutates `switch.csv`/`maxtask.csv` in place |
| `plot_trajectories.py` | PCA plot of count_ones/projection trajectories, shared basis per (task, n_ops) | n/a | `figures/pca_*.png` |
| `extract.py`, `run_mvp.py` | earlier/MVP-era extraction entry points | — | — |

## metrics/ — second-generation, regime- and manifold-aware

Added 2026-07-25 by the rigor audit; see `docs/rigor_audit.md`. These do NOT
replace the originals in place — the originals stay bit-compatible so cached
CSVs remain reproducible, and their biases are documented in their docstrings.

- `metrics/regime.py` — separates the ~15-step signal regime from the
  arithmetic-noise floor. Floor-aware `contraction_from_pair`,
  `step_cosine_converging`, `classify_shape_regime`, `h1_persistence_signal`.
- `metrics/surrogate.py` — on-manifold null. Preserves the sphere, the radial
  convergence profile and every consecutive angular step; randomises only
  rotational freedom. Replaces `winding.matched_random_walk`, which leaves the
  manifold (‖h‖ 76.37 → 154.76).
- `provenance.py` + `scripts/backfill_provenance.py` — sha256 sidecars and
  append-only `manifest.jsonl`; `task_seed` and `init_seed` are separate
  required fields.
- `eval_depth.py` + `scripts/run_depth_threshold.py` — H2 restated
  behaviourally as "at what depth does the answer become preferred". NEEDS GPU,
  not yet run.

## h3_toy_model/ — the A6 toy-model line

Standalone package (own scripts, own results under `h3_results/`) carrying the
**A6** H3 finding cited in `claims_ledger.md`: a from-scratch toy model showing
the contraction mechanism does not appear on a recurrent-over-**depth**
architecture matching Huginn's information flow, but does on a
recurrent-over-**time** counterpart. Files: `h3_validation.py`,
`sequential_h3_validation.py`, `sequential_rnn.py`, `spectral.py`,
`synthetic_tasks.py`, `train_tiny_recursive.py`, `instrument_huginn.py`,
`test_spectral.py`; outputs `h3_results/h3_sweep.csv`,
`h3_results/sequential_sweep.csv`, `h3_results/FINDINGS.md`.

**Maintenance gap, found 2026-07-25 (`rigor_audit.md` §14).** This directory
was in neither `pyproject.toml`'s `testpaths` nor ruff's `src`, so its 5 tests
were never collected and its code was never linted — while a claim resting on
it sat in the ledger. `testpaths` now includes it (all 5 pass). Ruff still
reports ~60 style errors here (vs 0 in `src/`), so it remains **linted but not
clean**; that is a known, scoped debt, not a silent one.

## GPU kernels — the derivation of every result marked "verified-live on GPU"

Run on Kaggle's free T4 tier. Each bundle is a **self-contained** `main.py` plus
`kernel-metadata.json`; they deliberately do NOT clone the repo, so no push to
anyone's remote was needed to run them. Logs are pulled back into `out/`, which
is what the offline analyses read.

| bundle | kernel id | log pulled |
|---|---|---|
| `kaggle_blayney_modk` | `arsen4ikvar/geometry-blayney-repro-modk-sweep` | no |
| `kaggle_depth` | `arsen4ikvar/geometry-depth-threshold` | yes |
| `kaggle_depth_fixed` | `arsen4ikvar/geometry-depth-fixed-length` | yes |
| `kaggle_jacobian` | `arsen4ikvar/geometry-jacobian-spectrum` | yes |
| `kaggle_modk_extended` | `arsen4ikvar/geometry-modk-extended-d15-followup` | no |
| `kaggle_precision_gate` | `arsen4ikvar/geometry-precision-gate` | yes |
| `kaggle_register` | `arsen4ikvar/geometry-register-probe` | yes |
| `kaggle_register_depth` | `arsen4ikvar/geometry-register-vs-depth` | yes |
| `kaggle_states` | `arsen4ikvar/geometry-position-states` | yes |
| `kaggle_v6_probe` | `arsen4ikvar/geometry-v6-experiments-6` | no |
| `kaggle_v6_rerun` | `arsen4ikvar/geometry-v6-rerun-post-fix` | no |
| `kaggle_v6_topk_fix` | `arsen4ikvar/geometry-v6-correctness-probe-top-k-fix` | no |

`register_depth` → **D37** (the register/depth dissociation);
Mapping to results: `precision_gate` → D30 (plan Step 1, the gate);
`jacobian` → D31 (Step 2, ρ exact); `register` → D32 (Step 4);
`depth` → D33 and `depth_fixed` → **D35** (Step 3, the confound-free version);
`states` → D34/D36 via `scripts/run_register_analysis.py`.

Note `scratch/` is otherwise throwaway by project convention. These six are
not: they are the only record of how the GPU numbers were produced, which is
why they are indexed here rather than left unattributed.

## results/ inventory — what's actually committed vs not

Present in `results/`: `blayney_repro.csv`, `convergence.csv`,
`counting_accuracy.csv`, `counting.csv`, `dissoc_multiinit.csv`,
`dissociation_15seed.csv`, `dissociation_results.csv`, `dissociation.csv`,
`fdr_correction.csv`, `forceloop.csv`, `full_synthetic_experiments.csv`,
`h2_loops.csv`, `homology.csv`, `maxtask.csv`, `pararule.csv`, `phase.csv`,
`power_curve.csv`, `power_loop_rate.csv`, `smoke_new_tasks.csv`,
`switch.csv`, `three_scale_modk.csv`, `three_scale_modk_extended.csv`,
`winding_null.csv`, `winding_variants_null.csv`, `manifold_null.csv`, `winding_permutation.csv`, `answer_probe.csv`, `regime_rederivation.csv`, `register_analysis.csv`, `register_window_recheck.csv`, `homology_null.csv`. Not yet produced (GPU-gated): `precision_check.csv`, `jacobian_spectrum.csv`, `register_probe.csv`, `depth_threshold.csv`.

`three_scale.csv` and `v6_correctness_probe.csv` both promoted into
`results/` proper 2026-07-23 — real, post-fix data, both confirmed via
Kaggle T4. Old copies under `scratch/kaggle_output_*/` predate the
coda-skip fix and stay garbage; don't cite from `scratch/` anymore, use
`results/` directly.

**Orphaned data, no current producer** (found 2026-07-23 while building
the automated consistency check below): `dissociation_results.csv` and
`h2_loops.csv`. Both committed in the earliest commits (`17a944a`/`8b5e418`,
"exp 2"/"exp 3"), predating `scripts/` entirely — almost certainly
notebook-era output (`notebooks/01_mvp_h2.ipynb`), with no `run_*.py`
anywhere in the current repo that regenerates either. Don't cite numbers
from these without first checking whether `dissociation.csv`/
`dissociation_15seed.csv` (the current, actively-produced equivalents)
supersede them — they cover overlapping ground. Not deleted here since
removing data isn't this doc's call to make unilaterally; flagged so
nobody cites them as if they were current.

## Decisions log (most recent first)

- **2026-07-25 — Ran the winding null test; the "it's blocked" premise was
  false, and the metric fails.** Docs had said for weeks that
  `winding_null_test` cannot run because raw per-step paths aren't saved.
  Wrong: **140 real Huginn trajectories with full raw `[T,5280]` paths sit in
  `results/trajectories/`** (untracked, no producer script, no manifest —
  which is why they were overlooked). Ran the test at 0 GPU cost:
  **82.1% of real trajectories wind LESS than a matched random walk**
  (median z=−5.61; 66.4% at p=1.000); only 9/140 beat the null after FDR.
  Mechanism verified: paths settle at step ≈14, then only jitter — a random
  walk turns that jitter into accumulated angle, the real path doesn't
  rotate. So the z-gap widens with budget (−1.62 at ns=64 → −15.99 at
  ns=128) while observed winding stays flat. Root cause of the ubiquitous
  ≈0.62 value: `winding_number` uses the trajectory's own centroid as
  rotation center, and a settling path collapses 18× toward it, sweeping
  ~0.6 turns by construction. Consequence: winding is near-constant on
  answer tokens (modk-N15 std **0.005**), so **H2 as operationalized was
  largely untestable there, not tested-and-refuted** — a sharpening of
  D15/D16. Caveat recorded: the banked trajectories' provenance is
  undocumented (and the `track_*` files `convergence.csv` used are gone), so
  this is a verdict on the *metric*, not on a specific condition. Full detail
  + all caveats: `claims_ledger.md` D22.

- **2026-07-24 — D15 replicated at N=15 (vs N=7): winding null holds up
  stronger, steps_settle effect strengthens.** `run_three_scale_modk.py
  --extended`, 270 real Huginn-3.5B configs, Kaggle T4, seq_len confirmed
  exactly constant (54, std=0.0) at the larger scale too. winding~active_len:
  Fisher-combined p=0.266 (was 0.something at N=7 too, but now with real
  power behind the null -- ~44-57% power to catch rho=0.5 at N=15, so this
  is a genuine absence, not just "still underpowered"). steps_settle~active_len:
  Fisher-combined p=0.0006 (was 0.015 at N=7) -- both moduli individually
  significant now (p=0.034, p=0.0016), essentially unchanged effect size at
  mod=5 (-0.750 -> -0.739), somewhat weaker at mod=2 (-0.775 -> -0.549) but
  still significant alone. Real replication, not just a bigger N repeating
  the same small-sample luck. See `claims_ledger.md` D15.
- **2026-07-24 — D12 rerun with top-k fix substantially revises the
  "small-number prior" story.** Two real bugs on the way: (1) forgot
  `logits` from `extract_trajectory` is numpy not torch, `.topk()`
  crashed 24/24 configs on the first Kaggle attempt -- fixed with
  `np.argpartition`, verified locally against `np.argsort` before
  relaunching. (2) The fix itself: single-token vs multi-token answers
  now distinguished (11/24 of D12's rows have negative targets, which
  ALSO split into 2 tokens like magnitude>=10 does -- a leading-"-"-token
  match confirms nothing about the actual value). Real result on the
  13 honestly-verifiable single-token rows: top-5 hit rate 13/13 (100%),
  usually by unroll step 1; strict argmax only 4/13. The model reliably
  shortlists the right count, a small-number bias usually wins the final
  pick. Old "doesn't count, just a prior" reading was too strong. See
  `claims_ledger.md` D12.
- **2026-07-24 — Added STATUS lines to 9 scripts that never had them**
  (`run_convergence/counting/dissociation/dissociation_multiinit/
  forceloop/homology/maxtask/phase/switch.py`) -- all notebook-derived,
  pre-session, had OWNER but no STATUS, unlike every script touched this
  session. Real, checkable inconsistency, not cosmetic: STATUS is where
  a reader learns a script's result is degenerate (D10), doesn't
  replicate (D13), or measures something narrower than its docstring
  implies (H1 on single curves, D-series pointers) without cross-referencing
  the ledger first. Content only, no logic changes; each STATUS line cites
  the specific claims_ledger.md ID it summarizes.
- **2026-07-24 — Real GPU runs, both Kaggle and DataSphere. First-ever
  loops observed outside forceloop.csv.** Kaggle T4: `diag_blayney_repro.py`
  (24 GSM8K examples x 2 conditions) and `run_three_scale_modk.py` (126
  configs) both completed. Blayney repro: 7/5445 (0.1286%) long_persona
  loops, matching Blayney's own 0.14% closely -- first real loops this
  project has ever seen outside the starved-budget sweep, some spanning
  multiple full turns (up to |winding|=7.18, vs forceloop's ~=0.65).
  Confirms the pipeline works and the phenomenon is real, just rare.
  modk sweep: seq_len confirmed exactly constant (42, std=0.0) across all
  126 rows -- clean null on winding~active_len, the project's first
  genuinely unconfounded H2 test. See `claims_ledger.md` D14/D15.
  DataSphere GPU (parallel, per explicit instruction to monitor closely
  and go incrementally): first-ever GPU job on this account. Two real
  infra bugs found and fixed in sequence -- (1) auto-discovered torch
  (2.13.0) bundles CUDA 13, incompatible with the gt4.1 node's driver
  (caps at CUDA 12.2); (2) pinning torch==2.4.1 to fix that broke
  Huginn's own model code, which imports `torch.nn.attention.
  flex_attention` (added in torch 2.5) -- needed >=2.5 AND still
  CUDA-12-compatible. Also hit and fixed a real `datasphere` CLI bug:
  `local-paths: []` (empty list) silently collapses to `None` via a
  truthy-check inconsistency, crashing job submission -- fix is a
  non-empty placeholder list. Full details in
  `code_env_info/yandex-cloud-smiles-access.md`'s new "GPU jobs" section.
  Balance check: `unitBalance` still displays `5000000` after ~15 minutes
  of combined real GPU time across 3 jobs -- confirmed this is a
  display/billing lag, not zero cost; don't trust it as a live tracker.
- **2026-07-24 — Two corrections + one new task generator, from being
  challenged on sloppy explanations.** (1) H3/task-reinjection: re-derived
  properly from `contraction_proof.md`'s own `h_{t+1}=R_theta(h_t;e)` --
  `e` held fixed throughout the proof, only `h_0`-differences shown to
  decay. This project's counting tasks make the count fully readable from
  `e` (full context, reinjected every unroll), so H3's mechanism doesn't
  bind on the count by construction here -- matches the A6 toy result
  exactly. Consequence: this project's own counting failure should not be
  narrated as confirming H3. (2) J-lens compute: was wrong calling it "one
  pass" -- verified against the actual method, needs a ~1000-prompt
  calibration corpus per layer, genuinely GPU-heavy. (3)
  `make_three_scale_modk_task` added (`synthetic.py`, 6 tests): fixes
  D11's prefix confound properly (constant total_len by construction, no
  prefix region) and generalises `make_switch_task`'s mod-2 to modulus
  2..10, single-token answer guaranteed. Also checked DataSphere credits
  live: original project balance unchanged display-wise despite 2 real
  jobs; found a second, empty-balance project in a newly-visible community.
- **2026-07-24 — Self-caught error: power analysis used wrong Blayney rate,
  fixed same day.** Downloaded Blayney et al.'s real PDF (WebFetch truncates
  the HTML), read Appendix C Tables 3-4 directly. The "2.81%" figure this
  project has cited since 2026-07-17 is a per-EXAMPLE rate ("any question
  token shows the behavior"), not per-token — `power_and_preregistration.md`'s
  first version (written earlier the same day) used it as a per-token
  Poisson rate anyway. Correct per-token ceiling is **0.14%** (Table 3,
  Long Persona). N-needed-for-5-loops corrected from 178 to ~3,571; the
  "current pool would show ~36 loops but shows 0" anomaly (which needed an
  awkward "rate doesn't transfer" caveat) is gone — corrected E[loops]=1.8,
  observing 0 is unremarkable (~16% chance under Poisson). Fixed
  `run_power_analysis.py`, `power_and_preregistration.md`, `claims_ledger.md`
  B9 (now has the full verified table), `project_plan.md` §7/§13. Also
  resolved Blayney's Orbit-specific rate (0.01%-0.13%, closer to this
  project's own metric) and their FFT-based classifier (Algorithm 1) as a
  second reference implementation worth comparing against `classify_shape`.
  Lesson: a WebFetch HTML summary of a 39-page paper with 63 figures is not
  the same as reading the actual tables — worth the PDF download when a
  number will be used in real arithmetic, not just cited.
- **2026-07-24 — Digit tokenization uncertainty resolved, live check not
  assumption.** `scripts/diag_tokenization.py`, tokenizer-only (CPU, no
  GPU/weights). Both spaced and unspaced digit strings give exactly one
  token per digit on this tokenizer (`make_count_ones_task`'s "n_ops
  tokens, always" claim confirmed) — the earlier "should we add spaces"
  question turns out not to change token count here. Second check confirms
  the already-flagged multi-digit-answer problem directly: answers 0-9 are
  one token, answers >=10 split in two with the first token being only the
  leading digit — any first-token-argmax accuracy check silently breaks
  above n_ops=9 on count_ones/counting/three_scale. `project_plan.md` §13
  updated (RESOLVED).
- **2026-07-24 — Phase 0 complete: power analysis + pre-registration
  written, the last §15 item.** `scripts/run_power_analysis.py`,
  `docs/power_and_preregistration.md`. Two power problems, both real:
  (1) loop-rate — Blayney baseline 0.02% needs 25,000 (token,trajectory)
  draws for E[loops]>=5; current real pool is 1,294 (E=0.26, correctly
  predicts the zero loops actually observed outside forceloop.csv,
  1,198/1,198 settle); keystone extraction needs >=500 prompts averaging
  >=100 tokens to clear the bar. (2) detection power — Monte Carlo (5000
  trials/cell) shows N=6 (this project's modal level count) gives only 30%
  power to detect even rho=0.7; need N>=20 for 80% power at that effect
  size. Both independently mean a null H2 result on current task designs
  is "untestable," not "refuted." Froze a 10-item pre-registered test list
  for Phase 1/2 (H1 items 1-3, H2 items 4-6, H3 items 7-8, validity checks
  9-10) so nothing gets added post-hoc after seeing keystone data. Fast
  tests pin the exact (non-simulated) arithmetic; the full Monte Carlo grid
  (~60s) stays out of the pytest suite, run via the script directly.
  **All 5 Phase 0 items from project_plan.md §15 now done.**
- **2026-07-24 — BH-FDR sweep done; plan's own prediction was wrong.**
  `scripts/run_fdr_correction.py`, 46 tests across 9 experiments, 0-GPU.
  Raw p<0.05: 22/46. Survives BH-FDR: 20/46 (project-wide and
  per-experiment families agree here). Plan expected the two "prime
  target" hits (maxtask winding~n_ops +0.943, dissociation local −0.943)
  to be FDR casualties — **wrong, both survive** (q=0.0123). Real problem
  elsewhere: maxtask's n_ops IS seq_len (D10, rank-corr=1.0, guard already
  raises on it); dissociation's local hit fails to replicate at 3x seeds
  (15-seed: rho=-0.543, p=0.27, genuinely dies there). Only 2 true FDR
  casualties, both marginal (p=0.0499). Lesson: don't trust a plan's own
  prediction of an outcome over the actual run, even your own plan's.
  `claims_ledger.md` D13, `project_plan.md` §0.2 corrected. Test pins the
  full result against cached CSVs.
- **2026-07-24 — Phase 0 rescues from `project_plan.md` §15, started.** Four
  checkpoints so far, each own commit on `feat/close-known-gaps`. (1)
  `multivariate_rank_control()` added to `correlate.py` — real code now,
  was inline ad hoc script. Reproduces D11's exact numbers against cached
  `three_scale.csv`. Wired into `run_three_scale.py`'s `main()`; `compute()`
  untouched. (2) `run_forceloop.py` now runs the actual Fisher exact test
  for B4 (was hand-computed, typed into ledger, no reproducing script).
  Matches ledger exactly: p=0.0101, 82/13/1 settle/loop/drift. (3)
  `partial_spearman`'s collinearity guard tightened 0.999 -> 0.95 — old
  threshold missed `dissociation.csv`'s real rho=0.9895 n_ops-vs-seq_len
  collinearity (a landmine, no live caller yet but nothing stopped one
  starting). New threshold still clears `pararule.csv`'s real rho=0.816.
  Tests added for all three. 31/31 green, ruff clean throughout. Next:
  BH-FDR sweep (§0.2), power analysis + pre-registration (§0.1).
- **2026-07-23 (round 4) — round 3's "-1 everywhere" was a probe bug, not a
  real finding.** Built `diag_v6_token_gap.py`, dumped real top-5 tokens.
  Target token used bare answer string ("2"). Model's real continuation:
  space-prefixed (" 2"). Bare token rank 27, prob 0.0023. Space-prefixed
  token: in the top-5, prob ~0.065. Wrong token checked, every prior run.
  Fixed: tokenize `" " + ans`. Old `v6_correctness_probe.csv` deleted, was
  invalid. Not yet rerun with the fix. `claims_ledger.md` D12 updated.
  Lesson: a suspicious all-negative result still needs a look before
  trusting it, same as a suspicious all-positive one.
- **2026-07-23 (round 3) — coda-skip fix confirmed correct on real hardware.
  `validate_logits` passed all 24/24 prompts, zero RuntimeErrors.** Round 2's
  fix (missing pre-coda `ln_f`) was right. `_replicate_coda_head` now trusted,
  not just reasoned-about. Real cost: correctness-timing data itself came
  back a clean null — `correct_at_step=-1` for every depth/seed tested.
  Cross-checked against `counting_accuracy.csv`: consistent with it, not
  contradicting, but N=3 seeds here is small — doesn't rule out a real but
  low hit rate. Full detail: `claims_ledger.md` D12. `results/v6_correctness_probe.csv`
  now real and committed, first time ever.
- **2026-07-23 (round 2) — the coda-skip fix's self-check caught that the
  fix was STILL wrong; found and fixed the real bug via direct source
  re-reading, not guessing.** Relaunched Kaggle for `run_v6_correctness_probe.py`
  after round 1's bugs were fixed. Result: `validate_logits=True` correctly
  rejected the reconstruction on **all 24/24** real prompts, consistent
  ~1.7-1.9 max abs logit diff (not noise — a systematic bug). This is
  exactly what the self-check exists to catch, and it worked as designed —
  the earlier "fix" was conceptually right (coda layers needed) but
  incomplete. Re-fetched `raven_modeling_minimal.py` directly and confirmed
  verbatim: `iterate_forward()` itself returns `self.transformer.ln_f(x)`
  — the recurrent loop's own output is already normalized by the time
  `forward()` receives it and feeds it to coda. `_replicate_coda_head` was
  feeding the raw, pre-ln_f hooked state straight into coda, skipping this
  first normalization entirely. Fixed: `ln_f -> coda -> ln_f -> lm_head`,
  not `coda -> ln_f -> lm_head`. Full detail in `hook.py`'s own GOTCHAS.
  A second, unrelated bug surfaced in the same run: `main()` crashed with
  `KeyError: 'depth'` trying to `groupby` an empty DataFrame after every
  config failed validation — fixed with an explicit empty-result check.
  Both fixes pushed, a third Kaggle run launched to confirm `validate_logits`
  actually passes now — genuinely unverified until that returns; this round's
  `v6_correctness_probe.csv` is empty/invalid, same as before.
- **2026-07-23 (round 1) — first real post-fix Kaggle run: `three_scale.csv` real
  data landed, `v6_correctness_probe.csv` still didn't run.** Launched a
  T4 Kaggle kernel (avoids the old P100 sm_60 CUDA issue entirely, no
  torch downgrade needed) against the now-fully-fixed
  `feat/close-known-gaps` branch. `run_three_scale.py`'s extraction
  succeeded completely (180/180 configs, ~17 min, `save_partial`
  checkpointing worked as designed) — but `main()`'s reporting step then
  crashed: `spearman()` returns `(rho, p)`, and the print code tried to
  format the whole tuple with `:.3f`, `TypeError: unsupported format
  string passed to tuple.__format__`. A real bug that static review
  (ruff, reading the code) never caught, because it's a runtime type
  mismatch, not a syntax issue — only surfaced by actually running it.
  The orchestrator script (`scratch/kaggle_v6_rerun/run_kaggle.py`,
  untracked) then propagated that failure and never attempted
  `run_v6_correctness_probe.py` at all — a second real bug, this one in
  the run orchestration itself (one experiment's failure shouldn't block
  an unrelated one). Fixed both: unpacked the tuple correctly in
  `run_three_scale.py`, and the orchestrator now isolates each experiment.
  The recovered real data itself is a genuine, citable negative-leaning
  result — see `claims_ledger.md` D11. `v6_correctness_probe.csv`,
  and with it the first real-hardware confirmation of the coda-skip fix
  itself, is still outstanding.
- **2026-07-22 — fixed the coda-skip bug in `hook.py`'s `return_logits=True`
  path.** Root cause: the V6 logit-lens hook ran `ln_f`/`lm_head` straight on
  `core_block[-1]`'s output, skipping the two `coda` layers that
  `RavenForCausalLM.forward()` actually runs first (verified against
  `raven_modeling_minimal.py`). Fix: `_replicate_coda_head()` runs the real
  coda→ln_f→lm_head tail; `validate_logits=True` (default) cross-checks the
  final step against a genuine `forward()` call and raises `RuntimeError` on
  mismatch — this has not been confirmed on real hardware yet (no local
  GPU), so the self-check exists specifically to catch a wrong
  reconstruction the first time this runs for real. Any `v6_correctness_probe.csv`
  from before this fix is invalid.
- **2026-07-22 — fixed sign-truncation bug in `run_v6_correctness_probe.py`**:
  compared against the answer's *last* token instead of its first, which
  silently ignored the sign on negative answers. Now uses `tok.encode(ans,
  ...)[0]`.
- **2026-07-22 — fixed non-deterministic file pick in `plot_trajectories.py`**:
  `glob.glob(pattern)[0]` could silently pick between e.g. `ns64`/`ns128`
  trajectory files for the same (task, n_ops, seed). Now raises `ValueError`
  on an ambiguous match instead of guessing; `num_steps=` param added to
  disambiguate explicitly.
- **2026-07-22 — documented `make_count_ones_task`/`make_projection_task`'s
  length confound** (same as every original synthetic task) directly in
  their docstrings, rather than redesigning them — `make_three_scale_task`
  is the task that actually solves this, these two don't need to.
- **2026-07-22 — `constants.py`**: documented that `MODEL_REVISION` is
  silently meaningless when `HUGINN_MODEL_ID` is overridden to a local path
  (e.g. a Kaggle dataset mount) — `from_pretrained(revision=...)` ignores
  `revision` for local paths, no error.
- **2026-07-19 — `steps_to_settle` split into two named metrics**: the
  original displacement-based `steps_to_settle` (every existing cached CSV
  uses this) was briefly overwritten with an acceleration-based
  redefinition; restored the original and kept the new one under its own
  name, `steps_to_settle_by_acceleration`, in `metrics/dynamics.py`.
- **2026-07-19 — `make_three_scale_task` added**: the only synthetic task
  generator that independently varies irrelevant/neutral/active-length,
  closing the project-wide length-confound gap (`claims_ledger.md` D10) for
  future experiments (not retroactively — existing E2/switch/maxtask data is
  still confounded, documented not redesigned).

## Branch map (as of 2026-07-22)

Local: `main`, `feat/close-known-gaps` (current), `fix/repro-and-scaffolding`,
`local-fixes`, `backup-all-3-fixes-2026-07-18`.
Remote `origin`: `main`, `pr-real-data`, `two-scale-real-fix`,
`feature/two-scale-latent-dynamics-paper`.
Remote `Arsenii324`: `feat/close-known-gaps` — the fork/branch the
unauthorized antigravity-cli work was pushed to; predates this session's
fixes. Do not treat code pulled from there as current.

## Open items

- ~~`three_scale.csv`/`v6_correctness_probe.csv` need a real GPU re-run~~ —
  done 2026-07-23, both real, both in `results/`, see decisions log + D11/D12.
- ~~DataSphere per-hour unit cost unverified~~ / ~~never actually run a job
  there~~ — both done 2026-07-23. `gt4.1` (T4, matches what already worked
  on Kaggle) = 129,600 units/hr = $1.38/hr, ~38.6hr runway on the
  5,000,000-unit budget. A real `c1.4` (CPU, cheapest tier) smoke-test job
  ran end to end — `status: SUCCESS`, real stdout retrieved, ~232 units
  spent. Working `config.yaml` pattern (needed real fixes beyond the docs'
  own example: `env.python.type: manual` + explicit `version: "3.11"`
  since `auto` fails when the local submitting interpreter is newer than
  DataSphere's supported 3.8-3.12; entry script needs a real
  `if __name__ == "__main__":` guard) recorded in
  `code_env_info/yandex-cloud-smiles-access.md`. GPU tier (`gt4.1`) itself
  still not exercised — only the CPU smoke test so far, deliberately, to
  prove the mechanism cheaply before spending on GPU time.
- `winding_null_test` (built this session) has never been run against real
  trajectory data — needs a fresh extraction saving raw `.npy` states, not
  just summary-stat CSVs like `three_scale.csv` has.
- `local_envs_venvs.md` in `code_env_info/` is an empty stub.
