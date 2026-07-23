# Architecture & experiment state — living doc

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
  the team's shared grant. Confirmed live via API: project
  `Online_project24_2` (id `bt14qn4u9t3n09nfjoqu`), unit balance **5,000,000,
  untouched**. Deployment mechanism is DataSphere Jobs
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
| `make_three_scale_task` | **independent** (irrelevant_len / neutral_len / active_len vary separately) | added 2026-07-19; the *only* task in the repo that actually decouples length from difficulty |

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
| `run_contrast.py`, `run_phase.py`, `run_forceloop.py` | contrast/phase/forced-loop-budget experiments | yes | `contrast.csv` (n/a locally), `phase.csv`, `forceloop.csv` |
| `run_homology.py` | persistent homology shape metric | yes | `homology.csv` |
| `run_three_scale.py` | V6 — three-scale length ablation (the decoupled task) | yes | `three_scale.csv` — **real, post-fix data as of 2026-07-23** (Kaggle T4, all 180 configs succeeded). Result leans against H2: `winding`/`steps_settle` track `irrelevant_len` far more strongly than `active_len` — see `claims_ledger.md` D11. |
| `run_v6_correctness_probe.py` | V6 — per-unroll logit lens, first-token correctness timing | yes | `v6_correctness_probe.csv` — real, post-fix data as of 2026-07-23 round 3. `validate_logits` passed 24/24 on real hardware. Data itself is a clean null (`correct_at_step=-1` everywhere) — see `claims_ledger.md` D12. |
| `run_smoke_new_tasks.py` | smoke test for count_ones/projection + normed acceleration | no (overwrites) | `smoke_new_tasks.csv` |
| `backfill_seq_len.py` | one-off: backfill `seq_len` onto switch/maxtask CSVs without GPU | n/a | mutates `switch.csv`/`maxtask.csv` in place |
| `plot_trajectories.py` | PCA plot of count_ones/projection trajectories, shared basis per (task, n_ops) | n/a | `figures/pca_*.png` |
| `extract.py`, `run_mvp.py` | earlier/MVP-era extraction entry points | — | — |

## results/ inventory — what's actually committed vs not

Present in `results/`: `convergence.csv`, `counting_accuracy.csv`,
`counting.csv`, `dissoc_multiinit.csv`, `dissociation_15seed.csv`,
`dissociation_results.csv`, `dissociation.csv`, `forceloop.csv`,
`full_synthetic_experiments.csv`, `h2_loops.csv`, `homology.csv`,
`maxtask.csv`, `pararule.csv`, `phase.csv`, `smoke_new_tasks.csv`,
`switch.csv`.

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
- ~~DataSphere per-hour unit cost unverified~~ — done 2026-07-23: `gt4.1`
  (T4, matches what already worked on Kaggle) = 129,600 units/hr = $1.38/hr,
  ~38.6 hours of runway on the 5,000,000-unit budget. Full config/pricing
  table in `code_env_info/yandex-cloud-smiles-access.md`. Not yet actually
  run a real job there — CLI verified working (`datasphere` via pipx,
  `GRPC_DNS_RESOLVER=native` needed to work around a local sandbox DNS
  issue), zero jobs on the project so far.
- `winding_null_test` (built this session) has never been run against real
  trajectory data — needs a fresh extraction saving raw `.npy` states, not
  just summary-stat CSVs like `three_scale.csv` has.
- `local_envs_venvs.md` in `code_env_info/` is an empty stub.
