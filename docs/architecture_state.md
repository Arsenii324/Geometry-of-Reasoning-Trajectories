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
| `run_dissociation*.py` (3 variants) | winding vs steps-to-settle dissociation | yes | `dissociation*.csv` |
| `run_convergence.py` | convergence metric sweep | yes | `convergence.csv` |
| `run_contrast.py`, `run_phase.py`, `run_forceloop.py` | contrast/phase/forced-loop-budget experiments | yes | `contrast.csv` (n/a locally), `phase.csv`, `forceloop.csv` |
| `run_homology.py` | persistent homology shape metric | yes | `homology.csv` |
| `run_three_scale.py` | V6 — three-scale length ablation (the decoupled task) | yes | `three_scale.csv` — **not currently in `results/`, only under `scratch/kaggle_output_{5,6}/repo/results/`** |
| `run_v6_correctness_probe.py` | V6 — per-unroll logit lens, first-token correctness timing | yes | `v6_correctness_probe.csv` — **existing copies (`scratch/kaggle_output_{5,6,7,8}/`) all predate the coda-skip fix in `hook.py` and are garbage; needs a fresh GPU run before the data means anything** |
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

**Not present** (only exist as pulled Kaggle kernel outputs under
`scratch/kaggle_output_*/repo/results/`, never copied/committed into
`results/` proper): `three_scale.csv`, `v6_correctness_probe.csv`. Treat
anything quoted from those two as coming from `scratch/`, not `results/`,
until someone deliberately promotes a post-fix run into `results/`.

## Decisions log (most recent first)

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

- `three_scale.csv`/`v6_correctness_probe.csv` need a real GPU re-run with
  the fixed `hook.py` before either is trustworthy; neither is in `results/`.
- DataSphere per-hour/per-job unit cost for this project's GPU config is
  still unverified — check the pricing page before running a real job there.
- `local_envs_venvs.md` in `code_env_info/` is an empty stub.
