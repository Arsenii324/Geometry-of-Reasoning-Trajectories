# Index — every document, tool, data location and external path

*Written 2026-08-11 19:30 against context compaction. **Read this to find things; read
`WORKING_KNOWLEDGE.md` to reason about them.** Paths relative to the repo root
`~/build-projs/barannikov-work/Geometry-of-Reasoning-Trajectories/` unless marked.*

---

## 0. If you arrived from GitHub — read this line first

**None of this work is on `main`.** `origin`'s default HEAD branch is `main`, and
`docs/submission/` does not exist there. Everything below lives on
**`claude/geometry-reasoning-recap-rhe0bp`**:

```sh
git clone https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git
cd Geometry-of-Reasoning-Trajectories && git checkout claude/geometry-reasoning-recap-rhe0bp
```

`upstream` is `Shtirmann/Geometry-of-Reasoning-Trajectories` and does **not** carry this branch at
all. A clone that skips the checkout sees an old tree and no submission material.

**What a fresh clone can and cannot do.** It *can*: read every document below, write from
`docs/submission/CLAIM_INVENTORY.md`, cross-check against the 200-row ledger, rebuild the paper
(`docs/submission/build/`), regenerate every figure and table from the banked result JSONs under
`scratch/*/`, run the 616-test suite, and run `scripts/local_smoke.py` — which builds the *real*
`RavenForCausalLM` at 27.7M params in ~1 s, **no GPU and no weights download**. It *cannot*: run
anything needing the 3.5B checkpoint, and it does not have `scratch/ds_embsep/out` (1.0 GB of raw
arrays, deliberately excluded — the derived numbers are in the ledger).

## 1. Read order after a compaction

1. `docs/submission/REVIEW_QUEUE.md` — **the actual remaining work.** Seven framing choices no
   author has seen. Says: do not polish before that review.
2. `docs/submission/CLAIM_INVENTORY.md` — **the primary writing source.** 21 claims with
   presuppositions, experiment links (kernel + job id), gates, safe-vs-strongest forms; Part 0 =
   the anti-2D-PCA methodological case; Part 5 = presupposition index; Part 6 = what must not be
   claimed.
3. `docs/submission/SUBMISSION_STATE.md` — deadline, compliance, placeholders by owner, template
   traps already solved.
4. `docs/WORKING_KNOWLEDGE.md` — the tacit layer: how instruments actually behave, load-bearing
   vs incidental numbers, fragility map, read-order for a result, failure signatures.
5. `docs/submission/WRITING_PACK.md` §C (anti-overclaim) and §D (traps) before writing prose.

## 2. The three canonical documents (CLAUDE.md §6)

| file | role | size |
|---|---|---|
| `docs/claims_ledger.md` | **200 rows.** Every quantitative claim with its instrument, verification status, and — where applicable — its amendment/withdrawal. **Do not write prose from this directly**: many rows are amended by later rows on the same day. | 695 KB |
| `docs/UNDERSTANDING.md` | Current synthesis, rewritten not appended. **§1.0-today** is the day's verification surface (24 rows, 7 amendments). §1.0a external calibration, §1.0b the transient, §1.3 the map, §6.x blind spots. | 63 KB |
| `docs/directions.md` | Unrun experiments, costs, dequeued-with-why. **§M** live queue, **§N** pre-registered predictions, **§O** failure-cause recap + ranked next actions, **§P** resumption state, **§Q** A47 abandoned, **§R** compaction handoff. | 117 KB |

## 3. Submission working set — `docs/submission/` (mirror) and `../files/paper_submission/` (live)

**The live copies are outside any git repo.** `docs/submission/` is the backup; keep both in sync.

| file | what |
|---|---|
| `CLAIM_INVENTORY.md` | 21 claims, presuppositions, experiments, gates, Part 0 anti-PCA, Part 6 prohibitions |
| `RESULTS_HANDOFF.md` | numbers with scope, A1–A11 — **the only content source the draft was written from** |
| `WRITING_PACK.md` | §A spine, §B six final paragraphs, §C anti-overclaim table, §D traps, §E draft OpenReview fields (**unreviewed**), §F figures + data paths, §G A47 contingency (resolved: omit), §H related work + per-ref claim, §I novelty boundaries, §J reviewer objections, §K build mechanics, §L self-audit vs deceptive-practice list |
| `SUBMISSION_STATE.md` | compliance checklist, placeholders by owner, template traps, dependency-ordered remaining work |
| `REVIEW_QUEUE.md` | the seven unvetted framing choices; what to do next |
| `APPENDIX_B_RETRACTIONS.md` | full withdrawal/amendment list, written out |
| `build/` | **the paper, buildable from a fresh clone** — `main.tex` (CP1251), `zapiski.cls` (CP1251+CRLF), `pic/`, `main.pdf`, `instructions.md`. Two `pdflatex` passes; verified 2026-08-11 from a clean copy of that directory alone: 10 pages, 266993 bytes, 0 errors, 0 undefined citations. The six `Font shape 'T2A/...' undefined` log lines are Cyrillic font substitutions, **not** missing references — do not re-diagnose. Supersedes the deleted `main.tex.cp1251` snapshot. |

**Live draft:** `../files/paper_submission/draft/{main.tex,main.pdf,zapiski.cls,pic/}`.
10 pages, main body ends p9, 0 errors, 0 undefined citations, 0 uncited bibitems.
**Template + instructions:** `../files/paper_submission/{instructions.md,template-zapiski-main/}`.

## 4. Tools written this session — none of these are obvious from the code

| script | what it prevents |
|---|---|
| `scripts/local_smoke.py` | **Run before any new kernel.** Builds the *real* `RavenForCausalLM` at 27.7M params from the released source in ~1 s (4.25 s total). Exercises the four intervention patterns and reproduces two GPU-killing bugs as regression checks. |
| `scripts/preflight.py --launch <dir>` | 6 checks before a launch: unpushed commits, imports vs pip lines, `build_items()` runs, copied item keys, **and check 6 = model attribute paths**. |
| `scripts/model_attr_check.py` | resolves every `model.<...>` path against an attribute map read verbatim from the released source. Catches name errors only, not shapes. |
| `scripts/experiment_registry.py` | A-number collisions (two happened). Exits nonzero. **Next free: A53.** |
| `scripts/outcome_variance_scan.py` | outcome fields with base rate exactly 0 or 1 — a degenerate outcome reads exactly like a clean null. |
| `scripts/parser_sweep.py` | ranks 10 extraction rules over 1424 banked generations by **lift over a measured chance rate**. |
| `scripts/answerpos_analysis.py` | re-locates the answer position **without the gold**, so wrong answers are not silently dropped. Carries a recorded caveat: `last_number` is family-dependent. |
| `scripts/arc_table.py` | regenerates all 25 ARC measurements from banked JSON — cannot go stale. |
| `scripts/regime_onset.py` | the probe experiment that retired an instrument class (A37/D165). |
| `scripts/strategic_audit.py` | repeated costs, repeated attempts, what the recent record is *about*. |
| `scripts/build_index.py` | generates `docs/EXPERIMENT_INDEX.md`. |

**The one implementation that must never be duplicated:**
`src/traj_geom/metrics/dynamics.py::rotation_power` — raises rather than returning an aliased
number when `tail % period` or `tail // period < 2`. Two analyses compare across it.

## 5. Data — where the banked results are

- **49 result JSONs** under `scratch/*/`. Naming: `scratch/<kernel>/<name>.json`.
- **Zero-GPU re-analysis of these produced ~20 of the 200 rows** — D157–D159, D163–D165, D168,
  D174, D177–D179, D181, D182, D194, D195. **Check the bank before launching anything.**
- Large banks: `kaggle_census/out/census.json` (4868 draws × 48 unrolls, rank curves),
  `kaggle_depthacc/out/depthacc.json` (1260 generations, 21 families, 5 depths — the only bank
  with generated **text**), `ds_embsep/out/embsep.tgz` (204 arrays of [64, 4, 5280], 51 nouns).
- Per-position spectra: `ds_periods/periods.json` (full rfft power at every position, two tails —
  **any period readable offline forever**).
- Per-generated-position geometry: `ds_answerpos/answerpos.json` (`per_pos` = token, rank curve,
  `rot`, `resid` at every generated position).

## 6. Other directories

**In this repo:** `src/traj_geom/` (metrics, extraction, shapes — the one-implementation-per-
quantity home), `tests/` (616 passing; `test_preregistration_implemented.py` catches unwired
gates), `configs/`, `notebooks/`, `results/`, `trajectories/`, `figures/`, `h3_toy_model/`.

**Parent `~/build-projs/barannikov-work/`** — predates this repo, still referenced:
`files/` (below), `archive/`, `results/`, `trajectories/`, `scratch/`, `scripts/`, `src/`,
`kaggle_job/`, `kaggle_kernel/`, `figures/`, `code_env_info/`.

**`../files/` — source material worth knowing about:**
`contraction_proof.md`, `theoretical_framework.md`, `Scaling up Huginn Critique.md`,
`Smiles26Barannikov Proposal.pdf`, the annotated Huginn paper PDF, `deep_research_A/`,
`deep_research_battery/`, `ai_docs/`, `paper_submission/`.

**External, outside the repo:**
- `~/build-projs/huginn-load/` — **12 papers with source, code, reviews.** `01-huginn-*` contains
  the released `recpre/raven_modeling_minimal.py` that `local_smoke.py` imports, and
  `code/recurrent-pretraining/README.md:28,31` which pins the ARC protocol.
- `~/.claude/skills/ai-research-writing-skill/` — installed; `references/citation-checklist.md`
  found three citation defects.
- `~/Downloads/guides-write/` — `NotGoodIdeas.md` used for the §L self-audit; README unread
  (7.5k tokens, beginner manuscript handbook).

## 7. Operations — DataSphere

- Project `bt12q57tmrs03pnt8drc`. Weights dataset **`bt102r0j5cb8r6r6nb36`** — mounts in 7.7 s vs
  262–282 s to download. Instance `gt4.1` (Tesla T4).
- `GRPC_DNS_RESOLVER=native` on every call; launch from the kernel's own config dir.
- `job attach --id` is **safe on a running job** (verified: 166 jobs before, 166 after) but
  **does not replay a job-side traceback** on an ERRORed one.
- **Concurrency: launch 2–3, not 5.** Four in flight meant ~1 h wall for 60 s of compute.
- Monitors: only a **top-level `run_in_background` call** notifies on completion. A detached
  `( … ) &` survives but wakes nobody. Check for the **task id**, not the process.

## 8. Practice notes — `docs/PRACTICE.md`

RC1 proxy-for-the-thing · RC4 recomputed-vs-banked · RC5 RC1 inside an audit tool ·
**RC6** a degenerate outcome reads like a clean null · **RC7** a monitor inside another command
is not a monitor (+ amendment: a detached one survives but cannot wake you) · **RC8** preflight
checked everything except the model · **RC9** a platform ERROR is often the kernel refusing on
its own gate, with no traceback ever raised.

## 9. Older documents — present, mostly superseded

`RESULT.md` (the argument document; **superseded for writing by `CLAIM_INVENTORY.md`**),
`PLAN.md`, `SELF_REVIEW.md`, `OPEN_THREADS.md`, `REMOTE_RUNS.md` (DataSphere operational
history), `EXPERIMENT_INDEX.md` (generated), `related_work.md`, `depth_profile.md`,
`research_inquiry_{1,2,3}.md`.
