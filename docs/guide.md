# Operating guide — Geometry of Reasoning Trajectories

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


> **STALE AS OF 2026-07-24** for status/results claims (commands and repo
> orientation are likely still fine). Predates the coda-skip fix's second
> round, D11-D15, and a full literature-verification pass. Read
> `project_plan.md` + `claims_ledger.md` for current status. See
> `docs/verifiability_and_accountability.md`. Not rewritten, just flagged.

A single-handed reference: what this repo actually is, how to run it, what's
broken, and what the team was doing when. Written from reading every module,
running the test suites (main and the open PR branch, in a disposable
worktree) and the full commit timeline — not from the READMEs alone, which
are in places aspirational or stale. See `docs/research_log.md` for the
open, evolving discussion; this file is the "how do I actually operate this"
reference and should stay reasonably current.

## 1. Mental model

Inference-only study of Huginn-3.5B (`tomg-group-umd/huginn-0125`), a
recurrent-depth transformer: one shared transformer block is applied `N`
times per forward pass instead of stacking `N` distinct layers. The question
is whether the *geometric shape* of a token's hidden state across those `N`
unrolls (settle to a point / loop / drift) encodes how much reasoning the
model is doing.

Nothing is trained. A prompt goes through `model.forward(input_ids,
num_steps=N)` — never `.generate()`, which doesn't expose per-unroll state — a
forward hook on `model.transformer.core_block[-1]` records the hidden state
after each of the `N` unrolls, and everything downstream is geometry/topology
on that `[N, hidden_dim]` (or `[N, seq_len, hidden_dim]` for all-position
extraction) array: winding number, persistent homology, Lyapunov-style
step-decay metrics, and (on the unmerged branch) Pappone et al.'s two-scale
acceleration/orthogonality metrics.

## 2. Branch map (this matters more than the file tree)

Four lines of work exist. They are **not** in sync with each other.
`main`'s authorship is **one person under two git identities** — "Shtirmann"
(local git config) and "Alexander Shiyanov" (GitHub web UI commits) are the
same GitHub account (confirmed: same numeric user ID surfaces in `gh pr view
1 --json author`), not two contributors.

```
main (Shtirmann, one identity, two git configs)
  b1bc19c Initial commit             2026-07-10 12:25
  ae7b5eb base (full scaffold)       2026-07-10 20:36
  7030d3d Added test pipeline        2026-07-11 19:56
  17a944a exp 2                      2026-07-12 02:36  <- overnight push begins
  dd071d4 minor changes              2026-07-12 02:54
  8b5e418 exp 3                      2026-07-12 15:01  <-- feature branch forks here
  fe00b8b retesting, exp 4           2026-07-12 22:50
  14e77eb Revise README              2026-07-12 22:53  <- 7 min before the 23:00 deadline

feature/two-scale-latent-dynamics-paper (forked from 8b5e418, jack)
  155c6d0 feat: Two-Scale analysis   2026-07-12 20:02  <- built in parallel with exp3/exp4,
                                                            ships with FABRICATED data (see §5)

  [PR #1: two-scale-real-fix -> this branch, opened 2026-07-15, MERGEABLE
   with zero conflicts, still 0 comments/0 reviews/0 CI as of 2026-07-18:]
  2ac075e fix(two-scale): real data  2026-07-15 16:42  <- Shtirmann catches + fixes the
                                                            fabrication, but the fix's own
                                                            "length-matched" design turns
                                                            out to fail its own stated bar
                                                            too (see §5)
  8efc31d added notebook             2026-07-15 16:46  <- contains the ACTUAL original
                                                            band-test code + its own
                                                            execution output (see §5) —
                                                            nobody opened this until 07-18

  [pr-real-data, forked from two-scale-real-fix's tip, NO open PR, jack:]
  900b8b4 docs: add real-data summary 2026-07-17 18:12  <- restates the band-test
                                                             coefficient as clean/final,
                                                             zero mention of §5's problem,
                                                             plus a new unreviewed
                                                             orthogonality result
```

Reading the timestamps: July 10 was scaffolding. July 11 evening built the
extraction pipeline and smoke-tested it on a Kaggle notebook. July 12 was an
overnight-to-deadline push — `exp 2` lands at 02:36am, the README is
finalized at 22:53, seven minutes before the task.md-documented "July 12,
23:00 UTC+3" pre-defense deadline. **In parallel**, at 20:02 that same
evening, `jack` forked off `exp 3` to build a second research direction
(two-scale latent dynamics, a different metric family from Pappone et al.)
— and under the same deadline pressure, shipped it with placeholder
`np.random.randn` data standing in for real Huginn output. Three days later
(July 15, ~16:42), Shtirmann caught this, fixed it honestly in a new branch,
and opened PR #1 against the feature branch — but (found 2026-07-18, not at
the time) the honest fix's own generating notebook shows its "length-matched"
design doesn't clear its own stated success bar either (§5). The PR sat
completely untouched — 0 comments, 0 reviews, 0 CI — until **2026-07-17**,
when `jack` pushed one more commit to a *different*, PR-less branch
(`pr-real-data`) that restates the disputed number as settled fact, with no
mention of the problem. **This is live, not historical** — as of the last
check (2026-07-18), nobody has told `jack` about it yet.

Every commit touching real Huginn output — the entire `main`-branch sweep,
every notebook, the two-scale real-data fix — traces to the one Shtirmann
identity. `jack`'s only contributions are one ~2hr feature commit (07-12)
and one ~20min docs commit (07-17). No commit anywhere in the repo's history
is from a third person, despite this being described as a 3-person team —
worth confirming out of band who that is and what they're doing, since the
repo gives zero evidence of it.

If you're picking up where they left off: the two-scale branch, PR #1, and
`pr-real-data` are the three things nobody has reconciled yet, and doing so
(merging PR #1 is a technically trivial, zero-conflict, <1hr job — the block
is purely that nobody's claimed it) is probably higher leverage right now
than any new experiment.

## 3. Setup (verified commands — README's reproduce block was wrong, fixed locally 2026-07-17, see §5)

```bash
cd Geometry-of-Reasoning-Trajectories
uv sync --extra dev              # base deps + pytest/ruff — NOT just `uv sync`
uv run pytest -q                 # -> 14 passed, 1 skipped (ripser) on main, post-fixes
uv run ruff check .

uv sync --extra tda              # adds ripser/persim for homology
uv sync --extra model            # adds torch/transformers/accelerate — needs a GPU to actually load Huginn
```

GPU work in this project has always meant **Kaggle**, not a local machine or
any CI runner — `notebooks/01_mvp.ipynb`'s own title is "MVP H2/H3 · Huginn
latent trajectories (**Kaggle**)" and it writes to `/kaggle/working`. There is
no cloud config, Docker, or `.github/` in the repo — GPU runs are manual.
`scripts/_common.py::cached()` is what makes this tolerable: every
`scripts/run_*.py` writes its result to `results/<name>.csv` on first run and
just reads it back after, so **almost every script here is analysis-only and
needs no GPU** as long as the CSV is already committed (which it is, for
every experiment except the two-scale ones — see §5).

If you're on a Mac (no CUDA): `extraction/hook.py` and `extraction/model.py`
hardcode `.to("cuda")` / `device="cuda"`. You can run everything CPU-side
(tests, metrics on synthetic or cached-CSV data) but not a fresh
`extract_trajectory*` call without either patching the device string or a
remote CUDA box.

## 4. Experiment catalog (what's actually been run, and how solid it is)

All numbers are per-level Spearman on group means unless noted — that's the
project's own convention (`analysis/correlate.py::spearman_by_level`), chosen
specifically to avoid pseudoreplication inflating significance; per-row rho
is reported as a secondary, weaker number throughout.

| Script | Hypothesis | Result | Solidity |
|---|---|---|---|
| `run_pararule.py` (E1) | H2: winding tracks PARARULE-Plus proof depth | winding~depth rho=+0.20, n.s. (N=4); everything settles | Low N (4 depths), underpowered, **currently unrunnable from scratch** (§5) |
| `run_counting.py` (E2) | H2/H3: winding/steps track count length | steps~n_ops rises but confounded with prompt length | Superseded by the dissociation control below |
| `run_dissociation.py` / `_multiinit.py` (E3, "the killer experiment") | H3: length-matched track (needs accumulation) vs local (doesn't) | track steps~length rho=+0.84 (N=10, p<.05), but per-seed rho ranges 0.17–0.87; local sign flips across seeds | Directionally right, unstable — the multi-init version exists specifically to show the instability, not hide it |
| `run_phase.py` / `run_forceloop.py` (E4) | H1: starving compute budget forces loop/drift regimes | At num_steps=16: unsettled (loop+drift) fraction rises **1/8 (n_ops=8) → 7/8 (n_ops=24)**, Fisher exact **p=0.0101**; 0% unsettled at num_steps≥24 | **Probably the project's single cleanest positive result** — fully reproducible (synthetic task, no missing loader), significant, independent of every problem afflicting the two-scale branch. Not itself an H2 test, but currently **missing from README's own Results section** — add it |
| `run_switch.py`, `run_maxtask.py` (E5) | Generalize state-holding beyond counting | switch (parity) replicates the counting trend; maxtask does **not** (rho negative) | Effect isn't universal — say so, don't cherry-pick switch |
| `run_accuracy.py` | Behavioral check: does the model even solve counting? | 38% at length 2, **0% at length ≥ 8** | Critical caveat: E2/dissociation geometry is regressed against a task the model is failing at long lengths |
| `run_homology.py` | H1: genuine topological loops (H1 persistence) | max normalized persistence ≈0.003 (ns=64) | Expected ≈0 by construction on 1D curves with only 15 banked trajectories; not a real test of H1 at this N |
| `run_convergence.py` | Model-free convergence diagnostics (Pappone/Movahedi-style) | step size shrinks ×52 by the end, conv_rate<0 for 9/9 | Solid, cheap, CPU-only — good sanity metric to lean on |
| `run_two_scale_depth.py` (unmerged PR, "band test") | H2, re-tried: does depth show up on CONTENT tokens instead of the answer token? | CONTENT accel depth-coef\|len = **+1.30**, 95% CI [1.05, 1.55], **p<1e-4**; ANSWER token ≈0 | **Do not treat as clean.** The coefficient is real (independently reproduced with a separate stats library) but the design's own generating notebook (`notebooks/02_two_scale.ipynb`, cell 14/21, opened 2026-07-18) shows it doesn't hit its own stated "len_lo≈len_hi" success bar (realized gaps of 18/21/29 tokens across the 3 bands) — this is now confirmed from the original honest author's own code and printed output, not an inference. Within-band collinearity is severe (point-biserial r=−0.89 to −0.90, condition number 6088, seq_len coefficient sign-flips depending on model spec). A boundary-restricted comparison and a permutation test both still find signal, so this isn't fake — but don't cite +1.30/[1.05,1.55] as a clean length-controlled estimate. Currently **cannot be rerun from scratch** (§5), and a second replication attempt on a different task (`run_two_scale_counting.py`, C6 in `claims_ledger.md`) found **no** dissociation. |
| `run_contrast.py` | n=1 easy vs hard prompt teaser | — | **Self-flagged as superseded** in its own docstring; don't cite |

Net read: on the answer token (main branch's entire approach), H2 is null.
On content tokens, length-matched, the unmerged branch finds a real,
well-powered effect. That's the single most actionable thread right now —
main only ever looked at the answer token because that's what
`extract_trajectory` returns; the PR's `extract_trajectory_allpos` is what
makes the content-token result possible at all.

## 5. Known-broken, verified by actually running it (not just reading)

**`traj_geom.data.loaders` does not exist, on any branch, in the repo's
entire git history.** `git rev-list --objects --all | grep loaders` returns
nothing. Cause: `.gitignore` line 2 is a bare `data/`, present since the very
first scaffold commit (`ae7b5eb`, July 10) — it matches *any* directory
literally named `data` anywhere in the tree (confirmed with `git check-ignore
-v src/traj_geom/data/`), not just a top-level raw-dataset folder, which is
what the comment ("# Data & artifacts") clearly intended. Every script that
touches PARARULE-Plus imports from this module:

- `scripts/run_pararule.py` (main) — top-level import, so even the
  cached/no-GPU analysis path is broken:
  `uv run python -c "import scripts.run_pararule"` →
  `ModuleNotFoundError: No module named 'traj_geom.data'` (verified).
- `scripts/run_two_scale_depth.py` (PR #1) — import is inside `compute()`,
  so the cached path survives, but a from-scratch rerun would fail the same
  way.
- `tests/test_two_scale.py` (PR #1) — **verified by actually running the PR
  branch's test suite in a disposable `git worktree`**: pytest aborts
  collection entirely (`ModuleNotFoundError`), so right now **none** of that
  branch's tests run in a normal `pytest` invocation, not just the two-scale
  ones (`--continue-on-collection-errors` shows the other 7 pass fine once
  you route around the missing module).

Net effect: nobody on the team can currently regenerate `pararule.csv`,
`band.csv`, or any other PARARULE-touching result independently — including
the strongest result in the project (§4, the band test). The cached CSVs are
real (see §6 for why I believe that) but not independently re-derivable right
now by anyone except whoever still has the file locally.

**This is a small, recoverable fix, not a lost-data problem.** The actual
loading logic is a thin wrapper around `datasets.load_dataset` for the public
HF dataset `qbao775/PARARULE-Plus-Depth-{2,3,4,5}`, and the full logic for
the `main`-branch version is recoverable verbatim from
`notebooks/01_mvp.ipynb`, cells 10–11:

```python
DEPTH_REPOS = {d: f"qbao775/PARARULE-Plus-Depth-{d}" for d in (2, 3, 4, 5)}
CTX, Q, LAB = "context", "question", "label"

def load_pararule(depth, n=15):
    ds = load_dataset(DEPTH_REPOS[depth], split="train").shuffle(seed=0).select(range(n))
    for i, ex in enumerate(ds):
        yield {"prompt": f"{ex[CTX]}\nQuestion: {ex[Q]}\nAnswer:",
               "depth": depth, "label": int(ex[LAB]), "prompt_id": f"d{depth}_{i}"}

def enrich(row, tok):   # notebook version takes a global `tok`; scripts/run_pararule.py
    row["seq_len"] = int(tok(row["prompt"], return_tensors="pt").input_ids.shape[1])
    row["answer_token_index"] = -1
    return row
```

The PR branch additionally needs `load_pararule_real` (loads more examples
per depth for length-matching, per `docs/two_scale.md`'s band design) and
`pararule_depth_of` (depth from `meta['QDep']`, falling back to a `-D<d>-` id
tag — see `tests/test_two_scale.py`'s own spec for its exact contract). Those
two aren't recoverable from the notebooks I've read; they exist only wherever
whoever wrote `run_two_scale_depth.py` (Shtirmann, per the PR) still has them
locally.

**Fix is two parts, both done locally 2026-07-17 (uncommitted):** (1)
`src/traj_geom/data/loaders.py` written with the functions above, matching
the OWNER/STATUS/TASK convention — verified: `python -m scripts.run_pararule`
now reproduces the README's exact published numbers; (2) `.gitignore`'s
`data/` rule anchored to `/data/` so it no longer swallows source
directories. Full test suite: 16 passed. The PR branch's equivalent
(`load_pararule_real`, `pararule_depth_of`) was also reconstructed and
verified (that branch's suite: 11 passed, 1 skipped) but *not* applied
there — not mine to commit on someone else's open PR; saved instead at
`docs/proposed_patches/` for whoever owns that branch to review and apply.
Neither fix has been committed or pushed anywhere — still sitting local,
your call when/whether to commit.

**README's own reproduce block was wrong as written — fixed locally
2026-07-17.** `uv sync ; uv run pytest` (the original literal text) failed:
`pytest`/`ruff` live in the `dev` extra, so plain `uv sync` doesn't install
them, and `uv run pytest` then silently falls back to whatever `pytest` is
on `$PATH` — on this machine that was a system Anaconda install with no
`traj_geom` in it, producing the same misleading `ModuleNotFoundError: No
module named 'traj_geom'`, for a completely unrelated reason than this
section's real bug. `README.md` now reads `uv sync --extra dev ; uv run
pytest ; ...`. Verified end-to-end with a from-scratch `.venv` (deleted and
rebuilt): `14 passed, 1 skipped`. Uncommitted, one line.

## 6. Why I trust the cached numbers despite the reproducibility gap

Two independent things back the committed CSVs:

1. `tests/test_two_scale.py::test_bootstrap_band_content_signal` pins the
   exact band-test numbers as a regression test (`est ≈ pytest.approx(1.30,
   abs=0.05)`, CI bounds, `p < 1e-4`) against the committed `results/band.csv`
   — that's a real statistical claim someone was willing to hard-code an
   assertion around, not just a printed number in a doc.
2. The commit-by-commit diffs on `main` show result CSVs growing/changing
   row counts across separate commits (e.g. `dissociation_15seed.csv` first
   appears with 181 lines in `exp 2`, grows to 300 in `exp 3`) consistent
   with genuinely re-running experiments at increasing seed counts over the
   course of July 12, not a one-shot fabrication.

That said, this is inference from process, not a substitute for actually
re-running it — which is exactly what's currently blocked. Worth prioritizing
the loaders.py fix before leaning on `band.csv` further in the paper.

## 7. Open questions for the team (not resolvable by reading code)

- Who is **"David"**? `results/FINDINGS.txt` and PR #1 both name him as the
  source of the fabricated `accel rho=0.997` deck numbers, and
  `tests/test_two_scale.py` credits him with the original per-position
  two-scale metric code — but no commit in this repo's history is authored
  by anyone named David (only `Shtirmann`/`Alexander Shiyanov` and `jack`).
  Worth asking directly, and confirming wherever those numbers were shown
  got corrected.
- **Does `jack` know about the band-test length-matching problem?** More
  urgent than the PR sitting untouched: `jack`'s own 2026-07-17 commit
  (`pr-real-data`) restates the disputed number as clean, after the problem
  was already independently findable in the same PR's own notebook. Tell
  him directly before this propagates further.
- Who is the project's third team member (task.md / the original framing
  describes 3 people; the repo shows commits from exactly 2)?
- Is the two-scale content-token result (§4) meant to become the project's
  primary H2 evidence, replacing winding — or run alongside it, or dropped
  in favor of leading with the forceloop result (§4, E4) instead? Five
  ranked options for what the paper actually argues are laid out in
  `research_log.md`'s 2026-07-18 entry — this is the single highest-leverage
  decision left, and it's a paper-scope call, not a code one.
- `src/spectral.py`/`src/h3_validation.py`/`src/sequential_h3_validation.py`
  (top-level `src/`, outside this package) now give a working, self-tested
  recipe for measuring ρ(∂ₕR) via power iteration, run on the toy TRM model
  since no GPU/Huginn access exists here — worth porting to a real Huginn
  hidden state if/when GPU access is available; would upgrade H3 from
  "theory + indirect consistency evidence" to "theory + one direct
  measurement on the real model."
