# Start here

Six docs accumulated in this folder over two sessions (2026-07-17,
2026-07-18) plus code artifacts elsewhere in the repo. This is the index —
what to read, in what order, for what purpose. Written 2026-07-18, after a
fresh `git fetch` confirmed no remote activity since the facts below were
gathered (`origin/pr-real-data` still at commit `900b8b4`, no new PR, repo
`pushed_at` unchanged at 2026-07-17T15:13:11Z).

## If you only have 5 minutes

0. **A project-wide methodology defect was found 2026-07-19** (`claims_ledger.md`
   row D10, explained in full in `narrative.md` §7b): in *every* synthetic
   task, prompt length is a strict deterministic function of the difficulty
   variable — one `seq_len` value per `n_ops` level, rank-correlation
   exactly 1.0. So the `partial_spearman` length control, which E2's
   headline "the signal is eaten by prompt length" conclusion rests on, is
   mathematically degenerate: it correlates against ~1e-14 of
   floating-point noise. The *qualitative* conclusion survives and is
   actually stronger (depth and length are perfectly confounded by
   construction, hence inseparable), but no length control was ever
   actually run on a synthetic task. Fixing it needs a task-template
   redesign, not a different statistic. Directly affects how E2 and
   maxtask should be described in the paper.
1. A 4th branch, `origin/pr-real-data` (jack, 2026-07-17), is restating the
   project's headline two-scale coefficient (+1.297, p<0.0001) as clean and
   final, with **zero mention** that its own generating notebook
   (`notebooks/02_two_scale.ipynb`, cell 21) states a match criterion the
   result doesn't clear. No PR is open on it — nobody is reviewing it.
   **Tell jack before this gets treated as settled.**
2. The project's actual best, cleanest result — a real, significant,
   fully-reproducible H1 finding (loop rate jumps 1/8→7/8 as task difficulty
   rises under a starved compute budget, Fisher p=0.01) — is **not in the
   README** at all. Cheapest high-value addition available.
3. No paper draft exists yet, 15 days out. That is now probably the
   project's single biggest schedule risk, not any specific number.
4. Everything below is analysis/local-fixes only. **Nothing has been pushed
   to `origin` by me.** Three small code fixes (the `.gitignore` bug,
   README's reproduce command, the `analysis/plots.py` stub) do exist as
   local commits on a branch called `fix/repro-and-scaffolding` — `git log
   main..fix/repro-and-scaffolding` shows them — but that branch has not
   been pushed anywhere either; `main` itself is untouched and matches
   `origin/main` exactly.

## Reading order, start to finish

1. **This file** — orientation, you're here.
2. **`narrative.md`** — **the main read; start here if you read only one
   thing.** One linear pass through the whole project in 14 sections:
   every concept defined before it's used and grounded in a concrete
   instance (real prompts, a real trajectory array dissected step by step,
   real numbers), then every experiment, then a scorecard table against
   H1/H2/H3, then what's open, then how to actually run the code. It's
   long but self-contained, and every number in it is the final corrected
   one. If you've read an older copy, §7b (the length-control defect) and
   §3 (concrete measurement walkthrough) are new.
3. **`research_log.md`** — read the **2026-07-18 entry first** (it
   supersedes several earlier framings), then skim backward through the
   2026-07-17 entries for how the picture got built up. This is the
   discussion trace — messiest but most complete, includes things that got
   found, then corrected, then found again by an independent check. If you
   read only one other doc, make it this one's newest entry.
4. **`claims_ledger.md`** — every specific number/claim in the project,
   tagged Verified-live / Trusted-cached / Confirmed-fabricated /
   Contradicted / Unimplemented, with the exact evidence for each. Use this
   when you need to know "is this specific figure actually true" rather
   than the general narrative.
5. **`end_to_end_guide.md`** — the longest, most complete single reference:
   full hypothesis statements, every experiment with exact commands and
   tunable parameters, statistical conventions, a cookbook for extending the
   code. This is what to reread before running or writing an experiment
   yourself. Fully reconciled against the 2026-07-18 review (branch map,
   two-scale section, statistics conventions, ranked issues, and the H3
   spectral-radius recipe — now actually run, on a toy model — are all
   current as of this file).
6. **`guide.md`** — shorter operating reference (setup commands, branch map,
   known-broken things). Reconciled, current.
7. **`infra.md`** — architecture/dependency-graph/conventions/cookbook, the
   "how is this codebase organized" reference. Reconciled, current — includes
   a real bug found in `analysis/correlate.py`'s significance-threshold
   table (§3b) not mentioned anywhere else.
7. **`proposed_patches/`** — reconstructed `data/loaders.py` for both
   branches (the module that was silently gitignored out of existence on
   every branch, forever). The main-branch one is verified good. **The
   two-scale-branch one is verified WRONG** — it guessed the wrong HF
   dataset; the real code was later found sitting in
   `notebooks/02_two_scale.ipynb` on that branch. Its `README.md` explains
   which is which.
8. **`architecture_state.md`** (added 2026-07-22) — not part of the reading
   order above, a living reference kept up to date going forward: current
   file/script/results inventory, an experiment status table, a dated
   decisions log, and pointers to the (repo-external) `code_env_info/`
   Kaggle/Yandex DataSphere notes. Check this before touching a file or
   asking "has this been run," rather than rereading the narrative.
9. **`open_question_depth_calibration.md`** (added 2026-07-22) — **not
   implemented**, eval-only proposed test: whether Huginn's depth-randomized
   training actually produced a difficulty-calibrated stopping signal, or
   whether the native exit criteria are heuristics riding on an uncalibrated
   process. No training involved; a clean negative result is treated as a
   real finding, not a failed check. Blocked on the same GPU rerun already
   queued elsewhere.

## Where the non-doc work lives

- **`../src/h3_results/FINDINGS.md`** (top-level `src/`, a separate,
  unrelated toy codebase, not part of the git-tracked
  `Geometry-of-Reasoning-Trajectories/` package): a from-scratch empirical
  test of H3 (the Contraction Bottleneck Theorem) on models this project can
  actually run without a GPU. Two sweeps: a recurrent-over-**depth** toy
  model (same information-flow pattern as Huginn — full context re-injected
  every step) where count information survives strong contraction almost
  perfectly (linear-probe R²≥0.996 across the whole β range); and a
  purpose-built recurrent-over-**time** counterpart (classic RNN, one token
  per step, matching what the theorem's own worked example assumes) where
  the same contraction produces a sharp collapse (R² craters to ≈0.00 the
  moment ρ drops meaningfully below 1). Real, striking, worth the team's own
  scrutiny — argued from architecture, not proven with the rigor of the
  Banach argument itself. Includes `spectral.py`, a self-tested
  power-iteration ρ(∂ₕR) estimator directly portable to real Huginn if/when
  GPU access is available.
- **`docs/infra.pdf`** — a 4-page PDF sitting next to `infra.md`, not
  something either work session produced; looks like a local export,
  presumably yours. Not analyzed further here.
- **`files/Scaling up Huginn Critique.md`** (parent `files/` folder, not
  git-tracked, no author trace): a same-day (2026-07-17) note asking
  whether Huginn even does latent CoT reasoning at all, citing Lu et al.
  2507.02199 (already in the README's own bibliography). The README already
  handles this fairly. Whoever wrote it should share it with the team —
  right now it's invisible to anyone else by construction.
- **`archive/`** (parent folder): the original pre-defense `task.md` and
  slide decks. `task.md` still shows Phases 1-3 unchecked despite them
  being substantially done on `main` by 2026-07-12 — actively misleading
  about project status if anyone still checks it.

## Recommendations, ranked

1. **Tell jack about the length-matching problem** before `pr-real-data`'s
   `docs/ANALYSIS_SUMMARY.md` propagates further. This is the one item on
   this list with a real, live cost to delaying.
2. **Merge PR #1.** Confirmed technically trivial — `gh pr view 1 --json
   mergeable` returns `MERGEABLE`, zero conflicts. The block is purely
   social (nobody has looked at it in 3 days), not code. Do this, then make
   one explicit decision about reconciling `main` vs the feature branch for
   the paper — don't let a 4th branch keep the two-scale line permanently
   separate from `main`.
3. **Decide the paper's actual argument now, not later.** Five ranked
   options are laid out in `research_log.md`'s 2026-07-18 entry (A through
   E). Given the evidence in hand, options **D** (frame as a methodology/
   audit paper — the claims-ledger process itself, including catching a
   real fabrication mid-project, is a legitimate citable contribution) or
   **C** (main-only, lead with the forceloop result, drop two-scale
   entirely) are the most defensible with the least remaining risk. Option
   **B** (two-scale band test as the headline positive finding) is
   explicitly the least defensible — an unmerged, unreviewed, non-
   reproducible result that failed its own author's stated success bar and
   didn't replicate on a second task design.
4. **Add the forceloop result to the README's Results section.** Already
   computed, already significant (Fisher p=0.01), zero additional work,
   currently the project's most underused asset.
5. **Start the actual paper draft.** Nothing exists yet — no `.tex`, no
   outline — 15 days from the deadline, and the target template
   (`files/Smiles26Barannikov Proposal.pdf`) is already in hand. Given the
   team's demonstrated pattern (strong burst velocity on experiments, weak
   on integration/writeup), this is now plausibly the biggest schedule risk
   in the project, independent of which number ends up being cited.
6. **Front-load any remaining GPU/Kaggle work before 2026-07-20.** Every
   commit that's touched real Huginn output traces to one person; if that's
   the person going dark for a week, anything not run before then likely
   doesn't happen before the deadline.
7. **Fix the significance-threshold bug** in
   `src/traj_geom/analysis/correlate.py` (`crit[4]=1.000` is mathematically
   unreachable — should read "n/a" like N>10 already does). Small, cheap,
   prevents a possible future false "sig" claim.
8. **Decide when to commit/push the local fixes** described in these docs
   (`data/loaders.py` on both branches, `.gitignore`, `analysis/plots.py`,
   `README.md`'s reproduce command) — they're verified working but sitting
   uncommitted; nobody else benefits from them until they're pushed, which
   is your call, not something done autonomously.
