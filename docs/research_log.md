# Research Log

Format: reverse-chronological entries. Each entry is a "knot" — whatever mix of
hypothesis update, evidence, literature, discussion, and open questions actually
belongs together for that problem. Don't force separate sections if the problem
doesn't have them. Every entry ends with an explicit **Status** and, if unresolved,
**Open questions**.

Tags in use so far: `h1` `h2` `h3` `two-scale` `integrity` `infra` `paper` `stats`

---

## 2026-07-18 — Deep multi-agent review (6 independent dimensions + cross-check): the picture changed twice while this ran

Ran a 6-dimension, 22-agent adversarial review (stats, architecture, narrative,
repro, process, literature; findings cross-checked by independent skeptical
re-verification). Two sub-agents hit a session-usage limit mid-run and didn't
complete (stats and one cross-check slot) — everything below is from the 20
that did, plus the cross-check pass on 14 of 42 flagged findings before that
limit was hit too. **This entry supersedes prior framings in several places —
read it in full before trusting an older entry on the same topic.**

### The most important thing: the situation is live, not static

**A fourth branch, `origin/pr-real-data`, appeared on 2026-07-17 — after my
own prior session's docs were written, and describing them as current was
already wrong by the time this review ran.** It's PR #1's two commits plus
one new one (`900b8b4`, jack, 2026-07-17T18:12:03+03:00), adding
`docs/ANALYSIS_SUMMARY.md`, which **restates the band-test coefficient as a
clean, final result — +1.297, 95% CI [1.055, 1.548], p<0.0001, per-band
+1.565/+1.266/+1.184 — with zero mention of the length-matching problem**,
plus a brand-new, never-reviewed orthogonality result (content rho=+0.139
p=0.008, answer rho=-0.004 p=0.946). No PR is open against it — it's less
visible than PR #1, which itself is still untouched. **`guide.md`'s "nothing
since July 15" framing and `research_log.md`'s implied timeline are both now
stale because of this**, confirmed independently against GitHub's own
`pushed_at` field. This is not a historical finding to log calmly — it is
happening while the team sleeps toward a deadline, and cross-check confirmed
it (`verdict: confirmed`).

### The band-test problem is no longer an inference — there's a smoking gun

My 2026-07-17 entries called the length-matching failure "most likely a
code/data provenance mismatch" — implying the right code might vindicate the
result if found. **That's resolved, and not in the result's favor.**
`notebooks/02_two_scale.ipynb` (committed in the very PR under review, commit
`8efc31d`, already named in `guide.md`'s own branch map — nobody had actually
opened it) contains the real generating code (cell 14, "CENTERPIECE:
length-matched depth test") and its own printed execution output. The design
doesn't do exact-pair matching at all — it restricts each side to a
pool-level *overlapping seq_len range* and independently samples up to 60
per side from within it, no per-example pairing. Its own printed diagnostic:
`len_lo=171 len_hi=153` (band 2-3), `210 vs 189` (3-4), `253 vs 224` (4-5) —
the exact same 18/21/29-token gaps my own statistical audit found from the
CSV alone, same direction (higher depth → shorter realized sample) in all
three bands. **The notebook's own closing cell states its own success
criterion in plain language: "If depth-coef|len is clearly >0 ... with
len_lo~len_hi → acceleration tracks depth beyond length."** By that bar, using
the design author's own numbers, it doesn't clear it. This is now a directly
observed property of the honest, non-fabricated generating procedure,
self-documented as falling short of its own stated bar — not a mystery
about which code ran. (Also: `docs/proposed_patches/two_scale_branch_data_loaders.py`,
which I built without finding this notebook, guessed the wrong HF dataset
entirely — real code streams from the combined `qbao775/PARARULE-Plus`
repo, not the per-depth splits — so don't apply that reconstruction; it
would not reproduce `band.csv` if run.)

Separately, independent statistical re-analysis of `band.csv` (not just the
notebook) found the design is worse than "disjoint ranges": within-band
point-biserial r(hi, seq_len) = **−0.89 to −0.90** (near-total collinearity,
not just non-overlap — the pooled −0.337 I reported understates this a lot),
condition number 6088, and the `seq_len` coefficient's *sign flips* between a
seq_len-only model (−0.024) and the full model with `hi` included (+0.019) —
textbook suppressor-variable behavior. A within-band permutation test still
finds signal (p=0/2000), and a boundary-restricted 10-vs-10 comparison per
band stays significant (p<0.001, diff +1.3 to +2.3) — so this is not "the
effect is fake," but the exact +1.30 point estimate should not be quoted at
face value. Both the collinearity finding and the notebook smoking-gun
finding survived independent adversarial cross-check (`confirmed`).

### The project's actual best result has been sitting unpromoted

`results/forceloop.csv`, reprocessed independently: at `num_steps=16`,
unsettled (loop+drift) fraction rises from **1/8 (n_ops=8) to 7/8 (n_ops=24)**,
Fisher exact p=0.0101, vanishing entirely by `num_steps≥24`. This is a clean,
fully-reproducible (synthetic task, no missing loader, no PARARULE
dependency), statistically real H1 finding on `main`'s native pipeline,
independent of everything wrong with the two-scale branch — **and it has
zero mention in README.md's own Results section.** Cross-check: `confirmed`.
This is very likely the strongest single number available for the paper
right now, stronger than the disputed +1.30.

### Five concrete, ranked options for what the paper actually argues

A dedicated narrative-review pass (not just claim auditing) ranked five
framings by defensibility, independent of effort required:
- **(B) Two-scale band test as the headline positive result — least
  defensible.** Rests on an unmerged/unreviewed PR, a data loader that's
  never been in git, a matching design that fails its own stated bar (above),
  and a failed replication attempt (C6: the same metric found no
  track/local dissociation on the one task where `main`'s own design got its
  best positive result). A reviewer with 10 minutes and one `groupby` call
  finds this themselves.
- **(A) Both branches, fully caveated** — moderately defensible if the paper
  is as rigorous as `claims_ledger.md` already is internally, but must
  address C6's non-replication head-on, not in a footnote.
- **(C) Main only, drop two-scale, lead with forceloop** — cleanest single
  pipeline, but wastes the strongest result if forceloop isn't promoted to
  the headline.
- **(D) Reframe as a methodology/audit paper** — "what we learned trying to
  measure this," using the claims-ledger verification taxonomy itself as a
  citable contribution (catching and correcting the fabrication mid-project,
  documenting per-level-vs-per-row Spearman changing a conclusion by an
  order of magnitude, documenting a length-matched design failing its own
  bar). Most defensible to a hostile reader given exactly the evidence in
  hand, for a graded school report with a hard deadline.
- **(E) H3/contraction-theorem as the theoretical spine**, all empirical
  results marshaled as *consistent with, not proving* it (correct proof,
  A1; zero direct ρ(∂ₕR) measurement, A2 — still true, see below); coherent
  without needing the two-scale branch at all.

The single most damaging thing a hostile reviewer would say, verbatim from
the review: *"You have exactly one statistically significant positive number
in the entire project's core hypothesis tests, and it comes from an
unmerged, zero-review, zero-CI branch, computed via a data loader that does
not exist anywhere in your git history, using a length-matching design whose
own committed data contradicts its own matching code's logic, and when you
tried to cross-check the same mechanism on a second task design it came back
null."* The honest answer available: concede it, and reframe the
contribution around options D/E rather than defend B.

**My own earlier "signal lives on content tokens, not the answer token"
framing (2026-07-17 (2) entry) is flagged as an overclaim** — cross-check
confirmed it: it rests on a cross-branch comparison with three uncontrolled
confounds (`num_steps` 64 vs 32, different and differently-sourced loaders,
and — most importantly — omitting that C6 is *direct counter-evidence*
against treating it as a general mechanism, not just an isolated honest
negative alongside it). Don't repeat that framing without those caveats.

### Other confirmed findings worth carrying forward

- **Test coverage claim in `infra.md` overclaimed** (confirmed on cross-check):
  roughly half the model-free layer's public functions — all of
  `metrics/dynamics.py` (the metric the project's own docstring calls the
  one that "carried the MVP result"), `winding_of` (the actual function
  every experiment calls, vs. only the lower-level `winding_number`
  primitive being tested), all four synthetic-task generators, and
  `partial_spearman` (which backs the B2 "eaten by length" null) — have zero
  direct tests, despite needing no GPU.
- **`analysis/correlate.py`'s significance table has a real bug at N=4**:
  `crit[4]=1.000` is mathematically unreachable by exact permutation test
  (max achievable two-tailed significance at N=4 is p=0.083, never <0.05) —
  should read "n/a" like N>10 does, not a numeric threshold that looks
  reachable but never is.
- **No multiple-comparisons correction anywhere** across ~15-20 reported
  tests (confirmed on cross-check) — doesn't threaten the band test itself
  (p=6.6e-23 survives trivially) but is directly relevant to results sitting
  right at their significance boundary (B2, B6's winding=+0.943 at N=6,
  crit=0.886).
- **Single point of failure, quantified**: every commit touching real Huginn
  output traces to one person (Shtirmann/"Alexander Shiyanov" are confirmed
  the same GitHub account, not two contributors — `guide.md`'s branch-map
  header should be corrected). `jack`'s only contributions are one ~2hr
  feature commit and one ~20min docs commit. No trace anywhere in git of a
  third team member. If the person unavailable July 20-26 is the one who's
  been running Kaggle, every remaining model-touching task is blocked for 7
  of the final 15 days with zero demonstrated redundancy.
- **No paper-drafting artifact exists yet** — no `.tex`, no outline, 15 days
  out, despite the target template (`files/Smiles26Barannikov Proposal.pdf`)
  already being in hand.
- **A same-day, git-untracked file** (`files/Scaling up Huginn Critique.md`,
  mtime 2026-07-17) raises a premise-level question — does Huginn even do
  latent CoT reasoning at all (citing Lu et al. 2507.02199, already in the
  README's own bibliography, on logit-lens evidence against it)? The README
  already handles this fairly, but the file itself is invisible to
  teammates by construction (not in git, no author trace) — same
  no-shared-tracking pattern as everything else this review found.
- Two architecture-dimension findings about a "PR merge hazard from 4
  diverged shared scripts" did **not** survive cross-check (`refuted`) —
  don't carry those forward as confirmed.
- A claimed power-analysis number (N=6 → ~13% power at rho=0.7) was also
  `refuted` on cross-check — the general point that power is never
  quantified anywhere in the docs is still probably worth adding, but don't
  cite that specific percentage as verified.

### Status

Not yet folded into `guide.md`/`claims_ledger.md`/`end_to_end_guide.md` line
by line — this entry is the durable record of what changed; those three
docs need a pass reconciling them against it (flagged, not done, given
session constraints this pass ran under). Two staleness banners added
directly to `guide.md` §2 and `infra.md` §1 pointing back here in the
meantime.

The review resumed after its first quota-limited run (21/22 agents this
time, 1 remaining failure was a cross-checker hitting a structured-output
retry cap, not a quota wall) — no new critical facts beyond the above, just
two more confirmed cross-checks: paper option D (methodology-paper framing)
independently re-confirmed as most defensible, and the "content tokens"
overclaim finding re-confirmed via a second independent reproduction of its
underlying evidence (num_steps 64 vs 32, different loaders, C6 counter-
evidence). Nothing here changes the synthesis above.

### Open questions

- Someone needs to tell jack about the length-matching problem before
  `pr-real-data`'s doc gets treated as ground truth by anyone else.
- Decide, deliberately, which number goes in the paper: +1.30 (raw), +1.09
  (interaction-corrected), or neither (option D/E instead).
- `archive/task.md` still shows Phases 1-3 unchecked despite being
  substantially done — either revive it as a living tracker or replace it;
  right now it actively misleads about project status.
- Merging PR #1 is confirmed technically trivial (`gh pr view 1 --json
  mergeable` → `MERGEABLE`, zero conflicts) — the blocker is entirely
  social/procedural, not code.

---

## 2026-07-17 (6) — TRM baseline actually run; FSA task looks broken

Closing out the same session as (4)/(5). Last unverified claim in the whole
ledger was A4 (the toy TRM baseline in the archived `presentation.md` /
top-level `src/train_tiny_recursive.py`, unrelated codebase to the Huginn
work) — hadn't actually executed it yet, just read it. Set up an isolated
`.venv` at the repo top level (torch+numpy, CPU, ~1 min/run) and ran it.

**Confirmed:** unconstrained (β=0) gives 100% in-dist / 0% extrapolation,
every time. Contracting (β=5.0) is noisier than the presentation implies —
the script sets no random seed anywhere, so I got 100%/100%/90% in-dist
across 3 repeated runs. The "90%" is real and reproducible-in-principle, just
not the modal outcome. Extrapolation robustly 0% in every run of either
condition, exactly as claimed.

**New, unrelated finding while I was in there:** ran the other two synthetic
tasks too (parity, FSA — `task.md` checks off "counting, parity,
state-tracking FSA" as a completed deliverable, but only counting had ever
actually been evaluated). Parity is fine (100% in-dist, 46.75%
extrapolation — a believable hard-extrapolation result). **FSA looks broken
as a task construction, not just difficult**: 30% in-distribution (~random
for 3 classes — the model isn't learning it at all) but 50.75%
*extrapolation* — higher than in-distribution, which is backwards for a task
that's supposed to get harder with length. Suspect the FSA's transition
rules make longer sequences' final-state distribution concentrate
(more predictable from label statistics alone, independent of real
tracking) — not fully diagnosed, just flagged. Full detail: `claims_ledger.md` A4/A5.

**Session status, all 4 entries today:** every claim in `claims_ledger.md`
is now either independently re-derived or explicitly marked as blocked on
GPU access. Fixed locally (all uncommitted): `data/loaders.py` (both
branches), `.gitignore`, `analysis/plots.py`, `README.md`'s reproduce
command. Found and precisely quantified one real methodological weak point
in the project's strongest result (the two-scale band test's broken
length-matching). Corrected two of my own earlier mislabels (B3/B3c
source-attribution, B6 maxtask's incomplete negative-control framing).
Verified the citation trail is clean (10/10 real papers, matching content).
Nothing pushed or committed anywhere.

**Status:** done for this session. Everything actionable is written down
in `guide.md`, `infra.md`, and `claims_ledger.md` — this log is now the
place to check for *why* those documents say what they say.

---

## 2026-07-17 (5) — Full numeric audit of every cached experiment, incl. a self-correction

Continuation of the same session as (4). Re-ran every `scripts/run_*.py` that
has a cached CSV against the project's own code (not a from-scratch
recompute, that still needs GPU — but a genuine re-derivation of every
printed number from the committed data, not just trusting prose). Two real
findings, one of them a correction of my own earlier notes in this log.

**New finding — `maxtask` isn't the clean negative control I said it was.**
Re-running `run_maxtask.py`: `steps~n_ops` rho=−0.771 (n.s.) as documented,
but `winding~n_ops` rho=**+0.943 (N=6, significant)** — never mentioned in
the README or my own earlier B6 entry. Unlike `run_counting.py`, this script
never computes a length-partial correlation, and `n_ops` (digit-stream
length) is just as collinear with `seq_len` here as in the counting task —
so this is likely the same length confound as B2, not a real effect, but
that's now an inference, not something the code actually checked. `switch`,
by contrast, re-ran clean: both its metrics are genuinely null. Fixed in
`claims_ledger.md` B5/B6.

**Self-correction — I had mislabeled which script produced the "+0.84,
N=10" dissociation number.** That figure belongs to
`run_dissociation_multiinit.py` (`results/dissoc_multiinit.csv`, N_OPS has
10 levels) — the *only* one of the three dissociation-family results that
clears the project's own significance bar, and only in the pooled fit (the
per-init-seed spread 0.17–0.87 is the real caveat, already correctly
captured). I had incorrectly attributed that same number to the base
`run_dissociation.py` (`results/dissociation.csv`/`dissociation_15seed.csv`,
N_OPS has only 6 levels) in my first pass through this repo — re-running
both confirms the base script's own numbers are actually **n.s. at N=6**
(rho=+0.812 for 5-seed, +0.714 for 15-seed). Fixed in `claims_ledger.md` B3.
Noting this one plainly: this is exactly the kind of error that motivated
building this audit in the first place, and it was in my own notes, not the
team's.

**Confirmed, exact matches:** `run_switch.py`, `run_homology.py` (needed
`uv sync --extra tda` again after a from-scratch `.venv` rebuild dropped
it — settle(ns=64)=0.003, tight(ns=16)=0.0, exact), `run_convergence.py`
(the README's "9/9" is specifically the ns=64 subgroup of the 15 banked
trajectories — confirmed exactly: shrink×51, cos−0.276, dlr 12.0 — and as a
bonus found all 15, not just those 9, have conv_rate<0), `run_forceloop.py`
(loops emerge at num_steps=16, gone by 24+, matches).

**Status:** done for this pass. Every B-row in `claims_ledger.md` has now
been either independently re-derived or explicitly marked as not yet
re-run from scratch (GPU-gated).

**Open questions:**
- Should `run_maxtask.py`/`run_switch.py` get the same
  `partial_spearman(..., seq_len)` treatment `run_counting.py` already has,
  so the winding~n_ops numbers are actually interpretable?

---

## 2026-07-17 (4) — 2-hour local work session: fixes, an independent statistical audit, and a citation sweep

Everything below is **local, uncommitted** (`.gitignore`, `src/traj_geom/data/`,
`src/traj_geom/analysis/plots.py`, `tests/test_plots.py`, `docs/`) — nothing
pushed, no branch touched on GitHub. Full detail/rationale for each fix is in
`guide.md` and `infra.md`; this entry is the "what changed and why" log
+ the one big new finding.

**Fixed, verified locally on `main`:**
- `src/traj_geom/data/loaders.py` — reconstructed from
  `notebooks/01_mvp.ipynb` cells 10-11 (`load_pararule`, `enrich`). Verified:
  `python -m scripts.run_pararule` now runs clean and reproduces the
  README's exact published numbers (winding~depth rho=+0.20, steps~depth
  rho=+0.80, N=4).
- `.gitignore` — anchored the `data/` rule to `/data/` so it stops
  swallowing `src/traj_geom/data/`. Confirmed with `git check-ignore -v`.
- `src/traj_geom/analysis/plots.py` — implemented the two stub functions
  (`scatter_winding_vs_depth`, `plot_pca_trajectory`); added
  `tests/test_plots.py` (3 smoke tests).
- Full suite: **16 passed, 0 skipped** (`uv sync --extra dev --extra tda`,
  then `pytest -q`), ruff clean.
- Also reconstructed the PR branch's equivalent (`load_pararule_real`,
  `pararule_depth_of`) in a disposable worktree off
  `origin/two-scale-real-fix` — lower confidence (no surviving notebook cell
  for this one; built from the exact test contract in `test_two_scale.py`
  plus the real HF dataset schema for `qbao775/PARARULE-Plus-Depth-*`,
  confirmed via the dataset viewer). Full branch suite: 11 passed, 1 skipped.
  Not applied to that branch (not mine to commit) — saved as reference at
  `docs/proposed_patches/`.
- `README.md`'s reproduce block fixed too (`uv sync` → `uv sync --extra
  dev`), verified against a from-scratch-rebuilt `.venv`: 14 passed, 1
  skipped.

**Major finding — the two-scale band test's "length-matched" design doesn't
actually hold in the committed data.** I independently re-fit the band
regression with statsmodels (completely separate code path from
`analysis/band.py`'s `np.linalg.lstsq`): got coef(hi)=+1.2974, CI[+1.039,
+1.556], p=6.6e-23 — matches the project's own bootstrap almost exactly, so
**not a bug in the regression code**. But checking the design's own premise:
`_length_match()` pairs rows by *exact* `seq_len` equality, which requires
the two groups to share `seq_len` values. They don't — per band, `hi=0` and
`hi=1` `seq_len` ranges are **completely disjoint** (band 2-3: [162,187] vs
[149,159], gap=3; 3-4: gap=5; 4-5: gap=11), and depth/length stay correlated
within-band (pooled r=−0.337, p=5e-11). That's logically impossible for
`_length_match()` as currently written to produce (exact-key matching cannot
yield non-overlapping partner sets) — strong evidence the code currently on
`two-scale-real-fix` isn't the exact code that generated `results/band.csv`.
Consequence: the regression's `seq_len` control is extrapolating across a
gap, not interpolating within a shared range. Sensitivity check: adding an
`hi:seq_len` interaction drops the coefficient to +1.09 and weakens it to
p=0.025 — could be genuine sensitivity or just multicollinearity from the
added near-collinear term; can't tell from the data alone. Full detail and
exact numbers: `claims_ledger.md` C3.

**Literature sweep — 10/10 real, all checked out.** Verified via web search:
Geiping 2502.05171, Blayney 2604.11791, Pappone 2509.23314, Merrill
2404.08819, Grazzi 2411.12537, Yang 2605.26733 (STARS), Movahedi 2606.18206
(FPRM), Tulchinskii 2502.17017 (co-authored by Barannikov himself — the
project's curator), plus two spot-checks from the broader lit review (Jiang
2606.23590, Tuci 2604.19740). All real, all matching their cited claims.
Deep-checked two: Blayney's Appendix C ("Non-Fixed-Point Limiting
Behavior," footnote reads near-verbatim "this is rare, the vast majority of
tokens reach a fixed-point") — real, structurally matches, but couldn't
re-extract the exact 0.02%/2.81% digits (arXiv HTML fetch truncated before
the appendix body). Pappone's actual equations — exact match to
`metrics/two_scale.py`'s `compute_acceleration`/`compute_step_orthogonality`
(their Eq. 4, Eq. 1, and the cosine formula are implemented faithfully); one
fidelity gap found: the paper also defines a *normalized* acceleration
variant (Eq. 5) that the repo doesn't implement, which could matter for the
band-test result (see `claims_ledger.md` C11). **Net: no evidence of
citation fabrication anywhere — the data fabrication found earlier (C1) was
isolated to that one placeholder script, the literature grounding is solid.**

**Status:** open. The band-test methodological issue is the one thing here
that could change what goes in the paper — worth the team resolving before
Aug 2.

**Open questions:**
- Does anyone have the actual code (or a notebook) that generated
  `results/band.csv`, to check it against what's committed on
  `two-scale-real-fix` now?
- Worth re-running the band test with a stricter matcher that actually
  produces overlapping `seq_len` ranges, or with the paper's normalized
  acceleration (Eq. 5) as a robustness check, before trusting +1.30 in the
  submission.

---

## 2026-07-17 (2) — Codebase deep-dive: process, infra, and a verified reproducibility hole

**Why/what/how, for anyone new to the repo:** the process is inference-only —
Huginn-3.5B (`tomg-group-umd/huginn-0125`, HF `trust_remote_code`, pinned
revision, hidden dim 5280) is never trained or fine-tuned. A prompt is run
through `model.forward(..., num_steps=N)` (never `.generate()` for trajectory
work — `generate` doesn't expose the per-unroll hidden state), a forward hook
on `model.transformer.core_block[-1]` records the hidden state after every one
of the `N` recurrent unrolls, and that stack (`[N, hidden_dim]` for one token,
or `[N, seq_len, hidden_dim]` for all positions via the PR's
`extract_trajectory_allpos`) is the object every metric operates on. All of
`metrics/`, `shapes/`, `analysis/` is pure numpy/scipy/sklearn on that array —
CPU-only, model-free, and that's exactly why it's the fully-tested part
(`tests/` covers winding/gate/convergence/homology/correlate on synthetic
paths with known ground truth, and the correlate/gate ones double as the H1/H3
spec, e.g. `classify_shape`'s settle/loop/drift thresholds).

**Workflow, confirmed from the notebooks:** prototype on a Kaggle GPU notebook
(`01_mvp.ipynb`, markdown title literally "MVP H2/H3 · Huginn latent
trajectories (Kaggle)", `CACHE = "/kaggle/working"`) → once a piece of logic
survives debugging, port it into `src/traj_geom/<module>.py` with an
OWNER/STATUS/TASK/I/O docstring header (the convention is consistent across
every main-branch module) → `scripts/run_*.py` wraps it as a reproducible
experiment, and `scripts/_common.py::cached()` writes/reads `results/<name>.csv`
so that once a result has been computed once on a GPU (Kaggle), it can be
re-analysed by anyone with zero GPU and zero model weights — this is why the
README's reproduce section works without CUDA for most scripts. No CI, no
Docker, no cloud config committed anywhere (`.github/` doesn't exist) — GPU
compute is manual/Kaggle, not automated.

**Verified bug — PARARULE loading is silently gitignored on every branch:**
`scripts/run_pararule.py` (main) and the PR's `scripts/run_two_scale_depth.py`
both import `traj_geom.data.loaders` (`load_pararule` / `load_pararule_real` +
`enrich`). That module **does not exist in git, on any branch, at any commit,
ever** (`git rev-list --objects --all | grep loaders` is empty) — because
`.gitignore:2` has a bare `data/` rule, which matches *any* directory literally
named `data` anywhere in the tree, including `src/traj_geom/data/` (confirmed
with `git check-ignore -v`), not just top-level raw-dataset dirs the comment
("# Data & artifacts") intended. Confirmed by actually running it:
`uv run python -c "import scripts.run_pararule"` → `ModuleNotFoundError: No
module named 'traj_geom.data'`. Consequence: nobody but whoever has that file
locally (not visible in this repo) can currently regenerate `pararule.csv`,
`band.csv`, or anything else that touches PARARULE-Plus, from scratch — the
cached CSVs are trustworthy as numbers but not independently re-runnable by
the team right now. The good news: the underlying data isn't a large private
asset, it's a thin wrapper around `datasets.load_dataset("qbao775/PARARULE-Plus-Depth-{d}")`
(HF Hub, public) — the full logic is recoverable verbatim from
`notebooks/01_mvp.ipynb` cells 10–11. This is a five-minute fix (write the
module, anchor the gitignore rule to `/data/` or rename the package dir) but
nobody's done it yet on either branch.

**Other staleness/gaps noticed while reading, not yet acted on:**
- README's own reproduce block (`uv sync ; uv run pytest ; uv run ruff check .`)
  is broken as literally written — `pytest`/`ruff` are in the `dev` extra, so
  bare `uv sync` doesn't install them and `uv run pytest` silently falls back
  to whatever `pytest` is on `$PATH` (verified: picked up a system Anaconda
  pytest with no `traj_geom` installed → same confusing `ModuleNotFoundError`
  as above, for an unrelated reason). Needs `uv sync --extra dev`. Confirmed:
  with that, `11 passed, 1 skipped` (skip = `ripser`, needs `--extra tda`).
- `src/traj_geom/analysis/plots.py` is a stub (`raise NotImplementedError`) —
  the figures actually in `figures/` were made ad hoc (probably in-notebook or
  inside the `run_*.py` scripts directly, e.g. `run_dissociation.py`/`run_phase.py`
  have their own inline `plot()`), not through this module.
- `run_contrast.py` is explicitly self-marked as a superseded teaser (n=1
  prompt per class) — don't cite its numbers.
- Who is "David"? `results/FINDINGS.txt` and the PR blame the fabricated
  `accel rho=0.997` deck numbers on someone named David, but git history on
  this repo only ever shows `Shtirmann` and `jack` as authors — David isn't
  one of the 3 official SMILES participants as far as this repo shows. Worth
  asking the team directly who that is and whether they've been told the
  numbers were fake.

**Status:** analysis complete, no code changed. The loaders.py gap blocks
independent verification of every PARARULE-based result (both branches) and
is worth fixing before leaning on `band.csv`'s p<1e-4 result in the paper.

**Open questions:**
- Fix `data/loaders.py` + `.gitignore` now (I can reconstruct it from the
  notebook faithfully), or wait and ask whoever has the working local copy to
  push it, to avoid a divergent reimplementation?
- Does the `dev`/`tda` extras gap in the README need fixing before other
  teammates hit the same confusing test failure?

---

## 2026-07-17 (3) — Git archaeology + wrote `docs/guide.md`

Went deeper: full commit timeline (both branches, timestamps), PR #1's own
comments/reviews/CI (all empty — zero engagement since it opened), and
actually ran the PR branch's test suite in a disposable `git worktree`
(cleaned up after) rather than just reading it.

**New finding:** PR #1's `tests/test_two_scale.py` imports
`traj_geom.data.loaders` too (`pararule_depth_of`, on top of
`load_pararule_real`) — same missing-module bug as the 2026-07-17 (2) entry,
but on this branch it's worse: pytest aborts *collection* entirely on that
import error, so **none** of that branch's tests run in a normal `pytest`
invocation right now, not just the two-scale ones
(`--continue-on-collection-errors` confirms the other 7 pass fine once you
route around it).

**Temporal read:** `main`'s commits (July 10 scaffold → July 11 evening
extraction pipeline → July 12 02:36–22:53 overnight push, README finalized 7
minutes before the documented 23:00 deadline) and the two-scale branch
(forked July 12 20:02, built in parallel by `jack`, same deadline pressure)
line up well with a rushed-placeholder-data explanation for the integrity
issue, without excusing it. Nothing on either branch since July 15 16:46 —
repo's `pushedAt` confirms this. Full detail in `docs/guide.md` §2.

Wrote `docs/guide.md`: a stable "how do I operate this repo" reference
(setup commands, branch map, experiment catalog with solidity ratings per
result, the loaders.py fix plan with recovered code, open questions for the
team). This log stays the chronological/discussion trace; the guide is the
thing to hand someone new or reread before doing paper-writing.

**Status:** analysis only, still no code changes.

---

## Current state (snapshot — keep this short, detail lives in the log below)

- **H1 (three regimes):** weak support. Only "settle" observed on answer-token
  trajectories so far; no loop/drift seen. But sample size is far below the
  ~0.02–2.8% base rate Blayney et al. report for non-fixed-point tokens, so this
  is underpowered, not disconfirming.
- **H2 (winding tracks depth):** not supported on answer-token trajectories
  (PARARULE, counting). **But**: the `two-scale-real-fix` branch (unmerged) finds
  a real, significant depth effect on CONTENT tokens (depth-coef +1.30, 95% CI
  [1.05, 1.55], p<1e-4) using Pappone et al.'s two-scale metric instead of
  winding — coefficient independently reproduced with a separate stats stack
  (statsmodels), so not a code bug. Possible resolution: wrong metric/token
  position, not wrong hypothesis. **Caveat added 2026-07-17 (4):** the
  claimed length-matching doesn't actually hold in the data (disjoint
  `seq_len` ranges between the compared groups in every band) — the
  regression is extrapolating across a length gap, not interpolating within
  matched pairs. Coefficient survives a sensitivity check but weakens
  (+1.30→+1.09, p<1e-4→0.025). Resolve before trusting this as the paper's
  headline number. See 2026-07-17 (2) and (4) entries.
- **H3 (contraction bottleneck forces looping):** proof is solid
  (`files/contraction_proof.md`). Empirical test (`run_dissociation.py`, track
  vs local, length-matched) gives a directionally right but unstable signal
  (per-seed rho 0.17–0.87).
- **Known integrity issue (fixed in an unmerged PR):**
  `scripts/run_two_scale_analysis.py` on `feature/two-scale-latent-dynamics-paper`
  generated `np.random.randn` data with a hand-coded `5+2*depth` decay and it was
  presented as real Huginn output (deck numbers `accel rho=0.997`, `orth=-0.5`).
  Caught and fixed in PR #1 (`two-scale-real-fix` →
  `feature/two-scale-latent-dynamics-paper`), open since 2026-07-15, not yet
  merged anywhere.

---

## 2026-07-17 — Two branches, one unmerged integrity fix, and a possible H2 rescue

**What happened:** `main` and `feature/two-scale-latent-dynamics-paper` diverged
after commit `8b5e418` ("exp 3"). `main` carries the winding/dissociation/homology
work described in the top-level README (H1–H3 as originally framed, answer-token
only). The feature branch carries a separate line of attack using Pappone et
al.'s (arXiv:2509.23314) two-scale latent-dynamics metrics, authored mostly by
Shtirmann with at least one commit from `jack`.

**Integrity finding:** the feature branch's original `run_two_scale_analysis.py`
was a synthetic placeholder (`np.random.randn` + injected exponential decay
`5+2*depth`) whose output was used as if it were real Huginn data — the numbers
`accel rho=0.997`, `orth=-0.5` that apparently made it into a deck came from this
generator, not the model. PR #1 (`two-scale-real-fix`) fixes this: adds
`extract_trajectory_allpos` (all token positions, not just the answer token), a
real pipeline (`run_two_scale_depth.py`, `run_two_scale_real.py`,
`run_two_scale_counting.py`), and an honest results doc (`docs/two_scale.md`).
The PR is open, targets the feature branch (not main), unmerged as of this entry.

**Why this might matter for H2:** the real-data result in that PR, on
length-matched CONTENT tokens (not the answer token), is `depth-coef|len =
+1.30`, 95% CI [1.05, 1.55], p<1e-4 — band-by-band 2-3: +1.57, 3-4: +1.27, 4-5:
+1.18. The ANSWER token gives ~0 (+0.04, CI [-0.60, 0.53]) — consistent with
`main`'s null result, which only ever looked at the answer token (see README
"Ограничения"). Reading both together: the depth signal may be real but live on
question/content tokens, not the answer token — which lines up with Geiping et
al.'s own observation that orbits show up on question/digit tokens, not the
answer token (also flagged as a limitation in `main`'s README).

**Status:** open — real finding, not yet reconciled with main, not yet in the
paper draft.

**Open questions:**
- Merge PR #1 into the feature branch, and then figure out how (or whether) to
  merge the feature branch into `main` — two different metrics (winding vs
  two-scale) on two different token positions, currently living in two
  different branches. Does the paper want both, or does the CONTENT-token
  two-scale result replace winding as the primary H2 test?
- Does `jack` (or Shtirmann) already know this PR is still open, 2 days old?
  Worth a direct check before touching it.
- If the two-scale CONTENT-token effect replicates, does the answer-token
  dissociation experiment (H3, `run_dissociation.py`) need to be re-run on
  content tokens too, now that `extract_trajectory_allpos` exists?
- Who saw the fabricated `accel rho=0.997` numbers, and where (which deck)?
  Not present in this repo's archived presentation — must be an external slide
  deck. Worth confirming it was corrected wherever it was shown, given the
  paper submission is public (OpenReview).

**Stashed ideas (not started, ranked by what came up first):**
- Re-run the dissociation experiment (H3) with more seeds to resolve the
  unstable per-seed rho (0.17–0.87).
- Full read-through/audit of `src/traj_geom` + tests before trusting any metric
  implementation further (winding sign-arbitrariness and steps-to-settle
  already flagged as limitations in the README; not independently verified).
- Compute ρ(∂ₕR) via power-iteration + JVP (Yang et al. method) — direct
  empirical test of the Contraction Bottleneck Theorem on real Huginn, not
  attempted yet.
- Redesign the H2 synthetic counting task — current one hits 0% accuracy at
  length ≥ 8, so depth-correlation is untestable on it by construction.

---

(add new entries above this line, newest first — keep the "Current state" block
above in sync when a status actually changes)
