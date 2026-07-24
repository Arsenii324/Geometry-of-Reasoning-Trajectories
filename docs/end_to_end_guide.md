# End-to-end guide: Geometry of Reasoning Trajectories

> **STALE AS OF 2026-07-24** for anything result-related (hypothesis status,
> claimed numbers) — commands/conventions below are likely still accurate,
> but every claimed *result* predates the coda-skip fix's second round,
> D11-D15, and a full literature-verification pass. Cross-check any specific
> number against `claims_ledger.md`; read `project_plan.md` for current
> status. See `docs/verifiability_and_accountability.md`. Not rewritten,
> just flagged.

One document meant to let you operate this project alone: understand why it
exists, review what's actually been established vs merely claimed, rerun any
experiment with different parameters, and write new ones that fit the
project's conventions. It draws on and cross-references the other three
docs in this folder (`research_log.md` — chronological discussion trace,
`guide.md` — shorter operating reference, `infra.md` — architecture deep
dive, `claims_ledger.md` — the exhaustive claim-by-claim audit this
document's numbers are drawn from) but is written to be readable on its own.
Every number below was independently re-derived or explicitly marked as not
yet re-derivable; see `claims_ledger.md` for the row-by-row evidence trail
if you want the receipts for a specific figure.

Written 2026-07-17. If you're reading this much later, distrust anything
time-sensitive (branch state, PR status, "current" numbers) and re-check
against a fresh `git fetch` and re-run — the whole point of this project's
`cached()` design is that re-checking is cheap.

---

## 1. What this project is and why

**The question:** models like Huginn-3.5B (Geiping et al., NeurIPS 2025)
don't scale test-time compute by writing more chain-of-thought tokens —
they iterate a single shared transformer block `R_θ` in latent space,
`h_{t+1} = R_θ(h_t; e)`, for a chosen number of recurrent unrolls `T`, before
ever producing a token. This project asks whether the *geometric shape* of
that latent path `h_0 → h_1 → … → h_T` — does it settle to a point, loop, or
drift? — encodes something about the depth or type of reasoning the model is
doing, and whether that shape is a necessary consequence of the model
needing to hold state (a running count, a proof-chain position) rather than
just noise.

**Three hypotheses**, stated precisely (from `presentation.md`, archived,
and `files/contraction_proof.md`):

- **H1 (a few shapes).** Each token's latent path falls into one of three
  regimes distinguishable by simple measurements: **settling** (fixed point,
  step displacement → 0, Lyapunov exponent λ<0), **looping** (limit cycle,
  λ≈0, nonzero winding number / persistent H1 homology), or **drifting**
  (λ>0, unbounded step displacement).
- **H2 (loops track reasoning depth).** When a path loops, its winding
  number scales with the logical depth of the problem — more reasoning
  steps needed, more turns in the latent path.
- **H3 (why looping is necessary — the Contraction Bottleneck Theorem).** If
  `R_θ` is forced to be a strict contraction (Lipschitz constant `c<1`,
  everywhere), the Banach fixed-point theorem guarantees convergence to a
  unique `h*(e)` independent of `h_0`, and the memory of any two initial
  states decays as `c^t`. Formally: to keep `N` distinguishable historical
  states apart under floating-point precision `ε`, the maximum useful unroll
  depth is `T_max ≤ log(D/ε) / log(1/c)` where `D` is the state-space
  diameter. **Consequence:** a model regularized to "always settle" (e.g.
  STARS's Jacobian Spectral Radius regularization, ρ(∂ₕR)<1) should be
  structurally unable to hold a running count or FSA state past `T_max`
  steps — such tasks should force looping or drifting, not settling. This
  proof is checked and correct (`claims_ledger.md` A1); what's *not* done
  yet is measuring the real Huginn's actual ρ(∂ₕR) to see where it sits
  relative to 1 (A2 — open, see §9.5 for how to attempt it).

**Why this matters beyond curiosity:** if H1–H3 hold, it suggests recurrent-
depth reasoning models have an interpretable internal signature of "how hard
they're thinking" and a principled reason certain training regularizations
(anything that pushes toward strict contraction, which is a natural thing to
do for training stability — see STARS, FPRM in the literature review) would
actively destroy state-tracking capability. That's a real tension worth
documenting even if H2 specifically turns out weak, which — spoiler for
§4 — it currently does on the metric the project started with (winding
number on the answer token) and does *not* on a different metric/token
position, which is the most important open thread in the whole project
right now.

---

## 2. The system under study

- **Model:** `tomg-group-umd/huginn-0125` (Huginn-3.5B), pinned at revision
  `bb6621b65e90b6a4b9b29ef88dc83866d450470c`. Loaded via
  `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` — it's
  a custom architecture (`RavenForCausalLM`), not a stock HF model class.
  Hidden dimension: **5280**.
- **Why `trust_remote_code` and a pinned revision matter:** the model ships
  custom modeling code from the HF Hub repo itself, and that code has only
  been verified to work with `transformers` **4.50–4.53** —
  `constants.py`'s own docstring documents exactly why, from hard debugging:
  below 4.50, `_get_initial_cache_position` lacks a `device` arg the custom
  code needs; at 4.54+, `key_cache` becomes a read-only property, breaking
  the custom code; at 5.x, tied-weights loading expects a different format.
  The project pins `transformers==4.53.3` exactly. **If you ever bump this
  dependency, expect the model load itself to break**, not a downstream
  metric — this is the single most fragile dependency in the project.
- **Inference only, never trained.** Every experiment is a forward pass;
  nothing here fine-tunes Huginn.
- **The recurrent unroll:** `model(input_ids=ids, num_steps=N)` — note
  `num_steps` must be a plain Python `int`, not a 0-d tensor (`len()` on a
  0-d tensor crashes the custom forward). Use `.forward()` (via `model(...)`)
  for trajectory work; use `.generate_with_adaptive_compute(...)` only when
  you want the model's actual decoded answer (accuracy checks), since
  `.generate()` doesn't expose per-unroll hidden states at all.
- **How the trajectory is captured:** a `torch` forward hook registered on
  `model.transformer.core_block[-1]` — the last sub-block of the recurrent
  container — fires once per unroll and appends the hidden state to a
  Python list. Three hard-won gotchas, all documented verbatim in
  `constants.py` and `extraction/hook.py`, and all worth re-reading before
  touching this code:
  1. `mod._forward_hooks.clear()` before registering, and
     `hook.remove()` in a `finally:` — a hook left behind by a crashed prior
     run silently doubles every subsequent capture.
  2. `h_0` (the very first recurrent state) is a **random initialization**
     (`torch.manual_seed(seed)` before the forward controls it) — it's an
     outlier relative to the rest of the path, so winding/shape metrics use
     a burn-in (default 4 steps) that drops it.
  3. The captured tensor must be `.detach().cpu()`'d immediately inside the
     hook, or GPU memory grows unbounded across a sweep.
- **Token position:** most of `main`'s experiments only ever look at
  `token_index=-1` (the answer/last token) — `extract_trajectory` returns a
  `[num_steps, hidden_dim]` array for one token position. The unmerged PR
  branch adds `extract_trajectory_allpos`, returning
  `[num_steps, seq_len, hidden_dim]` — every position at once, single GPU,
  sequential (no batching trick, just a bigger forward hook payload). This
  distinction — answer token vs. all/content tokens — turns out to be the
  single most consequential methodological choice in the whole project (see
  §4.2).

---

## 3. Repository landscape: branches, who did what, and current state

Four lines of work exist and are **not in sync**. This matters more than
the file tree — read this before doing anything else. `main`'s authorship is
**one person under two git identities** ("Shtirmann," local git config, and
"Alexander Shiyanov," GitHub web UI — confirmed the same account via a
matching numeric GitHub user ID), not two contributors as it can look at a
glance.

```
main (Shtirmann, one identity, two git configs)
  b1bc19c Initial commit             2026-07-10 12:25
  ae7b5eb base (full scaffold)       2026-07-10 20:36  <- .gitignore bug born here
  7030d3d Added test pipeline        2026-07-11 19:56
  17a944a exp 2                      2026-07-12 02:36  <- overnight push begins
  dd071d4 minor changes              2026-07-12 02:54
  8b5e418 exp 3                      2026-07-12 15:01  <-- feature branch forks here
  fe00b8b retesting, exp 4           2026-07-12 22:50
  14e77eb Revise README              2026-07-12 22:53  <- 7 min before the documented
                                                            "July 12, 23:00 UTC+3" deadline
  [nothing since]

feature/two-scale-latent-dynamics-paper (forked from 8b5e418, jack)
  155c6d0 feat: Two-Scale analysis   2026-07-12 20:02  <- built in parallel with main's
                                                            exp3/exp4, under the same
                                                            deadline pressure — shipped
                                                            with FABRICATED placeholder
                                                            data (see §4.8)

  [PR #1 "two-scale-real-fix" -> this branch, opened 2026-07-15 13:43 UTC,
   MERGEABLE with zero conflicts, still OPEN, 0 comments, 0 reviews, 0 CI]
  2ac075e fix(two-scale): real data  2026-07-15 16:42  <- Shtirmann catches + fixes
                                                            the fabrication himself —
                                                            but the fix's own "length-
                                                            matched" design turns out to
                                                            fail its own stated bar too
                                                            (§4.8)
  8efc31d added notebook             2026-07-15 16:46  <- contains the ACTUAL original
                                                            band-test code + its own
                                                            execution output — nobody
                                                            opened this until 07-18

  [pr-real-data, forked from two-scale-real-fix's tip, NO open PR, jack]
  900b8b4 docs: real-data summary    2026-07-17 18:12  <- restates the band-test
                                                            coefficient as clean/final,
                                                            zero mention of the length-
                                                            matching problem, plus a new
                                                            unreviewed orthogonality
                                                            result
```

Repo `pushedAt` (GitHub API): 2026-07-17 15:13 UTC, reconfirmed via a fresh
`git fetch` as of this writing. If you fetch and see anything newer, this
section is stale again — re-derive it (`git log --all --oneline`, `gh pr
list --state all`, and don't forget `git fetch` enumerates remote branches
you haven't explicitly fetched before, which is exactly how `pr-real-data`
was missed for a full day).

**Reading the timeline:** `main` was a solo overnight-to-deadline sprint —
scaffold day 1, extraction pipeline evening 2, then a ~20-hour push from
02:36 to 22:53 on day 3 that built out the entire H1–H3 experiment suite
(§4), finishing 7 minutes before the pre-defense deadline documented in the
archived `task.md`. In parallel, `jack` forked off mid-sprint (20:02 the
same evening) to try a second metric family (Pappone et al.'s two-scale
dynamics) — under the same time pressure, and shipped a script that
generated synthetic placeholder data and presented it as a real result in
some deck. Shtirmann caught this three days later and fixed it honestly,
opening PR #1 — but the honest fix's own design has its own, separate
problem (§4.8), only found 2026-07-18. The PR sat completely untouched
until **2026-07-17**, when `jack` pushed one more commit to a different,
PR-less branch that restates the disputed number as settled, with no
mention of the problem. **This is live, not historical** — as of the last
check, nobody has told `jack` about it.

Every commit touching real Huginn output — the entire `main`-branch sweep,
every notebook, the two-scale real-data fix — traces to the one Shtirmann
identity; `jack`'s only contributions are one ~2hr feature commit and one
~20min docs commit. No commit anywhere is from a third person, despite this
being described as a 3-person team — worth confirming out of band.

**If you're picking this up cold:** the two-scale branch, PR #1, and
`pr-real-data` are the three things nobody has reconciled yet — merging PR
#1 is confirmed technically trivial (zero conflicts), the blocker is purely
that nobody's claimed the hour of work. `main`'s `README.md` is
comprehensive and (mostly) honest about its own limitations, but is missing
its own best result (§4.4) — start there for the "official" project
narrative, then read this document for everything that narrative doesn't
say.

**Local fixes applied 2026-07-17, all uncommitted — nothing pushed, your
call when/whether to commit:**
- `src/traj_geom/data/loaders.py` written (was missing from git entirely,
  on every branch, forever — see §7). Verified against the cached CSV.
- `.gitignore`'s `data/` rule anchored to `/data/` so it stops silently
  swallowing source directories.
- `src/traj_geom/analysis/plots.py` implemented (was a stub).
- `README.md`'s reproduce command fixed (`uv sync` → `uv sync --extra dev`).
- A best-effort reconstruction of the PR branch's equivalent loader saved
  at `docs/proposed_patches/` — **confirmed wrong as of 2026-07-18** (see
  §4.8; don't apply it, the real code was later found in
  `notebooks/02_two_scale.ipynb`).

---

## 4. Full experiment walkthrough

Every experiment below: what it tests, the exact command, the tunable
parameters (for scenario 2 — rerunning with different values), the current
numbers, and the caveats (for scenario 1 — review). All scripts live in
`Geometry-of-Reasoning-Trajectories/scripts/`, are run as
`uv run python -m scripts.<name>` from that directory, and follow the same
pattern: `compute()` does the (GPU-requiring) work, `scripts._common.cached()`
transparently skips `compute()` and just reads `results/<name>.csv` if it
already exists, so **every command below runs with zero GPU right now** —
you're re-deriving the printed analysis from committed data, not
regenerating it, unless you delete the CSV first (which does require a GPU
and the `model` extra).

A note on comparability before you dive in: different experiments use
different `num_steps` budgets — `main`'s H1/H2/H3 experiments mostly use
**64** (`DEFAULT_NUM_STEPS` in `constants.py`), the two-scale branch's
experiments use **32**, and the accuracy check (`run_accuracy.py`) uses 32
for the `generate` call but 64 for the paired trajectory. This is a real
methodological wrinkle if you ever want to compare a "settle" classification
from one experiment against another — they aren't always looking at the
same compute budget.

### 4.1 E1 — PARARULE-Plus depth vs. answer-token winding (`run_pararule.py`)

- **Tests:** H2, on the answer token.
- **Command:** `uv run python -m scripts.run_pararule`
- **Parameters (top of file):** `N_PER_DEPTH = 10` (examples per depth
  level); depths are hardcoded to `(2, 3, 4, 5)` inside `compute()`;
  `num_steps=64` is hardcoded in the `extract_trajectory` call. To test a
  different depth range or more examples per depth, edit these directly —
  there's no CLI flag.
- **Data source:** `qbao775/PARARULE-Plus-Depth-{2,3,4,5}` on the HF Hub, via
  `traj_geom.data.loaders.load_pararule` (a thin wrapper: shuffled sample of
  `n` examples per depth, prompt = `f"{context}\nQuestion: {question}\nAnswer:"`).
- **Current result:** all 40 trajectories settle. `winding~depth`
  (per-level, canonical stat — see §6) rho=+0.20, N=4, **not significant**
  (critical value at N=4 is 1.000 — with only 4 depth levels, nothing short
  of a perfect correlation clears the bar). `steps~depth` rho=+0.80, also
  n.s. at N=4. Per-row partial (controlling for prompt length): rho=−0.23,
  p=0.15.
- **Caveat:** N=4 levels is inherently underpowered for a correlation test —
  this experiment cannot distinguish "no effect" from "real but small
  effect we can't see at N=4." If you want to strengthen this specifically,
  raising `N_PER_DEPTH` doesn't help (it doesn't add depth *levels*, just
  more rows per level); you'd need PARARULE-Plus at more depth values, and
  the public dataset only goes up to depth 5.
- **To rerun with different params:** raise `N_PER_DEPTH` for more rows per
  level (helps precision, not the N=4 significance-floor problem); this is
  the cheapest experiment to iterate on since it needs no length-matching
  logic.

### 4.2 E2 — synthetic counting task (`run_counting.py`)

- **Tests:** H2/H3, via a controlled task where reasoning depth = number of
  ±1 operations (`n_ops`).
- **Command:** `uv run python -m scripts.run_counting`
- **Parameters:** `N_OPS = (2, 4, 8, 16, 24, 32)`; `N_SEEDS = 5`; task text
  built by `shapes/synthetic.py::make_counting_task`.
- **Current result:** all settle. Per-level `|winding|~n_ops` rho=+0.943
  (N=6, **significant**) and `steps~n_ops` rho=+0.928 (**significant**) —
  but this is a **confounded, uninterpretable** significant result: in this
  task, `n_ops` and prompt length (`seq_len`) are collinear by construction
  (every operation adds a fixed number of tokens), so a per-level
  correlation with `n_ops` is indistinguishable from a correlation with raw
  length. The per-row, length-**partial** correlation (`partial_spearman`,
  controlling for `seq_len`) is what actually tests the depth-specific
  claim: rho=+0.036, p=0.85 — genuinely null. **When citing this
  experiment, cite the partial number, not the raw per-level one**, or a
  careful reader will reasonably ask why a "significant" number is being
  called a null result.
- **Why this task alone can't answer H2:** exactly this confound is why
  E3 (§4.3) exists.
- **Behavioral caveat that applies to this whole task family:** see §4.6 —
  the model's actual counting accuracy collapses to 0% at length ≥ 8, so
  for the longer `n_ops` values in this sweep, you're measuring the
  geometry of a computation the model is *failing at*.

### 4.3 E3 — the "killer experiment": length-matched track vs. local (`run_dissociation.py`, `run_dissociation_multiinit.py`)

This is the project's own name for its most carefully designed synthetic
test, and it's worth understanding the design even if you never touch the
code: `shapes/synthetic.py::make_variants(n_ops, seed)` generates a pair of
prompts sharing an **identical body** ("Start at 0. Add 1. Subtract 1. …")
and differing only in the final question — `track` asks for the running
total (needs accumulation across all `n_ops` steps), `local` asks only for
the last instruction (doesn't need to accumulate anything). Because the
body is identical, **prompt length is held fixed between the two
conditions by construction** — any difference in trajectory geometry
between `track` and `local` at the same `n_ops` isolates *state-holding*
from *raw length*, which is exactly the confound E2 couldn't resolve.

There are three separate scripts/CSVs in this family — **do not conflate
them**, an error I made myself in an earlier pass through this project (see
`research_log.md` 2026-07-17 (5)):

- **`run_dissociation.py --seeds 5`** (`results/dissociation.csv`) and
  **`--seeds 15`** (`results/dissociation_15seed.csv`): `N_OPS = (4, 8, 16,
  24, 32, 48)` → only **N=6** depth levels, single init seed, 5 or 15
  task-seeds. Track `steps~n_ops`: rho=+0.812 (5-seed) / +0.714 (15-seed),
  **both n.s.** at N=6 (critical value 0.886). Neither version of this
  script clears significance on its own.
- **`run_dissociation_multiinit.py`** (`results/dissoc_multiinit.csv`):
  `N_OPS = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64)` → **N=10** levels ×
  `N_TASK_SEEDS = 6` × `N_INIT_SEEDS = 5`. This is the **only** one of the
  three that reaches the project's own significance bar: pooled track
  `steps~n_ops` rho=+0.842 (N=10, critical value 0.648, **significant**).
  But its own point is to stress-test that pooled number: per-init-seed rho
  ranges **0.17 to 0.87** — the effect is real in direction but unstable in
  magnitude across random initializations, and the effect size itself is
  small (~1 extra step). `local`'s pooled rho=−0.207 (n.s.), with sign
  flipping across init seeds (−0.41…+0.28).
- **Command to rerun any of these:** `uv run python -m scripts.run_dissociation
  --seeds 15` (or `5`) / `uv run python -m scripts.run_dissociation_multiinit`.
  To test a different `n_ops` grid or seed count, edit the constants at the
  top of the relevant file directly.
- **Caveat worth flagging for review:** the base `dissociation.csv`'s local
  condition shows `winding~n_ops` rho=−0.943 (N=6, **significant**) — never
  discussed in the README, and (like E5's maxtask result below) never
  checked against a length-partial. Given `local`'s prompt body is identical
  to `track`'s at the same `n_ops`, this specific number isn't
  length-confounded in the usual sense — but it *is* an unexplained
  significant result on the condition that's supposed to be the negative
  control, worth someone's attention before it's ignored.
- **If you want to strengthen this experiment:** the multi-init version is
  already the strongest version, and its own limitation (per-init
  instability) is a sample-size problem at the init-seed level, not the
  `n_ops`-level — raising `N_INIT_SEEDS` beyond 5 is the most direct lever
  if you want a tighter per-init-seed rho range, at roughly linear GPU cost.

### 4.4 E4 — force-loop / phase diagram (`run_forceloop.py`, `run_phase.py`)

- **Tests:** H1 — do loop/drift regimes appear at all, and under what
  compute-budget conditions?
- **Commands:** `uv run python -m scripts.run_forceloop` /
  `uv run python -m scripts.run_phase`
- **Parameters:** `run_forceloop.py`: `NUM_STEPS = (16, 24, 32, 64)`,
  `N_OPS = (8, 24, 48)`, `N_SEEDS = 8`. `run_phase.py`: `NUM_STEPS = (10,
  14, 16, 18, 20, 24)`, `N_OPS = (4, 8, 16, 24, 32, 48)`, `N_SEEDS = 6`
  (produces `figures/phase.png`, a heatmap of "unsettled fraction" over the
  (budget × length) grid).
- **Current result, quantified 2026-07-18:** at `num_steps=16`, the
  unsettled (loop+drift) fraction rises **1/8 (n_ops=8) → 7/8 (n_ops=24) →
  6/8 (n_ops=48)**; Fisher exact test on the 8-vs-24 contingency table gives
  **p=0.0101**. Overall at `num_steps=16`: 82 settle / 13 loop / 1 drift
  across the full sweep; 100% settle at `num_steps≥24`.
  **This is probably the project's single cleanest positive result** —
  fully reproducible (synthetic task, no missing loader anywhere in the
  chain), a real significance test behind it, independent of every problem
  afflicting the two-scale branch (§4.8) — and it is currently **absent
  from `README.md`'s own Results section entirely**. If you add one thing to
  the paper for free, add this. **Interpretation the team draws, worth
  taking at face value:** looping here is a symptom of an
  under-provisioned compute budget, not of reasoning depth per se — this is
  evidence *for* H1 (loop is a real, distinct regime, reachable by
  manipulating the compute budget) but is explicitly *not* a test of H2.
- **If you want to explore the boundary more finely:** the interesting
  region is `num_steps` between 16 and 24 — neither script currently
  samples 17–23; adding those values to `NUM_STEPS` would sharpen exactly
  where the settle/loop transition sits.

### 4.5 E5 — generalization beyond counting: switch (parity) and maxtask (running max)

- **Tests:** whether the state-holding signature (steps rising with length)
  generalizes beyond arithmetic counting.
- **Commands:** `uv run python -m scripts.run_switch` /
  `uv run python -m scripts.run_maxtask`
- **Parameters:** both use `N_OPS = (4, 8, 16, 24, 32, 48)`; `N_SEEDS = 10`
  (switch) / `8` (maxtask).
- **Switch (parity — light on/off, flip/wait sequence):** `steps~n_ops`
  rho=+0.783 (n.s. at N=6), `winding~n_ops` rho=−0.086 (n.s.). **Clean
  null on both metrics** — this one is exactly what the README says it is.
- **Maxtask (running maximum over a digit stream):** `steps~n_ops`
  rho=−0.771 (n.s., wrong sign for the state-holding hypothesis anyway) —
  matches the README's framing of a failed replication. **But** `winding~
  n_ops` rho=+0.943 (N=6, **significant**) — this was not previously
  surfaced anywhere. **Important gap:** unlike `run_counting.py`, neither
  `run_switch.py` nor `run_maxtask.py` computes a `partial_spearman(...,
  seq_len)` control, and `n_ops` is just as collinear with prompt length
  here as it is in the counting task — so this significant winding result
  is very likely the same length confound as E2's, but that's an inference,
  not something the code has actually checked. **If you pick up one thing
  from this document to go implement, this is a good candidate:** add the
  same length-partial check `run_counting.py` already has to both of these
  scripts (a ~3-line change, following the existing pattern) before citing
  or dismissing either result.

### 4.6 Behavioral ground truth: does the model actually solve these tasks? (`run_accuracy.py`)

- **Tests:** whether the counting task's geometry (E2/E3) is being measured
  on a computation the model can actually perform.
- **Command:** `uv run python -m scripts.run_accuracy`
- **Parameters:** `N_OPS = (2, 4, 8, 16, 24, 32, 48)`, `N_SEEDS = 8`,
  `num_steps=32` for the `generate_with_adaptive_compute` call that decodes
  the answer (via `traj_geom.eval.counting_correct`), `num_steps=64` for the
  paired trajectory extraction — note the budget mismatch with the
  accuracy check itself.
- **Current result:** accuracy by length: 38% (n=2), 25% (n=4), **0%** at
  n=8, 16, 32, 48, and 12% (1/8) at n=24 (noisy, small sample). **This is a load-
  bearing caveat for every counting-based geometry claim in the project**:
  at the lengths where the geometry experiments (E2, E3) look for a depth
  signal, the model is failing the task outright, so "the geometry doesn't
  correlate with depth" and "the model doesn't have a depth-dependent
  internal computation to correlate with" are hard to tell apart on this
  particular task.
- **If you want a cleaner behavioral/geometry pairing:** this argues for
  either (a) restricting the geometry analysis to `n_ops` ≤ 4 where accuracy
  is non-trivial, at the cost of losing most of the length range, or (b)
  finding/designing a synthetic task the model can actually solve at the
  lengths you want to study (see §9.1).

### 4.7 Persistent homology and objective convergence diagnostics (`run_homology.py`, `run_convergence.py`)

These two don't sweep new trajectories — they read the 15 already-banked
`.npy` files in `trajectories/` (listed in `trajectories/manifest.csv`) and
score them, so they run instantly, need `uv sync --extra tda` for the
homology one (ripser), and nothing for the convergence one (pure numpy).

- **Homology** (`uv run python scripts/run_homology.py`): max normalized H1
  persistence ≈0.003 at `num_steps=64` (the "settle" regime), ≈0.000 at
  `num_steps=16` (the "tight budget" regime). Expected to be near-zero *by
  construction* at N=15 single curves — a genuine topological loop feature
  needs either many more sampled points on the same orbit or a population
  of many trajectories treated as one point cloud (see §9.4 for how you'd
  do the latter). Not evidence against looping, just underpowered. Ripser
  will print a "more columns than rows, did you mean to transpose?" warning
  on every call — this is a benign false positive given these are short
  curves (16–64 points) in a 5280-dim space, not an actual orientation bug.
- **Convergence** (`uv run python scripts/run_convergence.py`): the
  `num_steps=64` subgroup (9 of the 15 trajectories) shrinks step size by
  ×52 by the end, has `conv_rate<0` for all 9, cosine of adjacent steps
  ≈−0.28 (oscillatory-but-convergent), drift-to-loop ratio ≈12. All 15
  banked trajectories (including the `num_steps=16` ones) actually have
  `conv_rate<0`, a stronger finding than what's quoted anywhere — the
  `num_steps=16` group just shrinks much less (×10, not ×52) and has a
  *positive* mean cosine (+0.258, non-oscillatory), consistent with those
  being the tighter-budget trajectories where some genuinely loop rather
  than settle.
- **To add more banked trajectories:** there's no script that writes to
  `trajectories/` currently visible on `main` — the 15 `.npy` files there
  were added in bulk in one commit (`exp 3`). If you want to bank new ones
  for a homology/convergence analysis at a different parameter combination,
  you'd extract with `extraction.hook.extract_trajectory`, save with
  `numpy.save`, and add a row to `manifest.csv` with the same columns as
  the existing rows (`file`, `num_steps`, `n_ops`, plus whatever else the
  two analysis scripts read).

### 4.8 The two-scale branch (unmerged): a different metric, different token positions

Lives on `feature/two-scale-latent-dynamics-paper` / PR `two-scale-real-fix`
— not on `main`. Uses Pappone et al.'s (arXiv:2509.23314) metrics instead
of winding: for consecutive step-deltas `Δ(k) := h(k+1) − h(k)`,
**acceleration** `a(k) := ‖Δ(k) − Δ(k−1)‖₂` and **orthogonality**
`cos∠(Δ(k), Δ(k−1))`. Verified faithful to the paper's own Eq. 1/4 and
cosine formula (`claims_ledger.md` C11) — one gap: the paper also defines a
*normalized* acceleration `â(k) = ‖Δ(k)−Δ(k−1)‖ / (‖Δ(k)‖+‖Δ(k−1)‖+ε)` (its
Eq. 5) that the repo doesn't implement; only the raw, unnormalized version.

**Scripts (run from a checkout of that branch, not `main`):**

- **`run_two_scale_depth.py`** — the "MAIN" result. `BANDS = ((2,3), (3,4),
  (4,5))` (adjacent-depth pairs), `N_LOAD = 200` (pool size before
  matching), `N_PER_CELL = 60` (kept per side after matching),
  `NUM_STEPS = 32`. Produces `results/band.csv`. Regresses `accel_content ~
  hi + seq_len + C(adj)` (OLS, `analysis/band.py::depth_coef`), with a
  stratified bootstrap CI (`bootstrap_ci`, resampled within band×hi cells).
  **Result:** content-token depth-coefficient (net of the linear `seq_len`
  control) = **+1.2974**, 95% CI [+1.05, +1.55], p<1e-4 — independently
  reproduced with statsmodels (a completely separate implementation),
  matching almost exactly. Answer-token version: ≈+0.04, CI crosses zero,
  null.
- **The methodological problem you need to know before trusting this
  number — confirmed at the source as of 2026-07-18, not just inferred.**
  The design's own premise — that `hi=0` (low depth) and `hi=1` (high depth)
  rows within a band are matched on prompt length — does not hold in the
  committed CSV: `seq_len` ranges are **completely disjoint** in every band
  (2-3: gap=3 tokens; 3-4: gap=5; 4-5: gap=11), depth/length stay correlated
  within-band, and it's worse than a pooled number suggests — within-band
  point-biserial r(hi,seq_len) is **−0.89 to −0.90** (near-total
  collinearity), condition number 6088, and the `seq_len` coefficient's sign
  flips depending on whether `hi` is in the model (a textbook
  suppressor-variable symptom). **The generating code has since been found**
  — `notebooks/02_two_scale.ipynb` (already committed on this same branch,
  commit `8efc31d`, cell 14 "CENTERPIECE: length-matched depth test"). It
  turns out `analysis/band.py`'s `_length_match()` (exact-key pairing) isn't
  what actually ran; the real design restricts each side to a *pool-level
  overlapping seq_len range* and independently samples up to 60 per side
  from within it — no per-example pairing at all. Its own printed output
  (`len_lo=171 len_hi=153`, `210 vs 189`, `253 vs 224` across the three
  bands) matches the disjoint ranges found in `band.csv` exactly, confirming
  this notebook run produced the committed data. **The notebook's own
  closing cell states its own success bar: "If depth-coef|len is clearly >0
  ... with len_lo~len_hi → acceleration tracks depth beyond length." By the
  design author's own numbers, it doesn't clear that bar.** This is no
  longer a provenance mystery — it's a directly observed property of the
  honest, non-fabricated generating procedure, falling short of its own
  stated criterion. A within-band permutation test still finds signal
  (p=0/2000) and a boundary-restricted comparison stays significant in all
  3 bands, so the effect isn't fake — but the exact point estimate shouldn't
  be quoted as clean. Sensitivity check: adding an `hi:seq_len` interaction
  drops the coefficient to +1.09, p=0.025. **Before this goes in the paper
  as the headline H2 result:** either rerun with a matcher that pairs
  individual examples within a tolerance (not just a pool-level range
  restriction), or downgrade the language from "length-matched" to
  "length-band-restricted," and lean on +1.09/p=0.025 as the more
  defensible number. (The `docs/proposed_patches/` reconstruction of the
  data loader for this branch is now confirmed to have guessed the wrong HF
  dataset — don't use it; the notebook has the real code.)
- **`run_two_scale_real.py`** — full depth range, no length-matching.
  `DEPTHS = (2,3,4,5)`, `N_PER_DEPTH = 30`, `NUM_STEPS = 32`, `N_BINS = 10`
  (for a per-position acceleration map). `accel_content` by depth: 18.00,
  18.42, 18.53, 18.89 — per-level rho=+1.00 (N=4) but depth and `seq_len`
  are collinear on the full range, so this isn't interpretable as a clean
  depth effect on its own; it's honestly deferred to the band test above by
  the branch's own docs.
- **`run_two_scale_counting.py`** — track/local scored with two-scale
  acceleration instead of steps-to-settle/winding. `N_OPS = (4,8,16,24,32,
  48)`, `N_SEEDS = 3`, `NUM_STEPS = 32`. **Result: no dissociation** —
  content acceleration rises with `n_ops` for *both* track and local
  (length-confounded, not state-holding-specific); this is a genuine
  negative result, honestly reported, and it means the two-scale metric
  family does **not** replicate `main`'s E3 dissociation finding on the
  same underlying task.
- **The fabrication, still physically in the repo:** `run_two_scale_
  analysis.py` is the original synthetic-placeholder script (kept
  intentionally, headed "SYNTHETIC METRIC CHECK ONLY", so the diff shows the
  placeholder→real correction) — `np.random.randn(steps, 5280) * decay`
  where `decay = exp(-t/(5+2·depth))`. Its output (`results/
  two_scale_analysis.csv`, `two_scale_placeholder.csv`,
  `figures/two_scale_analysis.png`, `figures/two_scale_results.png`) is
  what got presented as real Huginn output somewhere (`accel rho=0.997,
  orth=-0.5`) — **confirmed fabricated independently this session**, not
  just taking the team's word for it: recomputed the per-depth means
  myself (4.89/5.84/6.70/7.45, suspiciously smooth) and found
  `mean_orthogonality` pinned at −0.496 to −0.500 on *every single row
  regardless of depth* — exactly the signature you'd expect from random
  noise with a depth-independent decay shape, not a real signal. These
  files are **still sitting in the repo tree**, unmerged branch — nothing
  at the filesystem level stops someone opening the wrong figure by
  accident; only a prose warning in `docs/two_scale.md` stands between a
  reader and the fake numbers. Worth deleting or moving to an obviously-
  named `archive/fabricated/` path before this branch is ever merged or
  shown to anyone external.
- **Who is "David"?** `results/FINDINGS.txt` and the PR blame the
  fabricated numbers on someone named David, and a test comment credits him
  with the original per-position metric code — but no commit anywhere in
  this repo, on any branch, is authored by anyone named David (only
  Shtirmann/Alexander Shiyanov and jack). Not resolvable from the repo
  alone; ask the team directly.
- **This is live, not settled.** A fourth branch, `pr-real-data` (forked
  from `two-scale-real-fix`'s tip, `jack`, 2026-07-17, no open PR), adds
  `docs/ANALYSIS_SUMMARY.md`, which restates the band-test coefficient as
  "+1.297, p<0.0001" — a clean, final-looking result table — with **zero
  mention of the length-matching problem above**, plus a new, never-
  reviewed orthogonality result (content rho=+0.139 p=0.008, answer
  rho=−0.004 p=0.946). Nobody has told `jack` about the confound as of the
  last check.

### 4.9 The TRM toy baseline (unrelated codebase, top-level `src/`)

Not part of `Geometry-of-Reasoning-Trajectories/` at all — lives at the
repository root, in `src/train_tiny_recursive.py` +
`src/synthetic_tasks.py`. A tiny (64-dim) weight-tied recurrent MLP, trained
(not just run at inference) on synthetic counting/parity/FSA tasks with an
optional Jacobian-regularization term (`compute_jacobian_reg_loss`, a
finite-difference local-Lipschitz estimate) that acts as a crude local proxy
for the STARS-style contraction regularization — this is the project's own
small, controllable testbed for H3, separate from the Huginn work, and the
team has since moved on to Huginn directly rather than extending this one.

- **Run it:** `cd src && python train_tiny_recursive.py` — needs `torch`
  and `numpy` only (no `uv` project here; an isolated venv with those two
  packages is enough, CPU is fine, ~1 minute per `train_and_evaluate` call).
- **Key parameters, all arguments to `train_and_evaluate()`:** `task_name`
  (`"counting"` / `"parity"` / `"fsa"`), `hidden_dim` (default 64),
  `num_steps` (recurrent unrolls during training, default 16),
  `num_epochs`, `beta_regularization` (0 = unconstrained; >0, e.g. 5.0,
  enforces contraction — this is the H3 lever). Train/test lengths (15/45)
  and dataset sizes (4000/800) are hardcoded inside the function, not
  exposed as arguments — edit them directly if you want a different
  extrapolation gap.
- **Confirmed by actually running it this session:** unconstrained (β=0)
  gives 100% in-distribution / 0% extrapolation, reliably. Contracting
  (β=5.0) is **noisier than the presentation implies** — the script sets no
  random seed anywhere (neither `torch.manual_seed` nor a numpy seed), so
  repeated runs gave 100%/100%/90% in-distribution across 3 tries.
  Extrapolation is robustly 0% for both conditions, every run. **If you
  rerun this for a paper figure, add a seed first** (`torch.manual_seed(k)`
  + `np.random.seed(k)` at the top of `train_and_evaluate`, looped over a
  few `k` to report a mean±std instead of one noisy number).
- **New problem found this session, not previously documented anywhere:**
  the **FSA task looks broken as a task construction**, not just hard. In a
  clean run: 30% in-distribution accuracy (barely above the 33% random-
  guess floor for 3 classes — the model isn't learning the FSA transition
  rules at all) but **50.75% extrapolation accuracy — higher than
  in-distribution**, which is backwards for a task meant to get harder with
  length. Likely cause, not fully diagnosed: `FSATrackingDataset`'s 3-state
  automaton (rules in `synthetic_tasks.py`) probably has a stationary state
  distribution that concentrates as sequences get longer, making the final
  label more predictable from statistics alone at length 45 than at length
  15, independent of whether the model tracks state correctly. If this
  baseline is revisited for the paper, don't use the FSA numbers until this
  is fixed — a quick diagnostic would be printing the label distribution at
  both lengths and checking whether it's near-degenerate at length 45.

---

## 5. Architecture reference (condensed — full version in `infra.md`)

```
Model-bound (needs CUDA + Huginn weights):
  extraction/model.py    — load_huginn()
  extraction/hook.py     — extract_trajectory() / extract_trajectory_allpos() [PR]
  eval.py                — counting_correct() via generate_with_adaptive_compute

Pure numpy/scipy/sklearn, model-free (fully unit-tested on synthetic
ground-truth paths — circles, spirals, lines):
  metrics/dynamics.py     — step_norms, steps_to_settle, contraction_rate
  metrics/winding.py      — winding_number, winding_of (PCA-projects to 2D first)
  metrics/projection.py   — pca_to_2d
  metrics/convergence.py  — convergence_rate, consecutive_step_cosine,
                             drift_to_loop_ratio, path_independence
  metrics/homology.py     — h1_persistence (needs ripser, `tda` extra)
  metrics/two_scale.py    — [PR only] compute_acceleration, compute_step_orthogonality
  shapes/gate.py           — classify_shape (settle/loop/drift)
  shapes/synthetic.py      — make_counting_task, make_variants, make_switch_task, make_max_task
  analysis/correlate.py    — spearman, partial_spearman, spearman_by_level
  analysis/band.py         — [PR only] depth_coef, bootstrap_ci
  analysis/plots.py        — scatter_winding_vs_depth, plot_pca_trajectory
  types.py                 — Trajectory dataclass (designed as the shared
                              contract; in practice NO run_*.py script
                              actually constructs one — everything passes
                              bare numpy arrays. Only test_types.py uses it.
                              Know this before assuming trajectories flow
                              through the codebase as Trajectory objects
                              anywhere except that one test.)

Glue (imports across both of the above, talks to results/*.csv):
  scripts/_common.py  — cached(), load_model() (deferred torch import)
  scripts/run_*.py    — one script per experiment, described in §4
```

**Every `torch`/`transformers` import in the model-free layer is deferred**
(inside function bodies, never at module top) — this is why importing
`traj_geom.metrics.winding` or running the test suite never requires a GPU
or even having `torch` installed.

**The caching contract, spelled out because it has a sharp edge:** the
`results/*.csv` files ARE the dataset. `cached(name, compute_fn)` reads the
CSV if it exists and never calls `compute_fn` — there is no hash or version
check tying a cached CSV to the code that produced it. If you edit a metric
formula (say, change `winding_of`'s burn-in from 4 to 8), every already-
cached `*winding*` number is now silently stale until you delete the CSV
and rerun with GPU access. There's no automatic detection of this — it's on
you to remember which CSVs depend on which code paths.

**Team-role convention (OWNER tags):** every `main`-branch module opens with
an `OWNER: <role>` docstring header — `Extraction+Winding` (13 files: model
loading, hook, winding/projection/convergence/homology/dynamics),
`Data+Analysis` (12 files: correlate, plots, eval, and most of the
experiment scripts), `Shapes+Gate` (5 files: gate, synthetic tasks,
force-loop/phase experiments). **Important:** on `main`, every one of these
files was written by one person (Shtirmann) despite the three-way split —
checked directly against git blame. The tags describe an intended division
of labor, not a record of who actually has hands-on history with any given
part. If you're deciding where to start contributing, all three lanes are
equally unclaimed in practice.

**Testing:** only the model-free layer has tests (`tests/test_convergence.py`,
`test_correlate.py`, `test_gate_synthetic.py`, `test_homology.py` [skipped
without `ripser`], `test_types.py`, `test_winding_synthetic.py`,
`test_plots.py` [added this session]). `extraction/` is untested — can't be,
without a GPU and the actual weights — so its correctness guarantees live as
prose gotchas in `constants.py` and the module docstrings instead of
assertions. Read those before changing anything in `extraction/`.

---

## 6. Statistical conventions — read this before writing a new experiment

The project has a deliberate, consistent statistical policy, established in
`analysis/correlate.py`. Follow it for any new experiment or you'll produce
numbers that don't compare cleanly to the existing ones.

- **The canonical statistic is per-level Spearman, not per-row.** Given a
  long-form DataFrame with repeated measurements at each level of an
  independent variable (e.g. multiple seeds at each `n_ops`),
  `spearman_by_level(df, xcol, ycol)` first collapses to one mean per level,
  then rank-correlates across levels. This deliberately avoids the inflated
  N (and false significance) of correlating every individual row — treating
  5 seeds × 6 lengths as N=30 independent observations when they're really
  N=6 independent length-levels with 5 repeated (non-independent) measures
  each is **pseudoreplication**, and this project's whole statistical
  posture is built around not doing that.
- **The significance threshold table is hardcoded** (`_SPEARMAN_CRIT_P05`
  in `correlate.py`): critical |rho| for p<0.05 two-tailed at N=4 through
  N=10 (1.000, 1.000, 1.000, 0.886, 0.786, 0.738, 0.700, 0.648 — note N=4
  and N=5 require a **perfect** correlation to ever reach significance;
  this is exactly why E1's N=4 depth levels can never demonstrate
  significance no matter how strong the true effect is). **Bug, confirmed
  2026-07-18:** the N=4 entry is worse than "requires a perfect
  correlation" — exact permutation enumeration shows it's mathematically
  *unreachable*, period (smallest possible two-tailed p at N=4 is
  2/24≈0.083, never <0.05, even at |rho|=1.0). Should read "n/a" like N>10
  already does. Every other entry is correct, checked the same way by
  exact permutation enumeration at each N — including N=8 (0.738), which
  an earlier pass wrongly flagged as a second bug (claiming 0.7143 was
  correct instead; that's wrong, since 0.7143 gives exact two-tailed
  p=0.058, not significant, while 0.738 gives p=0.046, which is). N=4 is
  the table's only real defect, and it has not produced a false "sig" in
  this project's actual N=4 experiments yet, but fix before adding a new
  one at this N. If you design a new experiment, more levels (not more
  seeds per level) is what buys you statistical power on this project's own
  terms.
- **No multiple-comparisons correction is applied anywhere**, across
  roughly 15-20 significance tests reported over ~8-10 experiments (grep for
  bonferroni/holm/fdr across the codebase: zero hits). Doesn't threaten the
  band test (p=6.6e-23 survives any plausible correction trivially), but is
  directly relevant to results sitting right at their per-level threshold
  (E2's winding/steps at N=6, or maxtask's winding result, §4.5) — worth
  either stating explicitly in the paper's methods section that no
  correction was applied (exploratory-analysis framing) or applying one
  before finalizing which results get called significant.
- **Report both, but lead with per-level.** Every existing script prints
  both `[per-level]` and `[per-row]` numbers, and where relevant a
  `[per-row, partial|L]` number too, precisely so a reader can see the
  difference and not be misled by whichever one happens to look better.
  Follow this pattern in new scripts (`fmt_by_level` gives you a one-line
  string with the significance verdict already computed).
- **Confounds get a `partial_spearman` control, when the confound is real.**
  `run_pararule.py` and `run_counting.py` do this (controlling for
  `seq_len`); `run_switch.py`/`run_maxtask.py` currently don't, and it shows
  (§4.5) — their significant winding results are uninterpretable without it.
  **When you design a new length-varying synthetic task, ask up front
  whether your independent variable is collinear with prompt length, and if
  so, either build a length-matched control (like E3's `track`/`local`
  design) or compute the partial correlation from the start.**
- **The two-scale branch's band-regression + bootstrap approach**
  (`analysis/band.py`) is a heavier-weight alternative worth knowing about
  if per-level Spearman isn't expressive enough for what you're testing —
  it fits `y ~ hi + seq_len [+ band dummies]` by OLS and gets a CI via a
  *stratified* bootstrap (resampling within each band×hi cell, preserving
  the design rather than naively resampling rows). This is the right tool
  when you have a genuine covariate to control for continuously (like
  `seq_len`) rather than a small number of discrete levels — but see §4.8
  for why its current instantiation has a data problem, not a code problem.
- **Winding sign is arbitrary.** PCA doesn't fix a sign convention, so every
  script that uses `winding_of` takes `abs()` of it before comparing across
  examples — don't compare raw signed winding across different trajectories
  unless you've separately confirmed the PCA sign is consistent (it isn't,
  in general).

---

## 7. Known issues, ranked by how much they matter

1. **`pr-real-data` is actively propagating the unresolved band-test problem
   right now** (§3, §4.8) — live, not historical; tell `jack` before this
   goes further.
2. **Two-scale band test's length-matching doesn't hold — confirmed at the
   source** (§4.8): the design's own generating notebook shows it fails its
   own stated bar, using its own printed numbers. Affects the project's
   single strongest quantitative result. Resolve (rerun with real matching,
   or downgrade the language and use the +1.09 interaction-corrected number)
   before the paper leans on the raw +1.30.
3. **The project's cleanest positive result is missing from README**
   (§4.4): forceloop's compute-budget × task-difficulty loop transition,
   Fisher p=0.01, fully reproducible, no missing-loader blockers. Add it —
   near-zero cost, meaningfully strengthens the paper regardless of what
   happens with #1/#2.
4. **`traj_geom.data.loaders` missing from git, both branches, ever**
   (`.gitignore`'s bare `data/` rule swallowing `src/traj_geom/data/` since
   the very first scaffold commit) — blocks independent regeneration of
   every PARARULE-touching result on both branches. Fixed locally 2026-07-17
   (§3), not yet committed anywhere; the PR-branch reconstruction was
   confirmed *wrong* on 2026-07-18 (real code found in
   `notebooks/02_two_scale.ipynb` — don't apply the one in
   `docs/proposed_patches/`).
5. **No paper-drafting artifact exists yet**, 15 days out, despite the
   target proceedings template already being in hand
   (`files/Smiles26Barannikov Proposal.pdf`) — plausibly the biggest
   schedule risk in the project at this point, independent of which number
   ends up cited.
6. **Single point of failure on GPU/Kaggle access** (§3): every commit
   touching real Huginn output traces to one identity, who is reportedly
   unavailable for roughly half the remaining time before the deadline.
   Front-load anything GPU-dependent.
7. **Fabricated placeholder artifacts still physically in the repo**
   (§4.8) — low effort to clean up (delete or rename 4 files), currently
   undone.
8. **`run_maxtask.py`/`run_switch.py` missing the length-partial check**
   `run_counting.py` already has (§4.5) — makes one significant result
   (maxtask winding) uninterpretable. Small, well-scoped fix.
9. **`analysis/correlate.py`'s significance-threshold table has a real bug
   at N=4** (§6) — small, mechanical fix, prevents a possible future false
   "sig" claim.
10. **FSA synthetic task's failure mode was misdiagnosed once already** —
    corrected 2026-07-18 (§4.9): it's not label-distribution concentration,
    it's the model underperforming a trivial baseline in-distribution. Don't
    use its numbers regardless until the training dynamics are actually
    investigated.
11. **README's reproduce command was wrong** — fixed locally 2026-07-17.
12. **PR #1 has sat untouched for 3+ days**, zero comments/reviews/CI, but
    is confirmed technically trivial to merge (zero conflicts) — a process
    gap, not a code one.
13. **"David"'s identity is unresolved** (§4.8) — ask the team.
14. **The Contraction Bottleneck Theorem (H3) has never been empirically
    connected to Huginn's actual ρ(∂ₕR)** — the proof is correct (A1), and
    a toy-model version of this measurement has now been done (§9.5,
    `src/h3_results/`) with a real, if unreviewed, finding: the mechanism
    doesn't show up on a recurrent-over-depth toy model (matching Huginn's
    information-flow pattern) but does on a recurrent-over-time
    counterpart. Still untested on the real model.
15. **No team member other than Shtirmann and jack leaves any trace in the
    repo's git history**, despite this being described as a 3-person
    project — worth confirming who the third person is and what they're
    responsible for.

---

## 8. Environment setup (verified, as of this session)

```bash
cd Geometry-of-Reasoning-Trajectories

# Base + tests/lint (this is what README.md now says, after this session's fix):
uv sync --extra dev
uv run pytest -q          # -> 16 passed (was 11 before this session's additions)
uv run ruff check .

# Add persistent homology support:
uv sync --extra tda       # ripser, persim

# Add model support (needs a real GPU to actually load Huginn-3.5B; on a
# Mac, extraction/model.py and hook.py hardcode "cuda" — you'd need to
# patch the device string or use a remote GPU box, e.g. Kaggle, which is
# how the team has always done the heavy runs — see infra.md sec 2/6):
uv sync --extra model
```

No CI, no Docker, nothing in `.github/` — GPU compute has always been
manual, on Kaggle (confirmed from `notebooks/01_mvp.ipynb`'s own title and
its `CACHE = "/kaggle/working"`). The `results/*.csv` caching layer (§5) is
what makes that tolerable for anyone without Kaggle access.

For the unrelated top-level TRM baseline (§4.9): no project file at all,
just needs `torch` + `numpy` in any Python 3.11 environment.

---

## 9. Cookbook: writing new experiments and code

### 9.1 Add a new synthetic reasoning task

Write `make_<name>_task(n_ops, seed) -> dict` in `shapes/synthetic.py`,
returning either `{"prompt", "answer"}` (single-condition) or `{"track",
"local"}` (if you want a length-matched dissociation control for free —
follow `make_variants`'s pattern: identical body, differing only in the
final question). Given §4.6's finding, consider up front whether the model
can actually solve your task at the lengths you care about — if not,
you're back in the same bind as the counting task.

### 9.2 Add a new metric

New file in `metrics/<name>.py`, pure function on a `[T, hidden_dim]` array,
no `torch`/model imports (keep it in the model-free layer), OWNER header,
and a test against a synthetic array with known ground truth — a circle for
anything winding/loop-related, a damping spiral for anything convergence-
related (both already exist inline in `tests/test_winding_synthetic.py` and
`tests/test_convergence.py`; worth factoring into a shared
`tests/synthetic_paths.py` if you add a third consumer).

### 9.3 Add a new experiment script

Copy the shape of `run_switch.py` or `run_maxtask.py` (~55 lines, one task,
one metric pass) rather than the larger multi-stage ones unless you
specifically need length-matching or multi-seed robustness (in which case,
`run_dissociation_multiinit.py` and `analysis/band.py` are the two existing
patterns for that). Use `cached()`, report with `fmt_by_level`, and — per
§6 — add a `partial_spearman` control against `seq_len` if your independent
variable could plausibly be collinear with prompt length.

### 9.4 Extending to all token positions, or a proper population-level TDA test

The PR branch's `extract_trajectory_allpos` (§2) already gives you
`[num_steps, seq_len, hidden_dim]` instead of one token's path — if you want
to test H2 with winding number the way `main` does it, but on content
tokens instead of the answer token, this is the function to build on; it
isn't used by any `main`-branch script currently. For a proper population-
level persistent-homology test of H1 (addressing §4.7's power problem),
you'd want to build a point cloud from *many* trajectories at once (e.g. all
tokens at a given depth, stacked) rather than treating one 16-64-point curve
as the whole population — `metrics/homology.py::h1_persistence` already
accepts any `[T, dim]` point cloud, so the metric code doesn't need to
change, just what you feed it.

### 9.5 Computing the model's actual spectral radius ρ(∂ₕR) — H3's missing empirical link

**Not implemented on real Huginn** (no GPU access when this was attempted),
but **implemented, self-tested, and run on the toy model**, 2026-07-18, at
top-level `src/` (outside this package): `spectral.py` does power-iteration
on the Jacobian-vector product of a recurrent step, alternating one JVP
(`torch.autograd.functional.jvp`) and one VJP through the same function
(equivalent to power-iterating on `J^T J`, converging to the operator norm
— the right quantity for a Lipschitz bound, not just the spectral radius,
which can understate it for a non-normal Jacobian). Validated against
known-answer linear/nonlinear maps in `test_spectral.py` (5/5 pass,
matching `torch.linalg.svd` to <1% relative error) before being trusted on
anything else.

Two sweeps were run with it: `h3_validation.py` (the recurrent-over-depth
toy model, same architecture pattern as Huginn) and
`sequential_h3_validation.py` (a recurrent-over-time counterpart built to
match the theorem's own §5 example). Result, in short (full detail
`src/h3_results/FINDINGS.md`): the depth-recurrent model keeps count
information ~perfectly decodable (linear-probe R²≥0.996) across the entire
β/ρ sweep, including deep contraction (ρ≈0.25); the sequential-RNN
counterpart shows a sharp phase transition — decodability craters from
R²≈0.83 to ≈0.00 the moment ρ drops meaningfully below 1, and stays there.
**This is not yet a test on real Huginn** — porting `spectral.py` to a real
Huginn hidden state (calling into the actual `core_block` step function
instead of the toy model's `step()`) is the natural next step if/when GPU
access is available; it would upgrade H3 from "theory + indirect
consistency evidence" to "theory + one direct measurement," which is a
categorically stronger paper regardless of which way the result comes out.

### 9.6 If you're touching `extraction/`

Read `constants.py`'s full docstring first — every gotcha there was learned
by debugging a crash, not written speculatively. In the order they'll most
likely bite you: wrong `transformers` version (must be 4.50–4.53, pin is
4.53.3), passing `num_steps` as a tensor instead of a plain int, a stale
hook left registered by a previous crashed run doubling captures (always
`_forward_hooks.clear()` + `try/finally`), calling `.generate()` when you
needed `.forward()` for trajectory work.

---

## 10. Literature grounding (verified this session, not just cited)

All checked via live web search against the actual arXiv listings, not
recalled from training data (several are dated after this assistant's
knowledge cutoff and had to be looked up fresh):

| Citation | Real? | What it's used for here |
|---|---|---|
| Geiping et al., arXiv:2502.05171 (NeurIPS 2025) | ✅ | The Huginn architecture itself |
| Blayney et al., arXiv:2604.11791 | ✅, Appendix C confirmed structurally (exact 0.02%/2.81% digits not re-extracted — fetch truncated) | The "underpowered, not disconfirmed" argument for H1 |
| Pappone, Crisostomi, Rodolà, arXiv:2509.23314 (NeurIPS 2025) | ✅, equations verified to match `metrics/two_scale.py` exactly (one gap: normalized variant Eq. 5 unimplemented) | The two-scale acceleration/orthogonality metrics |
| Merrill, Petty, Sabharwal, arXiv:2404.08819 | ✅ | "SSMs/transformers can't solve state-tracking" — theoretical backing for why counting might genuinely fail |
| Grazzi et al., arXiv:2411.12537 | ✅ | Parity/state-tracking impossibility results |
| Yang et al., arXiv:2605.26733 (STARS) | ✅ | Jacobian Spectral Radius regularization — the concrete H3-adjacent architecture cited in the motivation |
| Movahedi et al., arXiv:2606.18206 (FPRM) | ✅ | Fixed-point halting via pre-norm + iteration-wise residual scaling |
| Tulchinskii et al., arXiv:2502.17017 | ✅ — co-authored by **Serguei Barannikov**, this project's curator | Query-Key alignment probe, listed as future work |
| Jiang et al., arXiv:2606.23590 | ✅ (spot-check) | TDA on LLM hidden states, broader lit review |
| Tuci et al., arXiv:2604.19740 | ✅ (spot-check) | Edge-of-stability/fractal-attractor generalization theory, broader lit review |

10/10 spot-checked citations are real papers accurately characterized — no
evidence of citation fabrication anywhere. The only fabrication found in
this project is the isolated two-scale placeholder-data incident (§4.8),
which is a data problem, not a literature problem.

---

## 11. If you only read one section: what to do next

For **reanalysis/review**: start at §7 (ranked issues), cross-reference the
relevant §4 subsection and the corresponding `claims_ledger.md` row for
full derivation detail.

For **rerunning with different parameters**: every §4 subsection lists the
exact tunable constants and the command; nothing needs a CLI you don't
already see in the file — just edit the constants at the top of the script.

For **writing new experiments/code**: §9 (cookbook) + §6 (statistical
conventions, don't skip this one) + §5 (where your new file belongs in the
dependency layering).

The single highest-leverage next action, if you want one (updated
2026-07-18): **tell `jack` about the band-test length-matching problem
before `pr-real-data` propagates it further** — that's §7 items 1-2, and
it's live, not a someday-fix. Second: **add the forceloop result (§7 item
3, §4.4) to README's Results section** — near-zero cost, meaningfully
strengthens the paper regardless of how #1 resolves. Third: **start the
actual paper draft** (§7 item 5) — 15 days out with nothing written is now
plausibly a bigger risk than any single contested number. See
`docs/START_HERE.md` for the full ranked recommendation list and a map of
all five docs in this folder.
