> **FROZEN 2026-08-05.** Archived under CLAUDE.md §6 (three live documents only).
> Not maintained. Superseded by `docs/UNDERSTANDING.md`, `claims_ledger.md`, `directions.md`.
> Statements here may be contradicted by later work — several assert things since disproven
> (e.g. "H1 was never tested": it was, and the test itself was invalid — see D56, D65).

# The `pr-real-data` branch, end to end

A companion to `narrative.md`, written from the perspective of
`origin/pr-real-data` — the newest branch in this repository and the one
whose numbers are most likely to end up in the paper. Same promises as its
sibling: concepts introduced before use, abstractions grounded in real
prompts and real arrays, every number re-derived rather than quoted.

Read this one if you're working on the two-scale line of work or deciding
what to cite. Read `narrative.md` if you want the whole project including
`main`'s experiments — this document assumes its sections 1–4 (the
question, the model, what a measurement is, the three hypotheses) rather
than repeating them, and says so wherever it leans on them.

**Everything below is as of 2026-07-19**, verified against the branch
itself in an isolated worktree.

**Contents.**
1. [What this branch is, and the trap in calling it "latest"](#1-what-this-branch-is-and-the-trap-in-calling-it-latest)
2. [What lives here that lives nowhere else](#2-what-lives-here-that-lives-nowhere-else)
3. [The two-scale metrics](#3-the-two-scale-metrics)
4. [The placeholder-data incident](#4-the-placeholder-data-incident)
5. [Headline result 1: the acceleration band test](#5-headline-result-1-the-acceleration-band-test)
6. [Headline result 2: orthogonality — and a sign reversal](#6-headline-result-2-orthogonality-and-a-sign-reversal)
7. [What this branch inherits from main, including a broken control](#7-what-this-branch-inherits-from-main-including-a-broken-control)
8. [The unused mechanism: Huginn picks its own depth](#8-the-unused-mechanism-huginn-picks-its-own-depth)
9. [Scorecard and what to cite](#9-scorecard-and-what-to-cite)
10. [What to do next](#10-what-to-do-next)

---

## 1. What this branch is, and the trap in calling it "latest"

`pr-real-data` is the most recent commit in the repository — `900b8b4`,
jack, 2026-07-17 — and it is tempting to treat it as the project's current
state. **It is not a superset of `main`, and treating it as one would
silently lose work.**

The branch chain, verified with `git merge-base`:

```
main
  8b5e418  2026-07-12 15:01  "exp 3"          <- pr-real-data's ancestry forks HERE
  fe00b8b  2026-07-12 22:50  "retesting, exp 4"   ] main-only work,
  14e77eb  2026-07-12 22:53  "Revise README"      ] never on this branch
    │
    ▼ (forked from 8b5e418)
feature/two-scale-latent-dynamics-paper
  155c6d0  2026-07-12 20:02  jack        two-scale analysis + placeholder data (§4)
    │
    ▼  PR #1 opened against the above, 2026-07-15 — still open, 0 reviews
two-scale-real-fix
  2ac075e  2026-07-15 16:42  Shtirmann   replaces placeholder with real Huginn output
  8efc31d  2026-07-15 16:46  Shtirmann   adds the generating notebook
    │
    ▼ (forked again; no PR opened this time)
pr-real-data
  900b8b4  2026-07-17 18:12  jack        docs/ANALYSIS_SUMMARY.md
```

Because the fork happened at `exp 3`, before `main`'s last two commits,
these files exist on `main` and **are absent here**:

| missing here | consequence |
|---|---|
| `scripts/run_accuracy.py` + `src/traj_geom/eval.py` | the behavioral check — can the model even solve the task — cannot be run on this branch |
| `scripts/run_maxtask.py` | E5's running-maximum experiment cannot be rerun |
| `scripts/run_dissociation_multiinit.py` | the multi-init robustness run cannot be rerun |
| `scripts/run_convergence.py` + `metrics/convergence.py` + `results/convergence.csv` | convergence diagnostics gone entirely |
| `tests/test_correlate.py`, `tests/test_convergence.py` | two test files gone |

Most of the *data* survives (`maxtask.csv`, `forceloop.csv`,
`counting_accuracy.csv` and the rest are all still committed here) — it's
the *analysis code* that's missing. So results computed on `main` can still
be read here, but not regenerated.

**Practical consequence:** merging `pr-real-data` into `main` naively would
delete four experiments' worth of code. Any reconciliation has to be a
merge that keeps both sides, not a fast-forward or an overwrite.

## 2. What lives here that lives nowhere else

Against that, this branch is where all the two-scale work lives, and it is
substantial — 8,434 inserted lines versus `main`:

- `src/traj_geom/metrics/two_scale.py` — the acceleration/orthogonality
  metrics (§3)
- `src/traj_geom/analysis/band.py` — the length-matched band regression
- `scripts/run_two_scale_depth.py` — the headline analysis
- `scripts/run_two_scale_real.py`, `run_two_scale_counting.py`
- `results/band.csv` — 360 rows, the data behind both headline results
- `results/pararule_depth.csv`, `counting_two_scale.csv`, plus `.npy`
  per-position maps
- `notebooks/02_two_scale.ipynb` — **the real generating code**, with its
  own execution output saved inline (§5)
- `docs/two_scale.md` and `docs/ANALYSIS_SUMMARY.md`
- `tests/test_two_scale.py`
- an extended `extraction/hook.py` adding `extract_trajectory_allpos`

That last one matters more than its size suggests. `main`'s extraction
records **one** token position per prompt — the answer token. This branch
adds the ability to record **every** position at once, which is what makes
the content-token results below possible at all. It is the single most
consequential capability difference between the two lines of work.

## 3. The two-scale metrics

(`narrative.md` §3 covers what a trajectory is and how it's captured; this
assumes that.)

The metrics come from a different paper than `main`'s winding number:
Pappone et al., "Two-Scale Latent Dynamics for Recurrent-Depth
Transformers," NeurIPS 2025, arXiv:2509.23314 — confirmed a real paper,
its equations checked against this branch's implementation, which matches
faithfully (one gap: the paper's *normalized* acceleration variant isn't
implemented, only the raw one).

Write `Δ(k)` for the step vector from unroll `k` to unroll `k+1`. Two
quantities are built from it:

- **acceleration**, `a(k) = ‖Δ(k) − Δ(k−1)‖` — how much the step vector
  changes between consecutive unrolls. Large while the path is still
  turning or changing speed; small once it moves smoothly.
- **orthogonality**, `cos∠(Δ(k), Δ(k−1))` — whether consecutive steps
  point the same way (near +1), turn a corner (near 0), or reverse
  (near −1).

Each is computed at every token position and then split two ways, in
`row_metrics()` in the generating notebook:

- **answer-token** value — the last position, the one `main` studies.
- **content-token** value — the mean across every *other* position, i.e.
  the prompt body.

That content/answer split is the axis this whole branch turns on.

## 4. The placeholder-data incident

This branch's ancestry includes a data-integrity problem, caught and
corrected by the team. It needs stating precisely, because "a number turned
out to be fake" and "this line of work is compromised" are different
claims and only the first is true.

`scripts/run_two_scale_analysis.py` (jack, 2026-07-12, still present on
this branch) generates trajectories from `np.random.randn` noise with a
hand-injected decay shaped by depth, rather than from the model — and does
so **transparently**, in its own comments and printed output ("Generating
synthetic trajectories…"). Nothing about the script is disguised; it reads
like ordinary practice — validate a metrics pipeline on synthetic data
with known structure before spending GPU time.

Its output was later presented as if it were a real Huginn finding (deck
figures `accel rho=0.997`, `orth=-0.5`). The person who did that is named
consistently across this branch's own artifacts — `results/FINDINGS.txt`,
`docs/two_scale.md`, `tests/test_two_scale.py`,
`notebooks/02_two_scale.ipynb`, all written by Shtirmann — as "David":
*"David deck numbers (accel rho=0.997, orth=-0.5) = np.random.randn +
hand-coded 5+2\*depth. No model."*

**No commit anywhere in this repository is authored by anyone named
David.** Verified across all branches: exactly four author identities
exist, and David is not one of them. His name appears only inside file
contents. So whatever he did happened outside this repository entirely and
was never pushed — which reframes the incident: it looks far more like
someone's own uncommitted work being presented before the committed
pipeline behind it was verified, than like anyone editing shared code to
disguise a fake result.

That the presented numbers came from noise is independently confirmed here,
not taken on anyone's word: recomputing per-depth statistics from the
still-committed `results/two_scale_analysis.csv` shows
`mean_orthogonality` pinned between `-0.496` and `-0.500` on *every row
regardless of depth* — the signature of noise decaying at a
depth-independent rate.

Both placeholder files (`two_scale_analysis.csv`,
`two_scale_placeholder.csv`) are **still present on this branch**, kept
deliberately so the diff shows the correction. That is a live risk: nothing
at the filesystem level stops someone opening the wrong one.

## 5. Headline result 1: the acceleration band test

The honest fix replaced the noise script with a real pipeline against
actual Huginn output. Its central result, and this branch's strongest
claim.

**The design.** Compare *adjacent* PARARULE-Plus depths — 2-vs-3, 3-vs-4,
4-vs-5 — rather than the full range, restricting each comparison to
overlapping prompt-length ranges, then regress content-token acceleration
on a depth indicator while controlling for prompt length.

**The result**, from `results/band.csv` (360 rows, 60 per side per band),
independently re-fit here with `statsmodels` rather than the branch's own
hand-rolled regression:

```
accel_content ~ hi + seq_len + C(adj)      (robust standard errors)

  hi coefficient = +1.2974      p = 6.6e-23      95% CI [1.04, 1.56]
```

matching `docs/ANALYSIS_SUMMARY.md`'s published `+1.297, p<0.0001`
essentially exactly. Per-band: +1.565 (2-3), +1.266 (3-4), +1.184 (4-5) —
consistent in sign and magnitude across all three.

The answer-token control is null in the same fit: `-0.0440, p=0.88`. So on
this branch's own metric, the effect is on content tokens and not on the
answer token — consistent with `main`'s answer-token nulls on winding.

**The problem with it.** The design's premise is that the compared groups
are matched on prompt length. They are not. Within every band the two
groups' length ranges are **completely disjoint** — gaps of 3, 5 and 11
tokens between nearest edges — and within-band collinearity is severe
(point-biserial correlation −0.89 to −0.90 between length and the depth
indicator).

`docs/ANALYSIS_SUMMARY.md` describes the bands as "each with `seq_len
~200`." The actual group means, verified from `band.csv`:

| band | low-depth group | high-depth group | gap |
|---|---|---|---|
| 2-3 | 171.2 | 153.3 | 17.9 |
| 3-4 | 210.2 | 189.0 | 21.2 |
| 4-5 | 253.3 | 224.0 | 29.3 |

Not "~200" — 153 to 253 across the design, with a systematic ~20-token gap
*within* each supposedly matched pair. (Note the direction: the
higher-depth group is consistently *shorter*.)

This isn't an inference. The real generating code was located on this
branch's parent, in `notebooks/02_two_scale.ipynb` — already committed,
simply never opened. Its actual design doesn't pair examples at all: it
restricts each side to a pool-level overlapping range and samples up to 60
per side from within it. Its own printed diagnostics show
`len_lo=171 len_hi=153`, reproducing exactly the gap above. And its closing
cell states the author's own trust condition:

> *"If depth-coef|len is clearly >0 … with `len_lo~len_hi` → acceleration
> tracks depth beyond length."*

By that stated bar, using its own printed numbers, it isn't met.

**What survives.** A within-band permutation test (2,000 reshuffles per
band) still finds signal, and a comparison restricted to examples nearest
each band's length boundary stays significant in all three bands. Adding a
depth×length interaction drops the coefficient to `+1.09` and weakens it to
`p=0.025`. So: a real effect very likely exists; the specific
`+1.30, p<1e-4` figure should not be quoted as a clean length-controlled
result without this caveat.

## 6. Headline result 2: orthogonality — and a sign reversal

`docs/ANALYSIS_SUMMARY.md` reports a second result:

> CONTENT orthogonality: mean 0.1376, correlation with adj `+0.139`,
> p=0.008 → *"CONTENT orthogonality shows weak increase with depth."*

Those numbers reproduce exactly from `band.csv` (mean 0.1376, Spearman
+0.1393, p=0.0081). **The arithmetic is right. The conclusion is
backwards.**

Here's the problem. That `+0.139` correlates orthogonality against the
*band index* — 2-3 vs 3-4 vs 4-5 — which is a comparison **across** bands,
not the within-band matched comparison the acceleration result uses. And
band index is almost perfectly confounded with length:

```
spearman(band index, seq_len)      = +0.936
spearman(seq_len, orth_content)    = +0.195,  p = 0.0002
spearman(band index, orth_content) = +0.139   <- the reported number
```

Length predicts orthogonality *better* than the reported depth
relationship does. So the result has no length control at all, on a
variable that is 94% correlated with length.

Apply this branch's **own** design — the same `hi + seq_len + C(adj)` fit
that produced the acceleration headline — and the sign flips:

| metric | `hi` coefficient | p |
|---|---|---|
| `accel_content` | **+1.2974** | 6.6e-23 |
| `orth_content` | **−0.1586** | 1.2e-18 |
| `accel_answer` | −0.0440 | 0.88 |
| `orth_answer` | −0.1003 | 5.1e-10 |

And it is not a fragile artifact of pooling — it reverses in **every band
independently**, before and after controlling for length:

| band | low-depth mean | high-depth mean | raw difference | length-controlled `hi` coef | p |
|---|---|---|---|---|---|
| 2-3 | 0.1665 | 0.0598 | **−0.107** | −0.138 | 0.0007 |
| 3-4 | 0.2015 | 0.0957 | **−0.106** | −0.219 | <0.0001 |
| 4-5 | 0.2195 | 0.0825 | **−0.137** | −0.186 | <0.0001 |

So within every matched depth pair, the *deeper* problems show **lower**
content-token orthogonality — consecutive steps turning corners *less*,
not more. The published "+0.139, weak increase" comes entirely from
comparing across bands, where length rather than depth drives the number.

This is a textbook aggregation reversal, and the branch's own methodology
already contains the fix — it just wasn't applied to this second result.
The acceleration analysis controls for length within band; the
orthogonality analysis doesn't. That methodological inconsistency, inside
a single 54-line summary document, is what produces the wrong sign.

**Not a fabrication and not sloppiness in computing** — every number jack
published is arithmetically correct. It's an inference drawn from the wrong
comparison. But `docs/ANALYSIS_SUMMARY.md` currently states the opposite of
what this branch's own data supports, and this branch is the most likely
source for the paper's numbers.

**This finding is new as of 2026-07-19 and nobody on the team has been
told.** It should be raised before the orthogonality claim propagates.

## 7. What this branch inherits from main, including a broken control

`counting.csv`, `switch.csv`, `dissociation*.csv` and the rest are all
present here, unchanged from the fork point, along with
`analysis/correlate.py`. That means this branch inherits a defect
documented in `narrative.md` §7b, which is worth restating because two of
those CSVs are on this branch and could be cited from it.

In every synthetic task, prompt length is a **strict deterministic
function** of the difficulty variable `n_ops` — exactly one `seq_len` value
per level, zero variation across seeds:

| task | distinct `seq_len` per `n_ops` level | rank-corr(`n_ops`, `seq_len`) |
|---|---|---|
| counting | 1 | **1.0** |
| switch | 1 | **1.0** |
| maxtask | 1 | **1.0** |

After ranking, `n_ops` and `seq_len` are *literally the same array* (max
absolute difference exactly 0.0). So `partial_spearman`, the length
control, residualizes a variable against itself, leaving only
floating-point noise (4e-15 to 3e-14 depending on task) and then
correlates the metric against that noise.

The demonstration is simple: shuffle the row order — statistically a no-op,
identical data, identical ranks — and the reported correlation wanders and
**changes sign**: +0.036, −0.025, +0.050, −0.031, +0.063. A legitimate
statistic is invariant to row order.

So E2's `rho=+0.036, p=0.85` — cited as evidence that the counting signal
is "eaten by prompt length" — is not a clean null. It's an artifact of
float-operation ordering. The *qualitative* conclusion survives and is in
fact stronger (depth and length are perfectly confounded by construction,
hence inseparable) but the statistic offered as evidence never ran.

**Importantly, this does NOT affect this branch's headline results.**
PARARULE-Plus prompts are natural language, so their length genuinely
varies within a depth level (`spearman(depth, seq_len) = 0.816`, not 1.0;
8–10 distinct lengths per depth). The band test's regression-based length
control operates on real variation and is computable. The two-scale
results have real problems — §5 and §6 — but this is not one of them.

## 8. The unused mechanism: Huginn picks its own depth

Something no branch of this project uses, and which bears directly on the
central question.

Every experiment here sets the recurrence depth **manually** — a fixed grid
(`num_steps=32` for this branch's extraction, 64 on `main`) — and then asks
whether trajectory geometry correlates with problem difficulty. The
implicit assumption is that the model runs a fixed budget and we observe
what it does inside it.

But **Huginn ships with its own adaptive-depth mechanism**. Its
`generate_with_adaptive_compute` accepts a `criterion` argument selecting
among six built-in exit rules:

```
entropy-diff | latent-diff | cosine | kl / minp-kl | argmax-stability | none
```

Two of those — `latent-diff` and `cosine` — are *geometric*, measuring
essentially what this project measures by hand: has the latent state
stopped changing, and are consecutive steps still turning. The model's
authors already built a "has this settled?" detector.

**No branch of this project ever passes a `criterion`.** Verified with a
repo-wide search across `main`, `two-scale-real-fix` and `pr-real-data`,
covering both scripts and notebooks: every call site omits it. The default
is `criterion="none"`, which the model's own source comments describe as
*"adaptive compute is off by default, turn on by choosing an exit
criterion."*

So the one function in the codebase named for adaptive compute — and
described in `main`'s `eval.py` docstring as "Huginn's adaptive-compute
generation" — runs with adaptivity **disabled**, at a fixed step count.

That matters for interpreting everything above. The project's framing
assumes we must *infer* reasoning depth from geometry. But the model
exposes its own answer: run it with `criterion="latent-diff"` (or
`cosine`, or `kl`) and ask directly whether it exits later on harder
problems. That is a more direct test of the same underlying question, it
uses the model's native mechanism rather than a reimplementation, and
**it has never been run.**

It's also cheap relative to what's already been spent: the tasks and
prompts all exist, and the exit step per token comes back from the
model itself. Given that H2 is currently null on the answer token across
two independently implemented metric families, "ask the model when it
thinks it's done" is arguably the strongest remaining experiment available
to this project.

## 9. Scorecard and what to cite

Every number here appears above; nothing is newly introduced.

| Claim | Source | Verified? | Safe to cite? |
|---|---|---|---|
| Content-token acceleration rises with depth, `+1.297`, p<1e-4 | `ANALYSIS_SUMMARY.md`, `band.csv` | Reproduced exactly (`+1.2974`, p=6.6e-23) | **With the §5 caveat attached** — the "length-matched" premise fails by its own author's stated bar; interaction-corrected value is `+1.09`, p=0.025 |
| Answer-token acceleration null, `−0.044`, p=0.86 | same | Reproduced exactly | Yes — a clean, well-behaved control |
| Content-token orthogonality rises with depth, `+0.139`, p=0.008 | `ANALYSIS_SUMMARY.md` | Arithmetic reproduces; **conclusion reverses under the branch's own design** | **No — do not cite.** Within-band, length-controlled, it is `−0.159` (p=1e-18), negative in all three bands independently |
| The placeholder deck numbers were noise, not model output | `FINDINGS.txt` | Independently reconfirmed (§4) | Yes, and worth stating plainly |
| Counting's length control shows a null | inherited `counting.csv` | **Degenerate** (§7) | No — the control cannot run on this data |

**If the paper cites one number from this branch, cite the acceleration
band test with its caveat.** It is real, independently reproduced, and
survives permutation and boundary-restricted checks — it just is not the
clean length-controlled estimate its headline presentation implies.

## 10. What to do next

Ordered by urgency, given the 2026-08-02 deadline and that no paper draft
exists yet (no `.tex`, no outline, verified):

1. **Tell jack the orthogonality result reverses** (§6). This is the most
   time-sensitive item here: it's a published claim, on the newest branch,
   stating the opposite of what the data shows under the branch's own
   method, and nobody knows.
2. **Decide the merge strategy before merging anything** (§1). A naive
   merge of `pr-real-data` into `main` drops four experiments' worth of
   code. PR #1 is technically mergeable with zero conflicts and has sat
   unreviewed since 2026-07-15 — it is the correct thing to merge first,
   since `pr-real-data` builds on it.
3. **Restate the band test honestly** (§5) — either rerun with a matcher
   that pairs examples within a length tolerance, or downgrade the
   language from "length-matched" to "length-band-restricted, not
   achieving `len_lo≈len_hi` in the realized sample," and lead with the
   interaction-corrected `+1.09, p=0.025`.
4. **Run the adaptive-depth experiment** (§8). Cheapest genuinely new
   result available; needs no new task design, and directly addresses the
   question the project exists to answer.
5. **Delete or clearly quarantine the placeholder CSVs** (§4). They are
   still sitting in `results/` on this branch next to the real ones.

---

Companion documents: `narrative.md` is the full-project version, centred on
`main`'s experiments (E1–E5, homology, convergence) and containing the
background this document assumes — what a trajectory is, how it's
extracted, what the three hypotheses say. `claims_ledger.md` is the
row-by-row evidence trail for every number in both. `START_HERE.md` is the
short version with a ranked action list.
