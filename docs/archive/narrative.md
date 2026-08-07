> **FROZEN 2026-08-05.** Archived under CLAUDE.md §6 (three live documents only).
> Not maintained. Superseded by `docs/UNDERSTANDING.md`, `claims_ledger.md`, `directions.md`.
> Statements here may be contradicted by later work — several assert things since disproven
> (e.g. "H1 was never tested": it was, and the test itself was invalid — see D56, D65).

# The project, end to end

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


> **STALE AS OF 2026-07-24.** This file was last substantively updated
> 2026-07-22 and predates: the coda-skip fix's second round, D11's
> multivariate/prefix-confound refinement, D12's two corrections (bare-token
> bug, then the top-k re-score), D13-D15 (BH-FDR, the first real loops ever
> observed, the first clean H2 test), and a full literature-verification
> pass (Tulchinskii/Yang/Blayney/Geiping/Lu et al., all read from primary
> sources). Read `project_plan.md` first for current status;
> `claims_ledger.md` for any specific number. See
> `docs/verifiability_and_accountability.md` for how this banner came to
> exist. Content below is otherwise unchanged — not rewritten, just flagged.

A single linear read: what's being studied, why, how it's measured, what
each experiment actually found, and where that leaves the project's three
hypotheses. Written to be read once, start to finish, by someone who wants
to genuinely understand this project rather than get a status update.

Where the project measures something, you'll see an actual prompt, an
actual array, an actual number — not just the name of a metric.

Every number is the final, corrected one as of 2026-07-19, cross-checked
against `claims_ledger.md` (the row-by-row evidence trail — cite that
document, not this one, if you need receipts for a specific figure).
Several claims below are marked as checked or reproduced "this session" —
shorthand for the dedicated fact-checking pass that produced these
corrections. Where earlier notes got something wrong and it was later
caught, this document states only the corrected version; the history of
how a wrong number got fixed lives in `research_log.md`.

One scoping note: a separate, unrelated body of work — a toy
recurrent-model experiment built to test the theory in isolation, living
outside this git repository entirely — is deliberately excluded here.

**Contents.**
1. [The question, in plain terms](#1-the-question-in-plain-terms)
2. [The model: Huginn-3.5B](#2-the-model-huginn-35b)
3. [What one measurement actually looks like](#3-what-one-measurement-actually-looks-like)
4. [The three hypotheses](#4-the-three-hypotheses)
5. [How the codebase is organized](#5-how-the-codebase-is-organized)
6. [Where the compute actually happens](#6-where-the-compute-actually-happens)
7. [How "significant" gets decided — and a design flaw in the controls](#7-how-significant-gets-decided-and-a-design-flaw-in-the-controls)
8. [The main-branch experiments (E1–E5)](#8-the-main-branch-experiments-e1e5)
9. [The two-scale thread](#9-the-two-scale-thread)
10. [Branches, people, and how the work is actually distributed](#10-branches-people-and-how-the-work-is-actually-distributed)
11. [Defects found in the project's own infrastructure](#11-defects-found-in-the-projects-own-infrastructure)
12. [Where each hypothesis actually stands](#12-where-each-hypothesis-actually-stands)
13. [What's genuinely open](#13-whats-genuinely-open)
14. [How to actually run this](#14-how-to-actually-run-this)

Sections 1–7 build vocabulary; 8–11 are the evidence; 12–13 are the
synthesis; 14 is operational. Section 12's scorecard table is the fastest
way back in if you're returning to check one fact.

---

## 1. The question, in plain terms

Large language models can be made to "think longer" at inference time in
two structurally different ways.

The familiar one is **chain-of-thought**: the model writes out more text
tokens, and each token it writes becomes part of what it reads next. This
is expensive (every extra token costs a full forward pass through every
layer) and brittle (an early mistake compounds through everything after).

The less familiar way, and the one this project studies, is **recurrent
depth**. Instead of writing more tokens, the model runs a single shared
block of its own weights over and over on its internal state, before
producing any output token at all:

    h_{t+1} = R_θ(h_t; e)

`e` is a fixed encoding of the prompt, computed once. `h_t` is the model's
internal state after `t` iterations ("unrolls") of the shared block `R_θ`.
The number of unrolls `T` is a dial you turn at inference time — more
unrolls means more computation spent on the same problem, without writing
a single extra word.

Because `h_t` is a point in a high-dimensional space, the sequence
`h_0 → h_1 → … → h_T` traces out a **path** through that space. This
project's question:

> **Does the geometric shape of that path tell you anything about how much,
> or what kind of, reasoning the model is doing?**

If it did, that would be valuable: a way to read "how hard is this model
working" directly off its internals, without asking it, and a principled
reason to care about how the recurrent block is trained.

Section 4 turns that question into three testable hypotheses. But those
hypotheses are about the *shape* of a path, so it's worth seeing an actual
path first — that's section 3, right after a brief look at the model.

## 2. The model: Huginn-3.5B

The model is `tomg-group-umd/huginn-0125` (Geiping et al., "Scaling up
Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach,"
NeurIPS 2025, arXiv:2502.05171 — confirmed a real paper, correctly
characterized, via live web search). 3.5 billion parameters, a custom
architecture (`RavenForCausalLM`) loaded through HuggingFace with
`trust_remote_code=True`, pinned to an exact model revision and a narrow
`transformers` version window — outside which the custom code breaks in
several documented ways (exact pins in section 14).

**Its recurrent hidden state has 5280 dimensions.** So the "path" from
section 1 is a sequence of points in 5280-dimensional space. That number
matters for intuition later: it's a very large space, and a path of 64
points in it is a very sparse object.

Nothing in this project trains or fine-tunes the model. Every experiment
is a plain forward pass — `model(input_ids=ids, num_steps=N)` — never
`model.generate()`, which doesn't expose the intermediate per-unroll
states at all, only the final output token.

## 3. What one measurement actually looks like

This section is deliberately concrete. Everything after it will refer back
to these objects.

**Step 1 — a prompt.** Here is a real one, generated by this project's own
code (`make_max_task(n_ops=4, seed=0)` — the task this document will later
call **maxtask**):

```
Numbers: 7 7 1 5. Largest so far? A:
```

To answer, the model has to hold a running maximum across four numbers.
Tokenized, this prompt is **15 tokens** long — a quantity the code calls
`seq_len`, and which turns out to matter enormously in section 7b.

That "four" is the other quantity to name now: **`n_ops` is this project's
difficulty dial** — how many things the task makes the model track. Here
it's four numbers; in the counting task it'll be four arithmetic
operations. (The name is a slight misnomer for maxtask, where the items
are numbers rather than operations, but it's the same dial throughout.)
Nearly every experiment below works by turning `n_ops` up and watching
what happens to the shape of the path.

**Step 2 — run it, and watch.** The prompt goes through the model with
`num_steps=64`, meaning the shared recurrent block runs 64 times. A PyTorch
**forward hook** — a callback attached to `model.transformer.core_block[-1]`,
the last sub-module of the recurrent block — fires once per unroll and
records the internal state at that instant.

One thing the hook does *not* do: keep everything. The recurrent block
operates on all 15 token positions every unroll, but this hook records
just **one** — the last token of the prompt, where the answer will appear.
That's the choice almost every experiment in section 8 makes. Recording
every position instead is possible, and section 9's entire thread is what
happens when you do; it turns out to matter.

Three things about this hook matter, and each is documented verbatim in the
codebase's `constants.py` because each one broke something once:

- `h_0` is **randomly initialized** every call. That's Huginn's own design
  (see H3 in section 4), not an accident, and `torch.manual_seed(seed)`
  fixes it for reproducibility. Because it's random, the first few points
  are outliers relative to the rest of the path, so metrics discard them
  ("burn-in") before analyzing shape.
- `num_steps` must be a plain Python `int`, not a tensor — calling `len()`
  on a 0-dimensional tensor crashes the forward pass.
- A hook left registered by a previous crashed run silently *doubles* every
  subsequent capture, so the code always clears `_forward_hooks` and wraps
  the hook lifecycle in `try/finally`.

**Step 3 — what comes out.** A plain numpy array. This project banks a few
of these as files, so here is a real committed one — from a different
prompt than Step 1's, since no maxtask array happens to be banked (this is
`trajectories/local_n48_ns64_init0.npy`, the `local` condition of the E3
experiment you'll meet in section 8, at `n_ops=48`). The anatomy is
identical regardless of which prompt produced it:

```
shape: (64, 5280)    dtype: float32
```

64 rows (one per unroll), 5280 columns (the state's dimensions). That array
*is* the trajectory. Everything downstream is arithmetic on it.

**Step 4 — measure its shape.** The most basic thing you can ask is how far
the state moves at each step: the distance between consecutive rows. For
that real trajectory (shown from step 1, before the burn-in trim mentioned
above):

```
first 8 steps:  70.3, 55.8, 45.6, 37.6, 31.1, 24.9, 20.5, 18.2
last 4 steps:    1.05, 1.02, 1.00, 1.07
```

The steps shrink by a factor of about 65 from start to end. The path is
**settling** — moving less and less, converging toward a fixed point. That
is one of the three shapes hypothesis H1 will name in the next section.

Two quantities computed from arrays like this drive almost every result in
this document:

- **`steps_settle`** — how many unrolls until the step size first drops
  below a tenth of its maximum (a tenth is the code's default). For the
  trajectory above the largest step is 70.3, so the threshold is 7.0, and
  the path first falls below it at **unroll 13** — even though it's still
  creeping along at about 1.0 per step at unroll 64.

  That contrast is worth holding onto, because section 8 leans on it: the
  metric asks when the path stopped moving *fast*, not when it stopped
  moving. Loosely, "how long did the model keep changing its mind
  sharply."

- **`winding`** — an attempt to measure whether the path *curls*. It
  exists because of a specific prediction the next section will make: that
  paths which curl should curl *more* when the problem is harder. The raw
  path lives in 5280 dimensions, where "how many times did it go around"
  isn't directly meaningful, so the code projects the path down onto a 2-D
  plane (the plane of its own two directions of greatest variance, found by
  PCA, fit separately for each trajectory) and then adds up the signed
  angle the projected point sweeps around the path's own center, step by
  step. One full loop ≈ winding of 1.

  Two caveats worth carrying: the 2-D plane is fit per trajectory, so its
  orientation and the sign of the result are arbitrary from one trajectory
  to the next (which is why the code always takes `abs(winding)`); and this
  is a *proxy* for curling, not proof of a genuine topological loop. The
  project also computes real topological loop detection — persistent
  homology — separately, in section 8.

**Step 5 — label the shape.** Separately from those two numbers, a
classifier assigns each trajectory one of H1's three labels, by a rule
simple enough to state in full: if the **last** step is smaller than a
tenth of the largest step, call it **settle** (the path has converged —
which is how the trajectory above is labelled). Otherwise, if any two
points more than three unrolls apart come back within a quarter of the
path's overall extent, call it **loop** (it returned near somewhere it had
already been). Otherwise, **drift**. This rule is what produces E4's
headline result in section 8.

That's the whole measurement pipeline: prompt → forward pass with a hook →
`[64, 5280]` array → a couple of numbers describing its shape. An
"experiment" in this project means doing that a few dozen times across
prompts that vary in some controlled way, then correlating the shape
numbers against what was varied.

## 4. The three hypotheses

Now that "a path" and "its shape" are concrete, the hypotheses can be
stated precisely.

- **H1 — a few shapes.** Every token's latent path falls into one of a
  small number of distinguishable regimes: it can **settle** (converge
  toward a fixed point, steps shrinking toward zero — exactly what the
  example trajectory in section 3 does), it can **loop** (return near
  itself periodically, a limit cycle), or it can **drift** (keep moving
  outward without settling). These should be tellable apart by cheap
  measurements: how fast steps shrink, whether the path revisits itself,
  and its topological structure.

- **H2 — loops track depth.** When a path *does* loop, its winding number
  should grow with how much reasoning the problem actually requires. More
  logical steps needed → more turns in the latent path.

- **H3 — the Contraction Bottleneck Theorem.** This is the project's own
  proved result (`files/contraction_proof.md`; the algebra was checked
  independently this session and is correct — a standard Banach
  fixed-point argument). Suppose the recurrent map `R_θ` is a **strict
  contraction**: there's a constant `c<1` such that applying `R_θ` to any
  two states always brings them closer together by at least a factor `c`,

      ‖R_θ(x;e) − R_θ(y;e)‖ ≤ c‖x−y‖   for all x, y

  Then two things follow with mathematical certainty. **First**, the path
  converges to a single fixed point regardless of where it started — which
  is, notably, a *documented design goal* of Huginn itself: its authors
  deliberately randomize `h_0` so different starting points converge to the
  same steady state, a property they call "path independence." **Second**,
  the system's memory of where it started decays as `c^t`. So if you wanted
  to use the state to hold a running count, two different counts become
  indistinguishable — under floating-point precision `ε` — after a bounded
  number of steps:

      T_max ≤ log(D/ε) / log(1/c)

  where `D` is the diameter of the state space. Plug in `c=0.9`, `ε≈1e-6`,
  `D≈10` and you get `T_max ≈ 150` steps. Push a model to always settle
  (a real thing that training regularizers penalizing the map's Jacobian
  spectral radius do, to stabilize training) and, the theorem says, it
  becomes structurally unable to hold a running count past that bound —
  such tasks would be forced to loop or drift instead of settling.

  **One caveat runs through this entire project: nobody has ever measured
  Huginn's actual contraction constant.** The theorem is proven; whether
  real Huginn satisfies its premise (`c<1`) has never been tested. Every
  empirical result below is at best *consistent with* H3, never a direct
  test of it.

## 5. How the codebase is organized

The repository is a Python package, `traj_geom`, in three layers, split
strictly by whether a module needs the model loaded:

1. **Model-bound.** `extraction/model.py` (loads Huginn),
   `extraction/hook.py` (the hook from section 3, returning
   `[num_steps, hidden_dim]` for one token or
   `[num_steps, seq_len, hidden_dim]` for all positions), and `eval.py`
   (decodes the model's actual generated answer to check whether it got a
   task *right* — the only place generation-style decoding happens).
   Nothing else in the package imports `torch` or `transformers` at all.

2. **Pure numpy/scipy, model-free.** Everything that turns a
   `[T, hidden_dim]` array into a number: `metrics/` (step sizes, winding,
   persistent homology, convergence diagnostics), `shapes/` (the
   settle/loop/drift classifier and the synthetic-task generators),
   `analysis/` (statistical tests). Because this layer never touches the
   model, it's testable without a GPU against synthetic shapes with known
   answers — a perfect circle should wind exactly once, a straight line
   should have no loop, a geometrically shrinking sequence should register
   as settled. Roughly half this layer's public functions have no such
   test, including the step-size/settling function whose own docstring
   calls it the `"effective compute" signal that turned out to carry the
   MVP result` (MVP = minimum viable product) — a real, cheap-to-close gap.

3. **Glue.** `scripts/run_*.py`, one per experiment, each wrapping a
   `compute()` function behind a caching helper, `scripts/_common.py::cached()`.
   This helper is the single most consequential piece of infrastructure
   here: if `results/<name>.csv` already exists, it's read directly and
   `compute()` never runs — so the `torch` imports inside `compute()`
   (always written lazily, inside the function body) never fire either.
   **In practice, almost every experiment script can be rerun with zero
   GPU**, re-deriving the printed statistics from already-extracted data.

   The tradeoff: nothing ties a cached CSV to the code version that
   produced it. Edit a metric's formula after a result is cached and the
   stale numbers are silently reused until someone deletes the file.

A fourth piece exists but matters less than its own documentation claims:
`types.py` defines a `Trajectory` dataclass, described in its docstring as
"the ONE fully implemented module... every teammate codes against
`Trajectory`." In practice no `run_*.py` script anywhere constructs or
consumes one — every real script passes bare numpy arrays. The dataclass
works and has a passing test; it simply isn't how the codebase evolved.
One more documentation-vs-practice gap, alongside section 11's.

## 6. Where the compute actually happens

There is no CI, no Docker, and no cloud configuration checked into this
repository, on any branch, ever. GPU work has always been manual, on
**Kaggle** — confirmed from execution metadata embedded inside the
project's own notebooks, which carry real start/end timestamps and a
Kaggle-kernel Python version.

`notebooks/01_mvp.ipynb`, which produced essentially every main-branch
result in section 8, records **just over four hours of continuous Kaggle
runtime** on 2026-07-12, with most of its cells carrying saved output
inline. `notebooks/02_two_scale.ipynb` similarly records ~1h40m on
2026-07-13. (Exact timestamps and cell counts are in `claims_ledger.md`.)

**These notebooks — not the `scripts/run_*.py` files — are the literal
code that produced the numbers in `results/*.csv`.** The scripts are a
later, cleaner re-implementation of the same logic. The established
workflow, stated in every module's own docstring convention (`STATUS:
implemented (from notebooks/...)`), is: prototype on Kaggle, then port the
working logic into the package once proven.

That distinction matters for reading this document. No experiment here was
re-run against the real model during this document's fact-checking — there
has never been GPU access to do so. Every numeric claim below is one of
two kinds: a **direct recomputation from an already-committed CSV**
(rerunning `scripts/run_X.py` against its cache, which needs no GPU and
was done repeatedly), or a **reading of a notebook's own embedded
execution trace**. Neither is the same as independently re-extracting
fresh data from the model, and this document doesn't claim otherwise.

## 7. How "significant" gets decided — and a design flaw in the controls

Before any specific result means anything, two conventions have to be
stated. The second one is where this session found a real problem.

### 7a. Per-level correlation, and the N=4 bug

Almost every experiment has one independent variable at a handful of
discrete levels — say `n_ops ∈ {4, 8, 16, 24, 32, 48}`, six levels — with
several repeated seeds at each level. The **canonical statistic**,
`spearman_by_level()`, first collapses each level to its mean across seeds,
then rank-correlates across only the levels.

That deliberately avoids **pseudoreplication**: treating 10 seeds × 6
levels as 60 independent data points inflates the apparent sample size and
hence the apparent significance, when the seeds within a level aren't
independent evidence about the *level effect* — only about noise around it.
So `N` in every result below means *number of levels*, typically 4 to 10,
not number of trajectories.

The resulting correlation is compared against a hardcoded table of critical
values (`_SPEARMAN_CRIT_P05`) — the minimum `|rho|` needed for p<0.05 at
each `N`. Small `N` gives very little room: at `N=5` the table demands a
*perfect* correlation of 1.000.

**And at `N=4` the table had a real bug.** With only 4 levels there are
4! = 24 possible orderings; exactly 2 of them give a perfect correlation.
So the smallest achievable two-tailed p-value at `N=4` — even with a
flawless, perfect result — is 2/24 ≈ 0.083. Never below 0.05, for any
dataset that could ever exist. The old entry `4: 1.000` implied a bar that
would still prove nothing if cleared. It has been removed (a fix on a local
branch, section 10), so `N=4` now reports `crit=n/a`, the same path the
table already used for `N>10`. Every other entry was verified correct by
the same exact-permutation method, including `N=5`'s 1.000 (p = 2/120 ≈
0.017 — genuinely reachable) and `N=8`'s 0.738.

One more honest gap: across roughly 15–20 significance tests over ~10
experiments, **no correction for multiple comparisons is applied anywhere**
— no Bonferroni, no false-discovery-rate adjustment. That doesn't threaten
the most extreme results (section 9's p≈1e-23 survives any correction
trivially), but it matters for the several results sitting near threshold.

### 7b. The length confound — and why the control for it doesn't work

Here is the problem every synthetic experiment in this project faces, and
the thing this document's earlier versions got wrong.

Take the counting task. Its prompts look like this (real output of
`make_counting_task(n_ops=4, seed=0)`):

```
Start at 0. Subtract 1. Subtract 1. Add 1. Subtract 1. Final total? A:
```

You want to vary *difficulty* — more operations to track — so you increase
`n_ops`. But a prompt with 24 operations isn't just harder, it's also
**longer**. So if a shape metric correlates with `n_ops`, there are two
explanations, and they need separating:

1. the model is genuinely doing more reasoning (interesting), or
2. the prompt just has more tokens in it, which shifts these metrics for
   reasons having nothing to do with reasoning (boring).

The standard fix is a **partial correlation**: measure each prompt's actual
token count (`seq_len`), then ask "what's left of the metric↔`n_ops`
relationship once `seq_len` is held fixed?" The project implements exactly
this (`partial_spearman`), and E2's headline conclusion rests on it.

**That control does not work here, and this session established why.**

The technique needs `seq_len` to vary at least somewhat *independently* of
`n_ops` — you need some prompts that are unusually long or short for their
difficulty, so the two can be told apart.

This project has three synthetic tasks, and it's worth seeing all three
together, since the problem is identical across them. Counting is above;
the other two (both revisited in section 8's E5) are:

```
switch   (n_ops=4):  Light is off. Flip. Flip. Wait. Flip. Is the light on? A:
maxtask  (n_ops=4):  Numbers: 7 7 1 5. Largest so far? A:
```

`switch` is a parity task — track whether the light ends up on. `maxtask`
is the running-maximum task from section 3. Checked directly against the
committed data for all three, the independent variation the control needs
is exactly zero:

| task | `n_ops` levels | distinct `seq_len` values per level | rank-correlation(`n_ops`, `seq_len`) |
|---|---|---|---|
| counting | 6 | 1 | **1.0** |
| switch | 6 | 1 | **1.0** |
| maxtask | 6 | 1 | **1.0** |

Every single prompt at a given `n_ops` has *identical* token length. Not
similar — identical. Counting at `n_ops=4` is always 22 tokens; switch at
`n_ops=8` is always 28; maxtask at `n_ops=4` is always 15. The reason is
visible in the prompts themselves: the random variation between seeds
changes *which* words appear, never *how many tokens* they take. "Add 1."
and "Subtract 1." tokenize to the same length; so do "Flip." and "Wait.";
every digit 1–9 is one token.

So `n_ops` and `seq_len` aren't merely correlated — after ranking, they are
*literally the same array* (verified: maximum absolute difference between
`rankdata(n_ops)` and `rankdata(seq_len)` is exactly 0.0). Residualizing
one against the other leaves nothing but floating-point noise — measured at
4e-15 for counting, 2e-14 for switch, 3e-14 for maxtask — and the "partial
correlation" then correlates the metric against that noise.

**What this means concretely.** E2's reported partial correlation of
`rho=+0.036, p=0.85` is not evidence of a null effect. It's not evidence of
anything — it's a number produced by correlating against numerical dust.
The same applies to the equivalent controls on switch and maxtask (run for
the first time this session, section 8's E5).

**What survives.** The *qualitative* conclusion this statistic was offered
to support — that in these tasks you cannot separate reasoning depth from
prompt length — is not just still true, it's true more strongly than the
statistic suggested. They're perfectly confounded by construction. The
correct evidence for that claim is the perfect collinearity itself, not a
partial correlation that can't run. What has to be retired is the idea that
this control *tested* anything and came back clean.

**The right fix is a design change, not a statistic.** Either vary prompt
length independently of difficulty (padding, or phrasings of deliberately
different token lengths), or compare two conditions at matched length —
which is exactly what E3 was built to do, and why E3 is the best-designed
experiment in the project.

## 8. The main-branch experiments (E1–E5)

All of the following ran against real Huginn-3.5B on Kaggle (section 6) and
record the path of the **answer token only** — the last token of the
prompt, where the model's answer will appear. (A prompt is many tokens
long, and the recurrent block operates on all of them simultaneously every
unroll, but these experiments record just that one position. Section 9's
thread is what happens when you record all of them.) All committed CSVs
were directly reprocessed this session; numbers below are the corrected,
final ones.

**E1 — PARARULE-Plus: does winding track logical proof depth?**
PARARULE-Plus is a public logical-reasoning dataset whose examples carry a
labeled proof-tree depth: how many rules must be chained to reach the
answer. A real depth-2 example from the committed data, truncated:

```
The dinosaur is sleepy. The dinosaur is dull. ... The rabbit is furry.
Nice animals are lovely. If something is dull then it likes the squirrel.
If something likes the squirrel then it is slow. ...
Question: ... Answer:
```

and a real depth-5 one:

```
Bob is strong. Bob is high. Bob is big. Dave is little. ... Alan is poor.
Strong people are wealthy. If someone is little and short then they are
thin. If someone is sad and imperfect then they are rough. If someone is
wealthy and nice then they are kind. If someone is thin then they are
small. ...  Question: ... Answer:
```

This is the one dataset in the project where **prompt length is not a rigid
function of difficulty** — real sentences vary in length, so at depth 2
prompts run 118–176 tokens and at depth 5 they run 222–308, ranges that
genuinely overlap with neighbouring depths. That's why section 7b's
degeneracy problem does *not* apply here: the length control can actually
run on this data. Ten examples per depth, 64 unrolls each.

Result: **all 40 trajectories settle** — none loop, none drift. Winding,
measured anyway on the settling paths, has no relationship to depth:
`rho=+0.20` at `N=4`, not significant. Steps-to-settle gives `rho=+0.80`,
also not significant at `N=4` — and per section 7a, nothing at `N=4` could
have been.

**E2 — synthetic counting.** The task from section 7b: "Start at 0. Add 1.
Subtract 1. … Final total?" with `n_ops` from 2 to 32 standing in for
depth. Raw per-level correlations are large and formally significant:
`|winding|~n_ops` `rho=+0.943`, `steps~n_ops` `rho=+0.928` (`N=6`).

But as section 7b established, `n_ops` and prompt length are perfectly
collinear here, so those raw numbers cannot distinguish reasoning from
length — and the partial correlation offered as the control (`rho=+0.036`)
is degenerate, not a clean null. **The honest reading: this experiment
cannot separate the two explanations at all.** The project's own README
says the counting signal is "eaten by prompt length"; the accurate version
is that the design makes the two inseparable, so nothing was measured
either way.

One further caveat applies to E2 and E3 both, detailed below: the model is
failing the counting task outright at these lengths.

**E3 — the dissociation control, the project's best-designed experiment.**
This is the correct response to E2's problem. Two prompt variants share an
*identical body* and differ only in the final question (real output of
`make_variants(n_ops=4, seed=0)`):

```
track:  Start at 0. Subtract 1. Subtract 1. Add 1. Subtract 1. Final total? A:
local:  Start at 0. Subtract 1. Subtract 1. Add 1. Subtract 1. What was the last instruction? A:
```

`track` requires holding a running count across every operation. `local`
requires no accumulation at all — just read the last instruction. Same
body, same operations, same difficulty of *reading*; different demand for
*state-holding*. Any geometric difference between them isolates
state-holding from raw length by design, rather than by a post-hoc
statistical correction that (section 7b) can't work.

One precision correction, found this session: the two conditions are **not
exactly length-matched**, contrary to what this document previously
claimed. The differing question wording makes `local` consistently 3 tokens
longer (22 vs 25, 34 vs 37, 58 vs 61, … at every level). That constant
offset is far weaker than the confound E3 was built to eliminate, and
crucially, length rises *identically* with `n_ops` in both conditions — so
comparing the two conditions' *trends* remains valid. Worth stating
accurately rather than claiming a matching the data doesn't show.

Three runs exist and must not be conflated (an earlier version of the
project's own notes conflated them; here is the corrected picture):

- **Base run**, six levels, single random initialization: track
  `steps~n_ops` `rho=+0.812` (5 task-seeds) and `rho=+0.714` (15 seeds) —
  both **not significant** at `N=6`.
- **Multi-init robustness run**, ten levels × six task-seeds × **five
  different random initializations**: pooled track `steps~n_ops`
  `rho=+0.842` at `N=10`, which does clear significance. But this run's own
  purpose is to show that number is unstable — computed separately within
  each initialization, the same correlation ranges from `0.17` to `0.87`.
- The `local` negative control is null throughout, with one exception worth
  flagging rather than hiding: in the base 5-seed run, `local`'s winding
  correlates significantly (`rho=-0.943`, `N=6`), unchecked against any
  control. Treated here as an open question, not evidence.

**E4 — force-loop: the project's cleanest positive result.** If H1's three
regimes are real, then starving the compute budget (small `num_steps`) on a
task that genuinely needs state should push paths out of settling and into
looping or drifting.

It does. At `num_steps=16`, the fraction of trajectories failing to settle
rises from **1 in 8** at `n_ops=8` to **7 in 8** at `n_ops=24` — Fisher's
exact test on that 2×2 table, **`p=0.0101`** (a formal test that had not
been run before this session). It then dips slightly to 6 of 8 at
`n_ops=48`, a small non-monotonicity the test doesn't depend on and that
nobody has investigated.

The regime split is stark. **At `num_steps=16`: 10 settle, 13 loop, 1
drift, out of 24** — a majority of trajectories not settling. At
`num_steps ≥ 24`, across all three larger budgets: **100% settle**, 72 out
of 72. (Summed over all four budgets that's 82 settle, 13 loop, 1 drift
out of 96 — but the whole result lives in the starved-budget quarter of
that sweep.) So both "loop" and "drift" are real, observed regimes in this
project's own data — just only when the compute budget is too small for
the path to converge.

This result is fully reproducible with no external dependency — synthetic
task, no dataset fetch, no missing code — has a real significance test, and
is **absent from the project's own README Results section**, which lists E4
as an experiment but never reports the number.

**E5 — does state-holding generalize beyond arithmetic?** Two more
synthetic tasks needing running state:

```
switch  (n_ops=4):  Light is off. Flip. Flip. Wait. Flip. Is the light on? A:
maxtask (n_ops=4):  Numbers: 7 7 1 5. Largest so far? A:
```

`switch` (parity) is a clean **null** on both metrics: `steps~n_ops
rho=+0.783`, `winding~n_ops rho=-0.086`, neither significant at `N=6`.

`maxtask` is more complicated than the project's earlier notes recorded.
`steps~n_ops` is null and even wrong-signed (`rho=-0.771`), matching what
was reported. But its **`winding~n_ops` correlation is significant**
(`rho=+0.943`, `N=6`) and had gone entirely unreported.

Because maxtask has the same perfect `n_ops`/length collinearity as
counting (section 7b), that significant result cannot be attributed to
depth rather than length. Until this session neither switch nor maxtask had
*any* length control, because neither CSV recorded `seq_len` at all — the
column simply wasn't captured at extraction time in the original Kaggle
notebook. That has now been backfilled (reconstructing the exact prompts
from their generators, tokenizing them without needing the model, section
14) and the control run for the first time — but per section 7b, it comes
back degenerate rather than informative, exactly as it does for counting.
**So maxtask's significant winding result remains genuinely unexplained.**
It is not a clean negative control, and not a validated positive either.

**Behavioral ground truth: can the model even do these tasks?** A separate
check decodes the model's actual answer and compares it to the truth. It
runs its own sweep — `n_ops` 2, 4, 8, 16, 24, 32, 48, one level wider than
E2's — with 8 attempts each. Counting accuracy: **38% at `n_ops=2`** (3 of
8), 25% at 4, then at or near zero everywhere above: 0 of 8 at `n_ops` 8,
16, 32 and 48, with a single exception of 1 correct out of 8 at
`n_ops=24`.

This is load-bearing for E2 and E3 both. At the lengths where those
experiments look for a depth signature in the *geometry*, the model is
failing the underlying *task*. So "the geometry shows nothing" and "there's
no successful computation happening to have a geometric signature" are hard
to separate.

**Persistent homology.** Run on 15 saved trajectories (real `.npy` arrays
committed under `trajectories/`, like the one dissected in section 3). This
measures genuine topological loops — written *H1* in the
topological-data-analysis sense, a degree-1 homology class, which
unfortunately collides in name with hypothesis H1 and means something
entirely different.

The measure works like this: thicken every point of the trajectory into a
growing ball, and watch when a genuine hole appears in the resulting shape
and when it fills back in. How long it survives is its *persistence*,
divided here by the point cloud's own diameter so the score is comparable
across trajectories of different sizes. On that scale 1.0 would be an
unmistakable loop dominating the whole path. Observed: an average
per-trajectory value of `≈0.003` at `num_steps=64` (largest single value
0.009) and exactly `0.000` at `num_steps=16`.

That's expected *by construction* and is not evidence against looping: a
single 16-to-64-point curve in 5280 dimensions has essentially no
statistical power to register a topological loop. Independently, the
literature this project cites (Blayney et al., arXiv:2604.11791 — confirmed
real, Appendix C genuinely titled "Non-Fixed-Point Limiting Behavior")
reports only ~0.02% of tokens, up to 2.81% under some prompt conditions,
show non-fixed-point behavior in this model family. At that base rate,
15–40 sampled trajectories should show approximately zero loops even if
looping is real and common elsewhere.

**Convergence diagnostics** on the same 15 trajectories: those run at
`num_steps=64` shrink their step size roughly 52-fold start to end. (The
section 3 example is typical of this group rather than an outlier — its
65-fold figure is the ratio of its literal first step to its literal last
one, while the 52-fold average uses the run's own slightly different
shrink metric, under which that same trajectory scores ~53-fold.) They
show a mean step-to-step cosine of
`≈-0.28` — negative, meaning consecutive steps tend to point in partly
opposing directions, an oscillatory but still convergent approach. All 15,
including tighter-budget ones, show negative convergence rates.

## 9. The two-scale thread

Everything in section 8 uses one metric family (winding, steps-to-settle)
and one token position (the answer token). A separate line of work, on a
different git branch, varies both.

It borrows a different metric family from a different paper: Pappone et
al., "Two-Scale Latent Dynamics for Recurrent-Depth Transformers," NeurIPS
2025, arXiv:2509.23314 — confirmed real, equations independently checked
against this project's implementation, which matches faithfully (one gap:
the paper's *normalized* acceleration variant isn't implemented, only the
raw one).

Two quantities matter, both built from `Δ(k)`, the step vector from unroll
`k` to `k+1` — the same steps whose lengths were printed in section 3:

- **acceleration** `a(k) = ‖Δ(k) − Δ(k−1)‖` — how much the step vector
  changes between consecutive unrolls. Large while the path is still
  turning or changing speed; small once it moves smoothly.
- **orthogonality** `cos∠(Δ(k), Δ(k−1))` — whether consecutive steps point
  the same way (near +1), turn a corner (near 0), or reverse (near −1).

Crucially, this branch also adds the ability to record **every token
position's path at once** (`extract_trajectory_allpos`) rather than just
the answer token — which turns out to be the most consequential difference
between the two lines of work.

**The placeholder-data incident.** This branch's history includes a real
data-integrity problem, caught and corrected by the team itself. It needs
describing precisely, because "a number turned out to be fake" and "this
line of work is compromised" are different claims and only the first is
true.

An early script on the branch, `run_two_scale_analysis.py` (committed by
jack, 2026-07-12), generates trajectories from `np.random.randn` noise with
a hand-injected decay shaped by depth, rather than from the model — and
does so **transparently**, in its own comments and printed output
("Generating synthetic trajectories…"). Nothing about the script is
disguised; it reads exactly like standard practice, validating a metrics
pipeline on synthetic data with known structure before spending GPU time.

Its output was, at some later point, presented as if it were a genuine
Huginn finding (deck figures of `accel rho=0.997`, `orth=-0.5`). Whoever
did that presenting is named consistently across the honest fix's own
artifacts — `results/FINDINGS.txt`, `docs/two_scale.md`,
`tests/test_two_scale.py`, `notebooks/02_two_scale.ipynb`, all by Shtirmann
— as "David": *"David deck numbers (accel rho=0.997, orth=-0.5) =
np.random.randn + hand-coded 5+2\*depth. No model."* One of those files
also credits David with writing the original per-position two-scale metric
code, not just showing a slide.

**No commit anywhere in this repository is authored by anyone named
David.** His name appears only inside file *contents*, never as a
committer. So whatever he did happened outside this repository — his own
machine, his own Kaggle session — and was never pushed. That reframes the
incident considerably: it looks much more like *someone's own separate,
uncommitted work got presented before the committed pipeline behind it was
verified* than like anyone editing shared code to disguise a fake result.

That the presented numbers came from noise is independently reconfirmed
here, not taken on the team's word: recomputing per-depth statistics from
the still-committed `results/two_scale_analysis.csv` shows
`mean_orthogonality` pinned between `-0.496` and `-0.500` on *every row
regardless of depth* — the signature of noise decaying at a
depth-independent rate. The placeholder files remain in the tree on the
unmerged branch, kept deliberately so the diff shows the correction —
itself a residual risk, since nothing stops someone opening the wrong file.

**The honest fix and the band test.** The fix replaces the noise script
with a real pipeline against actual Huginn output, producing this branch's
central result: a **band test** comparing adjacent PARARULE-Plus depths
(2-vs-3, 3-vs-4, 4-vs-5) after restricting each comparison to overlapping
prompt-length ranges, then fitting **content-token acceleration** — the
acceleration metric above, averaged across every prompt token position
*except* the answer token — against a depth indicator while controlling for
length.

Headline: content-token acceleration rises **+1.30** per unit of depth, net
of the length control, 95% CI `[1.05, 1.55]`, `p<1e-4`. Independently
reproduced this session with a completely separate implementation
(`statsmodels` with robust standard errors, versus the project's
hand-rolled regression): `+1.2974`, `p=6.6e-23`. On the **answer** token
the same design gives a null (`≈+0.04`, CI crossing zero) — consistent with
section 8's answer-token nulls on the winding metric.

**Why "content tokens carry the signal" doesn't hold yet.** Read
optimistically, two null answer-token results plus one positive
content-token result suggest *the depth signal lives on content tokens, not
the answer token*. That framing was written into the project's working
notes and needs walking back.

It compares results differing in two uncontrolled ways at once: a different
unroll budget (64 steps on the answer-token work, 32 here) and a different,
separately-sourced data-loading function for the same nominal dataset. More
importantly, it ignores a more direct test **this branch's own team already
ran**: scoring E3's track/local design with the two-scale acceleration
metric instead of winding. That came back completely null — content-token
acceleration rose for *both* the state-holding and the non-state-holding
condition alike. A synthesis that sets aside its own team's
counter-evidence on the one task built to test exactly this isn't validated
yet.

**The length-matching problem.** The band-test coefficient carries a second
independent problem, traced fully to its source this session. The design's
premise is that the compared groups are matched on prompt length. Checked
against the committed data: they aren't. Within every band the two groups'
length ranges are **completely disjoint** — gaps of 3, 5, and 11 tokens
between nearest edges. Within-band collinearity is severe: point-biserial
correlation (between the continuous length and the binary depth-group
label) of `-0.89` to `-0.90`, and the length coefficient's sign flips
depending on whether the depth indicator is in the model. That sign flip is
the textbook symptom: when two predictors are nearly redundant, the
regression can't tell which of them owns the effect, so small changes to
the model swing their coefficients wildly — sometimes right past zero.

The working hypothesis was that the committed matching code (`_length_match()`,
which pairs examples by *exact* length equality) wasn't what generated the
committed results. That's now resolved, and not in the result's favor: the
real generating code was found on the same branch, in
`notebooks/02_two_scale.ipynb` — already committed, simply never opened.
Its actual design doesn't pair examples at all; it restricts each side to a
*pool-level* overlapping length range and samples up to 60 per side from
within it. Its own printed diagnostics (`len_lo=171 len_hi=153` for the
2-vs-3 band, similar for the others — larger gaps than the range-edge
numbers because these are group *means*) reproduce the same disjointness
found from the data alone, confirming this notebook run is the true source.

And the notebook's own closing cell states the condition for trusting the
result: *"If depth-coef|len is clearly >0 … with `len_lo~len_hi` →
acceleration tracks depth beyond length."* In plain terms: the coefficient
should be positive **and** the two groups' lengths should actually be
close. By its author's own stated bar, using its own printed numbers, it
isn't met. Not a provenance mystery — a directly observed property of an
honestly generated result falling short of its own standard.

**What survives.** None of this makes the coefficient fake. A within-band
permutation test (reshuffling the depth indicator 2,000 times per band and
refitting) still finds signal (`p≈0`), and a comparison restricted to
examples nearest each band's length boundary — the closest thing to a
genuinely matched comparison in the existing data — stays significant in
all three bands. A sensitivity check adding a depth×length interaction
drops the coefficient to `+1.09` and weakens it to `p=0.025`; whether
that's genuine specification sensitivity or just the documented
multicollinearity can't be told from the data. **Honest summary: a real
effect very likely exists; the specific `+1.30, p<1e-4` figure shouldn't be
quoted as a clean length-controlled result without this caveat.**

**Two smaller findings.** The full depth range without length-matching
shows a strong correlation its own documentation correctly flags as
uninterpretable (depth and length are near-perfectly collinear across the
unrestricted range) and honestly defers to the band test. And the attempted
reconstruction of this branch's missing data-loading function — made before
the real notebook surfaced — guessed wrong in several concrete ways (wrong
dataset configuration, wrong prompt format, different handling of a
metadata field). A useful reminder that a plausible, test-passing
reconstruction is not the genuine artifact.

## 10. Branches, people, and how the work is actually distributed

The repository has four pushed lines of history, not reconciled with each
other, and commits attributable to exactly two people.

**The two people.** The commit list appears to show three authors, but
"Shtirmann" and "Alexander Shiyanov" are the same GitHub account —
confirmed against GitHub's API, not inferred (same numeric account ID; one
identity comes from a local git client, the other from the web interface).
So: Shtirmann and "jack," and nobody else. No commit, PR, or issue
anywhere is attributable to a third person, despite the project being
described as a three-person effort — worth confirming with the team, since
the repository gives no evidence of who that is. (Nor of David, section 9,
who has no commits at all.)

**Who wrote the code, versus who owns which branch.** Every module in
`main`'s package carries an `OWNER:` tag naming one of three intended roles
— "Extraction+Winding" (13 files), "Data+Analysis" (12 files),
"Shapes+Gate" (5 files). Checked against git blame rather than trusting the
tags: **all 30 files, across all three roles, were written by Shtirmann
alone.** The three-role structure describes an intended division of labor,
not who has hands-on familiarity with any given part. Jack's one code
contribution (the two-scale branch) doesn't follow the convention: one new
file matches the documentation style, the rest have no header.

**The branch chain.** These are not four independent forks of `main` —
each continues the one before it, so reconciling the newest work with
`main` means walking back through all three intermediate steps:

```
main
  ...
  8b5e418  2026-07-12 15:01  Shtirmann   "exp 3"
  fe00b8b  2026-07-12 22:50  Shtirmann   "retesting, exp 4"
  14e77eb  2026-07-12 22:53  Shiyanov    "Revise README..."   <- main's tip, unchanged since
    │
    │  forked from 8b5e418, 2026-07-12 20:02 (while Shtirmann was still
    │  committing to main, three hours before main's own last commit)
    ▼
feature/two-scale-latent-dynamics-paper
  155c6d0  2026-07-12 20:02  jack        "feat: Add Two-Scale Latent Dynamics analysis"
                                          -> ships with the placeholder data (section 9)
    │
    │  pull request #1 opened against this branch, 2026-07-15 13:43
    │  — still open, 0 comments / 0 reviews / 0 CI
    ▼
two-scale-real-fix
  2ac075e  2026-07-15 16:42  Shtirmann   "fix(two-scale): replace synthetic placeholder
                                          with real-Huginn results"
  8efc31d  2026-07-15 16:46  Shtirmann   "added notebook"
                                          -> the real band-test generating notebook
                                             (section 9) lives here, unopened for 3 days
    │
    │  forked again, no pull request this time, 2026-07-17
    ▼
pr-real-data
  900b8b4  2026-07-17 18:12  jack        "docs: add concise real-data results summary"
                                          -> restates the band-test coefficient as settled,
                                             no mention of the length-matching problem
                                          -> current tip; most recent activity in the repo
```

PR #1, against `two-scale-real-fix`, was opened by Shtirmann — the person
who caught and fixed the placeholder data, not the one who introduced it —
three days after the fact; nobody touched the problem in that gap. Once
opened, it sat unreviewed for a further three days (0 comments, 0 reviews,
0 CI), and it is technically mergeable with zero conflicts, so what's kept
it unreviewed is coordination, not code. The fourth step, `pr-real-data`,
was pushed by jack two days after the fix, has no PR of its own, and is
consequently *less* visible than the PR that was already being ignored.

This is the live state, not a historical episode: a fresh fetch while
writing this shows no activity newer than `pr-real-data`'s single commit.
And it isn't low-stakes to leave open — this project has a **2026-08-02
submission deadline** (SMILES/Zapiski POMI proceedings format, per
`files/Smiles26Barannikov Proposal.pdf`), and no paper draft exists yet: no
`.tex`, no outline. That's plausibly the project's largest schedule risk,
independent of which number ends up cited.

**Two additional local-only branches** exist from this documentation and
fix work, never pushed, sitting on top of `main`:

- `fix/repro-and-scaffolding` (3 commits) — the `.gitignore` root-cause
  fix, a corrected README reproduce command, and the completion of a
  plotting module stubbed since the first commit.
- `feat/close-known-gaps` (1 commit) — restores the missing
  `data/loaders.py` (section 11), fixes two hardcoded CUDA device strings,
  removes the unreachable `N=4` table entry (section 7a), and adds the
  `seq_len` backfill plus length controls for switch/maxtask (section 8's
  E5).

## 11. Defects found in the project's own infrastructure

Independent of any experimental result, several defects were found and, in
most cases, verified by reproducing the failure rather than inferring it.

**The most consequential: a source file that has never existed in git.**
`src/traj_geom/data/loaders.py` — imported at module level by the
PARARULE-touching scripts on `main` and the two-scale branch — **does not
exist anywhere in this repository's history, on any pushed branch**,
confirmed by searching every object ever committed to every remote branch.

The root cause is one line in `.gitignore`, present since the first
scaffold commit: a bare `data/` rule, meant to exclude a top-level dataset
cache, which as written matches *any* directory named `data` anywhere in
the tree — including this real source package. Every `git add` has silently
dropped the file since day one, with no error or warning, because ignored
files simply never appear in `git status` at all.

Consequences, verified in an isolated fresh clone: `run_pararule.py` can't
even be imported, cached CSV or not — the one exception to section 5's
zero-GPU-rerun property. And the project's own documented lint command,
`ruff check .`, **fails** on a clean checkout, because the missing module
breaks import-order analysis in the file that imports it. (`pytest`, by
contrast, passes — no test imports the module.) Both are fixed on
`feat/close-known-gaps`, where the file is restored from a reconstruction
verified byte-identical against two independent earlier attempts, and the
ignore rule is anchored to `/data/`.

E1's numbers in section 8 were therefore re-derived by computing the
canonical per-level correlation directly from the committed
`results/pararule.csv`, not by running the script. One trap worth naming:
`notebooks/01_mvp.ipynb` itself prints a *different* number for the same
nominal quantity — `rho=-0.037` — because that's the pseudoreplicated
per-row correlation section 7a warns against, computed over all 40
trajectories at once rather than the four depth-level means. Don't confuse
it with the `+0.20` reported above.

**A related documentation defect:** the README's reproduce instructions
omit an installation flag that the very tools they invoke (`pytest`,
`ruff`) need to exist at all. Following them literally makes `pytest` fall
back to an unrelated system-wide installation with none of this project's
code, producing a failure that looks like a broken install rather than a
one-word omission.

**Smaller items.** A hardcoded CUDA-only device string existed in a second
location in the extraction code, bypassing the `device=` parameter that
`load_huginn()` already exposes — so passing `device="cpu"` never actually
worked. (No notebook ever tried; this was an unfulfilled promise in the
function's own signature rather than a live regression. Fixed on
`feat/close-known-gaps`.) A stray orphaned result file, produced only by an
inline notebook cell with no corresponding script, sits alongside the
properly scripted force-loop experiment it appears superseded by. And the
significance-table error from section 7a.

## 12. Where each hypothesis actually stands

Scorecard first; the reasoning is in sections 8–9, and every number here is
repeated, not newly introduced. `N` means number of levels (section 7a); a
blank means the test isn't a per-level correlation.

| Hyp. | Experiment | Test | Result | N | Significant? |
|---|---|---|---|---|---|
| H1 | E4 force-loop, `num_steps=16`, `n_ops` 8→24 | Fisher exact | unsettled fraction 1/8 → 7/8 | — | **Yes**, p=0.0101 |
| H1 | Persistent homology, 15 banked trajectories | max degree-1 persistence | ≈0 | — | Underpowered — not a real test either way |
| H2 | E1 PARARULE, winding~depth | Spearman | rho=+0.20 | 4 | No |
| H2 | E1 PARARULE, steps~depth | Spearman | rho=+0.80 | 4 | No |
| H2 | E2 counting, winding~n_ops | Spearman (raw) | rho=+0.94 | 6 | Yes — but uninterpretable, length perfectly confounded |
| H2 | E2 counting, length-controlled | partial Spearman | rho=+0.036 | — | **Degenerate — control cannot run (§7b)** |
| H2 | E5 switch (parity), both metrics | Spearman | rho=+0.78 / −0.09 | 6 | No |
| H2 | E5 maxtask, winding~n_ops | Spearman | rho=+0.94 | 6 | Yes — unexplained; control degenerate, so still open |
| H2 | Two-scale band test, **content** token | linear regression | +1.30 (interaction-corrected: +1.09) | — | Yes, p<1e-4 (corrected p=0.025) — see §9's caveat |
| H2 | Two-scale band test, **answer** token | linear regression | ≈+0.04 | — | No |
| H3 | E3 dissociation, base run (5-seed / 15-seed) | Spearman | rho=+0.81 / +0.71 | 6 | No |
| H3 | E3 dissociation, multi-init robustness | Spearman | pooled rho=+0.84, per-init range 0.17–0.87 | 10 | Yes pooled, unstable across seeds |
| H3 | Two-scale metric on E3's track vs. local | — | no dissociation found | — | No |

**H1 (settle / loop / drift).** All three regimes have genuinely been
observed in this project's data — but only under an artificially starved
compute budget. Under the model's normal budget every recorded trajectory
settles, and the sample sizes are far too small to detect looping at the
very low base rate the literature reports. **The project's strongest,
cleanest positive evidence anywhere is H1-relevant, not H2-relevant**: the
significant rise in unsettled trajectories as task difficulty increases
under a starved budget (E4).

**H2 (winding tracks depth).** Null, cleanly and consistently, on the
answer token, across two independently implemented metric families. A real,
independently reproduced positive coefficient exists on content tokens, on
one branch, with one metric — but that same branch's own attempt to
replicate the analogous effect on a purpose-built task came back null, and
the coefficient's "length-matched" design fails its own author's stated
criterion. Some real signal likely survives; the published figure shouldn't
be treated as settled.

Two supposed data points *against* H2 also have to be withdrawn as
evidence: E2's and maxtask's length controls are degenerate (section 7b),
so they neither support nor refute anything. Maxtask's significant winding
correlation is, at present, simply unexplained.

**H3 (contraction forces looping).** The theorem is proven and correct. On
the real model the empirical picture is one small, seed-unstable positive
correlate (E3's multi-init run) and no support from the two-scale branch's
attempt at the same question. And the premise itself — whether Huginn's
recurrent map is anywhere near a strict contraction — **has never been
measured**, in either direction.

## 13. What's genuinely open

**The experiment nobody has run.** Every experiment above sets the
recurrence depth by hand — a fixed grid, 64 unrolls on this branch, 32 on
the two-scale one — and then asks whether trajectory geometry correlates
with difficulty. But Huginn ships with **its own adaptive-depth
mechanism**: `generate_with_adaptive_compute` takes a `criterion`
argument selecting among six built-in exit rules (`entropy-diff`,
`latent-diff`, `cosine`, `kl`, `minp-kl`, `argmax-stability`). Two of them
are geometric, measuring essentially what this project measures by hand.

**No branch of this project ever passes a `criterion`** — verified by a
repo-wide search across scripts and notebooks on all branches. Every call
uses the default `criterion="none"`, which the model's own source
describes as *"adaptive compute is off by default."* So `eval.py`, whose
docstring advertises "Huginn's adaptive-compute generation," in fact runs
at a fixed step count with adaptivity disabled.

That leaves an obvious, cheap, never-attempted experiment: turn a
criterion on and ask directly whether the model exits later on harder
problems. It tests the same underlying question using the model's native
mechanism instead of a reimplementation, needs no new task design, and —
given that H2 is null on the answer token across two independent metric
families — is arguably the strongest remaining experiment available here.

**Measurement gaps.** Whether Huginn's actual contraction constant is
anywhere near the strict-contraction regime H3's proof assumes — the
single measurement that would move H3 from "theory plus indirect
consistency" to "theory plus evidence." Whether maxtask's significant
winding correlation is real or a length artifact — now known *not* to be
answerable by the existing control, and requiring a task template where
length can vary independently of difficulty.

**Design gaps.** Whether "content tokens carry the depth signal" survives
being tested properly — same data, same unroll budget, same loading code,
rather than compared across branches differing in all three. Whether the
band-test coefficient holds under a matching design that actually achieves
overlapping lengths rather than pool-level range restriction. Whether any
of the synthetic tasks can be redesigned so difficulty and prompt length
are separable at all — currently none of them are.

**Process gaps.** Whether PR #1 and its less visible sibling branch get
reconciled with `main` before the 2026-08-02 deadline, and whether anyone
tells the author of that sibling branch about the length-matching problem
before its numbers propagate further. Who the project's third team member
is. Who David is, and whether the venue where those deck numbers were shown
ever got a correction.

Every measurement item above traces back to the same root: nobody has
measured whether the model's recurrent map is close to the contraction H3
assumes. Until that exists, the project's central theoretical claim stays a
hypothesis consistent with the data, not one confirmed by it.

## 14. How to actually run this

```bash
cd Geometry-of-Reasoning-Trajectories

uv sync --extra dev          # base + pytest/ruff — NOT plain `uv sync`
uv run pytest                # -> 11 passed, 1 skipped (homology test needs ripser)
uv run ruff check .          # -> All checks passed (after §11's loaders.py fix)

uv sync --extra tda          # adds ripser/persim -> pytest now 13 passed
uv sync --extra model        # adds torch/transformers — only needed to load the model

# note: `uv sync --extra X` alone *replaces* the environment, dropping the
# others. To have several at once, name them together:
uv sync --extra dev --extra tda --extra model
```

Note `--extra dev` specifically: plain `uv sync` doesn't install `pytest`,
and `uv run pytest` then silently falls back to whatever is on `$PATH`,
producing a confusing unrelated error (section 11).

**Re-deriving any published result, no GPU:** every experiment script reads
its committed CSV and re-prints its statistics.

```bash
uv run python -m scripts.run_counting     # E2
uv run python -m scripts.run_forceloop    # E4, the cleanest result
uv run python -m scripts.run_switch       # E5
uv run python -m scripts.run_maxtask      # E5
uv run python -m scripts.run_dissociation # E3
uv run python -m scripts.run_pararule     # E1 — needs the loaders.py fix (§11)
```

**Extracting genuinely new trajectories requires a GPU** and, in practice,
Kaggle. `extract_trajectory(model, tok, prompt, num_steps, seed)` is the
entry point; read `constants.py`'s docstring first — every gotcha there was
learned by debugging a crash.

**Adding a new experiment**, following existing convention: write
`make_<name>_task(n_ops, seed) -> dict` in `shapes/synthetic.py`; copy the
shape of `run_switch.py` (~60 lines); use `cached()` so results persist;
report with `fmt_by_level()`. And — the lesson of section 7b — **check up
front whether your prompt's token length can vary independently of your
difficulty variable.** If it can't, no statistical control will rescue the
design later; fix it at the template. Also worth checking, per section 8's
behavioral result, whether the model can actually perform your task at the
difficulty levels you plan to test.

---

This document is self-contained and doesn't repeat what the other files in
this folder are for: `claims_ledger.md` has the row-by-row evidence trail
behind every number; `guide.md` and `infra.md` are operating references
(module dependency graph, extension cookbook); `research_log.md` is the
chronological trace, including mistakes made and caught along the way;
`START_HERE.md` is the short version with a ranked action list. Start there
if you want recommendations; start here if you want to understand the
project first.
