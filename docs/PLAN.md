# Plan — what to do next, and why

*Opened 2026-08-09, after the supervisor asked four questions the project had not
answered. Live document. `directions.md` holds every open thread; this holds the
ordered plan and the reasoning behind the order.*

---

## STATE — read this first after any context loss

*Last updated 2026-08-09, after D92–D94. Deadline 21:00 MSK.*

**THE HEADLINE, as of D92–D94** — this is now the project's spine, and it is
measured on data that was already on disk. Full argument in `depth_profile.md`.

> **Recurrent depth in Huginn does not compute the answer — it relocates it.**
> Top-1 accuracy at the answer position peaks at **unroll 5 (16.1%)** and collapses
> to a flat **6.2% by unroll 8**, identical across all 21 families from r=32 onward
> (family-paired sign test, 14 of 15 discordant families, **p = 9.8e-4**). Yet the
> answer's presence in the generated text **rises** 29.4% → 42.9% over the same
> range while a length-matched decoy **falls** 16.0% → 2.7% (specificity 1.8× →
> 15.9×). And `correct` — the label used project-wide — is `min_r rank_r == 1`, a
> minimum over unrolls: **33.9%, against 16.1% for the best selectable depth.**

**Running now**
- `geometry-h0bank` — RUNNING. 16 boundary prompts × 32 unseeded h₀ draws.
  The powered, **pre-registered** test of D91's lead. Analyse with
  `scripts/run_h0_within.py`, committed *before* the data existed.
  **Do not change that file after the data lands.**
- `geometry-clrs` — RUNNING. The CLRS-Text capability screen.

**Built, tested and queued, blocked only on a GPU slot** (Kaggle caps at 2)
- `scratch/kaggle_answerpos/` — **the decisive follow-up to D94.** Walks the model's
  own greedy continuation and ranks gold at *every* generated position, so
  "relocated" stops being an inference. Battery pinned byte-identical to
  `depthacc` by test, so the two banks join on (family, item, fmt, depth).
  Predictions pre-registered Q1–Q5 in the body docstring.
  Push: `uv run kaggle kernels push -p scratch/kaggle_answerpos`.

**Findings a reader must not lose**
- **D92** — `correct` is an oracle over unrolls. 33.9% vs 16.1% best fixed depth vs
  6.2% converged. **53% of the project's measured capability is unrealisable.**
- **D93** — depth *degrades* next-token accuracy, and is done degrading by r=8.
- **D94** — but containment *rises* and decoys *fall*, so the answer is relocated,
  not lost. The framing mechanism I first proposed is **refuted** (median rank 7.0
  framed vs 7.5 unframed at depth 32) and recorded as refuted.
- **D90, CORRECTED by D92/D93.** h₀ does *not* decide the converged answer: the
  last-unroll rank is a single integer across 10 draws in 7 of 8 prompts. h₀ moves
  the transient minimum, which is what `correct` reads. The consequence for the
  project stands and sharpens.
- **D89** — `correct` scores only the FIRST TOKEN of gold; 8 of 21 families have
  multi-token golds. This sits *on top of* D92, not instead of it.
- **D91 — the lead.** Correctness decodes at 68.4% (p = 0.010) at unrolls 6–18,
  chance out to 56; does not survive Bonferroni over 8 windows, n = 60. D93 now
  explains *why* that window and no other: it is where answer content stops moving.

**Next, in order**
1. Push `answerpos` the moment a slot frees. It is the one run that converts D94's
   central word ("relocates") from inference to measurement.
2. Analyse h0bank with the pre-registered script. If a transient window clears
   α = 0.00625 and the far tail does not, D91 replicates and the correctness nulls
   (D79/D84/D85) narrow to the *converged* region — which D93 independently says is
   the region where nothing changes.
3. Analyse the CLRS screen (§2).
4. Activation patching (§4a); the QK probe (§4b), still 0% done.

**Literature position** (`docs/related_work.md` for verification status). "Depth is a
contraction" is scooped by Blayney et al. (verified title/abstract). D92–D94 are not:
nothing found in that scouting pass measures accuracy against depth with a
length-matched containment control, and the contraction papers do not ask whether the
fixed point is the answer. Several 2026 arXiv numbers reached me via scouting agents
rather than PDFs and are marked unverified — **do not put them in the ledger without
checking the PDF.**

---

## 0. Four things settled from the model source, not from memory

Read out of `raven_modeling_minimal.py` at the pinned revision
`bb6621b65e90b6a4b9b29ef88dc83866d450470c`.

**What h₀ actually is.** `initialize_state` returns `torch.randn_like(input_embeds)`
re-drawn as a truncated normal at `std = config.init_values["std"]`, times
`emb_scale`. So the initial latent is **random noise shaped and scaled like the
embeddings — it is not the embeddings**. Unseeded, which is D78.

**What is injected, and how often.** Every unroll begins
`x = adapter(cat([x, input_embeds]))`. `input_embeds` here is the **prelude output**
— the token embeddings after two transformer blocks — not the raw embedding table.
So the prompt re-enters the state at *every* unroll, already partly processed. This
is why linear content probes are near-tautological (D70(4), D73).

**The block does not know which unroll it is on.** `block_idx` is threaded through
`core_block_forward` and used in exactly one place: `past_key_values.update(k, v,
block_idx)`, the KV-cache slot. It is **not** a timestep conditioning input. The
recurrence is therefore an **autonomous map** `x ← F(x, e)`, the same `F` every
iteration.

**Huginn's training mixture contains `tomg-group-umd/CLRS-Text-train`.** 430,000
rows, 30 classical algorithms rendered as text with step traces. It is in the
pre-training data, so it is *in distribution*, and it carries a clean integer
difficulty parameter. This is the single most useful fact in this document — see §2.

---

## 1. What the autonomous-map fact does to H1

H1 is the settle/loop/drift taxonomy. B4 found loop and drift **only** when the
compute budget is starved to `num_steps=16`, vanishing by 24.

**The autonomous-map fact is verified from source; the contraction is a
MEASUREMENT, and the difference matters.** An autonomous map converges only if it is
contractive — otherwise limit cycles and divergence are both available to it. So the
argument "loop and drift cannot persist" is *conditional on ρ<1*, and stating it as
though it were architectural was an overclaim on my part.

**Measured, 2026-08-09, over all 504 geomcap orbits at all five windows:**
ρ ∈ **[0.808, 0.920]**, median 0.859, and **zero orbits at or above 0.98**, let alone
1.0. So on this task set the condition holds with a healthy margin, and the argument
goes through. It does *not* establish ρ<1 for all inputs — 21 synthetic families at
one token position is not the space of prompts, and the supervisor's point stands
that a regime where contraction is not imposed is a different question.

Two things follow that the first draft of this section got wrong:

1. **ρ varies systematically with the task** — per-family medians run 0.829
   (`add_2d`) to 0.906 (`count16`), a real ordered spread. "Settles fast / settles
   slow" is therefore a live graded property even though settle/loop/drift is not a
   live taxonomy. But D85 already tested ρ against capability (ρ_Spearman = −0.131,
   n.s.) and against prompt length (**+0.665**), so the variation that exists is
   length, not difficulty.
2. **Contraction may be something Huginn's TRAINING imposes**, not a property of
   looped architectures generally — D80(7) measured the trained arm collapsing 8×
   faster than six untrained weight draws, which diffuse. If the hypothesis is about
   models where that has not been imposed, this model cannot test it, and saying so
   is different from refuting it.

The supervisor's intuition about the schedule is right and the source confirms it:
**more budget is more iterations of one fixed map, not passage through a schedule.**
Diffusion feeds the timestep to the network; Huginn's `block_idx` reaches only the
KV-cache slot. So *within the measured ρ range*, loop-and-drift-at-16 is a taxonomy
of where the recording stopped rather than of computations — D80's window law again.

---

## 2. The task problem, and CLRS-Text as the answer

**The gap, stated plainly.** The 21-family battery spans 0–100% accuracy, and 11
families sit in the informative 20–80% band. But difficulty varies *across* families,
not along a parameter *within* one. The only two parametric ladders are broken:

| ladder | accuracies | verdict |
|---|---|---|
| `count4 / count8 / count16` | 21% / 0% / 12% | at the floor, and non-monotone |
| `caesar1_letter / caesar1_word / rot13_word` | 8% / 0% / 0% | dead |

So the project has never had the design the supervisor described — Caesar × rotation
count, prefix sums × prefix length — with success in a readable band.

**Why CLRS-Text fixes it.**

1. **In distribution.** It is in Huginn's own pre-training mixture. Every task the
   project has used so far was invented here; if the model looks incapable, "off
   distribution" is a live explanation that CLRS removes.
2. **Two independent difficulty knobs**: the algorithm (30 of them, from
   `activity_selector` to `topological_sort` to `find_maximum_subarray_kadane`) and
   the problem size *n*.
3. **It can break the length/difficulty collinearity**, which is the property that
   matters most. D85 measured the geometry tracking prompt length, and D84/D88 showed
   the shape reading a single token — so any difficulty axis collinear with token
   count is unusable. In CLRS, array algorithms have input length ~n while graph
   algorithms have ~n², so **(algorithm, n) pairs exist with matched token counts and
   very different difficulty**. That is the design no synthetic family here has
   offered.
4. **The answer is separable.** `answer` is `trace | final`, so the final answer can
   be scored alone, and the trace gives a per-step ground truth if wanted.

**First run.** A capability screen over ~8 algorithms × n ∈ {4…12}, scored on the
final answer, to find the cells that land in 20–80%. Everything downstream needs that
map first, and it is cheap: no trajectory banking, one forward per item.

**Risk to check in that same run.** CLRS answers are long numeric strings, so
exact-match will be brittle and the first-token rank readout is not the right
instrument. Score the final field with normalisation, and record the full decode.

---

## 3. A measurement bug in the capability axis — CHECKED, real, D89

Confirmed the same day. `correct` scores *the first token of gold*, and **8 of 21
families have multi-token golds; 4 of them for every item**. `compare`'s 92% is
first-**digit** accuracy on a two-digit comparison, which magnitude alone often
decides. The ciphers may be deflated by the same mechanism — gold `'acuj'` tokenises
to `['acu','j']`, so a model emitting `'a'` scores zero.

**D85's conclusion survives.** Re-run against a tokenisation-free axis — decoded
strings from `geometry-depthacc`, same families, same item prefix — nothing reaches a
corrected threshold at any window. The two axes agree at ρ = +0.791 (containment),
so the ordering was substantially right.

**What must change in wording**: per-family accuracies in D75/D85 are first-token
accuracies for the eight families listed in D89(1), and `compare` must not be
described as a 92%-capable cell.

**What must change in future kernels**: score the full gold token sequence, or
restrict generators to single-token golds and gate on it in-kernel the way
`geometry-lenmatch` gates token counts. This is a hard requirement for the CLRS run
in §2, whose answers are long numeric strings — the first-token readout is simply
the wrong instrument there.

---

## 4. The method gap: no causal evidence at all

Everything in the project is observational. The one causal attempt, D47, is recorded
as **void rather than null** — the outcome variable had zero dynamic range before any
intervention, so it separated "the direction is unused" from "the readout is
degenerate" not at all.

Ranked by value per T4-hour, given no training budget:

**(a) Activation patching across unrolls — do this first.** Take a prompt the model
gets right and one it gets wrong, patch the state at unroll *r* from one into the
other, and read the answer. Sweep *r*. This asks *when* the answer is determined,
directly, and it needs one forward per patch and no gradients. It is also the
experiment that would put a causal floor under D86's "the answer is available early".
The design must fix D47's failure by construction: **choose cells with real dynamic
range in the outcome** — the battery now names them (`echo_digit` 100%, `compare`
92%, `add1` 79%, `sub1` 62%).

**(b) The QK-alignment probe — the supervisor's own open item.** `project_plan.md`
G2 lists "winding-vs-depth plus a query–key alignment probe" and records it as **0%
done**; it is item 6 of the pre-registration. A query·key dot product per unroll is
cheap to extract alongside the states. It has never been run, and it is the one
supervisor-nominated statistic still untouched.

**(c) Steering / activation addition.** Add a direction to the state at unroll *r*
and measure the output shift. Cheap, no training. Weaker than patching, because a
found direction is easier to over-interpret; do it after (a).

**(d) SAEs — not now.** Training a sparse autoencoder on 5280-d residuals needs a
training budget the project does not have, and on one model with no feature ground
truth the payoff is a dictionary nobody can validate. Revisit only if (a) finds a
localised effect worth decomposing.

**(e) Function vectors / task vectors.** Genuinely interesting here, because D88
showed the shape responds to a task *marker* token even when the task is unchanged.
The natural follow-up: is there a direction in state space that *is* the task, such
that adding it makes the model perform task B under marker A? That is a real
experiment and it sits naturally on top of (a)'s machinery. Second priority after
(a).

---

## 5. Connecting ρ to behaviour — H3's missing half

D52 is the project's best-replicated result and has **never been connected to
behaviour** (`UNDERSTANDING` §6.2). With CLRS giving accuracy in a readable band, the
test becomes available: does the per-prompt contraction rate predict whether *that
prompt* is answered correctly, at matched task and matched answer? The stratification
machinery for this already exists (`analysis/stratified.py`, D79) and so does the
bounding machinery (`analysis/reliability.py`, D85). Only the tasks were missing.

---

## 6. On training a network on synthetic winding/drift/loop shapes

Considered and **not recommended**, for two reasons that are worth stating rather
than just declining.

**The train/test mismatch is the strong kind.** Synthetic planar spirals live on a
2-D manifold; D74 measured the real orbit at ~13 effective dimensions over the
transient. A classifier fitted to synthetic shapes and applied to a 5280-d orbit is
not extrapolating a little, it is evaluating off its support entirely, and its errors
would be invisible.

**The model-free version of the idea is already done and is better.** "Let a learned
function decide what the shape carries" is exactly the shape-code classifier — but
trained *and* tested on real orbits, with no synthetic prior at all (D84, D87, D88).
It found the input at ceiling and the outcome at chance, against permutation nulls
and a measured detection floor. A synthetic-shape model could not have beaten that
and could easily have been worse in an undetectable way.

**Where synthetic data does belong, and is already used heavily:** validating the
analysis. Every test file plants a known effect and requires the pipeline to recover
it — a planted family-only signal must come back null, a planted within-stratum
effect must be found, a diffusing orbit must not show a depth law. That is the role
synthetic data can play honestly.

---

## 7. Order of work

1. ~~§3's tokenisation check~~ — **done, D89.** Defect real, D85's null survives.
2. **CLRS capability screen** — find the 20–80% cells. One T4-hour.
3. **Activation patching** on cells with real dynamic range (§4a). Two T4-hours.
4. **CLRS trajectory bank at graded n**, with token count measured and matched where
   possible — then re-run H2, H3-vs-behaviour, and the shape/outcome decode on a task
   family that is in-distribution and in-band.
5. **QK probe** (§4b) alongside the bank, since it is nearly free once states are
   being extracted.
6. **Function vectors** (§4e), if patching finds anything localised.

PARARULE-Plus stays on the list but below CLRS: it is textual and depth-graded, but
its depth is confounded with passage length, the project's one run had n=10 per depth
over 4 levels (underpowered by construction — the critical ρ at N=4 is 1.000), and it
is not in Huginn's training mixture. It becomes the natural *generalisation* test
once CLRS establishes the effect in-distribution.
