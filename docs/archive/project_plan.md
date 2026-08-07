> **FROZEN 2026-08-05.** Archived under CLAUDE.md §6 (three live documents only).
> Not maintained. Superseded by `docs/UNDERSTANDING.md`, `claims_ledger.md`, `directions.md`.
> Statements here may be contradicted by later work — several assert things since disproven
> (e.g. "H1 was never tested": it was, and the test itself was invalid — see D56, D65).

# Project plan — Geometry of Reasoning Trajectories in Huginn-3.5B

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


STATUS: **living plan, first authored 2026-07-24.** This is the authoritative
"what the project is, where it actually stands vs. its own proposal, and what
to do next and why" document. It supersedes scattered "further work" notes.
Built from: the curator's proposal (`files/Smiles26Barannikov Proposal.pdf`),
the H3 proof + team lit-survey (`files/`), every `docs/*.md`, a multi-agent
grounding+design+adversarial-verify pass over the real code, and independent
re-verification of the load-bearing empirical claims. Every non-obvious claim
is tagged **[verified]** (I re-derived/read it this session), **[grounded]**
(a subagent read it from source with file:line), or **[unverified]** (a lead
that needs an external read or curator input before it is trusted).

How this relates to the other docs: `claims_ledger.md` = every quantitative
claim, tagged; `architecture_state.md` = file/experiment inventory + decisions
log; `narrative.md` = the single-linear-read explanation; `deep_research_huginn_literature.md`
= verified external literature. This plan is the **forward** document; those
are the **backward** ones. When a number here contradicts an older doc, this
one is newer — but re-verify before citing.

---

## 0. TL;DR

The project set out (proposal, curator = Serguei Barannikov) to test whether
the **geometry of Huginn-3.5B's latent path** (as it iterates its shared core
block) reveals what it computes: **H1** three shapes (settle/loop/drift),
**H2** winding ∝ reasoning depth *when it loops*, **H3** forced contraction
destroys state-tracking. Goals **G1** (measure shapes at *every token, every
depth*), **G2** (winding-vs-depth **plus a query–key alignment probe** — the
curator's own method), **G3** (prove the contraction theorem + measure the
spectral radius on real Huginn).

**Where we actually are:** the theorem is proved but its *spectral radius has
never been measured on real Huginn*; the QK probe is **0% done** though it is
the curator's own method and half of G2; tracing is **answer-token-only**
(violating G1 and, per the project's own data, looking exactly where the
phenomenon is least present); and every direct winding-vs-depth test is null
or a length/position artifact. The honest current verdict: **the project has
mostly produced rigorous negatives**, and those negatives *independently
corroborate* a Huginn-specific critique (Lu et al. 2507.02199) via a different
methodology. **Update 2026-07-24 (D14)**: the phenomenon H1/H2 are about is
now confirmed real and detectable on this project's own pipeline — reproducing
Blayney et al.'s "Long Persona" condition found real loops (0.1286% per-token,
matching their 0.14%) for the first time outside the artificially-starved
`forceloop.csv` sweep, some spanning multiple full turns. The negatives above
are not a pipeline failure to see loops; they're a base-rate/design problem
(answer-token-only tracing, short synthetic prompts) that this result helps
pin down, not a reason to doubt the phenomenon exists.

**The plan** is one keystone GPU pass ("the efficient batch": all-token states
+ per-layer Q/K + per-step logits, saved once) that unblocks a fan of no-GPU
analyses, then a small number of GPU escalations (joint spectral radius on
real Huginn, QK-alignment), then writeup — total **~6–10 T4 GPU-hours ≈ one
Kaggle week**. The primary deliverable is reframed as **a rigorous audit**:
"here is what latent-trajectory geometry does and does not reveal about
reasoning in a recurrent-depth transformer, with the null models, power
analysis, and curator-aligned probes (QK, spectral radius, RTD) that prior
positive-looking claims lacked." A confirmed positive (QK routing predicts
shape/consistency; count decodable where readout fails) is upside on top.

---

## 1. Hypotheses, goals, and their exact form (from the proposal)

The antecedents and quantifiers matter; most of the project's difficulty comes
from ignoring them.

- **H1 (A few shapes).** Every token's path is settle / loop / drift,
  *distinguishable by three instruments*: Lyapunov exponent λ, self-return,
  persistent-homology signature. → the claim is that **all three agree** and
  separate the regimes.
- **H2 (Loops track depth).** **When the path loops**, winding number grows
  with required reasoning steps. → **conditional on looping**; a winding number
  on a *settling* path is not a measurement of the object H2 is about.
- **H3 (Why looping is necessary).** Forced strict contraction (ρ(∂ₕR)<1) ⇒
  cannot hold a running count ⇒ state-tasks must loop/drift.
- **G1** = instrument Huginn to record **each token's** path at **every depth
  t≤r**; chart which shapes appear and when (tests H1).
- **G2** = winding + loop-persistence vs depth (tests H2) **AND** whether the
  **query–key routing** relates to which shape arises, linking to
  logical-consistency scores — **Tulchinskii et al., EMNLP 2025, arXiv:2502.17017,
  co-authored by Barannikov** [verified from proposal refs]. QK is the
  proposal's *central mechanistic bet*: "within each repetition the only step
  that introduces fresh cross-token information is attention… the feed-forward
  layers then act position-wise" — confirmed true in Huginn's source
  (attention couples positions; `GatedMLP` is position-wise) [grounded].
- **G3** = state+prove the contraction theorem (done, `files/contraction_proof.md`,
  algebra re-verified [verified]) + confirm on a synthetic state-tracking task.
  Operationally: measure **spectral radius ρ(∂ₕR)** on Huginn.

---

## 2. Proposal-vs-actual gap matrix (the spine)

| Goal / item | Demanded | Done | The shortfall |
|---|---|---|---|
| **G1** each token, every depth | all positions | answer token (`index -1`) only; full-depth ✓ | **answer-token-only** — the single largest design flaw (§10.1) |
| **G2a** winding vs depth | on controllable tasks | measured; **null/negative** | tested on settling paths → not an H2 test (§10.2) |
| **G2b** QK-alignment probe | curator's own method | **0% — one code comment, no Q/K ever hooked** | the highest-curator-alignment gap (§8.2) |
| **G3a** contraction theorem | prove | proved + re-verified | — |
| **G3b** spectral radius ρ on Huginn | measure | **only on a 64-D toy model, never on Huginn** | the number the whole H3 story presupposes (§8.3) |
| Datasets | ProntoQA-OOD, PARARULE-Plus, MultiLogicEval (d≤6), GSM8K, synthetic w/ Lean/SAT labels | PARARULE d2–5 + synthetic (Python labels) | 3 of 4 reasoning datasets absent; PARARULE N=4 (significance structurally unreachable); no Lean/SAT labels |
| Metrics | Lyapunov, spectral radius, PH-H1, winding, depth-vs-winding fit, accuracy-vs-depth | winding ✓; PH-H1 degenerate; depth-vs-winding null; **λ and ρ never measured on Huginn**; accuracy measured vs task-length not vs recurrence-depth | see §9 |
| Baselines | KL-exit, second-diff exit, Yang, Movahedi | **0 of 4 run** | KL/second-diff runnable; Yang/Movahedi are training-time, inference-incompatible |

Per-hypothesis verdict [grounded + verified]:
- **H1** — covered but under-instrumented: regimes exist (force-loop B4, Fisher
  p=0.01) but λ is unmeasured, PH is ≈0 by construction, and the dominant
  "settle" is architecturally guaranteed (Geiping path-independence). The
  Fisher p=0.01 "cleanest positive result" **has no reproducing code** —
  `run_forceloop.py` computes no significance test [grounded]. Fix that first.
- **H2** — **not tested.** Its conditioning event (looping) essentially never
  occurs on the traced answer token; at Blayney's ~0.02% loop rate, the whole
  project pooled expects **≈0.3 loops** [grounded]. Every forced test is null
  (B1, C4) or a length/position artifact (D11, §9). Honest label: *untested,
  below detection threshold* — not refuted.
- **H3** — theorem correct; **applied claim architecture-dependent and, per the
  project's own toy result (A6), likely does not transfer to Huginn's
  context-reinjection topology**; ρ measurable inference-only but never done.

---

## 3. Strategic reframing — the negative/audit result is the primary thesis

This is a decision the project has been avoiding and the plan makes explicit
(adversarial completeness critic #2 [grounded]).

Given the base rates — Blayney ~0.02% looping, D11 winding tracks length not
content, D12 Huginn does not solve counting — **the most probable outcome of
even the improved experiments is more well-characterized negatives.** The
strongest, most defensible, most curator-aligned framing is therefore **not**
"geometry reveals reasoning" but:

> **A rigorous audit of what latent-trajectory geometry does and does not
> reveal in a recurrent-depth transformer** — with the power analysis,
> null models, all-position measurement, and curator-aligned probes (QK
> alignment, spectral radius, RTD) that prior positive-looking winding claims
> lacked; and an independent, methodologically-different confirmation of
> Lu et al. (2507.02199).

Why this is strong, not a consolation prize:
- **Independent corroboration.** Lu et al. (Huginn-specific, logit/coda lens)
  found "no clear temporal separation or structured latent reasoning pathway
  across recurrence steps" [verified from critique doc]. This project reaches
  the same place via *geometry/topology* — convergent evidence from an
  orthogonal method is a real contribution.
- **Methodology as contribution.** The claims-ledger discipline, the D10
  length-confound diagnosis, the winding null model, the FDR correction, the
  σ_max-vs-ρ precision — catching and fixing over-claims (including a real
  fabrication and a coda-skip bug mid-project) is a citable audit-methodology
  result.
- **Upside preserved.** If QK routing *does* predict shape/consistency, or the
  count *is* decodable where the readout fails, those are genuine positive
  discoveries layered on top — the plan pursues them, but does not bet the
  paper on them.

Pre-commit this fallback thesis now so it is a first-class deliverable, not a
scramble at the deadline.

---

## 4. The dependency spine: hypothesis → requirement → confound → control → statistic → interpretation

This is the structure the project has been missing — "what stems from what."
Read each row as: *to license a claim about the hypothesis, you need this
measurement, which is threatened by this confound, which is neutralized by this
control, reported with this statistic, and only then does a positive/negative
mean what it seems to.*

**H1 — shapes exist and separate.**
- Requires: shape classification at **all token positions**, with all three
  discriminators (λ, self-return, PH).
- Confounds: (a) answer-token-only → measures the most-settling position;
  (b) PCA manufactures apparent rotation from a high-D random walk;
  (c) single-curve PH ≈ 0 by construction; (d) "settle" is architecturally
  guaranteed so its dominance is not a discovery.
- Controls: all-position extraction (§5); `winding_null_test` vs matched
  random walk; population-cloud / delay-embedding PH (not single-curve);
  report only the *enriched* (question-token / starved-budget) condition as
  evidential.
- Statistic: cluster separability in (λ, self-return, PH) space (silhouette /
  GMM-BIC); cross-check that the three discriminators *agree*; BH-FDR.
- Interpretation: CONFIRM iff ≈3 separable clusters **and** the three
  discriminators agree **and** loops survive the null test. REFUTE/SCOPE iff a
  continuum, or the discriminators disagree, or only settle exists even when
  enriched.

**H2 — winding ∝ depth, conditional on looping.**
- Requires: (1) *induce* loops; (2) among genuine loopers only, winding vs
  reasoning depth over ≥~10 depth levels.
- Confounds: (a) the antecedent (looping) never satisfied; (b) length ≡
  difficulty by construction (D10); (c) **even three_scale's "decoupling" is a
  prefix-block length/position artifact** (§9, the deep one); (d) sign-arbitrary
  |winding| on a per-trajectory PCA basis discards H2's *directional* "grows"
  claim; (e) ~50 uncorrected tests.
- Controls: starved-budget + question/digit tokens to raise loop rate;
  winding_null_test as the loop filter; a **constant-total-length,
  constant-answer-position** task redesign (§9); a **shared global projection**
  (Geiping-style) for signed winding; BH-FDR; per-level canonical statistic.
- Statistic: `spearman_by_level` (not per-row) + multivariate rank control;
  power calc *conditional on genuine-loop rate* (§7).
- Interpretation: CONFIRM iff winding rises with depth **within the looping
  subset**, survives length control **and** the null test **and** FDR. REFUTE
  iff flat/length-tracking within loops. **UNTESTABLE** (a legitimate terminal
  outcome) iff no genuine loops arise even when enriched — must be stated as
  such, never dressed as a null.

**H3 — contraction destroys state.**
- Requires: (1) measure whether Huginn actually contracts (joint σ_max of ∂ₕR);
  (2) whether the running count survives (linear decodability under contraction).
- Confounds: (a) σ_max is a **joint** [S·E]² quantity — **not** per-token
  (§10.3); (b) operator-norm σ_max ≠ spectral radius ρ; (c) count "failure"
  (D12) partly a multi-digit measurement artifact (§9); (d) a step-index probe
  is trivially high (state settles monotonically).
- Controls: report **joint** σ_max only, labeled as operator norm;
  capped single-token answers; held-out probe R² beating a length-only
  baseline; a shuffled-dynamics control for any progress probe.
- Statistic: σ_max distribution across sampled iterates; held-out ridge-probe
  R² vs length, conditioned on the σ_max regime; logistic task-success ~ σ_max
  + probe-R² + depth.
- Interpretation: CONFIRM applied-H3 iff (σ_max<1 ∧ count-undecodable ∧ fails).
  **SCOPE-OUT** applied-H3 iff (σ_max<1 **yet** count decodable) — the A6 toy
  prediction, **now derived precisely, not just asserted (2026-07-24, re-read
  `contraction_proof.md` directly)**: the theorem's own map is
  `h_{t+1} = R_θ(h_t; e)`, with `e` an explicit, FIXED argument, never
  iterated — §3 of that doc proves `d(h_t,h_t') ≤ c^t d(h_0,h_0')` for two
  DIFFERENT initial states `h_0 ≠ h_0'` at the SAME `e`; it says nothing
  about `e`-dependence, and the fixed point `h*(e)` (§2, Banach) is itself a
  function of `e` — nothing in the proof constrains how rich `e ↦ h*(e)` can
  be. §5's jump from "h_0-difference decays" to "count states S_0..S_{N-1}
  collapse" is only valid if the count is encoded the way `h_0` is —
  i.e. as an evolving, accumulated, streamed state (the classic-RNN
  picture the theorem's own worked example assumes). **For this project's
  actual counting tasks, it isn't**: `e` is the *entire* prompt (all
  tokens, reinjected via input-injection every unroll — Huginn's own
  `adapter(cat[x, input_embeds])`), so "how many ones are in the sequence"
  is already fully present in `e` at t=0, readable by a static function of
  `e`, with zero need to iteratively accumulate it over recurrence steps.
  Contraction kills genuinely dynamical/streamed information; it says
  nothing about static functions of the always-visible context. This is
  exactly why `src/h3_results`'s own toy sweep (A6) shows the
  recurrent-over-**depth** model (full context reinjected every step,
  matching Huginn) keeps count-R²≥0.996 under strong contraction while the
  recurrent-over-**time** model (one token per step, no reinjection —
  genuinely streaming) collapses to R²≈0. **Sharper implication than
  previously stated**: if this holds, H3's contraction mechanism does not
  even *predict* a counting failure on Huginn's actual counting tasks —
  so this project's own empirical finding that Huginn fails to count
  (`counting_accuracy.csv`, 0% at n_ops>=8) needs a *different* causal
  story (training-distribution mismatch, insufficient capacity in the
  learned `e -> h*(e)` readout, task novelty) — it should **not** be
  narrated as "confirms the contraction bottleneck," since the theorem,
  applied honestly to this architecture, doesn't forbid counting here in
  the first place. Either way the proven theorem stands; only its *Huginn
  applicability*, and the causal story for the counting failure, are at
  stake.

---

## 5. The keystone — one "efficient batch" extraction pass

The single most important engineering fact [grounded, verified from `hook.py`
and `raven_modeling_minimal.py`]: **GPU time is dominated by the unrolls, and
the model already computes every token at every layer.** The hook already
stores the full `[1, n_tokens, 5280]` tensor per unroll (`hook.py:122`); only
the *return* (`hook.py:147`) throws all tokens but one away. Therefore:

- **All-token capture is a return-statement change with zero extra GPU** —
  memory grows ×n_tokens (~68 MB/prompt at 50 tokens, fp32).
- Hooking `core_block[i].attn.Wqkv` for **Q/K** (G2), reconstructing per-step
  **logits** via the already-validated `_replicate_coda_head` tail, and
  evaluating **σ_max** at settled states all add *negligible* marginal GPU —
  the unrolls are paid once.

So **one instrumented pass (~0.3–1.5 GPU-hr per condition-set) produces the
substrate for almost every analysis**, then everything downstream is offline.
This collapses G1 + the QK substrate + the logit substrate + most of the
rigor work into one Kaggle run.

Concrete deliverable: upgrade `extract_trajectory` to optionally return/persist,
per prompt, a self-describing record:
- `latents [num_steps, n_tokens, 5280]` (all positions);
- `qk` — reduced per-(layer, head, step) alignment scalars (see §8.2 for why
  raw Q/K cannot be persisted: ~13 GB/prompt);
- `logits` — top-k + answer-relevant, per step;
- wired through `Trajectory.save` → `.npz` (today `types.py` has **no live
  caller**; this makes it the backbone and gives `results/trajectories/*.npy`
  a reproducible producer at last).

Guardrails the adversarial pass flagged [grounded]:
- **Persist a *sampled* subset of full `[T,H]` paths, not every position of
  every prompt** — all-token × all-prompt ≈ 40 GB exceeds Kaggle's ~20 GB
  output cap (M7). Keep reduced summaries for all; keep full paths for a
  representative sample (enough for the null test and probes).
- **The QK statistic must be locked *before* this pass** (reduced online), so
  reading Tulchinskii 2502.17017 gates the GPU run, not parallel to it (M4).

---

## 6. The plan, phased and dependency-ordered

Compute tags: **[0-GPU]**, **[GPU: n hr]**. Curator-decision points marked **[C]**.

### Phase 0 — no-GPU rigor rescue (do first; gates the rest)
0.1 **Power analysis + pre-registration** [0-GPU]. Compute E[loops]=p·N per
   *condition* (answer-token/full-budget ≈ 0.3 loops project-wide; question-token
   + system prompt ≈ up to 2.8%), the N needed to expect ≥5 genuine-winding
   loops, and a Monte-Carlo Spearman power curve. Freeze the H1/H2/H3 test
   families for FDR. **This converts "we found no loops" into the defensible
   "the sampled condition is ~100× underpowered," and forbids citing any
   answer-token null as evidence against H2.** [C] sign-off on "≥5 loops" bar.
0.2 **DONE 2026-07-24 — Apply `benjamini_hochberg` to the existing ~50
   correlations** [0-GPU]. `scripts/run_fdr_correction.py`, 46 tests across 9
   experiments, project-wide and per-experiment families (agree here: 20/46
   survive either way). **Prediction corrected by the actual run**: the two
   prime-target hits (maxtask winding~n_ops +0.943, dissociation local
   winding~n_ops −0.943) both *survive* FDR (q=0.0123) — they are not FDR
   casualties. They're still not trustworthy, for reasons FDR doesn't touch:
   maxtask's n_ops is D10-degenerate (rank-corr(n_ops,seq_len)=1.0, so
   "winding~n_ops" ≡ "winding~seq_len"); dissociation's local −0.943 fails to
   replicate at 3× the seeds (15-seed: rho=−0.543, p=0.27, an actual
   casualty there). Only 2/22 raw-significant tests are pure FDR casualties
   (both `dissociation_5seed`'s marginal steps_settle~n_ops per-level,
   p=0.0499). See `claims_ledger.md` D13.
0.3 **PARTLY DONE — Retire the degenerate length-partials** [0-GPU]. Codified
   that counting/switch/maxtask/count_ones have rank-corr(n_ops, seq_len)=1.0
   and *cannot* be length-decoupled by construction (D10); the guard already
   raises for these, so none of their length-controlled numbers get reported
   as real. **DONE 2026-07-24: tightened the `partial_spearman` guard
   threshold 0.999 -> 0.95** — the old threshold missed dissociation's
   real 0.9895 collinearity (a landmine, no live caller yet). Not yet done:
   an explicit sweep to confirm no *other* uncaught degenerate confounder
   exists outside the four already-known tasks.
0.4 **Re-analyse three_scale properly, and correct D11** [0-GPU, partly done
   this session]. The shipped `run_three_scale.py` reports only per-row
   `spearman()`. The correct multivariate rank control (all three length scales
   simultaneously) gives [verified by me]: active_len β=+0.130 **p=0.056**,
   neutral_len −0.023 p=0.74, irrelevant_len −0.488 **p=6.9e-11**. The
   single-confounder "flip to +0.318" a subagent flagged is an artifact of
   controlling the *composite* seq_len — it does not survive the correct
   analysis. **But even this is not a clean H2 test** (§9, the prefix confound):
   because `irrelevant_len` is a prefix block, irrelevant_len's effect *is* a
   total-length / answer-position effect. Honest D11: **winding is a
   length/position artifact, not a reasoning-content signal.** Update the ledger.
0.5 **DONE 2026-07-24 — re-measured correctness without the single-token
   trap, and the result substantially revises D12.** `run_v6_correctness_probe.py`
   now separates single-token answers (0-9, honestly verifiable, 13/24 rows)
   from multi-token ones (negative/>=10, 11/24 rows, leading-token-only) and
   records `topk_correct_at_step` (top-5) alongside strict argmax. Real
   Kaggle rerun: **top-5 hit rate on the single-token subset is 13/13
   (100%)**, almost always by unroll step 1; strict argmax finds it only
   4/13 (all `target=0`). The model reliably shortlists the correct count in
   its top-5 candidates — it just usually loses the final argmax pick to a
   generic small-number bias. "Small-number prior, doesn't count" (the old
   D12 reading) was too strong; see `claims_ledger.md` D12's 2026-07-24
   update for the full, more accurate picture. N=13 still small, not a
   powered rate estimate, but 13/13 is a strong signal regardless.

### Phase 1 — the keystone extraction [GPU: ~1–2 hr]
1.1 Implement the §5 efficient-batch pass (all-token latents + Q/K + logits +
   `Trajectory.save`, sampled). Run over: three_scale (redesigned, §9 —
   `make_three_scale_modk_task` now exists, ready to use),
   PARARULE (extend loader to d≤6 [C]), a **starved-budget set (num_steps≈16)**
   — the only in-project loop-inducing lever tried so far — and
   **question/digit token positions**, not just the answer token. This one
   pass is the substrate for Phases 2–3.

   **Concrete build order (2026-07-24, expanded from the §5 sketch into
   actionable substeps — not yet implemented, this is the design).** Split
   into what's ready to build now vs. genuinely [C]-blocked, so the
   not-blocked parts don't sit idle waiting on a curator answer:

   a. **All-token latents [ready, 0 new risk].** `diag_blayney_repro.py`'s
      `_extract_all_positions` already prototypes this (mirrors `hook.py`'s
      own `return_logits=False` hook, which already captures every
      position — only the return statement slices). Promoting this into
      `hook.py` itself as a real `extract_trajectory(..., token_index=None)`
      mode (returning `[num_steps, n_tokens, 5280]`) is mechanical: copy the
      prototype's hook, keep the existing single-token path as the default
      for backward compat with every script that calls it today.
   b. **Logits, all positions [ready].** `_replicate_coda_head` already
      takes a `[1, n_tokens, hidden]` state and returns `[1, n_tokens,
      vocab]` logits — the existing per-step reconstruction already works
      for every position, `hook.py`'s current code just slices to
      `token_index` afterward (line ~146). Persisting top-k (not full
      vocab) per position per step keeps this cheap: `[num_steps, n_tokens,
      k]` instead of `[num_steps, n_tokens, vocab]` (a ~6000x reduction at
      k=5, vocab~=30k).
   c. **Q/K hooking mechanics [ready, the recipe is NOT the same as the
      search protocol].** Hook `core_block[i].attn.Wqkv` for each of the 4
      layers (Huginn-0125 is `(2,4,2)_I`: 2 prelude, 4 core_block, 2 coda,
      per Blayney et al. Table 1, read in full this session) — a SECOND
      hook per layer, alongside the existing `core_block[-1]` state hook.
      `freqs_cis` is a forward arg, not a module attribute, so it must be
      captured via a hook on a parent module or passed through explicitly
      (already flagged as a real gotcha, not newly discovered). Split the
      fused QKV output by `self.chunks`, apply `qk_bias` (config confirmed
      ON), then RoPE via `apply_rotary_emb_complex_like` — this exact
      sequence was verified against the real source this session (§8.2),
      not guessed. **What this step does NOT decide**: which (layer, head,
      unroll-step) to actually use for Tulchinskii's `S_QK` statistic —
      that's 1.1d below, and it's [C]-blocked.
   d. **QK statistic + search protocol [C]-blocked, do not implement past
      the hook itself without curator sign-off].** Tulchinskii's exact
      formula is now known (§8.2): `S_QK^(l,h) = q_{a_i}^(l,h) · k_s^(l,h)`,
      one `(layer, head)` selected via a 600-example calibration split
      (300/300 per class). For Huginn's weight-tied recurrence this becomes
      a `(layer, head, unroll-step)` search — 4 x 55 x 32 = 7,040
      candidates (matches the "~7,000 tests" figure already used
      project-wide, e.g. §9's FDR-family note). **Persist reduced scalars,
      not raw Q/K**: for a small, fixed set of (query-position,
      key-position) pairs of interest (e.g. query at each candidate answer
      token, key at the last question/statement token), compute and store
      the dot product directly during the extraction pass, at every
      `(layer, head, step)` — this is what makes the ~13 GB/prompt raw-Q/K
      problem (§8.2) go away: a `[4, 55, 32]` float array per
      query/key-position pair (~35 KB/prompt) instead of raw `[4, 55, 32,
      96]` tensors. **Still needs from the curator**: the consistency-
      labeled dataset (now known to just be ProntoQA-OOD/PARARULE
      Plus/Multi-LogiEval directly, all public — §8.2's RESOLVED note — so
      this is a scope/format decision, not a data-access blocker), and
      explicit confirmation that searching over unroll-step as a third axis
      is an acceptable port of the original 2D (layer,head) recipe.
   e. **`Trajectory.save` wiring [ready].** `types.py`'s `Trajectory`
      dataclass already has a working `.save()`/`.load()` `.npz` contract
      with zero live callers — wire steps (a)+(b) through it directly,
      giving `results/trajectories/*.npz` (not the orphaned legacy `.npy`
      files, see architecture_state.md) a real, reproducible producer for
      the first time.
   f. **Storage guardrail [ready, already specified in §5]**: full `[T,H]`
      paths for a *sampled* subset only (representative, enough for
      `winding_null_test`/probes), reduced summaries (winding/shape/
      steps_settle, already the existing convention) for every prompt.

   Net: (a), (b), (e), (f) and the Q/K *hooking mechanics* in (c) can all be
   built and tested (on a toy prompt, 0 GPU cost beyond a smoke test) before
   any curator conversation happens. Only (d)'s actual *search* — which
   (layer,head,step) triples get scored and reported as "the" QK signal —
   waits on [C].
1.2 **DONE 2026-07-24 — reproduced Blayney et al.'s exact loop-inducing
   condition, pipeline confirmed working, real loops observed for the
   first time.** `scripts/diag_blayney_repro.py`, `results/blayney_repro.csv`
   (Kaggle T4, 24 GSM8K examples x 2 conditions, all token positions).
   `long_persona`: 7/5445 loops (0.1286%), matching Blayney's own 0.14%
   per-token rate closely. `no_system_prompt`: 1/1677 (0.0596%) vs their
   0.02% baseline (N too small to pin down precisely, not inconsistent).
   **This project has now directly observed and quantified real winding
   loops for the first time ever outside the artificially-starved
   forceloop.csv condition** — several with |winding| well over 1 full
   turn (up to 7.18), unlike forceloop's own |winding|~=0.65 loops. See
   `claims_ledger.md` D14. Confirms the extraction pipeline is not the
   reason H1/H2 tests keep coming back null on this project's own
   answer-token-only synthetic tasks — the phenomenon is real and
   detectable, just very rare, exactly as the literature predicts.

### Phase 2 — no-GPU analyses on the substrate
2.1 **`winding_null_test` on every real winding** [0-GPU]. Adjudicate whether
   any winding — including the 13 force-loop "loops" (|winding|≈0.65, *under one
   full turn*) — beats a matched-random-walk-through-PCA null. Caveat (M5): the
   force-loop paths are only ~12 points; treat that adjudication as suggestive,
   run the definitive test on longer trajectories.
2.2 **Persistent homology done right** [0-GPU]. Replace single-curve H1 (≈0 by
   construction) with (i) delay-embedding (Takens/Perea–Harer periodicity) and
   (ii) population point clouds — each vs a surrogate null. This is **the
   curator's own TDA specialty**; consider RTD (Barannikov et al. 2201.00058)
   and persistence landscapes, which the team's own `theoretical_framework.md`
   names as *better-supported than winding* for exactly this (§14).
2.3 **All-token shape census** [0-GPU]. Chart shape-fraction by position class
   (question/digit/content/whitespace/answer) and by budget → satisfies G1;
   tests the literature's prediction that loops live on non-answer tokens.
2.4 **Linear count / state probe on saved states** [0-GPU]. Ridge probe
   `h_t(position) → running count`, held-out R², beating a length-only baseline,
   swept over positions and layers. **Interpretation is subtle (M2): a positive
   here SCOPES-OUT applied-H3** (count decodable under contraction = A6 on real
   Huginn), it does not support it. **Do a step-index probe only with a
   shuffled-dynamics control** (a settling state trivially encodes t) (L3).
2.5 **Native exit-criteria vs correctness** [0-GPU for the 2 geometric
   criteria; logits ride Phase 1]. If a *geometric* criterion (latent-diff,
   cosine) predicts correctness, that is **direct evidence geometry carries
   usable information** — a far stronger claim than any length-confounded
   correlation. Clean negative is citable.
2.6 **[DEEP] Geometry-vs-activations discriminator test** [0-GPU]. The
   headline thesis is "geometry reveals what's computed," but a linear probe on
   the 5280-d state proves the *state* holds the count, not that the *geometric
   summary* (λ, winding, shape, self-return) is informative. **Test that
   geometric features *alone* predict task identity / operation / correctness**
   above chance and above the raw-state probe's leakage. Without this the
   project proves things about activations while claiming things about geometry.
   (Adversarial completeness #9 — the deepest conceptual hole.)
2.7 **[DEEP] Path-independence null on the geometry itself** [rides Phase 1].
   Huginn is *designed* so random h₀ → the same fixed point. If a path's
   shape/winding is reproducible across init seeds, it may be an
   architecture/init artifact, not a computation signal. Measure same-prompt
   multi-init geometric reproducibility as a validity null. (Completeness #8.)
   **Confirmed 2026-07-24 as a live concern, not hypothetical, by reading
   Geiping et al. directly (§13)**: their own paper states "the same
   orbital patterns, fixed points, or directional drifts emerge regardless
   of initialization" — this project's own `path_independence` metric
   already exists (`dissoc_multiinit.csv`'s `lyap` column) but has never
   been read as a validity null on the *shape itself*, only reported as a
   secondary stat. Raises the credence that this test matters, not just
   that it's thorough to include.

### Phase 3 — GPU escalations
3.1 **Joint spectral radius / operator norm on real Huginn** [GPU: ~3–6 hr].
   Point the shape-fixed `spectral.py` at `core_block_forward`, evaluate at
   sampled iterates along saved trajectories. **Report the JOINT σ_max only**
   (the per-token version does not exist, §10.3), labeled *operator norm*, and
   the along-path finite-time contraction. Closes G3b. Requires: fix
   `spectral.py`'s 1-D tangent to `randn_like` + a new 2-D known-answer test
   (M6 — it does *not* run on Huginn's `[S,5280]` state as-is); force SDPA/eager
   + fp32 (JVP through flex_attention may be unsupported). **[C]/read** Yang
   2605.26733 before attributing the method to Yang (`spectral.py` cites Miyato).
3.2 **QK-alignment probe** [rides Phase 1 GPU; analysis 0-GPU]. §8.2 — the
   curator's method, half of G2. Treat as *research on weight-tied recurrence*,
   not a reimplementation (§9).
3.3 **Spectral → state-tracking link** [GPU: ~0.5–1 hr]. Accuracy vs
   **recurrence depth** (the proposal's axis; current data is vs task-length,
   the wrong axis) paired with σ_max and probe-decodability → the H3 test.
3.4 **Activation steering** [GPU: ~0.5–2 hr] — *only if 2.4 finds a decodable
   direction*. Causal upgrade; compare the probed direction vs random/shuffled.

### Phase 4 — baselines the proposal names [GPU: ~0.5 hr]
4.1 Run **KL-exit** [Geiping] and **second-difference exit** head-to-head vs the
   geometric criteria (does geometry beat cheap output-space heuristics?).
   Position **Yang / Movahedi honestly as training-time, inference-incompatible**
   context — cited, not run. (Completeness #5.)

### Phase 5 — writeup + figures [0-GPU, non-negotiable]
5.1 **Implement `analysis/plots.py`** (currently a stub — *no design can
   produce a single figure until this exists*), freeze a figure list, reserve
   calendar time. Treat the paper — not the last experiment — as the
   deliverable. (Completeness #1, the most existential gap.)

**Total GPU ≈ 6–10 T4-hours ≈ one Kaggle week**, DataSphere's 38.6 one-time
hours held in reserve for a confirmatory spectral run only. Compute ledger §11.

---

## 7. Power & the "untestable" outcome (why this gates everything)

H2's phenomenon may be **undetectable at any affordable N** in the sampled
condition, and that is a *result*, not a failure. At Blayney's ~0.02%, the
whole project pooled (1,294 answer-token trajectories, counted live) expects
**≈0.26 loops**; even their own "Long Persona" system-prompt condition's
per-token rate (**0.14%**, Table 3 — NOT the 2.81% figure, which is a
different per-example statistic, see §13 and `power_and_preregistration.md`)
expects only ≈1.8, and "non-fixed-point" ⊋ "≥1 full winding turn" — the only
real loops in-project carry |winding|≈0.65, under one turn [grounded]. See
`power_and_preregistration.md` (2026-07-24) for the full power calc,
including the N-needed-for-≥5-loops table. So the plan **must**:
(a) do the power calc *conditional on the genuine-winding-loop rate* before any
GPU spend on H2; (b) report "H2 untestable-because-underpowered on Huginn" as a
legitimate terminal outcome; (c) never treat an underpowered null as a
refutation — the single most likely reviewer-fatal error.

---

## 8. The three curator goals, explicitly (G1/G2/G3)

### 8.1 G1 — all-token, all-depth (the answer-token flaw)
Answer-token-only is *arguably the central design flaw* [grounded]: it
instruments the position where settling is strongest by design, where Geiping
reportedly observes *no* orbits (they are on question/digit tokens), and where
the project's own cross-branch split shows the depth signal is null (answer
token) vs. present (content tokens). Fix = §5 keystone (a return-statement
change; zero extra GPU). No current result should be read as characterizing
"Huginn's geometry" — only *the answer token's* geometry.

### 8.2 G2b — the QK-alignment probe (curator's own method, 0% done)
Q/K **are** recoverable: hook `core_block[i].attn.Wqkv`, split by `self.chunks`
→ `[B,S,55,96]` (55 heads, full MHA), **apply `qk_bias` (config has it ON)
*then* RoPE** via `apply_rotary_emb_complex_like` to match what attention
consumes [grounded]. But treat this as **research on weight-tied recurrence,
not a reimplementation** (adversarial H5 [grounded]):

**RESOLVED 2026-07-24 — read Tulchinskii et al. 2502.17017 directly (was in
the uncertainty register, §13).** The exact statistic:
`S_QK^(l,h)(c,s,a_i) = q_{a_i}^(l,h) · k_s^(l,h)` — a plain dot product
between the query vector at the *candidate answer token* `a_i` (`"true"` or
`"false"`) and the key vector at the token marking *the end of the
statement* `s`, at one specific (layer, head). Prediction = whichever
candidate (`true`/`false`) gets the higher score; **no threshold, no
learned classifier, no multiple continuations — one forward pass**. The
(layer, head) is not fixed a priori: it's **selected per task-setup on a
600-example calibration set (300 true / 300 false)**, then evaluated on a
held-out test set — i.e. head selection is itself part of the method, not
a hyperparameter to guess. **Datasets used are exactly the ones this
project already partially uses**: ProntoQA-OOD, PARARULE Plus, and
Extended-Multi-LogiEval (1,000+ samples) — all three named in the original
proposal (§2's gap matrix), of which this project currently runs only
PARARULE at d2-5. Reported accuracy: ~88-98% on ProntoQA-OOD Modus-Ponens
(depths 1-5) vs. ~61-67% baseline; weaker cross-dataset transfer (only
3/5 selected heads beat baseline by >10% on PARARULE Plus when the head
was chosen elsewhere) — **the method does not trivially transfer across
task distributions, so a head selected on one dataset should not be
assumed valid on another.** **Confirmed: the paper never discusses
recurrent, looped, or weight-tied architectures** — "all experiments were
performed with frozen pre-trained LLMs" (1.5B-70B, standard transformers).
The port to Huginn's weight-tied recurrence is genuinely novel, not an
application of an existing recipe — this independently confirms the
adversarial concern below, now grounded rather than hypothesized.

- **Weight-tying breaks "consistency heads":** the same 4 layers × 55 heads
  iterate 32× — a "head" now exists at 32 depths; which (layer, head, depth) is
  "the" consistency head is a *design decision*, not a port. **Now sharper**:
  the original method's head-selection step (calibrate per-setup on 600
  labeled examples) still applies — it just needs to search over
  `(layer, head, depth)` triples instead of `(layer, head)` pairs, and the
  calibration set requirement is now precisely known (>=300/300 per class,
  matching the original paper's scale).
- **`freqs_cis` is a forward arg, not on the module** → a second hook is needed;
  the probe is not "free plumbing."
- **~7,000 tests (4×55×32) with honest held-out head selection will most likely
  yield zero FDR survivors** — pre-register the likely null. Calibration-set
  selection (as the original paper does) is a legitimate alternative to
  raw FDR correction across all 7,000 — select on a held-out calibration
  split first, then test only the winning head on a separate test split,
  matching Tulchinskii's own protocol rather than inventing a new one.
- **"Logical-consistency scores" do not exist in-project as true/false
  labels** — Tulchinskii's own labels come directly from the three named
  benchmarks (ProntoQA-OOD, PARARULE Plus, Extended-Multi-LogiEval), not a
  separately-annotated "consistency" dataset — so the label-sourcing
  problem is smaller than previously framed: it is "extend the loader to
  the other two named datasets," not "invent a new labeling scheme."
  **[C] the curator must still confirm the exact `(layer, head, depth)`
  search protocol and sign off on porting a same-token-position dot
  product (query at the answer token, key at the statement-end token) to
  Huginn's shared-weight, per-unroll representation** *before* the GPU pass
  (M4 gates it). If no consistency dataset can be committed, keep only
  "QK-alignment vs shape" and drop the consistency link.
Still: this is the highest-curator-alignment deliverable and nearly free on the
extraction pass — **promote it to a standalone must-do** regardless of which
overall lens is chosen (completeness #7).

### 8.3 G3b — spectral radius on real Huginn
Never measured on Huginn (only a 64-D toy) [grounded]. `spectral.py` is
self-tested but **only on 1-D maps and measures σ_max (operator norm), not ρ
(spectral radius)** — a *defensible, stronger* choice for the contraction claim
(σ_max<1 ⟹ ρ<1, which is what Banach needs), but it must be *labeled* as
operator norm, not printed as "ρ(∂ₕR)."

**RESOLVED 2026-07-24 — read Yang et al. 2605.26733 directly (was in the
uncertainty register, §13): title "Stabilizing Recurrent Dynamics for
Test-Time Scalable Latent Reasoning in Looped Language Models" (STARS
method). The code's Miyato citation is correct; the README's Yang credit
is wrong, and it's not just an attribution slip — they compute genuinely
different quantities via genuinely different algorithms.** Yang's JSRR
does power iteration **directly on J**: `v <- Jv/‖Jv‖`, converging to J's
dominant eigenvector, estimating `ρ(J) ≈ ‖Jv‖₂` (Lyapunov-linearization
stability, matching H3's own `ρ(∂ₕR)` notation exactly). `spectral.py`
instead does power iteration **on JᵀJ** (via JVP-then-VJP composition),
converging to J's dominant *singular* vector — that's the classic Miyato
et al. 2018 spectral-normalization recipe, correctly self-cited, and it
estimates σ_max, not ρ. **These coincide only when J is normal (symmetric
in the real case); for a general nonlinear recurrent Jacobian there is no
reason to expect that, so `spectral.py`'s current estimator measures the
*wrong* quantity for a literal `ρ(∂ₕR)` claim** (it still correctly upper-
bounds ρ, since σ_max>=ρ always — the σ_max<1⟹ρ<1 contraction argument
above still holds, but a report of "ρ" specifically would need the
direct-J power iteration instead). Also: Yang's method is **training-time
only** (a regularization loss during SFT, `L_STARS = (1-λ)L_SFT + λ·L_JSRR`)
— confirms it's inference-incompatible as a baseline, already correctly
flagged (§2 gap matrix, "Yang/Movahedi are training-time"); they test on
Ouro-1.4B (a LoopLM, not Huginn), GSM8K, +4.01% peak.

Checkpoints before writeup, updated: (1) ~~read Yang 2605.26733~~ DONE —
`spectral.py`'s estimator and Yang's are confirmed different algorithms for
different quantities, not the same thing under two names; decide [C]
whether the paper report `σ_max` (already implemented, upper-bounds ρ) or
implement the cheaper direct-J power iteration to report actual `ρ`
(closer to what H3 literally states, ~1 extra JVP call, no VJP needed);
(2) fix the 1-D→`[S,5280]` tangent + re-test (M6); (3) report the **joint**
value only (§10.3), labeled correctly for whichever of the two is chosen.

---

## 9. Confounds & analytics register (the maths/stats layer)

The controls that must be in place, and the ones the project got wrong.

- **D10 (length ≡ difficulty).** Every synthetic task except three_scale has
  rank-corr(n_ops, seq_len)=1.0; length-partial is degenerate there. Guarded
  (raises) [verified]. Fix is task-design, not statistics.
- **The three_scale *prefix* confound [verified, the deep one].** `irrelevant_len`
  is a filler block placed *before* the sequence, so it lengthens total context
  **and pushes the answer token to a later absolute position**. The three scales
  are rank-orthogonal to *each other* but each is monotone in total seq_len and
  answer-token position. So D11's headline ("winding tracks irrelevant_len") is
  consistent with **winding tracking total length / absolute answer position** —
  exactly the artifact three_scale was built to defeat. **No three_scale number
  is an H2 test until the task is redesigned:** constant total token count,
  constant answer position, vary only the active/neutral *ratio*, single-token
  capped answer. [C] on the redesign. **DONE 2026-07-24 —
  `make_three_scale_modk_task`** (`src/traj_geom/shapes/synthetic.py`,
  6 tests) implements exactly this redesign: `neutral_len` derived so
  `total_len` (and therefore answer position) is constant by construction
  across an active_len/irrelevant_len sweep, all three symbol kinds
  shuffled into one sequence (no distinguished prefix region at all), and
  the answer is `active_len % modulus` — single-token for any active_len at
  any modulus 2..10 (generalises `make_switch_task`'s mod-2 parity to a
  full family; modulus itself is a new difficulty axis, distinguishing
  more residue classes needs more state per H3's own framing).
  **RUN 2026-07-24 on real Huginn** (`scripts/run_three_scale_modk.py`,
  Kaggle T4, 126 configs, all succeeded): `seq_len` confirmed exactly
  constant (42 tokens, std=0.0) across the whole sweep — the design's
  core guarantee held in practice. Result: **clean null**, no significant
  winding~active_len at either modulus tested (rho=+0.179 mod=2,
  rho=−0.036 mod=5, both n.s. at N=7). This is the project's first
  genuinely unconfounded H2 test, and it agrees with D11's negative
  reading on cleaner footing. **REPLICATED same day at N=15**
  (`--extended`, 270 configs, seq_len constant at 54): winding null holds
  up more decisively (Fisher-combined p=0.266, and at N=15 this project's
  own power curve gives real power to detect a moderate effect, so this
  is a genuine absence now, not just "still underpowered"). The
  unplanned steps_settle~active_len pattern flagged as "worth a
  follow-up" **replicated and got stronger, not weaker** (Fisher-combined
  p=0.0006 vs the N=7 run's p=0.015, both moduli individually significant
  at N=15). See `claims_ledger.md` D15 for both runs' exact numbers.
- **Joint-not-per-position σ_max [verified].** ∂h_{t+1}/∂h_t is one
  `[S·E]×[S·E]` causal Jacobian; σ_max is a single number for the whole state.
  "σ_max at question vs answer tokens" **does not exist** — drop that narrative;
  report the joint contraction factor as the H3 premise.
- **Multi-digit answers [grounded].** `make_count_ones`/`make_three_scale`/
  `make_counting` answers are uncapped (and counting can be negative); a
  first-token argmax sees only the leading digit/minus. Cap answers to a single
  token (0–9 or modular) *before* trusting any accuracy/probe-target claim.
- **"Lyapunov" is a convergence-rate proxy, not λ [grounded].** The implemented
  quantity is mean log step-norm ratio — it cannot distinguish a slow-settling
  path from a neutral loop. A true finite-time Lyapunov exponent needs the
  top-singular-value growth of the *composed* Jacobian (~T× the σ_max cost, not
  budgeted). Either budget it or demote H1 to two discriminators and concede λ
  is unmeasured.
- **Multiple comparisons.** ~48–50 uncorrected correlation tests project-wide;
  BH built, applied to nothing. Pre-register families — the family boundary
  *is* outcome-determining (pool the ~7,000 QK tests with the ~50 and all die).
- **Winding null model.** Built, never applied to real data; blocked until raw
  paths are saved (§5). Every winding number is currently un-null-tested.
- **Multivariate control absent.** `partial_spearman` handles one z; three
  simultaneous length scales need a rank/OLS multiple regression (I implemented
  a 10-line version this session — fold it into `correlate.py`).
- **Signed vs |winding|.** H2 claims winding *grows* (directional); the project
  reports |winding| on a per-trajectory PCA basis, discarding sign. Use a
  **shared global projection** (Geiping-style 6-D PCA) so sign is meaningful.

---

## 10. Deep problems / open holes (things that may not be solvable as framed)

1. **Answer-token blindness** (§8.1) — a re-run fixes it; cheap.
2. **H2 may be untestable on Huginn** (§7) — if genuine loops don't arise even
   when enriched, H2 is untestable, not refuted. Terminal-outcome, must be
   stated.
3. **Per-position spectral radius is undefined** (§9) — a narrative all three
   designs leaned on; unmeasurable. Only the joint value exists.
4. **Applied-H3 may be scoped out for Huginn's architecture** (§4 H3, derived
   precisely 2026-07-24 not just asserted) — the theorem's map
   `h_{t+1}=R_θ(h_t;e)` only proves *initial-state* (`h_0`) differences
   decay; `e` (the full reinjected prompt) is held fixed throughout the
   proof and is never shown to decay. Since this project's counting tasks
   make the count fully readable from `e` at t=0 (all tokens visible,
   reinjected every unroll), the theorem's mechanism doesn't bind on the
   count *by construction* here — it only would if the count were encoded
   the way `h_0` is (genuinely streamed/accumulated, one token at a time,
   no reinjection). A6's toy result is exactly this contrast: full-context
   reinjection (matching Huginn) keeps count-R²>=0.996 under strong
   contraction; a no-reinjection streaming counterpart collapses to
   R²~=0.00. **Consequence**: this project's own empirical counting
   failure (`counting_accuracy.csv`, 0% at n_ops>=8) should not be
   narrated as evidence *for* the contraction bottleneck — H3, applied
   honestly to an architecture with full-context reinjection, doesn't
   predict that failure in the first place. The real cause is a separate,
   still-open question (training-distribution mismatch, a learnability
   gap in the `e -> h*(e)` readout, task novelty) — confirming H3 on real
   Huginn (measuring σ_max, checking count-decodability) is still novel
   and worth doing, but it tests a narrower, more precisely scoped claim
   than "why Huginn can't count."
5. **Geometry vs activations** (§2.6) — the deepest conceptual hole: probing the
   raw state is standard activation probing, not evidence that the *geometric
   summary* is informative. Must be tested directly or the paper overclaims.
6. **Path-independence** (§2.7) — if geometry is reproducible across init, it may
   be an architecture artifact, not a computation signal.
7. **Lean/SAT labels unmet** — a curator-scope decision to make and defend, not
   silently skip.

---

## 11. Compute ledger

| Step | GPU | Platform |
|---|---|---|
| Phase 0 (all rigor rescue) | 0 | — |
| 1.1 keystone extraction (few conditions) | ~1–2 hr | Kaggle |
| 2.x offline analyses | 0 | — |
| 3.1 joint σ_max (real, could be 3–6 hr, M6/L6) | ~3–6 hr | Kaggle; DataSphere for a confirmatory 2nd run only |
| 3.2 QK (rides 1.1) | ~0 marginal | Kaggle |
| 3.3 accuracy-vs-depth | ~0.5–1 hr | Kaggle |
| 3.4 steering (conditional) | ~0.5–2 hr | Kaggle |
| 4.1 baselines | ~0.5 hr | Kaggle |
| **Total** | **~6–10 hr** | **≈ one Kaggle week (30 hr renewable)** |

Kaggle (renewable ~30 hr/week) is primary; DataSphere (~38.6 hr, one-time,
non-renewable) is the finite backup — reserve it for the one irreversible
confirmatory spectral run, not routine work.

---

## 12. Curator-decision points [C]

Consolidated — take these to Barannikov:
1. **QK statistic + a logical-consistency labeled dataset** (§8.2) — his own
   method; the exact head/position/aggregation recipe and the label source.
   Gates the QK GPU pass.
2. **three_scale / synthetic redesign** (§9) — constant-length, constant-answer-
   position, single-token answer; and whether to add ProntoQA-OOD / MultiLogicEval
   (for N≥6 depth levels) and GSM8K.
3. **Lean/SAT labels** — wire a checker, or de-scope with justification (§10.7).
4. **σ_max-vs-ρ reporting** (§8.3) — confirm reporting operator norm is
   acceptable for the H3 contraction claim.
5. **The "≥5 genuine loops" power bar** and pre-registered FDR families (§7).
6. **Primary-thesis decision** (§3) — commit the negative/audit framing as the
   paper's spine, with positives as upside.

---

## 13. Uncertainty register (verify before citing)

- **RESOLVED 2026-07-24 — Yang 2605.26733**, read directly. Confirmed: NOT
  the same estimator as `spectral.py`. Yang's JSRR does power iteration
  directly on J (estimates ρ, training-time-only regularizer); `spectral.py`
  does power iteration on JᵀJ (estimates σ_max) — correctly self-cited to
  Miyato 2018, not Yang. Full detail in §8.3. The README's Yang credit is
  wrong and should be corrected [C, whether to fix now or as part of a
  larger README pass].
- **RESOLVED 2026-07-24 — Tulchinskii 2502.17017 exact statistic**, read
  directly (WebFetch of the arXiv HTML, not abstract-only). Full detail in
  §8.2. Still open: the exact `(layer, head, depth)` search protocol for
  Huginn's weight-tied case is a [C] curator decision, not resolved by
  reading the paper.
- **RESOLVED 2026-07-24 — Lu et al.'s released code**
  (`github.com/wenquanlu/huginn-latent-cot`), fetched and diffed directly
  (`huginn-predrank/raven_modeling_minimal.py`, their patched fork of the
  model source, vs. this project's `hook.py`). Two concrete findings:
  1. **Independent confirmation this project's hard-won coda-reconstruction
     fix is mathematically correct.** Their probing code does exactly
     `ln_f -> coda -> ln_f -> lm_head` on an intermediate state
     (`raven_modeling_minimal.py`'s `core_block_forward`: `x_probe =
     ln_f(x)`, run through `self.transformer.coda`, then `x_probe =
     ln_f(x_probe)`, then `lm_head(x_probe)`) — the identical double-`ln_f`
     pattern this project's `_replicate_coda_head` uses, found independently
     via two rounds of `validate_logits` catching a systematic bug (§Errors,
     `claims_ledger.md` D12). A second, independent, published
     implementation doing the same reconstruction is strong external
     validation, not just an internal self-check.
  2. **Confirms the single-layer `[-1]` hooking concern concretely, with an
     exact granularity gap now known.** This project's `hook.py:110` hooks
     only `core_block[-1]` — one capture per full 4-layer unroll (32
     captures for `num_steps=32`). Lu et al. probe **after every one of the
     4 core_block layers**, inside the layer loop itself (128 captures for
     the same 32 unrolls) — 4x finer time resolution on exactly the
     question this project studies (per-step trajectory shape). Their
     `core_block_forward`'s probing branch uses a separate `x_probe`
     variable and never touches the real `x`/cache the forward pass returns
     — confirms the "probe without disturbing the real computation" pattern
     this project's hook also relies on is sound. **Not adopted here yet**
     (would require hooking all 4 `core_block[i]` layers, not just `[-1]`,
     ~4x the extraction memory/compute for the same unroll count) — flagged
     as a Phase 1 design choice, not implemented in this pass. One
     non-issue ruled out: their code needs `deepcopy(past_key_values)`
     during probing because `model.generate`'s incremental KV cache would
     otherwise get corrupted by the probe's coda pass; this project's
     `extract_trajectory` does a single non-cached forward pass (grepped
     `hook.py`: no `past_key_values`/`use_cache` reference anywhere), so
     that specific gotcha does not apply here.
- **RESOLVED 2026-07-24 — Blayney's exact digits**, read directly from the
  downloaded PDF's Appendix C Tables 3-4 (HTML fetches truncate this
  paper). Confirmed 0.02% and 2.81% are both real numbers **but measure
  different things**: 0.02%/0.14% (Table 3) are per-TOKEN rates; 2.81%
  (Table 4) is a per-EXAMPLE "at least one hit anywhere" rate, not directly
  comparable. This corrected a same-day error in
  `power_and_preregistration.md`'s first version, which had used 2.81% as
  a per-token rate. Full table in `claims_ledger.md` B9. Also newly
  available: their Orbit-specific (not broader non-fixed-point) per-token
  rate is 0.01%-0.13%, closer to this project's own winding-loop metric;
  and their Algorithm 1 (FFT-based orbit detection, τ=0.05, ρ=0.9) is a
  second, independently-designed classifier worth comparing against
  `classify_shape`.
- **RESOLVED 2026-07-24 — "Geiping observes orbits on question/digit
  tokens"**, read directly from the downloaded PDF (arXiv 2502.05171,
  §"Iteration Trajectories", Figures 11-12). **Partially right, and the
  imprecise part matters.** Confirmed: digit tokens do show orbits — their
  own worked example is literally the token `" 3"` in a GSM8K problem
  ("the state of the token quickly falls into an orbit pattern in all
  three pairs of PCA directions"). But "question tokens" is not quite what
  they say — that phrase actually describes a *different* figure (Fig. 11,
  norm-distance-to-fixed-point): "key parts of the question, and the start
  of the model response, are 'deliberated' much more in latent space" —
  slower *convergence*, not necessarily orbiting. The orbit examples they
  give beyond digits are specific structural/deliberation words — `"makes"`,
  `"thinks"` — "tokens... that determine the structure of the response,"
  not question tokens as a class. **Correction for this project's own
  usage: say "orbits on digit/arithmetic tokens and certain structural
  words; deliberation (slow convergence, not necessarily orbiting) is
  stronger on question tokens" — two distinct claims from two different
  figures, previously conflated into one.** Separately, their paper states
  path-independence directly (quoted in §2.7 above) — real, primary-source
  support for treating the geometry-vs-activations/path-independence
  concern as live, not merely hypothetical.
- **RESOLVED 2026-07-24 — Digit tokenization.** Loaded the real tokenizer
  (CPU, no GPU needed) and checked directly: `" ".join(digits)` gives exactly
  one token per digit for both `'0'`/`'1'` sequences (`make_count_ones_task`'s
  own docstring claim "n_ops tokens, always" is confirmed correct) — and,
  separately, so does the *unspaced* concatenation (`'01010110101'` also
  tokenizes 1 char/token on this tokenizer), so the earlier open question
  ("should we do spaces between them") turns out not to matter for token
  count on this specific tokenizer, only for whether a leading space
  attaches to each digit. **Also confirms the multi-digit-answer problem
  (§9) directly**: single-digit answers (0-9) are always exactly one token,
  but answers >=10 split into 2 tokens with the first token being only the
  leading digit (`" 12"` -> `[" 1", "2"]`, `" 20"` -> `[" 2", "0"]`) — a
  first-token-argmax check on any n_ops>9 count_ones/counting/three_scale
  item is checking the wrong thing, exactly as already flagged, now with a
  live confirmation instead of an assumption.
- **CORRECTED 2026-07-24 — J-Space / Jacobian-lens compute cost.** Earlier
  framing ("one linearized-attribution pass," cheap) was wrong, caught on
  challenge and re-verified against the actual methodology
  (transformer-circuits.pub/2026/workspace/). J-lens is **not** computed
  per-example. It requires a precompute pass over a ~1,000-prompt
  calibration corpus, `J_ℓ = E_{t,t',prompt}[∂h_final,t'/∂h_ℓ,t]`, averaged
  into a stored `d_model x d_model` matrix *per layer* — genuinely
  comparable in cost to a real extraction sweep (1000 forward+backward
  passes, times however many effective depths get distinguished for a
  weight-tied model, same "which (layer, depth)" ambiguity QK has), not a
  one-off probe. Deprioritization is now doubly justified: not in the
  original proposal (bonus, not required), AND genuinely GPU-heavy, not
  just under-researched. Still the strongest interpretability escalation
  path this project has found; revisit only after the required QK/spectral
  deliverables ship, and only with a real compute budget line for the
  calibration pass.

---

## 14. Architecture-decision record — what narrowed, and why

The team's own `theoretical_framework.md` envisioned a much broader program
(training a solver, comparing TRM/HRM/URM/Universal-Transformer, RTD/zigzag/
persistence-landscape TDA, grokking/rank/gradient training-dynamics metrics,
Sudoku/maze/graph tasks). The project narrowed to **inference-only Huginn** —
which is correct and matches the *proposal's* stated protocol (the framework was
one teammate's broader initial survey). Consequences the plan should honor:
- **Inference-only is the right scope** — it makes the work tractable and the
  proposal explicitly requires it. Training-based items (Yang, Movahedi,
  grokking metrics) are out by construction.
- **But the framework's *better TDA tools* still apply inference-only and are
  under-used:** the framework itself demotes winding to "auxiliary" and names
  **persistent homology / RTD / zigzag / persistence landscapes** as
  better-supported — and **RTD (Barannikov et al. 2201.00058) is another
  curator-aligned, unused method** that directly fixes the single-curve-H1
  problem by comparing trajectory *populations*. Fold RTD + persistence
  landscapes into Phase 2.2 as first-class, not winding-only.
- **Comparison architectures (URM, Universal Transformer)** are a clean future
  extension but out of scope for the paper's core Huginn claims.

---

## 15. Immediately actionable (the honest down-payment)

Everything in Phase 0 is doable now, no GPU, and rescues/re-scopes existing
claims at zero risk. The highest-value first moves, in order:
1. **DONE 2026-07-24 — BH-FDR across the ~50 existing correlations** (0.2) —
   `scripts/run_fdr_correction.py`. Recontextualized every "significant"
   claim; both prime-target hits survive FDR itself but are independently
   explained away (D10 confound, non-replication) — see D13.
2. **DONE — Correct D11 in the ledger** with the verified multivariate result +
   the prefix-confound caveat (0.4) — now real code
   (`multivariate_rank_control` in `correlate.py`), wired into
   `run_three_scale.py`, tested against the cached CSV.
3. **DONE — Reproduce the force-loop Fisher p=0.01 in code** (§2 verdict) —
   `run_forceloop.py`'s `main()` now runs the real Fisher exact test,
   verified to match the ledger exactly.
4. **Power analysis + pre-registration** (0.1) — the honesty gate. Not yet done.
5. **DONE — Implement the multivariate rank control in `correlate.py`** (§9) —
   done as part of item 2 above.

Remaining: item 4 (power analysis + pre-registration) is the last Phase 0
piece; it needs no compute and no curator input either, and gates any future
H2 claim about loop rate.
