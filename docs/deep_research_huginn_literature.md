# Deep-research findings: Huginn literature & interpretability grounding

STATUS: **PARTIAL**, run interrupted twice by session quota exhaustion
(2026-07-22/23, run ID `wf_b88b85c4-ee9`; a third resume attempt was
deliberately stopped before spending quota again — see below). 106 of a
planned agent set launched, 52 completed with real results, 54 killed
mid-flight by quota, synthesis step never ran. Raw, unmerged output — 8
adversarially confirmed claims (2-3 vote consensus) plus 17 single-source,
**not yet adversarially verified** claims, out of 117 total claims
extracted from 24 sources across 5 search angles.

**Most threads were closed out cheaply after that, without the expensive
multi-agent workflow** — plain `WebFetch` reads instead, after the second
quota exhaustion made another full resume look like a bad trade: "J-Space"
(resolved), the three 2026-dated arXiv candidates for a "later Huginn
analysis paper" (read directly, none of them turned out to be that), Lu et
al. 2507.02199 (resolved — the single most relevant related-work paper
this project has), Merrill/Grazzi re-verification (resolved, with a scope
correction), Anthropic circuit-tracing's actual architecture coverage
(resolved), and Huginn's own test-time-compute scaling numbers (resolved,
real GSM8K/HellaSwag/ARC-C figures). This is the cheaper pattern to prefer
over relaunching the full workflow for a narrow, single-source lookup.

**Still open, would need the full workflow (or more targeted manual
reads) to close**: the ACT/PonderNet/DEQ mechanistic details (real
sources found, zero adversarial passes — see below, still useful as-is
despite being unverified) and the winding-number null-model prior-art
search (never even reached — see below).

**To resume the full workflow for those remaining threads** (partially
free — the 52 already-cached agents replay instantly; still real new cost
for whatever's left, weigh that before launching):
```
Workflow({
  scriptPath: "/Users/a2mogus/.claude/projects/-Users-a2mogus-build-projs-barannikov-work-Geometry-of-Reasoning-Trajectories/1444efcb-22d8-4093-9d69-c781ae344e2e/workflows/scripts/deep-research-wf_b88b85c4-ee9.js",
  resumeFromRunId: "wf_b88b85c4-ee9",
  args: "<same research brief — see git history of this file / prior conversation>"
})
```

## Confirmed (2-3 vote adversarial consensus, direct primary-source quotes)

All from the real Huginn paper, arXiv **2502.05171** ("Scaling up
Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach") —
confirms this is the real, correct primary source.

1. **Recurrent-unroll count `r` is sampled stochastically during training**
   (3-0): `τ ~ N(log(r̄) − ½σ², σ)`, then `r ~ Poisson(e^τ) + 1`, with
   `r̄ = 32` and `σ = 1/2` for the main model. Heavy right tail above the
   mean. This is the exact formula for what this project's discussion has
   been calling "depth-randomized training" — cite this, not the vaguer
   description used earlier in this session.
2. **Truncated BPTT, k=8** (3-0, corroborated 2-1 in a second pass):
   gradients only flow through the last `k=8` recurrent iterations,
   regardless of total `r` unrolled that step. The prelude block is the
   exception — it gets a gradient update every step, since its output is
   re-injected at every iteration.
3. **No learned halting signal exists in training** (3-0, twice): training
   only randomizes `r` per the formula above; there is no halting loss, no
   stop/continue signal, nothing analogous to ACT's ponder cost. The only
   stopping criterion anywhere is a **post-hoc, zero-shot, inference-time**
   heuristic: halt when KL divergence between consecutive recurrent-state
   output distributions drops below **5×10⁻⁴**. This directly confirms
   what was reasoned from first principles three turns ago in this
   conversation — now with the exact threshold value, citable.
4. **Task-dependent convergence depth, split verdict (1-1, weak)**: the
   paper states low-reasoning tasks (OpenBookQA) plateau at lower iteration
   counts than high-reasoning tasks (GSM8K). This is the closest
   primary-source evidence bearing on the depth-calibration question, but
   the adversarial pass only reached 1-1 consensus — don't cite this one
   without re-verification.

## Unverified (single source, real primary papers, NOT adversarially checked)

Promising, specific, and plausible — but exactly the kind of claim this
project has already been burned by trusting once before verification.
Do not put any of these in a paper draft without independent confirmation.

**Training mechanics detail** (extends the confirmed items above):
- Exact code location: `recpre/model_dynamic.py`'s `iterate_forward()` —
  `num_steps_no_grad` iterations run inside `torch.no_grad()` (the "free"
  extra iterations beyond the truncated-BPTT window).
- The released `huginn-0125` checkpoint's specific config:
  `mean_recurrence=32`, `mean_backprop_depth=8` (consistent with the
  confirmed r̄=32/k=8 above), sampling scheme name
  `"poisson-lognormal-filling"`, one of ~15 alternative schemes present in
  source but not necessarily used for the released checkpoint.

**Test-time scaling — RESOLVED (2026-07-23, direct read of the full paper
text), real numbers, directly comparable to this project's own data**:
- **GSM8K (math reasoning)**: r=4 → ~9-10% accuracy; r=32 → ~28-38%
  (without/with system prompt); r=64 with weight averaging → 38.6% strict
  / 47.2% flexible. Large, real, monotone-ish gains from more recurrence
  on a genuinely hard reasoning task.
- **HellaSwag (commonsense)**: saturates fast — "only needs 8 iterations
  to achieve near peak performance" (their Figure 7).
- **ARC-C**: saturation point is **context-dependent** — without few-shot
  examples, saturates around 8-12 iterations; with 25-50 few-shot
  examples, saturation shifts to around 32 iterations (their Figure 9).
- **Their own stated general principle** (direct quote): *"saturation is
  highly task-dependent, on easier tasks the model saturates quicker,
  whereas it benefits from more compute on others."* This is the paper's
  own primary-source confirmation of task-dependent inference-time
  saturation — not the same claim as training-time depth-calibration
  (still unresolved), but real, concrete, and the closest thing to direct
  evidence for "the model's compute usage responds to difficulty" this
  research pass found. Directly comparable target for this project's own
  `steps_to_settle` distributions once real GPU data exists — do these
  project's own per-task saturation points land in the same 8-32 range,
  and does easy-vs-hard task ordering match theirs?
- Still unverified, not reached in this read: "mean successive-step KL
  divergence drops ~4 orders of magnitude between loop 2 and loop 16" —
  plausible, not yet confirmed against this specific source.
- **UPDATE 2026-07-23, now confirmed (single direct read, not the full 3-vote
  adversarial pass, but a real quote from the actual abstract) — with an
  important scope correction**: the "38% fewer loops" claim is real, source
  is arXiv **2607.14427**, exact quote: *"a training-free rule that halts
  each token once its output stabilizes attains uniform depth-8 quality at
  4.94 average loops (a 38 percent reduction in average depth)"*, beating a
  learned router which "requires nearly full depth and yields no
  reduction." **This is NOT about Huginn.** 2607.14427 studies a separate,
  much smaller (135M-parameter) depth-recurrent transformer trained on
  FineWeb-Edu — same architecture *class*, different model entirely. Do
  not cite this as a Huginn result; cite it as convergent evidence from a
  comparable architecture. Also reports token-wise convergence
  heterogeneity relevant to this project's own `steps_settle` spread:
  "median tokens stabilize by loop six, while roughly 10% continue
  updating at training depth," ordered by token type (whitespace
  shallowest, content words deepest) — a real, checkable point of
  comparison for this project's own per-token settling data.

**ACT / PonderNet / DEQ mechanistic contrast** (thread 3, entirely
unverified — real sources found, claims extracted, zero adversarial passes
completed):
- ACT (arXiv 1603.08983): learned sigmoidal halting unit, `N(t)` = first
  step cumulative halting-activation crosses `1−ε`; trained via an
  explicit "ponder cost" auxiliary loss with a hand-tuned `τ`; paper
  itself reportedly flags high sensitivity to `τ` and no principled way to
  set it.
- PonderNet (arXiv 2107.05407): halting distribution is a genuine
  (truncated) geometric distribution over per-step Bernoulli halting
  probabilities — not a cumulative-threshold rule like ACT. Two-term loss:
  expected prediction loss over halting steps, plus a KL-regularizer
  against a geometric prior. Claims **unbiased** gradient estimates,
  explicitly contrasted against ACT's claimed **biased** gradient (ACT's
  cost term only backprops through the final computational step).
- DEQ (arXiv 1909.01377): finds the fixed point directly via root-finding
  (implicit differentiation), equivalent to an infinite-depth weight-tied
  network; **constant memory** regardless of effective depth, since
  gradients come from the implicit function theorem, not stored
  activations across iterations.
- None of this has been checked against Huginn's actual mechanism yet
  beyond the confirmed items above — the mechanistic contrast argued
  informally three turns ago in this conversation is consistent with these
  unverified claims but not yet backed by a completed adversarial pass.

## Lu et al. 2507.02199 — RESOLVED (2026-07-23, direct read), the single most important related-work citation this project has

Full title recovered: **"Latent Chain-of-Thought? Decoding the
Depth-Recurrent Transformer"**. Not a generic critique — **it analyzes
Huginn-3.5B specifically**, the exact model this project studies, with
methodology strikingly close to this project's own V6 correctness probe:

- **Methods**: logit lens, "**Coda Lens**" probing, and **rank-trajectory
  tracking of result tokens on arithmetic tasks**. "Coda Lens" is almost
  certainly the published name for exactly what `hook.py`'s
  `_replicate_coda_head` does (run the real coda→ln_f→lm_head tail on an
  intermediate recurrent state) — worth citing precisely and checking
  whether this project's implementation matches their definition before
  claiming novelty for that technique.
- **Finding 1**: "limited evidence of interpretable latent CoT by tracking
  rank trajectories of final and intermediate result tokens" — i.e. they
  ran essentially this project's own V6 probe design (does the correct
  answer's rank/argmax emerge coherently across recurrent steps) and got a
  **negative-leaning** result.
- **Finding 2**: "significant probing inconsistencies across recurrent
  blocks, where the interpretability of hidden states depends heavily on
  both the layer index and the decoding method" — a direct, specific
  warning that this project's own choice of *where* to hook
  (`core_block[-1]`) and *how* to decode (coda vs. skipping it — the exact
  bug fixed earlier this session) materially changes what you see. Their
  finding that method choice matters this much is independent support for
  having fixed the coda-skip bug rather than leaving it as "probably fine."
- **Finding 3**: "increasing recurrence depth yields only marginal gains
  and falls well short of models that explicitly externalize reasoning
  steps" — a genuinely skeptical, negative-leaning verdict on whether
  Huginn's recurrence depth buys much reasoning benefit at all. This
  should be engaged with directly in this project's writeup, not
  footnoted — if this project's own (not yet GPU-verified) findings turn
  out more positive, that disagreement is itself the interesting result;
  if they agree, that's independent corroboration.

Code available: `github.com/wenquanlu/huginn-latent-cot` — worth diffing
against this project's own `extraction/hook.py` and
`run_v6_correctness_probe.py` before the GPU rerun, both to avoid
duplicating their exact setup uncredited and to check whether their
"probing inconsistency" finding suggests this project's own hook location/
decoding choice needs a robustness check across multiple layers, not just
`core_block[-1]`.

## Merrill et al. 2404.08819 / Grazzi et al. 2411.12537 — RESOLVED (2026-07-23, direct reads), with a scope correction worth noting

Both real, both confirmed, both **about a different architecture family
than Huginn** — worth being precise about this before citing them again.

- **Merrill et al.**: claims SSMs (Mamba-style) *and* transformers share
  the same TC⁰ expressiveness ceiling — "SSMs cannot express computation
  outside the complexity class TC⁰... cannot solve simple state-tracking
  problems like permutation composition," and "the 'state' in an SSM is an
  illusion: SSMs have similar expressiveness limitations to non-recurrent
  models like transformers." This is about **SSMs vs. standard
  transformers**, not specifically about weight-tied depth-recurrence —
  the excerpt available doesn't distinguish recurrent-over-depth from
  recurrent-over-time architectures at all.
- **Grazzi et al.**: narrower still — specifically about **linear RNNs**
  (Mamba, DeltaNet). Diagonal state-transition matrices restricted to
  `[0,1]` can't solve parity; allowing negative eigenvalues (range
  `[-1,1]`) fixes it and improves state-tracking generally.
- **Scope correction for this project's own prior citation**: this
  project's H3 toy-model `FINDINGS.md` cites both papers alongside its
  own recurrent-over-time vs. recurrent-over-depth comparison. That
  comparison's own two categories (classic Elman RNN vs. depth-recurrent
  weight-tied network) are neither of these papers' actual subject (SSMs /
  linear RNNs, a third family). The toy-model argument doesn't depend on
  Merrill/Grazzi being about the same architecture — but citing them as if
  directly supporting evidence for the depth-recurrent case specifically
  would be imprecise. Treat them as adjacent theoretical context (the
  general principle that "recurrent" doesn't automatically mean
  "state-tracking-capable"), not as findings about Huginn's architecture
  class.

## Anthropic circuit-tracing — RESOLVED (2026-07-23, direct read): confirms it hasn't been applied to recurrent-depth architectures

Direct read of `transformer-circuits.pub/2025/attribution-graphs/biology.html`
confirms: applied only to **Claude 3.5 Haiku** (a standard, non-recurrent
production transformer), plus a smaller comparison model and a finetuned
"secret goal" variant — all standard architectures. The methodology
description is built around "transformer-based language models" with
ordinary MLP/attention layers processing token sequences once; **no
mention of recurrent-depth, weight-tied, or iteratively-looped
architectures anywhere**. Also worth noting for calibrating ambition here:
the paper itself says only "approximately one quarter of the prompts"
yielded satisfying circuit-level insight, and frames results as existence
proofs, not comprehensive coverage — even on the architecture it was built
for. This directly confirms the assessment made earlier in this
conversation (circuit-tracing is realistically out of scope for this
project's size/timeline) with actual evidence rather than just informed
guessing: nobody has done the hard infrastructure work of applying this to
a recurrent-depth model yet, so there's nothing to borrow.

## Threads with sources found but no claims extracted yet before quota died

- **Null-model prior art for winding number** (thread 8) — no sources
  found at all. The 5-search-angle scoping phase appears to have folded
  this into another angle or dropped it; still fully open. (This session
  separately built and shipped a first-principles null-model check —
  `winding_null_test` in `src/traj_geom/metrics/winding.py` — without
  waiting on this literature search; worth reconciling once/if prior art
  turns up.)

## "J-Space" — RESOLVED (2026-07-23, direct read of transformer-circuits.pub/2026/workspace/)

Real, and it's exactly the kind of Anthropic 2026 release my own knowledge
cutoff (Jan 2026) would miss. **"J-Space: A Global Workspace in Language
Models"** — a subset of a model's internal representations functioning
like a global workspace: "a small, privileged set of representations that
[models] can report, manipulate, and reason with, amidst a much larger
volume of processing." Five functional properties claimed: verbal report,
directed modulation, internal reasoning (medium for multi-step inference),
flexible generalization (one representation routes to multiple downstream
operations), selectivity (small fraction of total representational
content). Identified via the **Jacobian Lens (J-lens)**: "the average
linearized effect of an activation on the model's likelihood of producing
a particular token" — surfaces *unverbalized* reasoning, not just output.

**Relevance to this project, worth flagging directly**: J-lens and this
project's H3 toy-model `spectral.py` both build on Jacobians of internal
computation, but for different purposes — J-lens attributes an
activation's effect on token probability (interpretability/circuit
framing), `spectral.py` estimates the operator norm for contraction
analysis (dynamical-systems framing). Not the same tool, don't conflate
them. But J-Space's actual definition — a small, verbalizable, selective
subset of representations used for multi-step reasoning — is a strikingly
direct match for the "linear probe for an internal step-counter/progress
signal" interpretability escalation already named as a next step (see
`open_question_depth_calibration.md` and the interpretability-ladder
discussion). If Huginn has anything resembling a J-Space, it's the most
promising concrete escalation path available: look for a small, selective,
verbalizable subset of the recurrent state that reasoning depth routes
through, using J-lens-style linearized attribution rather than a bare
linear probe. Not attempted here — flagging as the strongest lead this
research pass produced.

## 2026-dated arXiv papers — read directly, none is Huginn's "later analysis paper"

All three checked directly (single reads, cheap, no multi-agent fan-out
needed):
- **2602.08864** — not about Huginn. Proposes its own framework (ANIRA) for
  token-level adaptive computation in recurrent transformers on synthetic
  tasks. Finding: compute allocation can track task complexity without
  explicit supervision, but doesn't imply algorithmic generalization —
  models fail on unseen input sizes.
- **2606.29983** — not about Huginn. Addresses instability in Looped
  Transformers during length extrapolation; proposes randomizing loop
  count during training (matches Huginn's own r-sampling approach,
  independently) plus an RL-learned stochastic halting mechanism
  (RL-Halting). Tested on binary addition, Dyck-1.
- **2607.14427** — not about Huginn either, but this is the source of the
  "38% fewer loops" finding above (135M-param FineWeb-Edu model).

**None of these is a "later Huginn analysis paper."** They're independent,
convergent work on the same general problem (depth-recurrent/looped
transformer adaptive computation) from around the same time, not sequels
to Geiping et al. If a specific later-analysis-of-Huginn paper exists, it
wasn't among these three and remains unfound.
