# Deep-research findings: Huginn literature & interpretability grounding

STATUS: **PARTIAL**, run interrupted twice by session quota exhaustion
(2026-07-22/23, run ID `wf_b88b85c4-ee9`; a third resume attempt was
deliberately stopped before spending quota again — see below). 106 of a
planned agent set launched, 52 completed with real results, 54 killed
mid-flight by quota, synthesis step never ran. Raw, unmerged output — 8
adversarially confirmed claims (2-3 vote consensus) plus 17 single-source,
**not yet adversarially verified** claims, out of 117 total claims
extracted from 24 sources across 5 search angles.

**Two threads were separately closed out cheaply, without the expensive
multi-agent workflow** — direct `WebFetch` reads instead, after the second
quota exhaustion made another full resume look like a bad trade: "J-Space"
(resolved, see below) and the three 2026-dated arXiv candidates for a
"later Huginn analysis paper" (read directly, none of them turned out to
be that — see below). This is the cheaper pattern to prefer over
relaunching the full workflow for narrow, single-source lookups.

**Still open, would need the full workflow (or more targeted manual
reads) to close**: Lu et al. 2507.02199 full read, Merrill/Grazzi
re-verification against this project's specific attributed claims, the
ACT/PonderNet/DEQ mechanistic details (real sources found, zero
adversarial passes), whether circuit-tracing has been applied to any
recurrent-depth architecture, and the winding-number null-model prior-art
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

**Test-time scaling** (thread 2 — the comparison against this project's
own steps-to-settle findings, not yet actually done):
- Model card claim: benchmark accuracy improves with more unrolling up to
  roughly **num_steps=64**, flat beyond that. If true and verified, this
  is directly comparable to this project's own `DEFAULT_NUM_STEPS=64`
  choice — worth checking whether that constant was chosen for this exact
  reason or independently landed on the same number.
- Mean successive-step KL divergence drops **~4 orders of magnitude
  between loop 2 and loop 16** — a concrete, checkable number against this
  project's own `steps_to_settle` distributions if it holds up.
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

## Threads with sources found but no claims extracted yet before quota died

- **Lu et al. 2507.02199** (thread 4) — real paper, confirmed to exist
  (html + pdf both found), plus a related repo
  (`github.com/wenquanlu/huginn-latent-cot`) and a blog post
  (`blog.bluedot.org/p/interpreting-latent-reasoning-in`) — genuinely
  promising leads not yet read.
- **Merrill et al. 2404.08819 / Grazzi et al. 2411.12537** (thread 5) —
  both confirmed real (matches what this project already cites), not yet
  re-verified against the specific claims this project attributes to them.
- **Anthropic circuit-tracing** (thread 7) —
  `transformer-circuits.pub/2025/attribution-graphs/{biology,methods}.html`
  found (the real "On the Biology of a Large Language Model" work), but
  whether it's been applied to any recurrent-depth/weight-tied
  architecture specifically was never checked.
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
