# Deep-research findings: Huginn literature & interpretability grounding

STATUS: **PARTIAL, run interrupted twice by session quota exhaustion**
(2026-07-22/23, run ID `wf_b88b85c4-ee9`). 106 of a planned agent set
launched, 52 completed with real results, 54 killed mid-flight by quota,
synthesis step never ran. This is raw, unmerged output — 8 adversarially
confirmed claims (2-3 vote consensus) plus 17 single-source, **not yet
adversarially verified** claims, out of 117 total claims extracted from 24
sources across 5 search angles. Treat the "unverified" section as leads,
not facts, until re-checked.

**To resume and get the remaining ~54 agents + synthesis** (free — cached
agents replay instantly, only new/failed ones re-run):
```
Workflow({
  scriptPath: "/Users/a2mogus/.claude/projects/-Users-a2mogus-build-projs-barannikov-work-Geometry-of-Reasoning-Trajectories/1444efcb-22d8-4093-9d69-c781ae344e2e/workflows/scripts/deep-research-wf_b88b85c4-ee9.js",
  resumeFromRunId: "wf_b88b85c4-ee9",
  args: "<same research brief — see git history of this file / prior conversation>"
})
```
Quota reportedly resets 1:50pm Europe/Moscow (as reported by the failed
run itself — not independently confirmed).

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
- **Most interesting unverified claim, flag for priority re-check**: "a
  training-free, activation-based halting rule (successive-output KL
  threshold) matches uniform depth-8 output quality using 38% fewer
  average loops (4.94 vs 8), and beats a learned linear router trained on
  convergence labels." If real, this bears directly on the depth-calibration
  open question from earlier in this conversation — a stronger, more
  specific version of "does the model have a legible internal stopping
  signal" than what was proposed. Source not pinned down precisely before
  quota died; likely one of the 2026-dated arXiv IDs below.

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

## "J-Space" — still unresolved, one real lead

No claim in either confirmed or unverified list identifies what this is.
One candidate surfaced in the source list, unverified:
`transformer-circuits.pub/2026/workspace/` — a 2026-dated page on
Anthropic's own transformer-circuits site. Worth checking directly first
when resuming; do not assume this is "J-Space" without reading it.

## Other 2026-dated sources found, unread — possible "later analysis paper"

Three arXiv IDs with 2026 timestamps turned up in the source list, topically
relevant, contents unknown (never reached extraction before quota died):
`arxiv.org/abs/2602.08864`, `arxiv.org/abs/2606.29983`,
`arxiv.org/abs/2607.14427` (also seen as `arxiv.org/html/2607.14427`). Any
of these could be the "slightly later analysis paper" referenced earlier in
this conversation, or the source of the "38% fewer loops" claim above.
Reading these should be the first priority on resume.
