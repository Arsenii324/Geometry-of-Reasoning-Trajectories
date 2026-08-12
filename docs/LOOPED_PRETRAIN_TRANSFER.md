# What this project actually knows that bears on the looped-pretraining task

*Written 2026-08-12. Scope: the T-Lab test task asks for a looped transformer (<=10M params,
<=100M tokens, FineWeb, Qwen3 base) in which **many** loops stay useful. This document is not a
proposal for that task. It is an inventory of what our own Huginn-3.5B measurements license us to
say about looped models, ranked by how hard it would be to reach the same conclusion from public
papers or from first principles.*

**Provenance discipline, because the task warns about it.** The task says to use your own
background and to avoid LLM-generated ideas. Everything in §1--§5 below is arithmetic on numbers
*we measured*, with the ledger row named. §2 is the one place where I extend a measurement into a
prediction; it is flagged as such and carries its own falsifier. Treat §2 as something to verify
before relying on, not as a result.

---

## 0. The one-paragraph version

Loop saturation is not one phenomenon. Our measurements separate it into three, which have
different fixes and are routinely conflated: the map is **time-invariant and contractive** so
iteration provably stops doing work; the **readout commits at unroll ~4 of 48**, and in 17 of 21
task families the answer is *better in the transient than at the fixed point*, so converging
destroys information; and with weight tying plus contraction the **weight gradient is dominated by
the last loops**, which are the ones doing the least. The third gives a quantitative prediction:
the number of loops that can contribute to learning is about `1/(1-rho)`, which for our measured
`rho` is 5.7--12.2 and brackets the observed knee in Huginn's own published depth curve. If that
is right, the lever is `rho`, not the loop count -- and two obvious candidate fixes are already
refuted by evidence sitting in the released Huginn checkpoint.

---

## 1. Saturation is three problems

### 1.1 Dynamical: the map cannot help converging

`block_idx` is threaded through `core_block_forward` and **never enters the block computation** --
its only use is the KV-cache update (D187c, read in the released source). So the unrolled map is
genuinely time-invariant: loop *t* and loop *t+1* apply the identical operator. Combined with
measured contraction (D115), iteration past the fixed point is a provable no-op, not an empirical
disappointment.

Two consequences worth stating separately:

- Saturation of a weight-tied, time-invariant, contractive loop is **architectural**. No training
  schedule fixes it, because the fixed point is a property of the map.
- Huginn *has the plumbing* for depth-conditioning and does not use it. A depth-dependent
  modulation (scale/shift from a depth embedding) costs `O(d)` parameters, which does not grow with
  model size in the way the task's own counter-example -- a large learnable value-embedding matrix
  -- does.

### 1.2 Read-out: the answer is decided long before the state settles

D159, over 21 census families x 4868 draws x 48 unrolls: **the median unroll at which the gold
token attains its best rank is 4**, and *the gold is best available in the transient and worse at
the fixed point in 17 of 21 families*. Convergence does not merely stop adding information; it
**removes** it.

This is the finding that reframed our project, and it is the one I would most want on the table
before designing a loop schedule. It says the ceiling on "useful loops" may be set by **when the
decoder commits**, not by how long the state keeps moving. Those need different interventions:
deep supervision, per-loop readout, or anything that delays commitment -- not more depth.

### 1.3 Credit assignment: the gradient is dominated by the idle loops

Weight-tied loop, loss at the end:

```
h_{t+1} = R(h_t; e; W),   L = L(h_T)
dL/dW = sum_t  (dL/dh_T) . (prod_{s>t} J_s) . (dR/dW)|_t ,    ||prod_{s>t} J_s|| ~ rho^(T-1-t)
```

The contribution of loop *t* is damped by `rho^(T-1-t)`. The **late** loops -- the converged ones
doing the least work -- carry almost all of the weight gradient; the early loops, where D159 says
the actual computation happens, are suppressed geometrically. Summing the weights gives an
effective count of `sum_k rho^k -> 1/(1-rho)`.

---

## 2. The prediction: effective loop horizon ~ `1/(1-rho)`

**Claim to test, not a result.** The number of loops that can contribute to the update is about
`1/(1-rho)`. Beyond it you buy forward FLOPs and approximately no gradient.

Our measured `rho` (D115: per-family contraction fitted from step-size decay, 608 orbits, 21
families) runs **0.8239 (`rot13_word`) to 0.9181 (`count16`)**, between-family sd 0.0301:

| rho | `1/(1-rho)` |
|---|---|
| 0.824 | 5.7 |
| 0.87 (mid) | 7.7 |
| 0.918 | 12.2 |

Against Huginn's own published ARC-Easy depth curve: **34.89 (r=1) -> 49.07 (4) -> 65.11 (8) ->
69.49 (16) -> 69.91 (32)**. As a fraction of the total r=1->32 gain: **40.5% by r=4, 86.3% by r=8,
98.8% by r=16**. The knee sits at r~8. Predicted horizon 5.7--12.2 with midpoint 7.7, for a
one-parameter heuristic that used none of this curve.

**Why this is worth a day rather than a footnote:**

- It converts "how many loops help" from an expensive sweep into a measurement available in the
  first minutes of training.
- It names the lever as **`rho`, not `T`**. To make 32 loops carry gradient you need `rho ~ 0.97`.
  Interventions that raise `rho` (residual scaling `h + alpha*f(h)`, normalisation placement,
  spectral control) are then the design space -- not loop count.
- It quantifies the tension the task itself names: DEQ *rewards* fast convergence, i.e. small
  `rho`, i.e. the **minimum** possible learning horizon. The two objectives are directly opposed
  and the trade is numeric.

**Precedent that `rho` predicts timescales here.** This is not the first time. D173 registered
`ln(0.05)/ln(0.83) = 16.1` unrolls *before* the run as the relaxation time after a mid-trajectory
parameter swap; measured 15.5 (an upper bound, detector lag). Same style of argument, applied to
state relaxation rather than gradient flow, and it held.

**Falsifier.** Deliberately vary `rho` (e.g. residual scale `alpha`), measure the saturation knee.
If the knee does not move as `1/(1-rho)`, the model is wrong. This is cheap at 10M params and a
clean negative result either way -- which the task explicitly counts as a good outcome.

**What weakens it.** The bound `||prod J|| ~ rho^k` is loose; non-normal Jacobians admit transient
growth even when the spectral radius is below 1. Our `rho` is a **step-decay fit, not a spectral
radius** -- we never computed `rho(J)`, and that is a stated limitation of our own paper. So the
constant is uncertain even if the scaling is right.

---

## 3. Two obvious candidate fixes are already refuted, in the released checkpoint

Both of these look attractive on first pass and cost real budget to discover. Neither is prominent
in the Huginn paper; both are recoverable from the released artefacts, which is how we got them.

### 3.1 "Randomise the loop count during training so the model learns to use many loops"

Huginn already does exactly this. From `raven_modeling_minimal.py:793-797`, training depth is
sampled `rate ~ lognormal(log(t+s) - sigma^2/2, sigma=0.5)`, `p ~ Poisson(rate) + 1`. Simulated at
400k draws: **mean 33.0, median 29, p10 15, p90 56, p99 93**, `P(depth >= 32) = 0.443` (D187b).

A model trained with a heavy right tail of depths, seeing 56+ loops about a tenth of the time,
still saturates by r~8--16 on its own benchmark. **Depth-randomised training alone is not
sufficient.** Whatever makes many loops useful, this is not it.

### 3.2 "The recurrence has to learn to hold state"

D53: on identical prompts, a **randomly initialised** Huginn carries the lagged running count at
cross-validated `R^2 = 0.7498` against the **trained** model's `0.7175`. The untrained model is, if
anything, slightly better. The register is **architectural, not learned** -- training did not build
it.

This reframes the problem. The bottleneck is not the recurrence's capacity to carry state across
loops; that is present at initialisation. The bottleneck is that the model does not learn to *use*
it. That points at credit assignment and readout (§1.2, §1.3), not at representational width or
at exotic memory mechanisms.

---

## 4. The sharpest design constraint we can offer

Combine two of our results:

- D159: the readout commits at median unroll **4**.
- D198: rotating and settling prompts are **indistinguishable over unrolls 0--11** (0.2292 vs
  0.2296, separation 0.0004) and separate only from window-start 16 onward (0.862 vs 0.298 at the
  tail).

Huginn *does* possess a sustained non-convergent regime -- a damped period-6 rotation, switchable
by a single instruction word at fixed token count, causally controlled from one embedding row. But
**the entire regime difference lives after the readout has committed.** The interesting dynamics
are causally downstream of the decision.

So: **producing non-convergence is not sufficient. It has to happen before readout commitment.**
Most "stop it converging" interventions will add activity in exactly the window Huginn already
fills with activity that does not reach the output. That is a non-obvious constraint and it
disqualifies a whole family of otherwise reasonable ideas cheaply.

**Where exploration is injected matters more than how much.** The task lists "exploration during
loops" as a lever. We ran the closest thing we could: a difference-of-means vector added to the
re-injected parameter `e` at **every** unroll, so it acts `r` times rather than once -- a
dose-response knob a feedforward model does not have. It moved nothing. Largest `|delta R|`
anywhere was **0.0065**, with **zero threshold crossings in 132 conditions** spanning both regimes,
identity gate exact. The dose was adequate, not too small: oracle accuracy degraded from 0.560 to
0.480 at large coefficients, so the perturbation demonstrably acted. Our reading is the **injection
site**, not the idea -- the regime's positional onset is 33 positions upstream of where we applied
it. The transferable warning: a perturbation applied at the wrong site can be applied `r` times,
visibly perturb the model, and still leave the loop dynamics untouched. Choose the site
deliberately and gate it on a measured change in the dynamics, not on the perturbation's norm.

Corollary for early-exit / Q-exit designs: `rho` is **task-type-dependent**, not a model constant
(D115 -- between-family sd 0.0301 across a 0.094 span; and D115's headline is that it tracks task
type, not prompt length and not difficulty). A single global exit threshold is therefore
mis-specified by construction. Relatedly, the asymmetry in D173 -- **entering the non-settling
regime costs ~15.5 unrolls, leaving it costs 1.0** -- says collapse to a fixed point is cheap and
escape is expensive. An exit gate that fires early is close to irreversible.

---

## 5. An operational definition of "useful compute", and a free diagnostic

The task asks to extract "as much useful computation as possible". That needs an operational
definition before it can be optimised. Ours, transposed from D159:

> A loop is **useful** iff it changes the output distribution.

Diagnostic: bank `KL(p_t || p_T)` per loop, per token, every eval. If it reaches ~0 by loop *k*,
the effective depth is *k* regardless of *T* -- and no amount of extra looping will show up in the
loss. At 10M parameters this is nearly free, and it measures the thing the task is actually about,
which validation perplexity alone does not.

Two further measurement notes that follow from our data rather than from taste:

- **Average PPL is the wrong instrument for the question, even though it is the target.** D177:
  recruited depth moves **+0.23 unrolls** with difficulty *within* a task but **+5.56** with task
  identity. Depth benefit is heterogeneous across *kinds* of token, not across *difficulty* of
  instance. An average over FineWeb tokens is dominated by tokens that need one loop. Report PPL
  stratified by per-token depth benefit as well as the headline number, or a real effect will be
  invisible.
- **The same diagnostic is also a loss lever, and a data lever.** If per-loop `KL(p_t || p_T)`
  identifies which tokens depth actually helps, that set is directly usable: upweight those tokens
  in the loss, or oversample the documents containing them, inside the fixed 100M-token budget.
  This is motivated rather than speculative -- D177 measured the benefit as heterogeneous by *kind*
  of token (+5.56 unrolls for task identity) and near-flat within a kind (+0.23 for difficulty), so
  the useful axis for weighting is token type, not an instance-level difficulty estimate. The
  honest caveat: on FineWeb the "kinds" are not labelled the way our census families were, so the
  diagnostic has to define them, and that is the part that could fail.
- **On a normalised architecture, "has it converged" is an angular question.** Every recorded
  Huginn state is an RMSNorm output on a sphere of radius **76.386** (relative sd ~5e-05), so two
  random points on that sphere sit **108.03 +- 0.75** apart -- exactly `R*sqrt(2)`, confirmed both
  by simulation at the measured radius and in closed form. A `||delta h||` threshold is an angle in
  disguise and its null is not zero. Huginn's own `embed_scale` is `72.6636084983398`, which is
  **exactly `sqrt(5280)`**; the geometry is set by construction, not learned.

---

## 6. Measurement traps that each cost us a result

The task's second criterion is implementation and verification, and it warns that coding agents
"will happily take the wrong tokeniser or forget to save a checkpoint". Ten ledger rows in this
project carry an explicit withdrawal or amendment marker, and four of those were killed by controls
we had registered *in advance* rather than by later work. These are the transferable ones; each is
cheap to guard at 10M params.

1. **Unseeded latent init.** Huginn draws `h_0` from an unseeded RNG and *no kernel seeded it
   before we did* (D78). Two forwards of one prompt were not comparable. For a looped pretrain:
   seed it, and **measure run-to-run PPL variance before believing any ablation delta**. We have a
   concrete instance of what happens otherwise: a correlation that read `+0.842` (N=10,
   significant) had a per-init-seed range of **0.17--0.87**.
2. **A degenerate outcome reads exactly like a clean null.** D148 reported "0 of 48 units changed"
   where the outcome variable was False in **432 of 432** orbits. An early-exit gate that never
   fires and one that fires but does nothing are indistinguishable in the loss. **Log the base rate
   of every gate and every outcome field.**
3. **The identity condition must be bit-exact.** Every intervention run we trust gated on
   "modification off reproduces baseline at `0.000e+00`". It caught real bugs three times. For the
   test task: `loops=1` or `modification disabled` must reproduce the baseline exactly, not
   approximately.
4. **Bank the primitive, not the summary.** We banked one FFT bin and had to re-run; banking the
   whole spectrum later answered a question the run was not designed for. Bank per-loop logits and
   states, not just final loss.
5. **A failed probe is not an absent representation.** A linear probe of the class we had used for
   several nulls could not recover a label that is a *guaranteed deterministic function of its own
   input features*: 0.690 against a 0.600 baseline, where the correct one-dimensional statistic
   reaches 0.980 (D165). Never conclude "the model doesn't represent X" from a probe failure.
6. **Protocol can dominate the number you are attributing to architecture.** 25 ARC measurements
   partition cleanly and non-overlappingly by whether the option list was in the prompt
   (0.392--0.495 with, 0.592--0.658 without) -- a gap larger than the spread within either group.
   The PPL analogue is tokeniser, sequence packing, document-boundary handling and BOS: all can
   exceed the architectural delta being measured. Fix and version them before the first ablation.
7. **A longer analysis window is worse for a damped signal.** Our first spectral pass failed its
   own gate because over a 48-step window the decay transient dominated the low-frequency bins.
   If you measure loop dynamics over the whole trajectory, the transient will swamp what you want.
8. **Per-step geometry is confounded with step norm unless normalised** (the two-scale
   reproduction): before normalisation the orthogonality statistic partly tracks how far the state
   moved; after, its correlation with step norm falls to -0.0436.

---

## 7. What we cannot claim

Stated so the rest stays usable.

- We **never trained a looped model.** Every measurement above is inference-only on one checkpoint,
  one revision, seeded per condition. The transfer to a 10M-parameter pretrain is an argument, not
  an observation.
- We **never computed the spectral radius `rho(J)`.** Our `rho` is a step-decay fit. §2's constant
  depends on that gap.
- Our regime results come from an **instruction-following setup**, not next-token pretraining on
  web text. There is no "instruction word" in FineWeb. What transfers is the structural claim (the
  loop's qualitative dynamics are selected by input surface form, not by required computation), not
  the specific nouns.
- Huginn is **3.5B**; the task is **<=10M**. Contraction, credit assignment and readout commitment
  are all plausibly scale-dependent. `rho` in particular should be re-measured, not assumed.
- The `1/(1-rho)` horizon is a heuristic with a named falsifier (§2), not a theorem.

---

## 8. If I had to pick three things to carry over

1. **Measure `rho` and the per-loop `KL(p_t || p_T)` curve from the first training run**, before
   any architectural choice. They give the effective depth for free and turn the central question
   into a measurement (§2, §5).
2. **Treat readout commitment as a separate axis from depth** (§1.2, §4). The single most
   surprising thing we found is that the answer is *better in the transient than at the fixed
   point* in 17 of 21 families. If that reproduces at small scale, "more loops" is the wrong frame
   and "later commitment" is the right one.
3. **Take the two refutations in §3 for free.** Depth-randomised training is already done in Huginn
   and does not break saturation; and the recurrence's state-carrying capacity is present at
   initialisation and is not what training improves. Both save budget that would otherwise be spent
   confirming them.
