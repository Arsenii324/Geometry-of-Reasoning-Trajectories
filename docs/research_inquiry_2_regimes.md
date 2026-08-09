# Deep research inquiry #2: when do weight-tied recurrent transformers *fail* to contract? (loops, drift, and whether our null is expected)

**One question, stated up front.** We measured that a specific recurrent-depth
transformer's latent iteration is a **uniform contraction** — every orbit pair we
tested converges, none loops or drifts. Our supervisor's hypothesis predicts that
looping/drifting regimes should exist and matter. **We need to know whether our
result is (a) expected and already known, (b) an artefact of where or how we
measured, or (c) a genuine finding worth reporting.** And critically: **under
what conditions, if any, do such models exhibit non-contracting dynamics?**

---

## 1. The system

**Huginn-3.5B** (`tomg-group-umd/huginn-0125`, Geiping et al.,
[arXiv:2502.05171](https://arxiv.org/abs/2502.05171)), a recurrent-depth
transformer:

```
embed -> prelude(2 blocks) -> [core_block(4 blocks)] x r -> coda(2 blocks) -> ln_f -> lm_head
```

Facts we have confirmed by reading the model source, which matter for the
dynamics:

- The core block is **weight-tied** and iterated r times, r chosen at inference.
- The initial latent `h_0` is **random** (truncated-normal noise shaped like the
  embeddings) and, in the public code path, **unseeded** — so repeated runs of
  the same prompt start from different points.
- The **prelude output is re-injected every iteration** through an adapter on
  `[h, e]`.
- The iteration index reaches only the KV-cache slot; there is **no timestep
  conditioning** of the weights. So the recurrence is a **strict autonomous map**:
  the same function applied repeatedly, not a time-varying one.

That last fact is important to our reasoning and we would like it stress-tested:
an autonomous map with contraction factor < 1 cannot sustain a limit cycle or
drift, so loop/drift regimes would be *ruled out by construction* if the map
contracts globally.

## 2. What we measured

The quantity is the **per-step contraction factor of the map**, estimated the
honest way: take two orbits of the **same prompt** started from **different
`h_0`**, and fit the decay rate of the distance between them. (We are aware that
fitting how fast a *single* trajectory approaches its own endpoint is
near-tautological for any convergent sequence; we do not do that.)

Result over 16 prompts × 32 random initialisations, ~960 orbit pairs, reading the
state at the last core block:

- **median per-step factor ρ = 0.855**; per-prompt medians span 0.832–0.901.
- **0.0% of pairs had ρ ≥ 1.** Not one loop, not one drift.
- The residual gap two orbits settle to is **bounded away from zero** (median
  ~2e-2), i.e. different initialisations converge to *nearby but distinct*
  endpoints, not to one identical fixed point.
- An independent estimator from a different route gave ρ ∈ [0.808, 0.920],
  consistent.

We separately found the residual gap is ~100× larger for counting-type tasks than
for arithmetic/comparison ones — unexplained.

We also ran a causal test: replacing the state at iteration r with one from a
different initialisation of the same prompt. The perturbation's effect **decayed
monotonically and vanished entirely by iteration 32**, and never changed the final
output. (This experiment was underpowered for its primary question and we treat it
as suggestive only, but the decay is consistent with ρ ≈ 0.855.)

## 3. Why this is a problem worth researching

Our supervisor's hypotheses (which carry a strong prior for us) distinguish
**settling / looping / drifting** regimes and propose that **contraction may not
be the only or best regime** — that useful computation might live in orbits that
loop or drift rather than converge. Our measurement says this model contracts
everywhere we looked.

Before we report "no loop/drift regime exists here", we must rule out that we
simply measured in the wrong place or the wrong way.

**Specific worries we cannot resolve from our own data:**

- We read the state at **one layer** (the last of the 4 core blocks) and at **one
  token position** (the answer position). A cycle *across the 4 blocks within an
  iteration* would be invisible to us — we would see each block converge while the
  cycle lives in the composition. We have seen at least one paper suggesting
  exactly this: that each layer in the cycle converges to a distinct fixed point
  and the recurrent block therefore follows a cyclic trajectory. **If the "loop"
  in the literature means an inter-layer cycle rather than a non-contracting map,
  our measurement and the hypothesis may not even be about the same object.**
- Convergence is reportedly **non-uniform across token positions** — one paper
  reports the median token converged by iteration 6 while ~10% are still updating
  at 8. We measure one position, so we may be averaging over a mixture.
- Contraction might depend on **prompt type, depth budget, or numerical
  precision**, and we sampled a narrow set of prompts chosen for other reasons.

## 4. The questions, in priority order

**Q1 (most important). Is there published evidence of weight-tied /
looped / recurrent-depth transformers exhibiting genuinely non-contracting
latent dynamics** — limit cycles, quasi-periodic orbits, sustained drift, or
chaotic behaviour — **at inference, in a trained model?** If yes: which model,
which architecture feature enabled it, how was it measured, and does it require
something Huginn lacks (timestep conditioning, gating, no re-injection, a
different normalisation)?

**Q2. Is a contraction factor around 0.85 typical, expected, or notable** for
this class of model? Is there theory predicting trained weight-tied recurrent
transformers should be contractive — e.g. arguments via normalisation layers,
residual scaling, or training dynamics that select for stability? A pointer to
"this is well known and here is why" would be extremely valuable and would let us
demote our result to a replication.

**Q3. What exactly do people mean by a "loop" or "orbit" in this literature?**
We need the distinction sharpened between (i) a non-contracting map with a limit
cycle in the iteration index, and (ii) a cyclic path *through the layers within
one iteration*, which is compatible with each layer converging. These are
different objects and we suspect they are being conflated — including possibly by
us. Which does the literature actually measure?

**Q4. Given per-token non-uniformity of convergence, is measuring at a single
token position (the answer position) known to be misleading?** Is there guidance
on which positions carry the computation in recurrent-depth models?

**Q5. Does anything in the literature predict the residual-gap structure we saw**
— different random initialisations of the same prompt converging to nearby but
distinct endpoints, with the gap much larger for counting tasks than arithmetic
ones? Is this related to solution multiplicity, or to the task requiring
information the fixed point must hold?

**Q6. Are there architectural or inference-time conditions that induce
non-contraction** — very large depth budgets, low precision, temperature or noise
injection, adversarial prompts? If a regime change is reachable by turning a knob
we control at inference, that is directly actionable for us.

## 5. What a great answer looks like

- A clear verdict on Q1: **either** concrete instances of non-contracting trained
  looped transformers with citations, **or** a well-supported statement that no
  such instance is documented and contraction is the norm. Both outcomes are
  useful; the second lets us report a clean replication rather than a discovery.
- On Q3, an explicit disambiguation with references, because it may reveal that
  our null and the hypothesis are about different quantities — which would be the
  single most valuable thing you could tell us.
- Where possible, quantitative anchors: reported contraction rates, convergence
  iteration counts, fraction of tokens/orbits not converging.

**Mark every claim** `[V]` (you read the primary source and confirmed it),
`[T]` (title/abstract/docs only), or `[U]` (inferred/unverified). We will not act
on `[U]`. **"I could not find evidence for this" is a valid and useful answer** —
please state it rather than filling space.

## 6. Relevant work we have already partly checked

Please verify rather than trust these; our own tags are noted. Do not restrict
yourself to this list.

- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning*,
  [arXiv:2502.05171](https://arxiv.org/abs/2502.05171) — the model itself. `[V]`
  We have read its benchmark tables and the modelling code. It mentions a token's
  state falling into an "orbit pattern" in PCA projections, qualitatively, with no
  quantified prevalence — **we would very much like to know what that claim
  actually rests on.**
- A mechanistic analysis of looped reasoning language models reporting that each
  layer in the cycle converges to a distinct fixed point, so the recurrent block
  follows a consistent cyclic trajectory. `[T]` — **this is the paper most likely
  to resolve Q3.**
- A paper on per-token fixed-point convergence in depth-recurrent transformers
  reporting fast but non-uniform convergence across tokens. `[T]` — relevant to Q4.
- A paper arguing that in weight-tied looped transformers the algorithm lives in
  the *head* of the trajectory, and that instruments reading the converged tail
  provably saturate. `[V]` — relevant to whether our read window is wrong.

## 7. Out of scope

- Not asking about training looped transformers, only inference on a frozen model.
- Not asking about task selection or benchmark choice (separate inquiry).
