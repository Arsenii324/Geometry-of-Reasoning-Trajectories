# Deep research inquiry #3: causal interpretability methods that actually work on weight-tied recurrent-depth models

**The question.** Standard mechanistic-interpretability tooling assumes a stack of
**distinct** layers. Our model applies **the same 4 blocks 32–64 times**. We need
to know which causal methods survive that change, which break, and what the
recurrent-depth-specific literature has already established — **before** we spend
more GPU budget re-deriving it.

---

## 1. The system, and why it is not a normal transformer

**Huginn-3.5B** (`tomg-group-umd/huginn-0125`,
[arXiv:2502.05171](https://arxiv.org/abs/2502.05171)):

```
embed -> prelude(2 blocks) -> [core_block(4 blocks)] x r -> coda(2 blocks) -> ln_f -> lm_head
```

Properties that break standard assumptions:

- **Weight-tied recurrence.** There is no "layer 17" to attribute anything to —
  there is one function applied 32+ times. "Which layer does X" is ill-posed;
  the meaningful axis is **iteration index**, which shares parameters.
- **Depth r is chosen at inference**, so the computational graph's length is not
  fixed.
- **The initial latent is random and unseeded**, so the same prompt run twice
  gives different trajectories and sometimes different answers.
- **The prelude output is re-injected every iteration**, so information is
  continually re-supplied rather than only propagated.
- Empirically the iteration is a **contraction** (per-step factor ≈ 0.855, no
  exceptions found across ~960 orbit pairs). **A perturbation injected at
  iteration r decays and is completely gone by iteration 32.**

That last point is the practical crux: **activation patching late in the
recurrence may be causally inert by construction**, because the contraction erases
the intervention before the output head reads it. We observed exactly this — a
patch changed the trajectory transiently and never changed the final answer.

## 2. What we have tried, and how it went

- **Activation patching**: replace the answer-position state at iteration r with
  the corresponding state from a run of the same prompt that got the answer
  *right*, sweeping r. Our implementation is validated (a no-op replay reproduces
  the original run exactly). The experiment nonetheless failed its
  pre-registered gate because too few prompts had outcomes that varied at all —
  a *task supply* problem, not an implementation problem. Separately, the
  patch effect decayed to nothing by iteration 32.
- **Reading trajectory shape** to decode correctness: mostly null under properly
  stratified, permutation-tested analysis.
- **A query–key alignment probe** at the answer position: running now.
- **Sparse autoencoders**: deliberately deprioritised — training one on 5280-d
  residuals is beyond our budget and we would have no ground truth to validate the
  dictionary against.

## 3. The questions, in priority order

**Q1 (most important). What causal interpretability has actually been done on
recurrent-depth, weight-tied, or looped transformers** (Huginn, Ouro, Universal
Transformers, looped/CoT-looped transformers, deep equilibrium models)? Which
techniques did they use, what worked, and what did they report as failing?

**Q2. How should activation patching be adapted when the map is contractive?**
If interventions are provably washed out, patching late is meaningless and only
early interventions can matter. Is there established practice here — patching the
*re-injected* stream rather than the state, patching at all positions rather than
one, patching a whole iteration's output, or intervening on the fixed point
itself? Related: is there work on causal intervention in **deep equilibrium
models**, where the object is a fixed point rather than a path? That literature
seems structurally closest to our situation and we have not surveyed it.

**Q3. What is the right unit of attribution when weights are shared across
iterations?** Standard circuit analysis attributes to (layer, head). Here the same
head appears 32 times. Is there published guidance on attributing to
**(iteration, head)**, on treating the recurrence as an unrolled graph, or on
whether iteration-indexed attribution is even well-defined?

**Q4. Do steering vectors / activation addition / function vectors work in this
setting?** We are interested because we found the trajectory's *shape* responds to
a task-marker token even when the marker selects the *same* computation — which
raises the question of whether a "task direction" exists in state space that could
be added to make the model perform task B under marker A. Has anything like this
been done on a recurrent-depth model? Would the contraction destroy an added
direction the same way it destroys a patch?

**Q5. Is there a method that exploits, rather than fights, the recurrence?**
Since the same function is applied repeatedly, tools from dynamical systems —
fixed-point analysis, Jacobian spectra, basin structure, bifurcation with respect
to inputs — might be more natural than layer-wise circuit tracing. Has anyone done
mechanistic interpretability *as dynamical systems analysis* on a trained LM, and
did it yield anything a circuit analysis would not?

**Q6. Given our model's random unseeded initial latent, is there interpretability
work that treats the initialisation as an experimental handle?** We use it to get
within-prompt outcome variation. Are there pitfalls, or better uses?

## 4. What a great answer looks like

For each recommended method:

> **Method** — what it does, in one sentence.
> **Evidence it works on recurrent/weight-tied/equilibrium models**, with citation.
> **What it needs**: number of forward passes, gradients or not, training or not.
> **Why it survives (or doesn't) a contractive map.**
> **Concrete first experiment** we could run on a frozen 3.5B model on one GPU.

We would rather have **two methods with real evidence and a runnable first
experiment** than a taxonomy of ten.

**Mark every claim** `[V]` (primary source read and confirmed), `[T]`
(title/abstract only), `[U]` (inferred). We will not act on `[U]`. **If the honest
finding is "almost no causal interpretability has been done on this model class",
say so** — that is itself decision-relevant, and would tell us we are in open
territory rather than behind the field.

## 5. Constraints on anything you propose

- The model is **frozen and public**; no fine-tuning, no retraining.
- Budget is roughly **one T4/A100-class GPU for hours, not days**. Methods needing
  large-scale dictionary training are out unless you can argue the payoff.
- We read outcomes as **the rank of the gold answer's first token** at a single
  position, at every iteration (cheap: one forward gives all depths).
- We have banked raw trajectories on disk (hundreds of orbits, full state at the
  answer position for every iteration), so **methods that work offline on
  recorded states are especially attractive** — they cost no GPU at all.

## 6. Out of scope

- Task/benchmark selection (separate inquiry).
- Whether the model's dynamics contract (separate inquiry) — assume they do.
- Training-time interpretability or checkpoint-comparison studies.
