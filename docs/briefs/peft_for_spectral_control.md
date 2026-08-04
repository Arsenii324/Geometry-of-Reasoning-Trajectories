# Research brief: selecting a PEFT method to *steer the spectral radius* of a weight-tied recurrent transformer

**Audience.** A frontier research agent with web access. You have no prior context
on this project; everything needed is below. **Read the objective in §3 carefully —
it is not the usual PEFT objective, and an answer optimised for the usual one will
be wrong here.**

---

## 1. The decision you are being asked to inform

We will fine-tune **Huginn-3.5B** (`tomg-group-umd/huginn-0125`), a
recurrent-depth transformer, on a single Kaggle T4. We must choose:

1. the **PEFT family and variant** (LoRA and its descendants, or something else),
2. the **adapter initialisation** scheme,
3. the **target modules** inside the recurrent block,
4. the **rank / scaling** regime,
5. the **optimiser**,
6. the **loss/penalty formulation** for a *spectral* objective,
7. the **truncation scheme** for backprop through the recurrence.

We want a **defensible, specific recommendation with a fallback**, not a survey.

---

## 2. The system, stated precisely

Huginn-3.5B is a *recurrent-depth* (looped) transformer. Verified against the
model's own `raven_modeling_minimal.py` at revision `bb6621b6…`:

- **Structure:** `embed → prelude(2 layers) → [core_block(4 layers)] × r → coda(2 layers) → ln_f → lm_head`.
- **`core_block` is weight-tied and applied `r` times.** `r` is sampled per training
  step (log-normal-Poisson; `mean_recurrence = 32`) and is chosen freely at
  inference. This is the crux of the whole brief.
- **Context re-injection:** the prompt embeddings are concatenated back into the
  state at *every* unroll, which makes the core an autonomous map
  `h_{t+1} = F_e(h_t)` parameterised by the (fixed) prompt embedding `e`.
- **RMSNorm terminates every block**, so the state lies on a sphere:
  `‖h_t‖ = 76.37 ± 0.007` (0.007% variation). Radial dynamics do not exist; all
  dynamics are angular.
- `n_embd = 5280`. Total ≈ 3.5B parameters.
- **Training used truncated backprop through a sampled window** of the last *k*
  unrolls, not through all `r`.
- **Initialisation** (their own `_init_weights`, not an HF default):
  `std = √(2/(5·d))`, `out_proj_std = std / √(2·effective_expected_depth)`,
  truncated normal at ±3σ, RMSNorm weights = 1. Note the **depth-scaled** output
  projection — the scaling accounts for the *unrolled* depth, not the 4 physical
  layers.

**Primary source:** Geiping et al., *"Scaling up Test-Time Compute with Latent
Reasoning: A Recurrent Depth Approach"* (arXiv 2502.05171). Please verify the
recurrence, sampling and truncation details against the paper and the released
modelling code rather than against secondary summaries.

---

## 3. The objective — read this twice

**We are not trying to improve task accuracy.** A few hundred steps on a free-tier
T4 will not deliver capability gains, and we are not asking for them.

We want to **change, in a controlled and measurable way, the contraction rate ρ of
the iterated map** — the spectral radius of the Jacobian `∂h_{t+1}/∂h_t` of the
weight-tied core block at its fixed point.

Why ρ specifically. In this model we measured, across **14 independent
weight-sets** (5 random inits, 9 published training checkpoints of the same run),
using two-orbit convergence `‖h_t^{(1)} − h_t^{(2)}‖ ~ ρ^t`:

| arm | ρ | n |
|---|---|---|
| randomly initialised | **0.7048 ± 0.0087** | 5 |
| trained | **0.8577 ± 0.0139** | 9 |

Complete separation (Mann-Whitney U=0, p=5×10⁻⁴, Cohen d=13.2). Independently,
Arnoldi on Jacobian-vector products gives ρ ≈ 0.80, and the leading eigenvalue is
**complex** in 3/3 prompts (rotation period ≈2.6–6.0 unrolls).

**ρ is the only quantity we have found that training demonstrably changes** — every
content-level property we measured (linear decodability of task variables,
effective dimensionality, endpoint geometry) is matched or *exceeded* by a randomly
initialised model. Slower contraction is also what makes test-time depth do
anything: a map with ρ=0.70 has a time constant of 3.0 unrolls and is finished by
r≈8 however many you give it, whereas ρ=0.86 gives 6.6 unrolls and is still moving
at r=32.

**Every ρ result we have is observational** — read off weight-sets other people
produced. A PEFT run that *moves ρ by a predicted amount* would be the first
intervention on the quantity, which is the entire scientific motivation.

**Success = a measurable, attributable Δρ under a controlled edit.** Capability is
not a success criterion and need not be preserved (though catastrophic divergence
is a failure — see §5).

---

## 4. Why this is not a standard PEFT question

Please do not answer this as "which LoRA variant scores best on downstream
benchmarks". Four things make it different, and a recommendation that ignores them
is not usable:

1. **The adapter is applied `r` times per forward pass.** The core block is
   weight-tied, so a rank-*r* update ΔW does not perturb one layer's output — it
   perturbs the *map being iterated*. Effects on the fixed point compound over
   unrolls in a way that standard single-application LoRA analysis does not model.
   We would like to know whether any existing work analyses this.
2. **The target is a spectral property of a Jacobian, not a token-level loss.**
   This is closer to Lipschitz-constrained training, spectral normalisation, and
   **Deep Equilibrium Model (DEQ) stabilisation** than to instruction tuning. We
   believe *Bai, Koltun & Kolter, "Stabilizing Equilibrium Models by Jacobian
   Regularization"* and the surrounding DEQ literature is directly relevant — it
   solves the problem of controlling the spectral radius of an iterated map — and
   we would like it evaluated for transfer, including its Hutchinson-style
   stochastic Jacobian-norm estimator and whether that estimator is usable as a
   *training* penalty at this scale.
3. **Stability is a hard constraint, not a preference.** ρ must stay below 1. An
   edit that pushes ρ ≥ 1 makes the model non-convergent and the run worthless.
4. **The state is on a sphere** (RMSNorm terminates every block). Any analysis
   assuming unconstrained residual-stream geometry may not transfer.

---

## 5. Hard constraints

| | |
|---|---|
| Hardware | Kaggle free tier: **NVIDIA T4, ~14.56 GiB usable**, 2 concurrent notebooks, ~30 GPU-h/week, ≤9–12 h per kernel |
| Precision | **T4 is Turing: fp16 yes, bf16 NO.** Huginn was trained in bf16. We have separately measured that this model is unusually precision-sensitive: bf16 rounding makes it *appear* to converge ~4.6× sooner than it does, so any ρ measured under reduced precision needs its own precision check. |
| Memory | 3.5B in 4-bit NF4 ≈ 2 GB, so weights are not the blocker. **Activation memory through `r` unrolls is.** |
| Framework | PyTorch + HF `transformers` + `peft`; `trust_remote_code` custom architecture, so `target_modules` must be named explicitly. |
| Failure mode to avoid | ρ ≥ 1 (divergence), or a Δρ that cannot be attributed to the edit rather than to precision/estimator noise. Our ρ estimator has a measured upward bias of +0.002…+0.015 and a between-weight-set sd of ~0.014, so **a credible Δρ must exceed ~0.03.** |

---

## 6. Questions, in priority order

**Tier 1 — these determine whether the experiment is possible at all.**

1. **Is there any published work on PEFT / low-rank adaptation of weight-tied,
   looped, or recurrent-depth transformers** (Huginn, Universal Transformers,
   looped transformers, ACT/PonderNet, DEQs)? If yes, what does it say about
   adapters that are re-applied every iteration? If genuinely none exists, say so
   plainly — that is a useful answer.
2. **What is the best way to make ρ (or a differentiable surrogate) trainable?**
   Candidates we know of: in-graph power iteration on Jacobian-vector products;
   Hutchinson estimators of `‖J‖_F` (the DEQ-regularisation route); spectral norm
   of ΔW as a crude proxy. Which is tractable at 3.5B on one T4, and what is the
   bias of each as a stand-in for the *spectral radius* rather than a norm?
   (`‖J‖_F ≥ ρ` always — how loose is that in practice?)
3. **Which modules inside the block most control ρ?** Attention output projection,
   MLP/GLU down-projection, QKV, the RMSNorm gains? Is there theory or measurement
   on which weights dominate the Jacobian's spectral radius in a transformer block?
   RMSNorm gains are a tiny parameter set with a plausibly direct scaling effect —
   is tuning *only* those a known, viable method?

**Tier 2 — method selection.**

4. **Which LoRA descendant** for this objective? We are aware of **PiSSA**, **DoRA**,
   **rsLoRA**, **LoRA+**, **MoRA**, **VeRA**, **OLoRA**, **EVA**, **LoftQ**,
   **HQQ+**. Rank them *for spectral control under repeated application*, not for
   benchmark accuracy. Specifically: does DoRA's magnitude/direction decomposition
   give a more direct handle on a spectral scale than plain LoRA? Does PiSSA's
   principal-singular initialisation matter more than usual when the adapter is
   applied 32× (a poor start compounds)? Is MoRA's high-rank-at-equal-parameters
   design relevant if a low-rank ΔW turns out unable to move ρ?
5. **Optimiser.** Is there evidence for **Muon** (orthogonalised momentum for 2-D
   parameters), **SOAP**, **Adam-mini**, or **Shampoo**-family methods on adapter
   matrices, and does any of them interact well or badly with a spectral penalty?
   Muon's orthogonalisation constrains the update's singular values — is that
   helpful or actively counterproductive when the *goal* is to change singular
   structure?
6. **Truncated backprop.** Huginn trained with a sampled window. For a *spectral*
   objective, does truncating the unroll bias the gradient of ρ, and is there a
   principled window length? DEQ work uses implicit/one-step gradients — does that
   apply when we deliberately do **not** run to convergence?

**Tier 3 — worth knowing, not blocking.**

7. Would **breaking weight-tying** (a separate adapter per unroll, or per unroll
   group) be a better instrument, or a category error that changes the object of
   study? Any precedent?
8. Known **fp16 failure modes** for PEFT at 3B+ scale, and the standard mitigations
   (loss scaling, fp32 master weights, keeping norms in fp32).
9. Any work **measuring** — not just bounding — how the spectral radius of a
   trained transformer's Jacobian changes over training. Our finding that ρ rises
   0.705 → 0.858 with **91% of the change occurring before the first published
   checkpoint (step 6144)** may or may not be novel; we would like to know.

---

## 7. What a strong answer looks like

- **A concrete primary recommendation**: method + init + target modules + rank +
  scaling + optimiser + LR + penalty formulation, with reasoning tied to §3–§4,
  plus **one fallback** if the primary is infeasible on a T4.
- **Citations to primary sources** (papers, official repos, model code), with dates.
  Please distinguish clearly between *confirmed against the primary source*,
  *plausible but unverified*, and *contradicted*. We have twice been burned this
  project by trusting a secondary summary over the actual source.
- **Explicit statements of what will NOT work and why** — negative guidance is as
  valuable here as positive.
- **Quantitative where possible**: memory estimates for the T4, expected magnitude
  of achievable Δρ if anything is known, parameter counts.
- **An honest "no literature exists on X"** wherever that is the truth. We would
  much rather hear that than receive a forced analogy.

## 8. What a weak answer looks like

- A ranked survey of LoRA variants by GLUE / commonsense / MMLU scores. Our
  objective is not downstream accuracy and those rankings do not transfer.
- Anything that ignores that the adapter is applied `r` times per forward.
- Anything assuming bf16, or >14.5 GiB of VRAM.
- Recommending a method without saying how ρ becomes differentiable (question 2 is
  the load-bearing one).
