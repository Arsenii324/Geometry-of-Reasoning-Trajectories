# Related work — with verification status per item

*Opened 2026-08-09. Every entry carries how it was checked. This project has already
had one citation ("Movahedi et al.") that could not be located, and a scouting pass
surfaced several 2026 arXiv numbers read through summarisers rather than PDFs.*

**Verification levels**
- **[V]** — I fetched the arXiv page myself and the quoted content is in what I read.
- **[T]** — Title, authors and abstract verified by me; deeper claims (appendix
  numbers, which model was used) **not** verified.
- **[U]** — Reported by a scouting agent from its own read; **not** verified here.
  **Nothing at [U] may enter `claims_ledger.md` or a paper draft without a PDF check.**

---

## Directly on Huginn / recurrent depth

| item | status | what it says | what it does to us |
|---|---|---|---|
| Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning* (huginn-0125), [arXiv:2502.05171](https://arxiv.org/abs/2502.05171) | **[V]** model card + `raven_modeling_minimal.py` at the pinned revision + paper body (HTML, fetched 2026-08-09) | The model itself. Source facts we rely on: `initialize_state` is truncated-Gaussian noise shaped like the embeddings; the **prelude output** is re-injected through the adapter every unroll; `block_idx` reaches only the KV-cache slot, so the recurrence is an **autonomous map**. Training mixture includes `tomg-group-umd/CLRS-Text-train`. **Benchmark numbers, quoted (answers A10):** Table 1, final model, r=32: ARC-E 69.91, ARC-C 38.23, HellaSwag 65.21, MMLU 31.38, OpenBookQA 38.80, PiQA 76.22, SciQ 93.50, WinoGrande 59.43. Table 2, r=32: GSM8K flexible/strict 28.05/28.20, GSM8K-CoT 32.60/34.57, Minerva MATH 12.58, MathQA 26.60; with weight averaging at r=64, GSM8K rises to 47.23% flexible / 38.59% strict. Table 4, early checkpoint, r=1 vs r=32: ARC-E 34.01→53.62, HellaSwag 29.19→48.80, GSM8K-CoT 0.00/0.15→9.02/10.24 — genuine, substantial task accuracy that scales with recurrence depth, not a floor. Also: *"the state of the token quickly falls into an orbit pattern in all three pairs of PCA directions"* (qualitative, no fraction quantified in the body). | Foundation. The autonomous-map fact is what makes H1 ill-posed *given* ρ<1, and the CLRS entry is why `scratch/kaggle_clrs/` exists. **The benchmark numbers show Huginn genuinely solves natural-language reasoning tasks in its training distribution (GSM8K, ARC, HellaSwag) at non-trivial rates that scale with r — so a capability floor is real on SOME tasks, not universal.** This does NOT directly disambiguate this project's battery, since GSM8K/ARC are natural-language and in-distribution while the battery is synthetic/symbolic (ciphers, parity, indexing) and largely OOD — so the battery's near-zero families could still be genuine OOD difficulty, D89-style scoring artefacts, or both; the paper doesn't resolve which. |
| Blayney et al., *A Mechanistic Analysis of Looped Reasoning Language Models*, [arXiv:2604.11791](https://arxiv.org/abs/2604.11791) | **[T]** for the paper; **its central claim is now [V] BY OUR OWN MEASUREMENT** — D98 hooked all 4 core blocks and found each converges to its own fixed point with the four separated by ~52% of the state norm (ratio 95-730). | Abstract, verbatim: *"each layer in the cycle converges to a distinct fixed point; consequently, the recurrent block follows a consistent cyclic trajectory in the latent space."* | **Scoops "depth is a contraction."** But read the object carefully: that is a cycle **across the 4 core layers**, not rotation of the state at a fixed layer. This project reads out at `core_block[-1]` only, so it measures one layer's convergence — a different quantity. Demote our contraction result to replication and state the distinction. |
| *ibid.*, **Appendix C, Tables 3–4 — RESOLVED 2026-08-09** | **[V]** (full HTML read in a sourced literature pass; supersedes the three conflicting fragments previously circulating in our notes) | Non-fixed-point prevalence on Huginn-0125 over GSM8K. **Table 3 (% of tokens):** no system prompt 0.02% (0.01 orbit / 0.01 slider); Short Math 0.01%; Long Persona 0.14% (0.13 orbit); Long Persona Padded 0.05%. **Table 4 (% of examples with ≥1 such token):** Long Persona 2.81% (2.50 orbit), no system prompt 0.76%, Short Math 0.45%. **Criterion matters more than the number and must always be quoted with it:** a heuristic classifier on per-token cosine similarity of the final recurrent layer's output across 128 recurrences vs the final residual stream, τ=0.05, fixed-point fraction 0.9; the authors warn absolute values "vary significantly depending on the algorithm hyperparameters." | **The apparent contradiction between our three recorded versions was per-token (Table 3) vs per-example (Table 4), not an inconsistency.** Authoritative figure for non-fixed-point prevalence; corroborates our H1 nulls in **sense (i)** while the same paper's Prop. 4.1 supplies **sense (ii)**, which D98 independently measured. |
| *ibid.*, **Prop. 4.1 + the untrained-model finding** | **[V]** via the same pass; **Prop. 4.1's content independently measured by us in D98** | "Cyclic fixed point": the per-layer fixed points "are not necessarily the same … the cycle of layers can instead trace out an arbitrary cycle in latent space." Separately: cyclic-fixed-point behaviour is **emergent from architecture and appears in randomly-initialised untrained models** when input injection is present. | **This is the source that predicts D98's untrained control.** If the cycle exists untrained, D98's *existence* claim is architectural and only the cycle's geometry could be a fact about training. `scratch/ds_cyclenull/` tests exactly this. |
| *Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers*, [arXiv:2607.14427](https://arxiv.org/abs/2607.14427) | **[T]** (title + search abstract) | Successive-output KL falls 3.9e-1 → 8.5e-6 by loop 16; median token converged by loop 6; ~10% of tokens still updating at depth 8. | Independent confirmation that convergence is fast and **non-uniform across tokens**. The non-uniformity matters: we read at one position. |
| *When Does Recurrence Become an Algorithm? Convergence Selection in Weight-Tied Looped Transformers*, [arXiv:2607.20594](https://arxiv.org/html/2607.20594v1) | **[V]** | *"tail instruments only seeing the converged region where they provably saturate"*; the algorithm, if there is one, lives in the **head** of the trajectory. | **The strongest objection to our nulls, and the best explanation of D80.** It predicted D91's window location before we measured it. `geometry-h0bank` + `scripts/run_h0_within.py` is the pre-registered test. |
| *Adaptive Depth in Looped Transformers*, [arXiv:2607.20519](https://arxiv.org/abs/2607.20519) | **[T]** | Halting gates and trajectory readouts on Ouro-1.4B/2.6B. | Adjacent; a second architecture to generalise to if anything survives. |

## The three senses of "loop" — the project's standard vocabulary from 2026-08-09

Adopted from a sourced literature pass. Conflating these is what made our H1 null
look like it contradicted the supervisor's hypothesis.

| sense | what it is | status in the literature | our measurement |
|---|---|---|---|
| **(i)** genuine **non-contracting** limit cycle, \|λ\|=1 | state at a fixed layer returns to itself across recurrences | **never measured in a trained looped LM.** ρ≥1 appears only as a *training-time* failure mode that papers then eliminate (Parcae, STARS) | **D94, D97 — absent.** A clean replication of the field's norm, not a discovery |
| **(ii)** cycle **across layers** within one iteration | each layer converges to its **own** fixed point; the k-block cycle traces a closed path while every component contracts | Blayney Prop. 4.1; also emergent in **untrained** models with input injection | **D98 — present and large** (four fixed points ~52% of a state norm apart) |
| **(iii)** **damped rotation**, complex λ with \|λ\|<1 | inward spiral that shape heuristics label "settle" | "Fixed-Point Reasoners" has an explicit damping section; Pappone's "spiral-like iterate behaviour" | **D55 — present** (complex leading Jacobian eigenvalue 3/3 prompts, 27/30 top modes complex, period 2.6–6.0 unrolls) |

**Consequence for how we write.** "No loop, no drift" is a misleading summary of
D94/D97 and must not be repeated: the map *does* rotate (iii) and there *is* an
inter-block cycle (ii). What is absent is sense (i) — and no one has ever measured
sense (i) in a trained looped LM, so absence is the expected result.

## Contradicting the outcome null

| item | status | what it says | our answer |
|---|---|---|---|
| Three papers reporting correctness decodable from trajectory geometry (AUC ≈ 0.90–0.91; TRACED AUROC 0.75–0.83; scale-invariant shape features beating MSP on 41/45 pairs) | **[U]** | Correctness is readable off the trajectory. | **They obtain within-prompt variance from stochastic rollouts.** Huginn's decode is deterministic and our main block draws one h₀ per item, so our null was scoped to *between-prompt* variation. D90 found the missing handle (h₀), and `geometry-h0bank` runs their design. IDs and numbers unverified — check before citing. |
| Zhou et al., carrier-invariant design: position clusters by surface form (0.85 vs 0.26), **curvature clusters by logic** (0.53 vs 0.11–0.13) | **[U]** | The opposite decomposition to ours. | If real, the sharpest antagonist. Defences to check against the PDF: layer-wise paths in feedforward models vs unroll paths under input injection; clustering similarity vs a discriminative classifier. **Also kills any claim that "the input control is the field's gap"** — we would not be first. |

## What survives clean

- **D88's same-computation control.** Nothing found in this literature holds the
  *computation* fixed while varying one token. The nearest [U] item holds *content*
  fixed. This is the project's strongest single contribution.
- **D85's reliability-bounded null.** None of the antagonist papers bounds its own
  detection floor. Ours measures the correlation a perfect relation would have shown.
- **D86.** Exact-match → 0% by r=8 while containment rises to 33–39% is a
  decomposition nothing found here performs.

## Do not cite

- "Du et al. (2025)" on correct-vs-incorrect trajectory separability — **no scout
  could locate it.** Dropped, not rebutted.
- "Movahedi et al." — already recorded in this repo as unlocatable.
