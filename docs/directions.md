# Directions register

**Purpose.** Anything recognised as worth doing goes here the moment it is
recognised, not into working memory. Context gets compacted; this file does not.
A direction that exists only in a conversation is lost at the next compaction, and
this project has already re-derived things it had previously noticed.

**How this differs from the neighbouring documents.**

| file | holds |
|---|---|
| `claims_ledger.md` | what was measured, with evidence and retractions |
| `backlog_not_done.md` | specific experiments considered and not run, with reasons |
| **`directions.md`** | **open threads, requests, and ideas — with status and owner** |
| `state_of_knowledge.md` | the synthesised picture |

**Status vocabulary.** `open` · `in-progress` · `done` (with the ledger row or
file that closed it) · `parked` (with why) · `dropped` (with why).

---

## A. Standing requests from the curator/user

| # | request | status | where it lives |
|---|---|---|---|
| A1 | Audit for helpful scaffolding — templating, backgrounding, less verbose output, without cutting observability | **done** — §C, all 8 items | §C below; `scratch/_lib/`, `scripts/build_kernel.py`, `tests/test_kernel_common.py` |
| A2 | Caesar cipher as a task, several fundamental variants | **open, designed** | §B1 |
| A3 | Observability metrics — do we need more, fewer, or different ones to spot the unexpected | **done** — `docs/observability.md`; it paid for itself immediately (D55) | §D |
| A4 | How are "random" weights selected? Is Huginn's init scheme the right one for a recurrent transformer? | **partly answered** | §B2 |
| A5 | Is SoTA parameter-efficient fine-tuning feasible on Kaggle for a 3.5B recurrent model? | **assessed (§B3); research brief written for an external agent** | §B3, `docs/briefs/peft_for_spectral_control.md` |
| A6 | Keep recognised ideas in a file, not in working memory | **done** | this file |

**Standing constraints** (do not re-litigate): push only to `Arsenii324`, never
`origin`. Kaggle free tier, ~30 GPU-h/week, max 2 concurrent notebooks — **treat
GPU as plentiful**; the scarce resource is AI tokens, so avoid subagents/workflows
unless the task genuinely needs judgement that cannot be computed.

---

## B. Research directions

### B1. Caesar cipher task family — **the highest-value open direction**

**Why it matters more than it looks.** Every content-level finding in this project
was deflated by an untrained control (D40, D41, D48, D53), and the one capability
claim that would have made D41 sharp — "decodability and capability move in
opposite directions" — had to be **retracted** because the counting task has *no
capability contrast*: the trained model scores 1/60 and 0/120 at M=64, the
untrained 0/60, Fisher p=1.0 (D41(3), D47). **A task where the trained model
measurably succeeds and the untrained cannot is the missing ingredient for half
the project's claims.** Caesar decoding is a strong candidate: it is genuinely
solvable by a trained LM, trivially impossible for random weights, and its
difficulty is continuously controllable.

**Variants to run** (the point is that these differ in *what kind of work* they
require, not merely in difficulty):

| variant | axis it isolates |
|---|---|
| shift given vs shift must be inferred | retrieval/apply vs **search over 25 hypotheses** — the second should need more depth if depth is search |
| `"Caesar cipher"` named vs full explicit instruction vs no framing | how much of the work is task identification |
| space-separated letters vs natural spacing | tokenisation burden separated from the cipher itself |
| short (one word) vs long (a sentence) | evidence accumulation; a long text over-determines the shift |
| shift = 1, 13, 25 | 13 is self-inverse (ROT13) and likely memorised; 1 and 25 are near-identity in letter space but not in token space |
| real English plaintext vs random letter strings | how much is language-model prior vs cipher computation |
| several rephrasings of the same instruction | prompt-form sensitivity as a nuisance axis to measure, not eliminate |

**Pre-registered predictions, written before running** (so the result can fail):
1. Trained accuracy > 0 on at least the *shift-given, short, ROT13* cell.
   Untrained accuracy = 0 everywhere. → supplies the capability contrast.
2. Required depth is **higher for shift-inferred than shift-given** at matched
   text length. If depth is search, this must hold; if it does not, "depth =
   search" is wrong for this architecture.
3. Per D52, the untrained model's states converge by r≈8 and the trained by r≈32,
   so any depth effect beyond r≈8 cannot be architectural.
4. The plaintext-letter identity will be linearly decodable from the state
   *earlier* than the model can emit it — the availability/use gap again (D38,
   D54). Untrained control required before claiming it.

**Instrumented follow-up, built and verified locally (`scratch/kaggle_caesar2/`),
launches once the screen reports.** The screen gives one number per cell, which is
enough to decide whether to continue and nothing else. The follow-up keeps **one
record per output character** with four covariates that are free — they are
properties of data already being generated:

- **wrapped** — decoding needs the modular reduction exactly when the ciphertext
  letter index is `< k`. A model doing real modular arithmetic is indifferent;
  one doing naive subtraction fails precisely there. Verified: exactly `k` of 26
  letters wrap, so the shift sweep also sweeps how often it matters (k=1 → 3.8%,
  k=13 → 50%, k=25 → 96%). **`char_acc | wrapped` vs `| not wrapped` is a direct
  test of computation versus approximation.**
- **shift** — k=13 (ROT13) and k=3 are *named* and over-represented in any corpus;
  k=1 and k=25 are not. Accuracy spiking only at 13 would mean retrieval.
- **prior** — `classic` (famous strings plausibly present verbatim as ROT13
  examples) / `fresh` (ordinary English) / `random` (letter strings that cannot be
  memorised or guessed). **Without the random arm, high accuracy on English is
  uninterpretable.**
- **tokenisation** — spaced letters are one token each; joined text is a ragged
  many-to-many map. Separates the cipher from the tokeniser.

24 cells × 10 items × 2 arms. Per-character records are written out so the analysis
can be redone locally without a rerun.

**Cheap first step (do this before any sweep):** one kernel, ~6 cells × 20 items,
scoring accuracy only. If trained accuracy is 0 everywhere, the whole family is a
counting-task repeat and should be **dropped rather than elaborated** — that is the
decision this step exists to make.

### B2. What "random weights" means here — **partly answered, 2026-08-04**

**Answered.** Huginn is not initialised by a generic HF scheme. Its own
`raven_modeling_minimal.py` defines `_init_weights`, and `from_config` invokes it,
so the untrained arm *is* "Huginn at step 0" under the paper's scheme:

```
std        = sqrt(2 / (5 * dim))                        # dim = 5280
out_proj   = sqrt(2 / (5 * dim)) / sqrt(2 * num_layers)  # DEPTH-SCALED
embedding  = sqrt(2 / (5 * dim))
embed_scale= sqrt(dim)
```
truncated normal, `a=-3σ, b=+3σ`; RMSNorm weights set to 1; `num_layers =
config.effective_expected_depth`, i.e. the scaling accounts for the *unrolled*
depth, not the 4 physical core layers. That depth-scaled `out_proj` is the
recurrence-aware part and is what keeps a deep unroll stable at init.

**Still open.** (a) Verify empirically on GPU that a `from_config` model's
parameter statistics match those formulas — cheap, and it is the difference between
reading the code and checking the model. (b) The untrained arm samples **one init
family**; D52's five seeds vary the draw but not the scheme. Whether ρ≈0.705 is a
property of *this* init or of random weights generally is untested — a sensible
control is to vary `std` by ±2× and see whether ρ moves. If it does, "untrained
ρ = 0.705" is really "ρ at Huginn's chosen init scale".

### B3. Parameter-efficient fine-tuning — **assessed; the reason to do it is ρ, not capability**

**The scientifically interesting fact, which is specific to this architecture.**
Huginn's core block is **weight-tied across unrolls**. An adapter on it is therefore
applied *N times per forward pass*, so a rank-r perturbation ΔW does not shift the
computation once — it changes the **iterated map** `h ← F(h; W+ΔW)`, and hence
directly changes the Jacobian and its spectral radius. D52 established that **ρ is
the one quantity training demonstrably changes** (0.7048 → 0.8577, complete
separation over 14 weight-sets), and D55 that its *argument* sets the rotation.

> A LoRA on the core block is the most direct causal handle on ρ available. Every
> ρ result so far is observational — read off weight-sets someone else produced. A
> short PEFT run that *moves* ρ by a predicted amount would be the project's first
> intervention on its own central quantity.

Concretely testable: add an explicit penalty on the measured contraction rate
(estimable in-graph from two orbits) and see whether ρ can be steered, and what
happens to accuracy and to decodability when it is. **That experiment does not
require the fine-tune to make the model better at anything** — which is fortunate,
since a few hundred Kaggle steps will not.

**Methods worth using (not vanilla LoRA).**

| method | why here |
|---|---|
| **PiSSA** | initialises the adapter from the *principal* singular directions of W, so training starts in the high-energy subspace instead of at zero — matters when the adapter is applied 32× and a bad start compounds |
| **DoRA** | decomposes into magnitude × direction; the magnitude term is a near-direct scale knob on the block, which is plausibly the closest thing to a ρ dial |
| **rsLoRA** | rank-stabilised scaling (γ = α/√r rather than α/r) — the standard α/r scaling misbehaves at higher rank, and rank is a variable we would want to sweep |
| **LoRA+** | different learning rates for A and B; cheap and reliably better than equal-rate |
| **EVA / OLoRA** | data-driven or orthonormal init; alternatives to PiSSA worth one comparison, not three |
| **MoRA** | high-rank update via a square matrix at equal parameter count — relevant if a low-rank ΔW turns out unable to move ρ at all, which is itself a finding |

**Optimisers.** **Muon** (orthogonalised momentum for 2-D parameters) is the
strongest recent default for matrix-shaped weights and is a natural fit for
adapter matrices; **SOAP** and **Adam-mini** are reasonable fallbacks; **GaLore**
only if optimiser memory ever binds, which at adapter scale it will not.

**Feasibility on Kaggle: yes, with three architecture-specific obstacles.**

1. *Memory is not the blocker.* 3.5B in 4-bit NF4 ≈ 2 GB against a T4's 14.56 GB.
2. *The recurrence is.* Backprop through `num_steps` unrolls of a weight-tied block
   multiplies activation memory by the unroll count. **Huginn's own training used
   truncated backprop through a sampled window**, and any fine-tune must do the same
   (short window + gradient checkpointing). This changes what is being optimised and
   must be stated, not glossed.
3. *T4 is Turing: fp16 yes, **bf16 no**.* Huginn trained in bf16, which has the same
   exponent range as fp32; fp16 does not. D30 already showed this model is unusually
   precision-sensitive (bf16 rounding makes it *appear* to converge 4.6× sooner than
   it does), so loss scaling or fp32 master weights are mandatory, and any ρ measured
   under fp16 needs its own precision check.

**A research brief for an external frontier agent** is at
`docs/briefs/peft_for_spectral_control.md`. It is self-contained and states the
objective as *spectral control*, not capability, because a survey answering the
usual PEFT question would not be usable here.

**Order of work, if started:** (a) confirm a LoRA on the core block moves ρ at all;
(b) if it does, sweep rank and see whether Δρ scales with capacity; (c) only then
ask whether a ρ-targeting penalty can steer it. Stop at (a) if the answer is no —
that would itself say the contraction rate is not reachable by low-rank edits, which
is worth knowing.

### B4. Carried over from `backlog_not_done.md`

| # | direction | status |
|---|---|---|
| B4.1 | Is ρ constant across task families? | **open** — D45's claim was retracted; n=3/family had no power |
| B4.2 | Recompute D46's arm comparison with the corrected window | **done** → D53 |
| B4.3 | Untrained control for D35 | **dropped** → D54(3): degenerate by construction, the untrained model has no answer process |
| B4.4 | Why are 8/108 trained orbit fits below the R²>0.9 bar while 0/60 untrained are? | **open** — a second, unexplained way the trained operator differs (D52(3)) |
| B4.5 | What happens to ρ *before* step 6144? | **parked** — no published checkpoint exists; would need training from scratch |
| B4.7 | **Measure accuracy on the 8 task families that never had it** (parity, running-max, projection, three-scale, three-scale-modk, running-count, nesting-depth, count-ones). One generation pass each. | **open — the single highest-value gap** (D56(2)); decides whether 8 tasks' worth of geometry describes reasoning or failure |
| B4.8 | **Give persistent homology a null, or drop it.** 15 trajectories, H₁ 0–26, no surrogate ever computed. | **open** (D56(3)) — the last untested topological claim, and the most distinctive thing the project set out to look at |
| B4.9 | **Sweep the Jacobian spectrum properly**, n=3 → tens of prompts, both arms. Magnitudes gave ρ (D31), arguments gave rotation (D55). | **open** — best value per unit cost in the project |
| B4.10 | **Re-operationalise or retire H1's regime labels.** As they stand they partition without separating (Kruskal–Wallis p=0.48). | **open** (D56(1)) |
| B4.6 | **Jacobian eigenvalue ARGUMENTS.** D31 ran Arnoldi on J-vector products and reported only the magnitude (ρ≈0.79–0.81). A contracting map rotates iff its eigenvalues are complex, at a rate given by their argument — a quantity needing no trajectory, no window and no null. The rotation question that five trajectory statistics failed to settle (D22/D26/D28/D32) is one cheap run away, and the data may already be on disk. | **DONE → D55.** Answered from data already on disk: leading eigenvalue complex in 3/3 prompts, period ≈2.6–6.0 unrolls, rotation real but surviving only ~4 turns and sampled at ~3 points/turn. Zero GPU. A proper multi-prompt sweep is now the follow-up. |

---

## C. Scaffolding audit (request A1)

**Measured, not guessed:** 23 kernel bundles, 4563 lines, **28% of 8-line windows
appear in more than one kernel**; `fit_rho`, `orbit`, `measure`, `summarise` each
written 8–9 times. That duplication re-imported a fixed bug (`np.arange(1, n+1,
float)`) **three times**, costing one GPU run.

| # | friction | fix | status |
|---|---|---|---|
| C1 | Kernel boilerplate duplicated 8–9× | `scratch/_lib/kernel_common.py` + `scripts/build_kernel.py` inline tested blocks into a self-contained `main.py` | **done** |
| C2 | Shared blocks were never tested | `tests/test_kernel_common.py` — 15 tests, incl. an **AST** check for the arange bug | **done** |
| C3 | Kaggle logs are JSON-lines-wrapped; the same parser rewritten 3+ times | `scripts/klog.py` — clean stdout, `--tail`, and `--json` to recover embedded records | **done** (it is how D55 was recovered) |
| C4 | push/status/pull/watch typed out each time; 3 hand-written watcher scripts | `scripts/kg push\|st\|pull\|watch\|run` — reads the kernel name from `kernel-metadata.json` so bundle and name cannot drift | **done** |
| C5 | Ledger rows appended by ad-hoc heredocs ~12 times | `scripts/ledger.py` — **validates, does not generate**: each row needed bespoke prose, but duplicate ids, missing cells, dangling `D-nn` citations and unpointed retractions are checkable. Found 2 real gaps on first run | **done** |
| C6 | test+lint+commit+push retyped every time | `scripts/ship` — also refuses to touch `origin` | **done** |
| C7 | Over-verbose reads filling context (`git ls-files scratch/` dumped 100+ lines; one grep produced a 30 KB persisted file) | default to counts/`head`; reach for full dumps deliberately | **habit, not tooling** |
| C8 | Foreground polling of GPU jobs | background watchers | **done in practice** |
| C9 | **Generation was the entire cost and was ~50× slower than necessary.** Measured on the Caesar screen: 239 min total, of which model load was **0.6 min (0.25%)** — so a Kaggle-dataset weights cache would have saved nothing. The remaining 238 min was 240 completions at **~1 min each**, because the hand-rolled loop re-ran the whole sequence through all 32 unrolls for every token at batch size 1, ignoring the `HuginnDynamicCache` and four `generate_*` methods the model ships. | `batched_generate` block: uses `generate_minimal` (cached, batched), buckets by exact token length because `forward` sets `prepared_attn_mask = None` so **padding is unmasked** and equal-looking prompts tokenise to 15 *or* 16 tokens | **done** |

**What must NOT be "optimised" away.** Reading full logs when a result is
surprising; per-instance distributions rather than means (D49 exists only because
the mean hid a bimodal population); cross-checking a number against a second,
unrelated source. Terseness is for *routine* output only. When something is
unexpected, verbosity is the point.

---

## D. Observability (request A3) — see `docs/observability.md`

Short version, recorded here so the register is self-contained: this project had
**many** rotation metrics (winding, chord-to-arc, H₁ persistence, participation
ratio, subspace dimension) and they **all failed together**, because they shared an
unexamined assumption — that the recording window contained signal rather than
arithmetic noise. Metric *diversity* gave false confidence because the metrics were
diverse in *statistic* and identical in *assumption*. The fix is not more metrics;
it is metrics that fail differently, plus a small set of standing diagnostics that
would have caught the failures actually observed.
