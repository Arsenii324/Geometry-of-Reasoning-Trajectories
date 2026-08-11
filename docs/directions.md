# Directions register

**Purpose.** Anything recognised as worth doing goes here the moment it is
recognised, not into working memory.

> **Read `OPEN_THREADS.md` first if you want to know what to DO next.** That file
> is the one-line-per-thread, prioritised INDEX; this file is the store that holds
> the reasoning, the evidence and the dead ends behind each entry. The index exists
> because this file passed 800 lines, at which point "everything is written down"
> stops meaning "anything is findable". **Neither supersedes the other**: a thread
> with no line in the index is invisible; a thread with no detail here is a slogan.
> `tests/test_docs_consistency.py` fails if the index references a section of this
> file that does not exist. Context gets compacted; this file does not.
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

## 0. Runs in flight, and premises already ruled out

**Why this section exists.** Everything below is about what to do next; nothing said
what is happening NOW, and the previous session's results were lost because they
existed only in a container. A reader opening this repo should be able to see the
live state without asking. Updated 2026-08-09.

| run | where | question | status |
|---|---|---|---|
| `geometry-h2-rotation` | Kaggle T4 | B4.15 — H2 on arg(lambda), length-matched | **DONE -> D83.** Paired track−local vs n_ops: rho=+0.543, exact p=0.328 over 2048 arrangements. H2's last instrument returns nothing |
| `geometry-geomcap` | Kaggle T4 | does trajectory geometry track capability ACROSS the 21 families — and what is its own reliability ceiling? | **DONE -> D85.** 608/608 banked. 0 of 18 (statistic, window) cells track capability; 6 track prompt length; reliability 0.95-1.00 so a perfect relation would have shown |rho| >= 0.80. Both determinism controls passed |
| `geometry-lenmatch` | Kaggle T4 | **the test D84 demands** — two computations at an IDENTICAL token count | **DONE -> D87.** Gate passed absolutely (188/188 kept, 0 dropped, token multisets identical). Shape decodes the marker at 96.9-100%, pooled 98.9%, p = 0.0025. D84's length confound removed |
| `geometry-marker` | Kaggle T4 | the control D87 needs: A and C defined IDENTICALLY, B differently | **DONE -> D88.** A-vs-C (same computation) 98.0-100% against A-vs-B's 96.0-100%. **No gap: the shape reads the TOKEN, not the computation.** The pre-registered prediction failed and is recorded as the result |
| `geometry-depthacc` | Kaggle T4 | does the model SAY what its rank curve says it knows? closes D68(5) | **DONE -> D86.** Exact match hits 0% by r=8 while containment rises to 33-39% (p = 0.0215): depth wraps the answer in prose. rank->output link phi = +0.600 at r=4, gone by r=32 |
| `geometry-battery` | Kaggle T4 | 21-family capability screen, graded readout | **DONE -> D75.** 5 families >=50%, control at chance 21/21, and B6 unblocked |
| `geometry-h0bank` | Kaggle T4 | **the powered, PRE-REGISTERED test of D91's lead** — 16 boundary prompts x 32 unseeded h_0 draws, within-ITEM correctness decode with the window grid fixed in advance | **DONE -> D93.** NOT CONFIRMED: best window (6-18) 52.0% vs 46.8% null, p=0.0125, misses corrected alpha=0.00625. The P8 detection floor was found broken (n_perm=40 could never clear that alpha), fixed to n_perm=400, re-run: now catches a planted 0.5 sd effect at 100%, so the null is trustworthy, not underpowered. D91's transient-window mechanism is ruled out |
| `geometry-clrs` | Kaggle T4 | capability screen on CLRS-Text, which is IN Huginn's training mixture and has a clean integer difficulty knob — the task gap the supervisor named | RUNNING |
| `geom-eigenplane` | DataSphere g1.1, job `bt1mvf3juvrcjpl6kl1i` | does the orbit rotate at the rate the Jacobian predicts? | **DONE** -- no: 0.24x, and D74 explains why **THIS ROW IS ALSO B3's ANSWER (D119).** `OPEN_THREADS` listed B3 as *"Never done — the only route to a second, valid estimate of rho"* while this run had already measured the FULL unroll operator: \|λ\| median 0.8098, arg λ separating tasks (`track` 61–63°, `local` 19.5–24.3°). Filed under a RUN name, indexed by a THREAD name, so it was invisible to the index. |
| `geom-bank` | DataSphere g1.1, job `bt1hd3oqb17690amgolg` | B14 -- bank raw states for BOTH arms; the untrained control D74 needs | **DONE -> D76.** Training flips the step cosine -0.379 -> +0.541, disjoint at every window |
| `geom-seeds` | DataSphere g1.1 | closes D76's one-draw limit | **DONE.** Five draws agree (cos -0.339..-0.389); trained/untrained completely disjoint, p=1.05e-12 |
| `geometry-b6-bank` | Kaggle T4 | **B6 re-run** | **DONE -> D79.** Bounded null: no geometric difference at matched answer, powered to 1.5 sd, in the 1 family of 4 that the design could test |
| `geometry-patch` | Kaggle T4, queued (built + CPU-verified, blocked on a slot — Kaggle caps at 2) | **first non-void causal test.** Patch the answer-position state at unroll r from a correct h_0-draw into an incorrect h_0-draw of the SAME prompt (D90's mechanism); sweep r over {0,8,16,24,32,48}; does the flip rate concentrate in D91's window? | BUILT, `scratch/kaggle_patch/`, 8/8 CPU dry-run tests pass. Push with `kaggle kernels push -p scratch/kaggle_patch` |
| `qk-alignment-probe-smoke` | **Yandex DataSphere**, project `bt12q57tmrs03pnt8drc` (personal account, ~80k RUB remaining, valid to end of year — confirmed by the user 2026-08-09, NOT the expired smiles2026 grant) | the supervisor's own open item (`project_plan.md` G2, 0% done): per-unroll query-key cosine alignment at the answer position, on the SAME paired track/local length-matched design as H2/D83 | **DONE, descriptive only — not yet a claim.** First submission's self-check failed correctly (root cause: ground-truth spy captured the wrong layer, see git log 2026-08-09 for the fix); fixed and reproduced/confirmed locally before resubmitting. Resubmission (`bt164frpr04jvqguht5j`) SELF-CHECK PASSED (exact 0.0 error) and banked 24/24 forwards. **Printed final-unroll means show `track > local` in all 12 (n_ops, seed) pairs**, by a small and shrinking margin (~0.001-0.014) as n_ops grows — directionally consistent but this is NOT a computed statistic: `config.yaml` never declared an `outputs:` block, so DataSphere did not retrieve `results/qk_probe_smoke.json` (the per-unroll data) and only the stdout-printed final-unroll means survive. **No proper D83-style paired permutation test has been run on this.** Before spending more of the personal Yandex budget on a powered follow-up: (1) fix `config.yaml` to declare `outputs:` so the raw JSON comes back, (2) decide whether the pattern is worth scaling given it may just be D87/D88's "shape reads the token" effect showing up in Q/K space too (track and local prompts end in different words). **This is a spend decision — flagged for the user rather than launched unilaterally (CLAUDE.md §8).** |

**Answered offline since, with no GPU, from data already on disk.**

- **D80 — the shape statistics are mostly a clock.** Sliding a fixed-width window
  along the unroll axis over all 152 banked trained orbits: participation ratio
  falls 8.55 → 4.55 between unroll 0 and 28 (down in 147/147), the step cosine
  rises +0.276 → +0.710 (up in 134/147), and the contraction rate does not move
  (p = 0.32). Each shift is ~3× the entire prompt-to-prompt spread. This resolves
  the apparent conflict between D74 (PR 13.3 at a ~21-unroll window) and the
  asymptotic values (PR 2.3 at ~90), and it means **any shape statistic quoted
  without its window is uninterpretable**.
- **D81/D82 — the map is linear where it matters, so D52 stands.** Held-out
  one-step prediction recovers a median 91% of what the fitted subspace can
  express. A spectrum-built surrogate reproduces the dimensional collapse (5/7
  prompts) but not the rotation rate (2/7); D82 shows that is about which modes
  Arnoldi returned, not about linearity, since a gapless spectrum makes later
  steps leave the earlier span without any nonlinearity being involved.

**Premises checked and FALSE — do not re-derive these.** Each cost minutes to check
and would have cost hours to discover downstream.

- **`results/trajectories/*.npy` contains no multi-initialisation information.**
  `init{N}` is the TASK seed; h_0 is fixed across those files, proven by a prompt
  collision at n_ops=2 giving bit-identical arrays. So the two-orbit contraction
  estimator (`contraction_from_pair`, the quantity H3 is about) is NOT computable
  from them, and neither is UNDERSTANDING §6.9's ρ(r≤32) vs ρ(r≤98) question. The
  top-level `trajectories/` with its `manifest.csv` is the directory that does vary
  h_0, and it only goes to ns=64. *(Checked 2026-08-09; the provenance sidecars say
  so explicitly, and `contraction_from_pair`'s own docstring warns about it.)*
- **Fitting ρ from a single trajectory's approach to its own endpoint is the
  tautology this project already made once.** It is near-tautological for any
  convergent sequence and is called out in `metrics/regime.py`. So §6.9 cannot be
  rescued by a single-trajectory estimator either; §6.10's "this needs a rerun"
  stands, and was verified rather than assumed.

## A. Standing requests from the curator/user

| # | request | status | where it lives |
|---|---|---|---|
| A1 | Audit for helpful scaffolding — templating, backgrounding, less verbose output, without cutting observability | **done** — §C, all 8 items | §C below; `scratch/_lib/`, `scripts/build_kernel.py`, `tests/test_kernel_common.py` |
| A2 | Caesar cipher as a task, several fundamental variants | **open, designed** | §B1 |
| A3 | Observability metrics — do we need more, fewer, or different ones to spot the unexpected | **done** — `docs/observability.md`; it paid for itself immediately (D55) | §D |
| A4 | How are "random" weights selected? Is Huginn's init scheme the right one for a recurrent transformer? | **partly answered** | §B2 |
| A5 | Is SoTA parameter-efficient fine-tuning feasible on Kaggle for a 3.5B recurrent model? | **assessed (§B3); research brief written for an external agent** | §B3, `docs/briefs/peft_for_spectral_control.md` |
| A6 | Keep recognised ideas in a file, not in working memory | **done** | this file |
| A7 | Build a TESTED parser for answer/gold extraction that correctly handles Huginn's own BOS/EOS and request/response markers, not just naive first-token rank — rule the D89 defect in or out property-by-property rather than family-by-family | **open** | §E1 below |
| A8 | For every battery family currently reading low/zero accuracy: is it a genuine model failure, or an artifact (multi-token gold, prompt formatting, insufficient prompting/depth)? Disentangle before trusting the capability axis's LOW end the way D89 disentangled the multi-token defect | **open, partly informed by D89** | §E1 below |
| A9 | Explain the shape-code classifier (D84/D87/D88/D92) in plain terms — answered inline 2026-08-09, see the chat | **answered** | `src/traj_geom/metrics/shape_code.py`, D84 |
| A10 | Do published Huginn papers report genuine TASK accuracy (not trajectory geometry) on any benchmark — could sanity-check whether the battery's low numbers are a real capability floor or an artifact of this project's own prompting/scoring | **open** | §E2 below |
| A11 | Did the large multi-family battery run from earlier in the session complete, and is its data preserved? | **answered inline 2026-08-09** — yes: `geometry-battery` (D75, 21 families, `results/battery.csv`) and `geometry-geomcap` (D85, 608/608 orbits banked, `results/geomcap.csv` + raw `.npy` states) both completed and are committed. Nothing from either run was lost. | D75, D85 |

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
| B4.1 | ~~Is rho constant across task families?~~ | **ANSWERED 2026-08-10 by D115: NO.** Over 21 families rho spans **0.8239-0.9181**, between-family spread **4.2x** the within-family spread, and it survives length matching (7.5x at an identical 29 tokens). D116 adds that the rotation ANGLE is a second, independent task-level parameter (38-83 deg, uncorrelated with the modulus at +0.11). D45's retracted claim is superseded by a measurement with 21 families rather than 3. |
| B4.2 | Recompute D46's arm comparison with the corrected window | **done** → D53 |
| B4.3 | Untrained control for D35 | **dropped** → D54(3): degenerate by construction, the untrained model has no answer process |
| B4.4 | Why are 8/108 trained orbit fits below the R²>0.9 bar while 0/60 untrained are? | **open** — a second, unexplained way the trained operator differs (D52(3)) |
| B4.5 | What happens to ρ *before* step 6144? | **parked** — no published checkpoint exists; would need training from scratch |
| B4.7 | **Measure accuracy on the 8 task families that never had it** (parity, running-max, projection, three-scale, three-scale-modk, running-count, nesting-depth, count-ones). One generation pass each. | **open — the single highest-value gap** (D56(2)); decides whether 8 tasks' worth of geometry describes reasoning or failure |
| B4.8 | ~~Give persistent homology a null, or drop it.~~ | **DONE 2026-08-05 (D64)**, and EXTENDED 2026-08-09 (D123): the signal is concentrated in the orbits that rotate — median normalised H1 persistence 0.0241 vs 0.0000 settling, 17× its own manifold-matched null, 124/124. **This row said `open` for four days while `UNDERSTANDING.md` line 703 said CLOSED**, and the contradiction caused D123's author to overwrite the existing instrument and reinvent its known-bad predecessor. |
| B4.9 | **Sweep the Jacobian spectrum properly**, n=3 → tens of prompts, both arms. Magnitudes gave ρ (D31), arguments gave rotation (D55). | **open** — best value per unit cost in the project |
| B4.15 | **H2's FIRST REAL TEST: rotation rate vs difficulty on the Jacobian eigenvalue argument, n ≫ 3.** D55 has 3 points (n_ops 8/32/64 → \|arg\| 2.397/2.388/1.050) on the right instrument, never analysed as H2. Needs ~30 prompts × 4 difficulty levels, both arms. | **open — the single most valuable unrun experiment** (D65(5)); one kernel |
| B4.16 | ~~H1 re-operationalised on a working instrument.~~ | **DONE 2026-08-10 by D123 and D126.** Persistent homology carries the question instead of winding, against a manifold-matched surrogate preserving radius, radial profile and step norms. Over 308 orbits the signal is real and concentrated: rotating orbits give median normalised H1 persistence **0.0241 against a surrogate 0.00139 (17x), beating all 8 of their own surrogates in 124/124**, while settling orbits give exactly **0.0000**. D126 then shows it carries no correctness at matched answer value (p = 0.93). A real positive and a matched negative. |
| B4.17 | **H3's consequence, never tested.** ρ<1 is established four ways; the implication for counting capacity was never turned into a measurement. | **open** (D65(3)) |
| B4.13 | **ρ is fitted over 65–98 unrolls; Huginn trains at mean r=33.** All 12/12 prompts have ρ characterised beyond 2× the mean training depth, so the project's central quantity describes a regime the model was never trained for. Whether ρ(r≤32) = ρ(r≤98) is untested and bears on D42/D44/D52/D59. | **open** — requires a rerun, because of B4.14 |
| B4.14 | **Kernels persist the FIT, not the CURVE.** `eps_sweep.json` stores fitted ρ and n_used but not the per-unroll separation curves, so B4.13 cannot be answered without re-running the GPU job. Rule: persist the curve you fitted. Costs a few MB; not doing it costs a rerun per unanticipated question, and this project generates about one such question per experiment. | **rule adopted; existing kernels not retrofitted** |
| B4.11 | **DEPTH WAS NEVER VARIED AGAINST ACCURACY.** Audited 2026-08-05: every accuracy kernel in this project fixed `num_steps=32`. The kernels that sweep depth all score geometry. On a recurrent-depth model whose authors report GSM8K rising 9-10% at r=4 to 28-38% at r=32, the single most obvious experiment for the architecture had not been run. | **in progress** — `geometry-prompt-depth` sweeps r ∈ {4,8,16,32,64} on the best of five phrasings |
| B4.12 | **Huginn's own inference modes were never used.** `generate_with_adaptive_compute`: 0 kernels. `continuous_compute` (warm-starting the latent across tokens — the "continuous CoT" mode that is the architecture's distinctive feature): 0 kernels. | **partly in progress** — `continuous_compute` is now an arm in `geometry-prompt-depth`; adaptive compute still unused |
| B4.10 | ~~Re-operationalise or retire H1's regime labels.~~ | **RETIRED 2026-08-10.** They partitioned without separating (Kruskal-Wallis p = 0.48, D56), and the classifier behind them fires on straight lines, random walks and decaying spirals alike (D25(3)). Every claim resting on it is withdrawn: **B4** (the project's cleanest positive result), **D14** (its only positive loop observation) and **D100** (a working difficulty ladder). What replaces them uses none of it: D98, D112/D123, D127/D132. |
| B4.6 | **Jacobian eigenvalue ARGUMENTS.** D31 ran Arnoldi on J-vector products and reported only the magnitude (ρ≈0.79–0.81). A contracting map rotates iff its eigenvalues are complex, at a rate given by their argument — a quantity needing no trajectory, no window and no null. The rotation question that five trajectory statistics failed to settle (D22/D26/D28/D32) is one cheap run away, and the data may already be on disk. | **DONE → D55.** Answered from data already on disk: leading eigenvalue complex in 3/3 prompts, period ≈2.6–6.0 unrolls, rotation real but surviving only ~4 turns and sampled at ~3 points/turn. Zero GPU. A proper multi-prompt sweep is now the follow-up. |


### B5. THE READOUT IS THE CONFOUND — **DONE → D68, D69; it gated B6-B8 and B10**

**The problem.** Every capability claim here is scored by *greedy generation plus
string match*: D56, D60, D61, the Caesar screen, and the unrun `capcontent`. That
instrument has failed loudly twice — D62 returned empty output for every prompt,
D60 showed format alone moving `copy` 15% → 100% — and it has **no dynamic
range**. Once accuracy is 0, "the model cannot do this" and "the readout cannot
see it" are the same measurement. UNDERSTANDING §6.1 makes that the project's
largest open threat, because every *content is architectural* result (D40, D41,
D48, D53) was measured precisely there.

**The Caesar screen is the sharp case, and its own raw output shows it.** Trained
exact-match is 0.0% in all six cells. The second metric it added to avoid this
trap does not survive inspection: **the untrained arm beats the trained one in
four of six cells on `char_acc` (5.8% vs 0.0%)**, and the trained arm's best
cells come from emitting a memorised pangram (`'the cat sat on the mat'` →
`'The quick brown fox jumps over the lazy dog'`) — D61's retrieval mode, not
partial competence. So B1's "drop this family" verdict rests on two blunt
metrics, one of them chance-dominated and sign-inverted.

**The instrument already exists and was never pointed here.**
`traj_geom/eval_depth.py` scores teacher-forced `log P(gold)` per unroll by
hooking `core_block[-1]` and decoding each unroll through a replicated coda+head.
Built for D35; never used for capability. Two properties decide it:

- **One forward at `num_steps=R` yields the whole `r=1..R` curve**, because the
  hook fires once per unroll. Generation needs a separate decode per depth. This
  is the difference between a depth × task × arm sweep costing minutes and hours
  (C9: generation was the *entire* cost, ~1 completion/min).
- It already computes `log_softmax` over the **full vocabulary**, so **rank is one
  line away** and was simply never recorded.

**Why rank and not Acc@k.** Acc@k truncates exactly where the variation lives: a
model at 0% accuracy may hold gold at rank 3 or rank 8000, and Acc@5 scores both
as a miss. Full-vocab rank separates them by three orders of magnitude; chance is
32768. Recorded per unroll, "does depth help" gains a gradient instead of a step.

**Status: DONE -> D68, D69.** 5 tasks spanning capability
(`echo_digit`, `add1`, `count4`, `count16`, `rot13_word`) × 24 items ×
trained/untrained, recording rank, `log P(gold)`, distractor margin and top-1 at
every unroll. Four pre-registered predictions in the kernel docstring, including
the one that kills the direction cheaply: **if trained median rank is ~chance on
the 0%-accuracy cells, the graded readout adds nothing** and "the model cannot do
these tasks" is robust. Persists the curves (B4.14). ~1 GPU-h.

### B6. GEOMETRY CONDITIONED ON CORRECTNESS, AT MATCHED DIFFICULTY — **the strongest available test of the founding hypothesis**

Every geometric measurement in this project has **pooled correct and incorrect
trajectories**, and since accuracy is ~0 almost everywhere, that means the
geometry describes *failure*. The project set out to ask whether trajectory shape
encodes reasoning; it has never compared the shape of a trajectory that got the
answer right against one that got it wrong.

**Design.** Use B5's graded readout to locate cells where the model *succeeds*
(rank 1) and *fails* (rank ≫ 1) **within the same task at the same difficulty and
prompt length**, then measure ρ, the Jacobian eigenvalue argument, and the
convergence geometry on each group. Task, length, format and difficulty are all
held fixed by construction; only correctness varies. That is the contrast the
project has never had.

**KNOWN LIMITATION OF THE RUNNING VERSION, found by self-review while it was in
flight.** Geometry is captured at `[0, -1, :]` — the appended **gold-token**
position — while correctness is read at `n_p - 1`, the prompt-final position.
They are one token apart, and `[-1]` no longer matches the project's prior
convention either: earlier work appended nothing, so `-1` *was* the
answer-predicting position. The comparison is not void — the gold token is the
same for both classes, so nothing confounds class with position content — but the
geometry is measured one step downstream of where correctness is decided, on a
token the model did not choose. **Read the result with that caveat, and treat a
re-run at `n_p - 1` as the follow-up.** Not killed mid-flight because the run is
interpretable and the GPU time is already spent.

**Status: VOID as first run (D72), and now UNBLOCKED (D75).** The first run's correctness was a deterministic function of the answer value, so the label permutation broke the prompt-geometry link at the same time. D75's screen supplies what it needed: `count_mod3`, `sub1`, `nth_item` and `local_last` each have a gold value carrying BOTH a success and a failure, so the contrast can be run WITHIN a fixed answer. The re-run should stratify on gold value and use those families. Original status line follows.

**`geometry-correctness` RAN 2026-08-08.** Split criterion
is depth-free — correct = gold reaches rank 1 at any unroll ≤ 64 — because D68
showed the accuracy peak moves with task, so a fixed depth would make the split a
function of my choice rather than of the model. Analysis path validated offline
first: 0 false positives on a null, finds exactly the planted metric at 2 sd,
refuses with "too few" at 1-vs-63 balance, and the verdict withholds when the
threshold is unattainable.

**Pre-registered prediction.** If trajectory geometry tracks computation, correct
and incorrect trajectories must differ on *some* geometric statistic at matched
difficulty. **If they do not differ on any of ρ, |arg λ| or settling time, then
geometry is not tracking the computation** — a clean, falsifiable negative that no
amount of pooled measurement could have produced. Requires B5 to find successes
first; ~1-2 GPU-h after that.

### B7. B4.11 (DEPTH vs ACCURACY) REDESIGNED — **one forward per item, not one generation per depth**

B4.11 is "the single most obvious experiment for this architecture" and remains
unrun: `geometry-prompt-depth` attempted it by generation, OOM'd (D66 shows that
OOM is still unexplained), and its `continuous_compute` arm cannot work at all
(D66). The teacher-forced curve gives **every depth from a single forward**, which
sidesteps the decode loop entirely — no KV cache (D62, unresolved), no format
swing (D60), no `continuous_compute` (D66, blocked), no OOM.

**Predictions.** Rank improves monotonically with `r` on tasks the model can do
and is flat on those it cannot; and D35's *threshold depth scales with difficulty*
should generalise from its original tasks to the capability ladder — or fail to,
which would scope D35. Essentially free once B5's kernel exists.

### B8. RE-OPEN CAESAR BEFORE DROPPING IT — **cheap, and it changes what B1 means**

B1 is the "highest-value open direction" and its screen said drop. That verdict
came from the blunt readout. If `rot13_word` shows **low gold rank while top-1 is
a pangram**, the finding is not "cannot decode" but **"knows but does not emit"** —
which is a different claim, rehabilitates the family as the capability-contrast
candidate the project needs, and gives D61's retrieval modes a mechanism rather
than a taxonomy. Already an arm of B5's kernel, so cost is zero.

### B9. STANDING RULE PROPOSED FROM B5

CLAUDE.md §1 already says *"before scoring geometry on a task, ask whether the
model can do the task."* B5 supplies the missing half: **that question must be
asked with a graded readout, not exact match** — because exact match cannot
distinguish an absent capability from an invisible one, and every claim built on
a 0% cell inherits that ambiguity.

### B10. THE 6.1 MODERATION TEST, WITH A CAPABILITY AXIS THAT CAN SEE — **DONE → D70**

`geometry-cap-content` was built to settle UNDERSTANDING §6.1 and measures its
capability axis by **greedy generation plus exact match** — the instrument D68
shows is blind exactly there (0.0% by exact match, median rank 22 against chance
32768). Run as written its *moderator* would be ~0 across the whole ladder for
measurement reasons, and it would "confirm" the headline by construction.

That kernel is **not edited** (§3: say what is wrong rather than rewrite someone
else's experiment); the warning sits in UNDERSTANDING §6.1 and
`geometry-cap-graded` is a separate kernel with the capability axis replaced by
log₁₀ rank. Content is the **same** ridge probe as D41/D48/D53, so the comparison
is like-for-like. Two gates, both simulated before launch: `cv_r2`'s permutation
null must be ≤0.05 (a positive null means the probe interpolates), and the
untrained arm must sit at chance (the control that made D68 credible). **P4 is
written as equally publishable**: a ~0 gap at every rung, including ones the model
demonstrably does, means "content is architectural" survives its strongest
challenge. N=4 rungs, so read as direction plus effect sizes, never a p-value.

**Still open after this:** D40/D41/D48/D53 have not themselves been re-measured
with a graded readout — only new tasks have. That is the remaining decisive gap.

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
| C9 | **Generation was the entire cost.** Measured on the Caesar screen: 239 min total, model load **0.6 min (0.25%)** — so a Kaggle-dataset weights cache would have saved nothing. The remaining 238 min was 240 completions at **1.00/min**, batch-1 with no KV cache. | `batched_generate` runs the decode loop KNOWN to work over length-homogeneous batches. **It does NOT use the KV cache**: `generate_minimal` (batched + cached) returned empty output for every prompt (D62) and both of my proposed mechanisms were refuted against the source, so the cause is unresolved and I sidestepped it rather than debug a borrowed decode loop on a borrowed GPU. Buckets must be length-homogeneous because `forward` sets `prepared_attn_mask = None`, so padding is unmasked. | **partial — speedup NOT YET MEASURED.** An earlier version of this row claimed "~50× slower than necessary"; that figure was never measured and has been removed. The only measured throughput is the 1.00/min baseline. |
| C11 | **The uncached loop needs an activation budget.** Without a KV cache it re-runs the full GROWING sequence every step, so cost is `batch x seq` and rises as generation proceeds. float32 weights are ~14.1 GB of a T4's 14.56 GB, leaving ~450 MB against a gated MLP of inner width 17920 — batch 16 x ~100 tokens died inside `nonlin(x_fc_1) * x_fc_2` in `geometry-prompt-depth`. | `max_batch_tokens` (default 1024) splits each length-bucket so `chunk x (prompt_len + max_new) <= budget`. Real cells now run at batch 14 (short answers) and 8 (CoT), so batching survives. Two tests: the budget binds when it should, and does NOT split when it needn't. | **done** |
| C12 | **The reverted generator is VALIDATED on GPU.** `geometry-prompt-depth`'s smoke test printed `4/4 copied -> ['banana','orange','puzzle','kitten']` before the OOM. So the uncached batched loop produces correct text, and D62's empty output was specifically the `generate_minimal` cache+batch path — not batching, and not my decode logic. | — | **done** |
| C13 | **A third GPU push is REJECTED, not queued.** `kaggle kernels push` with two sessions live returns `Maximum batch GPU session count of 2 reached` and the push fails outright — `code_env_info/kaggle-cli-guide.md` §5 says to "expect it to QUEUE rather than run or fail", which is wrong for this account/CLI version. `kg enq` exists for exactly this and is the correct path. | use `scripts/kg enq`, never a bare third push | **corrected 2026-08-08** |
| C10 | **No queue: Kaggle slots idle between turns.** Two concurrent slots and ~30 GPU-h/week were available throughout, but launches were reactive — one or two at a time, with slots idle while ledger rows were written. The inner loop was optimised (load profiled, batching added) and the outer loop was not. | `scratch/QUEUE` + `kg watch` pops the next bundle whenever fewer than 2 kernels are busy | **done** |

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

### B11. A CONTENT PROBE WHOSE TARGET IS NOT IN THE INPUT — **running, and it can refute D70**

D70's deepest finding: every content probe in this project (D40, D41, D48, D53,
D70) targets a quantity that is a **linear function of the input token bag** — a
count, a digit, a sum — and Huginn re-injects the prompt embeddings at *every*
unroll through the adapter on `[h, e]`. So the bag is in every state by
architecture, a random-weight model preserves it (untrained R² = **1.0000** twice),
and the probe answers *"is the input linearly recoverable"* rather than *"did the
model compute anything"*. Both arms pass trivially; the gap had no headroom to move.

**The fix is the target, not the probe.** `geometry-nonlinear-content` holds the
probe fixed — the same ridge `cv_r2` as D41/D48/D53 — and varies how linearly
available the target is, all five derived from one bit-string so they share one
forward: `count` (linear, control), `parity` (not linear — the canonical case),
`last` (needs position), `max_run` and `alt` (order-dependent). The theory is
already in this project's bibliography: Grazzi et al. 2411.12537 on LRNNs and
parity, Merrill et al. 2404.08819 on the illusion of state.

**P5 is written so the run can refute D70 itself:** if the *untrained* arm decodes
parity well, the re-injection story is wrong and D70(4) must be amended. Gated on
`has_dynamic_range`, so a repeat of D70's ceiling withholds the verdict rather than
reporting a rank correlation over slivers.

### B13. IS D73's GAP CONTENT OR LINEARITY? — **open, cheap, and it can refute D73**

D73 found the trained-minus-untrained gap ordered by distance from a linear
function of the input bag (-0.044 count → +0.805 alt). Two objections remain,
and one of them is now the only live one.

**Dead: the rank-1 objection.** D48 measured the untrained count representation at
participation ratio 1.0, which would have explained the entire table. It is
refuted from data already on disk (D73(7)): `last` is at most r2=0.0743
explainable by `count`, yet the untrained arm decodes `last` at 0.7528. One
dimension cannot carry both.

**Live: 6.7, and it has never actually been tested.** Every content number in
this project — D41, D48, D50, D53, D70, D73 — comes from a LINEAR ridge. If the
trained model encodes order NONLINEARLY while random weights preserve the input
linearly, a linear probe manufactures exactly D73's pattern for reasons unrelated
to information content. UNDERSTANDING §6.7 claimed this was under test; it was
not. `cv_r2_nonlinear` is inlined in `kaggle_capcontent` alone, which has never
run, and `geometry-nonlinear-content` used the linear probe by design.

**Design.** Same 96 items, 16 bits, r=32, both arms, same five targets — so the
comparison is like-for-like with D73 — running four things off one forward per
item: (1) `cv_r2` linear, as a positive control that the pipeline reproduces D73;
(2) `cv_r2_nonlinear` (PCA-8 + degree-2 ridge, the design already tuned against
synthetics); (3) participation ratio per arm, the descriptive capacity number;
(4) **the states persisted** (96 x 5280 float32 ≈ 2 MB per arm), per B4.14, so the
next unanticipated question does not need another GPU hour.

**Pre-registered predictions.**
1. *P1, gate.* Linear R2 reproduces D73 within ±0.05 on all 10 cells. If not, the
   pipeline differs and nothing else in the run may be read.
2. *P2, gate.* Untrained participation ratio > 1. A PR of 1.0 would contradict an
   inference already drawn from disk (D73(7)) and must be resolved first.
3. *P3, gate — the one that decides whether P4/P5 are readable at all.* The
   nonlinear probe must beat the linear one on a PLANTED curved target measured in
   the same run. `cv_r2_nonlinear`'s own docstring records that three earlier
   designs failed this, and **a weak nonlinear probe reports "no lift" and thereby
   confirms the headline it exists to challenge.**
4. *P4.* Nonlinear lifts untrained `max_run`/`alt` by ≥ +0.3 over linear ⇒ D73's
   gap is a linearity artifact, D73(3) must be amended, and 6.7 is a real threat.
5. *P5.* Lift < +0.1 ⇒ the information is absent in any form this probe reaches,
   D73(3) stands, and 6.7 is answered negatively for these targets. P4 and P5 are
   mutually exclusive; the 0.1–0.3 band is reported as partial, not rounded to
   either.

Every gap reported must pass `gap_is_readable` (both arms are otherwise free to be
below their own nulls, which is how D73's parity headline happened). ~1 GPU-h.

### B14. BANK THE STATES, ANALYSE OFFLINE — **the structural fix, and it subsumes several rows**

**The problem, stated as a pattern rather than an incident.** Every GPU run in this
project computes its own statistics and returns conclusions. So every question the
design did not anticipate costs another GPU hour, and this project generates about
one such question per experiment:

- §6.9/B4.13 cannot be answered because `eps_sweep.json` stored the fitted ρ and
  not the per-unroll separation curves (§6.10, B4.14).
- D73's rank-1 objection cannot be settled because the kernel stored `r2` and the
  targets but not the STATES, so participation ratio per arm is not computable.
- D72's confound needed the item set REGENERATED from the kernel's own seeding
  because prompt and gold were not persisted.

B4.14 adopted "persist the curve you fitted". This goes one step further and is the
version that actually closes the class: **persist the STATES, and make the GPU run
a data-collection step with no analysis in it at all.**

**Why it is affordable.** One trajectory at 128 unrolls x 5280 dims x float32 is
2.7 MB. A 24-prompt x 2-arm grid is ~130 MB — nothing, against a 20 GB output
allowance. Storing bf16-exact float32 keeps D30's arithmetic-floor analysis valid.

**What one such run would then make CPU-local and re-runnable**, with no further
GPU at all: every winding variant W1-W9; the eigenplane projection; ρ by any
estimator over any window (which is §6.9); participation ratio per arm (D73's open
confound); persistent homology; step-cosine by regime; and any statistic invented
later. It also makes the analysis unit-testable, which is where three kernels in
this project produced wrong conclusions.

**What must be recorded alongside the states**, learned from the rows above: the
prompt, the gold, the per-unroll rank of gold (so correctness is known AT EVERY
DEPTH rather than at one), the arm, the task seed, and the tokenised length. The
per-unroll rank is what makes correctness a variable rather than a constant, and it
comes free from the same forward via the D71-validated coda readout.

**Caveat on the readout, worth stating because it bears on the geometry.** Decoding
intermediate states through the coda head is the project's most exotic instrument;
D71 established it reproduces the model's own logits exactly at the FINAL unroll,
which is where the identity is guaranteed. At intermediate unrolls the model never
applies coda, so the readout is a probe rather than the model's own computation. It
is the right instrument for "is the answer available at depth r" and it is NOT
evidence about what the model would emit at depth r. The geometric quantities do not
depend on it at all, which is a reason to bank the states rather than only the
readout.

**Status: designed, not yet run.** It should be the next GPU job after the three in
flight, and it plausibly replaces B13 and part of B4.13 rather than adding to them.

### B12. THE CONTENT NULLS THEMSELVES, RE-MEASURED — **open, and still the decisive gap**

D40/D41/D48/D53 have never been re-run with either a graded capability axis or a
target outside the input bag. B10 and B11 measure *new* tasks; the original nulls
still stand on the old instrument. Until they are redone, "content is
architectural" is supported only indirectly.

| — | **DEQUEUED UNRUN 2026-08-09: `kaggle_capcontent` and `kaggle_promptdepth`.** Both were queued from a local session that predates D70–D89. `capcontent` tested UNDERSTANDING 6.1 (is "content is architectural" an artefact of zero-capability tasks) with a linear+nonlinear probe of the answer-position state — the design D70 showed is at ceiling in both arms and D73 showed is near-tautological, since Huginn re-injects the prompt every unroll and random weights decode a count at R²=0.99999. `promptdepth` swept prompt phrasing × recurrence depth; D85/D86 supersede it with 21 families and a graded readout. Neither would add anything to the current record. The supervisor killed the running `geometry-cap-content` job. | **dropped** |

## E. New open threads, 2026-08-09 (post machine-switch session)

### E1. A tested parser for gold/answer extraction — closing A7/A8 properly

D89 found ONE defect (first-token-only scoring) affecting 8 of 21 battery families,
and showed D85's conclusion survives it. That is not the same as ruling out every
scoring artefact across the battery's LOW-accuracy end. The user's framing is right:
disentangle GENUINE model failure from INSTRUMENT failure, property by property,
not family by family guessed at.

**What a proper audit needs to check, per family:**
1. **Gold tokenisation** (D89's defect) — already checked, `docs/claims_ledger.md`
   D89(1) has the full family list.
2. **Prompt-format sensitivity** — D69 showed ONE instruction ("Reply with only the
   answer") moved `echo_digit` 0%→83%. D86 showed the SAME instruction is worth only
   +9.5 points at r=4 averaged over 21 families and ~0 elsewhere. So format
   sensitivity is family-specific and UNMEASURED per-family beyond that one test.
3. **Depth/`num_steps` sensitivity** — D68 showed accuracy is NON-monotone in r,
   peaking shallow for some families. Every battery run so far used ONE fixed
   depth (`geometry-battery`/`geometry-geomcap` at whatever `NUM_STEPS` each used).
   A family reading 0% at that one depth could be >0% at another (D86 is the
   existing evidence this happens).
4. **Huginn's own special tokens** — the user's specific question: does this
   project's prompt-building correctly use Huginn's begin/end-of-text,
   begin/end-of-turn tokens (ids 65504/65505/65508/65509, confirmed in
   `raven_modeling_minimal.py`'s generation config and used correctly in
   `batched_generate`'s stop-token set) EVERYWHERE, or did any kernel roll its own
   parsing that diverges? Worth a static AST-level audit across
   `scratch/kaggle_*/body.py`, mirroring `tests/test_kernel_tasks.py`'s existing
   drift guards (which already check `coda_head` and `items()` for drift — extend
   the same discipline to tokenisation/stop-token handling).

**Proposed design, not yet built.** A tested module
(`src/traj_geom/scoring.py`, following the `numerical-research-code` skill's
discipline: single-source the scoring logic, pin known-answer cases, fail loudly on
ambiguous input) with:
- `extract_gold_rank(logits_row, gold_ids)` — the SAME first-token-rank logic
  everywhere, but with an explicit `multi_token: bool` flag surfaced in every output
  record, computed from `len(gold_ids) > 1`, so "first-token defect" becomes
  something every downstream table can filter on rather than something that has to
  be re-discovered per-analysis the way D89 was.
- `extract_decoded_answer(text, format: Literal["bare","constrained"])` —
  containment/exact scoring (already exists in `run_depth_accuracy.py`/
  `run_correctness_decode.py` as `normalise`/`score`; the ask is to CONSOLIDATE
  these into one tested module rather than the current per-script duplication).
- A **per-family diagnostic table**: family × {multi_token_gold, best_depth_used,
  constrained_format_tried, accuracy_at_best_depth_and_format} — run offline from
  data ALREADY BANKED (`geomcap.csv`, `depthacc.json`) wherever possible, no new
  GPU needed for most of it.

**On the user's workflow/subagent suggestion.** Given this is fundamentally a
STATIC, single-file audit task (read kernel bodies, check for a specific pattern)
rather than an open-ended research question, it does not obviously need a
multi-agent workflow — a single focused pass would likely be cheaper and just as
thorough. Revisit if it turns out to require reading and cross-referencing many
kernel bundles in parallel.

### E2. Do published Huginn papers report genuine task accuracy anywhere?

The user's question, unanswered: does Geiping et al.'s own paper, or any of the
2026 follow-ups the literature scout found (`docs/related_work.md`), report Huginn
achieving non-trivial accuracy on some standard benchmark — which would sanity-check
whether this project's battery reading mostly 0-30% is a genuine capability floor of
a 3.5B model or an artefact of THIS project's own prompting/scoring (D89-style).

**Not yet checked.** `docs/related_work.md`'s scouting pass focused on trajectory
geometry / looped-transformer mechanics, not Huginn's benchmark numbers. Geiping et
al.'s own abstract (fetched earlier, `docs/PLAN.md` §0) mentions the model is
"surprisingly capable in reasoning" — worth reading the actual benchmark table
(GSM8K, ARC, etc. are the likely candidates given the training mixture includes
`nvidia/OpenMathInstruct-1`, `meta-math/MetaMathQA`, `hkust-nlp/gsm8k-fix`) rather
than trusting the marketing-style abstract line. If Huginn scores e.g. 30-40% on
GSM8K at higher `num_steps`, that directly bounds what this project's near-zero
battery accuracies could mean — either the battery's synthetic tasks are
harder/more OOD than GSM8K-style problems (plausible: ciphers, parity, indexing are
not what the training mixture emphasises), or something in the scoring is still
wrong beyond D89's fix.


### E3. How to MINE for good task statements and setups — the strategy, and its traps

**Why this needs a strategy at all.** Two runs have now failed on task supply
rather than on method: D47 (no dynamic range in the outcome) and D95 (2 of 9
items split, gate VOID). `geometry-census` searches the EXISTING pool of 21
hand-authored families x 12 items. **If that pool contains no good items, the
census returns nothing and we have learned only that the pool is bad.** Searching
a fixed pool is not the same as generating good tasks.

**What "good" means here is FIVE criteria that do not coincide, and that is the
whole difficulty.**

| criterion | why | who violates it |
|---|---|---|
| accuracy strictly inside 0-1, ideally near 0.5 | a pinned outcome carries no correctness information however much h_0 moves it (D90(6)) | most of the battery: `echo_digit` at 100%, `rot13_word` at ~0% |
| difficulty set by an explicit knob | the supervisor's actual ask -- vary required reasoning steps while holding everything else fixed | the battery families are single-difficulty by construction |
| token count NOT covarying with difficulty | else geometry tracks LENGTH, which it demonstrably does (D26, D84, 6 of 18 cells in D85) | almost everything; only the D87 lenmatch design controls it |
| single-token gold | else the rank readout scores the leading token, not the answer (D89) | 8 of 21 battery families |
| in-distribution enough that the model can do it | Huginn gets GSM8K 32.6%, ARC-E 69.9% on natural-language in-mixture tasks, but our synthetic symbolic families read near zero | the synthetic battery |

**Criteria 2 and 5 pull directly against each other**, and that tension is the
core problem: clean difficulty knobs come from synthetic tasks the model is bad
at; tasks the model is good at come from natural-language data with messy,
uncontrolled difficulty.

**Five mining strategies, roughly in order of expected value.**

1. **A DIFFICULTY x DEPTH GRID, swept in one pass -- NOT a binary search.**
   *(Design corrected 2026-08-09 after the supervisor pointed out the flaw in the
   first version, which proposed binary-searching n for the accuracy threshold.
   That was wrong: accuracy at a given n is a noisy Bernoulli estimate, so a
   staircase that commits to a direction at each step chases its own noise, and
   near the 20-80% band -- exactly where we want resolution -- the noise is
   largest. It also throws away every level it steps past.)*

   Instead: define each family as a GENERATOR with an explicit integer knob n,
   and sweep **the whole n-grid at once with a small number of h_0 draws per
   cell**, reporting the full accuracy surface rather than one threshold.

   **The reason this is nearly free is a property of the harness that must not be
   re-derived: one forward already yields correctness at EVERY depth.** The
   read hook applies the D71-validated `coda_head` at each unroll of a single
   continuous run (`rank_curve_only`, `bank_one`), so a forward at
   `num_steps=48` returns the gold rank at all 48 depths. **Never re-run a
   prompt per depth.** So the cost of the sweep is
   `families x n-levels x h_0-draws` forwards, and the DEPTH axis comes for
   free, giving a (difficulty x depth) accuracy surface at the price of a
   difficulty-only scan.

   Three things then fall out of the same data, which is why this design
   dominates: (a) the n where accuracy enters the measurable band, per family;
   (b) the depth at which each n becomes solvable -- directly the "more
   reasoning steps for harder problems" object H2 is about, measured
   behaviourally rather than geometrically; (c) the h_0 spread per cell, i.e.
   the D90 confound, visible as the within-cell variance instead of being a
   separate experiment. Small per-cell n is fine precisely because the grid is
   dense and smooth in n -- neighbouring levels pool.
2. **CLRS-Text, the rare intersection.** It is IN Huginn's training mixture
   (`tomg-group-umd/CLRS-Text-train`, confirmed in the model card) AND has a
   clean integer problem-size knob. That is the one place criteria 2 and 5 are
   satisfied at once. `geometry-clrs` is testing it now.
3. **Prompt format as a deliberate lever, not a nuisance.** D69 moved
   `echo_digit` from 0% to 83% with ONE added instruction. So format can move a
   task into the measurable band without changing the computation at all. Any
   titration should sweep format as a second axis, and D86 shows the effect is
   family-specific rather than global (+9.5 points at r=4, ~0 elsewhere).
4. **Length-matching by construction, verified not assumed.** The D87 trick --
   two prompts with byte-identical bodies differing in one marker character --
   is the only design in this project proven to escape the length confound, and
   it passed its gate absolutely (188/188, token multisets identical). Any new
   difficulty-controlled family should be built this way where possible, and the
   token-count gate must be CHECKED in the kernel, as D87 checked it.
5. **h_0 as the micro-knob.** Once an item sits near the boundary, the unseeded
   h_0 ensemble supplies within-item outcome variance for free (D90). This is
   the last mile, not the search itself.

**THE TRAPS, and one of them is already live in `geometry-census`.**

- **Winner's curse / regression to the mean -- RAISED, THEN CHECKED, AND IT IS
  SMALL. The rule first written here was wrong and is corrected in place.** The
  concern was that `geometry-census`'s `summarise` pools the 4 stage-1 screening
  draws with the 20 stage-2 draws, so items selected for looking balanced would
  be re-scored using the very draws that selected them. That is a real mechanism,
  so this entry originally required the offline analysis to use **stage-2 draws
  only**. **Simulation refutes that requirement.** Over 4000 simulated items under
  the kernel's own selection rule (both classes present in 4 screening draws):

  | estimator | bias in apparent balance | mean \|estimate − truth\| |
  |---|---|---|
  | stage-2 only | −0.0180 | 0.0778 |
  | **pooled** | **−0.0014** | **0.0698** |

  Pooling wins on both. The selection condition is weak -- it excludes only items
  pinned near 0 or 1 -- so it induces little curse, while discarding 4 of 24 draws
  costs real precision; and the residual bias is dominated by noise passing
  through the non-linear \|f − 0.5\| transform, which penalises the SMALLER
  sample. `scripts/run_census_analysis.py` therefore reports the **pooled**
  estimate as primary, prints the stage-2 estimate beside it, and prints the
  measured gap so the curse's size is visible rather than asserted. *(The stage
  split is still computed, because it is what makes the comparison possible at
  all -- rows are stored in append order and each item's first 4 are stage 1.)*
- **Circularity: selecting on the outcome, then predicting the outcome.** If
  items are chosen because correctness varies and we then ask whether geometry
  predicts correctness, the selection and the analysis share data. The correct
  pattern is the one `geometry-h0bank` used: **select the boundary prompts from
  a DIFFERENT run's ranks (geomcap), so selection cannot be circular with this
  run's labels.** Any census-selected item set must be used the same way -- as a
  selector for a FRESH banking run, never analysed on the census's own draws.
- **Selection restricts generalisation.** Results on titrated boundary items are
  claims about boundary items, not about the model's geometry in general. That
  must be stated in whatever claim they support, as D79's "in the 1 family of 4
  that the design could test" states it.
- **D89 and the length confound do not go away** just because the items are
  better chosen; the single-token-gold and token-count gates still have to run.

**Concrete next build, if the census returns too few splitting items:** the
titration kernel in (1), over parametric generators with an explicit n, is the
principled replacement for a fixed item pool and is the thing to build rather
than re-running a wider census over the same hand-authored families.

---

## F. Audit findings, 2026-08-09 — recovered from a workflow whose verify stage never ran

**Provenance and status.** A 5-agent audit of the D93-D102 work was launched, paused
for an unbounded verifier fan-out, and its findings were then recovered directly from
the agent transcripts at zero further agent cost. **These are UNVERIFIED auditor
claims.** Each is either checked and actioned below, or left explicitly open. Do not
treat any of them as established until its line says so.

### F1. CHECKED AND ACTIONED

- **F2.1 — CONFIRMED, and D96 is corrected.** Recomputed directly: pooled
  rho = +0.750 / p = 2.4e-05; **item-clustered permutation p = 0.093**; within
  A-B rho = +0.566 (p = 0.055); **within A-C rho = exactly 0.000**. The auditor
  was right that the pooled figure is largely a between-contrast offset. D96(3b)
  now carries this, and the net effect is that D96 becomes *more* inconclusive:
  neither "QK tracks the computation" nor "QK tracks difficulty" is established.
- **F2.11 — CONFIRMED, folded into D96(3c).** 0.00015 is below the exact floor
  1/4096 of a 12-pair sign-flip test. No conclusion changes; the endpoint is not
  attainable and the exact test should be used when the pattern space < N_PERM.
- **F2.5 — CONFIRMED and WORSE than reported.** The auditor said 1 of 6 prompts
  fall in the quoted 0.09-0.11 residual band; the true count is **0 of 6**
  (actual range 0.052-0.402), and the reduction is **159x-1185x, median 384x**,
  not "~600x". Root cause: I quoted `echo_digit`'s four PER-BLOCK residuals as
  though they were the across-prompt range. Corrected in the ledger, RESULT.md
  and UNDERSTANDING.md. **D98's conclusion survives** — the minimum
  separation/residual ratio across prompts is 95.4.
- **F2.2 — CONFIRMED, propagated.** The retracted D94 claims and the n = 960
  figure were fixed in the ledger only. Now corrected in `UNDERSTANDING.md`,
  `RESULT.md`, `PLAN.md`, and both research inquiries: rho is quoted raw (0.855)
  with the +0.033 bias named and the ~0.82 corrected value, against D31's
  independent 0.79-0.81, and n_eff is stated as 16 prompts.
- **F2.3 — CONFIRMED, fixed.** `RESULT.md`'s second headline no longer says "no
  loop or drift regime exists anywhere we have looked"; it is scoped to the
  fixed-block map with the other two senses named.
- **F2.6 — CONFIRMED, fixed.** `research_inquiry_3_interp.md` carried the
  inverted attenuation direction; corrected, along with the deeper point that
  same-prompt patching is inert at both ends by construction.
- **F2.7 — CONFIRMED, and it goes further than the auditor said.** D31's row
  ends: *"NEXT: measure the FULL operator (all positions perturbed and read)
  rather than the diagonal block, which is the version whose spectrum should
  match the observed orbit decay."* So the "no line attractor" claim rests on a
  sub-operator our own ledger says is not the governing one, and the full
  operator has still never been measured. Added as a third caveat in
  `UNDERSTANDING.md`. **It also weakens D94 further:** comparing the pair-based
  ρ against D31's 0.79–0.81 is not a like-for-like comparison of the same object.
- **NEW, found while checking F2.7 — D32 is a positive counterweight on H3 that
  no synthesis document cited.** After residualising out position (which alone
  explains 98.5% of a running count, so the raw probe is worthless), the latent
  still predicts the count at **R² = +0.601** on residuals of sd 1.06 — D32's own
  reading: *"evidence that a running count is maintained."* Task b (nesting
  depth) is cleaner: R² = +0.590 / +0.718 against position's 0.160. Now cited in
  `UNDERSTANDING.md`'s H3 section, where it converges with the scope argument and
  the re-injection architecture.

**Guard added, because the pattern here is propagation rather than analysis:**
when a ledger row is corrected, grep every other document for the retracted
phrasing **in the same commit**. Four of the six items above existed only because
that was not done.

### F2. STATUS AS OF 2026-08-10 — **9 of 13 verified RESOLVED, 3 still live, 1 undecidable**

**Read this before the table below.** The table was written when all 13 were open and
still reads that way; every row is preserved verbatim because a corrected audit item is
evidence about how the correction went, but **the table alone overstates the number of
live errors by four times.** Each verdict below was checked against the file, not
against memory.

| # | verdict | what the check showed |
|---|---|---|
| F2.1 | **RESOLVED** | `run_qk_analysis.py` 205-206 now runs an item-clustered permutation and states "Both are reported; **the clustered one decides**", at p = 0.093 — within rounding of the auditor's 0.095. |
| F2.2 | **RESOLVED** | None of "two unrelated instruments", "independently measured", "n = 960" appears in `UNDERSTANDING.md`, `RESULT.md` or `PLAN.md`. |
| F2.3 | **RESOLVED** | "no loop or drift regime exists" does not appear in `RESULT.md`. |
| F2.4 | **RESOLVED** | `PLAN.md` 164 now attributes [0.808, 0.920] to the `contraction` column of `results/geomcap.csv`, not to D52. |
| F2.5 | **STILL LIVE** | `RESULT.md` 72 still reads "per-block residual 58–71 → **~0.1**". The auditor's point stands: that is one prompt's value presented as the general one, with three prompts 2–3.5x above it. The *conclusion* survives (minimum separation/residual across prompts is 95); the quoted number does not. **Cheapest fix: quote the per-prompt range.** |
| F2.7 | **RESOLVED** | No line-attractor claim survives in `UNDERSTANDING.md` — removed in the 2026-08-10 rewrite. |
| F2.8 | **RESOLVED** | Re-simulated under a bimodal prior; `run_census_analysis.py` amended. |
| F2.9 | **RESOLVED** | `split_stages` fixed so a failed screening draw consumes its slot; pinned by `tests/test_census_stages.py`. |
| F2.10 | **STILL LIVE, but cosmetic** | `run_h0_within.py` 33 still Bonferronis over 8 windows. That is correct for its own pre-registered P5 and D93's verdict stands; what needs restating is **D93(5)'s SECONDARY multiplicity argument**, which is vacuous rather than informative. A wording fix in one ledger row, no re-run. |
| F2.11 | **RESOLVED** | `require_exact_test_if_small` is called at `run_qk_analysis.py` 112, with the 4096-point-space incident recorded in the comment above it. |
| F2.12 | **STILL LIVE** | `related_work.md` carries 9 `[V]` tags; the contradictory pair was not reconciled. Cheap, and it matters because [V] is the only thing standing between us and a repeat of the Movahedi/Du citation failures. |
| F2.13 | **RESOLVED** | `RESULT.md` 300-303 now states outright that the null survived a tokenisation-free re-test **and** that "every number quoted from the first-token axis is a first-token number and should be read as one". |
| F2.6 | **UNDECIDABLE HERE** | `research_inquiry_3_interp.md` exists and was already sent; no live harm either way, and re-reading it cannot unsend it. Left alone deliberately. |

**What the resolution pattern says.** The four that remain are all *wording* — a
single-prompt number quoted as a range, a vacuous secondary argument, an inconsistent
tag. Every item that touched a **number or a code path** was fixed. That is the
opposite of the failure mode the table's closing paragraph predicted, and it is worth
recording: the propagation guard worked, and what it does not catch is prose.

### F2 (original table). OPEN — highest consequence first

| # | claim | file | why it matters |
|---|---|---|---|
| F2.1 | **`run_qk_analysis.py`'s confound test is pseudoreplicated.** The 24 observations are 12 items x 2 contrasts, and both members of a pair share the same `A` value, so they are not independent; scipy used df=22 rather than at most df=10. The auditor computes the clustered p as **0.095, not 2.4e-05**. | `scripts/run_qk_analysis.py` ~185; D96 | **This is the number D96 uses to demote the QK result to "confounded with difficulty".** If it does not hold, D96's verdict needs restating in EITHER direction — the confound may still be real but is not established at the quoted strength. Same error class as D94's. |
| F2.2 | **The retracted D94 claims were fixed in the ledger row only.** `UNDERSTANDING.md`, `RESULT.md` and `PLAN.md` reportedly still assert "two unrelated instruments", "independently measured", and the un-caveated rho = 0.855, and still quote n = 960 rather than n_eff = 16. | `docs/UNDERSTANDING.md`, `docs/RESULT.md`, `docs/PLAN.md` | A correction that reaches the ledger and not the two documents CLAUDE.md section 6 designates as the current-state record is not a correction. **This is a process failure, not a typo.** |
| F2.3 | **`RESULT.md` still says "no loop or drift regime exists anywhere we have looked"** — the exact absolute form the three-sense taxonomy retired, and which D98 (sense ii, present) and D55 (sense iii, present) contradict. | `docs/RESULT.md` | The argument document asserts an absolute negative that two of our own measurements refute. |
| F2.4 | **The [0.808, 0.920] interval is misattributed.** The auditor says it is not D52's at all but the single-orbit self-convergence fit that `metrics/regime.py` documents as the near-tautological mistake `contraction_from_pair` exists to replace. | `docs/PLAN.md` section 1; D94(3) | If true, **D94(3)'s own correction identified the wrong source**, and the cross-method support for rho is weaker still: only one estimator family behind 0.855. |
| F2.5 | **D98's per-block residual "0.09-0.11 / ~0.1" is not the data.** Only 1 of 6 prompts lands in that band; three sit 2-3.5x above it. The "~600x reduction" holds for one prompt (per-prompt ratios ~187x to ~1735x). | D98; `docs/RESULT.md`; `docs/UNDERSTANDING.md` | The conclusion survives (minimum separation/residual across prompts is 95) but the quoted numbers are a single prompt presented as the range. |
| F2.6 | **`research_inquiry_3_interp.md` still states the attenuation direction backwards** — the error corrected in D95 and `RESULT.md` was never propagated to the inquiry, which was then sent to an external researcher. | `docs/research_inquiry_3_interp.md` | Asks outsiders how to adapt patching under inverted physics. Already returned, so no live harm, but the file is wrong in the repo. |
| F2.7 | **`UNDERSTANDING.md`'s H3 line-attractor conclusion omits D31's own caveat** that it measured a sub-operator its author argues does not govern the orbit, and that the full operator was never measured. | `docs/UNDERSTANDING.md`; D31 | H3's strong mechanistic form ("no line attractor, so no running count") would rest on the spectrum of an operator our own ledger says is not the governing one. **This compounds with the D99/toy-model correction, which already narrowed that claim.** |
| F2.8 | **`run_census_analysis.py`'s pooling justification may invert under a bimodal prior.** My simulation drew true accuracies from Uniform(0,1); this repo's items are mostly pinned near 0 or 1, where the "splits in 4 draws" condition is highly informative and the curse is larger. | `scripts/run_census_analysis.py`; `directions.md` E3 | **I already corrected this rule once, in the direction the auditor now questions.** The census has not returned, so the decision is not yet load-bearing — but re-run the simulation with a bimodal prior before using either estimator. |
| F2.9 | **`split_stages` mis-assigns stage-2 draws to stage 1 whenever a screening draw failed**, corrupting the exact comparison the winner's-curse check rests on. | `scripts/run_census_analysis.py` 63-83 | Cheap fix, and it matters only if the census had failed forwards — check the census output for `ok: false` rows before trusting the printed curse block. |
| F2.10 | **`run_h0_within.py`: alpha is Bonferroni over 8 windows while 25 tests are run**, and at the honest 24-test alpha the permutation resolution (1/401) makes significance unreachable. | `scripts/run_h0_within.py`; D93(5) | The pre-registered P5 verdict stands on its own terms; the SECONDARY multiplicity argument in D93(5) is vacuous rather than informative and should be restated. |
| F2.11 | **A reported permutation p (0.00015) is below the exact floor of its own test** (12 sign-flip pairs give 2^-12 = 0.000244). Monte-Carlo artefact of `(hits+1)/(N+1)` with N=20000 over a space of 4096 patterns. | `scripts/run_qk_analysis.py` 85-102; D96 | No conclusion changes (all windows still clear alpha under the exact test) but the quoted endpoint is not an attainable p-value. **Use the exact test when the pattern space is smaller than N_PERM.** |
| F2.12 | **`related_work.md` gives the same paper two contradictory verification statuses** and stretches [V] to cover a scouting read, which the doc's own key assigns to [U]. | `docs/related_work.md` | The verification ladder is what prevents a repeat of the Movahedi/Du citation failures; if [V] no longer means "I read the primary source myself", it has stopped discriminating. |
| F2.13 | **The D89 wording amendment `PLAN.md` mandates was never applied** to `RESULT.md` or `UNDERSTANDING.md`: the capability axis behind the central null was first-token-scored for 8 of 21 families. | `docs/RESULT.md`, `docs/UNDERSTANDING.md` | D89 reports the null survives a tokenisation-free re-test, so the conclusion holds — but the independent variable is quoted as exact-match accuracy when it is not. |

**The pattern worth naming.** Most of these are not analysis errors; they are
**propagation failures** — a correction made in the ledger and not carried into the
synthesis documents, or a number quoted from one prompt as though it were a range.
That is a different failure mode from the four caught earlier today by reading raw
output, and it needs a different guard: **when a ledger row is corrected, grep the
other documents for the retracted phrasing in the same commit.**

---

## G. Unmerged parallel work, and new hypotheses (2026-08-09 evening)

### G1. `claude/geometry-remote-continued` is NOT merged and contains real results

A parallel remote session halted with a handoff note
(`../files/claude-files/9-aug-11_40-claude-halt.md`, read 2026-08-09 evening).
Its branch carries commits `92600a2` and `44255b2`, confirmed **not ancestors of
our HEAD**. It holds:

- **A result we do not have:** *"depth does not compute the answer, it relocates
  it"* — accuracy at the answer position peaks at unroll 5 (16.1%), flat at 6.2%
  from unroll 8, identical across all 21 families from r=32 (family-paired sign
  test 14/15, p = 9.8e-4), containment rising 29.4% → 42.9% while a
  length-matched decoy falls 16.0% → 2.7%. With `docs/depth_profile.md`,
  `scripts/run_depth_profile.py` (regenerates every number with no GPU) and 11
  tests.
- **The D103 defect**, which we have now independently replicated.
- **A parsing hazard worth knowing:** Huginn leaks role markers with no
  separator (`'4Huginn'`), which silently defeated a boundary-matched scorer on
  exactly the low-depth blurts.
- `scratch/kaggle_answerpos/` built, linted, tested, **not pushed**.

**A LEDGER NUMBERING COLLISION EXISTS: that branch's D92–D94 are different claims
from ours.** Merging requires renumbering one side. Deliberately deferred rather
than attempted under deadline — but **the branch must not be lost**, and whoever
merges must renumber, not overwrite.

### G2. New hypotheses, formed from synthesising today's results

Each is stated with the test that would kill it, and **all three of the first are
runnable offline on already-banked data**, which matters because GPU is the
current bottleneck.

**G2.1 — The computation lives in the fixed point *h\*(e)*, not in the path.**
Everything we measure about the *path* is dominated by the contraction's clock
(D80) and by architecture (the period-4 cycle, D98/D100, invariant across tasks
and difficulties). The task-relevant information enters through *e*, which is
re-injected every unroll, and determines *where* the fixed point sits rather than
*how* the path reaches it. This single hypothesis retro-explains: D88 (shape reads
the input token — because *e* is a function of the token), the total absence of a
geometric H2 signal (path shape is architecture plus clock), the weakness of
correctness decoding from shape (D92/D93), and D32's positive result (a running
count IS decodable — from the state, i.e. from *h\**).
**Test, zero GPU:** on banked orbits, decode task variables from the FINAL state
versus from the shape code, matched feature counts. The hypothesis predicts the
final state wins substantially. **Killed if** shape matches or beats it.

**RUN 2026-08-09, AND THE TEST AS SPECIFIED WAS MIS-DESIGNED — recorded because
the mistake is instructive.** I ran it within-prompt on the 320 splitting h0bank
orbits: shape 0.520 (null 0.471, p = 0.030); final state h\* 5280-d **0.451**
(p = 0.79); final state at a matched 66 random projections 0.493 (p = 0.42).
That looks like a refutation and is not one. **Within a prompt, *e* is FIXED, so
*h\*(e)* is the same target for all 32 draws** — after within-prompt centring
there is almost nothing left in the endpoint to decode, which is exactly what
G2.1 itself predicts. The test cannot discriminate; it needed a BETWEEN-prompt
contrast, where *e* varies.

**What it does establish, and this is worth keeping:** whatever correctness
signal exists within a prompt lives in the **path, not the endpoint** — shape is
weakly above its null while the raw final state sits at chance on both a full
5280-d and a matched-dimension projection. Combined with D93's timing control
(`best_depth` alone decodes correctness about as well as the best shape window),
the most economical reading is that **within-prompt correctness is a fact about
WHEN the transient peaks, not about where it ends up.**
**Redesigned test, still zero GPU:** decode the gold VALUE across items within a
family at matched token count (the D87/lenmatch banks) from *h\** versus from
shape. There *e* genuinely varies, so the hypothesis is testable.

**G2.2 — The period-4 cycle is an architectural carrier; any task signal lives in
deviations from it.** D100 found every cycle statistic invariant across tasks and
difficulty — which is exactly what a carrier looks like. **Test, zero GPU on the
block-resolved banks (6 prompts in `ds_blockcycle`, 12 in `ds_cyclegeom`):**
compute the mean cycle across prompts, subtract it, and decode task identity from
the residual. **Killed if** the residual decodes no better than chance.

**G2.3 — The dynamics are ANGULAR, and every Euclidean statistic we have computed
is a chord approximation.** D99 measured ‖h‖ constant to 1.3e-4, so the state
moves on a sphere of radius 76.39. Participation ratio, step cosine, winding and
inter-vertex distance are all ambient-space quantities on a curved manifold.
**This is a candidate explanation for why every rotation/winding instrument
failed** (D28, D74(6), D83): rotation on a sphere must be measured in the tangent
space, not in ambient coordinates. **Test, zero GPU:** recompute the key
statistics as geodesic/tangent-space quantities on the banked orbits and see
whether anything sharpens. **Killed if** the geodesic versions track the Euclidean
ones to within noise — which, given the tiny radius variation, is the honest prior.

**G2.4 — Depth degrades the answer at the read position, and H2 should be asked
about the PEAK.** D103 measures gold top-1 at 11% for the best fixed depth and
0.7% at r=47. Combined with D86 (containment rises while exact-match collapses),
the picture is that the answer becomes available early and is then buried. So the
H2-shaped question is not "do harder problems need more depth" but "does the peak
move later" — precisely D101's `best_depth`. **This makes D101 the right
instrument rather than a consolation prize**, and a powered re-run of it on
`nth_item_k` and `addk` alone is the highest-value H2 experiment available.


## H. Runs landed 2026-08-10 (overnight battery) — status, not narrative

Three DataSphere jobs launched and landed in one night, plus four local re-analyses of
already-banked orbits. Ledger rows carry the controls and limits; this table exists so
nothing here is later called "never run".

| run | job | question | outcome |
|---|---|---|---|
| **A23** `ds_embsep` | `bt19lv117dbh0bqrcbcv` | does the token EMBEDDING select the rotation regime? | **DONE -> D135, D136.** NULL. Six classifiers, best-of-six 0.551 against a null-of-max 0.589, p = 0.808, below the 0.592 majority baseline. Refutes D134's untested "by elimination" inference in both its linear and nonlinear forms. |
| **A24** `ds_einterp` | `bt1igml4vt3so359mdp3` | interpolate `e` between a settling and a rotating noun — where does the regime flip, and how sharply? | **DONE -> D140.** All 8 cross-regime paths cross the threshold exactly once, every crossing inside one 0.05 grid step. At t = 1 the orbit matches the rotating noun's own R within the h_0 floor despite a foreign h_0 — D111/D113's parameter claim, causally. **The rotating set is NOT convex**: settling->settling crosses twice. P1 unevaluable (unseeded h_0, D78). |
| **A21** `ds_gmres` | `bt13o2rva6j2mk5ls0u8` | re-solve DR3 Method 2's adjoint with GMRES instead of Neumann | **DONE -> D142.** Converged **24/24** at tol 1e-3 in 39-48 matvecs, against D131's **1/24** exhausting a 120-term cap; the count lands in DR3's predicted band. P4 (the instrument's own null) passes only weakly at mean rho = +0.3732, so the marker's median rank 13/51 vs a random direction's 33 is informative but not localising. |
| **B4c** `ds_nth` | `bt1j97ks2han4ptnks5q` | the orthogonal difficulty ladder, third attempt | **DONE -> D143.** Every structural gate passes for the first time (37 tokens at all levels, 8/8 levels live, no multi-token golds). **H2 null once list position 1 is removed**: +0.2825 (p = 0.0051) collapses to -0.0286 (p = 0.7986). |
| **D137** (local) | — | is the regime a thresholded continuum? | **DONE, then AMENDED by D141.** `word` is a detector false positive, `set` genuinely straddles. |
| **D138** (local) | — | does the structure replicate on other banks? | **DONE -> partially RETRACTED by D141.** The gap replication was the wrong statistic; the threshold replication (691/696 orbits across three banks) survives and is stronger. |
| **D139** (local) | — | does symbol diversity modulate the regime continuously? | **DONE.** D129 survives: 0 of 132 orbits cross the threshold. The graded arm does not survive the correct unit (p = 0.0143 -> 0.0973). |

**Two kernels died on setup bugs before these landed**, each after a full ~12-minute
clone + install + weight download: a missing `scipy` (the lean install recipe does not
carry repo dependencies) and a `KeyError` from item keys the builder never wrote. Both
are now `scripts/preflight.py` checks 4 and 5, each verified against the genuine pre-fix
git revision and against all 24 kernels for false positives. See `PRACTICE.md` RC3.

### H1. What these leave open

- **A seeded re-run of A24** would establish patch fidelity, which its P1 could not.
  One cheap job; D78's prescription (seed h_0 per forward) is already in `constants.py`.
- **Extending A21's P4 to all 24 prompts** turns rho = +0.3732 from an estimate into a
  bound. Brute-force ablation is ~51 forwards per prompt, so under an hour.
- **B4c's ladder is reusable and its limit is recorded**: structural difficulty, not
  behavioural — Spearman(k, accuracy) = -0.1350, p = 0.1769.


## I. Strategic audit, 2026-08-10 23:45 — answers, not just state

First run of `scripts/strategic_audit.py`. The state it assembled, then the five answers.
Recorded here rather than in a message because the answers change what gets queued.

**The signal that matters most, and it is uncomfortable:** of the last twelve ledger rows,
**6 are bounded negatives, 5 are repairs of our own record, and 1 is a new positive.** The
project is currently spending most of its output correcting itself. That was necessary —
the repairs found a scorer defect that overturned a headline, an axis that reads 1.000 on a
task the model cannot do, and a control eight rows depended on that had never been checked.
But it is not a state to stay in, and noticing it required counting rather than recalling.

**1. What is the binding constraint?** **The model.** Task supply was the answer once and
D130 closed it. The capability axis was the answer next and D145/D147/D153 characterised
it. What is left is Huginn itself: the one difficulty axis where theory says recurrence is
*required* is out of its reach entirely (D147, a constant responder); we cannot reproduce
its own published ARC-Easy figure (D153, 0.407 against 0.699); and the one dramatic
phenomenon we did find has no behavioural consequence (D148, 0 of 48 answers changed).
**Every functional road now ends at "the model is too weak to carry the question."** Since
the scope is Huginn-only by instruction, **the remaining tractable questions are geometric,
not functional** — which is also where the supervisor's interest lies.

**2. What is being paid repeatedly?** Weight downloads: **≥ 4.8 GPU-hours** across runs
visible locally, and the platform's job count is higher still. `REMOTE_RUNS.md` has
recorded a verified mechanism to remove it for a day. **Now delegated** to a `c1.4` CPU job
that builds the weights into a mountable dataset without touching any existing kernel.
Second: **`scratch/RUNS.json` registers 22 runs against 146 jobs on the platform** — the
operational registry `runs_status.py` reads is 15% complete, so any "have we run this?"
question answered from it is unreliable. That is the same class of defect as D119, where a
thread was called never-done while a bank had already answered it.

**3. What class of result cannot be produced, and why?** A **functional consequence of any
geometric phenomenon**. This is a MODEL limit, not an instrument or design limit, and the
distinction is now measured rather than assumed: A26 demonstrated the instrument is
sensitive — the readout tracks displacement of the map's parameter at ρ = **+0.82** (D154) —
and still found no effect of the regime. More GPU does not fix a model limit.

**4. What would fail if queued now?** **Another capability-flavoured run on Huginn.**
D118 (0% above one level), D128 (depth ran backwards), D143 (null once the trivial level is
removed), D147 (constant responder), D153 (29 points below the published number) are five
independent instances of the same outcome. Geometry runs (A31) are not in that class.

**5. Ten times the compute, or a tenth?** **The same answer both ways** — geometry on banked
data plus a small number of targeted interventions. **Compute is therefore not the
constraint**, and "queue a GPU job" should stop being the default action. The corollary:
prefer re-analysis of the 63 banked runs over new ones, which is how D138, D141, D146, D152
and D154 were obtained at zero GPU cost.


## J. RESUMPTION STATE, 2026-08-11 06:10 — written before a context compaction

Everything a fresh context needs to pick this up. Ledger is at **D162**.

### J1. No GPU jobs are running

All of the overnight battery landed and is written up. Nothing is in flight; the
DataSphere job list's top entries are the weights-dataset agent's `c1.4` jobs, already
complete.

| run | kernel | row | outcome |
|---|---|---|---|
| A26 | `ds_regimebehav` | D148 | behavioural null now **bounded**: 0 of 48 units change correctness; floor catches 4.7 unrolls at 99.3% |
| A27 | `ds_statetrack` | D147 | Huginn is a **constant responder** on state tracking; `min(rank)` reads 1.000 on it |
| A28 | `ds_boundary` | D149 | hyperplane **refuted**; region bounded, 49× anisotropic |
| A29 | `ds_arcrepro` | D153 | ARC-Easy 0.407 vs published 0.699 |
| A30 | `ds_arcproto` | D162 | **protocol refuted** — text arms 0.433/0.413, gap is real |
| A31 | `ds_aniso` | D156 | null, design powerless by construction |
| A32 | `ds_wteswap` | D161 | **embedding row causally controls the regime**; 10% edit destroys rotation |
| A33 | `ds_aniso2` | D160 | settling subspace survives further, p = 0.010, opposite to prediction |

### J2. The single most important thing learned, and what it implies

**D159: the answer lives in the transient, not at the fixed point.** Median argmin unroll
**4**; 17 of 21 families reach best rank by unroll 8; then the rank **plateaus at a stable
worse value** for forty-plus unrolls. Scored at the final unroll, **19 of 21 families read
0.000** (D158) and the census's 41 live items become **0** (D157, 5.29× inflation).

**Most of this project's geometry reads the converged state.** If that is where the answer
is not, then contraction rates, fixed-point similarity, the rotation regime and h\*
decoding are all measuring a state that does not carry the answer. **This is the strongest
unexploited lead in the project** and it is testable on banked data: decode correctness
from the state at the per-orbit argmin depth versus from h\*, on `kaggle_b6bank` (128
orbits, has `correct` + `rank_curve` + states) or `ds_blockbank` (130 orbits, 4 blocks).
Nobody has run it. D109/D125 tested h\* only.

### J3. The queue, in priority order

1. **Few-shot ARC** — D162's own cheapest next test and the only candidate sized right to
   close a 29-point gap (D60 measured **85 points** from prompt format alone on `copy`).
   A prompt change to `scratch/ds_arcproto/job.py`, no new machinery. If few-shot closes
   it, our capability axis is calibrated for the first time; if not, the checkpoint or the
   150-item prefix are next.
2. ~~**Transient-vs-fixed-point decode** (J2). Zero GPU.~~ **CLOSED by D163, negatively
   and without a run.** D93's powered sweep already covers the argmin window and shows no
   trend (Spearman(start, accuracy) = +0.14 / −0.31, both null, argmin window at chance).
   The direct version at n = 32 cannot resolve below Cohen's d = 10 raw, or d ≈ 2 even
   after projecting to 5 dims, against real effects of d ≈ 0.6. **Do not run it.**
3. **Bank the top-1 token at the fixed point.** D158/D159 both end at the same unanswered
   question — *what displaces the gold?* — and neither can answer it because only the gold's
   rank was banked. Costs nothing to add to any future kernel.
4. **Bounded chords between named tokens, many pairs.** D160's honest next step; A33's
   random-directions-in-a-span design cannot be interpreted.

### J4. Infrastructure now available and unused

**Weights dataset `bt102r0j5cb8r6r6nb36`** removes the 262–282 s download; mounting costs
**7.7 s**. Stanza and loader snippet are at the end of `REMOTE_RUNS.md`. **No kernel uses
it yet** — adopting it is a 4-line change per kernel and saves ~4.5 min per run.

Also there: the page-cache trap (`output-datasets` snapshots the block device, so a job can
report SUCCESS with data that never entered the dataset — `os.sync()` and let it return).

### J5. Practice state

`PRACTICE.md` now carries RC1–RC5 and the wake invariant. **RC5 is the live one**: RC1's
proxy-for-the-thing error committed *inside audit instruments*, which is worse because an
audit's output is trusted without re-checking. Four instances in one session, including two
in `build_index.py` and one in D150's own exposure count. The rule: **a tool's first output
gets the same premise-check an experiment's first run gets.**

`scripts/strategic_audit.py` exists and its first run is answered in section I above. Its
conclusion stands: **compute is not the binding constraint**, re-analysis of the 63 banked
runs outranks new jobs, and that is where D138, D141, D146, D152, D154, D157, D158, D159
all came from at zero GPU cost.

---

## §K State at 2026-08-11 09:00 MSK — in flight, landed, and what each closes

Written while three jobs run, per the standing instruction not to wait for compaction.

### K1 In flight

| id | job | question | status |
|---|---|---|---|
| A34 | `bt199d2ndbsrd91q0bcv` `ds_arcshots` | does few-shot close the 29-point ARC gap (D153/D162)? | EXECUTING |
| A36 | `bt146oseqilb4tbqbdpq` `ds_normchord` | is D164's `e`-vs-`wte` contrast curvature or norm? | EXECUTING |
| A38 | `bt13mu7dnv74kef3ogn5` `ds_genscore` | does the GENERATION contain the answer D166 puts one token behind? | EXECUTING |

### K2 Built, preflighted, NOT launched

- **A39 `ds_hyster`** — swap `e` at unroll k and leave it swapped, both directions, 8 switch
  points over 96 unrolls. Asks whether the regime follows the current parameter or the
  history. **Registers the first quantitative prediction that ties ρ to an observable:** at
  ρ ~ 0.83 the state closes 1% of the gap to a new attractor in ln(0.01)/ln(0.83) = **24.7**
  unrolls and 5% in **16.1**, so a relaxation of 16–25 unrolls confirms the contraction picture
  causally and closes UNDERSTANDING.md §6.2 ("ρ has never been connected to behaviour").
  Launch when a slot frees.

### K3 Landed since the last entry, at zero GPU

- **D164** — the rotating region is chord-convex in `e` (0 of 3 chords leave) and not in `wte`
  (2 of 2 leave), on the **matched pair** `symbol`↔`array`: min R **0.768** drawn in `e`,
  **0.400** drawn in `wte`. Resolves A32's failed P5 control as a coordinate artefact.
- **D165** — the linear-probe design cannot recover a label that is a guaranteed deterministic
  function of the features it is given: **0.690** from the raw states of the window the label
  is computed on, against a **0.600** baseline, while R as a single feature gets **0.980**.
  Retires every "not linearly decodable at n ~ 50" null, and kills the probing route to the
  onset question — which is why A39 asks it causally instead.
- **D166** — what displaces the gold is **`The`**, a sentence opener, not a role marker. The
  gold's final rank is median **2.5**, top-5 for 24 of 32. Refutes D145's hypothesis; keeps
  D157/D158's numbers and replaces their reading.

### K4 Dequeued, with why

- **Dating the regime's onset by probing.** Dead by D165 — no probe of that class can do it,
  at any n available here. Replaced by A39's causal version.
- **Re-establishing D135/D136 with a better classifier.** Dead by D165 for the same reason;
  a positive would not be believable and a null is uninformative by construction.
- **Any further regex-over-prose audit of the ledger.** Failed three times to identify claim
  structure; not rebuilt (RC5).

---

## §L Implementation available in `huginn-load`, and what it would have saved (2026-08-11)

The papers directory ships **code**, not only text: 60 `.py` files under `01/code/`, 18 under
`02/code/`. Read against this project's actual defect record, three items are worth adopting.

**L1 — run lm-eval directly instead of hand-rolling benchmark scoring.** `01/code/
recurrent-pretraining/evaluate_raven/local_lm_eval.py` is a thin wrapper over lm-eval's own
`TaskManager`/`simple_evaluate`; `saturation_eval_dist.py` gives the whole recipe — set
`model._model.config.mean_recurrence = num_steps`, pass `num_fewshot`, and let the harness
score. **This is the single highest-leverage item, because scoring is where we keep breaking.**
D175's `T_chat` arm was voided by a leading space moved into the prompt (BPE span
misalignment); the 13-arm 0.392–0.495 group in D182 is a prompt format *we invented*; D179 had
to sweep ten extraction rules because we score by hand. None of that arises inside the harness.

**L2 — `test_time_noise` is a built-in perturbation API we have been working around.** Five
schedules (`geom`/`sqrt`/`line`/`chi`/`fixed`), applied every unroll before the adapter, with
optional renorm through `core_block[-1].norm_4`. Our `e`-patching and state-injection kernels
hand-roll hooks to do less than this offers. Also `iterate_forward(init_scale=…)` — the h₀
scale is a free parameter we have never varied (A47 now touches it indirectly).

**L3 — what their code does *not* do, checked rather than assumed.** Paper 02's `*_inter_*`
files mean **intermediate recurrence steps, not intermediate token positions**:
`coda_lens_exp_inter.py` iterates `16*4` = 16 recurrences × 4 blocks, all at one position. So
the interior-position question **A46** is testing is unaddressed by paper 02 as well as by us —
only paper 01's qualitative figures touch it. A46 is novel against both.

*Not adopted today: L1 and L2 are changes to how we run, and every result now in flight was
launched under the current scheme. Adopting mid-flight would make the comparisons
cross-protocol, which is the error D183 just caught us making in the other direction.*

---

## §M ACTIONABLE QUEUE, 2026-08-11 15:35 — replaces §K, which was stale by six hours

*§K listed A34/A36/A38 as in flight; all landed long ago. Today's follow-ups have been living
in ledger row notes rather than as a queue, which is how they stay unactioned. This is the list.*

### M1 In flight
- **interp / steering vector on `e`** — subagent running. The one genuinely novel-by-literature
  direction (D188: zero steering work on any looped LM).

### M2 Broken, needs a decision
- **A47 `ds_h0inject` — ERRORED.** Output `h0inject.json` never written, so it died before
  banking. The replayed attach tail does not carry the job-side traceback and the local CLI
  logs are from my own attach calls. **Diagnosis inconclusive.** Options: re-run with the
  wall budget lowered so it banks a partial, or drop it — note that D190(a) makes a null there
  *partly predicted* by the paper's own path-independence claim, so its value fell today.

### M3 Cheap, decisive, not yet run
- **Period sweep on their word problem** (D192(4)). A46 found 0/35 rotating positions on
  *"Claire makes a 3 egg omelette…"*, max R 0.119 — but `rotation_power(period=6)` detects
  period 6 **only**. If their orbit has another period we would read zero by construction.
  One kernel; banks trajectories rather than R so the period can be swept offline afterwards.
  **If it also comes back empty, D185 strengthens considerably.**
- **`E_reason` arm of A43** — never ran (wall budget). Reason-then-mark is untested.
- **A system-prompt arm** (D189(c)) — paper 02 solved our exact problem with
  *"Always return only the final answer straightway."* D193 licenses only *"no **user-turn**
  format instruction works"*; this is the missing form.

### M4 Recorded but unbelieved — needs its own run before anyone quotes it
- **Padding improves final-unroll accuracy** (D186(4)): `echo_digit` 0.00 → **1.00**, pooled
  0.000 → 0.188, apparently suppressing D174's prose frame. Unpredicted and uncontrolled.

### M5 Blocked, and on what
- **Reconciling D98/D146 against papers 03 and 05** is blocked on D190(c): their `s*` is taken
  at **128** iterations and post-`ln_f`; ours at **64** and pre-`ln_f`. Equalise one side first.
- **"Sliders"** (D187d) — a third structure they name and we have never looked for. No
  definition extracted yet, so not yet runnable.

### M6 Dequeued, with why
- **Papers 04, 06, 10** — triage says "no bearing". *Recorded as an accepted verdict, not a
  checked one* — the same acceptance cost us 07/09, which turned out to hold D191.
- **Adopting lm-eval (§L1) mid-flight** — would make today's arms cross-protocol, which is
  the error D183 caught. Correct after the current batch, not during.
