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
| `geometry-h0bank` | Kaggle T4 | **the powered, PRE-REGISTERED test of D91's lead** — 16 boundary prompts x 32 unseeded h_0 draws, within-ITEM correctness decode with the window grid fixed in advance | **COMPLETE 2026-08-09.** Output pulling now; `scripts/run_h0_within.py` to be run exactly as pre-registered, unedited, once the data lands |
| `geometry-clrs` | Kaggle T4 | capability screen on CLRS-Text, which is IN Huginn's training mixture and has a clean integer difficulty knob — the task gap the supervisor named | RUNNING |
| `geom-eigenplane` | DataSphere g1.1, job `bt1mvf3juvrcjpl6kl1i` | does the orbit rotate at the rate the Jacobian predicts? | **DONE** -- no: 0.24x, and D74 explains why |
| `geom-bank` | DataSphere g1.1, job `bt1hd3oqb17690amgolg` | B14 -- bank raw states for BOTH arms; the untrained control D74 needs | **DONE -> D76.** Training flips the step cosine -0.379 -> +0.541, disjoint at every window |
| `geom-seeds` | DataSphere g1.1 | closes D76's one-draw limit | **DONE.** Five draws agree (cos -0.339..-0.389); trained/untrained completely disjoint, p=1.05e-12 |
| `geometry-b6-bank` | Kaggle T4 | **B6 re-run** | **DONE -> D79.** Bounded null: no geometric difference at matched answer, powered to 1.5 sd, in the 1 family of 4 that the design could test |
| `geometry-patch` | Kaggle T4, queued (built + CPU-verified, blocked on a slot — Kaggle caps at 2) | **first non-void causal test.** Patch the answer-position state at unroll r from a correct h_0-draw into an incorrect h_0-draw of the SAME prompt (D90's mechanism); sweep r over {0,8,16,24,32,48}; does the flip rate concentrate in D91's window? | BUILT, `scratch/kaggle_patch/`, 8/8 CPU dry-run tests pass. Push with `kaggle kernels push -p scratch/kaggle_patch` |
| `qk-alignment-probe-smoke` | **Yandex DataSphere**, project `bt12q57tmrs03pnt8drc` (personal account, ~80k RUB remaining, valid to end of year — confirmed by the user 2026-08-09, NOT the expired smiles2026 grant) | the supervisor's own open item (`project_plan.md` G2, 0% done): per-unroll query-key cosine alignment at the answer position, on the SAME paired track/local length-matched design as H2/D83 | **First submission (`bt1jfkkdf71ms695e6id`) SELF-CHECK FAILED, correctly, and the run halted before any experimental number — the instrument-null discipline (§5) working as designed.** `max\|q_hat-q_true\|=14.96, max\|k_hat-k_true\|=13.82`. Root cause found and fixed same session: `verify_extraction`'s ground-truth spy captured the FIRST `scaled_dot_product_attention` call in the whole forward and assumed it was `core_block[-1]`'s — but a real forward runs prelude (2 attn blocks) -> core (4) -> coda (2), so the first call is the first PRELUDE block's, an unrelated layer's weights entirely. Fixed by bracketing the spy with forward-pre/post hooks on `attn` itself so it only arms while `core_block[-1]`'s own forward is on the stack. Reproduced the bug AND confirmed the fix locally with a tiny structural fake (prelude+core+coda, real `CausalSelfAttention`/`apply_rotary_emb_complex_like`, no download): old logic gives q_err=1.59/k_err=1.76, fixed logic gives exact 0.0. Resubmitted as job `bt164frpr04jvqguht5j` 2026-08-09 ~12:27 MSK. Poll with `GRPC_DNS_RESOLVER=native datasphere project job get --id bt164frpr04jvqguht5j` (the `native` DNS resolver env var is REQUIRED on this Mac — the default c-ares resolver fails against its VPN/Tailscale-assigned DNS server even though the OS resolver and `curl` work fine; also note `job get` takes `--id` alone, no `-p`) |

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
| B4.1 | Is ρ constant across task families? | **open** — D45's claim was retracted; n=3/family had no power |
| B4.2 | Recompute D46's arm comparison with the corrected window | **done** → D53 |
| B4.3 | Untrained control for D35 | **dropped** → D54(3): degenerate by construction, the untrained model has no answer process |
| B4.4 | Why are 8/108 trained orbit fits below the R²>0.9 bar while 0/60 untrained are? | **open** — a second, unexplained way the trained operator differs (D52(3)) |
| B4.5 | What happens to ρ *before* step 6144? | **parked** — no published checkpoint exists; would need training from scratch |
| B4.7 | **Measure accuracy on the 8 task families that never had it** (parity, running-max, projection, three-scale, three-scale-modk, running-count, nesting-depth, count-ones). One generation pass each. | **open — the single highest-value gap** (D56(2)); decides whether 8 tasks' worth of geometry describes reasoning or failure |
| B4.8 | **Give persistent homology a null, or drop it.** 15 trajectories, H₁ 0–26, no surrogate ever computed. | **open** (D56(3)) — the last untested topological claim, and the most distinctive thing the project set out to look at |
| B4.9 | **Sweep the Jacobian spectrum properly**, n=3 → tens of prompts, both arms. Magnitudes gave ρ (D31), arguments gave rotation (D55). | **open** — best value per unit cost in the project |
| B4.15 | **H2's FIRST REAL TEST: rotation rate vs difficulty on the Jacobian eigenvalue argument, n ≫ 3.** D55 has 3 points (n_ops 8/32/64 → \|arg\| 2.397/2.388/1.050) on the right instrument, never analysed as H2. Needs ~30 prompts × 4 difficulty levels, both arms. | **open — the single most valuable unrun experiment** (D65(5)); one kernel |
| B4.16 | **H1 re-operationalised on a working instrument.** The regimes were retired on winding, which fails its own null. Do settle/loop/drift differ in ρ, in \|arg\|, or in whether the orbit converges vs limit-cycles? | **open** (D65(1)); reuses the ρ pipeline, one kernel |
| B4.17 | **H3's consequence, never tested.** ρ<1 is established four ways; the implication for counting capacity was never turned into a measurement. | **open** (D65(3)) |
| B4.13 | **ρ is fitted over 65–98 unrolls; Huginn trains at mean r=33.** All 12/12 prompts have ρ characterised beyond 2× the mean training depth, so the project's central quantity describes a regime the model was never trained for. Whether ρ(r≤32) = ρ(r≤98) is untested and bears on D42/D44/D52/D59. | **open** — requires a rerun, because of B4.14 |
| B4.14 | **Kernels persist the FIT, not the CURVE.** `eps_sweep.json` stores fitted ρ and n_used but not the per-unroll separation curves, so B4.13 cannot be answered without re-running the GPU job. Rule: persist the curve you fitted. Costs a few MB; not doing it costs a rerun per unanticipated question, and this project generates about one such question per experiment. | **rule adopted; existing kernels not retrofitted** |
| B4.11 | **DEPTH WAS NEVER VARIED AGAINST ACCURACY.** Audited 2026-08-05: every accuracy kernel in this project fixed `num_steps=32`. The kernels that sweep depth all score geometry. On a recurrent-depth model whose authors report GSM8K rising 9-10% at r=4 to 28-38% at r=32, the single most obvious experiment for the architecture had not been run. | **in progress** — `geometry-prompt-depth` sweeps r ∈ {4,8,16,32,64} on the best of five phrasings |
| B4.12 | **Huginn's own inference modes were never used.** `generate_with_adaptive_compute`: 0 kernels. `continuous_compute` (warm-starting the latent across tokens — the "continuous CoT" mode that is the architecture's distinctive feature): 0 kernels. | **partly in progress** — `continuous_compute` is now an arm in `geometry-prompt-depth`; adaptive compute still unused |
| B4.10 | **Re-operationalise or retire H1's regime labels.** As they stand they partition without separating (Kruskal–Wallis p=0.48). | **open** (D56(1)) |
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

