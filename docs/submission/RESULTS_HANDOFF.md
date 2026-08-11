# Results handoff for the draft — numbers with their scope attached

**Read this instead of reconstructing from `claims_ledger.md`.** The ledger has 200 rows, many
amended or withdrawn, several by later rows on the same day. An agent reading it cold will
quote a number whose interpretation was retracted. Every figure below is current as of
**2026-08-11 18:00** and carries the scope it is licensed at. Row IDs are given so anything can be
checked, and `docs/UNDERSTANDING.md` §1.0-today is the same material in argument form.

**Do not add, sharpen or extrapolate. If a claim you want is not here, it is not established.**

---

## The one-sentence version

*The latent trajectory encodes the input — its shape and its endpoint alike — while difficulty
never reaches the dynamics, the dynamics never reach the readout **because the answer is fixed
before the dynamics acquire the structure we measure**, and much of what this project spent
months scoring was not the readout but the model's willingness to answer in the format we
scored.*

## Model and protocol

Huginn-3.5B (`tomg-group-umd/huginn-0125`), revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`,
inference only, float32. Architecture: `embed → prelude(2) → [core_block(4)]×r → coda(2) → ln_f
→ lm_head`, with the prelude output `e` re-injected at every unroll via
`adapter(cat[x, input_embeds])`. All measurements hook `core_block[-1]`. Depth `r` = 32 or 48
unless stated.

---

## A. The strongest results

**A1. One instruction word switches the recurrent map between settling and a damped period-6
rotation.** *"report the largest **symbol**"* rotates 36/36; *"largest **element**"* and
*"largest **item**"* rotate 0/36 — same sequences, same required answer, token count verified at
53 (**D132**). The selector is not semantic, not form, not tokenisation: `symbol`'s semantic
neighbours rotate **0/144**, `element`'s rotate **96/120**, form-matched controls (`symptom`,
`cymbal`, `symmetry`) rotate **72/72** (**D134**). A single threshold on a continuous rotation
statistic reproduces the binary label on **691/696** orbits and **688/696** held out
leave-one-bank-out (**D141**).

**A2. It is causally controlled from the token's embedding row.** Editing one row of `wte`, with
the prompt string and every token id untouched, flips the regime. Both gates bit-exact: t=0
reproduces the unpatched orbit at 0.000e+00 and t=1 reproduces the target's own orbit
(**D161**). *Scope:* the within-regime control failed there and was only partly resolved — a
norm-matched chord rescues one pair and not another (**D167**), so quote "the embedding row
causally determines the regime within a small, direction-dependent neighbourhood", not more.

**A3. The regime is a property of the prompt, not of the position sampled.** Measuring at every
token position: rotating prompts rotate at **62.0%** of positions (range 0.614–0.623, n=9),
settling prompts at **0.000** — no overlap. Rotation switches on at a single boundary and holds
to the end (**D192**). *This was checked because the paper's figures are drawn at interior
positions and ours at the last one; the record survived the check.*

**A4. No hysteresis, and ρ predicts an observable.** Swapping `e` mid-trajectory: **62 of 64**
switches take, at switch points from unroll 1 to 48 — the regime follows the current parameter,
not the history. Entering rotation takes median **15.5** unrolls, leaving it **1.0** — a 15×
asymmetry on the identical detector. The entering time brackets a prediction registered *before*
the run from ρ ≈ 0.83: ln(0.05)/ln(0.83) = **16.1** (**D173**). *Scope:* the detector's window
adds lag, so the measured value is an upper bound; the asymmetry is the robust part.

**A5. Much of the capability record measured format compliance.** First-token exact-match reads
**0.125** where the answer is actually produced in **0.781** of generations, against a *measured*
cross-item chance rate of 0.042–0.152 (**D172**). What displaces the gold is `The` — it opens
**39.4%** of 1260 banked generations (**D170**), and depth *raises* prose-opening 20× (0.032 →
0.647) while leaving "starts with the gold" flat (p = 0.515) (**D174**). On tasks the model can
do, the gold's final rank never exceeds **35** in 2720 draws (**D168**).

**A6. Two independent routes reproduce the published ARC-Easy number.** Published **0.699** at
r = 32. The paper's own protocol — pinned from their code as `--num_fewshot=0` through lm-eval
with `acc_norm` — is our bare-prompt, no-option-list, character-normalised arm: **0.658**
(**D175**, **D183**). Independently, five in-context examples under letter-argmax give **0.723**,
paired, 35 items fixed against 4 broken, exact McNemar **p = 3.35e-07** (**D169**). *Use D175 as
"the" reproduction; D169 is a second route, not their protocol.*

**A7. Instruction fails, demonstration works — and the system turn does not rescue it.** *"Reply with only the answer"* gives parsed-exact
**0.000**, identical to bare; asking for `\boxed{}` produced *fewer* `\boxed{}` than not asking
(2/36 vs 4/36) (**D193**, amended — its arm-level numbers were parser artefacts; on a repaired
metric no arm beats bare, B_only +3 net at p = 0.375). Two in-context examples move oracle
accuracy **+0.221** (exact McNemar p = 9.3e-09) and take `echo_digit`/`add1`/`sub1` from 0.000 to
**1.000** at the final unroll (**D180**).

*Placing the identical instruction in the **system** turn rather than the user turn — the form
another group used to solve this exact problem — buys **one item in 24** (paired, p = 1.0), so
the failure is not about the turn (**D199**). The one qualified exception is an instruction that
does not fight the prose frame but appends to it: "work through it, then write `Answer:` on the
last line" gives parsed-exact **0.167 against 0.000**, gained 4 lost 0, **p = 0.125** — literal
output `'Answer: 1'`. Suggestive at n = 24, not established, and consistent with the prose being
where the computation happens (A9).*

**A8. Context costs depth, not accuracy — and it is length, not exemplars.** Padding a bare
prompt with *irrelevant* prose to the 5-shot token length (match verified, median ratio 1.000)
collapses the gold's start rank **31 → 329** (paired p = 2.9e-09), where five real exemplars give
58.5. At matched length neutral prose is **5.6×** worse than exemplars. Oracle accuracy barely
moves (0.500 → 0.479) while `best_depth` rises **4.73 → 7.29** (p = 0.0001) (**D186**).

**A9. The prose is computation, not decoration.** The forward pass carrying the answer is deeper
than position 0 (4.07 → **7.89**, p = 0.0031) *and* than the surrounding prose positions (7.89 vs
3.65, **p = 0.0004**), concentrated on arithmetic (`sub1` 4.33 → **15.67**) while `echo_digit`
barely moves (2.00 → 2.50) (**D184**).

**A10. Steering the map's parameter does not reproduce what in-context examples do — and the
two nulls are one finding.** The first steering-vector experiment on a recurrent-depth model
(no such work exists across the 12 papers surveyed). `Δe = mean(e[last] | 2-shot) −
mean(e[last] | 0-shot)`, added to `e` at every unroll, swept over α ∈ {0, ±0.5, 1, 2, 4}.
**P1 exact** (α=0 adds zero; first token identical 25/25). Behaviourally null: first-token
accuracy 0.000 → 0.080, with a **norm-matched random control indistinguishable everywhere**;
`The` opens 19/25 at α=0 and 18/25 at α=4. The dose *was* adequate — oracle degrades 0.560 →
0.480 at large α, which is what makes the null readable rather than too-small (**D196**). The
companion run banks `rotation_power` on base prompts spanning **both** sides of the threshold
(66 rotating / 66 settling, baseline R from 0.077 to 0.968): **R moves by at most 0.0065, zero
threshold crossings in 132 conditions**, P1 exact at 0.000e+00 (**D197**).

*Scope, registered before the runs: D192 found rotation switches on at position 20 of 53, so a
last-position intervention sits 33 positions downstream of where the property is established.
These rows license "steering the last position does nothing", **not** "formatting and regime are
orthogonal directions". Since the same intervention moved neither behaviour nor geometry, the
economical reading is the wrong site, not two independent axes. The discriminator — apply Δe at
all positions — is unrun.*

**A11. The regime and the answer occupy different, non-overlapping windows of the same
trajectory — which reframes the behavioural nulls rather than retracting them.** Measuring
`rotation_power` over the **decision window** (unrolls 0–11, the earliest a period-6 orbit is
definable at all, since the statistic requires two cycles) and over the tail, on the same
orbits: rotating prompts read **0.229** early and settling prompts **0.230** — **separation
0.000**. Over the tail the same prompts read **0.862** vs **0.298** — separation **0.565**. A
sliding 12-unroll window dates the onset sharply: the two populations are *identical* through
unroll 12, then diverge at u16 (0.56 vs 0.34), u20 (0.74 vs 0.40), u24 (0.84 vs 0.45),
saturating at 0.96 vs 0.55 (**D198**).

*The early window is not quiet — its non-DC power is **1.6e+05** against tail power of 1943 and
59, roughly a hundredfold more energetic. It simply has no period-6 structure.*

**Why this matters for A1–A4.** The answer is best-ranked at median unroll ~4 (D159; position-0
`best_depth` 4.07, D184), and at unroll 4 the two regimes are indistinguishable. So the earlier
finding that a causally-induced regime change alters almost nothing behaviourally (1 of 18
paired units, p = 1.0, **D176**) is **not** evidence that the regime is a dynamical
epiphenomenon: it manipulated a property that had not yet come into existence when the readout
was settled. **This is a timing claim, not a causal one** — it does not show the regime *could
not* affect behaviour, only that on this architecture the answer is fixed ~12 unrolls before the
regime becomes measurable. It also bounds the instrument: **no period-6 statistic can describe
the state at the moment of decision**, and that should be stated wherever the regime is
connected to behaviour.

---

## B. The hypotheses, as they now stand

- **H1 (settle / loop / drift)** — answered, "both at different levels". Drift is excluded by
  construction (RMSNorm sphere, radius 76.386). A large period-4 cycle exists *across* the four
  core blocks; *within* a block the state settles or rotates by instruction token (**D98**,
  **D146**, **D99**).
- **H2 (harder problems recruit more depth)** — **architecturally untestable on this model.**
  The literature's H2 is *per-position adaptive depth*: Universal Transformers measure it
  directly and get 2.3 ± 0.8 → 3.1 ± 1.1 → **3.8 ± 2.2** ponder time as supporting facts go
  1 → 2 → 3. **Huginn has no halting head — `num_steps=r` unrolls the whole sequence together,
  so every position gets `r`.** There is nothing to allocate (**D191**). Depth *does* buy
  capability at fixed difficulty (their GSM8K CoT 0.00 → 34.80 from r=1 → 32, **D190**; our
  **D171**), and difficulty *within* a task moves recruited depth by +0.23 unrolls against task
  identity's +5.56 (**D177**).
- **H3 (contraction ⇒ no running register)** — its strong form overstates its evidence.
  Contraction is real (ρ ∈ 0.79–0.87 by four instruments) but the limit set has diameter ≈1 in
  5279 tangent dimensions, and the count is linearly decodable from the latents (**D32**).

---

## C. What must be said about method — this is a strength here, not an admission

The project retracted or amended claims when controls came back against them, and the report
should say so plainly; the review criteria name "reliability of results" explicitly.

- **D148's behavioural null was vacuous** — `correct` was False in **432 of 432** orbits, so
  there was no variance for the regime to change. Re-tested on an outcome that varies, the
  conclusion returned (1 of 18 units flip, p = 1.0) (**D176**).
- **D164's headline was withdrawn by its own registered control** within hours (**D167**).
- **A linear probe of the class used for several nulls cannot recover a label that is a
  guaranteed deterministic function of its own features** — 0.690 against a 0.600 baseline where
  the correct 1-D statistic gets 0.980 (**D165**). Those nulls are uninformative, not negative.
- **The extraction rule dominates the number**: over 1424 banked generations, `exact` recovers
  0.038 and `last_number` 0.240 at half the false-positive rate (**D179**).

## D. Known limits to state, not hide

- **Surface-form bias**: our ARC numbers and the published 0.699 share an uncorrected
  first-token preference in likelihood scoring (**D195**).
- **We do not reproduce the paper's orbit on its own word-problem example, and it is not our
  detector.** 0/35 positions at period 6 (**D192**); sweeping *every* period the tail resolves
  finds no peak anywhere, because there is almost nothing to peak — median non-DC power **2.096**
  against **665–3139** for rotating prompts, with their trivia and capital-of-France prompts at
  **0.023** and **0.363** (**D200**). The detector is validated to **4e-08** against its own
  definition, and our rotating prompts peak at period 6 *specifically* (0.820, 0.962). *Honest
  scope: a prompt in the shape of their example, not their exact string; their figures are PCA
  projections of a 128-iteration run against our 64. The claim is that numerical-reasoning
  content is **not sufficient** to produce the orbit we measure.*
- **The paper attributes orbiting to "prompts requiring numerical reasoning"; our data
  contradicts that attribution** on three grounds while crediting them the phenomenon
  (**D185**).
- **Cross-checkpoint / cross-protocol traps**: paper 02 evaluates an SWA-merged checkpoint, not
  ours; paper 01 computes `s*` at 128 iterations where we use 64 and post-`ln_f` where we read
  pre-`ln_f` (**D190**).
