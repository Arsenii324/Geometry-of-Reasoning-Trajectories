# Claim inventory — every strong result, with its presuppositions, experiment, and gates

*Written 2026-08-11 19:10. **Purpose: to be used directly while writing, so that no claim is
stated without its presupposition, no experiment goes unlinked, and no claim is stated weaker
than the evidence supports.** Each entry gives: the claim at its **strongest defensible** form,
the **safe** form if they differ, what must be **presupposed**, the **experiment** (kernel + job
id), the **gates that passed**, and **what would break it**. Dense by design. Numbers here are
authoritative; `RESULTS_HANDOFF.md` is the prose-facing subset of the same facts.*

---

# PART 0 — The methodological thesis: against visual 2D PCA

**This is a contribution in its own right and the paper currently under-states it. It should be
stated early, because every geometric claim we make depends on having replaced the field's
default instrument.**

**The default practice.** The model's own paper establishes its latent-dynamics claims by
**projecting 5280-dimensional trajectories onto the first two or six principal components and
judging the resulting pictures by eye.** It reports orbits, "sliders", and path independence
this way, and describes them as *"a few qualitative examples"*. No period is stated anywhere. No
null is computed. No threshold separates an orbit from a non-orbit.

**Why that is not sufficient, stated concretely rather than as a complaint:**

1. **A 2D projection of a 5280-d trajectory discards >99.9% of the variance directions.** Whether
   a loop appears in the projection depends on the basis, and the basis is fitted to the data
   being judged.
2. **Damping is confounded with rotation in a projection.** A monotone settle along a curved
   path renders as an arc; an arc read as a partial orbit is unfalsifiable by eye. *We hit this
   exact failure ourselves* — see Part 0's self-indictment below.
3. **No projection supports a negative.** "This prompt does not orbit" cannot be established by
   looking at a picture in which it does not obviously orbit.
4. **It cannot support a threshold, and thresholds are what claims need.** Without one, "rotating"
   is a judgement per figure, so no leave-one-out validation, no population statement, no
   cross-prompt comparison.

**What we substituted, and its validation chain** — this is the part to state, because a
replacement instrument is only worth as much as its own null:

| step | what | number |
|---|---|---|
| the statistic | `rotation_power(traj, period=6, tail=24)` — fraction of non-DC tail power in the period-6 rfft bin | — |
| it is a **scalar per orbit**, on the full 5280-d state, no projection | — | — |
| **spectral self-consistency** | recomputing the bin from the full banked spectrum reproduces the function | **4.0e-08** (tail 24), **5.2e-08** (tail 48) |
| **binary-label agreement** | one threshold reproduces the hand-assigned regime label | **691/696** orbits |
| **held-out validation** | leave-one-bank-out | **688/696** |
| **specificity** | rotating prompts peak *at* period 6, not merely somewhere | `symbol` **0.820**, `symptom` **0.962**, dominant bin exactly 6.00 |
| **it supports negatives** | their word-problem prompt has non-DC power **2.096** against **665–3139** for rotating prompts; two further free-form prompts **0.023**, **0.363** | — |

**The payoff, and why this belongs in the paper rather than in a methods appendix.** With a
scalar, thresholded, validated statistic we could do four things no projection allows, and
**three of them changed our conclusions**:

- **Population statements.** 62.0% of positions rotate in a rotating prompt; **0.000** in a
  settling one (D192). No overlap. A projection cannot produce that sentence.
- **A negative with force.** Their illustrative example does not orbit *at any period the tail
  resolves* (D200). Under 2D PCA this claim is unavailable in principle.
- **A timing decomposition.** The same statistic on two windows gives early separation **0.000**
  and tail separation **0.565** (D198) — the paper's hinge. A projection cannot be windowed
  without changing the basis and hence the picture.
- **Causal experiments with exact nulls.** `e`-swap and `wte`-swap runs gate on
  `|ΔR| = 0.000e+00` at the identity condition. A visual method has no identity check.

**The self-indictment that makes this credible, and it should be in the paper.** *We made the
projection-style error ourselves, in the same statistic, and caught it only because the
statistic had a gate.* A49's first pass computed the spectrum over a 48-unroll tail while
comparing against `rotation_power`'s 24-unroll default. The run **failed its own P1 consistency
check (0.1999 vs 0.7676)** and was voided. The cause is exactly the damping/rotation confound in
(2) above: over a 48-unroll window the decay transient dominates the low-frequency bins, so
`symbol` peaked at **period 48** — the slowest bin — rather than 6. **A longer window is worse,
not better, for a damped orbit.** With a picture, that error would have produced a plausible
figure and no alarm.

**How to phrase the claim so it is not a swipe at the prior work.** The phenomenon is theirs;
they saw it first and named it. What we contribute is (a) an instrument that makes it a
measurable property with a threshold and a null, (b) the finding that their *attribution* of it
to numerical-reasoning content is wrong, and (c) a negative on their own illustrative case that
their method could not have produced in either direction.

---

# PART 1 — The regime line

## C1. One instruction word switches the recurrent map between settling and damped period-6 rotation

- **Strongest form:** a single noun in the instruction, with computation, required answer,
  sequence content, marker and token count all held fixed, deterministically switches the
  recurrent map between a settling regime and a damped period-6 rotation.
- **Numbers:** *"report the largest **symbol**"* rotates **36/36**; *"largest **element**"* and
  *"largest **item**"* rotate **0/36**. Token count verified at **53** for all three. Paired
  within sequence: the word alone changes the regime in **36 of 36** (sequence, marker) cells.
- **Experiment:** A20, `scratch/ds_wordswap`, job `bt19t87hq0fmmget8p36`. 18 sequences × 3
  instructions × 2 markers.
- **Presupposes:** (i) the regime label is well-defined → C2's threshold; (ii) h₀ is seeded, or
  two forwards of one prompt cannot be compared (D78 — `initialize_state` draws from an unseeded
  RNG and no kernel seeded it before us); (iii) token count identity, which is *verified* not
  assumed.
- **Would break it:** a token-count confound (excluded by construction), or the regime label
  being an artefact of the statistic (excluded by C2).
- **Links:** C2 (threshold), C3 (what selects it), C4 (causal control), C6 (positional scope).

## C2. The regime is binary and a single threshold reproduces it

- **Claim:** a continuous rotation statistic separates the two populations, and **one** threshold
  R > **0.6677** reproduces the hand-assigned binary label on **691 of 696** orbits, and **688 of
  696** held out leave-one-bank-out.
- **Experiment:** D141, re-analysis over three banks (`ds_nounsweep`, `ds_embsep`,
  `ds_wordswap`), zero GPU.
- **Presupposes:** the label being reproduced was assigned independently of the statistic —
  true; it came from D132/D134's binary rotate/not calls.
- **Also records a retraction, and this is worth keeping:** D141 *replaced* D137/D138, which had
  argued "gapped distribution ⇒ two regimes" against a **uniform** null. Any clustered
  distribution beats a uniform null, so that test never tested the claim; checked against the
  labels the gap splits the *settling* population (40 settling + 1 rotating below it).
- **Would break it:** a bank where the threshold fails to separate. None found in 696 orbits.

## C3. What selects the regime is not meaning, form, or tokenisation

- **Claim:** the selecting property is idiosyncratic to the token and is predicted by none of the
  three obvious candidates.
- **Numbers:** `symbol`'s semantic neighbours (`character`, `digit`, `glyph`, `numeral`, `sign`,
  `token`) rotate **0 of 144**. `element`'s neighbours (`component`, `member`, `term`, `value`,
  `entry`) rotate **96 of 120**. **Form-matched** controls chosen to share form with `symbol`
  while denoting nothing like it (`symptom`, `cymbal`, `symmetry`) rotate **72 of 72**. Within
  the 53-token group the split is **exactly 50%**, so token count predicts nothing; subword count
  fails too (`cymbal` splits 3 ways and rotates, `glyph` splits 2 and does not).
- **Experiment:** A22, `scratch/ds_nounsweep`, job `bt1p04j521i2udj5civd`. 16 nouns × 12 shared
  sequences × 2 markers = 384 orbits.
- **Strength note:** the semantic result is an **inversion**, not a null — `symbol`'s synonyms
  uniformly do *not* rotate while `element`'s mostly do. State it that way; it is much stronger
  than "no semantic effect".
- **Would break it:** a fourth candidate predictor. The one that survived is the embedding row
  itself → C4.

## C4. The regime is causally controlled by one row of the embedding matrix

- **Strongest form:** editing a single row of `wte`, with the prompt string, every token id,
  sequence, length and seeded h₀ untouched, flips the regime.
- **Numbers:** both gates **bit-exact**: at t = 0 (row unchanged) R reproduces the unpatched
  orbit at **0.000e+00** on all 12 curves; at t = 1 (row *is* the target's) R reproduces the
  target's own orbit at **0.000e+00**. All four settling targets flip with exactly **one**
  crossing; `digit` and `letter` at t\* = **0.10**, `element` **0.15**, `token` **0.85–0.90**.
- **Experiment:** A32, `scratch/ds_wteswap`, job `bt1nerk2q8p5a8q39j4o`. 252 orbits.
- **Presupposes:** the noun is a single token (asserted in-kernel; run voids otherwise).
- **SAFE FORM — do not overstate:** *"the embedding row causally determines the regime within a
  small, direction-dependent neighbourhood."* **Not** "the embedding encodes the regime."
  Reason: A32's within-regime control **failed** — interpolating toward another *rotating* noun
  also left the regime (`array` 2 crossings, `block` 4). A36 (`ds_normchord`, job
  `bt146oseqilb4tbqbdpq`) then showed norm-matching **rescues `array`** (0 crossings, 21/21
  rotating, min R 0.768 vs the plain chord's 0.400) but **not `block`** (still 2 and 4 crossings
  at min 0.568/0.561), while settling targets still leave under matching (`digit` → 0.433,
  `element` → 0.068), so the matched arm is not merely pinning R high.
- **Consequence to state:** *one* chord of two exits the rotating set for reasons that are not
  norm. That is the same "real but not generic" footing D144 put the `e`-space version on.
- **Retraction attached:** D164's headline (chord-convex in `e`, not in `wte`, on a matched pair)
  is **withdrawn** — the decisive pair was normalisation (D167).

## C5. The map is memoryless in its parameter, and entering rotation costs ~15× leaving it

- **Claim:** swapping `e` at unroll k and leaving it swapped re-aims the dynamics at the new
  regime regardless of k — no hysteresis — and the two directions are strongly asymmetric.
- **Numbers:** **62 of 64** switches take, at k ∈ {0,1,2,4,8,16,32,48}; `set→rot` **32/32** at
  every k including 48, `rot→set` 30/32 with both misses at k = 48. Relaxation: **leaving**
  rotation median **1.0** unroll (range 0–2); **entering** median **15.5** (range 5–25). P1
  exact: all 8 k = 0 runs reproduce the donor's own orbit at **0.000e+00**.
- **Experiment:** A39, `scratch/ds_hyster`, job `bt1gltcn8nas02ilq32r`. 96 unrolls, both
  directions, 2 chords × 2 sequences.
- **The ρ connection — this is the only place a contraction rate predicts an observable, and it
  was registered before the run.** At ρ ≈ 0.83 a state closes 5% of the gap to a new attractor in
  ln(0.05)/ln(0.83) = **16.1** unrolls, 1% in **24.7**. Measured entering time median **15.5**,
  range 5–25.
- **SAFE FORM:** the agreement is **suggestive, not tight** — the relaxation detector slides a
  width-30 window and carries its own lag, so the measured value is an **upper bound**. True
  entering time ≤ 15.5, consistent with ρ in D115's 0.79–0.83 family range; it is *not* a
  confirmation of the 1%-closure figure. **The asymmetry is the robust part** (identical detector
  both directions), and P5 registered its direction in advance: *a periodic orbit is not a point,
  so entering one may cost more than leaving one.*
- **What it establishes about framing:** `e` is the map's **parameter**, `h` its **state** — and
  this is now a code-level fact, not an analogy: `block_idx` is threaded through
  `core_block_forward` and **never enters the block computation** (its only use is the KV-cache
  update), so the unrolled map is genuinely time-invariant (D187c).

## C6. The regime is a property of the prompt, not of the position measured

- **Claim:** measuring at *every* token position rather than only the last, rotating prompts
  rotate at **62.0%** of positions (range 0.614–0.623, n = 9) and settling prompts at **0.000**
  (all 9). **No overlap, no intermediate case.**
- **Structure found, not looked for:** rotation switches on at a **single boundary** and holds to
  the end — in `symbol`, positions **0–19 settle, 20–52 rotate**. The boundary is **not** the
  selecting noun (position 14) but where the second rule clause begins.
- **Experiment:** A46, `scratch/ds_posrot`, job `bt169rn8jvoi82aogsn8`. 21 prompts, all
  positions, 64 unrolls, 112 s.
- **Why it exists:** the prior work's figures are drawn at *interior* positions; every
  measurement of ours read the **last** prompt position. Had interior positions rotated inside
  settling prompts, D141's threshold would have been fitted on an unrepresentative population and
  **fourteen rows would have needed a scope line.** They did not. **State that the check was run
  and could have failed** — it is evidence about method.
- **Bounds C7 and the steering nulls:** because the onset is at position 20 of 53, a
  last-position intervention is **33 positions downstream** of where the property is set.

## C7. The regime and the answer occupy non-overlapping windows — the hinge

- **Strongest form:** on this architecture the answer is fixed roughly twelve unrolls before the
  regime becomes measurable at all, so behavioural nulls about the regime were testing a property
  that did not yet exist.
- **Numbers:** decision window (unrolls 0–11) rotating **0.2292** vs settling **0.2296**,
  separation **0.0004**. *(Recomputed 2026-08-11 from `transrot.json`: medians 0.229177 and
  0.229603. The earlier "0.229 vs 0.230, separation 0.000" is the same numbers rounded, but it
  reads as an arithmetic error — 0.230−0.229=0.001 — so the paper states four decimals.)* Tail: **0.862** vs **0.298**, separation **0.565**. Sliding 12-unroll
  window: identical through u12 (0.23/0.24/0.24/0.29 vs 0.23/0.24/0.25/0.29), then u16 **0.56 vs
  0.34**, u20 **0.74 vs 0.40**, u24 **0.84 vs 0.45**, saturating **0.96 vs 0.55**.
- **The floor cuts the right way and must be quoted:** early-window non-DC power **1.6e+05** vs
  tail **1943** (rotating) and **59** (settling). **The early window is ~100× more energetic and
  simply has no period-6 structure.** Without this, "0.229 vs 0.230" could be dismissed as two
  ratios of nothing.
- **Experiment:** A52, `scratch/ds_transrot`, job `bt1ori1l3q449kmbhgpd`. 48 prompts.
- **P1 gate:** `R_tail` reproduces the labels on all four nouns (`symbol` 0.767, `symptom` 0.958,
  `element` 0.077, `token` 0.519).
- **Presupposes:** the answer is settled early — D159 (median best-rank unroll **4** across 4868
  census draws; 17 of 21 families by unroll 8) and D184 (position-0 `best_depth` **4.07**).
- **SAFE FORM, mandatory:** *this is a timing claim, not a causal one.* It does **not** show the
  regime could not affect behaviour. It also **bounds the instrument**: `rotation_power` requires
  two cycles, so **12 unrolls is the earliest it can say anything** — no period-6 statistic can
  describe the state at the moment of decision. That is a property of measurement and model
  together.
- **Registered in advance:** written as `directions.md` §N3 **before** the run, both outcomes
  specified.
- **What it rescues:** D148 (0 of 48 — later found vacuous, see C13) and D176 (**1 of 18**,
  exact McNemar p = 1.0) are no longer evidence that the regime is a dynamical epiphenomenon.
  **The word "epiphenomenon" must not appear in the paper.**

## C8. Their attribution of orbiting to numerical-reasoning content is wrong

- **Claim:** the phenomenon is theirs; the stated cause is not supported.
- **Their claim, verified in-file** (`arxiv_february_2025.tex:182`): *"context-dependent
  behaviors emerge in latent space, such as ``orbiting'' when responding to prompts requiring
  numerical reasoning"*, illustrated at `:788` by a trivia question that converges *"without
  orbiting"*.
- **Three independent grounds:** (a) the **same nominal task** does both — `max_vs_min` rotates
  **32/32** in one bank and **0/22** in another, same task, different wording (D127d); (b) one
  word flips it with computation and answer held fixed (C1); (c) the selector is not semantic at
  all (C3).
- **Not confined to one template:** across **12 (bank, family) strata spanning 318 orbits, 6
  rotate in 100% of orbits and 6 in 0% — not one stratum mixed** (D127b).
- **And their own exemplar does not orbit:** sweeping every period the tail resolves, the
  word-problem prompt's median non-DC power is **2.096** vs **665–3139** for rotating prompts;
  trivia **0.023**, capital-of-France **0.363**. Experiment A49, `scratch/ds_periods`, job
  `bt13tdktmeldau9tb3p4`, both P1 gates passing at 4.0e-08 / 5.2e-08.
- **SAFE FORM:** *numerical-reasoning content is **not sufficient** to produce the orbit we
  measure.* We ran a prompt **in the shape of** their example, not their exact string; their
  figures are projections of a **128**-iteration run against our **64**.
- **Self-correction to record:** D192 first attributed our failure-to-reproduce to **our own
  detector** being period-6-blind. D200 shows that was wrong — there is no peak at any period
  because there is almost no non-DC power to distribute.

---

# PART 2 — The measurement line

## C9. First-token scoring reads 0.125 where the answer is produced in 0.781

- **Numbers:** first-token exact **0.125**, containment **0.781** (r = 32; 0.797 at r = 48) — a
  **6.3×** gap. **Chance rates measured, not assumed:** `echo_word` **0.000**, `echo_digit`
  **0.042**, `sort_min` **0.062**, `add1` **0.152**, against observed containments of
  1.000/0.750/0.625/0.781. Lifts **+0.562 to +1.000**.
- **Experiment:** A38, `scratch/ds_genscore`, job `bt13mu7dnv74kef3ogn5`. 128 generations, r = 32
  and 48.
- **P1 replication gate:** `echo_word`'s argmax is the gold's first token **16/16**; the other
  three families read first-token **0.000**, matching D158.
- **Presupposes:** containment is informative on these families — which is exactly what the
  chance-rate arm establishes. **Never quote a containment number without its chance rate.**
- **Rule this establishes:** of ten extraction rules over **1424** banked generations, the one
  this project used throughout (`exact`) recovers **0.038**; `last_number` recovers **0.240** at
  half the false-positive rate; `contains` **0.312** (D179). *And `last_number` is family-
  dependent: it mislabels 5 of 5 correct `compare` items, because "83 is larger than 23" puts the
  answer first.*

## C10. What displaces the answer is prose, and depth makes it worse

- **Numbers:** `The` opens **496 of 1260** banked generations (**39.4%**), then `To` 7.4%, `Text`,
  `There`, `Letter`, `Sequence`. Exact-match **0.020** vs containment **0.255** on that bank — a
  **12.75×** gap; only **8.3%** of generations begin with the gold.
- **The depth result, which is the sharp one:** rate of opening with `The` runs **0.032 → 0.099 →
  0.631 → 0.560 → 0.647** across r = 2/4/8/16/32 (**20×**; Fisher r=2 vs r=32 **89/252 → 206/252,
  p = 1.09e-12**), while the rate of **starting with the gold does not move**: 23/252 → 18/252,
  **p = 0.515**.
- **Experiments:** D170/D174, `scratch/kaggle_depthacc` (1260 generations, 21 families, 5 depths),
  zero GPU re-analysis.
- **Literal evidence, and it should appear in the paper as a table:** `sub1` gold `3` →
  *"The answer is 4 - 1 = 3"* → scored **wrong**. `sort_min` gold `18` → *"The smallest of these
  numbers is 18"* → **wrong**. `echo_digit` gold `4` → *"The number 4 is repeated exactly."* →
  **wrong**. `compare` gold `83` → *"83 is larger than 23."* → right.
- **The anti-correlation, stated plainly:** spending more compute makes the standard accuracy
  metric worse while making the underlying output better.
- **Mechanism for the `echo_word`/`echo_digit` dissociation:** the natural prose answer to
  *"repeat this word"* **begins with the word** (argmax is the gold's own first token 8/8:
  `cand`→candle, `g`→garden, `Sil`→silver); to *"repeat this number"* it begins with `The`.

## C11. The answer is never lost, and the collapse has two causes

- **Numbers:** restricted to the **10 census families with oracle ≥ 0.35** (the only ones where a
  final-unroll zero can mean anything), **2720 draws: the gold's final rank never exceeds 35, and
  0 of 2720 sit beyond rank 100** in a 65k vocabulary. `final_rank` is **uncapped** in that bank
  (it reaches **8971** elsewhere), so this is a real bound, not clipping.
- **The split:** copy/compare/select (1420 draws) median final rank **2.0**, top-5 **0.896**,
  worst 15. Arithmetic `add1`/`sub1`/`add_2d` (864 draws) median **16.0**, top-5 **0.000**, worst
  35.
- **Experiment:** D168, `scratch/kaggle_census` (4868 draws), zero GPU.
- **Interpretation, amended once — keep the amendment:** I first called arithmetic *"uncertain
  about the answer itself"*. A38 shows `add1` containment **0.781** with literal text *"One more
  than 1 is 2"* — gold correct, in the first clause. **The model computes in prose; the answer is
  not at position 1.** The rank-16 measurement stands; the reading was narrowed.
- **Consequence:** D157's 5.29× oracle-to-final inflation is **decomposed** rather than restated
  — formatting on one half, genuine rank displacement on the other. **No single inflation factor
  should ever be quoted** (2.06× on our tasks, 2.00× on ARC, 5.29× on the census).

## C12. The published benchmark number reproduces, by two protocols

- **Their protocol, pinned from their released code**, not inferred:
  `code/recurrent-pretraining/README.md:28` — *"All benchmark scores reported in the paper are
  computed using the lm-eval harness"*; `:31` gives the invocation with `--num_fewshot=0` and
  `mean_recurrence=32`; the paper's table caption (`tex:353`) says **zero-shot**, normalized
  accuracy; `:370` gives **ARC-E 69.91** at r = 32.
- **Route 1 — theirs:** bare prompt, no option list, option-text likelihood, character-normalised
  = **0.658**, off by **0.041**. A40, `scratch/ds_archarness`, job `bt1mspb9iukr5pbb3d74`, 120
  items, 4 arms.
- **Route 2 — independent:** five in-context examples under letter-argmax = **0.723**; paired
  over 101 items, **35 fixed against 4 broken, exact McNemar p = 3.35e-07**; 0-shot arm 0.416
  against D153's 0.407 (replication gate). A34, `scratch/ds_arcshots`, job
  `bt199d2ndbsrd91q0bcv`.
- **MANDATORY FRAMING:** *route 1 is the reproduction; route 2 is a coincidentally matching second
  route.* Closeness is not the criterion — protocol is. Getting this backwards is the single most
  likely writing error.
- **The partition, across 25 measurements** (`scripts/arc_table.py` regenerates it): zero-shot
  **with** the option list in the prompt **0.392–0.495** (13 arms); **without** it **0.592–0.658**
  (6 arms); few-shot letter-argmax **0.713–0.723** (2 arms). **Non-overlapping**, and the gap
  exceeds the spread within either group.
- **The mechanism, not just the number:** the letter arm gains **30.7** points from shots while
  the option-text arms gain ~**2** — a likelihood score never requires the model to *emit*
  anything.
- **Retraction attached:** D162 concluded *"protocol is refuted as the explanation"* after varying
  only the **scoring rule** at fixed prompt. Prompt format was the whole story and was never
  varied.
- **Known shared bias, must be disclosed:** likelihood scoring carries an uncorrected first-token
  preference; `acc_norm` divides by characters and corrects length, not that. **Our numbers and
  the published 0.699 share it**, so the *comparison* survives while the *absolute* accuracy of
  both is questionable in the same direction (D195).

## C13. Instruction fails, demonstration works — with one qualified exception

- **Instruction:** *"Reply with only the answer"* → parsed-exact **0.000**, identical to bare
  (and `kaggle_depthacc` found the same at every depth). Asking for `\boxed{}` produced **fewer**
  `\boxed{}` than not asking (**2/36 vs 4/36**) and moved the opener from `The` (27/36) to `To`
  (22/36) — i.e. *more* explanation. A43, `scratch/ds_fmtsweep`, job `bt10qd5t3soji8j3rjui`.
- **The system turn is not the missing ingredient:** the identical text in the system turn vs the
  user turn gives, paired over 24 items, parsed-exact net **+1**, containment net **−1**,
  first-token net **+1**, all **p = 1.0**. A50, `scratch/ds_sysprompt`, job
  `bt1qu10g915hnuatdj2m`. *This closes the gap that another group's system-prompt workaround
  opened.*
- **The exception:** *"work through it, then write `Answer:` on the last line"* → parsed-exact
  **0.167 vs 0.000**, gained 4 lost 0, **p = 0.125, n = 24**; literal output `'Answer: 1'`,
  `'The number 4 is repeated exactly.\n\nAnswer: 4'`. **Suggestive, not established.**
- **The interpretable pattern — state this, it is the point:** every *failing* instruction asks
  the model to **suppress** prose; the exception asks it to **append after** prose. It does not
  fight the frame. Consistent with C15 (the prose is where computation happens).
- **Demonstration:** two in-context examples move **oracle** accuracy **0.327 → 0.548** (paired
  41 gained / 4 lost, exact McNemar **p = 9.3e-09**) and final-unroll **0.071 → 0.327**
  (p = 7.0e-11); `echo_digit`/`add1`/`sub1` go **0.000 → 1.000** at the final unroll. A42,
  `scratch/ds_censusshots`, job `bt1a0btj3kp3m1huge8v`.
- **Registered-prediction failure worth reporting:** I predicted few-shot would fix *emission*
  only and leave oracle flat. **Oracle moved +0.221.** So few-shot improves what the model can
  do, not only what it will say — and *"the task is beyond the model"* was never established at
  zero shots (`succ_letter` goes **0.000 → 0.750** on the oracle axis).
- **And more is not better:** k = 5 is significantly **worse** than k = 2 on availability (net
  **−15**, p = 7.3e-04).

## C14. Context costs depth, not accuracy — and it is length, not exemplars

- **Design:** a zero-shot prompt padded with **irrelevant neutral prose** (rivers, bridges; no
  digits, no questions) to the 5-shot token length. **Length match measured, not assumed:**
  median ratio **1.000** (min 1.000, max 1.008; 131 vs 130 tokens).
- **Numbers:** padding collapses the gold's start rank **31 → 329** (paired Wilcoxon
  **p = 2.9e-09**) where five real exemplars give **58.5**. So padding reproduces **1084%** of the
  k=5 degradation — it overshoots tenfold. **At matched length neutral prose is 5.6× worse than
  exemplars** (329 vs 58.5, p = 1.9e-05). Oracle barely moves (0.500 → 0.479) while `best_depth`
  rises **4.73 → 7.29** (p = 0.0001). Consistent across **7 of 8** families.
- **Experiment:** A45, `scratch/ds_padlen`, job `bt1vouev933eh21iebus`.
- **Registered fork resolved against my own framing:** I posed "exemplars have negative returns"
  vs "the readout degrades with context length". **Length wins, and exemplars are the
  mitigation, not the cause.** *"Too many examples hurt" is the wrong lesson.*
- **It extends their claim:** their paper reports the model using more recurrence when more
  context is given, framed as extracting more information. We measure the same direction
  (**+2.56 unrolls**) for context carrying **no information at all**. So the mechanism is not
  extraction — **context costs recurrence whether or not there is anything in it to extract.**
- **Unbelieved side finding, flagged as such:** padding *improved* final-unroll accuracy
  (`echo_digit` 0.00 → 1.00, pooled 0.000 → 0.188). Unpredicted, uncontrolled, needs its own run.

## C15. The generated prose is computation, not decoration

- **Claim:** the forward pass that carries the answer is deeper than both position 0 and the
  surrounding prose positions.
- **Numbers:** answer position `best_depth` **7.89** vs position 0 **4.07** (Wilcoxon
  **p = 0.0031**) and vs other generated positions **3.65** (Mann-Whitney **p = 0.0004**).
  Per family: `sub1` 4.33 → **15.67**, `add1` 5.40 → 9.40, `sort_min` 4.60 → 7.40, against
  `compare` 4.33 → 4.67 and `echo_digit` 2.00 → 2.50.
- **Experiment:** A44, `scratch/ds_answerpos`, job `bt11guq5ipmvk7vv6t10`. 36 generations,
  every generated position, 48 unrolls.
- **Internal control — this is what rules out an artefact:** `echo_digit` needs no computation and
  gains **0.50** unrolls at its answer position; `sub1` gains **11.34**. A generic "later
  positions are deeper" effect would not respect that.
- **Registered fork:** DECORATION predicted the answer position would be *shallower* (the token
  already decided in the prompt's pass). It is **deeper by both comparisons**.
- **A result I nearly published and withdrew — worth one sentence in the reliability section:**
  re-locating the answer without the gold first gave a depth–correctness split of 9.13 vs 4.54,
  p = 0.0359. **Artefact of the locator**: `last_number` mislabels all 5 `compare` items the model
  got *right* (*"83 is larger than 23."*, gold 83). Under a locator-independent label: **8.04 vs
  5.50, p = 0.2901.** Direction preserved, significance gone.
- **Blind spot this closed:** every other geometry measurement in the project reads one forward
  pass at the last **prompt** position — i.e. the pass in which the model decides to write `The`.

## C16. Steering the map's parameter reproduces neither the benefit nor the geometry — one finding, not two

- **Novelty:** the **first steering-vector experiment on a recurrent-depth model.** Across 12
  surveyed papers there are **zero** occurrences of steering vectors, activation addition,
  representation engineering, SAEs, path patching, attribution patching, causal scrubbing or
  circuit discovery; activation patching appears twice and neither applies it to Huginn's
  recurrent state (D188).
- **Why the architecture suits it:** `e` is re-injected at **every** unroll, so a vector added to
  `e` applies **r times** — a dose–response knob no feedforward model has.
- **Behavioural arm:** `Δe = mean(e[last] | 2-shot) − mean(e[last] | 0-shot)`, α ∈ {0, ±0.5, 1, 2,
  4}. **P1 exact** (α = 0 adds zero; first token identical 25/25). First-token accuracy 0.000 →
  0.080; **norm-matched random control indistinguishable everywhere**; `The` opens 19/25 at α = 0
  and 18/25 at α = 4. **The dose was adequate** — oracle degrades 0.560 → 0.480 at large α, which
  is what makes the null readable rather than too-small. A48, `scratch/ds_steer`, job
  `bt1h7da466buqpgpsrik`.
- **Geometry arm:** same intervention with `rotation_power` banked, base prompts spanning **both**
  sides of the threshold (66 rotating / 66 settling, baseline R **0.077 to 0.968**): **largest
  |ΔR| anywhere 0.0065, zero threshold crossings in 132 conditions**, P1 exact at 0.000e+00. A51,
  `scratch/ds_steerregime`, job `bt13g7rgplheqpkqffi6`.
- **MANDATORY FRAMING:** these are **one intervention measured twice**, and the registered
  reading (written before A51 ran, `directions.md` §N1) is that the **site** is wrong, not that
  the directions are orthogonal — C6 puts the regime's onset **33 positions upstream** of where
  the vector was applied. **Do not write "formatting and regime are orthogonal directions."**
- **Named discriminator, unrun:** apply `Δe` at **all** positions.
- **Honest gap in my own design:** `‖Δ‖ = 2.1947` was banked but `‖e[last]‖` was not, so α is not
  expressible as a fraction of the parameter's scale.

---

# PART 3 — Hypotheses

## C17. H2 is architecturally untestable on this model, in the sense the literature means it

- **The literature's H2 is per-position adaptive depth**, and it is measured directly elsewhere:
  ponder time grows with difficulty when a halting cost is trained in (ACT); a universal
  transformer with dynamic halting reports mean per-symbol depth **2.3 ± 0.8 → 3.1 ± 1.1 → 3.8 ±
  2.2** as required supporting facts go 1 → 2 → 3, with the across-position histogram sharpening;
  PonderNet states the target as computation growing with problem complexity rather than input
  size.
- **Huginn has no halting head.** `num_steps = r` unrolls the whole sequence together; every
  position receives exactly `r`. **There is nothing to allocate.**
- **Therefore:** D177 (+0.23 unrolls for difficulty vs +5.56 for task identity), D143's null
  ladder, D115's 0.0033 — all measured the only quantity the model can vary, the **global** r.
  **Not evidence against H2 as a hypothesis about recurrent models.**
- **Three-way distinction to state explicitly:**
  1. *allocation by difficulty* — needs a trained halting objective; Huginn has none.
  2. *depth buying capability at fixed difficulty* — **real and published**: their GSM8K CoT
     **0.00 → 34.80** and ARC-E **34.89 → 69.91** from r = 1 → 32; we confirm independently
     (D171: depth puts the answer in the output on **5 of 14** families past Bonferroni, incl.
     `sub1` 1/12 → 12/12, `add_2d` 0/12 → 11/12).
  3. *difficulty recruiting depth within a task* — tested here, not found, and no mechanism by
     which it could be.
- **Supporting fact:** training depth was **not** fixed at 32 — it is `p ~ Poisson(rate)+1` with
  `rate ~ lognormal(log 32 − σ²/2, σ = 0.5)`; simulated mean 33.0, median 29, **p90 56, p99 93**.
  So **r = 48 is the 83rd percentile of the training distribution, not outside it** (D187b) —
  this **withdrew** a scope condition we had imposed on ourselves.
- **`parity8` is not an anomaly:** parity is the **S₂** word problem and is argued unsolvable by
  this architecture class, with harder state tracking above **S₅**. Depth making it *worse*
  (9/12 → 0/12, p = 0.0003) is expected, not noise.

## C18. H1 and H3, as they stand

- **H1 (settle/loop/drift) — answered, "both at different levels."** Drift is excluded **by
  construction**: every recorded state is an RMSNorm output on a sphere of radius **76.386**
  (per-orbit relative sd ~5e-05), and two random points on that sphere lie **108.03 ± 0.75**
  apart — which is the correct null for every distance claim we make. A large period-4 cycle
  exists **across** the four core blocks (vertex separation ≈52% of the state norm,
  separation-to-residual ratio median **232**, cycle perimeter **1.6×** the total distance
  travelled); **within** a block the state settles or rotates by instruction token. On rotating
  orbits the per-block last-step residual is **2.9152** vs **0.0185** for settling — a **157×**
  gap — while the cycle's vertex geometry is regime-invariant.
- **The sphere radius is not arbitrary:** `embed_scale` = **72.6636084983398** = exactly
  **√5280**, so our ‖h‖ = 76.386 implies a learned `ln_f` gain of **1.05123**.
- **H3 (contraction ⇒ no running register) — strong form overstates its evidence.** Contraction
  is real (ρ ∈ 0.79–0.87 by four instruments) **but ρ is family-level**, spanning **0.8239**
  (`rot13_word`) to **0.9181** (`count16`), with between-family spread **4.2×** the within-family
  spread — *do not quote ρ as a single model constant.* The limit set has diameter ≈1 in **5279**
  tangent dimensions, ample room to separate 64 states, and the count **is** linearly decodable
  from the latents on both of Barannikov's tasks (D32) — while the *winding* readout of it shows
  no quantisation, which is what the original instrument actually measured. Register is
  **architectural, not learned**: an untrained Huginn carries the lagged running count at
  cross-validated R² **0.7498** against the trained model's **0.7175** (D53).
- **H3's behavioural test remains unrun** — the h₀-injection experiment failed three times and was
  abandoned. Say nothing about it; the H3 position above is bounded and complete without it.

---

# PART 4 — Method as a result

## C19. Four claims were killed by controls we had registered in advance

*Ordered by how much they cost. This is the reliability argument; §B4 of `WRITING_PACK.md` has
the prose.*

1. **A behavioural null computed on a variable with no variance.** D148 reported *"0 of 48 units
   changed correctness"*. `correct` is False in **432 of 432** orbits — the gold never reaches
   rank 1 anywhere, best rank achieved **3**. There was no correctness for the regime to change.
   Re-run measuring the emitted text (D176, `scratch/ds_regimeout`, job `bt17rlmq1u51qsbui3li`):
   the model is right in **10/18 settling and 9/18 rotating**, and correctness changes in **1 of
   18** (p = 1.0). *The conclusion returned; the evidence for it is now an outcome that varies.*
2. **A geometry headline killed by its own norm control**, within hours (C4/D167).
3. **A spectral run refused by its own consistency check** (Part 0's self-indictment).
4. **A scoring comparison measuring the parser rather than the model.** D193's `parsed_exact` read
   0.000/0.000/0.056/0.028; re-scored on first-line containment the same 144 generations give
   **0.444/0.528/0.389/0.389** — the harness arm off by **14×**, because the model answers
   correctly on line 1 and then **invents its own follow-up questions and answers them**, and a
   fixed-order parser took the last `Answer:` marker. *Conclusion survived on the repaired
   metric: paired against bare, no arm wins.*

## C20. An instrument class was retired by a positive control

- **Claim:** a linear probe of the class this project used for several nulls **cannot recover a
  label that is a mathematically guaranteed deterministic function of the features it is given.**
- **Numbers:** the label is `rotation_power > 0.6677`, which integrates unrolls 40–63 — so
  features from that window determine it **exactly** and a perfect classifier provably exists.
  Given **R itself** as one feature, leave-one-noun-out scores **0.980**. Given the **raw states
  of the same window**, **0.675** at PCA-40 and **0.690** at PCA-100 and PCA-150, against a
  majority baseline of **0.600**. *Four times the components buys 0.015.*
- **Experiment:** `scripts/regime_onset.py` over `scratch/ds_embsep` (204 arrays, 51 nouns), zero
  GPU. Leave-one-**noun**-out, per-fold scaler and PCA fit on training nouns only.
- **Consequence:** every null of the form *"X is not linearly decodable from the state at n ≈ 50"*
  is uninformative. D135/D136 stay withdrawn and **no future run may re-establish them with a
  probe of this class.** This is the empirical version of D155's simulated floor (Cohen's d ≈ 10).
- **It also killed the question it was built for:** dating the regime's onset by probing is not
  possible; the causal version (C5, C7) replaced it.

## C21. Other method facts worth one line each

- **A degenerate outcome reads exactly like a clean null.** `scripts/outcome_variance_scan.py`
  flags outcome fields whose base rate is 0 or 1 across 38 banked result files. Two runs flag:
  one is the failure above, one (`ds_statetrack`, D147) is the row's actual **finding** — the
  oracle axis reads **1.000** on a model that answers `C` regardless of the question. *A flag is a
  question, not a verdict.*
- **Preflight checked everything except the model.** Two kernels died on a GPU for reasons a
  4.25 s local run catches: a fixed-length `e` concatenated onto a **growing** sequence, and a
  monkeypatch of an attribute that lives on the model rather than on the `ModuleDict`.
  `scripts/local_smoke.py` builds the **real** `RavenForCausalLM` at 27.7M params from the
  released source and reproduces both as regression checks.
- **A platform ERROR is not necessarily a crash.** One kernel's own gate could never pass
  (donor/recipient token counts 49 vs 40), so it dropped **18 of 18** items and returned 1 with
  **no traceback ever raised**. Diagnosis order: *did the kernel refuse on its own gate* →
  *undefined name* (`ruff --select F821`) → *only then* look for a traceback.
- **Exemplar leakage caught pre-launch by an assertion:** `echo_digit` has only **ten** distinct
  prompts, so a naive exemplar pool would have shown the model the exact test item with its
  answer. Exemplars are filtered against the test set; three families cannot supply five distinct
  non-test prompts and run at the k they can, with `k_actual` banked.

---

# PART 5 — Presupposition index

*If a claim below is used, the listed presupposition must hold and should be stated once,
early, rather than defended per-claim.*

| presupposition | needed by | established by |
|---|---|---|
| h₀ is seeded on every forward | C1, C4, C5, C7, and every paired comparison | D78; `torch.manual_seed(SEED)` before each forward in every kernel |
| the regime label is well-defined and binary | C1, C3, C4, C6, C7, C8 | C2 — 691/696, 688/696 held out |
| `rotation_power`'s window is the settled part, not the transient | C2, C7, C8 | Part 0 self-indictment; `tail=24` vs the failed `tail=48` |
| the state is norm-constrained, so distances have a fixed null | C18, any injected-state work | ‖h‖ = 76.386, sd ~5e-05; random-pair distance 108.03 ± 0.75 |
| `e` is the map's parameter and may be held fixed across unrolls; `h` is the state and may not | C4, C5, C16 | `block_idx` never enters the block computation (D187c); `ds_estream` documents the state-pinning failure |
| containment is informative on the families where it is used | C9, C10, C11, C13 | measured cross-item chance rates 0.000–0.152 |
| the census families are the same items across runs | C11, C13 | `items()` copied verbatim with `zlib.crc32` seeding (not `hash()`, which Python salts per process) |
| oracle accuracy ≥ 0.35 screens which families can speak at all | C11 | D168; families below it are capability failures and carry no formatting information |
| r = 48 is inside the training distribution | any claim at r = 48 | D187b — 83rd percentile of a Poisson-lognormal depth sampler |

---

# PART 6 — What must NOT be claimed

*Consolidated from every correction made on 2026-08-11. `WRITING_PACK.md` §C is the short
version; this is the complete one.*

1. **"The regime is a dynamical epiphenomenon."** Killed by C7. The word must not appear.
2. **"Few-shot reproduces their published number."** C12 — route 1 is theirs.
3. **Any single inflation factor.** C11.
4. **"Formatting and regime are orthogonal directions in `e`-space."** C16 — site untested.
5. **"The embedding encodes the regime."** C4 — neighbourhood-local only.
6. **"H2 is refuted."** C17 — architecturally untestable.
7. **"The regime is not linearly decodable."** C20 — the instrument failed, not the phenomenon.
8. **"We refuted their orbiting claim."** C8 — the phenomenon is theirs; the attribution fails,
   and only *sufficiency* of numerical content is refuted.
9. **"ρ = 0.83."** C18 — family-level, 0.8239–0.9181.
10. **"Reason-then-mark works."** C13 — p = 0.125 at n = 24, suggestive.
11. **A containment number without its chance rate.** C9.
12. **"Depth hurts accuracy."** C10 — depth raises prose-opening; answer-opening is flat.
13. **"The model cannot follow instructions"** without the `Answer:` exception. C13.
14. **Novelty for:** orbiting itself, path independence over h₀, depth-buys-capability,
    per-position adaptive depth, or the surface-form-bias correction. C8, C17, D190.
