# State of understanding — 2026-08-11

Rewritten, not appended (CLAUDE.md §6). §1 below is the current synthesis; the
2026-08-09 text it replaces is kept from §1-old onward because the difference is
itself evidence. Every figure cites the ledger row carrying its controls and limits.

---

## 1. The synthesis

**One sentence.** *The latent trajectory encodes the input — its shape and its endpoint
alike — while difficulty never reaches the dynamics, the dynamics never reach the readout, the
readout is finished long before the dynamics are, and what we spent months scoring was not the
readout but the model's willingness to answer in the format we scored.*

**The second clause is new as of 2026-08-11 and it is the day's main result.** Huginn answers
in prose sentences. `The` opens **39.4%** of 1260 banked generations; exact-match reads 0.020
where the answer is actually present in 0.255 of them; and the literal text is unambiguous —
`sub1` gold `3` produces *"The answer is 4 - 1 = 3"*, scored **wrong** (D170). Supplying the
format recovers the benchmark: ARC-Easy goes **0.416 → 0.723** on five in-context examples
against a published 0.699, while option-likelihood scoring — which never asks the model to emit
anything — moves ~2 points over the same shots (D169). And on tasks the model can do, the gold
never leaves the top 35 of a 65k vocabulary at any depth (D168).

*This does not say Huginn is accurate.* `last_item` confidently answers *"The last number in
the list is 4"* when the gold is `2`, and five families contain the gold in 0.000 of their
generations. It says our accuracy numbers measured format compliance, and that the two must be
separated before any of them can be read as capability.

### 1.0-today. What changed on 2026-08-11, and what it cost

*A verification surface: 24 new rows (D164–D187) and 7 amendments, in one place. The
through-line is a single sentence — **for most of this project we were measuring format
compliance and calling it capability** — and it cuts both ways.*

**Added after the papers arrived (D182–D187).** Reading Huginn's own source and config settled
four things our own runs could not, and two of them go against us:

| row | what it did |
|---|---|
| **D183** | **pins their ARC protocol from their code** — `--num_fewshot=0`, lm-eval harness, acc_norm, r = 32 → ARC-E **69.91**. That is our `H_bare_chr` **0.658** arm, *not* D169's 5-shot 0.723. Quoting the five-shot route as "we reproduced their number" would misdescribe the comparison. |
| **D182** | across **25** ARC measurements the option-list partition is clean and non-overlapping — zero-shot with options **0.392–0.495**, without **0.592–0.658** |
| **D184** | **the prose is computation.** The answer-carrying forward pass is deeper than position 0 (4.07 → **7.89**, p = 0.0031) *and* than the surrounding prose (7.89 vs 3.65, **p = 0.0004**), concentrated on arithmetic (`sub1` 4.33 → **15.67**) |
| **D185** | their attribution of orbiting to *"prompts requiring numerical reasoning"* is **contradicted** — same task rotates 32/32 in one wording and 0/22 in another; one noun flips it at fixed token count |
| **D186** | **it is length, not exemplars.** Irrelevant padding collapses the start rank 31 → **329** (p = 2.9e-09) where five exemplars give 58.5 — at *matched* length. Context costs **depth** (4.73 → 7.29, p = 0.0001), not accuracy |
| **D187** | **§6.9 withdrawn in our favour** — training depth is Poisson-lognormal, so r = 48 is the **83rd percentile**, not "outside the training regime". And `block_idx` never enters the block computation, so the DEQ framing is a code-level fact |

**Two things I would not have found without the papers, and one is a hole in our strongest
line.** Their orbit figures are drawn at **interior token positions**; every measurement in this
project reads the **last** prompt position only. If interior tokens rotate inside prompts whose
last position settles, D141's threshold was fitted on an unrepresentative population and
fourteen rows need a scope line. **A46 is testing it now.** They also name a third structure,
**"sliders"**, which we have never looked for.

*Two of my own registered predictions were refuted today (D180's P2, D186's fork) and one
result I nearly published was an artefact of my own locator (D184(3)). Those are in the rows.*

**Retracted or narrowed (5).** These are the expensive half and they come first.

| row | what fell |
|---|---|
| **D148** | its correctness null was **vacuous** — `correct` is False in **432 of 432** orbits, so there was nothing for the regime to change. *(Then re-tested and restored by D176 on an outcome that varies: the model is right ~50% of the time there when you read what it writes.)* |
| **D162** | *"protocol is refuted"* is **wrong**. It varied the scoring rule at fixed prompt; **prompt format was the whole story** and was never varied. |
| **D164** | headline **withdrawn** by its own registered control — the decisive `symbol`↔`array` pair was normalisation, not curvature (D167). |
| **D168(2)** | *"uncertain about the answer itself"* narrowed by D172 — `add1` writes *"One more than 1 is 2"*, correct, in the first clause. |
| **D177(4)** | *"depth tracks the kind of operation"* **superseded** by D178 — it tracks where the answer starts (family-level Spearman **+0.885**). |

**Established (the load-bearing four).**

1. **The answer is never lost, and the metric was the problem.** On the 10 census families the
   model can do, the gold's final rank never exceeds **35** in **2720** draws, none beyond 100
   in a 65k vocabulary (D168). It is displaced by a prose opener — `The` takes 39.4% of 1260
   generations (D170) — and first-token scoring reads **0.125** where the answer is produced in
   **0.781**, against a *measured* chance rate of 0.042–0.152 (D172).
2. **Two independent routes now reproduce a published number**, the first in 169 rows: five
   in-context examples at **0.723** (D169) and the harness's own bare prompt at zero shots at
   **0.658** (D175), against a published **0.699**.
3. **ρ predicts an observable, closing §6.2.** Swapping `e` mid-trajectory: **62 of 64** switches
   take, so the regime follows the current parameter with no hysteresis, and entering rotation
   costs **15×** what leaving it does (median 15.5 vs 1.0 unrolls) against a pre-registered
   prediction of 16.1 (D173).
4. **Depth reaches behaviour, and it is not H2.** More unrolls put the answer in the output on
   **5 of 14** families past Bonferroni (D171) — but depth also raises prose-framing **20×**
   while leaving starts-with-gold flat (D174), difficulty moves recruited depth **+0.23 unrolls
   against task identity's +5.56** on a format-independent measure (D177), and demonstration
   moves the ladder from floor to **ceiling** without ever opening it (D180).

**Two instruments were themselves refuted (D165, D179)**, which is why the above is quotable: a
linear probe of the class this project used cannot recover a label that is a *guaranteed*
deterministic function of its own features (**0.690** against a 0.600 baseline, where the correct
1-D statistic gets **0.980**); and of ten extraction rules over 1424 banked generations, the one
used throughout scores **0.038** where `last_number` scores 0.240 at half the false-positive rate.

**Still open, and registered rather than assumed:** whether the 5-shot regression is exemplars
or context length (A45), whether any format *instruction* works at zero shots (A43), and whether
Huginn's prose is computation or decoration (A44).

---

### 1.0 Where the three starting hypotheses stand

D65 returned H1, H2 and H3 all to **open** on the correct grounds — each had been retired
using winding, an instrument that fails its own null (window-governed, D28; sampled near
the aliasing limit, D55), and *"the measurement failed" is not "the phenomenon is
absent."* Enough has been measured since to give each a verdict. Stated here because
they are otherwise scattered across fifty rows.

**H1 (settle / loop / drift) — ANSWERED, and the answer is "both, at different levels".**
*Drift was never available:* every recorded state is an RMSNorm output on a sphere of
radius 76.386, so unbounded drift is excluded by construction rather than by measurement
(D23, D99). *The loop exists and is large:* hooking all four core blocks instead of one,
each converges to **its own** fixed point, the four separated by **22–58** where states
have norm 70–76 — about **52% of the state norm** — with a separation-to-residual ratio of
median **232** and a cycle perimeter **1.6×** the entire distance the state travels
(D98). *And "each block converges" holds only for two thirds of prompts:* on rotating
orbits the per-block last-step residual is **2.9152** against **0.0185** for settling ones,
a **157×** gap, while the cycle's vertex geometry is regime-invariant (D146). So: **loops
across blocks always; within a block, settles or rotates depending on one instruction
token.** The hypothesis was not wrong — the instrument that retired it read one block.

**H2 (harder problems recruit more depth) — STILL NOT ESTABLISHED, BUT ITS PRECONDITION IS
NO LONGER ABSENT.** *(Revised 2026-08-11.)* Every H2 test below reads a *rate* or a *rank*.
Read instead as **what the model writes**, depth does reach behaviour: over 1260 banked
generations at r = 2…32, the answer's appearance in the output rises with depth on **5 of 14
families after Bonferroni** — `sub1` **1/12 → 12/12**, `add_2d` **0/12 → 11/12**, `local_last`
**3/12 → 12/12**, `add1` **1/12 → 9/12** — with `parity8` moving strongly the other way
(**9/12 → 0/12**). Output length grows only up to r = 8; **above it length is flat and
containment still rises, 61/252 → 91/252, p = 0.0048**, so this is not the length confound
(D171). The gains land exactly on the arithmetic families D168 identified as *uncertain about
the answer* rather than *blocked on format*, and at r = 32 `sub1` writes out *"The answer is
4 - 1 = 3"* — it shows the working. **Depth buys computation where computation is what is
missing.** That is not H2, which claims *difficulty recruits depth*; it is the precondition H2
needs, and the record had been treating it as absent.

**And the reason H2 cannot be tested has changed, which is worth stating precisely.** Re-running
the census's own items at 0/2/5 shots: two examples raise **availability**, not just emission —
pooled oracle **0.327 → 0.548**, paired gains 41 against 4 losses, exact McNemar **p = 9.3e-09**
(final-unroll accuracy moves 0.071 → 0.327, p = 7.0e-11). So *"the task is beyond the model"*
was never established at zero shots (D180). **`succ_letter` goes 0.000 → 0.750 on the oracle
axis.** But the ladder still cannot be walked, for the opposite reason it could not before:
`add1` **0.750 → 1.000** and `add_2d` **0.250 → 1.000** at k = 2 — *both at ceiling*. The
difficulty axis is squeezed between floor and ceiling, and demonstration moves it from one to
the other rather than opening it. Five families stay at **0.000 at every k** — `last_item`,
`count8`, `max_run`, `caesar1_word`, `rot13_word` — so the count ladder's middle rung is still
missing. *(And more is not better: k = 5 is significantly worse than k = 2 on availability, net
−15, p = 7.3e-04.)*

**Everything below is therefore scoped to the zero-shot condition, including the H2 test that
follows.**

**The formatting discovery does NOT rescue H2, and that is measured rather than
assumed.** The obvious worry after D174 — that every H2 null was an artefact of a metric
anti-correlated with competence — is testable on a measure that has nothing to do with emission:
**`best_depth` conditioned on the model being right**, i.e. when the answer becomes *available*.
On 4868 census draws, difficulty moves it in H2's predicted direction and by almost nothing:
1-digit → 2-digit addition gives **+0.23 unrolls (d = 0.297, p = 0.0048)** — the project's first
significant difficulty-on-depth effect — against matched-difficulty controls at **+0.16** and
**+0.05**, and against **task identity at +5.56 unrolls (d = 3.18)**. Between-family means span
**2.20 to 8.29** with a within-family sd of 1.08 (Kruskal–Wallis p = 2e-237). That is 0.5% of a
48-unroll budget (D177).

This replicates D115's shape on an unrelated quantity: **the map's behaviour is set by which
task it is, not by how hard the instance is.**

**And the large between-family effect is not recruitment either.** It looked like computational
demand — retrieval 2.2–2.9 unrolls, arithmetic 4.5–4.9, character transformation 8.3. It is
mostly **where the answer starts**: family-level Spearman between the gold's rank at unroll 1
and mean `best_depth` is **+0.885** (item-level +0.528, p = 1.6e-146). The matched pair makes it
concrete — `echo_word` and `echo_digit` are the same instruction at an identical 21-token
prompt, and the word's gold sits at median rank **862** at unroll 1 against the digit's **15**,
which is why it takes 6.15 unrolls to arrive against 2.73 (d = 2.87, p = 3.2e-83). Multi-token
golds are excluded as the cause (6.15 vs 6.08 within `echo_word`, p = 0.25). **The recurrence
climbs from wherever the prior puts the answer; it does not allocate itself by need** (D178).

So the one large depth effect in the census is a property of the *initial readout* — the input
embedding and the untrained prior — rather than of the computation the task requires.

So the evidence below that the *rate*- and *rank*-based tests are null is now corroborated by a
format-independent test rather than being merely a statement about instruments.

**The original entry, unchanged:** Where it could be tested it is null, with bounds: difficulty moves the contraction
rate by **0.0033** when the count quadruples (D115, D116); the dynamics do not reach the
readout (ρ = +0.118, p = 0.65, D114); the readout is finished at median unroll **4** with
87% of the journey ahead (D112); the first structurally clean ladder gives ρ = **−0.0286,
p = 0.7986** once the trivial level is removed, bounded at |ρ| ≥ 0.216 (D143). *(The
"causally induced regime change alters 0 of 48 answers" that used to sit here is **withdrawn**:
`correct` is False in 432 of 432 of that run's orbits, so there was no correctness to
change — see §1.3.)* **But the decisive fact
is D147**: the one difficulty axis where theory says recurrence is *required* — sequential
state tracking — is out of Huginn's reach entirely, since it answers with a constant. Every
axis we could build is either shortcuttable in principle or beyond the model. **That is a
limit of Huginn, not of the hypothesis**, and no amount of further GPU on this model
changes it.

**H3 (contraction ⇒ no running register) — ITS STRONG FORM OVERSTATES ITS OWN EVIDENCE.**
Contraction is real (ρ in 0.79–0.87 by four measurements, §1.3). But the limit set has
**diameter ≈1 in 5279 tangent dimensions**, ample room to separate 64 states, so
contraction excludes an **unbounded** register and says nothing about a bounded counter.
And a bounded register is not merely possible but *observed*: the count is linearly
decodable from the latents on both of Barannikov's tasks (D32) — while the *winding*
readout of it shows no quantisation, which is what the original instrument actually
measured. A further scope condition: the register is **architectural, not learned** — an
untrained Huginn carries the lagged running count at cross-validated R² = **0.7498**
against the trained model's **0.7175** (D53).

---

### 1.0a The one external calibration now works, and it explains the rest

Huginn's paper reports **0.699** on ARC-Easy at r = 32. Zero-shot this project measured 0.407
by letter-argmax and 0.433/0.413 by option-text likelihood, and D162 concluded the gap was not
the scoring rule. It was not. **It was elicitation:** letter-argmax runs **0.416 → 0.713
(2-shot) → 0.723 (5-shot)**, clearing the published figure at two shots, paired over 101 items
with 35 fixed against 4 broken, exact McNemar **p = 3.35e-07** (D169).

The gain is **protocol-specific and that is the point**: over the same shots the option-text
arms move ~2 points while the letter arm moves **30.7**. A likelihood score never asks the
model to emit anything; the letter arm asks for a bare option letter at the first generated
position, and zero-shot Huginn opens with prose there (D166). **The answer was available and
could not get out.** §1.0b below is the same phenomenon measured from the inside.

### 1.0b The answer lives in the transient, and most of our instruments read the fixed point

This is the structural fact that reframes the rest, and it was found in banked data at no
GPU cost.

Across all 21 census families — 4868 draws, 48 unrolls — **the median unroll at which the
gold token reaches its best rank is 4**, and **17 of 21 families reach it by unroll 8**.
After that the rank does not decay toward noise; it **settles at a stable worse value and
holds it for forty-plus unrolls**. `add1` is rank 1 at unroll 4 and rank **19** from ~16 to
47. `echo_digit` is rank 1 at unrolls 1–5 and rank **4** thereafter. Median final/best rank
ratio **1.6×**, worst **19×** (D159).

**So the map converges to a state whose readout does not prefer the gold.** The gold's
rank-1 moment is a *transient* feature. Two families are the exception and they are the
only ones that keep their answer: `echo_word` holds rank 1 for **43 of 48** unrolls, and
`compare` holds rank 2.

Three consequences, all uncomfortable and all measured:

1. **Scored at the final unroll, 19 of 21 families read exactly 0.000** — including
   `echo_digit`, the easiest task in the suite, at **0.997 → 0.000** (D158). *That number
   stands and its original reading — "Huginn cannot hold a copied digit" — does not.* It
   holds the digit at rank 2–4; what outranks it is **`The`** (D166, below).
2. **The observability census certifies availability, not production.** Its 41 items in the
   20–80% band become **0** on the final-unroll axis, at an inflation of **5.29×** — larger
   than D103's 2.06× and A29's 2.00×, so the factor is a property of task and budget and
   **no single "inflation factor" should be quoted** (D157).
3. **Most of this project's geometry measures the converged state** — contraction rates,
   fixed-point similarity, the rotation regime, h\* decoding. **The state that carries the
   answer is the one at unroll ~4, and it is not the state those instruments read.**
   *But the tempting corollary — that transient geometry therefore predicts correctness
   where the fixed point does not — is **checked and not supported**. D93's powered window
   sweep already covers the argmin (start 0, width 12) and finds no trend across eight
   windows: Spearman(start, accuracy) = +0.14 (p = 0.75) for shape, −0.31 (p = 0.46) for
   position, with the argmin window itself at chance. Where the answer is **available** and
   where the geometry **predicts success** are different questions (D163).*

**And the displacer is now named, which changes what the whole axis means.** Every kernel
here banked the gold's rank and nothing else, so "what wins instead" was open. A35 banks the
top-5 tokens and log-probabilities at every unroll. Templated (n = 32): the gold's best rank
is median **1.0 at unroll 3.5**, its **final** rank is median **2.5**, and it is still in the
top 5 for **24 of 32** items. The final argmax is a **prose sentence-opener** — `The` ×12,
`One` ×3, `Number` ×2, `To` ×2 — with `The` taking `sort_min` **8 of 8** at logprob −0.079
(~92%) while the gold sits at rank 2. **D145's role-marker hypothesis is refuted: it is not
`user`, it is `The`** (D166).

Three things follow, and they matter more than the axis itself:

- **The final-unroll axis measures answer *formatting*.** It scores whether the model opens
  with the bare gold token at the first generated position; Huginn opens a sentence. The
  answer is one token behind, not gone.
- **D158's dissociation has a mechanism.** `echo_word` holds because the natural prose answer
  to *"repeat this word"* **begins with the word** — its argmax is the gold's own first token
  in 8 of 8 (`cand`→candle, `g`→garden, `Sil`→silver). The natural prose answer to *"repeat
  this number"* begins with `The`.
- **The chat template is load-bearing.** Removing it does not remove the displacer; it makes
  the answer disappear. Bare, the gold's final rank is median **10** with a maximum of **509**,
  and `\n` wins **23 of 32** as the model continues a document instead of answering.

**At census scale that halves.** Re-scoring the 4868 banked draws, restricted to the **10
families with oracle accuracy ≥ 0.35** — the only ones where a final-unroll zero can mean
anything, since the rest are capability failures — gives **2720 draws in which the gold's final
rank never exceeds 35, and 0 of 2720 sit beyond rank 100**, in a 65k vocabulary. `final_rank`
is uncapped in that bank (it reaches **8971** across all rows), so that is a real bound. **On
families the model can do, the answer is still there at the end of the computation, always.**

But the pool splits sharply, and only one half is formatting (D168):

| group | draws | median final rank | top-5 | worst |
|---|---|---|---|---|
| copy / compare / select | 1420 | **2.0** | 0.896 | 15 |
| arithmetic (`add1`, `sub1`, `add_2d`) | 864 | **16.0** | 0.000 | 35 |

A prose opener costs one or two rank positions — which is exactly the copy group. It cannot
explain arithmetic at median 16. A35's literal top-5 for `add1` shows what does:
`['One', 'The', 'To', '1']` — prose openers **and the input digit**, with the gold `2` outside
the top five. **For arithmetic the model is uncertain about the answer itself, and that is a
capability statement, not a formatting one.**

*(A prediction of mine failed here and is recorded as failing: I expected survival to track
word-gold against number-gold. It does not — 1 of 6 word families clear 0.05, 1 of 15 number
families do, and the five failing word families all have oracle ≤ 0.222, so they are capability
failures carrying no information about formatting.)*

So the sharpest version of §1.0b is this: **the recurrence reaches the answer at unroll ~3–4,
never subsequently loses it, and spends the remaining ~44 unrolls committing to a prose frame —
on the tasks it can do. On arithmetic it also never resolves which answer it means.** And this
decomposes D157's 5.29× inflation rather than merely restating it: formatting on one half,
genuine uncertainty on the other, which is why one factor should never be quoted across both.

### 1.1 What the latent carries

The trajectory's **shape** identifies the input at ceiling, separating two prompts one
token apart whether or not that token changes the computation (D84, D87, D88), and
carries at most a weak, unreplicated trace of whether the computation succeeded
(D92 positive, D93 not-confirmed under a floor that detects a planted 0.5 sd effect
100% of the time).

**The same is now true of the endpoint, which is the harder case.** D109 found the
fixed point h\* decoding the specific answer while the shape sits at chance — but in
that design the answer is a *function* of the input, so both readings predict it. A
paired-marker experiment separates them: sharing the required **answer** moves h\*
similarity by **+0.00016 (p = 0.70)**; sharing the **input** moves the same statistic
on the same orbits by **+0.0034**. The answer effect is **4.3%** of the input effect
(D125). The h\*-versus-shape dissociation stands; its interpretation narrows to
*h\* encodes the input, from which the answer is recoverable.*

And the topological signal behaves the same way: real, well-nulled, and outcome-blind
(D123 positive, D126 null at matched answer value).

### 1.2 Why H2 returns nothing — a chain, not a list

1. **Difficulty does not reach the dynamics.** The contraction rate spans 0.824–0.918
   across 21 families (between/within 4.2×, surviving length matching at 7.5×), yet
   quadrupling the count in `count4→count8→count16` moves it **0.0033** (D115). The
   rotation angle behaves identically, independently of the modulus, with an untrained
   control (D116). Both are **rates, not durations**, so D80's clock-reading critique —
   which explains away every earlier H2 instrument — cannot apply.
2. **The dynamics do not reach the readout.** ρ(contraction rate, best_depth) =
   **+0.118, p = 0.65**, against a floor detecting effects three times smaller than the
   variation present (D114).
3. **A causally manipulated regime change produces no behavioural difference.** In
   D132's paired design — same sequence, same marker, same required answer, only the
   instruction's noun differing — the rotating arm and its settling counterparts differ
   in neither the depth at which the answer is best available (16.0 vs 14.5, p = 0.156)
   nor the best rank reached (6.5 vs 7.0, p = 0.880). This is the strongest of the three
   because it manipulates rather than correlates.
4. **Because the readout never waits.** On solved orbits the answer is top-ranked at
   **median unroll 4**, with **87% of the state's journey still ahead**; the state
   reaches a 1% residual only at unroll 32–36 (D112, replicated across banks; threshold
   dependence measured in D97's correction).

Every H2 instrument this project built measured the trajectory. The answer was already
decided in the first few unrolls.

### 1.3 The map itself is well characterised, and it is a task-level object

Four instruments of three kinds put the contraction rate in **0.79–0.87**: causal
state-injection **0.8335 ± 0.028** over 16 recipients (D113), full-operator Arnoldi
**0.8098** (D119), diagonal-block Arnoldi 0.79–0.81 (D31), passive orbit decay ≈0.82
(D94 — whose +0.033 bias correction toward ~0.82 was **withdrawn**: it fitted 3 of 12 real
prompts and the 0.058 gap to Arnoldi remains unexplained, D58). Training moves it from
**0.7048 ± 0.0087 → 0.8577 ± 0.0139** across 14 weight-sets, with *total* separation
(U = 0, p = 5.0e-04, d = 13.2), and is done by the earliest public checkpoint (D52; the
0.7042 → 0.8582 figures previously quoted here are D120's, and D120(1) is retracted as a
rediscovery of D52).

**These are not four estimates of one number.** Each instrument ran on a different prompt
set and ρ is family-level: across 21 families it spans **0.8239** (`rot13_word`) to
**0.9181** (`count16`), with between-family spread **4.2×** the within-family spread
(D115). The defensible statement is that the map contracts on every prompt measured, at a
rate the *task* sets.

**And there is a loop — across the four core blocks, not within one.** Every instrument
in this project reads `core_block[-1]`, and at that block the state contracts. Hooking all
four instead: each block converges tightly to **its own** fixed point (per-block residual
~61–64 → **0.052–0.402**, a 159×–1185× reduction, median 384×), and those four fixed points
are separated by **22–58** in a space where states have norm 70–76 — about **52% of the
state norm**. The ratio of separation to residual is **95–730, median 232**, and the cycle's
perimeter is **1.6× the entire distance the state travels** from initialisation to
convergence (D98). **The loop exists, it is large, and single-block reading was blind to it
by construction.** This is the scope on the contraction rates above: they describe the
iteration-to-iteration map *at a fixed block*, and every absolute "no loop anywhere"
statement this project once made is retired.

**And a single word selects the qualitative regime.** *"largest **symbol**"* rotates
36/36; *"largest **element**"* and *"largest **item**"* rotate 0/36 — same task, same
sequences, **identical 53-token count**, flipping in **36 of 36** within-sequence cells
(D132). Symbol diversity is refuted as the cause (D129). This has a mechanism already
in the ledger: `e` is the map's **parameter** (D111, D113), and the spectrum at h\*(e)
is task-dependent (D115, D116, D119) — D132 localises that dependence to one token.

**And the three nouns request the same computation and the same answer.** `symbol`,
`element` and `item` all mean *take the maximum of this sequence*, and the banked golds
are identical across all three in **36 of 36** cells. Computation fixed, answer fixed,
token count fixed, sequences shared — **and the dynamics change completely.** This is
D88 one level down: D88 found the trajectory's *shape* reads the input token rather than
the computation; D132 finds the same of the *dynamics*. **The latent geometry tracks the
surface form of the instruction, not what the instruction asks for** — which is the
sharpest statement of §1.1 available, and it is causal rather than correlational.

**What selects the regime is still unknown, and one round of guessing has been spent.**
Meaning, surface form and subword structure are all refuted (D134). D134 then inferred
*by elimination* that the property must live in the token's learned embedding; A23 tested
that survivor on 51 nouns and it **fails** — leave-one-out from the embedding is 0.551
against a permuted null of 0.530, below the 0.592 majority baseline, and the separating
direction is not low-dimensional (D135). Six classifiers, linear and nonlinear, then give
a **best-of-six of 0.551 against a null-of-max of 0.589, p = 0.808** (D136).

**Those two rows do NOT refute the embedding hypothesis, and reading them as if they did
was an error of mine.** Planting a linear class separation of known strength into the same
49 embeddings shows the design cannot resolve anything below **Cohen's d ≈ 10** — where
0.8 is conventionally a *large* effect — and the observed result is indistinguishable from
planted separations of d = 0.25, 0.5 and 1.0, all of which return the identical 0.551
(D155). **D135 and D136 are uninformative, not refutations.** Combined with D149 removing
the hyperplane picture that would have explained a linear-in-`e` structure being invisible
in the embedding, the honest state is: **we do not know where the property lives, and the
one attempt to rule out the embedding had no power to.** A32 asks it causally instead —
interpolating the token's `wte` row directly — because manipulation is what worked for
`e` when fitting failed.

**The regime is bimodal, and D132's determinism survives at larger scale.** A continuous
rotation statistic R separates the two populations (median **0.896** vs **0.336**,
p = 0.0002) with no marker or sequence confound. Of the two nouns that failed D135's
determinism gate, `word` is a **detector false positive** — all four of its orbits sit at
R = 0.26–0.40, well below the decision threshold, so the period detector's single
"rotating" call on it is spurious — and only `set` is genuinely intermediate, its four
orbits at 0.559, 0.560, 0.570 and **0.737** straddling the threshold, with the one above
it being exactly the one the detector also called rotating. The two instruments agree on
`set` and disagree only on `word`. **50 of 51 nouns are deterministic** (D137, sharpened
by D141) — which retires D135's own reading that the regime depends on the token *and*
the sequence.

**And a single threshold on that statistic reproduces the regime label across three
banks.** R > 0.6677 agrees with the independently computed period-6 label on **384/384**
(`nounsweep`, 16 nouns × 12 sequences), **108/108** (`wordswap`), and **199/204**
(`embsep`) — **691 of 696 orbits**, across banks with different nouns, sequence counts
and templates (D141). *An earlier version of this paragraph claimed a distributional
**gap** instead; that test used a uniform null the data beats for reasons unrelated to
the hypothesis, and the gap it found sits inside the settling population in two of the
three banks. It is retracted and replaced by the threshold result, which is what the
claim needed in the first place.*

**And its behavioural consequence is now *partly* bounded and partly untested — this
paragraph previously overstated it.** A26's regression discontinuity compares the two
parameter values **0.05 apart in `e`** on opposite sides of the boundary: same tokens, same
required answer, regime flipped, in 48 of 48 units with exactly one crossing. `best_depth`
is null, with a floor that catches a planted 0.5 sd (4.7 unrolls) 99.3% of the time and
0.25 sd never, so effects below ~4.7 unrolls remain untested. The only signal is a
**one-rank shift** in 16 of 48 pairs, 13 in the same direction, exact p = 0.0213 — which
does *not* survive correction over the three measures tested, and is one position out of a
median rank of 8 (D148).

**But the correctness half is vacuous and is withdrawn.** `correct` is `False` in **432 of
432** orbits: the gold never reaches rank 1 anywhere in that run, in either regime, at any
grid point, and the best rank any orbit achieves is **3**. So "correctness never changes,
0 of 48" describes a variable that is identically zero on both sides — *there was no
correctness for the regime to change*. The task is "report the largest/smallest {noun}" with
golds 6/7/8/9, and Huginn does not do it. **This is the check §1 of CLAUDE.md exists for and
it was not run.** D166 supplies a likely mechanism for the floor: prose openers hold the top
ranks, and a best rank of 3 is what two permanently-ahead format tokens would produce.

**Re-tested the right way, the conclusion comes back — and D148's zero was the metric,
entirely.** A41 reruns the same boundary comparison measuring the **emitted text**. On that
very task the model is right in **10 of 18** units settling and **9 of 18** rotating — it
writes *"The largest token of the sequence is 8."* with gold **8** — where D148's rank-based
scorer recorded 0 of 432. With a real base rate to move, crossing the boundary changes
correctness in **1 of 18** paired units (exact McNemar p = 1.0000), and the first token is
identical in **18 of 18**. The generated text does differ in 10 of 18, but lexically — *"digit
**of** the sequence"* → *"digit **in** the sequence"*, and *"the largest **letter**"* → *"the
largest **signal**"*, that last being the interpolated instruction noun leaking into the
model's own echo of it, which varies continuously with t and is not a regime effect (D176).

So **"dynamical epiphenomenon" is supported again — but now by a test that could have detected
a change.** The bound is weak: n = 18 paired units excludes only large effects, far weaker than
D148's `best_depth` arm, which caught a planted 0.5 sd at 99.3%. *Surface form moves with the
parameter; the answer does not.*

**The regime is causally controlled by `e`, and the boundary is a surface rather than a
split.** Interpolating `e` along straight lines between nouns, all eight cross-regime
paths cross the threshold **exactly once**, every crossing inside a single 0.05 grid
step — sharper than the design resolves — and at t = 1 the orbit matches the rotating
noun's own R to within the h₀ noise even though h₀ came from the settling prompt, which
supports D111/D113's parameter claim causally. A24 could not verify its own patch
fidelity — its check demanded 1e-6 agreement between two forwards while h₀ is drawn from
an unseeded RNG (D78) — but **A25 seeded h₀ and the same check now returns exactly
0.000e+00 on all 8 chords**, so the hook writes `e` and only `e`, retrospectively
licensing A24's numbers (D144).

**And direction, not distance, is what moves the regime — which is the finding that makes
this geometric rather than anecdotal.** A25's off-distribution control travels the *same
euclidean distance* from a carrier's `e` in a random bearing. **None of 8 random bearings
rotates** (R = 0.054 … 0.579, all below threshold), and on a rotating carrier a random
bearing of that magnitude **destroys the rotation in 3 of 3 cases** — 0.845 → 0.579,
0.977 → 0.271, 0.974 → 0.131 — while the chord toward another *rotating* noun keeps all
21 of 21 points inside the regime. So the rotating set is a structured region of
parameter space, not a neighbourhood that any large perturbation escapes.

**Which set is the non-convex one — corrected here.** This paragraph previously asserted
that *the rotating set* is not convex and then cited evidence that it is; the label was on
the wrong set, and the banked counts settle it. In `e`-space, **within-rotating chords leave
the regime 0 of 3** (all 21 grid points rotating, min R 0.768 / 0.846 / 0.885), while
**within-settling chords pass through rotation 2 of 5**, rising to R = 0.74 before falling
to 0.05, with t\* varying 0.28–0.74 by pair. So in `e` it is the **settling** set whose
chords escape, and 2 of 5 is below A25's pre-registered bar of 3 of 6 — demonstrated, and
not to be read as typical (D140, D144).

**Redrawn in `wte`-space, within-rotating chords leave the regime 2 of 2** — `symbol`→`array`
dipping to R = 0.400 with two threshold crossings, `symbol`→`block` to 0.546 with four, on both
sequences (D164). That looked like a coordinate effect, on a matched pair appearing in both
runs. **It was mostly norm, and the control says so.**

A36 reran those chords rescaled to the source row's norm at every grid point, holding
everything else. It replicates the plain arm bit-for-bit, and the confound turns out to be
large: the embedding rows are nearly orthogonal (cos **+0.039 to +0.096**), so linear
interpolation drops the interpolant to **69.5–75.5%** of the source norm at its minimum.

- **On `symbol`→`array` — the pair that made D164 decisive — the excursion vanishes:** 0
  crossings, **21 of 21** points rotating, min R **0.768** where the plain chord gave 0.400.
  So *that* contrast between `e` and `wte` was normalisation, and **D164's headline is
  withdrawn.**
- **On `symbol`→`block` it does not vanish:** still 2 and 4 crossings at min R 0.568/0.561. So
  **at least one straight line between two rotating embeddings exits the rotating set for
  reasons that are not norm.**
- **The control holds:** settling targets still leave under norm-matching (`digit` landing at
  0.433, `element` at 0.068), so the matched arm is not merely pinning R high (D167).

**What this leaves is narrower and should be quoted narrowly:** non-convexity in `wte` is
demonstrated on **one chord of two**, which is the same "real but not generic" footing D144 put
the `e`-space version on. What distinguishes `array` from `block` is unmeasured — `block` has
the larger norm (0.874 against the source's 0.797 and `array`'s 0.742), so the target's norm
relative to the source's is the first thing to check.

**The region is bounded and anisotropic, and it is NOT a half-space.** Bisecting from two
rotating carriers along 64 rays: **81.2%** of random directions escape the regime, and of
32 antipodal pairs **both** sides cross in 20, one side in 12, neither in 0 — where a
hyperplane would predict ~0 pairs crossing on both. Crossing distances vary by a factor of
**49** (0.053–2.572 chord units). So the rotating set is extended along the directions
connecting rotating tokens and thin in random ones (D149). **This kills the tidiest
explanation of D135/D136**: had the boundary been a hyperplane in `e`, those embedding
nulls would have been predicted, since `e` is the prelude's nonlinear image of the
embedding. It is not, so they remain unexplained.

That pointed at `e` itself — the prelude's nonlinear image of the embedding, and the
map's parameter (D111, D113). **A24 tested it causally rather than by fitting**, since at
51 nouns power rather than the hypothesis was what bound; its result is the next
paragraph.

### 1.4 What the corrections did to the older record

An independent audit on 2026-08-10 found that **three older positives all rest on one
instrument**: `classify_shape`, whose loop branch D25(3) proved fires on straight lines,
random walks and decaying spirals alike. B4 ("the project's cleanest positive result" —
loops under a starved budget) is retracted: its settle test is the last step over the
*largest* step, which truncation inflates by construction. D14's claim to validate the
pipeline against Blayney's rate is withdrawn — a detector with an unknown false-positive
rate cannot be validated by matching a published number. D100's "working difficulty
ladder" is withdrawn (3 forwards per level; only 0/33/67/100 observable).

Two more dissolved rather than resolved: D97 vs D30 is a **threshold**, not a
disagreement (11→47 unrolls for 30%→0.1%); D100 vs D101 is two underpowered estimates.
And D55's near-aliasing diagnosis is withdrawn — its samples-per-turn came from the
diagonal-block operator D31(3) calls "the wrong operator"; the trajectory is sampled
3–9× above Nyquist, not near it. **D55's third prompt, at 60.1°, was already reading the
mode D112/D132 later characterised.**

### 1.5 The strongest objection to all of it

Capability. The census yields **31 usable items across 5 families** (D130), none of them
natural-language reasoning; multi-term addition is at 0% (D118); binary-search depth runs
*opposite* to Huginn's difficulty (D128); CLRS was never tested in the form its
deep-research pass specified (D124). So every null here is a null *on synthetic,
largely-OOD tasks*, and the D124 ambiguity — genuine OOD difficulty versus scoring
artefact — is unresolved. The one experiment that would most sharpen the picture,
D125's answer-versus-input test, is limited by exactly this: at 4.7% accuracy on one
marker it cannot separate "no answer encoded" from "no answer computed."

---


## 1-old. The 2026-08-08 synthesis, kept unedited

### The one-paragraph version (as of 2026-08-08)

The project asked whether latent-trajectory *geometry* in Huginn-3.5B encodes
reasoning depth. It does not, in the form originally proposed. The 2026-08-05
revision then held that **training changes the model's dynamics, not the content
of its state** — ρ rising 0.7048 → 0.8577 across 14 weight-sets with complete
separation (D52), while every content measurement was matched or beaten by random
weights (D40, D41, D48, D53). **That second half is now in serious doubt, and the
reason is that the readout was the confound.** Every content-null result was
measured on tasks scored 0% by greedy generation plus exact match. Scored instead
by rank of the gold token over the full vocabulary, the trained model holds the
answer *far* below chance on those same tasks — `count16` at median rank 22
against a chance of 32768 — and the trained-minus-untrained gap is **ordered by
capability**: +4.05, +3.43, +2.96, +2.81, +1.37 in log₁₀ rank from `echo_digit`
down to `rot13_word` (D68). Where the model can do the task, its weights carry
four orders of magnitude more answer-information than random ones. That is a
content difference, and a large one.

## 1b. What changed on 2026-08-08, and what it costs

Three findings, in the order they landed.

- **The instrument was wrong, not the model** (D68). Teacher-forced rank per
  unroll, from one forward per item — an instrument `eval_depth.py` already
  contained and that had been used for D35 and never for capability.
- **Depth makes a discourse choice** (D68, amended). Accuracy is non-monotone in
  `r`: `echo_digit` reaches 96% rank-1 at r=4 and 0% by r=8, while top-1 moves
  from the answer digit to a prose opener — `'The'` for 96–100% of items at r=64
  across every task. Every accuracy kernel in this project ran at `num_steps=32`,
  past that transition. **Amended the same day**: the "answer merely drops to rank
  2–3 behind *The*" reading holds only for the trivial task; on `count4`/`count16`/
  `add1` gold sits 10–25 places back, so genuine uncertainty rides on top of the
  discourse choice. It is a decomposition, not one mechanism.
- **The computation survives depth** (D69). With *"Reply with only the answer"*,
  `echo_digit` at r=64 goes 0% → **83%**, median rank 1, top-1 the actual digit.
  Same model, same depth, one instruction. Depth did not destroy the computation;
  the bare prompt's discourse convergence hid it.

The cost: D69's kernel **printed the opposite conclusion** and had to be rejected
on its own raw output, because the arm I nominated as decider was degenerate
(argmax = partial-UTF-8 byte fragments for 24/24 items on every task) and the
verdict logic never checked. That is D62 repeated in new code, and it is why the
`preflight` guards now exist.

## 1c. The strongest defensible result, and the strongest objection

**Best replicated:** D52 — ρ separates trained from untrained across 14 independent
weight-sets with no overlap (U=0, p=5e-04, d=13.2). Unchanged by any of this.

**Most consequential, least replicated:** D68's capability-ordered rank gap. It has
a working control — the untrained arm sits *at* chance (14668–46119), which a
broken decode could not produce alongside rank 3 in the other arm — but it is one
run, n=24 per task, one seed.

**The strongest objection I cannot answer:** rank is not capability. A low rank
means the answer is available to the readout, not that the model would emit it, and
the whole point of D68 is that those differ. Everything in §1 therefore rests on a
proxy whose relationship to behaviour is exactly what D69 began measuring and has
established for one trivial task only.

**The decisive missing experiment follows directly:** D40/D41/D48/D53 — the
content-null results — have *not* been redone with a graded readout. Until they
are, "content is architectural" is neither established nor refuted; it is measured
with an instrument now known to be blind in that regime. Note that `capcontent`,
the queued kernel for exactly this question, still scores capability by generation
and would repeat the error as written.

## 2. The architecture, verified against source

Facts below are read from `raven_modeling_minimal.py` at the pinned revision, not
from the paper's prose.

- `embed → prelude(2) → [core_block(4)]×r → coda(2) → ln_f → lm_head`, `n_embd=5280`.
- **The core block is weight-tied** and applied `r` times; `r` is sampled during
  training (log-normal-Poisson, mean 33) and chosen freely at inference.
- **Prompt embeddings are re-injected every unroll** via an adapter on the
  *concatenation* `[h, e]`. This makes the core an autonomous map `h ← F_e(h)`.
- **RMSNorm terminates every block**, so the state lives on a sphere.
  **Measured on our own banked data (D99):** `‖h‖ = 76.386`, with per-orbit
  relative sd **median 5.0e-05, max 1.6e-04 across all 512 h₀bank orbits**, and
  1.3e-05 to 8.9e-05 per token across unrolls on the full per-token grids. So the
  state is confined to S⁵²⁷⁹ of radius 76.386 to four significant figures. All
  dynamics are angular; there is no radial mode (independently consistent with
  D58(3), where the orbit difference is 100% tangential).
  **Two consequences that are not decoration.** (a) **Unbounded DRIFT is not an
  available asymptotic regime**, so H1's settle/loop/drift trichotomy is
  *exhaustive* rather than three guesses, and reduces to the sign of the top
  Lyapunov exponent. (b) It fixes the correct random baseline for any distance
  between states: two random points on that sphere are **108.03 ± 0.75** apart
  (= R·√2 exactly), which is what makes D98's cycle vertices at ~38 meaningful —
  0.35× random, i.e. far *closer* than chance.
- Init is Huginn's own, not an HF default: `std = √(2/5d)`, out-projections scaled
  by `1/√(2·d_eff)` with `d_eff = 132` — **depth-scaled for the unrolled depth**.
  So "random weights" here means Huginn at step 0, not a generic init.
- Training used **truncated backprop through the last k=8 unrolls**.
- The model ships a **chat template** (`<|begin_header|>`/`<|end_header|>`/
  `<|end_turn|>`, assistant role `Huginn`) and four `generate_*` methods.

## 3. What is established

**Tier 1 — measured across many weight-sets, survives controls.**

- **Training slows the contraction.** ρ 0.7048 → 0.8577, 5 random inits vs 9 trained
  checkpoints, Mann-Whitney U=0 (no overlap), p=5e-04, d=13.2 (D52). **But 91% of
  the shift is present at the earliest published checkpoint** (step 6144), and the
  within-training trend flips significance under a fit-quality filter — so it is a
  step change before anything saved, not a gradual property.
- **Content is architectural.** Untrained models match or beat trained ones on:
  the counting register (cv R² 0.7498 vs 0.7175 on identical prompts, D53), total
  decoding precision (0.046 vs 0.934 counts, D41), and effective dimensionality
  (PR 1.0 vs 4.8 — the untrained count representation is *literally
  one-dimensional*, PC1 correlating +0.999973 with the count, D48).
- **The trajectory rotates.** Jacobian leading eigenvalue complex in 3/3 prompts,
  27/30 top modes complex, period 2.6–6.0 unrolls (D55).

**Tier 2 — measured once, well-powered, not yet replicated.**

- **ρ is steerable.** Scaling `attn.proj` and `mlp.proj` by (1+ε): paired slope
  **+0.2853 ± 0.0450**, p=2.3e-04, 9/12 prompts monotone (D59). The pre-registered
  prediction `dρ/dε = ρ = 0.887` **failed** at 0.32× — so the branches carry only
  about a third of the contraction and the skip/RMSNorm structure carries the rest.
- **Prompt format is worth ~85 points** on a task the model can do (copy: 15% raw →
  100% chat, D60).
- **Failure is structured, not noisy.** Seven distinct retrieval modes; none is an
  attempted computation (D61).

**Tier 3 — real but scoped.**

- bfloat16 rounding makes the model appear to converge ~4.6× sooner than it does
  (D30).
- Required depth scales with difficulty (D35) — but this lives in the *answer*
  process, not in when the count becomes decodable (D54), and is structurally
  immune to the untrained control that deflated everything else, because random
  weights have no answer process to time.

## 4. What was retracted, and the pattern

Twelve claims retracted or superseded **by later work in this same project**. The
pattern is worth more than any individual retraction:

| what failed | why | row |
|---|---|---|
| winding as evidence | window-governed; sign flips with `num_steps` | D28 |
| the counting register's *direction* | one-token window misalignment; `v` is 97% the current-token contrast | D50 |
| the ρ→usable-depth bound | flipped under estimator and threshold choice | D43 |
| ρ constant across task families | ANOVA never run; family is significant | D45 |
| "decodability vs capability" | accuracy figures came from a different config; neither kernel measured accuracy | D41(3) |
| causal patching verdict | outcome had zero dynamic range before intervention | D47 |
| H1's settle/loop/drift | classes closer together than their own scatter, p=0.48 | D56 |

**Three of these were caught only by reading code or verbatim outputs, not
summary numbers.** Two were caught by a control that had been built and then
excluded from the analysis that depended on it (D47, D62).

## 5. Methodological state

**What works.** Tests that encode *retractions* executably — the only mechanism
here that survived context compaction with zero re-reading. Pre-registered
predictions inside kernel docstrings, which is why D59's failure was informative
rather than rationalised. `ledger.py` validation (found two real gaps first run).

**Known-broken or unresolved.**

- **The 0.058 estimator gap.** Two-orbit gives ρ=0.887, Arnoldi ~0.80. All three
  candidate explanations were tested and **refuted**: tangent projection changes
  nothing (radial fraction 0.0000), the envelope fit works on 3/12 real prompts and
  implies a period contradicting Arnoldi, and the orbits do converge (d_end/d0 =
  4e-05, so not a limit cycle). **Unexplained.** The paired design in D59 sidesteps
  it without resolving it.
- **The KV cache is unused and the reason is unresolved** (D62). `generate_minimal`
  returned empty output; both proposed mechanisms were refuted against the source.
  Sidestepped, not diagnosed.
- **Between-prompt sd of ρ is 0.056**, larger than the 0.03 effect we call
  detectable — so unpaired ρ comparisons are uninterpretable.

---

## 6. Critical re-analysis: blind spots

### 6.1 The deepest one — the untrained controls ran on tasks the model cannot do

Every "content is architectural" result (D40, D41, D48, D53) was measured on
counting-family tasks where the trained model scores **~0%** (D56, D60, D61). If
the model cannot perform the task, there is *nothing for training to have encoded*,
and finding that random weights represent the input equally well is close to
tautological — both models are carrying an unused input.

**This is the single largest threat to the project's headline claim.** "Training
changes dynamics, not content" may be an artefact of only ever testing content on
tasks with no content to change. The claim needs at least one task the trained
model demonstrably performs. Today the only such task in the entire project is
`copy` (100% under chat format, D60), which is trivial.

*Status: PARTLY ANSWERED, 2026-08-08, and the answer runs AGAINST the headline.
D68 measured the trained-minus-untrained gap with a graded readout across a
capability ladder and found it **ordered by capability**: +4.05 log₁₀ rank on
`echo_digit` down to +1.37 on `rot13_word`. So the content gap is real and large
where the model can do the task, and shrinks where it cannot — which is what 6.1
predicted would happen if the confound were real. The counting-family nulls
(D40, D41, D48, D53) sit at the far, capability-zero end of that ladder.*

*NOT closed, for two reasons. (a) D68 is one run, n=24 per task, one seed, and
rank is not capability. (b) The content-null results themselves have not been
REDONE with a graded readout — only new tasks have been measured with it. Until
D40/D41/D48/D53 are re-measured, "content is architectural" is neither
established nor refuted.*

*A WARNING ABOUT THE QUEUED KERNEL. `geometry-cap-content` was built to settle
this and still scores its capability axis by greedy generation plus exact match —
the instrument D68 shows is blind in exactly the regime that matters. Run as
written, its moderator variable would be near-zero across the ladder for
measurement reasons, and it would "confirm" the headline by construction. It
needs the rank readout wired into its capability axis before it is launched.*

### 6.2 ρ has never been connected to behaviour

**CLOSED 2026-08-11 (D173).** Swapping `e` mid-trajectory and leaving it swapped re-aims the dynamics at the new regime in **62 of 64** switches, at switch points from unroll 1 to unroll 48 — no hysteresis, so the regime follows the parameter the map currently has rather than the trajectory's history. And the *time* it takes connects to ρ: the prediction registered before the run was ln(0.05)/ln(0.83) = **16.1** unrolls to close 5% of the gap to a new attractor, and the measured time to ENTER rotation is median **15.5** (range 5–25). Leaving rotation takes median **1.0**. The 15× asymmetry is the robust part — both directions use the identical detector — while the agreement with ρ is suggestive rather than tight, because the width-30 sliding window carries its own lag and the measurement is therefore an upper bound.

D43 tried and was retracted. So ρ — the project's central quantity, the one thing
training changes, now steerable — has **no demonstrated behavioural consequence**.
D59 moves it; nothing measures what moving it does to anything the model outputs.
The ε-sweep should have carried an accuracy arm and did not.

### 6.3 Depth was never varied against accuracy

Audited 2026-08-05: every accuracy kernel fixed `num_steps=32`. On a recurrent-depth
model whose authors report GSM8K 9-10% at r=4 → 28-38% at r=32. `geometry-prompt-depth`
addresses this; it has not yet produced a result.

### 6.4 Huginn's own inference modes are untouched

`generate_with_adaptive_compute`: 0 kernels. `continuous_compute`: 0 kernels until
today. The latter is the architecture's distinctive feature.

### 6.5 Untested but assumed

- ρ is measured at the **final token position only**, on 12 prompts, one seed.
- The untrained arm samples **one init family**; whether ρ≈0.705 is a property of
  random weights or of *Huginn's chosen init scale* is untested (vary `std` ±2×).
- ~~Persistent homology: no null ever computed~~ — **CLOSED 2026-08-05 (D64).** It
  now has one, and it *passed*: real trajectories exceed all 40 manifold-matched
  surrogates in 9/9 cases with loops, 8.7 vs 1.9 mean count. Heavily qualified
  though — the excess is in the NUMBER of cycles, not their prominence (max
  persistence ratio 1.003), every cycle is at the arithmetic-noise scale
  (0.0000–0.0090 of the diameter), and the metric is still budget-governed: 0
  cycles at `num_steps=16` in all 6 runs, 7–26 at 64 in all 9, p=6e-04. **The first
  topological positive in the project, and it is a positive about counts of
  noise-scale cycles that only exist past a recording threshold.**
- `attn` vs `mlp` split is underpowered and its own gate says don't interpret (D63).

### 6.7 Every content measure in this project is LINEAR — found on the second pass

`cv_r2` is ridge. D41, D48, D50 and D53 are all linear decodability. That licenses
one reading of the headline and forbids another that **no measurement here can
currently distinguish**: training may encode the same content *nonlinearly*. If the
trained model represents the quantity in a curved or distributed way while random
weights preserve it linearly — which prompt re-injection would do — then a linear
probe favours the **untrained** arm for reasons unrelated to information content.

That is precisely the observed pattern (D41: untrained 19× more precise; D48:
untrained representation literally 1-D). So it is a live alternative explanation of
the headline, not a hypothetical.

*Status: NOT UNDER TEST — corrected 2026-08-09. The probe has never touched a
real state. `cv_r2_nonlinear` is inlined in exactly one bundle, `kaggle_capcontent`,
and that kernel has never been launched (§6.1 warns it would repeat the instrument
error). `geometry-nonlinear-content`, the one that did run, used the LINEAR probe
deliberately — its own docstring says "not to a nonlinear probe, which would only
re-open D68's 6.7 worry". So 6.7 is fully open, and it now bears directly on D73:
if a nonlinear probe lifts the untrained arm on `max_run`/`alt`, D73's
capability-ordered gap is a LINEARITY artifact rather than a content difference.
See directions.md B13, which is the experiment that settles it. Original status
text, retained because the probe design work it records is real:*

*UNDER TEST in the same kernel, via a nonlinear probe. Getting that probe
right took four designs — MLP(64) reached only 0.42 on a **clean linear** target,
RBF kernel ridge 0.02 with a positive null, PCA-24+poly2 0.36 — because a weak
nonlinear probe reports "no lift" and falsely confirms the headline it exists to
challenge. The accepted design (PCA-8 + degree-2 ridge) is asserted in tests to
recover a linear latent (0.73), beat the linear probe by 0.76 on a curved one, and
give a negative null. Two of the four rejections came from tuning on **isotropic**
synthetics before noticing the real states are low-rank (PR 1.0–4.8, D48).*

### 6.8 Not yet examined at all

- **No non-recurrent baseline.** Everything is Huginn. "Training changes dynamics,
  not content" may be true of any LM of this size — there is no architecture
  contrast, so nothing attributes it to recurrent depth.
- **No natural-language task.** Every task is a synthetic bit-string or word. The
  model's *reported* competence (GSM8K, ARC-C, HellaSwag) is on natural benchmarks,
  so "the model cannot do the task" may partly indict our task construction.

### 6.9 ρ is characterised entirely OUTSIDE the training regime — found on the third pass

**CORRECTED 2026-08-11 (D187): this heading is wrong and the limitation it states does not
apply.** Training depth was never fixed at 32. `raven_modeling_minimal.py:793-797` samples
it as `p ~ Poisson(rate) + 1` with `rate ~ lognormal(log 32 − σ²/2, σ = 0.5)` — mean 33.0,
median 29, **p90 = 56, p99 = 93** over 400k simulated draws. So **P(depth ≥ 48) = 0.169**:
our r = 48 runs sit at the **83rd percentile** of the training distribution and r = 64 at
the **94th**. They are in the tail, not outside it. Only r ≳ 96 would be genuinely
out-of-distribution. *The scope condition below is withdrawn; the ρ numbers do not need it.*


Huginn was trained with `r` sampled log-normal-Poisson, **mean 33, median 29, mode
24**. Every ρ in this project is fitted over the pre-floor window of a 128-unroll
run, and that window is **65–98 points long — all 12/12 prompts beyond 2× the mean
training depth**.

So the project's central quantity, the one thing training demonstrably changes and
the only quantity it has managed to steer, describes the map's behaviour in a
regime the model was never trained to operate in. Whether ρ measured within r≤32
equals ρ fitted over r≤98 is **unknown and untested**, and it bears directly on
every claim built on it: D42, D44, D52, D59, and the retracted D43.

*Status: open. It is a rerun, not an analysis — see 6.10.*

### 6.10 The sweeps saved conclusions, not data

6.9 cannot be answered from what is on disk. `eps_sweep.json` stores the fitted ρ
and the number of points used, **not the per-unroll separation curves** — so
refitting on a restricted window requires re-running the GPU job. The same is true
of the checkpoint sweeps.

This is a scaffolding failure with a clear rule attached: **a kernel should persist
the curve it fitted, not just the fit.** The marginal cost is a few MB; the cost of
not doing it is a GPU rerun for any question the original analysis did not
anticipate — and this project has generated such questions at a rate of roughly one
per experiment.

### 6.6 What a reader should not conclude

- Not "Huginn cannot count" — only "does not, at these sizes, under these prompts."
- Not "geometry is irrelevant" — winding failed; the rotation it was trying to
  measure is real (D55).
- Not "training does nothing to representations" — see 6.1; the test was weak.
