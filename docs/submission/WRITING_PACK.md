# Writing pack — for the post-compaction, writing-focused pass

*Written 2026-08-11 18:10 while full context is still held. `RESULTS_HANDOFF.md` has the
numbers; this has **the reasoning that links them**, the **scope lines that each cost a whole run
to establish**, and **final prose for the paragraphs that are hard to write without context.**
If you are a later pass with a compacted context: the numbers are safe, the *argument* is what
you would otherwise have to reconstruct, and §C is the device that stops you overclaiming.*

---

## 0. Read `CLAIM_INVENTORY.md` first

21 claims with presuppositions, experiment links (kernel + job id), gates, and safe-vs-strongest
forms. This pack is the *prose* layer over it. If a sentence here and the inventory disagree, the
inventory wins — it carries the caveats this pack compresses.

## A. The argument spine — the paper's logic, in order

The paper has **one narrative** with a hinge in the middle. Do not present it as two unrelated
result sections; the hinge is what makes it interesting.

1. **Setup.** A recurrent-depth transformer reuses one weight-tied block; the latent trajectory
   has been treated as a trace of reasoning. We measure that directly on Huginn-3.5B.
2. **The map has sharp qualitative structure.** One instruction word switches it between
   settling and a damped period-6 rotation, at an identical token count. The switch is causal
   from the embedding row. The regime is a property of the prompt, not of the position sampled.
   Swapping the injected parameter mid-run re-aims the dynamics with no hysteresis.
3. **THE HINGE.** *We then tried to connect that structure to behaviour and could not — and
   investigating why revealed that most of what we had been scoring as capability was answer
   formatting.* This is the pivot. Everything after it reinterprets what came before.
4. **The formatting result.** First-token scoring reads 0.125 where the answer is present in
   0.781. What displaces the answer is `The`. Depth *raises* prose-opening while leaving
   leading-with-the-answer flat. Supplying the format recovers the published benchmark number by
   two independent routes.
5. **The timing result closes the loop.** The regime and the answer occupy **non-overlapping
   windows of the same trajectory** — the answer is fixed at unroll ~4, the regime is not
   measurable until ~unroll 16. So the behavioural nulls were never evidence that the regime is
   inert; they measured a property that had not yet come into existence.
6. **The hypothesis verdict.** H2 as the literature means it (per-position adaptive depth) is
   architecturally untestable here — there is no halting head, so every position gets `r` and
   there is nothing to allocate.
7. **Method, presented as a result.** Ten claims were withdrawn or amended, four of them by
   their own pre-registered controls. This is the reliability argument, not an apology.

**The unifying sentence, if one is wanted:** *we set out to measure the geometry of latent
reasoning, found that most of what we were measuring was output format, and fixing that changed
what the geometry results mean.*

---

## B. Final prose for the paragraphs that are hard without context

*Use these close to verbatim. Each encodes a distinction that took a full run to establish and
that is easy to garble.*

**B1 — the timing result (this is the subtlest thing in the paper).**

> Measuring the same rotation statistic over two windows of the same orbit separates two facts
> that had been conflated. Over the decision window — unrolls 0–11, the earliest a period-6
> orbit is definable at all, since the statistic requires two cycles — rotating and settling
> prompts are indistinguishable (0.229 against 0.230, separation 0.000). Over the tail they
> separate completely (0.862 against 0.298). A sliding window dates the divergence to between
> unrolls 12 and 16. The early window is not quiet: its non-DC power exceeds the tail's by about
> two orders of magnitude. It is energetic and simply has no period-6 structure. Since the
> answer is best-ranked at a median unroll of about four, the earlier finding that a
> causally-induced regime change alters almost nothing behaviourally is not evidence that the
> regime is a dynamical epiphenomenon; it manipulated a property that had not yet come into
> existence when the readout was settled. **This is a timing claim rather than a causal one.** It
> does not show that the regime could not affect behaviour, only that on this architecture the
> answer is fixed roughly twelve unrolls before the regime becomes measurable.

**B2 — the formatting result, stated so it is not mistaken for "the model is bad".**

> Scoring the first generated token, the model answers correctly on 12.5% of items. Reading the
> generated text, the answer is present in 78.1% — against a measured cross-item false-positive
> rate of 0.042 to 0.152, so the gap is not an artefact of a permissive scorer. What occupies the
> first position is prose: the token `The` opens 39.4% of 1260 banked generations. Depth makes
> this worse rather than better — the rate of opening with `The` rises twentyfold from r = 2 to
> r = 32 while the rate of opening with the answer does not move at all (p = 0.515). **The model
> is not failing to compute the answer; it is failing to put the answer where our metric looks.**

**B3 — the benchmark reproduction, with the right route identified.**

> The model's published ARC-Easy score is 69.9% at r = 32. Read from the released code, that
> number is produced by `lm-eval-harness` at zero shots with normalised accuracy — a bare
> continuation prompt with no option list. Scored that way we obtain **65.8%**, within 4.1 points.
> A second and independent route reaches the same place: five in-context examples under
> option-letter argmax give **72.3%**, paired, 35 items fixed against 4 broken (exact McNemar
> p = 3.4e-07). *The first is their protocol; the second is a coincidentally matching route and
> should not be described as the reproduction.* Across 25 measurements the partition is clean and
> non-overlapping: zero-shot arms that list the options in the prompt read 0.392–0.495, and those
> that do not read 0.592–0.658.

**B4 — the retraction section's opening, so it reads as strength.**

> We report the claims this project withdrew alongside the ones it kept, because several of the
> withdrawals were produced by controls we had registered in advance and are therefore evidence
> about the method rather than only about the errors. In four cases a result was killed by its
> own pre-registered gate: a matched-pair geometry claim did not survive the norm control it had
> itself specified; a spectral run was refused by its own consistency check; a behavioural null
> turned out to have been computed on a variable that took the same value in 432 of 432 orbits;
> and a scoring comparison was found to be measuring the parser rather than the model.

**B5 — H2, stated so it is not read as "we refuted H2".**

> The hypothesis that harder problems recruit more depth is, in the sense the literature means
> it, not testable on this model. Adaptive-computation architectures measure per-position ponder
> time and find it rising with difficulty (2.3 ± 0.8, 3.1 ± 1.1, 3.8 ± 2.2 as the number of
> required supporting facts goes from one to three). Huginn has no halting head: `num_steps = r`
> unrolls the whole sequence together, so every token position receives exactly `r` iterations
> and there is nothing to allocate. **Our null results on this axis were measuring the only
> quantity the model can vary — the global iteration count — on an architecture with no mechanism
> to vary it per instance.** Depth does buy capability at fixed difficulty, which the model's own
> paper shows and we confirm independently; what is absent is allocation *by* difficulty.

**B6 — the steering result, as one finding rather than two.**

> We applied a difference-of-means steering vector in the map's injected parameter, fitted on the
> contrast between two-shot and zero-shot prompts and added at every unroll. It reproduced
> neither the behavioural benefit of in-context examples nor any change in the rotation regime:
> first-token accuracy moved from 0.000 to 0.080 with a norm-matched random control
> indistinguishable throughout, and the rotation statistic moved by at most 0.0065 with zero
> threshold crossings in 132 conditions. The dose was adequate — oracle accuracy degraded at
> large coefficients, so the perturbation acted. **We read this as the wrong injection site rather
> than as two orthogonal directions**, because the regime's positional onset is 33 positions
> upstream of where the vector was applied, and because a null on both measures is more
> economically explained by the site than by a coincidence of two independent nulls.

---

## C. Anti-overclaim table — what you may say, and what you may not

*The single most useful section for a compacted writer. **Left column is safe. Right column is
false or unsupported.** Every right-hand entry is a sentence I actually had to correct today.*

| may say | may **not** say |
|---|---|
| the regime and the answer occupy non-overlapping windows | "the regime is a dynamical epiphenomenon" *(killed by D198)* |
| the answer is fixed ~12 unrolls before the regime is measurable | "the regime does not affect behaviour" *(untested — timing ≠ causation)* |
| numerical-reasoning content is **not sufficient** for the orbit | "we refuted their orbiting claim" *(the phenomenon is theirs; the attribution is what fails)* |
| the harness protocol reproduces their number at 0.658 | "few-shot reproduces their number" *(D169 is a different protocol from theirs)* |
| inflation is 2.06× / 2.00× / 5.29× **depending on task and budget** | any single "inflation factor" |
| no format **instruction** works; demonstration does | "the model cannot follow instructions" *(and note the `Answer:` exception)* |
| appending a marker after prose is the one qualified exception, p = 0.125, n = 24 | "reason-then-mark works" *(suggestive, not established)* |
| steering at the last position moved nothing | "formatting and regime are orthogonal directions" *(site not tested)* |
| the embedding row causally determines the regime **within a small, direction-dependent neighbourhood** | "the embedding encodes the regime" *(the within-regime control partly failed)* |
| ρ is family-level, spanning 0.8239–0.9181 | "ρ = 0.83" as a single model constant |
| H2 is architecturally untestable per-position here | "H2 is refuted" |
| our ARC numbers and theirs share an uncorrected first-token bias | quoting 0.658 as a clean capability estimate |
| the linear probe class cannot resolve this question at n ≈ 50 | "the regime is not linearly decodable" *(the instrument failed, not the phenomenon)* |
| depth raises prose-opening while leaving answer-opening flat | "depth hurts accuracy" |
| `contains`-style numbers, **always beside their measured chance rate** | a containment number alone |

---

## D. Traps a post-compaction writer will fall into

1. **Writing "epiphenomenon".** It was the working reading for weeks and it is *wrong* as of
   D198. Search the draft for it before submitting.
2. **Crediting D169 as the reproduction.** D175/D183 is theirs. This one is easy to get backwards
   because 0.723 is closer to 0.699 than 0.658 is — *closeness is not the criterion, protocol is.*
3. **Merging D196 and D197 into "two nulls".** They are one intervention measured twice.
4. **Dropping the scope line from D161.** Its within-regime control partly failed; the claim is
   neighbourhood-local.
5. **Presenting the retractions defensively.** They are evidence about method. B4 is the tone.
6. **Quoting D148 at all without its amendment.** Its correctness half is vacuous; D176 is the
   version that has variance.
7. **Saying "the regime is not linearly decodable".** D165 shows that probe class fails on a
   label it is *guaranteed* to contain — the instrument is what failed.
8. **Forgetting `grep -a`.** The draft `main.tex` is CP1251; plain `grep` reports nothing and you
   will conclude the file is empty or the numbers are missing. This already caused one false alarm.

---

## E. Draft OpenReview fields — **unreviewed, do not paste blindly**

*I wrote these; no author has read them. The abstract in particular commits to a framing
("two lines that meet") that was never discussed. Treat as a proposal to be challenged.*

**Title**
> What the Latent Trajectory of a Recurrent-Depth Transformer Encodes, and What Our Measurements
> Were Actually Measuring

**TL;DR**
> We set out to measure the geometry of latent reasoning in a recurrent-depth transformer, found
> that most of what we were measuring was output formatting, and fixing that changed what the
> geometry results mean.

**Keywords**
> recurrent-depth transformers, latent reasoning, mechanistic interpretability, evaluation
> methodology, dynamical systems, test-time compute

**Abstract** *(~215 words; every number in it is in `RESULTS_HANDOFF.md`)*
> We study the latent trajectory of Huginn-3.5B, a recurrent-depth transformer that applies one
> weight-tied block a variable number of times before decoding. We report two lines of results
> and the connection between them. First, the recurrent map has sharp qualitative structure: a
> single instruction word switches it between a settling regime and a damped period-6 rotation
> (36 of 36 paired cells at an identical token count), the switch is causally controlled by one
> row of the embedding matrix, the regime is a property of the prompt rather than of the token
> position sampled, and swapping the map's injected parameter mid-trajectory re-aims the dynamics
> with no hysteresis in 62 of 64 switches. Second, attempts to connect that structure to
> behaviour revealed that much of what we had measured as capability was answer formatting:
> first-token scoring reads 0.125 where the answer is present in 0.781 of generations, depth
> raises prose-opening twentyfold while leaving the rate of leading with the answer flat, and
> supplying the output format recovers the model's published benchmark number by two independent
> routes. Finally, the two lines meet: the regime and the answer occupy non-overlapping windows
> of the same trajectory, the answer being fixed some twelve unrolls before the regime becomes
> measurable. We report the claims this work withdrew alongside those it kept.

---

## F. Figures and tables worth having, with the data behind each

*None of these exist yet. In priority order, if there is time.*

1. **The onset curve** (strongest visual, and it carries the paper's hinge). Sliding 12-unroll
   rotation power vs unroll, two lines (rotating / settling), with a marker at the median
   answer-decision unroll (~4). Data: `scratch/ds_transrot/transrot.json`, key `slide`.
2. **Rank-vs-unroll for one item**, showing the gold reaching rank 1 early and being displaced.
   Data: any `rank_curve` in `ds_transrot` or `kaggle_census`.
3. **The ARC partition table** — 25 measurements, options-listed vs not. Regenerate with
   `scripts/arc_table.py`.
4. **Position-wise R for one rotating and one settling prompt**, showing the onset boundary at
   position 20 of 53. Data: `scratch/ds_posrot/posrot.json`, key `R_by_pos` with `tokens`.

---

## G. Contingency for the one unresolved job

**A47 (`ds_h0inject`, `bt1vuemjmeg0onc52shp`)** — injects a donor state at h₀ and asks whether it
reaches the readout before the map forgets it. At 18:06 it was at 4/18, ~25 min from done.

- **If it lands and P1/P2 pass:** it is an H3 result and belongs in §5 as one paragraph. **A null
  is partly predicted** — the model's own paper claims path independence over h₀ — so the
  informative arm is P4, whether the recipient briefly ranks the *donor's* answer. Say "content
  transfer" only if P4 is positive.
**RESOLVED 18:26 — it errored a third time. OMIT ENTIRELY.** Do not mention it, not even as attempted. The paper is complete without it and H3's verdict is unchanged.

- **If it errors or does not land by 19:00:** **omit it entirely.** Do not mention it as
  in-progress; the paper is complete without it. Two prior attempts failed on a data-path gate,
  which is recorded in the ledger and needs no space in the paper.

---

## H. Related work — a real defect, and the material to fix it

**RESOLVED 2026-08-11 19:25 — do not re-do this.** The defect was real: the draft declared 6
`\bibitem`s and contained exactly one `\cite`, and LaTeX's "0 undefined citations" check runs the
other way so a green compile hid it. **It is now fixed and verified by counting: 6 `\bibitem`s, 6
`\cite` commands, keys matching one-to-one.** The table below is kept because it records *what
each reference is for*, which is still useful when writing — not because anything is outstanding.

*One correction to the table: the state-tracking row's key is `grazzi2025statetracking`, not
`merrill2024ssm`. The parity/S₂/S₅ claims come from arXiv 2411.12537; an earlier draft cited
2404.08819 for them, which was a misattribution and has been fixed.*

**What each is for, and the sentence it supports.** *All claims here are verified in-file from
`/Users/a2mogus/build-projs/huginn-load`; quotes are exact.*

| ref | cite it where | the point it carries |
|---|---|---|
| `geiping2025huginn` | throughout | the model, the architecture, the published benchmark numbers (ARC-E 34.89 / 49.07 / 65.11 / 69.49 / **69.91** at r = 1/4/8/16/32; GSM8K CoT **0.00 → 34.80** from r = 1 → 32), and the orbiting observation we contradict the *attribution* of |
| `graves2016act` | the H2 section | the positive control for adaptive depth: *"ponder time appears to grow linearly with difficulty"* — in networks **trained with a ponder cost** |
| `dehghani2019ut` | the H2 section | per-position ponder time rising **2.3 ± 0.8 → 3.1 ± 1.1 → 3.8 ± 2.2** with the number of required supporting facts; this is what H2 means in the literature and what Huginn cannot express |
| `banino2021pondernet` | the H2 section | states the target directly: computation *"grows with the size of the inputs, but not with the complexity of the problem being learnt"* |
| `merrill2024ssm` | the state-tracking / `parity8` discussion | parity is the **S₂** word problem and *"cannot be solved by modern architectures"*; harder state tracking sits above **S₅**. This is why `parity8` getting *worse* with depth is expected rather than anomalous |
| `bai2019deq` | where `e` is called the map's parameter | the DEQ framing. **Strengthen it with a fact rather than an analogy:** `block_idx` is threaded through `core_block_forward` and never enters the block computation, so the unrolled map is genuinely time-invariant |

**Uncited but worth one sentence each, if space allows:** the dedicated interpretability study of
this model reports a **negative** result (*"Token rank trajectory analysis provides little
evidence for latent CoT reasoning"*), and evaluates an **SWA-merged checkpoint**, so any numeric
comparison to it is cross-checkpoint. A separate paper proves a **tail-blindness** proposition —
past the loop where a fixed point is reached, local Jacobians and any attribution graph built
from them agree across loops, so *"none of these quantities constrain the computation"*. That is
directly relevant: it means Jacobian-space analysis read on the converged tail is provably
uninformative, and our transient/tail result (§B1) is on the informative side of that line.

## I. What we may NOT claim novelty for

*Get this wrong and a reviewer's first paragraph writes itself.*

- **Orbiting itself is theirs.** They observed it and named it. **Our contribution is that their
  attribution is wrong** — the same nominal task rotates 32/32 in one wording and 0/22 in
  another, and one noun flips it at fixed token count.
- **Path independence over h₀ is theirs** (*"re-initializing from multiple starting states… the
  model moves in similar trajectories"*). If A47 lands, frame it as a *readout-window* question,
  not as discovering path independence.
- **Depth buying capability at fixed difficulty is theirs**, and their numbers are stronger than
  ours (GSM8K CoT 0.00 → 34.80). We confirm it independently; we do not discover it.
- **The per-position adaptive-depth idea is ACT/UT/PonderNet's.** We contribute the observation
  that Huginn cannot express it, which is why our H2 nulls are not evidence against H2.
- **Surface-form bias in likelihood scoring is a known problem** with a known correction (PMI-style
  normalisation against an unconditional score). We inherit it uncorrected — *as does the number
  we compare against* — and say so.
- **What is ours:** the single-word regime switch and its causal control from the embedding row;
  the regime being a property of the prompt rather than the sampled position; the no-hysteresis
  and entry/exit-asymmetry result; the formatting-versus-capability decomposition; the
  transient/tail timing result; the first steering-vector experiment on a recurrent-depth model
  (a null); and the reproduction of their benchmark number by two protocols.

## J. Reviewer objections, and the honest answer to each

*Pre-empt these in the text. Every one is fair; three are already conceded in the record.*

1. **"n is small."** True in places — 18 paired units for the behavioural null, 24 items per arm
   for the format sweep, 12 base prompts for the steering geometry. **Answer: say so and give the
   bound.** Where a null is quoted, state what effect size it excludes. Do not defend the n.
2. **"Single model, single checkpoint, single seed."** True. **Answer:** the project is
   Huginn-scoped by design; every forward is seeded (the model draws h₀ from an unseeded RNG and
   nothing seeded it before we did); cross-checkpoint comparisons are flagged where they occur.
3. **"Your accuracy numbers are protocol-dependent."** *That is the paper's own finding.* Turn it
   from an objection into the contribution — 25 measurements partition cleanly by whether the
   option list is in the prompt.
4. **"The rotation statistic is hand-picked."** Fair. **Answer:** it is validated to 4e-08
   against its own spectral definition, a single threshold on it reproduces the binary label on
   691/696 orbits and 688/696 held out, and our rotating prompts peak at period 6 *specifically*
   rather than merely somewhere.
5. **"You changed your conclusions repeatedly."** **Answer: §B4.** Four were killed by controls
   registered in advance. This is the reliability argument.
6. **"The steering null is one draw."** Conceded explicitly — one contrast, one aggregation, one
   injection site, and we name the discriminator we did not run.
7. **"Why should we believe the formatting story rather than low capability?"** Strongest answer
   is the *paired* one: the same items, the same model, the same depth — only the prompt format
   changes — and accuracy moves 0.416 → 0.723 with 35 items fixed against 4 broken.

## K. Regenerating the draft — mechanics, so a later pass need not rediscover them

```
cd files/paper_submission/draft
pdflatex main.tex && pdflatex main.tex      # twice: refs/labels
grep -c "^!" main.log                        # must be 0
grep -oE "Output written on main\.pdf \([0-9]+ page" main.log
file main.tex                                # must report ISO-8859 (CP1251), not UTF-8
```

- **`main.tex` must stay CP1251.** Editing it with a UTF-8 tool will silently corrupt the Russian
  block. Verify with `file` after any edit, and **use `grep -a` to search it**.
- `zapiski.cls` and `pic/` must sit beside `main.tex`; they are already there.
- The Russian metadata block sits **before** `\end{document}` (the shipped template puts it after,
  where it never typesets). Do not move it back.
- Do **not** reintroduce the three `\renewcommand` masthead overrides from the distributed
  template — with a CP1251 `.tex` they are unnecessary and were deliberately removed.
- Content source is `RESULTS_HANDOFF.md` only. **It is currently ahead of the compiled PDF by
  D198, D199, D200 and the A7 amendment — regeneration is required, not optional.**

---

## L. Self-audit against a published list of deceptive practices (18:45)

*Checked the draft against `~/Downloads/guides-write/NotGoodIdeas.md`. Most of that list targets
method-proposal papers — inflating compute, tweaking baselines, cherry-picking architectures —
and does not apply: we propose no method and have no baseline to sabotage. **Four items do
apply, and three are already handled.***

- **5.8, "surreptitiously add hints to test prompts, comparing few-shot and zero-shot."** This is
  the sharpest one for us. We *do* compare 0.416 zero-shot against 0.723 five-shot and call it a
  reproduction. **Handled: §B3 identifies the zero-shot harness arm (0.658) as their protocol and
  explicitly labels the five-shot route as coincidentally matching, not the reproduction.** Keep
  that sentence; it is the difference between an honest and a misleading claim.
- **5.1/5.2, "report only the metrics/datasets that improved."** We report all 21 census families
  including the ones reading 0.000, and every null with its bound. **Clean, and worth saying so
  in the text** — it is unusual enough to be evidence about the method.
- **2.4, "cherry-pick random seeds."** Single seed throughout, stated. Not cherry-picked, but
  also not a seed sweep. **Disclose as a limitation** rather than leaving it inferred.
- **5.9, "indirectly overfit by leaking data or seeds."** A real near-miss: the few-shot census
  run would have shown the model the exact test item with its answer, because `echo_digit` has
  only ten distinct prompts. **Caught pre-launch by an assertion, and exemplars are now filtered
  against the test set.** Worth one clause in the text — it is a concrete instance of a control
  working.

---

## M. Checked against the writing guides — my earlier dismissal was wrong

*I had skipped `~/Downloads/guides-write/guide-{1,2,3}` as "wrong stage". Their headings map
directly onto our risks. Checked the draft mechanically against them; two real gaps, three
clean.*

**GAP 1 — zero figures, and guide-3 is blunt about it:** *"Figures are a big deal, and figure 1
is the most important figure. Many readers will literally skip all your text."* We have **0**
`figure` environments. **The onset curve is our Figure 1** — it carries the paper's hinge (C7) in
one image: two lines identical through unroll 12, diverging to 0.96 vs 0.55. Data:
`scratch/ds_transrot/transrot.json`, key `slide`. Three further candidates with data paths in §F.

**GAP 2 — guide-3 asks the introduction to answer three questions explicitly**, and we answer
only two: *how seriously should I take the main claims*, *what flavour of evidence will there
be*, and — **missing** — *what kinds of evidence should you not expect, even if it would have
been nice to have had this*. **We have that material and it is unusually strong**: no behavioural
h₀ test (three failed attempts), single model and checkpoint and seed, an uncorrected
surface-form bias shared with the number we compare against, and no per-position depth to
allocate. Stating it in the intro converts a limitation section into an asset.

**CLEAN — checked, not assumed:**
- *Intensifiers* (guide-1: jettison "extremely, very, completely, essentially, rather…"): all 17
  occurrences of "rather" are the contrastive `X rather than Y`, not the intensifier `rather
  large`. One "completely", one "barely", both load-bearing.
- *Introduction length* (guide-3: ≤1–1.5 pages): ours is ~0.4 pages.
- *"A sin of omission is better than a sin of commission"* (guide-1) — this is exactly what
  `CLAIM_INVENTORY.md` Part 6 enforces, and independently validates keeping it.

**Also worth heeding, not yet done:** guide-1 *"cite generously — the papers you ought to cite
are likely written by the people who will review your paper"*. We went from 1 citation to 6
today; 6 is still thin for a paper that surveys 12.

### M2 — full read of all three guides; ranked deltas

*Read whole, not skimmed. Ranked by cost of not doing.*

1. **Figure 1 does not exist.** Guide-3: *"the most important figure… many readers will literally
   skip all your writing and go straight to figure 1"*; in 1-column it belongs at the top of
   page 2. **Ours is the onset curve** (C7): sliding 12-unroll rotation power, two lines,
   identical through u12 then diverging to 0.96 vs 0.55, with a marker at the median
   answer-decision unroll (~4). Data `scratch/ds_transrot/transrot.json` key `slide`. Guide-1
   adds two constraints: the figures must tell the story alone **and** the text must stand
   without them; caption 1–3 lines, and say which direction is better.
2. **No contributions bullet list.** Guide-3: 2–4 items, each ≤1 line, in the intro.
3. **Intro does not say what evidence *not* to expect.** Guide-3 lists this as one of three
   things the evidence paragraph must answer. We have strong material for it.
4. **Abstract follows neither formula.** Guide-2: sentence 1 = something every reader agrees
   with; sentence 2 = surprising but following from it. Guide-3: (1) what achieved, (1) why hard,
   (1) how, (2) evidence including the most remarkable number. Ours opens *"We study the latent
   trajectory of Huginn-3.5B"* — neither. **Candidate opener in the guide-2 shape:** *"A
   recurrent-depth transformer's latent trajectory is usually read as a trace of its reasoning.
   We find that most of what such readings measure is the model's willingness to answer in the
   format being scored."*
5. **Six citations from a twelve-paper survey.** Guide-1: *cite generously*, *cite throughout*,
   *exhaust the references limit* — and notes the uncited authors are often the reviewers.

**Validated, no change needed:** *avoid hostages to fortune* and *a sin of omission is better
than a sin of commission* are exactly what `CLAIM_INVENTORY.md` Part 6 enforces. Intensifier
sweep already clean (all 17 "rather" are contrastive).

**On guide-3's LLM-drafting advice — considered and does not apply here.** Its premises are
about a *human* author using an LLM on work the LLM did not do: *"if you are a good writer you
are better than LLMs"*, *"if you are a bad writer you need the practice"*, *"you will learn more
about your work by writing it"*. None parses in this setting — no author is being deprived of
practice or understanding. Its fifth premise, *"LLMs are currently not great at explaining novel
things"*, fails specifically: the novelty here is documented claim-by-claim with instrument,
gate and job id in `CLAIM_INVENTORY.md`, which is the opposite of paraphrasing unfamiliar work.
**Only the style premise is live** — *"annoying, preachy, long-winded text with a distinctive
style"* — and that is a check already run (intensifier sweep clean; anti-overclaim table; terse
discipline), not a prohibition. Recorded because I initially over-applied it.
