# Deep research inquiry #1 (HIGHEST PRIORITY): finding tasks a 3.5B recurrent-depth model can *partly* do, with a difficulty knob

**You are being asked one concrete question.** Everything below is context for it.
Please read the "What a great answer looks like" section before you start, because
it defines success precisely, and the failure mode we most need you to avoid is
returning a list of benchmark names without evidence about accuracy.

---

## 1. The situation, briefly

We are studying the **geometry of latent reasoning trajectories** in
**Huginn-3.5B** (`tomg-group-umd/huginn-0125`, Geiping et al.,
[arXiv:2502.05171](https://arxiv.org/abs/2502.05171)), a *recurrent-depth*
transformer. Its architecture is:

```
embed -> prelude(2 blocks) -> [core_block(4 blocks)] x r -> coda(2 blocks) -> ln_f -> lm_head
```

The core block is **weight-tied and iterated r times at inference**, where r is
chosen at test time (the paper's headline is that accuracy improves with r). We
record the hidden state at each iteration r and study the resulting trajectory.

**A useful property of our instrumentation, which shapes what tasks are usable:**
because the recurrence is iterated, one forward pass at `num_steps=48` lets us
apply the model's output head at *every* intermediate depth. So a single forward
yields the model's answer-token rank at all 48 depths. **Depth resolution is free;
what is expensive is prompts.**

## 2. The problem that blocks us, stated exactly

To relate trajectory geometry to *reasoning*, we need tasks where the model's
outcome actually **varies**. Specifically we need per-item accuracy that is
**strictly between 0 and 1** — ideally near 50%. An item the model always gets
right, or always gets wrong, carries no information about what distinguishes
success from failure, no matter how much its geometry moves.

Two of our experiments have now failed *not on method but on task supply*:

- A causal activation-patching experiment required at least 4 of 9 candidate
  items to show both correct and incorrect outcomes across random-initialisation
  draws. Only 2 did. The experiment declared itself void.
- An earlier experiment failed the same way for the same reason.

**We have measured the following on Huginn ourselves** (21 hand-written synthetic
families, 24 items each, graded by whether the gold answer's first token reaches
rank 1):

| family | our measured accuracy |
|---|---|
| `echo_digit` (repeat a digit) | ~100% |
| `compare` (which is larger) | ~92% |
| `add1` | ~79% |
| `sub1` | ~62% |
| most counting / parity / cipher / indexing families | ~0–30% |
| `rot13_word` | ~0% |

So our synthetic tasks are almost all **pinned at one end or the other**. Very
few sit in the measurable band.

**Meanwhile the model is genuinely capable on in-distribution natural-language
benchmarks** (numbers quoted from the paper's own tables, at r=32):
GSM8K 28.05 / GSM8K-CoT 32.60, ARC-Easy 69.91, ARC-Challenge 38.23,
HellaSwag 65.21, MMLU 31.38, OpenBookQA 38.80, PiQA 76.22, WinoGrande 59.43,
Minerva MATH 12.58, MathQA 26.60. At r=1 versus r=32 on an early checkpoint,
ARC-E goes 34.01 -> 53.62 and GSM8K-CoT 0.00 -> 9.02.

**The tension we cannot resolve by ourselves, and the heart of this inquiry:**

> Tasks with a *clean difficulty knob* (ciphers with k rotations, prefix sums of
> length n, parity of n bits) are synthetic and **out of distribution — the model
> scores ~0**. Tasks the model is genuinely *good at* (GSM8K, ARC, HellaSwag) are
> natural-language and have **messy, uncontrolled difficulty** and long
> multi-token answers.
>
> **We need the intersection: tasks with a controllable difficulty parameter that
> a ~3B-parameter model solves at intermediate rates.**

## 3. Hard constraints on any task you propose

Please check proposals against all five. Partial matches are still useful — just
say which criteria fail.

1. **Intermediate accuracy.** A ~3B model should get roughly 20–80% correct.
   Evidence for this matters more than anything else in your answer.
2. **An explicit difficulty parameter** — an integer we can turn (number of
   reasoning hops, sequence length, problem size, number of rule applications)
   that monotonically increases required computation.
3. **Token count should NOT grow with difficulty**, or should be controllable
   independently. This is critical: we have repeatedly found trajectory geometry
   tracking *prompt length* rather than task difficulty, so if harder items are
   also longer, the experiment is confounded. Tasks where difficulty rises while
   length stays fixed are enormously more valuable to us. (Example of the design
   we want: two prompts with byte-identical bodies differing in one marker
   character that selects a different computation.)
4. **Short answers, ideally a single token.** We score by the rank of the gold
   answer's first token. Multi-token answers ("forty-two", "Add") mean we are
   measuring the leading token, not the answer. Single digits, single letters, or
   yes/no are ideal.
5. **Plausibly in or near Huginn's training distribution.** Its published mixture
   includes (among much else) `tomg-group-umd/CLRS-Text-train`,
   `nvidia/OpenMathInstruct-1`, `meta-math/MetaMathQA`, `hkust-nlp/gsm8k-fix`.

## 4. What we already know or suspect (do not spend effort re-deriving)

- **CLRS-Text** is the most promising lead we have identified, because it is *in
  the training mixture* and has an explicit integer problem-size knob. We are
  running a capability screen on it now. **Anything you can tell us about which
  CLRS-Text algorithms and sizes are tractable for small models, and what the
  answer format looks like, is high value.**
- **Prompt format is a large lever**: adding one instruction moved one of our
  tasks from 0% to 83%. So a task that looks impossible may just be badly
  prompted. Evidence about prompt formats that work for base-ish small models is
  useful.
- Accuracy is **non-monotone in depth** for some tasks — more iterations can hurt.
- The model's answers get **wrapped in prose** at higher depth: exact-match
  scoring collapses while containment scoring rises. So scoring method matters.

## 5. The questions, in priority order

**Q1 (most important). Which concrete datasets or task generators satisfy
criteria 1–5, especially 1 and 2 together?** For each: what is the difficulty
knob, what is the answer format, and what accuracy should we expect from a
~3B-parameter model? Candidates we are aware of but have *not* verified include
CLRS-Text, PARARULE-Plus, ProofWriter, RuleTaker, bAbI, LogiQA, ProsQA, and
multi-hop QA with a controllable hop count — but we do not know which are
tractable at this scale, and we would rather have three well-evidenced options
than twenty names.

**Q2. For CLRS-Text specifically:** which algorithms and problem sizes do small
models actually get partly right? What exactly does an item look like (prompt and
expected answer string)? Does answer length grow with problem size — i.e. does it
violate criterion 3?

**Q3. Is there published work on *difficulty-controlled* evaluation of small
language models** where the difficulty parameter is explicit and accuracy is
reported per level (an accuracy-versus-difficulty curve rather than a single
number)? We want to reuse an existing calibrated ladder rather than build one.

**Q4. Are there tasks where difficulty rises while token count stays fixed?**
This is the hardest constraint to satisfy and the most valuable to us. Any task
family, published or constructible, with this property is a major find.

**Q5. What is known about *which* tasks benefit from test-time depth/recurrence**
in looped or recurrent-depth transformers (Huginn, Ouro, universal transformers,
looped transformers)? If some task class is known to improve with more iterations,
that class is where our hypothesis is testable, and it is where we should look
first.

## 6. What a great answer looks like

For each recommended task, we want a row we can act on:

> **Name** — where to get it (HF dataset id / repo / generator code).
> **Difficulty knob**: what integer we turn, and its useful range.
> **Item example**: literal prompt text and literal expected answer.
> **Answer length**: single token or not.
> **Does length covary with difficulty?** yes / no / controllable.
> **Expected accuracy for a ~3B model**, with a citation or measured number.
> **Confidence**, and what you could not verify.

Three such rows, well-evidenced, beat a survey of thirty datasets.

**Please mark every factual claim with how you verified it:**
`[V]` you read the actual paper/dataset/code and confirmed it;
`[T]` title/abstract or documentation only;
`[U]` unverified or inferred.
We maintain this discipline in our own notes and will not act on `[U]` claims. A
smaller, honestly-tagged answer is far more useful than a confident one we then
have to re-check. **If the honest answer to Q4 is "no such family is known", that
is a genuinely valuable result — say so plainly rather than padding.**

## 7. Explicitly out of scope

- Do not propose fine-tuning, training, or modifying the model. We evaluate a
  frozen public checkpoint.
- Do not propose tasks requiring long generated chains of thought. We read the
  answer token's rank at a single position.
- We do not need trajectory-geometry or interpretability literature here — that
  is covered by separate inquiries.
