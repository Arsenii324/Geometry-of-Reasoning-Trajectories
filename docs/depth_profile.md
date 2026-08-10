# What depth actually does in Huginn, and what that costs this project

*Opened 2026-08-09 from D92-D94. Everything here is measured on data already
banked; `scripts/run_depth_profile.py` regenerates every number with no GPU.*

---

## The one-paragraph version

Recurrent depth in Huginn does not make the answer better. Over 504 orbits in 21
task families, top-1 accuracy at the answer position **peaks at unroll 5 (16.1%)
and collapses to a flat 6.2% by unroll 8**, where it stays for the remaining 57
unrolls — better at r=5 in 14 of 15 discordant families, exact sign test
p = 9.8e-4. But the answer is not destroyed. Over 1260 generations, the gold string's
presence in the output **rises** from 29.4% to 42.9% across the same depth range,
while a length-matched decoy **falls** from 16.0% to 2.7%. Depth moves the answer out
of the immediate next-token slot while making the output more correct and more
discriminating. And the label this project has called `correct` throughout —
`min_r rank_r == 1` — is a minimum over unrolls: it reads 33.9%, against 16.1% for
the best depth anyone could actually select.

## Why this reframes almost every result in the ledger

The project's nulls were computed on shape codes read from windows that are mostly
in the converged region, against a label that is an oracle over the transient. Those
two facts point the same way:

- **The converged region is where the answer stops changing.** From r=32 onward all
  21 families give bit-identical accuracy. A geometric statistic read there is
  reading a state whose answer content is frozen — which is exactly
  arXiv:2607.20594's argument that tail instruments "provably saturate", and exactly
  where D91 found correctness decodable (unrolls 6–18) and nowhere later.
- **The label was noisier than anyone accounted for.** `correct` credits any unroll
  at which gold happened to top the distribution. D90 showed h₀ moves that minimum by
  up to 6×. So the label carries h₀ variance that the converged model does not have.

None of this rescues a null into a finding. A null measured against a noisy label is
still a null. What changes is the *scope*: "the shape does not encode the outcome"
was always a statement about the converged region, and the project did not know that.

## The correction to D90

D90 recorded that h₀ "decides the answer". The manifest it was read from already
contained the refutation, in a column D90 did not look at.

| | across 10 unseeded h₀ draws of one prompt |
|---|---|
| `best_rank` = min over unrolls | moves in **6 of 8** prompts, up to **6.0×** |
| rank at the **last** unroll | a **single integer** in **7 of 8** prompts |

So h₀ is *forgotten* by the time the recurrence converges — which is what a
contraction with ρ ∈ [0.808, 0.920] requires, and it is reassuring that the two
measurements agree. What h₀ controls is the depth of a transient excursion. D90's
consequence for the project survives intact and is arguably sharper: every capability
number here is a distribution over h₀ **because the label reads the transient**, not
because the model's converged behaviour is stochastic. It is not.

## What is measured, and what is still inferred

**Measured.** The accuracy-vs-depth curve and its family-paired significance; the
plateau onset at r≈8 and exact convergence by r=32; the containment rise against a
length-matched decoy; the h₀ contrast between converged and transient rank.

**Inferred, not measured.** *Where* the answer moved to. Every rank in this project
is taken at one position — the token the answer would start at. "Relocated" is the
only reading consistent with accuracy falling while specific containment rises, but
it is a reading.

**Refuted, and recorded as refuted.** The first mechanism proposed on seeing
`'The number 4 is repeated exactly.'` was that depth prepends conversational framing
and pushes gold out of first place. That predicts a worse gold rank in framed
generations at fixed depth. Measured at depth 32: **median rank 7.0 with framing, 7.5
without**. The framing word-list is too coarse to be the mechanism, and the honest
move is to measure the position rather than a proxy for it.

`scratch/kaggle_answerpos/` does that: it walks the model's own greedy continuation
and ranks gold at *every* generated position, on a battery that
`tests/test_kernel_tasks.py::test_answerpos_battery_is_byte_identical_to_depthacc`
pins to be identical to the generation bank so the two join cell by cell. Its
predictions are pre-registered in the body docstring (Q1–Q5).

## One bug worth keeping on the record

The first containment measure boundary-matched the gold string. Huginn's chat
template leaks role markers into the decoded text with no separator — `echo_digit` at
depth 2 emits `'4Huginn\n\n'` — so the match **missed blurted answers**, which are
concentrated at low depth. That is precisely the bias that would manufacture the
reported rise out of nothing. Fixing it raised depth-2 containment the most
(21.4% → 29.4%) and the effect grew rather than shrank. The miss rate across depths
was 10/12/6/14/12 — real, but not monotone in depth, so it was never the trend.
