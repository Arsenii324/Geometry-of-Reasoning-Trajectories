# Working knowledge — what I know that isn't in the claims

*Written 2026-08-11 18:00, before a context compaction near the deadline. The ledger records
**what** was found; `UNDERSTANDING.md` records **what it means**. This file records the things I
would otherwise have to rediscover: how the instruments actually behave, which numbers are
load-bearing, which claims are fragile and why, and the reasoning patterns that repeatedly paid
off. None of this is obvious from the rows.*

---

## 1. The instruments, and what they are secretly doing

**`rotation_power(traj, period=6, tail=24)`** is the spine of the regime line, and three of its
properties are non-obvious:

- **`tail` and `period` are coupled, not independent.** The function requires `tail % period == 0`
  (so the period lands on an exact rfft bin) and `tail // period >= 2` (so a periodic orbit is
  distinguishable from a single excursion). **A49's first pass died on exactly this**: I set the
  spectrum tail to 48 and compared against `rotation_power`'s default 24 — different windows,
  never comparable.
- **A longer tail is *worse*, not better, for a damped orbit.** This is counterintuitive and I
  got it backwards once. The orbit decays, so a 48-unroll window includes the transient, whose
  envelope dominates the low-frequency bins — `symbol` then peaks at period *48* rather than 6.
  `tail=24` exists to look at the **settled** part.
- **It cannot see anything before unroll 12** (two cycles at period 6). Since the answer is
  settled at unroll ~4, **no period-6 statistic can describe the state at the moment of
  decision** — that is D198, and it is a fact about the instrument as much as the model.

**The oracle axis vs the final-unroll axis.** `min(rank)==1` over the unroll budget (the
"oracle") is what most older accuracy numbers use. It is inflated relative to final-unroll
scoring by **2.06× (D103), 2.00× (ARC), 5.29× (census)** — the factor is *task- and
budget-dependent and must never be quoted as a constant*. Rule of thumb: oracle ≈ "was the
answer ever available"; final ≈ "did the model open its answer with it". After D166–D174, the
second is mostly a formatting measurement.

**The state lives on a sphere.** RMSNorm terminates every block, ‖h‖ = **76.386** with per-orbit
relative sd ~5e-05. Consequences that come up constantly: drift is excluded *by construction*,
not by measurement; two random points on that sphere are **108.03 ± 0.75** apart, which is the
correct null for any distance claim; and **any injected state is renormalised**, so an
off-sphere donor measures the projection, not your intervention.

**`e` is the map's parameter; `h` is its state.** Not a metaphor — `block_idx` is threaded
through `core_block_forward` and *never enters the block computation* (its only use is the KV
cache update), so the unrolled map is genuinely time-invariant. This is why `e` may be held fixed
across unrolls but **the state may not**: overwriting `h` every unroll pins the trajectory and
measures nothing (`ds_estream` documents this).

## 2. Numbers that are load-bearing, and numbers that only look it

**Load-bearing — if these are wrong, whole sections fall:**
- **0.6677** — D141's rotation threshold. Everything in the regime line is binarised by it.
- **76.386** — the sphere radius; every geometric null derives from it.
- **ρ ≈ 0.79–0.87** — four instruments, but D115 says these are *not* competing estimates of one
  number; ρ is family-level and spans 0.8239–0.9181 across tasks.
- **unroll ~4** — where the answer is settled (D159/D184). Half of today's reinterpretations
  turn on this.
- **position 20 of 53** — where rotation switches on within a prompt (D192). This is what makes
  the steering nulls (D196/D197) about the *site* rather than the direction.

**Looks load-bearing, isn't:**
- Any single "inflation factor" (see above).
- `‖Δ‖ = 2.1947` in the steering runs — banked without `‖e[last]‖`, so α is not expressible as a
  fraction of the parameter's scale. The null survives only because oracle degraded at α = 4,
  proving the dose was adequate.
- The 62.0% figure in D192 — it is the fraction of *positions* rotating in a rotating prompt, not
  a confidence. The load-bearing part is that settling prompts read **0.000**, i.e. no overlap.

## 3. Fragility map — what would break what

- **If 0.6677 were miscalibrated:** D132, D141, D161, D164, D167, D173, D176, D185, D192, D197
  all move. Mitigation already in place: D141 validated it leave-one-bank-out (688/696).
- **If the answer-settles-at-4 finding were wrong:** D198's entire reframing collapses, and
  D157/D158's inflation story weakens.
- **If `contains`-style scoring is chance-level on a family:** D170 and D171 lose their evidence
  there. Guarded by D172's measured cross-item rates (0.042–0.152) — *always report those beside
  a containment number.*
- **What is deliberately not load-bearing:** the interp line (D196/D197). It is one contrast, one
  aggregation, one site. Nothing else depends on it.

## 4. How to read a result here, in order

1. **The P1/gate line first, never the headline.** Two runs today were void on their own gates
   and would have read as findings. `0.000e+00` is the expected value for a no-op patch — anything
   else means the hook writes something other than what the docstring says.
2. **Then the base rate.** A null on a variable with no variance is not a null (D148: `correct`
   False in 432/432). `outcome_variance_scan.py` catches this class.
3. **Then the raw strings**, not the summary. D110 printed a capability verdict over empty
   strings; D193 reported parser artefacts as model behaviour. `grep -a` if the file is CP1251.
4. **Then the chance rate**, for anything containment-like.
5. **Only then the effect size.**

## 5. Failure signatures — recognise these fast

| symptom | almost certainly |
|---|---|
| ERROR, no output file, no traceback in `job attach` | the kernel **refused on its own gate** — check drop counters and `return 1` paths against real data (A47: a length gate that could never pass, 18/18 dropped) |
| ERROR immediately after model load, nothing printed | undefined name / import-time error — `ruff check --select F821`, milliseconds |
| a metric reading exactly 0.000 across every arm | you are measuring your own parser or a floor, not the model |
| an effect that vanishes under a matched control | the arms differed in something you did not name (norm, length, tail) |
| suite fails on "constant defined and never read" | a pre-registered gate was written in the docstring and never wired |

**`job attach` is safe on a running job** (verified: 166 jobs before, 166 after) but **does not
replay a job-side traceback** on an ERRORed one. Do not spend time there.

## 6. Reasoning patterns that repeatedly paid off

- **Register the prediction before the run, both ways.** §N1 predicted A51's null *and* why it
  would be uninformative; that is the only reason D197 is quotable as a bounded claim rather than
  an over-read. Every time I did this it constrained the interpretation usefully.
- **When a control fails, the control is usually right.** D164's headline died to its own
  registered control within three hours. A49 died to its own gate. Both were correct to fail.
- **Re-analysis of banked data outranks new GPU.** D138, D141, D146, D152, D154, D157–D159, D163,
  D164, D165, D168, D174, D177, D178, D179, D181, D182, D194, D195 — all zero-GPU, all from data
  already on disk. **Check the bank before launching.**
- **Read the source, not the paper, for protocol.** D183 pinned the ARC comparison from
  `--num_fewshot=0` in their README; D187 corrected our own §6.9 from their sampler code.
- **When two of my findings disagree, look for the window/site/scale they differ in** before
  concluding either is wrong. That is how D198 and D197 both got their scope lines.

## 7. Operational facts that cost time to learn

- **DataSphere serialises concurrent jobs.** Four in flight meant ~1 h wall for 60 s of compute.
  Launch 2–3, not 5.
- `GRPC_DNS_RESOLVER=native`, launch from the config dir, weights dataset `bt102r0j5cb8r6r6nb36`
  mounts in 7.7 s vs 262–282 s to download.
- **A monitor only survives as the top-level command of its own background call.** `nohup … &`
  and `( … ) &` inside another command both die silently (RC7). One monitor, listing every live
  id, re-armed as a set.
- **Gate every push on a green suite.** I pushed red twice today by running pytest and then
  committing unconditionally.
- `local_smoke.py` builds the **real** `RavenForCausalLM` at 27.7M params in ~1 s. Any new
  intervention pattern should be made to pass there **before** it goes in a kernel.
- **CP1251 files are invisible to plain `grep`.** The draft `main.tex` is CP1251; use `grep -a`.

## 8. Things I believe but have not established

*Kept separate deliberately — none of these is in the ledger as a claim.*

- The exemplar benefit probably lives in **attention over the exemplar tokens**, not in `e`. It
  would explain why a length-matched pad reproduces context's *cost* (D186) but not its benefit,
  and why steering `e` reproduces neither (D196).
- The regime is probably **set by the prelude and then merely expressed** by the recurrence —
  D192's sharp positional boundary and D198's late onset both point that way, but nothing tests
  the prelude directly.
- Their word problem probably **does not orbit at any period** on this checkpoint (A49's first
  pass showed ~10× less non-DC power than our settling prompts), which would strengthen D185.
  A49's second pass is the test.
- `parity8` getting *worse* with depth (D171) is probably the S₂-impossibility (D189b) showing up
  as the model committing harder to a wrong constant, not as noise.
