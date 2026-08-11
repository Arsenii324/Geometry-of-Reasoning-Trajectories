# Appendix B content — the full withdrawal/amendment list

*Generated from `claims_ledger.md` on 2026-08-11 17:40. Ten rows carry an explicit marker.
Drop-in replacement for the Appendix B placeholder in `draft/main.tex`. Every entry states what
fell and what survived, because in every case something survived.*

**Earlier work (2026-08-04), inherited:**

1. **D34 — superseded by D50.** The direction it measured is 97% the current-token contrast, so
   `register_r` was a window artefact. *The register claim itself survives; the mechanism
   described in this row does not.*
2. **D36 — superseded by D50.** Same cause: the ℤ-action results describe token identity rather
   than the claimed structure.
3. **D43 — retracted.** Was: *"the contraction rate predicts how deep Huginn can usefully
   think."* The direct measurement in D44 flips the verdict.
4. **D51 — superseded by D52 (same day).** Partial: 4 of 8 checkpoints and n = 1 untrained.
   *Its numbers are subsumed; its caveat about the early jump was confirmed.*

**The regime line:**

5. **D137/D138 — retracted and replaced by D141.** Their "gapped, therefore two regimes" test
   used a uniform null, which any clustered distribution beats, so it never tested the claim
   made. Checked against the labels, the gap splits the *settling* population. **The replacement
   is stronger:** a single threshold reproduces the binary label on 691/696 orbits and 688/696
   leave-one-bank-out.
6. **D135/D136 — superseded 2026-08-11.** Their headline refuted D134's "by elimination"
   inference. Three later rows overturn that: D155 (their design cannot resolve a separation
   below Cohen's d ≈ 10), D161 (editing one `wte` row flips the regime — the inference confirmed
   by manipulation), D165 (a probe of this class fails on a label it is *guaranteed* to contain).
   *What survives: five held-out replications, and the finding that `set` and `word` are mixed
   rather than deterministic.*
7. **D164 — amended by D167, its own registered control.** The decisive matched pair
   (`symbol`↔`array`) does not survive norm-matching: the excursion vanishes, 0 crossings,
   min R 0.768 against the plain chord's 0.400. *The measurements replicate bit-for-bit; the
   interpretation in the headline is what fails. A narrower claim survives on a second chord.*

**The evaluation line:**

8. **D148 — the correctness half is vacuous and withdrawn.** `correct` is False in **432 of 432**
   orbits, so there was no variance for the regime to change. *The instrument null, the
   sensitivity control, the `best_depth` null with its 4.7-unroll floor, and the ±1 rank effect
   all stand. Re-tested on an outcome that varies (D176), the conclusion returned: 1 of 18
   paired units, p = 1.0.*
9. **D162 — headline wrong, amended by D169 and D175.** It refuted "protocol" after varying only
   the *scoring rule* at fixed prompt. **Prompt format was the whole story and was never
   varied.** *Its own numbers are unchanged and its arms replicate.*
10. **D193 — amended the same day by reading its own raw output.** Its `parsed_exact` column
    measured the parser, not the model: re-scored on first-line containment the arms read
    0.444 / 0.528 / 0.389 / 0.389 against the reported 0.000 / 0.000 / 0.056 / 0.028 — the
    harness arm off by 14×. *The conclusion survives on the repaired metric: paired against
    bare, no arm wins.*

**Interpretations narrowed without the row being withdrawn:** D168(2) (arithmetic is *not*
"uncertain about the answer" — it computes in prose, D172), D177(4) (depth tracks where the
answer starts, not the kind of operation — D178), and D196/D197 (bounded to "steering the last
position", not to orthogonality of directions — §N1).

**Two runs were voided by their own preregistered gates rather than by later work:** A49's first
pass (spectrum tail 48 vs `rotation_power`'s tail 24 — the gate refused it) and A43's `E_reason`
arm (never executed; the job hit its wall budget and banked cleanly).
