> **FROZEN 2026-08-05.** Archived under CLAUDE.md §6 (three live documents only).
> Not maintained. Superseded by `docs/UNDERSTANDING.md`, `claims_ledger.md`, `directions.md`.
> Statements here may be contradicted by later work — several assert things since disproven
> (e.g. "H1 was never tested": it was, and the test itself was invalid — see D56, D65).

# Exit criteria: Huginn's own vs. what this project used

Verified 2026-07-19 against `raven_modeling_minimal.py` at the pinned revision
`bb6621b65e90b6a4b9b29ef88dc83866d450470c`, and against every call site on
`main`, `origin/two-scale-real-fix` and `origin/pr-real-data` (scripts and
notebooks).

## Huginn's built-in exit criteria

Passed as `criterion=` to `model.generate_with_adaptive_compute(...)`.
Resolved by `get_adaptive_exit_evaluator(model, criterion, exit_threshold)`.

| criterion | measures | default threshold (`exit_threshold="auto"`) | needs coda pass? |
|---|---|---|---|
| `latent-diff` | relative latent change: `‖h(k) − h(k−1)‖ / ‖h(k)‖`, averaged over positions | `0.03` | no — geometric |
| `cosine` | `1 − cos(h(k), h(k−1))` between successive latent states | `1e-3` | no — geometric |
| `entropy-diff` | absolute change in output-distribution entropy | `1e-3` | yes |
| `kl` | KL divergence between successive output distributions | `0.001` | yes |
| `minp-kl` | same as `kl`, min-p filtered | `1e-5` | yes |
| `argmax-stability` | number of consecutive steps the argmax token is unchanged (exits when `>=` threshold) | `5` steps | yes |
| `none` | never exits — `NoOpExitEvaluator`, returns an all-`False` mask unconditionally | — | no |

Two groups:

- **Geometric** (`latent-diff`, `cosine`) — read the trajectory directly, no
  decoding required, so they are cheap per step.
- **Output-space** (`entropy-diff`, `kl`, `minp-kl`, `argmax-stability`) —
  require running the coda to obtain logits at each checked step, so they
  cost more but measure what the model actually predicts.

Relevant extra knobs on the same call: `exit_threshold`, `min_steps`,
`check_criterion_every_n_steps`, `do_not_exit_in_prefill`, `exit_evaluator`
(plug in a custom `PerIterationExitEvaluator`).

**`criterion="none"` is the signature default.** The model's own source
comments it as: *"adaptive compute is off by default, turn on by choosing an
exit criterion."*

## What this project actually used

**No call site anywhere passes `criterion`.** Verified across all three
branches, in both `scripts/*.py` and `notebooks/*.ipynb`. Every invocation of
`generate_with_adaptive_compute` therefore runs with `NoOpExitEvaluator`, so
`exit_reached` stays all-`False`, the `break` at the end of the recurrence
loop is unreachable, and the loop runs `for compute_step in range(max_steps)`
to completion.

Depth is instead a hardcoded constant everywhere:

| script / notebook | `num_steps` |
|---|---|
| `run_pararule`, `run_counting`, `run_dissociation`, `run_dissociation_multiinit`, `run_switch`, `run_maxtask` | 64 (fixed) |
| `run_accuracy` | 32 for generation, 64 for the trajectory |
| `run_forceloop` | grid `(16, 24, 32, 64)` |
| `run_phase` | grid `(10, 14, 16, 18, 20, 24)` |
| `run_two_scale_{depth,real,counting}` | 32 (fixed) |
| `notebooks/01_mvp.ipynb` | 16 / 32 / 64 inline |
| `notebooks/02_two_scale.ipynb` | 32 |

## The project's own hand-rolled settling detectors

None of these ever stopped computation. All are post-hoc measurements on an
already-complete fixed-length array.

| metric | where | definition | used as an exit? | used in a headline? |
|---|---|---|---|---|
| `steps_to_settle` | `metrics/dynamics.py` (main) | first `k` where `‖Δ(k)‖ < 0.1 × max‖Δ‖` for that trajectory | no | yes — the `steps~n_ops` results |
| `exit_step_by_acceleration` | `metrics/two_scale.py` (two-scale) | first `k` where acceleration `< 0.1 × max`, sustained 2 consecutive steps | no | no — committed as `exit_answer` in `band.csv`, unused |
| `exit_step_by_norm` | `metrics/two_scale.py` (two-scale) | first `k` where step norm `< 0.1 × max` | no | no — committed as `settle_answer` in `band.csv`, unused |

## Notes worth carrying

1. **`steps_to_settle` vs `latent-diff` differ in the denominator.** The
   project normalizes by that trajectory's *maximum* step — which is almost
   always the first step, driven by the random `h_0` initialization. Huginn's
   `latent-diff` normalizes by the *current* latent norm. So the project's
   metric is anchored to an initialization artifact; the model's is not.

2. **`cosine` vs the two-scale `orthogonality` are near-siblings.** Huginn:
   `1 − cos(h(k), h(k−1))` on consecutive *states*. Two-scale branch:
   `cos(Δ(k), Δ(k−1))` on consecutive *step vectors*. Closely related
   quantities — one ships with a validated threshold and a working exit
   mechanism, the other was reimplemented as a descriptive statistic.

3. **`settle_answer` is not inert.** It is committed in `band.csv` and never
   used, but it differs significantly by depth (length-controlled `hi`
   coefficient −3.96, p = 3.7e-26), and adding it as a control shrinks the
   band-test headline from +1.2974 to +1.1301 (p = 2.9e-14).

4. **Accuracy was measured at one depth only.** `run_accuracy.py` uses
   `num_steps=32`. No `num_steps` sweep against correctness exists anywhere;
   `README.md` itself lists "полный accuracy-vs-depth" among the
   unimplemented items. So "0% accuracy at length ≥ 8" means "0% at 32
   unrolls", not "0% at any depth".

## The experiment this suggests

Run counting accuracy under each of the six built-in criteria, recording both
accuracy and mean steps consumed, against a fixed-depth baseline sweep
(`num_steps` ∈ {8, 16, 32, 64, 128}). That answers two open questions at once:
whether the 0%-at-length-8 result is a capability ceiling or a compute-budget
artifact, and whether any stopping rule beats an arbitrary constant. If a
geometric criterion (`latent-diff`, `cosine`) wins, that is direct evidence
the trajectory geometry carries usable information — a far stronger claim than
correlating geometry against a length-confounded difficulty label.

Caveat: selecting the criterion that maximizes accuracy on the same data used
to evaluate it is overfitting. Split by task instance, or report it as tuned.
