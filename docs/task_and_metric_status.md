# Where the tasks and the topological metrics actually stand

Written 2026-08-04 because the ledger had grown to 55 claims about contraction
rates and probes while the *base layer* — do the tasks work, do the topological
metrics mean anything — had never been summarised in plain terms. Everything below
is read off `results/*.csv`, not recalled.

---

## 1. The three original hypotheses

| | hypothesis | status |
|---|---|---|
| **H1** | Trajectories fall into **settle / loop / drift** regimes | **Vacuous as operationalised.** The three regimes have statistically indistinguishable winding: settle 0.6618 ± 0.0047 (n=27), loop 0.6606 ± 0.0054 (n=31), drift 0.6595 ± 0.0003 (n=2). Kruskal–Wallis **p = 0.48**; between-regime spread 0.0023 against within-regime sd 0.0041, **ratio 0.57**. Worse, *every* trajectory has winding ≈ 0.66 regardless of label, so the labels partition the data without separating it. `drift` has **n = 2**. |
| **H2** | Winding ∝ reasoning depth | **Dead as a metric, alive as a phenomenon.** Winding is governed by the recording budget — its effect reverses sign with `num_steps` (D28). But the Jacobian spectrum shows the state genuinely rotates (D55): leading eigenvalue complex in 3/3 prompts, period ≈2.6–6.0 unrolls. The rotation survives ~4 turns and is sampled at ~3 points per turn against a Nyquist floor of 2, so winding over a 64–128-unroll window was measuring aliased noise. |
| **H3** | Contraction (ρ<1) forbids unbounded counting | **Premise measured, consequence not tested.** ρ ≈ 0.80 by Arnoldi (D31), 0.705→0.858 across 14 weight-sets by two-orbit convergence (D52). The *implication* for counting capacity was never turned into a test. |

---

## 2. Tasks: what exists, and what was measured on it

Ten task generators exist in `src/traj_geom/shapes/synthetic.py`. **The critical
column is the last one.**

| task | geometry measured? | **accuracy measured?** |
|---|---|---|
| `make_counting_task` (count ones) | yes — `counting.csv`, 30 rows | **yes** — `counting_accuracy.csv`, 56 rows |
| `make_switch_task` (parity) | yes — `switch.csv`, 60 rows | **no** |
| `make_max_task` (running max) | yes — `maxtask.csv`, 48 rows | **no** |
| `make_projection_task` | yes — in `manifold_null.csv` | **no** |
| `make_three_scale_task` | yes — `three_scale.csv`, 180 rows | **no** |
| `make_three_scale_modk_task` | yes — `three_scale_modk*.csv`, 396 rows | **no** |
| `make_running_count_task` (Barannikov a) | yes — `register_analysis.csv` | **no** |
| `make_nesting_depth_task` (Barannikov b) | yes — `register_analysis.csv` | **no** |
| `make_count_ones_task` | yes — `dissociation*.csv`, `forceloop.csv` | **no** |
| ParaRule (external) | yes — `pararule.csv`, 40 rows | **no** — the `label` column is ground truth, not model output |

### The structural gap, stated plainly

**Geometry was measured on eight task families. Model correctness was measured on
one.** So for eight of ten tasks we do not know whether the model can do the thing
whose "reasoning trajectory" we were characterising.

This is not a small caveat. Where accuracy *was* measured, it is close to zero at
any interesting difficulty:

- counting: 37.5% at n_ops=2, 25% at n_ops=4, **0% at n_ops = 8, 16, 32, 48**
  (one 12.5% blip at 24); overall 10.7% carried entirely by the easy cells
- `v6_correctness_probe.csv`: the model is ever exactly correct in **4 / 24** cases
  (16/24 for top-k)
- the causal kernel, independently: **0 / 120** at M=64

If the model cannot perform the task, its latent trajectory is the geometry of a
model *failing*, which is a legitimate object of study but is **not** "the geometry
of reasoning". Every geometric result on the eight unmeasured tasks inherits this
ambiguity.

---

## 3. Topological / geometric metrics

| metric | what was actually computed | verdict |
|---|---|---|
| **winding number** | thousands of rows across 10 files | **Retire as evidence.** Window-governed, sign flips with `num_steps` (D28); near-aliased relative to the true rotation (D55). Keep as a worked example. |
| **persistent homology H₁** | `homology.csv` — **15 trajectories**, 2 kinds, 3 difficulty levels. Loop counts 0–26, 9/15 nonzero | **Uninterpretable, not negative.** No null, no surrogate, no follow-up run. A random walk on a sphere also has H₁ cycles at some scale; without a surrogate the counts mean nothing. The single least-developed metric in the project. |
| **settle/loop/drift regime label** | `phase.csv` (216 rows), `h2_loops.csv` (60) | **Vacuous** — see H1 above. The labels do not separate the data. |
| **chord-to-arc ratio** | `classify_shape_regime` in `metrics/regime.py` | Sound construction (sampling-density invariant) but shares winding's window dependence. |
| **`steps_to_settle`** | 8 files | **Retire.** Thresholds at 10% of the maximum step, which is always the `h₀` transient, making it a function of ρ alone (D24(3)). |
| **TwoNN intrinsic dimensionality** | implemented, barely used | Unexploited. Participation ratio did the work instead (D48) and was more informative. |
| **Jacobian spectrum** | `kaggle_jacobian` — 3 prompts | **The most informative metric per unit cost, and the least used.** Magnitudes answered ρ (D31); the arguments answered rotation (D55) and had been ignored for a week. n=3 — a proper sweep is overdue. |

---

## 4. What this implies for what to do next

Ordered by how much they would change the picture, not by effort.

1. **Measure accuracy on the eight tasks that never had it.** Cheap — it is one
   generation pass per task — and it decides whether eight tasks' worth of geometry
   describes reasoning or failure. Nothing else on this list matters as much.
2. **Screen for a task the model can actually do** (Caesar, running). Without one,
   "geometry of reasoning" has no positive instance anywhere in the project.
3. **Give persistent homology a null, or drop it.** 15 trajectories with no
   surrogate is not a result in either direction. It is the last untested topological
   claim, and H₁ was the most distinctive thing the project set out to look at.
4. **Sweep the Jacobian spectrum properly** (n=3 → tens of prompts, both arms).
   Best value per unit cost in the project.
5. **Retire H1's regime labels** or re-operationalise them on a quantity that
   separates the classes. As they stand they are a partition, not a finding.
