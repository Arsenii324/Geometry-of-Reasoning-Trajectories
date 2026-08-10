# Experiment index — GENERATED, do not hand-edit

Regenerate with `uv run python scripts/build_index.py`. Source of truth is
`claims_ledger.md`; this file only cross-references it against the code, the runs
and the banked data, so that no experiment is unfindable.

**159 claims · 33 distinct jobs referenced · 61 kernels · 3261 banked artifacts across 92 directories**

| row | claim (first sentence) | kernel | job | analysis | banked |
|---|---|---|---|---|---|
| **D1** | README's reproduce block: uv sync ; uv run pytest ; uv run ruff check . | — | — | — | — |
| **D2** | 11 tests pass / 1 skip on main with the correct setup | — | — | — | — |
| **D3** | trajgeom.data.loaders exists somewhere and produced the committed PARARULE CSVs | — | — | — | — |
| **D4** | "David" authored the fabricated two-scale numbers and the original per-position metric code | — | — | — | — |
| **D5** | Repo activity has stopped | — | — | — | — |
| **D6** | pr-real-data restates the two-scale band-test coefficient as clean and final | — | — | — | — |
| **D7** | analysis/correlate.py's significance-threshold table is fully correct | — | — | — | — |
| **D8** | The project has 3 active contributors | — | — | — | — |
| **D9** | No paper-drafting artifact exists yet | — | — | — | — |
| **D10** | The length-confound control (partialspearman against seqlen) is a working control wherever it's applied | — | — | — | — |
| **D11** | makethreescaletask (the fix for D10) — first real result, 2026-07-23: does \ | — | — | `run_three_scale.py` | — |
| **D12** | The extracttrajectory(returnlogits=True) coda-reconstruction self-check (validatelogits=True) — does it actually pass on real hardware? | — | — | `diag_v6_token_gap.py`, `run_v6_correctness_probe.py` | — |
| **D13** | Project-wide BH-FDR correction (projectplan.md §0.2/§15 item 1) — do the project's 50 uncorrected correlation tests survive multiple-comparisons corre | — | — | `run_fdr_correction.py` | — |
| **D14** 🔻 | Phase 1.2 positive control (projectplan.md §Phase 1.2) — does this project's extraction pipeline see genuine winding loops at all under Blayney et al. | — | — | `diag_blayney_repro.py` | — |
| **D15** | First genuinely clean (unconfounded) H2 test — makethreescalemodktask sweep, totallen constant by construction (unlike D11's threescale.csv): does \ | — | — | `run_three_scale_modk.py` | — |
| **D16** | Cross-task synthesis (the strongest single anti-H2 datapoint): does \ | — | — | — | — |
| **D17** | The honesty headline: is the model even solving the counting task at the depths where winding/stepssettle "rise with reasoning depth"? | — | — | — | — |
| **D18** | Cross-task pattern in stepssettledifficulty: is the sign explained by whether a task's answer accumulates vs. | — | — | — | — |
| **D19** | Init-robustness ranking: which trajectory metrics are driven by the computation vs. | — | — | — | — |
| **D20** | Two structural facts a truly-clean report must disclose: (a) where the Blayney loops actually sit in the prompt; (b) a residual confound in the "clean | — | — | — | — |
| **D21** | Adversarial audit of the project-state recap itself (2026-07-25): do the summary claims survive independent re-derivation? | — | — | — | — |
| **D22** | The winding metric adjudicated against a matched-random-walk null — the foundational check the project had declared impossible. | — | — | `run_winding_null.py` | — |
| **D23** | The recurrent state is architecturally confined to a sphere — which invalidates the rotation-center, the null model, and the "two-phase" reading of th | — | — | — | — |
| **D24** 🔻 | Whole-project rigor re-audit: metrics, hyperparameters, arbitrary pre/post-processing, extraction settings, task text, tokenization, statistical assum | — | — | `backfill_provenance.py`, `run_mvp.py` | — |
| **D25** 🔻 | Second audit pass: an arithmetic error of mine retracted, the winding null REVERSED under a valid surrogate, H1 found to have never been tested, and t | — | — | `run_manifold_null.py` | — |
| **D26** | H2 tested without any surrogate model, converged-state answer-clustering tested, the last two unaudited metrics closed, and a derivation of what the n | — | — | `run_winding_permutation.py` | — |
| **D27** | The project's first positive signal about computation (SUGGESTIVE, NOT ESTABLISHED): the answer is linearly decodable from the converged state at fixe | — | — | `run_answer_probe.py` | — |
| **D28** | RESOLVED: there is no rotational signal in winding. | — | — | `run_manifold_null.py` | — |
| **D29** | Plan Step 5 executed: the invalidated metrics re-derived floor-aware wherever raw paths survive — and the sign of cos is shown to be set by the RECORD | — | — | `run_regime_rederivation.py` | — |
| **D30** | GATE PASSED: the post-convergence residual IS arithmetic — confirmed on real GPU across three dtypes. | `kaggle_precision_gate` | *geometry-precision-gate* | — | kaggle_precision_gate: 1 |
| **D31** | ρ(∂ₕR) measured EXACTLY on real Huginn for the first time — the quantity A2/D21 record as never measured. | `kaggle_jacobian` | *geometry-jacobian-spectrum* | — | kaggle_jacobian: 1 |
| **D32** | Barannikov's Task a and Task b, run: the register content IS linearly decodable from the latents for both — but the winding readout shows NO quantisat | `kaggle_register` | *geometry-register-probe* | — | kaggle_register: 1 |
| **D33** | H2 restated behaviourally: required recurrent depth DOES scale with difficulty (rho=+0.626, p=0.0014) — but on a length-confounded task, exactly as pr | `kaggle_depth` | *geometry-depth-fixed-length*, *geometry-depth-threshold* | `run_depth_threshold.py` | kaggle_depth: 1 |
| **D34** 🔻 | SUPERSEDED BY D50 (2026-08-04): the direction is the current-token contrast and registerr is a window artefact. | — | *geometry-position-states* | — | — |
| **D35** | SCOPED BY D54 (2026-08-04): the effect is in the answer process, not in when the count becomes decodable; and the untrained control that deflated D40/ | `kaggle_depth_fixed` | *geometry-depth-fixed-length* | — | kaggle_depth_fixed: 1 |
| **D36** 🔻 | SUPERSEDED BY D50 (2026-08-04): computed along a direction that is 97% the current-token contrast, so the ℤ-action results describe token identity, no | `kaggle_states` | — | — | kaggle_states: 99 |
| **D37** | A sharp dissociation: the counting register's CONTENT is fully present after ONE unroll and does not grow with depth, while its DIRECTION rotates mono | `kaggle_register_depth` | *geometry-register-vs-depth* | — | kaggle_register_depth: 27 |
| **D38** | What the extra depth is FOR: readout. | `kaggle_readout` | *geometry-readout-vs-depth* | — | kaggle_readout: 5 |
| **D39** | "Geometry encodes the computation" is TRUE — at the answer token, across prompts. | `kaggle_readout` | — | — | kaggle_readout: 5 |
| **D40** | THE CONTROL THAT SHOULD HAVE BEEN RUN FIRST: an UNTRAINED model with random weights reproduces the project's positive findings, most of them MORE stro | `kaggle_baseline` | *geometry-untrained-baseline* | — | kaggle_baseline: 2 |
| **D41** 🔻 | Decodability and capability move in OPPOSITE directions: the untrained model decodes the count 20× more precisely than the trained one and cannot coun | `kaggle_rho_direct` | *geometry-readout-vs-depth*, *geometry-untrained-depth* | — | kaggle_rho_direct: 2 |
| **D42** | Training SLOWS the contraction, ρ ≈ 0.66 → 0.91, lengthening the depth time-constant 2.4 → 10.6 unrolls. | — | — | — | — |
| **D43** 🔻 | RETRACTED 2026-08-04 (see (6)). | — | — | — | — |
| **D44** 🔻 | D42 CONFIRMED ON THE OPERATOR: the untrained map contracts at ρ=0.7150±0.0069, the trained at ρ=0.8866±0.0362. | — | *geometry-rho-direct* | — | — |
| **D45** 🔻 | HEADLINE RETRACTED 2026-08-04, same day, after adversarial review. | — | — | — | — |
| **D46** 🔻 | D40 CONFIRMED at matched n=220, and the omitted control comes out AGAINST the deflationary reading: the register direction is NOT the tokenizer's embe | `kaggle_baseline` | *geometry-untrained-baseline* | — | kaggle_baseline: 2 |
| **D47** | The causal patching experiment is BROKEN, not null — the intervention never reached the output. | `kaggle_causal` | *geometry-causal-patch* | — | kaggle_causal: 2 |
| **D48** | The geometric form of D41: the untrained model's count representation is literally ONE-DIMENSIONAL (participation ratio 1.0, PC1 correlates +0.999973  | — | — | `plot_regimes.py` | — |
| **D49** | Per-instance, the trained model resolves the exact integer count for 74/220 prompts and the untrained for 220/220 — and 99 trained instances sit insid | — | — | `plot_regimes.py` | — |
| **D50** | THE COUNTING REGISTER IS REAL — cv R² = 0.782 for the lagged count after strict controls — BUT D34/D36 FOUND THE WRONG DIRECTION. | — | — | `recheck_register_window.py`, `run_register_analysis.py` | — |
| **D51** 🔻 | SUPERSEDED BY D52 (same day): partial, 4 of 8 checkpoints and n=1 untrained. | — | *geometry-rho-ckpt-a*, *geometry-rho-direct* | — | — |
| **D52** 🔻 | D44's pseudoreplication is CLOSED on both sides. | — | *geometry-rho-ckpt-a*, *geometry-rho-seed-a* | — | — |
| **D53** | THE COUNTING REGISTER IS ARCHITECTURAL. | — | *geometry-register-correct* | `recheck_register_window.py` | — |
| **D54** | D35's depth-scaling is NOT a decodability effect: the depth at which the count becomes linearly decodable does not scale with difficulty in EITHER arm | — | — | — | — |
| **D55** 🔻 | THE TRAJECTORY DOES ROTATE — the Jacobian's leading eigenvalue is complex in 3/3 prompts (27/30 top modes complex), with rotation period ≈2.6–6.0 unro | `kaggle_jacobian` | — | — | kaggle_jacobian: 1 |
| **D56** | H1's settle/loop/drift regimes are VACUOUS: the three classes have statistically indistinguishable winding (Kruskal-Wallis p=0.48), and every trajecto | — | — | — | — |
| **D57** | The Caesar screen is 0% in every cell — but the trained model answers almost every prompt with the memorised pangram "The quick brown fox jumps over t | — | *geometry-caesar-screen* | — | — |
| **D58** | The log-linear fit used for every ρ in this project is biased +0.033 at Huginn's own ρ and rotation period — an order of magnitude worse than the +0.0 | `kaggle_eps_sweep` | *geometry-eps-sweep* | — | kaggle_eps_sweep: 2 |
| **D59** | ρ IS STEERABLE by a gradient-free static rescale — paired slope +0.2853 ± 0.0450, r=+0.913, p=2.3e-04, 9/12 prompts monotone. | — | *geometry-eps-sweep* | — | — |
| **D60** | Prompt format matters enormously — a trivial copy task goes 15% → 100% under Huginn's own chat template — but it does NOT rescue counting or Caesar, w | — | *geometry-chatfmt* | — | — |
| **D61** | When Huginn cannot do a task it does not fail randomly — it RETRIEVES something adjacent, by one of at least seven distinct routes, and which route fi | — | *geometry-caesar-instrumented*, *geometry-caesar-screen*, *geometry-chatfmt* | — | — |
| **D62** | The task-accuracy run is INVALID: batchedgenerate returned an empty string for every prompt. | `kaggle_taskacc` | *geometry-task-accuracy* | — | kaggle_taskacc: 2 |
| **D63** | The attn-vs-mlp split is UNDERPOWERED and its own pre-registered gate says do not interpret it: the reproduction arm missed D59 by 0.095, larger than  | — | *geometry-eps-split* | — | — |
| **D64** | THE FIRST TOPOLOGICAL POSITIVE: H1 loop COUNT beats a manifold-matched null in 9/9 trajectories that have any loops at all (real 8.7 vs surrogate 1.9) | — | — | `run_homology_null.py` | — |
| **D65** | H1, H2 and H3 ALL RETURN TO OPEN. | — | — | — | — |
| **D66** | The activation budget was never the problem, and continuouscompute CANNOT WORK in the uncached loop. | `kaggle_memprobe` | *geometry-mem-probe*, *geometry-prompt-depth* | — | kaggle_memprobe: 2 |
| **D67** | REPRODUCTION AUDIT: with the banked raw trajectories restored, the three analysis CSVs that depend on them re-derive from scratch — manifoldnull.csv a | — | — | — | — |
| **D68** | THE MODEL KNOWS ANSWERS IT NEVER EMITS, AND DEPTH CONVERGES TO A DISCOURSE CHOICE. | `kaggle_graded` | *geometry-discourse*, *geometry-graded-readout* | — | kaggle_graded: 2 |
| **D69** 🔻 | (A) DISCOURSE CONFIRMED, BUT BY THE ARM I DID NOT NOMINATE. | `kaggle_discourse` | *geometry-correctness*, *geometry-discourse*, *geometry-graded-readout* | — | kaggle_discourse: 2 |
| **D70** | 6.1's CONFOUND IS NOT SUPPORTED: the content gap is 0 even where the trained model is 100% capable — but the content probe is AT CEILING in both arms, | `kaggle_capgraded` | *geometry-cap-content*, *geometry-cap-graded* | — | kaggle_capgraded: 2 |
| **D71** | THE READOUT UNDER D68/D69 IS EXACT: my per-unroll decode reproduces the model's own logits to max | `kaggle_headcheck` | *geometry-head-check* | — | kaggle_headcheck: 2 |
| **D72** 🔻 | B6 IS VOID AS DESIGNED: four cells cleared Bonferroni and NONE can be read, because correctness is very nearly a function of the ANSWER VALUE. | `kaggle_geomcorrect` | *geometry-correctness* | — | kaggle_geomcorrect: 2 |
| **D73** | THE CONTENT NULLS WERE A TARGET-SELECTION ARTIFACT. | `kaggle_nonlinear` | *geometry-nonlinear-content* | — | kaggle_nonlinear: 2 |
| **D74** | THE ORBIT IS NOT LOW-DIMENSIONAL, AND THAT EXPLAINS WHY EVERY PLANAR TRAJECTORY METRIC IN THIS PROJECT FAILED. | — | `bt1mvf3juvrcjpl6kl1i` ⚠unregistered | `run_effective_dim.py` | — |
| **D75** | THE MODEL DOES FAR MORE THAN THIS PROJECT THOUGHT, AND B6 IS UNBLOCKED. | `kaggle_battery` | *geometry-battery* | `run_battery.py` | kaggle_battery: 2 |
| **D76** | TRAINING MAKES THE LATENT PATH GLIDE INSTEAD OF ZIG-ZAG, AND CONFINES IT TO A BOUNDED SUBSPACE. | — | `bt1hd3oqb17690amgolg` ⚠unregistered | `run_arm_geometry.py` | — |
| **D77** | HOW FAR bf16 STORAGE SUPPORTS TRAJECTORY GEOMETRY, MEASURED RATHER THAN ASSUMED: exactly to the unroll where the step norm crosses the rounding scale, | — | *geom-bank* | — | — |
| **D78** | THE PER-UNROLL RANK READOUT HAS AN UNCONTROLLED SOURCE OF RUN-TO-RUN VARIANCE: initializestate draws h0 from an UNSEEDED RNG at a shape that depends o | — | *geometry-b6-bank*, *geometry-battery* | — | — |
| **D79** | B6 ANSWERED, AND THE ANSWER IS A BOUNDED NULL: at matched answer value, correct and incorrect trajectories do NOT differ in effective dimensionality,  | — | *geometry-b6-bank* | `run_b6.py` | — |
| **D80** | TRAJECTORY SHAPE IS MOSTLY A CLOCK: the geometry statistics are governed by HOW FAR THE CONTRACTION HAS RUN, not by the input. | — | *geometry-geomcap* | `run_window_law.py` | — |
| **D81** | A LINEAR CONTRACTION FROM A RANDOM START REPRODUCES THE ORBIT'S DIMENSIONAL COLLAPSE BUT NOT ITS ROTATION RATE. | `ds_bank`, `ds_eigen` | — | `run_surrogate.py` | ds_bank: 97, ds_eigen: 1 |
| **D82** | ONE UNROLL IS A LINEAR MAP ON THE STATES THE ORBIT VISITS, so D52 and every Jacobian quantity stand. | — | — | `run_linearity.py` | — |
| **D83** | H2 IS NOT SUPPORTED BY ROTATION EITHER: the paired track-minus-local difference in rotation per unroll does not scale with reasoning length (rho = +0. | `kaggle_h2rot` | *geometry-h2-rotation* | `run_h2_rotation.py` | kaggle_h2rot: 2 |
| **D84** | THE ORBIT'S SHAPE IDENTIFIES WHICH PROMPT IS BEING PROCESSED, AT CEILING -- AND STILL DOES NOT CARRY CORRECTNESS. | — | — | `run_shape_decode.py` | — |
| **D85** | NO GEOMETRIC STATISTIC TRACKS CAPABILITY ACROSS 21 TASK FAMILIES SPANNING 0% TO 100% -- AT ANY OF FIVE WINDOWS -- WHILE THE SAME STATISTICS ON THE SAM | — | *geometry-geomcap* | `run_geomcap.py` | — |
| **D86** | DEPTH DOES NOT DESTROY THE ANSWER, IT WRAPS IT IN PROSE -- measured on DECODED STRINGS across 21 families, not on rank. | — | *geometry-depthacc* | `run_depth_accuracy.py` | — |
| **D87** | THE ORBIT'S SHAPE SEPARATES TWO COMPUTATIONS AT AN IDENTICAL TOKEN COUNT, at 96.9-100% balanced accuracy. | `kaggle_lenmatch` | *geometry-lenmatch* | `run_shape_decode.py` | kaggle_lenmatch: 190 |
| **D88** | THE SHAPE READS THE INPUT TOKEN, NOT THE COMPUTATION -- the control I expected to confirm the opposite. | `kaggle_marker` | *geometry-marker* | `run_shape_decode.py` | kaggle_marker: 218 |
| **D89** | A MEASUREMENT DEFECT IN THE CAPABILITY AXIS ITSELF: correct scores the FIRST TOKEN of gold, and 8 of 21 families have multi-token golds -- 4 of them f | `kaggle_geomcap` | *geometry-depthacc*, *geometry-lenmatch* | — | kaggle_geomcap: 610 |
| **D90** | WHETHER HUGINN ANSWERS A FIXED PROMPT CORRECTLY IS PARTLY DECIDED BY ITS OWN RANDOM INITIAL LATENT. | `kaggle_geomcap` | *geometry-geomcap* | — | kaggle_geomcap: 610 |
| **D91** | A LEAD, NOT A FINDING, AND THE MOST CONSEQUENTIAL ONE THE PROJECT HAS: correctness may be decodable from the orbit's shape in the TRANSIENT and nowher | — | *geometry-geomcap*, *geometry-h0bank* | `run_h0_within.py` | — |
| **D92** | A STALE-CSV BUG SURFACED A REAL, REPRODUCIBLE, WEAK CORRECTNESS SIGNAL THAT D84's ORIGINAL TEST COULD NOT SEE: shape decodes correctness at 61.0% bala | — | *geometry-h0bank* | `run_correctness_decode.py`, `run_h0_within.py` | — |
| **D93** | D91's LEAD DOES NOT REPLICATE IN THE POWERED, PRE-REGISTERED TEST -- AND THE NULL IS TRUSTWORTHY BECAUSE THE INSTRUMENT'S OWN DETECTION FLOOR WAS ITSE | — | *geometry-h0bank* | `run_h0_within.py` | — |
| **D94** | H3 IS MEASURED, FOR THE FIRST TIME, WITH THE ESTIMATOR H3 IS ACTUALLY ABOUT -- AND THE MAP CONTRACTS UNIFORMLY: median rho = 0.8550 over 16 prompts, a | — | *geometry-h0bank* | `run_h3_contraction.py` | — |
| **D95** 🔻 | THE FIRST CAUSAL EXPERIMENT RAN, ITS INSTRUMENT VALIDATED ITSELF, AND ITS OWN PRE-REGISTERED GATE DECLARES THE RESULT VOID -- NOT NULL. | `kaggle_patch` | *geometry-patch* | — | kaggle_patch: 2 |
| **D96** | THE QK-ALIGNMENT PROBE (G2) IS MEASURED AT LAST, ITS PRE-REGISTERED PREDICTION IS REJECTED, AND THE RESULT IS STILL INCONCLUSIVE -- because the effect | `ds_qkprobe` | `bt1ji09c30ksvttripsv` () | `run_qk_analysis.py` | ds_qkprobe: 1 |
| **D97** | G1 IS DONE, AND IT REMOVES THE LEADING ALTERNATIVE EXPLANATION FOR THIS PROJECT'S NULLS: every one of 179 token positions across 12 prompts classifies | `ds_pertoken` | `bt1gefs432d0i0f8qhfo` () | — | ds_pertoken: 1 |
| **D98** | THE LOOP EXISTS. | `ds_blockcycle` | `bt1f31eeijhc7t6mt859` () | — | ds_blockcycle: 1 |
| **D99** | THE STATE IS CONFINED TO A SPHERE, AND THAT MAKES THE SETTLE/LOOP/DRIFT TRICHOTOMY EXHAUSTIVE RATHER THAN A LIST OF THREE GUESSES. | `ds_pertoken`, `kaggle_h0bank` | *geometry-h0bank* | — | ds_pertoken: 1, kaggle_h0bank: 547 |
| **D100** 🔻 | H2 RETURNS NOTHING EVEN ON THE CYCLE -- the one object where it was still live -- WHILE THE SAME RUN DELIVERS THE PROJECT'S FIRST WORKING LENGTH-MATCH | `ds_cyclegeom` | `bt17gufvdcnlp777c5cg` () | — | ds_cyclegeom: 1 |
| **D101** | THE FIRST POSITIVE DIRECTION FOR H2 IN THE PROJECT'S HISTORY -- AND IT IS A LEAD, NOT A FINDING, BECAUSE THE KERNEL'S OWN p-VALUES WERE PSEUDOREPLICAT | `ds_grid` | `bt1vgfjlofgu7sut7kb1` () | — | ds_grid: 1 |
| **D102** 🔻 | THE UNTRAINED CONTROL FOR D98 STILL HAS NOT RUN, AND ITS SECOND ATTEMPT PRINTED A CONFIDENT SCIENTIFIC VERDICT OVER ZERO DATA -- caught by reading the | `ds_cyclenull` | `bt102roip8snofu5kosh` ⚠unregistered, `bt11tr5ekqe600cfi6eg` ⚠unregistered, `bt17krflb4tqtuffonu6` ⚠unregistered | — | ds_cyclenull: 1 |
| **D103** | EVERY ACCURACY NUMBER IN THIS PROJECT IS AN ORACLE OVER DEPTH, AND IT INFLATES BY 2x. | `ds_grid` | *geometry-census*, *geometry-remote-continued* | — | ds_grid: 1 |
| **D104** | THE BLOCK CYCLE IS LEARNED, NOT ARCHITECTURAL -- and my pre-registered prediction that it would be architectural is REFUTED. | `ds_cyclenull` | `bt11u1jd652fmsoc2neo` () | — | ds_cyclenull: 1 |
| **D105** 🔻 | CLRS-TEXT, THE ONE TASK SOURCE INSIDE HUGINN'S OWN TRAINING MIXTURE, SCORES 0.0% EXACT AND 14.9% CONTAINMENT AT DEPTH 4 -- BUT THE RUN IS NOT YET EVID | — | *geometry-clrs* | — | — |
| **D106** | THE H2 NULLS ARE NOT A COORDINATE ARTEFACT: correcting every shape statistic for the fact that the state lives on a SPHERE changes it by 5%, against a | `kaggle_h0bank` | — | — | kaggle_h0bank: 547 |
| **D107** | RE-SCORED HONESTLY, THE CAPABILITY AXIS FALLS BY 1.66x -- AND THE INFLATION IS SO FAMILY-DEPENDENT THAT IT REVERSES WHICH LADDER THE PROJECT SHOULD US | `ds_grid` | — | — | ds_grid: 1 |
| **D108** 🔻 | THIS PROJECT HAS A WORKING CAUSAL INSTRUMENT. | `ds_estream` | `bt1piv6aglul4h2nbtnk` () | — | ds_estream: 2 |
| **D109** | A GRANULARITY DISSOCIATION: within ONE computation at a verified-identical token count, the FIXED POINT decodes the specific answer while the PATH'S S | `kaggle_lenmatch` | — | — | kaggle_lenmatch: 190 |
| **D110** | H2's BEHAVIOURAL ARM DOES NOT MERELY FAIL -- IT RUNS BACKWARDS, AND THE REVERSAL IDENTIFIES DIFFICULTY ITSELF RATHER THAN THE ANSWER TOKEN. | `ds_addk`, `ds_grid` | `bt1vuup3tfvmj2j3bmsr` () | — | ds_addk: 1, ds_grid: 1 |
| **D111** | THE PARAMETER/STATE DISSOCIATION IS COMPLETE, AND THE STATE ARM'S DECAY YIELDS THE PROJECT'S FIRST CAUSAL ESTIMATE OF THE CONTRACTION RATE -- WHICH MA | `ds_estream` | `bt1ac05vfako35rndads` () | — | ds_estream: 2 |
| **D112** | THE READOUT DOES NOT WAIT FOR THE FIXED POINT, AND THE SLOWEST MODE IS A DAMPED COMPLEX PAIR OF PERIOD EXACTLY 6. | `kaggle_lenmatch` | — | — | kaggle_lenmatch: 190 |
| **D113** | THE CAUSAL CONTRACTION RATE IS A PROPERTY OF THE MAP, NOT OF ONE PROMPT -- rho = 0.8335 ACROSS 16 DISTINCT RECIPIENTS, sd 0.028. | `ds_estream_b1c` | `bt1j30cub5aifduql4fl` () | — | ds_estream_b1c: 1 |
| **D114** | THE READOUT IS DECOUPLED FROM THE DYNAMICS: a prompt's contraction rate does NOT predict when its answer becomes readable. | `ds_addk`, `ds_estream_b1c` | — | `run_h0_within.py`, `run_rho_vs_behaviour.py` | ds_addk: 1, ds_estream_b1c: 1 |
| **D115** | THE CONTRACTION RATE IS A PROPERTY OF THE TASK TYPE, NOT OF PROMPT LENGTH AND NOT OF DIFFICULTY. | `kaggle_geomcap` | — | — | kaggle_geomcap: 610 |
| **D116** | THE ROTATION ANGLE IS A SECOND, INDEPENDENT TASK-LEVEL PARAMETER OF THE MAP -- AND IT SEPARATES TRAINED FROM UNTRAINED BY A FACTOR OF TWO -- BUT DIFFI | — | — | — | — |
| **D117** | D85's GEOMETRY-CAPABILITY NULL NOW COVERS THE DYNAMICS, NOT ONLY THE SHAPE -- AND THE SAME RUN REPRODUCES PROMPT LENGTH AS THE ONLY LIVE PREDICTOR. | — | *geometry-capability* | — | — |
| **D118** 🔻 | B4b IS VOID FOR H2 -- AND THE REASON IS ITSELF A RESULT: HUGINN CANNOT ADD MORE THAN ONE NONZERO DIGIT. | `ds_fixedsum` | `bt1tos7luo04f7370fpu` () | — | ds_fixedsum: 1 |
| **D119** 🔻 | B3 WAS ALREADY DONE AND THE INDEX SAID IT WAS NOT: the FULL one-unroll operator's spectrum has been measured since 2026-08-08, it gives a fourth agree | `ds_eigen` | `bt1mvf3juvrcjpl6kl1i` ⚠unregistered | — | ds_eigen: 1 |
| **D120** 🔻 | TRAINING PUSHES THE RECURRENT MAP TOWARD THE EDGE OF STABILITY, AND IT IS DONE BY THE EARLIEST CHECKPOINT AVAILABLE. | `kaggle_rho_ckpt_` ⚠MISSING | *geometry-rho-ckpt-a*, *geometry-rho-seed-a* | `preflight.py` | — |
| **D121** | A4b CANNOT BE SETTLED FROM BANKED DATA, AND THE FIRST ANALYSIS THAT SAID IT COULD WAS A CONFOUND WORTH RECORDING. | — | — | — | — |
| **D122** | TWO NUMBERS IN THIS LEDGER ARE GEOMETRIC IDENTITIES OF THE SPHERE, NOT MEASUREMENTS OF HUGINN. | — | — | — | — |
| **D123** | THE PROJECT'S TOPOLOGICAL SIGNAL IS THE DAMPED ROTATION, AND D64 UNDERSTATED IT 11x BY POOLING. | — | — | `run_homology_null.py` | — |
| **D124** 🔻 | THE CANCELLED CLRS RUN'S OUTPUT SURVIVED, AND READING IT RETRACTS HALF OF D105 AND SHOWS THE RUN NEVER IMPLEMENTED DR1's SPECIFICATION. | `kaggle_clrs` | *geometry-clrs* | — | kaggle_clrs: 2 |
| **D125** | THE FIXED POINT ENCODES THE INPUT, NOT THE ANSWER -- WHICH SCOPES D109's HEADLINE RATHER THAN ITS DISSOCIATION. | `ds_paired` | `bt1hh12jorfr1mat9r2p` () | — | ds_paired: 1 |
| **D126** | THE TOPOLOGICAL SIGNAL IS REAL AND DOES NOT CARRY CORRECTNESS -- D123's H1 joins the shape statistics in the outcome null. | — | — | — | — |
| **D127** | THE DAMPED ROTATION IS AN ALL-OR-NOTHING PROPERTY OF THE PROMPT TEMPLATE, AND IT IS NEITHER LENGTH NOR TASK. | — | `bt16fuonalgnb2fldhpk` () | — | — |
| **D128** | DR1's FIXED-BODY BINARY-SEARCH LADDER WORKS STRUCTURALLY AND FAILS AS A DIFFICULTY AXIS: HUGINN IS BETTER AT THE POSITIONS THE ALGORITHM FINDS HARDER. | `ds_binsearch` | `bt1t7ogigg1cm83pgdo1` () | — | ds_binsearch: 2 |
| **D129** | SYMBOL DIVERSITY IS REFUTED AS THE SELECTOR OF THE ROTATION -- AND THE RESIDUAL POINTS AT A SINGLE WORD. | `ds_diversity` | `bt1l8elecek91qtvkdcp` () | — | ds_diversity: 133 |
| **D130** | THE OBSERVABILITY CENSUS LANDS: 31 USABLE ITEMS WITH GENUINE DYNAMIC RANGE, ACROSS 5 FAMILIES THAT CAN EACH CARRY A WITHIN-FAMILY DESIGN. | `kaggle_census` | *geometry-census* | `run_census_analysis.py` | kaggle_census: 3 |
| **D131** 🔻 | DR3's 'NOVEL, PUBLISHABLE CORE' DOES NOT CONVERGE ON HUGINN: THE IMPLICIT-DIFFERENTIATION ADJOINT NEEDS 257 VJPs, NOT THE 30-40 PREDICTED, AND ITS OWN | `ds_implicit` | `bt1i4qge4opib5adikll` () | — | ds_implicit: 1 |
| **D132** | ONE NOUN IN THE INSTRUCTION SWITCHES THE RECURRENT MAP BETWEEN SETTLING AND A DAMPED PERIOD-6 ROTATION -- 36 OF 36 PAIRED CELLS, AT AN IDENTICAL TOKEN | `ds_wordswap` | `bt19t87hq0fmmget8p36` ⚠unregistered | — | ds_wordswap: 109 |
| **D133** 🔻 | D86's CONTAINMENT RISE IS SPECIFIC TO THE CORRECT ANSWER, NOT A SIDE-EFFECT OF LONGER OUTPUT -- imported from a parallel branch and INDEPENDENTLY REPR | `kaggle_depthacc` | *geometry-remote-continued* | — | kaggle_depthacc: 3 |
| **D134** | THE WORD THAT SELECTS THE DYNAMICAL REGIME IS NOT PREDICTED BY MEANING, BY FORM, OR BY TOKENISATION -- IT IS IDIOSYNCRATIC TO THE TOKEN. | `ds_nounsweep` | `bt1p04j521i2udj5civd` ⚠unregistered | — | ds_nounsweep: 385 |
| **D135** | THE REGIME IS NOT A LINEAR FUNCTION OF THE TOKEN EMBEDDING EITHER, WHICH REFUTES D134's OWN 'BY ELIMINATION' INFERENCE -- AND ROTATION TURNS OUT NOT T | `ds_embsep` | `bt19lv117dbh0bqrcbcv` () | — | ds_embsep: 206 |
| **D136** | THE REGIME IS NOT RECOVERABLE FROM THE TOKEN EMBEDDING BY ANY OF SIX SIMPLE RULES, LINEAR OR NONLINEAR. | — | — | `run_embsep_nonlinear.py` | — |
| **D137** 🔻 | THE TWO 'MIXED' NOUNS RESOLVE INTO ONE INTERMEDIATE TOKEN AND ONE DETECTOR ERROR. | `ds_embsep` | — | `run_rotation_continuum.py` | ds_embsep: 206 |
| **D138** 🔻 | THE TWO-REGIME STRUCTURE REPLICATES ON TWO INDEPENDENT BANKS, AT THE SAME PLACE, AND ONE NOUN IN SEVENTY IS AMBIGUOUS. | — | — | `run_rotation_continuum.py` | — |
| **D139** | D129 SURVIVES A MORE POWERFUL RE-TEST: SYMBOL DIVERSITY DOES NOT SELECT THE REGIME, AND DOES NOT MODULATE IT EITHER AT THE CORRECT UNIT. | `ds_diversity` | — | — | ds_diversity: 133 |
| **D140** | THE REGIME IS CAUSALLY CONTROLLED BY e, THE TRANSITION IS SHARPER THAN A 0.05 GRID CAN RESOLVE, AND THE ROTATING SET IS NOT CONVEX. | `ds_einterp` | `bt1igml4vt3so359mdp3` (einterp-regime-boundary) | `run_einterp_analysis.py` | ds_einterp: 1 |
| **D141** 🔻 | RETRACTION AND REPLACEMENT: D137's AND D138's 'GAPPED, THEREFORE TWO REGIMES' TEST WAS THE WRONG TEST, AND THE CORRECT ONE IS STRONGER. | — | — | — | — |
| **D142** | GMRES FIXES D131's SOLVER OUTRIGHT -- 24/24 CONVERGED AGAINST 1/24 -- AT THE ITERATION COUNT THE THEORY PREDICTED; THE ATTRIBUTION IT YIELDS IS ONLY M | `ds_gmres` | `bt13o2rva6j2mk5ls0u8` (gmres-implicit-attribution) | — | ds_gmres: 1 |
| **D143** 🔻 | THE FIRST STRUCTURALLY CLEAN DIFFICULTY LADDER THIS PROJECT HAS BUILT, AND H2 IS NULL ON IT ONCE LIST POSITION 1 IS REMOVED. | `ds_nth` | `bt1j97ks2han4ptnks5q` (nth-orthogonal-ladder) | — | ds_nth: 1 |
| **D144** | THE ROTATING REGION OF e-SPACE IS A STRUCTURED SET, NOT A BLOB: THE SAME DISTANCE TRAVELLED LEAVES IT IN A RANDOM DIRECTION AND STAYS INSIDE IT ALONG  | `ds_nonconvex` | `bt12cjm70cqfbgqu4qhr` ⚠unregistered | — | ds_nonconvex: 1 |
| **D145** | HUGINN'S CHAT TEMPLATE LEAKS THE NEXT-TURN ROLE HEADER ONTO THE ANSWER WITH NO SEPARATOR, AND IT HAS BEEN CORRUPTING EVERY DECODED-ACCURACY NUMBER IN  | `kaggle_depthacc` | — | — | kaggle_depthacc: 3 |
| **D146** | D98's 'EACH BLOCK CONVERGES TO ITS OWN FIXED POINT' IS FALSE FOR ROTATING ORBITS -- THEY ARE STILL MOVING AT 4% OF THE STATE NORM AT UNROLL 63, A 157x | `ds_blockbank` | `bt16fuonalgnb2fldhpk` () | — | ds_blockbank: 131 |
| **D147** | HUGINN CANNOT STATE-TRACK AT ALL: IT IS A CONSTANT RESPONDER, AND THE PROJECT'S STANDARD CAPABILITY AXIS SCORES THAT 100%. | `ds_statetrack` | `bt12s3ifcf9u6383e468` ⚠unregistered | `run_statetrack_analysis.py` | ds_statetrack: 1 |
| **D148** | CROSSING THE REGIME BOUNDARY MOVES THE ANSWER'S RANK BY EXACTLY ONE POSITION AND FLIPS NO ANSWERS -- THE BEHAVIOURAL NULL IS NOW BOUNDED, WHICH D132's | `ds_regimebehav` | `bt1jp231g4u3d1fbp168` ⚠unregistered | `run_regimebehav_analysis.py` | ds_regimebehav: 1 |
| **D149** | THE ROTATING REGION IS A BOUNDED, ANISOTROPIC REGION -- NOT A HALF-SPACE. | `ds_boundary` | `bt17mi3ab789pavgrg3c` ⚠unregistered | — | ds_boundary: 1 |
| **D150** | SYSTEMATIC EXPOSURE AUDIT OF THE ORACLE CAPABILITY AXIS: 22 OF 149 ROWS CARRY A NUMBER FROM IT -- NOT THE 44 A NAIVE COUNT GIVES. | — | — | — | — |
| **D151** 🔻 | THE SCORER IS REPAIRED, D89's NULL SURVIVES AND STRENGTHENS, AND D145's CONTAINMENT SENTENCE IS WITHDRAWN AS MY OWN RULE-MISMATCH ERROR. | `kaggle_depthacc` | — | `run_addk_arms.py`, `run_depth_accuracy.py`, `run_depth_profile.py` | kaggle_depthacc: 3 |
| **D152** | fromconfig IS DETERMINISTIC AND THE UNTRAINED ARMS REPRODUCE AT THE WEIGHT LEVEL -- AND 70% OF dsseeds' RECORDING IS BELOW THE FLOATING-POINT FLOOR, T | `ds_bank`, `ds_seeds` | — | — | ds_bank: 97, ds_seeds: 61 |
| **D153** | WE DO NOT REPRODUCE HUGINN'S OWN PUBLISHED ARC-EASY NUMBER: 40.7% AGAINST 69.9%, A 29-POINT MISS OF A PRE-REGISTERED +/-10 BAND. | `ds_arcrepro` | `bt11qtigf5o3aj8hbad0` ⚠unregistered | — | ds_arcrepro: 1 |
| **D154** 🔻 | WHAT MOVES THE READOUT IS DISPLACEMENT IN e, NOT THE REGIME: rho(t, bestrank) = +0.8163 WHILE THE PARTIAL rho(R, bestrank | `ds_regimebehav` | — | — | ds_regimebehav: 1 |
| **D155** | D135 AND D136 ARE NOT BOUNDED NULLS: THEIR DESIGN CANNOT SEE A LINEAR SEPARATION BELOW COHEN'S d  10. | `ds_embsep` | — | — | ds_embsep: 206 |
| **D156** | A31 IS A NULL, AND THE REASON IS THAT ITS DESIGN COULD NOT HAVE WORKED: RANDOM DIRECTIONS IN 279,840 DIMENSIONS ARE ALL ORTHOGONAL TO A 6-DIMENSIONAL  | `ds_aniso` | `bt1mi0pjmouua0nil59f` ⚠unregistered | — | ds_aniso: 1 |
| **D157** | THE OBSERVABILITY CENSUS CERTIFIES AVAILABILITY, NOT PRODUCTION -- ITS 41 LIVE ITEMS BECOME 0 ON THE FINAL-UNROLL AXIS, AND THE INFLATION IS 5.29x, NO | `kaggle_census` | — | — | kaggle_census: 3 |
| **D158** | ON THE PRODUCTION AXIS HUGINN HAS TWO USABLE TASK FAMILIES, NOT FIVE -- AND IT CAN HOLD A COPIED WORD TO THE END OF ITS COMPUTATION BUT NOT A COPIED D | `kaggle_census` | — | — | kaggle_census: 3 |
| **D159** | THE ANSWER IS NOT DECAYING WITH DEPTH -- IT IS DISPLACED TO A STABLE WRONG RANK. | `kaggle_census` | — | — | kaggle_census: 3 |

## Claims with no code, job or script reference

These cannot be re-derived by anyone but their author, and that is the defect
`run_addk_arms.py` was written to fix for D110. Listing them is not an accusation —
many are re-analyses of data banked elsewhere, or reasoning about the model source.
But a load-bearing number in this list should get a script.

- **D1** — README's reproduce block: uv sync ; uv run pytest ; uv run ruff check . *(evidence column: main README.md)*
- **D2** — 11 tests pass / 1 skip on main with the correct setup *(evidence column: —)*
- **D4** — "David" authored the fabricated two-scale numbers and the original per-position metric code *(evidence column: results/FINDINGS.txt, PR #1 body, tests/testtwoscale.py comment)*
- **D5** — Repo activity has stopped *(evidence column: GitHub API (pushedAt), git fetch --all)*
- **D6** — pr-real-data restates the two-scale band-test coefficient as clean and final *(evidence column: git show origin/pr-real-data:docs/ANALYSISSUMMARY.md)*
- **D7** — analysis/correlate.py's significance-threshold table is fully correct *(evidence column: src/trajgeom/analysis/correlate.py::SPEARMANCRITP05)*
- **D8** — The project has 3 active contributors *(evidence column: git log --all (all branches), gh pr view 1 --json author)*
- **D9** — No paper-drafting artifact exists yet *(evidence column: find . -iname '.tex' -o -iname 'draft' (repo-wide, excluding .venv))*
- **D10** — The length-confound control (partialspearman against seqlen) is a working control wherever it's applied *(evidence column: results/counting.csv, results/switch.csv, results/maxtask.csv, src/tra)*
- **D16** — Cross-task synthesis (the strongest single anti-H2 datapoint): does \ *(evidence column: winding\)*
- **D17** — The honesty headline: is the model even solving the counting task at the depths where winding/stepssettle "ris *(evidence column: results/countingaccuracy.csv (56 rows; accuracy scored by generation +)*
- **D18** — Cross-task pattern in stepssettledifficulty: is the sign explained by whether a task's answer accumulates vs. *(evidence column: Per-level rhos recomputed from the same results/.csv as D16; adversari)*
- **D19** — Init-robustness ranking: which trajectory metrics are driven by the computation vs. *(evidence column: results/dissocmultiinit.csv (600 rows = 6 nops × 2 kinds × 6 task-seed)*
- **D20** — Two structural facts a truly-clean report must disclose: (a) where the Blayney loops actually sit in the promp *(evidence column: (a) results/blayneyrepro.csv; (b) results/threescalemodkextended.csv; )*
- **D21** — Adversarial audit of the project-state recap itself (2026-07-25): do the summary claims survive independent re *(evidence column: The workflow's per-cluster verdicts (H2, correctness, H1, H3, stepsset)*
- **D23** — The recurrent state is architecturally confined to a sphere — which invalidates the rotation-center, the null  *(evidence column: ravenmodelingminimal.py @ pinned revision bb6621b65e90b6a4b9b29ef88dc8)*
- **D42** — Training SLOWS the contraction, ρ ≈ 0.66 → 0.91, lengthening the depth time-constant 2.4 → 10.6 unrolls. *(evidence column: Re-analysis of the D41 curves through this project's own derived law ()*
- **D43** — RETRACTED 2026-08-04 (see (6)). *(evidence column: Derivation from D42 + Geiping et al.'s reported numbers as recorded in)*
- **D45** — HEADLINE RETRACTED 2026-08-04, same day, after adversarial review. *(evidence column: Same run; 3 prompts each in counting / nesting-depth / arithmetic word)*
- **D54** — D35's depth-scaling is NOT a decodability effect: the depth at which the count becomes linearly decodable does *(evidence column: Local re-analysis of perinstanceerror.json for both arms; no new compu)*
- **D56** — H1's settle/loop/drift regimes are VACUOUS: the three classes have statistically indistinguishable winding (Kr *(evidence column: results/h2loops.csv, phase.csv, homology.csv and a census of all resul)*
- **D65** — H1, H2 and H3 ALL RETURN TO OPEN. *(evidence column: Re-reading of D26/D28/D32/D56 against D55; CLAUDE.md §5 instrument rul)*
- **D67** — REPRODUCTION AUDIT: with the banked raw trajectories restored, the three analysis CSVs that depend on them re- *(evidence column: 140 results/trajectories/.npy (80 at ns128, 285 MB) restored from the )*
- **D116** — THE ROTATION ANGLE IS A SECOND, INDEPENDENT TASK-LEVEL PARAMETER OF THE MAP -- AND IT SEPARATES TRAINED FROM U *(evidence column: Banked data only, zero GPU: 608 kagglegeomcap orbits for (a); 156 dsba)*
- **D121** — A4b CANNOT BE SETTLED FROM BANKED DATA, AND THE FIRST ANALYSIS THAT SAID IT COULD WAS A CONFOUND WORTH RECORDI *(evidence column: Banked data only, zero GPU: 188 kagglelenmatch orbits with their manif)*
- **D122** — TWO NUMBERS IN THIS LEDGER ARE GEOMETRIC IDENTITIES OF THE SPHERE, NOT MEASUREMENTS OF HUGINN. *(evidence column: )*
- **D150** — SYSTEMATIC EXPOSURE AUDIT OF THE ORACLE CAPABILITY AXIS: 22 OF 149 ROWS CARRY A NUMBER FROM IT -- NOT THE 44 A *(evidence column: docs/claimsledger.md cross-referenced against scratch//{job,main,body})*

## Banked data no claim references

Directories holding `.npy`/`.json` that NO ledger row cites AND no script,
test or module reads. Each is either a dead
run worth deleting or an unanalysed asset worth mining — and this project has
already found one of the latter (three banks re-read with a new statistic became
D138/D141 at zero GPU cost).

- `scratch/datasphere_smoke_test` — 5 artifact(s)
- `scratch/ds_startup_probe` — 18 artifact(s)
- `scratch/ds_weightsds` — 5 artifact(s)
- `scratch/kaggle_blayney_modk` — 16 artifact(s)
- `scratch/kaggle_capcontent` — 1 artifact(s)
- `scratch/kaggle_chatfmt` — 2 artifact(s)
- `scratch/kaggle_eps_split` — 2 artifact(s)
- `scratch/kaggle_modk_extended` — 16 artifact(s)
- `scratch/kaggle_output` — 15 artifact(s)
- `scratch/kaggle_output_2` — 15 artifact(s)
- `scratch/kaggle_output_3` — 15 artifact(s)
- `scratch/kaggle_output_4` — 15 artifact(s)
- `scratch/kaggle_output_5` — 15 artifact(s)
- `scratch/kaggle_output_6` — 15 artifact(s)
- `scratch/kaggle_output_7` — 15 artifact(s)
- `scratch/kaggle_rho_ckpt_b` — 2 artifact(s)
- `scratch/kaggle_rho_ckpt_c` — 2 artifact(s)
- `scratch/kaggle_rho_ckpt_d` — 2 artifact(s)
- `scratch/kaggle_rho_seed_a` — 2 artifact(s)
- `scratch/kaggle_rho_seed_b` — 2 artifact(s)
- `scratch/kaggle_v6_probe` — 1 artifact(s)
- `scratch/kaggle_v6_rerun` — 1 artifact(s)
- `scratch/kaggle_v6_topk_fix` — 16 artifact(s)
