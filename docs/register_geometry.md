# Can Huginn hold a counting register? What the norm constraint forces

STATUS: derivation plus numerical checks, 2026-07-26. No GPU used; every number
is either from the model source at the pinned revision or from the banked
trajectories. **No claim here is an empirical finding about how Huginn actually
counts** — this is an argument about what the architecture *permits*, which
constrains how the curator's Task a/b should be probed.

MOTIVATION: Prof. Barannikov's framing is that the minimal algorithm for a
running count is a register with values in ℕ, updated by the translation
`T₁ : s ↦ s+1`. The natural probe is then "is there a direction `v` in latent
space such that seeing a `1` displaces the state by ≈`v`?" This document argues
that probe is looking for something the architecture forbids, and derives what
to look for instead.

---

## 1. The constraint

Every `SandwichBlock` ends in an RMSNorm (`raven_modeling_minimal.py` L535), so
every recorded state is a normalised vector. Measured over 20 trajectories ×
128 unrolls:

```
||h|| = 76.37 +- 0.007      relative variation 0.0073%
||h|| / sqrt(n_embd) = 1.0511
```

So states occupy a **thin spherical shell** of radius `R = 76.37` and relative
half-thickness `δ ≈ 7.3e-5`.

## 2. A translation register does not fit

Displace a state `p` on the sphere by `v = v_∥ + v_⊥` (radial + tangential).
To second order,

```
||p + v|| - R  ≈  v_∥  +  |v_⊥|² / (2R)
```

The radial term must satisfy `|v_∥| < Rδ = 0.0056` — a radial component is
almost entirely forbidden. The tangential term is the interesting one: it is
quadratic, so small steps are nearly free, but a *straight-line* register
accumulates. Requiring the total tangential displacement `D` to keep the state
in the shell gives

```
D² / (2R) < Rδ      =>      D < R sqrt(2δ) = sqrt(2 R · Rδ) = 0.923
```

Over Barannikov's `m = 64` positions that is **0.0144 per increment**.

For comparison, from the banked trajectories: the bf16 noise floor is **≈0.9**
per step and real signal-regime steps are **6.9 – 72.5**. So a straight
translation register would have to operate **62× below the arithmetic noise
floor**.

**Robustness.** The bound scales as `sqrt(δ)`, so it is very insensitive to the
shell estimate:

| assumed shell δ | max total D | per increment | vs noise floor |
|---|---|---|---|
| 7.3e-5 (measured) | 0.92 | 0.0144 | 62× below |
| 7.3e-4 (10×) | 2.92 | 0.0456 | 20× below |
| 7.3e-3 (100×) | 9.23 | 0.1442 | 6× below |
| 7.3e-2 (1000×) | 29.18 | 0.4560 | 2× below |

The conclusion survives a **1000×** underestimate of the shell thickness. Only
at δ ≈ 0.5 — i.e. no norm constraint at all — would translation become viable.

**Caveat, stated explicitly.** δ was measured across *unrolls* at a single
token position. A register lives across *token positions*, and RMSNorm's output
norm is `‖û ⊙ g‖` for the unit input direction `û`, which varies somewhat with
direction. The across-position shell could be thicker than 7.3e-5. The table
above is exactly why this does not matter: the argument tolerates three orders
of magnitude of error. Measuring the across-position shell is nevertheless a
cheap GPU check worth doing (backlog 1.11).

## 3. What the sphere does permit: rotation

The norm-preserving actions on a sphere are rotations. The natural ℤ-action is
therefore `T₁ = R_{θ₀}`: a fixed rotation in some 2-plane, one application per
increment. Encoding `s` as an angle `s·θ₀` costs nothing in norm, exactly.

Feasibility, from the same numbers:

| register spans | angle/increment | chord/increment | vs noise floor |
|---|---|---|---|
| 10° of arc | 0.156° | 0.208 | below |
| 30° of arc | 0.469° | 0.625 | comparable |
| 70° of arc | 1.094° | 1.458 | above |

A register occupying 30–70° of arc gives per-increment motion at or above the
noise floor. For scale, the observed transient sweeps ~70° of geodesic, so a
register of this size is well within the range the state actually uses.

**So: a rotation register is feasible; a translation register is not.** This
does not say Huginn implements one — only that if it counts at all, this is the
form available to it.

## 4. Consequences for probing Task a / b

1. **Probe for a rotation, not a translation.** The object to look for is a
   2-plane and a fixed angle `θ₀`, such that each `1` in the string advances
   the state by `θ₀` within that plane. Concretely: fit the plane, then test
   whether the angle is linear in the running count `y_i`.

2. **Use tangent coordinates.** Over a small angular range the sphere is
   locally flat, so a tangent-space (log-map) translation at a reference point
   is an excellent approximation to a rotation — the two differ only through
   curvature. **This rescues Barannikov's framing rather than contradicting
   it**: a linear probe for `s_i` in tangent coordinates is the right test, and
   is equivalent to the rotation picture at small angle. What must not be done
   is a linear probe in raw ambient coordinates *interpreted as translation*,
   because the radial direction there is architecturally frozen.

3. **This reconnects the winding number to the hypothesis, in the right place.**
   If the register is a rotation by `θ₀` per increment, then the angle swept
   across the string is `y_m · θ₀`, so

   > **winding across token positions ∝ the count**

   That is H2 — but measured across *positions in the prompt*, not across
   *unrolls at the answer token*, which is where this project has always
   measured it and where the state merely converges. The metric was not
   wrong in kind; it was applied to the wrong index.

4. **Task b makes the winding number a genuine invariant.** The deepest
   mathematical defect in this project's winding metric is that a winding
   number is an element of `π₁(ℝ²∖{p}) ≅ ℤ` and requires a **closed** curve,
   while a trajectory is an open arc — so `Δθ/2π` was a real number with no
   quantization and no homotopy invariance. Barannikov's Task b uses
   **balanced** parentheses: depth starts at 0 and returns to 0. If the depth
   register is a rotation, the state **returns to its starting point**, and the
   curve closes. On a closed curve the winding number is a true topological
   invariant taking integer values.

   This yields a sharp, falsifiable prediction: for balanced strings the
   position-indexed winding should be **near-integer**, and for deliberately
   unbalanced strings it should not. That is a much stronger test than any
   correlation, because it predicts *quantization*, not just monotonicity.

## 5. What would falsify this

- The across-position shell turns out to be δ ≳ 0.1 (no effective norm
  constraint) — then §2's argument collapses and translation is back on the
  table.
- A linear probe in tangent coordinates recovers `s_i` with high R² while the
  fitted displacement direction has a large *radial* component — that would
  mean the norm constraint is being evaded in a way this analysis does not
  model.
- Position-indexed winding on balanced strings shows no tendency to
  integers and no dependence on maximum depth.

## 6. What this does NOT claim

That Huginn counts; that a register exists; that any rotation is present. Every
statement here is about what the architecture permits and therefore about
experimental design. The empirical tests are backlog items 1.11–1.13 and remain
unrun.
