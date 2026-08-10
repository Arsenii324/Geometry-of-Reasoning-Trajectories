"""A24's analysis: where along `e` does the rotation regime flip, and how sharply?

Implements the predictions pre-registered in `scratch/ds_einterp/job.py`, written
before the run returned. Nothing here is chosen after seeing the data; the one number
picked by hand is SHARP_MAX below, and it is quoted in the kernel docstring too.

  P1   instrument null -- R(t=0) must equal the unpatched settling orbit's R, because
       at t=0 the patch substitutes e_s for e_s and is a no-op BY CONSTRUCTION. A
       mismatch means the hook does not write what it claims and VOIDS the run.
  P1'  prediction, NOT an instrument check -- R(t=1) matching the rotating noun's own
       orbit would show the attractor is fixed by `e` alone even from a foreign
       initial state (D111/D113's DEQ claim). Failure here is a finding about the
       model, so it is reported separately and never gates anything.
  P2   PRIMARY -- the 10-90% transition width in t. D137 found the population GAPPED,
       which predicts a step: width < 0.25. Width > 0.5 is a ramp and contradicts
       D137, whose gap would then be a 51-noun sampling artifact.
  P3   the load-bearing control -- within-regime interpolations must NOT transition.
       Without this, "any perturbation of `e` moves the dynamics" explains P2 equally
       well.
  P4   consistency of t* across the four cross-regime pairs. Described, not tested:
       four pairs cannot support a test.
  P5   does `set`, D137's one in-gap token, project near t*?

Run:  uv run python scripts/run_einterp_analysis.py
"""

from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch/ds_einterp/out/out"
SHARP_MAX = 0.25          # P2's pre-registered threshold for "step, not ramp"
RAMP_MIN = 0.50           # above this, D137's gap is contradicted
NOOP_TOL = 1e-6           # P1: a no-op patch must reproduce the orbit exactly


def crossing(ts, rs):
    """Midpoint crossing t* and the 10-90% width, by linear interpolation.

    Deliberately not a sigmoid FIT: with 21 points and a possible step, a fitted
    logistic's width parameter is dominated by the two points that straddle the jump,
    and reports a confident number where the grid cannot resolve one. Reading the
    crossings off the data says exactly what the grid supports and no more.
    """
    lo, hi = float(np.min(rs)), float(np.max(rs))
    if hi - lo < 1e-9:
        return None, None, (lo, hi)
    z = (np.asarray(rs) - lo) / (hi - lo)

    def cross_at(level):
        for i in range(len(z) - 1):
            a, b = z[i], z[i + 1]
            if (a - level) * (b - level) <= 0 and a != b:
                return ts[i] + (level - a) / (b - a) * (ts[i + 1] - ts[i])
        return None

    t10, t50, t90 = cross_at(0.1), cross_at(0.5), cross_at(0.9)
    width = abs(t90 - t10) if (t10 is not None and t90 is not None) else None
    return t50, width, (lo, hi)


def main() -> int:
    man = json.loads((OUT / "manifest.json").read_text())
    rows = [r for r in man if r.get("ok")]
    if not rows:
        print("no usable rows")
        return 1
    curves = collections.defaultdict(list)
    for r in rows:
        curves[(r["rot_w"], r["set_w"], r["kind"], r["seq"])].append(r)
    for v in curves.values():
        v.sort(key=lambda r: r["t"])
    print(f"{len(rows)} orbits, {len(curves)} interpolation curves\n")

    # ---- P1 / P1' -------------------------------------------------------------
    print("P1 INSTRUMENT NULL -- R(t=0) vs the unpatched settling orbit (no-op patch):")
    void = False
    for k, v in sorted(curves.items()):
        r0 = v[0]
        d = abs(r0["R"] - r0["base_R"])
        flag = "ok" if d < NOOP_TOL else "MISMATCH"
        if d >= NOOP_TOL:
            void = True
        print(f"  {k[0]:>7s}<-{k[1]:<7s} seq{k[3]} {k[2]:<11s} "
              f"R(0)={r0['R']:.6f} base={r0['base_R']:.6f}  d={d:.2e}  {flag}")
    if void:
        print("\n  VOID: the patch is not a no-op at t=0, so it does not write what it "
              "claims.\n  Nothing below may be read.")
        return 1
    print("  -> the hook writes exactly `e`, and only `e`\n")

    print("P1' PREDICTION (not a gate) -- does the attractor follow `e` from a foreign h0?")
    for k, v in sorted(curves.items()):
        r1 = v[-1]
        print(f"  {k[0]:>7s}<-{k[1]:<7s} seq{k[3]} {k[2]:<11s} "
              f"R(1)={r1['R']:.4f} vs rotating noun's own {r1['base_rot_R']:.4f}"
              f"   d={abs(r1['R'] - r1['base_rot_R']):.4f}")

    # ---- P2 / P3 --------------------------------------------------------------
    print("\nP2 PRIMARY (cross-regime) and P3 CONTROL (within-regime):")
    widths = {"cross": [], "within": []}
    stars = []
    for k, v in sorted(curves.items(), key=lambda kv: kv[0][2]):
        ts = [r["t"] for r in v]
        rs = [r["R"] for r in v]
        t50, w, (lo, hi) = crossing(ts, rs)
        grp = "cross" if k[2] == "cross" else "within"
        span = hi - lo
        if grp == "cross" and w is not None:
            stars.append((k, t50))
        widths[grp].append((span, w))
        wtxt = f"{w:.3f}" if w is not None else "  -  "
        ttxt = f"{t50:.3f}" if t50 is not None else "  -  "
        print(f"  {k[2]:<11s} {k[0]:>7s}<-{k[1]:<7s} seq{k[3]}  "
              f"R span [{lo:.3f}, {hi:.3f}] = {span:.3f}   t* {ttxt}   10-90% width {wtxt}")

    cw = [w for _, w in widths["cross"] if w is not None]
    cs = [s for s, _ in widths["cross"]]
    ws = [s for s, _ in widths["within"]]
    print(f"\n  cross-regime : median R span {np.median(cs):.3f}"
          f"{f', median 10-90% width {np.median(cw):.3f}' if cw else ''}")
    print(f"  within-regime: median R span {np.median(ws):.3f}" if ws else
          "  within-regime: no curves")
    if cw:
        w = float(np.median(cw))
        verdict = ("STEP -- consistent with D137's gap" if w < SHARP_MAX else
                   "RAMP -- CONTRADICTS D137's gap" if w > RAMP_MIN else
                   "INTERMEDIATE -- neither prediction is met")
        print(f"  P2 verdict: 10-90% width {w:.3f} -> {verdict}")
    if ws and cs:
        print(f"  P3 verdict: within-regime span is {np.median(ws) / max(1e-9, np.median(cs)):.2f}x "
              f"the cross-regime span"
              f" -> {'CONTROL HOLDS' if np.median(ws) < 0.5 * np.median(cs) else 'CONTROL FAILS -- any e-perturbation moves R, and P2 means nothing'}")

    # ---- P4 / P5 --------------------------------------------------------------
    if stars:
        sv = [t for _, t in stars if t is not None]
        print(f"\nP4 CONSISTENCY -- t* across {len(sv)} cross-regime curves: "
              f"{[round(t, 3) for t in sv]}")
        print(f"  median {np.median(sv):.3f}, range [{min(sv):.3f}, {max(sv):.3f}]")

    probes = {(r["rot_w"], r["set_w"], r["seq"]): r["probe_t"] for r in rows
              if r["kind"] == "cross" and r.get("probe_t") is not None}
    if probes:
        print(f"\nP5 -- where `set` (D137's in-gap token) projects onto each axis:")
        for k, v in sorted(probes.items()):
            star = next((t for kk, t in stars if (kk[0], kk[1], kk[3]) == k), None)
            print(f"  {k[0]:>7s}<-{k[1]:<7s} seq{k[2]}  set at t={v:+.3f}"
                  + (f"   t*={star:.3f}   |set - t*| = {abs(v - star):.3f}"
                     if star is not None else ""))
    else:
        print("\nP5 -- no probe projections (shape gate dropped them); not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
