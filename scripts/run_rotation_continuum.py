"""D137: is the rotation regime a THRESHOLDED CONTINUUM rather than a binary label?

WHY THIS EXISTS. D135's P3 gate found that 2 of 51 nouns (`set`, `word`) are MIXED --
some of their four orbits rotate and some do not. D135 read that as evidence for a
continuous quantity being thresholded rather than a label being looked up. **That
reading was an interpretation, not a measurement**, and this project has a standing
rule against elaborating on an untested premise -- which is exactly the error D135
itself caught in D134.

The reading makes a sharp prediction that the ALREADY-BANKED orbits can test, with no
GPU: if a continuum is being thresholded, then a continuous measure of rotation
strength should place the mixed nouns AT THE BOUNDARY between the two populations. If
instead the mixed nouns sit at typical rotating or typical settling values -- i.e. the
mixing is measurement noise in the period estimator, or a genuine bistability with no
intermediate -- the continuum reading is wrong and D135's third paragraph must be
withdrawn.

THE CONTINUOUS STATISTIC. For each orbit, the period-6 spectral power of the tail,
as a fraction of total non-DC power:

    R = |F[6-per-period bin]|^2 / sum_{k>=1} |F[k]|^2      averaged over coordinates

Binary "does it rotate" is the same quantity thresholded, so R is the natural
continuous extension rather than a new instrument. R in [0, 1]; a pure period-6 orbit
gives R -> 1, a monotone settle gives R -> 0.

PREREGISTERED PREDICTIONS (written before running):

  P1  SEPARATION. R must separate the two deterministic populations -- median R of the
      29 rotating nouns above that of the 20 settling ones, Mann-Whitney p < 0.01.
      **If it does not, R is not measuring the thing the binary label measures and
      nothing below may be read.** This is the instrument's null, per CLAUDE.md §5.

  P2  PRIMARY -- BOUNDARY PLACEMENT. Under the continuum reading, the mixed nouns'
      orbits sit between the populations. Quantified as the mean over mixed orbits of
      that orbit's percentile within the pooled deterministic R distribution: the
      prediction is that mixed orbits land near the CROSSOVER percentile (where the
      two populations' densities cross), not near 0 or 1. Tested against a null that
      draws the same number of orbits at random from the deterministic pool.

  P3  BIMODALITY. If the quantity is genuinely thresholded, the deterministic R values
      are BIMODAL with a gap; if it is continuous with an arbitrary cut, they are
      unimodal. Reported as the size of the largest gap in the sorted R values
      relative to the range, against a permuted null. This distinguishes "threshold on
      a continuum" from "two genuinely distinct dynamical regimes" -- which are
      different mechanisms, and D135's wording assumed the first without checking.

  P4  ORBIT-LEVEL CONFOUND. Each noun contributes 4 orbits = 2 sequences x 2 markers.
      If the mixing tracks the MARKER or the SEQUENCE rather than sitting at a
      boundary, that is a cleaner explanation than a threshold and is reported first.

Run:  uv run python scripts/run_rotation_continuum.py
"""

from __future__ import annotations

import collections
import glob
import json
import pathlib
import random

import numpy as np

from traj_geom.metrics.dynamics import rotation_power

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch/ds_embsep/out/out"
PERIOD = 6
TAIL = 24
N_PERM = 5000
SEED = 0


def dominant_period(seq: np.ndarray) -> int:
    n, lo = len(seq), len(seq) - TAIL
    d = {p: float(np.median([np.linalg.norm(seq[t] - seq[t + p]) for t in range(lo, n - p)]))
         for p in range(1, PERIOD * 2 + 1)}
    return min(d.items(), key=lambda kv: kv[1])[0]


def mannwhitney_p(a, b, rng) -> float:
    obs = float(np.median(a) - np.median(b))
    pool = np.concatenate([a, b]); n = len(a)
    cnt = 0
    for _ in range(N_PERM):
        p = pool.copy(); rng.shuffle(p)
        cnt += abs(float(np.median(p[:n]) - np.median(p[n:]))) >= abs(obs)
    return (cnt + 1) / (N_PERM + 1)


def main() -> int:
    man = {m["tag"]: m for m in json.loads((OUT / "manifest.json").read_text())}
    rows = []
    for f in sorted(glob.glob(str(OUT / "*.npy"))):
        m = man.get(pathlib.Path(f).stem)
        if not m:
            continue
        s = np.load(f).astype(np.float64)[:, -1, :]
        rows.append({**m, "R": rotation_power(s, PERIOD, TAIL), "rot": dominant_period(s) == PERIOD})

    by = collections.defaultdict(list)
    for r in rows:
        by[r["pair"]].append(r)
    mixed = sorted(w for w, v in by.items() if len({r["rot"] for r in v}) > 1)
    det = {w: v for w, v in by.items() if len({r["rot"] for r in v}) == 1}

    rot = np.array([r["R"] for v in det.values() for r in v if r["rot"]])
    set_ = np.array([r["R"] for v in det.values() for r in v if not r["rot"]])
    mix = [r for w in mixed for r in by[w]]
    rng = random.Random(SEED)

    print(f"{len(rows)} orbits; {len(det)} deterministic nouns, {len(mixed)} mixed {mixed}")
    p1 = mannwhitney_p(rot, set_, rng)
    print(f"\nP1 INSTRUMENT NULL -- does R separate the two known populations?")
    print(f"  rotating  n={len(rot):3d}  median R {np.median(rot):.4f}  "
          f"[{np.percentile(rot, 10):.4f}, {np.percentile(rot, 90):.4f}]")
    print(f"  settling  n={len(set_):3d}  median R {np.median(set_):.4f}  "
          f"[{np.percentile(set_, 10):.4f}, {np.percentile(set_, 90):.4f}]")
    print(f"  permutation p = {p1:.4f}  ->  {'OK' if p1 < 0.01 else 'FAILS -- STOP HERE'}")
    if p1 >= 0.01:
        print("\n  R does not measure what the binary label measures. Nothing below is read.")
        return 1

    print(f"\nP4 CONFOUND -- what do the mixed nouns' four orbits split on?")
    for w in mixed:
        for r in sorted(by[w], key=lambda r: (r["seq"], r["marker"])):
            print(f"  {w:6s} marker={r['marker']}  seq=[{r['seq']}]  "
                  f"rot={str(r['rot']):5s}  R={r['R']:.4f}")

    pooled = np.sort(np.concatenate([rot, set_]))
    pct = [float(np.searchsorted(pooled, r["R"]) / len(pooled)) for r in mix]
    cross = float(np.searchsorted(pooled, 0.5 * (np.median(rot) + np.median(set_))) / len(pooled))
    obs = float(np.mean([abs(p - cross) for p in pct]))
    draws = []
    for _ in range(N_PERM):
        s = [pooled[rng.randrange(len(pooled))] for _ in mix]
        draws.append(float(np.mean([abs(np.searchsorted(pooled, v) / len(pooled) - cross)
                                    for v in s])))
    pv = (sum(1 for d in draws if d <= obs) + 1) / (N_PERM + 1)
    print(f"\nP2 PRIMARY -- do mixed orbits sit at the crossover?")
    print(f"  crossover percentile {cross:.3f}; mixed orbit percentiles "
          f"{[round(p, 3) for p in pct]}")
    print(f"  mean |percentile - crossover| = {obs:.3f}   random-orbit null "
          f"{np.mean(draws):.3f} (sd {np.std(draws):.3f})")
    print(f"  p = {pv:.4f}  ->  {'AT THE BOUNDARY' if pv < 0.05 else 'NOT DISTINGUISHABLE'}")

    gaps = np.diff(pooled)
    g = float(gaps.max() / (pooled[-1] - pooled[0]))
    gd = []
    for _ in range(N_PERM):
        u = np.sort(np.array([rng.uniform(pooled[0], pooled[-1]) for _ in pooled]))
        gd.append(float(np.diff(u).max() / (u[-1] - u[0])))
    pg = (sum(1 for d in gd if d >= g) + 1) / (N_PERM + 1)
    print(f"\nP3 BIMODALITY -- is there a gap, i.e. a threshold rather than a continuum?")
    print(f"  largest relative gap {g:.4f} at R = {pooled[int(np.argmax(gaps))]:.4f}; "
          f"uniform null {np.mean(gd):.4f}, p = {pg:.4f}")
    print(f"  -> {'GAPPED (two regimes)' if pg < 0.05 else 'no gap (continuum)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
