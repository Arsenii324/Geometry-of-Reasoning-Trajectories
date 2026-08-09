"""Does a prompt's CONTRACTION RATE predict when its answer becomes readable?

This is the first test in the project that puts a geometric quantity and a
behavioural quantity on the SAME prompts. It costs zero GPU: B1c measures rho per
recipient prompt, B4 measured best_depth per problem, and every one of B1c's 16
recipients (v in {0,1,2}, k in 1..7) exists in B4's 36-problem set. The join is
free; it was not available before D111 because B1b's pairs all shared one recipient.

THE PREDICTION, WRITTEN BEFORE THE JOIN IS RUN.

D111 measures rho per prompt from how fast an injected state perturbation dies:
potency ~ rho^(R-r). D110 measures best_depth per problem, the unroll at which the
gold answer is best available. If the second is DOWNSTREAM of the first -- if the
answer becomes readable once the state has settled -- then:

  P1  rho and best_depth are POSITIVELY correlated. A larger rho is a SLOWER
      contraction, so the fixed point arrives later and the answer with it.
  P2  Since D110 found harder problems peak EARLIER, mediation requires
      rho(k, rho_contraction) < 0: harder problems must contract FASTER.
  P3  If P1 holds and P2 fails, the two effects are independent and D110's
      reversal is NOT a contraction-rate phenomenon -- which is the more
      interesting outcome, because it means difficulty acts on the readout
      rather than on the dynamics.

WHAT WOULD FALSIFY THE WHOLE FRAME. A NULL or NEGATIVE P1. Note that a null here is
strongly expected on one reading of the existing numbers, and it is worth saying so
in advance rather than discovering it afterwards: at rho = 0.822 a displacement is
only halved by unroll 3.5 (0.822^3.5 = 0.50), yet best_depth sits at ~3.5 and D97
puts full convergence at a median of 34 unrolls. **The answer is decodable while
roughly half the initial displacement is still present.** If the readout does not
wait for the fixed point, then the rate at which the fixed point is approached need
not govern when the answer appears, and P1 should fail. That is the readout-blind-
spot story of thread A1 (arXiv:2606.24898) arriving from a second direction.

THE FLOOR, SO A NULL FROM THIS IS TRUSTWORTHY (CLAUDE.md section 5). Simulated at
n = 16 with rho refitted from synthetic potency curves through the SAME code path,
60 draws per cell, two-sided alpha = 0.05:

    planted rho-spread   noise sd 0.05   noise sd 0.25
      0.000 (no effect)        5%              5%      <- correct false-positive rate
      0.012                  100%             45%
      0.024                  100%             98%
      0.045                  100%            100%

B1b's per-donor R^2 of 0.83-0.98 implies a log-potency noise sd in the 0.05-0.25
band, and its observed rho spread was 0.075 (0.791-0.866). So this test detects an
effect three times SMALLER than the spread already seen, at the worst noise level
plausible. A null here means absence, not incapacity -- which is exactly what D93's
detection floor was built to establish and what run_h0_within.py originally failed
to check.

The join was also validated end-to-end on a synthetic B1c file with a planted
rho = 0.78 + 0.02 * best_depth: recovered at rho = +0.97, p = 0.0001, while the
unplanted P2 arm correctly returned p = 0.92.

Either way this resolves something. Run it when B1c lands.
"""

from __future__ import annotations

import collections
import json
import pathlib
import random
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from traj_geom.rigor import require_units  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
B1C = ROOT / "scratch/ds_estream_b1c/out/estream.json"
B4 = ROOT / "scratch/ds_addk/out/grid.json"
FIT_R = (8, 16, 24, 32, 40)


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def perm_two_sided(x, y, n=20000, seed=0):
    obs = spearman(x, y)
    rng = random.Random(seed)
    yy = list(y)
    ge = le = 0
    for _ in range(n):
        rng.shuffle(yy)
        r = spearman(x, yy)
        ge += r >= obs
        le += r <= obs
    pp, pn = (ge + 1) / (n + 1), (le + 1) / (n + 1)
    return obs, min(1.0, 2 * min(pp, pn))


def main() -> int:
    if not B1C.exists():
        print(f"B1c output not present yet: {B1C}\nNothing to do -- rerun when it lands.")
        return 1

    b1c = json.loads(B1C.read_text())
    if "records" not in b1c:
        print(f"B1c returned {b1c.get('verdict', 'no records')} -- VOID, nothing to join.")
        return 1

    # rho per RECIPIENT, refit here from the raw records rather than trusting the
    # kernel's summary (CLAUDE.md section 4).
    import numpy as np
    per = collections.defaultdict(dict)
    meta = {}
    for x in b1c["records"]:
        per[x["pair"]][x["r"]] = abs(x["state_potency"])
        meta[x["pair"]] = x.get("rec_gold")
    rho_by_pair = {}
    for pair, d in per.items():
        pts = [(r, d[r]) for r in FIT_R if d.get(r, 0.0) > 0.0]
        if len(pts) < 4:
            continue
        xs = np.array([p[0] for p in pts], float)
        ys = np.log(np.array([p[1] for p in pts], float))
        slope, _ = np.polyfit(xs, ys, 1)
        rho_by_pair[pair] = float(np.exp(-slope))

    # B4: best_depth and accuracy per (k, gold) problem
    rows = json.loads(B4.read_text())["rows"]
    cells = collections.defaultdict(list)
    for r in rows:
        if r.get("ok"):
            cells[(r["level"], str(r["gold"]))].append(r)
    bd = {k: st.median(x["best_depth"] for x in v) for k, v in cells.items()}
    acc = {k: st.mean(x["correct"] for x in v) for k, v in cells.items()}

    # B1c's recipient index -> its (v, k); regenerate the item list the job used
    items = [{"v": v, "k": k, "gold": str(v + k)}
             for v in range(0, 10) for k in range(1, 8) if v + k <= 9]

    joined = []
    for pair, rho in rho_by_pair.items():
        rec_i = int(pair.split("->")[0])
        if rec_i >= len(items):
            continue
        it = items[rec_i]
        key = (it["k"], it["gold"])
        if key in bd:
            joined.append({"k": it["k"], "v": it["v"], "gold": it["gold"],
                           "rho": rho, "best_depth": bd[key], "acc": acc[key]})

    print(f"joined {len(joined)} prompts present in BOTH B1c and B4")
    if len(joined) < 8:
        print("  too few to test -- report as VOID, not as a null.")
        return 1
    require_units([(j["k"], j["gold"]) for j in joined], min_units=8,
                  what="rho-vs-behaviour join")

    rr = [j["rho"] for j in joined]
    dd = [j["best_depth"] for j in joined]
    kk = [j["k"] for j in joined]
    aa = [j["acc"] for j in joined]
    print(f"  rho: median {st.median(rr):.4f}, range {min(rr):.4f}-{max(rr):.4f}, "
          f"sd {st.pstdev(rr):.4f}")

    o, p = perm_two_sided(rr, dd)
    print(f"\nP1  rho(contraction) vs best_depth : {o:+.4f}  two-sided p = {p:.4f}"
          f"   [predicted POSITIVE]")
    o2, p2 = perm_two_sided(kk, rr)
    print(f"P2  difficulty k vs rho(contraction): {o2:+.4f}  two-sided p = {p2:.4f}"
          f"   [mediation needs NEGATIVE]")
    o3, p3 = perm_two_sided(rr, aa)
    print(f"    rho(contraction) vs accuracy    : {o3:+.4f}  two-sided p = {p3:.4f}"
          f"   [not predicted; reported to catch a confound]")

    print("\nREADING:")
    if p > 0.05:
        print("  P1 NULL. The rate at which the fixed point is approached does not"
              "\n  predict when the answer becomes readable. Combined with best_depth"
              "\n  ~3.5 against D97's median-34 convergence, the readout does not wait"
              "\n  for the fixed point -- the A1 readout-blind-spot mechanism, reached"
              "\n  independently. D110's reversal is then NOT a dynamics effect.")
    elif o > 0:
        print("  P1 CONFIRMED. Slower-contracting prompts hold their answer later,"
              "\n  so the readout IS downstream of the dynamics. Check P2 before"
              "\n  claiming this mediates D110.")
    else:
        print("  P1 REVERSED -- faster contraction, LATER answer. Neither the"
              "\n  mediation story nor the blind-spot story predicts this; treat as"
              "\n  a finding needing its own explanation, not a confirmation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
