"""Figure 1 -- the onset curve (C7 / D198), regenerated from the banked run.

WHAT THE X AXIS IS, because it is ambiguous three ways and the paper's hinge number is read
off it. `ds_transrot/job.py:208` computes

    slide = [(s, rotation_power(T[s:s + 12], period=6, tail=12))
             for s in range(0, T.shape[0] - 12 + 1, 4)]

so `s` is the window's **START** and the window spans unrolls **[s, s+12)**. The point plotted
at s = 16 therefore summarises unrolls 16-27, not unroll 16. "Onset between unrolls 12 and 16"
means the window beginning at 12 still shows no separation and the one beginning at 16 does.
Plotting this against a window *centre* or *end* would move the paper's hinge while producing a
perfectly plausible curve, which is failure class 1 in directions.md section O.

The 12-unroll width is not a choice: `rotation_power` needs two full cycles at period 6, so 12
is the smallest window the statistic accepts. That is why no period-6 statistic can describe
the state at the moment the answer is decided (median unroll 4, D159).

GATE: the medians recomputed here must equal the values C7/D198 quote. The script raises rather
than drawing a figure that has drifted from the ledger.
"""
import json
import pathlib
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROT, SET = ("symbol", "symptom"), ("element", "token")
THRESHOLD = 0.6677          # D141, the regime label's threshold
DECISION_UNROLL = 4         # D159: median unroll at which the gold attains its best rank
WIDTH = 12                  # = ds_transrot EARLY; two cycles at period 6, the statistic's floor

# C7's quoted numbers, as an exact regression check on the whole pipeline.
EXPECTED = {
    "rotating": {0: 0.23, 4: 0.24, 8: 0.24, 12: 0.29, 16: 0.56, 20: 0.74, 24: 0.84, 52: 0.97},
    "settling": {0: 0.23, 4: 0.24, 8: 0.25, 12: 0.29, 16: 0.34, 20: 0.40, 24: 0.45, 52: 0.55},
}

root = pathlib.Path(__file__).resolve().parent.parent
data = json.loads((root / "scratch/ds_transrot/transrot.json").read_text())
reg = [r for r in data["rows"] if r["kind"] == "regime"]
assert data["early"] == WIDTH, f"window width changed: {data['early']}"

curves = {}
for grp, name in ((ROT, "rotating"), (SET, "settling")):
    v = [r for r in reg if r["label"] in grp]
    starts = [s for s, _ in v[0]["slide"]]
    curves[name] = (starts, [st.median([dict(x["slide"])[s] for x in v]) for s in starts], len(v))
    # the first sliding window IS R_early by construction -- free identity check
    for x in v:
        assert abs(dict(x["slide"])[0] - x["R_early"]) < 1e-12, "slide[0] != R_early"

for name, exp in EXPECTED.items():
    starts, med, _ = curves[name]
    got = dict(zip(starts, med))
    for s, want in exp.items():
        assert abs(got[s] - want) < 0.005, f"{name} u{s}: {got[s]:.4f} != documented {want}"

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.axvspan(0, DECISION_UNROLL, color="0.88", zorder=0)
ax.axhline(THRESHOLD, color="0.55", ls=":", lw=0.9, zorder=1)
ax.text(53, THRESHOLD + 0.015, "regime threshold", ha="right", va="bottom",
        fontsize=7.5, color="0.35")
ax.text(DECISION_UNROLL + 1.0, 0.045, "answer decided\n(median unroll 4)",
        fontsize=7.5, color="0.35", va="bottom")

style = {"rotating": ("o", "-", "0.05"), "settling": ("s", "--", "0.45")}
for name in ("rotating", "settling"):
    starts, med, n = curves[name]
    m, ls, c = style[name]
    ax.plot(starts, med, ls, marker=m, ms=3.4, lw=1.4, color=c,
            label=f"{name} prompts (n={n})", zorder=3)

ax.set_xlabel("start unroll $s$ of the 12-unroll window $[s,\\,s{+}12)$")
ax.set_ylabel("rotation power $R$ (period 6)")
ax.set_xlim(-1.5, 54)
ax.set_ylim(0, 1.0)
ax.set_xticks(range(0, 53, 8))
ax.legend(loc="lower right", fontsize=8, frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout(pad=0.3)

for out in (root / "docs/submission/build/pic/onset.pdf",
            root.parent / "files/paper_submission/draft/pic/onset.pdf"):
    fig.savefig(out)
    print("wrote", out)
print("gate passed: medians match C7/D198; slide[0] == R_early on all rows")
