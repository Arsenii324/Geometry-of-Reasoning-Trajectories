"""Stage 0 + Stage 1: reconcile the rho estimator, then steer rho by a static rescale.

THE INSTRUMENT, AND WHY IT IS NOT PEFT
    An external research pass (docs/briefs/peft_for_spectral_control.md; report in
    files/deep_research_A/) concluded that a low-rank adapter is the WRONG
    instrument for *moving* rho in a predicted way, and that the right one is a
    gradient-free scalar rescale of the core block's two output projections.

    Huginn's own init downscales every out-projection by 1/sqrt(2 * d_eff) with
    d_eff = l_P + r_bar*l_R + l_C = 2 + 32*4 + 2 = 132 (paper section 4.1). Those
    projections therefore carry the architecture's built-in per-unroll
    branch-magnitude knob. In a sandwich-residual block the Jacobian is a
    skip/norm structure S plus a branch Jacobian B that is left-multiplied by the
    out-projection weights, so scaling `attn.proj` and `mlp.proj` by (1+eps)
    scales B by (1+eps) to first order.

    A-PRIORI PREDICTION, made before running: if the contractive deficit (1-rho)
    is branch-dominated then d(rho)/d(scale) ~ rho, so

        Delta_rho ~ rho * eps

    Concretely: eps = +0.072 should take rho 0.858 -> 0.92, and eps = +0.13 is the
    hard-abort band (rho -> 0.97). SUCCESS IS NOT "rho moves" -- rho will surely
    move. Success is that rho(eps) TRACKS rho*eps within the noise floor at small
    |eps|. A measured slope that departs from rho calibrates the branch's spectral
    share instead, which is a result either way.

STAGE 0 IS BLOCKING AND IS RUN AT eps = 0
    Our two estimators disagree by 0.058 -- two-orbit convergence gives 0.858,
    Arnoldi on Jacobian-vector products gives ~0.80 -- and that gap is TWICE the
    0.03 we declared detectable. An intervention cannot be attributed to "rho"
    while two estimates of rho differ by more than the effect. Three fixes are
    applied here, each recommended by the report:

    (a) TANGENT PROJECTION. RMSNorm pins the state to a sphere (||h|| = 76.37 with
        0.007% variation), so the radial direction is degenerate and the meaningful
        spectrum lives on the (d-1)-dimensional tangent space. The orbit difference
        is projected orthogonal to the state before its norm is taken.
    (b) ENVELOPE FIT. The leading eigenvalue is complex (3/3 prompts, period
        2.6-6.0 unrolls), so ||h1-h2|| oscillates around an exponential envelope. A
        log-linear fit over a non-integer number of periods is biased in a way our
        stated +0.002..+0.015 estimator bias does not capture. Both fits are
        reported: plain (as in D52) and envelope-through-peaks.
    (c) BETWEEN-PROMPT VARIANCE, which is unmeasured. Our Arnoldi figure came from
        3 prompts. If between-prompt sd is >~0.03 it swamps the detectability floor
        and every target must become a prompt-set mean. 12 prompts across 4
        families are used and the sd is reported.

    Also recorded: whether the orbits actually converge (fixed point) or plateau
    (limit cycle), since a complex leading eigenvalue admits either, and at r=32
    with a ~6.6-unroll time constant we may not be near a fixed point at all.

float32 throughout, to stay comparable with every prior rho measurement in this
project (D44, D52) and because this model is precision-sensitive (D30).
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run prompts_by_family


def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


def prompts_by_family(n_per=3, m=64):
    """Four task families with token lengths spanning ~15 to ~74.

    Stratifying by family is what lets a per-model quantity be tested for
    constancy across tasks instead of assumed (claims_ledger D45). Length varies
    WITH family here by design, so the two are collinear -- a real limitation,
    recorded rather than hidden.
    """
    import random
    out = []
    for i in range(n_per):
        rng = random.Random(i * 7919)
        bits = [1 if rng.random() < 0.5 else 0 for _ in range(m)]
        out.append(("counting",
                    "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:"))
    for i in range(n_per):
        rng = random.Random(i * 104729)
        seq, d = [], 0
        for _ in range(m // 2):
            if d == 0 or (rng.random() < 0.5 and d < 8):
                seq.append("(")
                d += 1
            else:
                seq.append(")")
                d -= 1
        seq += [")"] * d
        out.append(("nesting",
                    "String: " + " ".join(seq) + ". What is the maximum nesting depth? A:"))
    for i in range(n_per):
        rng = random.Random(i * 15485863)
        a, b, c = rng.randint(11, 99), rng.randint(3, 19), rng.randint(2, 9)
        out.append(("arith",
                    f"A shop had {a} boxes. It sold {b} boxes each day for {c} days. "
                    f"How many boxes are left? A:"))
    stems = ["The man picked up the heavy suitcase and walked toward the platform. He",
             "She opened the oven, checked the bread, and decided it needed more time. Then she",
             "The dog heard the doorbell, ran into the hallway, and started barking. Next it"]
    for i in range(n_per):
        out.append(("commonsense", stems[i % len(stems)]))
    return out

import json

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MAX_R = 128
EPSILONS = (-0.10, -0.07, -0.05, -0.02, 0.0, 0.02, 0.05, 0.07, 0.10, 0.13)
ABORT_RHO = 0.98          # above this the map is effectively non-contracting


def fit_plain(y):
    """Log-linear fit over the pre-floor regime -> (rho, n_used, fit_r2)."""
    import numpy as np
    y = np.asarray(y, float)
    if len(y) < 6:
        return float("nan"), 0, float("nan")
    floor = float(np.median(y[-max(3, len(y) // 4):]))
    if not np.isfinite(floor) or floor <= 0:
        return float("nan"), 0, float("nan")
    n = 0
    for v in (y > 3.0 * floor):
        if not v:
            break
        n += 1
    if n < 4:
        return float("nan"), int(n), float("nan")
    t = np.arange(n, dtype=np.float64)
    ly = np.log(y[:n])
    sl, ic = np.polyfit(t, ly, 1)
    pred = sl * t + ic
    den = ((ly - ly.mean()) ** 2).sum()
    r2 = float(1 - ((ly - pred) ** 2).sum() / den) if den > 0 else float("nan")
    return float(np.exp(sl)), int(n), r2


def fit_envelope(y):
    """Fit through the LOCAL MAXIMA, so an oscillating decay is not fitted through
    its troughs. Returns (rho, n_peaks, fit_r2, half_period).

    The floor cut is applied to the PEAKS, not to a leading contiguous run. A
    contiguous-run rule terminates at the first trough of an oscillating series
    rather than at the arithmetic floor -- a synthetic check at rho=0.86,
    period=3.2 found zero peaks that way, which is precisely the case this fit
    exists to handle.

    The returned spacing is between successive maxima of a NORM, which oscillates
    at twice the rotation frequency (an elliptical spiral passes its major axis
    twice per turn), so the rotation period is ~2x this value.
    """
    import numpy as np
    y = np.asarray(y, float)
    if len(y) < 8:
        return float("nan"), 0, float("nan"), float("nan")
    floor = float(np.median(y[-max(3, len(y) // 4):]))
    if not np.isfinite(floor) or floor <= 0:
        return float("nan"), 0, float("nan"), float("nan")
    pk = [i for i in range(1, len(y) - 1)
          if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] > 3.0 * floor]
    if len(pk) < 3:
        return float("nan"), len(pk), float("nan"), float("nan")
    t = np.asarray(pk, float)
    ly = np.log(y[pk])
    sl, ic = np.polyfit(t, ly, 1)
    pred = sl * t + ic
    den = ((ly - ly.mean()) ** 2).sum()
    r2 = float(1 - ((ly - pred) ** 2).sum() / den) if den > 0 else float("nan")
    half_period = float(np.mean(np.diff(t))) if len(t) > 1 else float("nan")
    return float(np.exp(sl)), len(pk), r2, half_period


def orbit(model, ids, seed, max_r):
    import numpy as np
    import torch
    cap = []
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()
    h = mod.register_forward_hook(
        lambda m, i, o, c=cap: c.append(o.detach()[0, -1, :].float().cpu().numpy()))
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            model(input_ids=ids, num_steps=max_r)
    finally:
        h.remove()
    return np.stack(cap)


def measure(model, tok, prompts, label):
    """rho per prompt, by four estimators: {plain, envelope} x {ambient, tangent}."""
    import numpy as np
    import torch
    rows = []
    for i, (family, p) in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        a = orbit(model, ids, 1000 + i, MAX_R)
        b = orbit(model, ids, 2000 + i, MAX_R)
        d = a - b
        amb = np.linalg.norm(d, axis=1)
        # (a) TANGENT PROJECTION: remove the radial component along the state
        mid = 0.5 * (a + b)
        nrm = np.linalg.norm(mid, axis=1, keepdims=True)
        unit = mid / np.maximum(nrm, 1e-12)
        tan = np.linalg.norm(d - (d * unit).sum(1, keepdims=True) * unit, axis=1)
        rp, np_, fp = fit_plain(amb)
        rt, nt, ft = fit_plain(tan)
        re_, ne, fe, per = fit_envelope(tan)
        rows.append({"prompt": i, "family": family,
                     "rho_plain_ambient": rp, "n_plain": np_, "r2_plain": fp,
                     "rho_plain_tangent": rt, "n_tan": nt, "r2_tan": ft,
                     "rho_envelope_tangent": re_, "n_peaks": ne, "r2_env": fe,
                     "half_period": per,
                     "d0": float(amb[0]), "d_end": float(amb[-1]),
                     "radial_frac": float(np.median(1 - tan[:20] / np.maximum(amb[:20], 1e-12))),
                     "norm_h": float(np.linalg.norm(a[-1]))})
        del ids
        torch.cuda.empty_cache()
    print(f"    {label}: " + "  ".join(
        f"{k.split('_')[1][:4]}{'/'+k.split('_')[2][:3] if len(k.split('_'))>2 else ''}="
        f"{np.nanmean([r[k] for r in rows]):.4f}"
        for k in ("rho_plain_ambient", "rho_plain_tangent", "rho_envelope_tangent")),
        flush=True)
    return rows


def main():
    run("pip install -q 'transformers>=4.50,<4.54' scipy")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    prompts = prompts_by_family(n_per=3)
    print(f"{len(prompts)} prompts across {len({f for f, _ in prompts})} families",
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True).to(torch.float32).to("cuda").eval()

    # the two out-projections in every core block -- the branch-magnitude handle
    targets = []
    for blk in model.transformer.core_block:
        for path in ("attn.proj", "mlp.proj"):
            mod = blk
            for part in path.split("."):
                mod = getattr(mod, part)
            targets.append((path, mod))
    print(f"targets: {len(targets)} matrices "
          f"({[(p, tuple(m.weight.shape)) for p, m in targets[:2]]})", flush=True)
    orig = [m.weight.detach().clone().cpu() for _, m in targets]

    results = {}
    for eps in EPSILONS:
        with torch.no_grad():
            for (_, m), w in zip(targets, orig, strict=True):
                m.weight.copy_(w.to(m.weight.device) * (1.0 + eps))
        print(f"\n=== eps = {eps:+.3f} ===", flush=True)
        rows = measure(model, tok, prompts, f"eps{eps:+.2f}")
        results[f"{eps:+.3f}"] = {"eps": eps, "rows": rows}
        with open("eps_sweep.json", "w") as f:
            json.dump(results, f, indent=1)
        r_env = np.nanmean([r["rho_envelope_tangent"] for r in rows])
        if np.isfinite(r_env) and r_env > ABORT_RHO:
            print(f"  ABORT BAND: rho_envelope = {r_env:.4f} > {ABORT_RHO}; "
                  "stopping the sweep", flush=True)
            break

    # restore
    with torch.no_grad():
        for (_, m), w in zip(targets, orig, strict=True):
            m.weight.copy_(w.to(m.weight.device))

    print("\n=== STAGE 0: estimator reconciliation at eps = 0 ===")
    z = results.get("+0.000", {}).get("rows", [])
    if z:
        for k, lbl in (("rho_plain_ambient", "plain / ambient  (as in D52)"),
                       ("rho_plain_tangent", "plain / tangent"),
                       ("rho_envelope_tangent", "envelope / tangent")):
            v = np.array([r[k] for r in z], float)
            v = v[np.isfinite(v)]
            print(f"  {lbl:<28} {v.mean():.4f} +- {v.std(ddof=1):.4f}  (n={len(v)})")
        print(f"  median radial fraction of the orbit difference: "
              f"{np.median([r['radial_frac'] for r in z]):.4f}")
        hp = np.nanmean([r["half_period"] for r in z])
        print(f"  peak spacing {hp:.2f} unrolls -> rotation period ~{2 * hp:.2f} "
              f"(Arnoldi gave 2.6-6.0)")
        print(f"  d_end/d0 (fixed point if <<1, limit cycle if ~1): "
              f"{np.mean([r['d_end'] / max(r['d0'], 1e-12) for r in z]):.2e}")
        bp = np.nanstd([r["rho_envelope_tangent"] for r in z], ddof=1)
        print(f"  BETWEEN-PROMPT sd = {bp:.4f}  -> "
              f"{'SWAMPS' if bp > 0.03 else 'below'} the 0.03 detectability floor")
        print("  Arnoldi anchor (D31, ambient, 3 prompts): 0.80")

    print("\n=== STAGE 1: does rho(eps) track the prediction rho*eps? ===")
    print(f"  {'eps':>7} {'rho_env/tan':>12} {'rho_plain/tan':>14} "
          f"{'predicted':>10} {'resid':>8}")
    base = np.nanmean([r["rho_envelope_tangent"] for r in z]) if z else float("nan")
    xs, ys = [], []
    for _key, v in sorted(results.items(), key=lambda kv: kv[1]["eps"]):
        e = v["eps"]
        re_ = np.nanmean([r["rho_envelope_tangent"] for r in v["rows"]])
        rp = np.nanmean([r["rho_plain_tangent"] for r in v["rows"]])
        pred = base * (1 + e)
        xs.append(e)
        ys.append(re_)
        print(f"  {e:>+7.3f} {re_:>12.4f} {rp:>14.4f} {pred:>10.4f} {re_ - pred:>+8.4f}")
    xs, ys = np.array(xs), np.array(ys)
    ok = np.isfinite(ys)
    if ok.sum() >= 4:
        slope, icpt = np.polyfit(xs[ok], ys[ok], 1)
        print(f"\n  measured d(rho)/d(eps) = {slope:+.4f}")
        print(f"  PREDICTED               = {base:+.4f}   (the prediction is slope == rho)")
        print(f"  ratio measured/predicted = {slope / base:.3f}")
        if abs(slope / base - 1) < 0.35:
            print("  => the linearisation HOLDS. rho is steerable by a scalar rescale,")
            print("     with a predicted magnitude. This is a predicted intervention.")
        else:
            print("  => the linearisation FAILS. The branch is not the dominant")
            print("     contractive term; the measured slope calibrates its share.")
        eps_star = (0.92 - base) / slope if slope else float("nan")
        print(f"  eps needed for rho -> 0.92: {eps_star:+.4f} "
              f"(a-priori estimate was +0.072)")


main()
