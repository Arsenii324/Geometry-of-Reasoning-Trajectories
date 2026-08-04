"""Where does Huginn's contraction live: the attention branch or the MLP branch?

WHAT D59 LEFT OPEN
    D59 scaled `attn.proj` AND `mlp.proj` together by (1+eps) and found rho is
    steerable -- paired slope +0.2853 +- 0.0450, p=2.3e-04 -- but that the
    pre-registered prediction d(rho)/d(eps) = rho = 0.887 FAILED, measured/predicted
    = 0.322. So the two out-projections together carry only about a third of the
    contraction and the skip/RMSNorm structure carries the rest.

    That was a single scalar applied to 8 matrices at once, so it measures the
    branches' AGGREGATE share. D59(4) named the split as the obvious next step.

PRE-REGISTERED PREDICTIONS, written before running
    1. ADDITIVITY. To first order the two branches enter the Jacobian as separate
       additive terms, so slope(attn) + slope(mlp) should equal slope(both) =
       +0.2853 within error. A large super- or sub-additivity means the branches
       interact through the sandwich norms and the first-order picture is wrong.
    2. The MLP branch should dominate: its inner width is 17920 against attention's
       5280, and the gated-SiLU MLP is where most of the block's parameters and
       most of its nonlinearity sit.
    3. REPRODUCTION. Two `both` points (eps = -0.10, +0.10) are re-measured. They
       must reproduce D59's values within the paired noise, or the comparison
       across runs is invalid and nothing here can be attributed to the split.

Paired analysis throughout: the same 12 prompts at every setting, so the
between-prompt sd (0.056, D59(3)) cancels rather than swamping a 0.03 effect.
"""
# @needs: run prompts_by_family

import json

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MAX_R = 128
EPSILONS = (-0.10, -0.05, 0.0, 0.05, 0.10)
ARMS = ("attn", "mlp", "both")          # "both" is the D59 reproduction arm
BOTH_EPS = (-0.10, 0.10)                # only two points needed to reproduce
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

    def targets_for(arm):
        paths = {"attn": ("attn.proj",), "mlp": ("mlp.proj",),
                 "both": ("attn.proj", "mlp.proj")}[arm]
        out = []
        for blk in model.transformer.core_block:
            for path in paths:
                mod = blk
                for part in path.split("."):
                    mod = getattr(mod, part)
                out.append(mod)
        return out

    all_mods = targets_for("both")
    orig = {id(m): m.weight.detach().clone().cpu() for m in all_mods}
    print(f"targets: attn {len(targets_for('attn'))}, mlp {len(targets_for('mlp'))}, "
          f"both {len(all_mods)}", flush=True)

    def restore():
        with torch.no_grad():
            for m in all_mods:
                m.weight.copy_(orig[id(m)].to(m.weight.device))

    results = {}
    for arm in ARMS:
        mods = targets_for(arm)
        for eps in (BOTH_EPS if arm == "both" else EPSILONS):
            restore()
            with torch.no_grad():
                for m in mods:
                    m.weight.mul_(1.0 + eps)
            print(f"\n=== {arm}  eps = {eps:+.3f} ===", flush=True)
            rows = measure(model, tok, prompts, f"{arm}{eps:+.2f}")
            results[f"{arm}|{eps:+.3f}"] = {"arm": arm, "eps": eps, "rows": rows}
            with open("eps_split.json", "w") as f:
                json.dump(results, f, indent=1)
            r = np.nanmean([x["rho_plain_tangent"] for x in rows])
            if np.isfinite(r) and r > ABORT_RHO:
                print(f"  ABORT BAND: rho={r:.4f}", flush=True)
                break
    restore()

    print("\n=== PAIRED SLOPES: where does the contraction live? ===")
    from scipy.stats import linregress
    base = {x["prompt"]: x["rho_plain_tangent"]
            for x in results["attn|+0.000"]["rows"]}
    slopes = {}
    for arm in ARMS:
        pts = sorted(((v["eps"], v["rows"]) for k, v in results.items()
                      if v["arm"] == arm), key=lambda t: t[0])
        xs, ys = [], []
        for e, rows in pts:
            dr = [r["rho_plain_tangent"] - base[r["prompt"]] for r in rows
                  if np.isfinite(r["rho_plain_tangent"]) and np.isfinite(base[r["prompt"]])]
            xs.append(e)
            ys.append(float(np.mean(dr)))
        if len(xs) >= 3:
            lr = linregress(xs, ys)
            slopes[arm] = (lr.slope, lr.stderr, lr.pvalue)
            print(f"  {arm:>5}: slope {lr.slope:+.4f} +- {lr.stderr:.4f}  p={lr.pvalue:.3g}")
        else:
            d = dict(zip(xs, ys, strict=True))
            sl = (d[BOTH_EPS[1]] - d[BOTH_EPS[0]]) / (BOTH_EPS[1] - BOTH_EPS[0])
            slopes[arm] = (sl, float("nan"), float("nan"))
            print(f"  {arm:>5}: slope {sl:+.4f} (2-point, reproduction arm)")

    if {"attn", "mlp", "both"} <= set(slopes):
        sa, sm, sb = (slopes[k][0] for k in ("attn", "mlp", "both"))
        print(f"\n  PREDICTION 1 (additivity): attn + mlp = {sa + sm:+.4f} "
              f"vs both = {sb:+.4f}   ratio {(sa + sm) / sb if sb else float('nan'):.3f}")
        print(f"  PREDICTION 2 (mlp dominates): mlp/attn = "
              f"{sm / sa if sa else float('nan'):.2f}  -> "
              f"{'CONFIRMED' if sm > sa else 'REFUTED, attention dominates'}")
        print(f"  PREDICTION 3 (reproduces D59 slope +0.2853): both = {sb:+.4f}, "
              f"deviation {sb - 0.2853:+.4f}")

    print("\n=== STAGE 0: estimator reconciliation at eps = 0 ===")
    z = results.get("attn|+0.000", {}).get("rows", [])   # eps=0 is arm-independent
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

    print("\n  (Stage 1 is superseded here by the PAIRED SLOPES block above, which\n"
          "   is the correct analysis: unpaired means carry the 0.056 between-prompt\n"
          "   sd, and this run's whole purpose is a difference between arms.)")


main()
