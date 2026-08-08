"""Does the trajectory rotate at the rate the Jacobian says it should?

THE GAP THIS CLOSES. B4.15 measures arg(lambda) -- rotation per unroll read off the
operator at the converged state -- and treats it as H2's instrument because it has
no recording window and therefore cannot be governed by the recording budget the
way winding was (D28: winding's sign flips with `num_steps` alone). But that
argument has never been checked against the thing it claims to describe. The whole
Jacobian picture rests on the dynamics being LOCALLY LINEAR near h*, and nothing in
this project has tested that.

THE TEST IS A POINT PREDICTION, WHICH IS WHY IT IS WORTH RUNNING. If h_{t+1} - h* ~
A (h_t - h*) with A's leading eigenpair lambda = rho e^{i phi} and eigenvector v,
then in the real 2-D plane spanned by Re(v) and Im(v) the trajectory must rotate at
EXACTLY phi radians per unroll. So:

    measured angular rate in the eigenplane  /  |arg lambda|   ==   1.0

That ratio is the positive control the Jacobian approach has never had. If it holds,
rho and arg mean what D52/D55/B4.15 say they mean. If it does not, the map is not
locally linear at the scales we record and BOTH quantities are in question --
including D52, the project's best-replicated result.

AND IT IS ALSO THE ONLY CORRECT USE OF WINDING. Every winding variant W1-W9 failed
because it measured angle in an ARBITRARY plane (2-D PCA), about an arbitrary centre
(the centroid, which a settling path collapses toward), over an arbitrary window.
Here the plane is the one rotation actually happens in, the centre is the fixed
point, and the window is the pre-floor regime whose length rho itself predicts. The
metric is not being asked to support H2; it is being asked whether the operator
describes the orbit.

SECOND MEASUREMENT, FREE FROM THE SAME EIGENVECTORS: WHICH MODE CARRIES THE ANSWER.
d = grad_h logit(token) at h* is the direction in state space that moves a token's
logit. Its overlap |cos(d, v_i)| with each dynamical mode says whether the modes the
map rotates are the ones the readout reads. This project has measured geometry and
content separately for months and never connected them; one backward pass does it.

PRE-REGISTERED (CLAUDE.md section 1).
P1 -- GATE. The leading eigenpair must be complex and eigs must converge, else there
      is no eigenplane and nothing below can be computed.
P2 -- GATE, on the linear model itself. The projected orbit must be a decaying
      spiral: radius in the eigenplane must fall roughly geometrically over the
      pre-floor window (R^2 >= 0.9 on log radius vs t). If it does not, the leading
      mode is not what the orbit follows and the ratio in P3 is meaningless.
P3 -- THE POINT PREDICTION. measured_rate / |arg| in [0.8, 1.25] on a majority of
      prompts. Deliberately two-sided and not a p-value: this is a calibration
      check, and the interesting outcomes are "1.0" and "not 1.0", not "significant".
P4 -- CONNECTION. |cos(d_top1, eigenplane)| is compared against the same overlap for
      random directions in 5280-d, where chance is ~1/sqrt(5280) = 0.0138. An
      overlap at chance means the rotating modes are orthogonal to what the readout
      reads, which would be a clean negative and would bound how much trajectory
      geometry can possibly explain about the output.

Persists the full eigen-decomposition, the projected 2-D orbit and the radius curve
per prompt (B4.14: persist the curve, not the fit), so a question this design did
not anticipate can be answered without another GPU hour.
"""

import json
import math
import os
import random
import time
import traceback

import numpy as np

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
N_SETTLE = 96          # long enough that the pre-floor window is well sampled
N_EIGS = 8
TOL = 1e-5
MAX_MATVEC = 4000
N_OPS = (4, 8, 16, 32)
KINDS = ("track", "local")
SEEDS = (0,)
OUT = os.environ.get("RESULT_DIR", ".") + "/eigen.json"


def make_variants(n_ops, seed=0):
    """Verbatim from src/traj_geom/shapes/synthetic.py; see tests/test_kernel_tasks.py."""
    rng = random.Random(seed)
    ops = [rng.choice([1, -1]) for _ in range(n_ops)]
    body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1." for o in ops)
    return {
        "track": body + " Final total? A:",
        "local": body + " What was the last instruction? A:",
    }


def eigenplane_orbit(traj, h_star, v):
    """Project the orbit into the real plane of a complex eigenvector.

    Re(v) and Im(v) span an invariant real 2-D subspace of A for a complex pair.
    Gram-Schmidt makes them orthonormal so angles are undistorted -- skipping that
    would shear the plane and bias the measured rate, which is precisely the class
    of error that made PCA-plane winding uninterpretable.
    """
    e1 = np.real(v).astype(np.float64)
    e1 /= np.linalg.norm(e1) + 1e-30
    e2 = np.imag(v).astype(np.float64)
    e2 -= (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2) + 1e-30
    d = traj - h_star[None, :]
    return np.stack([d @ e1, d @ e2], axis=1)


def spiral_stats(xy, floor_frac=3.0):
    """Angular rate and radial decay of a 2-D orbit over its PRE-FLOOR window.

    The window is chosen by the data, not by a constant: the tail radius sets an
    arithmetic-noise floor (bf16 rounding, D30) and only steps a clear multiple
    above it carry signal. D28's lesson is that a statistic averaged over a window
    that mixes signal with floor reports the window, so the window is reported too.
    """
    r = np.linalg.norm(xy, axis=1)
    floor = float(np.median(r[-8:]))
    keep = np.where(r > floor_frac * floor)[0]
    if len(keep) < 6:
        return {"ok": False, "why": f"pre-floor window too short ({len(keep)} steps)",
                "floor": floor, "n_pre_floor": int(len(keep))}
    a, b = int(keep[0]), int(keep[-1])
    th = np.unwrap(np.arctan2(xy[a:b + 1, 1], xy[a:b + 1, 0]))
    steps = b - a
    rate = float(abs(th[-1] - th[0]) / steps)
    lr = np.log(np.clip(r[a:b + 1], 1e-30, None))
    t = np.arange(len(lr), dtype=float)
    slope, icept = np.polyfit(t, lr, 1)
    pred = slope * t + icept
    ss = float(1 - ((lr - pred) ** 2).sum() / max(((lr - lr.mean()) ** 2).sum(), 1e-30))
    return {"ok": True, "rate": rate, "n_pre_floor": steps + 1, "win": [a, b],
            "floor": floor, "radial_rho": float(np.exp(slope)), "radial_r2": ss,
            "total_turns": float(abs(th[-1] - th[0]) / (2 * math.pi))}


def main():
    import subprocess
    subprocess.check_call("pip install -q 'transformers>=4.50,<4.54' scipy", shell=True)
    import torch
    from scipy.sparse.linalg import LinearOperator, eigs
    from transformers import AutoModelForCausalLM, AutoTokenizer

    wb = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wb = wandb.init(project="geometry-of-reasoning-trajectories",
                            name="ds-eigen-" + str(int(time.time())),
                            config={"n_settle": N_SETTLE, "n_eigs": N_EIGS,
                                    "n_ops": list(N_OPS), "model": MODEL_ID})
        except Exception as exc:
            print(f"wandb unavailable: {exc}", flush=True)

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cuda").eval()

    def core_step(h, emb, freqs):
        bi = torch.tensor(0, device=h.device, dtype=torch.long)
        x = model.transformer.adapter(torch.cat([h, emb], dim=-1))
        for block in model.transformer.core_block:
            bi = bi + 1
            x = block(x, freqs, bi, None, None)
        return x

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    records = []

    def save():
        with open(OUT, "w") as fh:
            json.dump({"n_settle": N_SETTLE, "n_eigs": N_EIGS, "records": records}, fh)

    for n_ops in N_OPS:
        for seed in SEEDS:
            v = make_variants(n_ops, seed=seed)
            for kind in KINDS:
                t0 = time.time()
                tag = f"{kind} n_ops={n_ops} seed={seed}"
                try:
                    ids = tok(v[kind], return_tensors="pt").input_ids.to("cuda")
                    freqs = model.freqs_cis[:, : ids.shape[1]]
                    with torch.no_grad():
                        emb = model.transformer.wte(ids)
                        if model.emb_scale != 1:
                            emb = emb * model.emb_scale
                        for b in model.transformer.prelude:
                            emb = b(emb, freqs, torch.tensor(0), None, None)
                        h = model.initialize_state(emb)
                        traj = []
                        for _ in range(N_SETTLE):
                            h = core_step(h, emb, freqs)
                            traj.append(h[0, -1, :].detach().float().cpu().numpy())
                    traj = np.stack(traj)
                    hs = h.detach()
                    dim = hs.shape[-1]

                    # The readout direction: which way in state space raises a
                    # token's logit. One backward pass, and it is what connects the
                    # dynamics to the output.
                    x = hs.detach().clone().requires_grad_(True)
                    lg = coda_head(x, freqs)[0, -1]
                    top1 = int(lg.argmax().item())
                    (d_top1,) = torch.autograd.grad(lg[top1], x, retain_graph=False)
                    d_top1 = d_top1[0, -1, :].detach().float().cpu().numpy().astype(np.float64)

                    calls = [0]

                    class _Cap(Exception):
                        pass

                    def _rev(vec):
                        u = torch.zeros_like(hs)
                        u[0, -1, :] = torch.as_tensor(np.asarray(vec, dtype=np.float32),
                                                      device=hs.device, dtype=hs.dtype)
                        xx = hs.detach().clone().requires_grad_(True)
                        y = core_step(xx, emb, freqs)
                        (g,) = torch.autograd.grad(y, xx, grad_outputs=u)
                        return g[0, -1, :]

                    def matvec(vec):
                        calls[0] += 1
                        if calls[0] > MAX_MATVEC:
                            raise _Cap(f"matvec cap {MAX_MATVEC}")
                        return _rev(vec).detach().float().cpu().numpy().astype(np.float64)

                    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
                    # eig(J^T) = eig(J); the eigenVECTORS of J^T are the LEFT
                    # eigenvectors of J. For the invariant-plane test that is the
                    # right object only if A is normal, which it is not -- so this
                    # measures the plane the ADJOINT rotates in. Recorded explicitly
                    # rather than glossed; P2's radial-fit gate is what decides
                    # whether the plane found actually describes the orbit.
                    vals, vecs = eigs(op, k=N_EIGS, which="LM", tol=TOL, maxiter=400)
                    order = np.argsort(-np.abs(vals))
                    vals, vecs = vals[order], vecs[:, order]
                    osc = [i for i in range(len(vals)) if abs(vals[i].imag) > 1e-8]
                    rec = {"kind": kind, "n_ops": n_ops, "seed": seed,
                           "matvecs": calls[0], "n_tokens": int(ids.shape[1]),
                           "rho": float(abs(vals[0])),
                           "lead_is_real": bool(abs(vals[0].imag) <= 1e-8),
                           "n_osc": len(osc),
                           "eigs": [[float(z.real), float(z.imag)] for z in vals]}
                    if osc:
                        j = osc[0]
                        arg = float(abs(np.angle(vals[j])))
                        vv = vecs[:, j]
                        xy = eigenplane_orbit(traj, traj[-1], vv)
                        st = spiral_stats(xy)
                        rec.update({"arg": arg, "rho_osc": float(abs(vals[j])),
                                    "spiral": st,
                                    "xy": xy[::1].round(6).tolist(),
                                    "radius": np.linalg.norm(xy, axis=1).round(8).tolist()})
                        if st["ok"]:
                            rec["rate_over_arg"] = st["rate"] / arg if arg > 0 else None
                            rec["predicted_turns"] = arg * st["n_pre_floor"] / (2 * math.pi)
                        # P4: is the rotating plane visible to the readout at all?
                        e1 = np.real(vv) / (np.linalg.norm(np.real(vv)) + 1e-30)
                        dn = d_top1 / (np.linalg.norm(d_top1) + 1e-30)
                        rng = np.random.default_rng(0)
                        rnd = rng.normal(size=(64, dim))
                        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
                        rec["cos_readout_mode"] = float(abs(dn @ e1))
                        rec["cos_readout_random_mean"] = float(np.abs(rnd @ dn).mean())
                        rec["top1_token"] = top1
                    rec["ok"] = True
                except Exception as exc:
                    rec = {"kind": kind, "n_ops": n_ops, "seed": seed, "ok": False,
                           "why": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()}
                rec["secs"] = round(time.time() - t0, 1)
                records.append(rec)
                save()
                brief = {k: rec[k] for k in
                         ("kind", "n_ops", "rho", "arg", "rate_over_arg",
                          "cos_readout_mode", "secs", "ok") if k in rec}
                print(json.dumps(brief), flush=True)
                if wb:
                    wb.log({f"{kind}/n_ops": n_ops, **{f"{kind}/{k}": val
                            for k, val in brief.items() if isinstance(val, (int, float))}})

    save()
    ok = [r for r in records if r.get("ok") and r.get("spiral", {}).get("ok")]
    print(f"\n=== {len(ok)}/{len(records)} usable ===", flush=True)
    if ok:
        ratios = [r["rate_over_arg"] for r in ok if r.get("rate_over_arg")]
        r2s = [r["spiral"]["radial_r2"] for r in ok]
        print(f"  P2 radial fit R^2: median {np.median(r2s):.3f} (gate wants >= 0.9)")
        print(f"  P3 measured_rate/|arg|: median {np.median(ratios):.3f} "
              f"(point prediction is 1.0), range [{min(ratios):.3f}, {max(ratios):.3f}]")
        cs = [r["cos_readout_mode"] for r in ok if "cos_readout_mode" in r]
        ch = [r["cos_readout_random_mean"] for r in ok if "cos_readout_random_mean" in r]
        if cs:
            print(f"  P4 |cos(readout, mode)|: median {np.median(cs):.4f} vs random "
                  f"{np.median(ch):.4f}")
    if wb:
        wb.finish()
    print(f"saved {OUT}\nDONE", flush=True)


main()
