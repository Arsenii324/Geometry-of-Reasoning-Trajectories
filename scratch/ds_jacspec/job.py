"""A53: the multi-prompt Jacobian-spectrum sweep D55 asked for, and the test that connects
the project's two rotation lines.

WHERE THIS COMES FROM. D31 measured the recurrence's Jacobian spectrum exactly -- implicitly
restarted Arnoldi on autodiff Jacobian-vector products -- and reported only the magnitudes:
rho = 0.7935 / 0.8042 / 0.8083 on three prompts. D55 then pulled the eigenvalue ARGUMENTS out of
the same banked log at zero further compute: the leading eigenvalue is complex in 3/3 prompts,
27/30 top modes are complex, rotation period 2.6-6.0 unrolls, surviving only ~4 turns while the
project's trajectory statistics sampled it at ~3 points per turn. That is why five separate
trajectory statistics failed on the rotation question -- aliasing, not absence. **D55 named a
multi-prompt sweep as its own follow-up and it was never run.** Three prompts is the entire
evidence base for the strongest structural claim in the record.

THE QUESTION THIS ADDS, WHICH IS NOT JUST "MORE PROMPTS". The project has two independent
rotation lines that have never been put in the same room:

  * the JACOBIAN line -- the map's spectrum is complex, period 2.6-6.0 unrolls (D31, D55);
  * the REGIME line -- one instruction noun flips the trajectory between settling and a damped
    **period-6** rotation, 36/36 paired cells at fixed token count, causally controlled by one
    embedding row, threshold reproducing the label on 691/696 orbits (D132, D141, D161).

Those periods overlap. If they are the same phenomenon, then a prompt this project labels
ROTATING should carry an eigenvalue whose argument implies a period near 6, and a prompt it
labels SETTLING should not -- and the regime statistic would be reading the Jacobian's
oscillatory mode. Nobody has checked. It costs one kernel.

PREREGISTERED PREDICTIONS. Written before the run.

  P1  GATE -- the instrument reproduces the project's contraction range. Median rho over all
      prompts lands in 0.75-0.90, where four independent methods have put it. If it does not,
      nothing below may be read. (Not a claim to reproduce D31's three numbers exactly: the
      prompts differ.)

  P2  PRIMARY, REGISTERED BOTH WAYS. Comparing prompts labelled rotating (`symbol`, `symptom`)
      against settling (`element`, `token`), at identical sequence, marker and token count:
        (a) if the two lines are the same phenomenon -- rotating prompts carry an oscillatory
            eigenvalue with implied period in roughly 5-7 unrolls, settling prompts either have
            no oscillatory mode near that period or a much weaker one. The regime statistic is
            then a trajectory-space read of the Jacobian's spectrum.
        (b) if the spectrum is the same in both arms -- the regime difference is NOT in the
            leading local dynamics, and the two lines are separate. That is equally publishable
            and it bounds what the regime statistic can be measuring.
      Either way this is decided by the eigenvalue arguments, which need no trajectory, no
      recording window and no null -- the three things that broke the five earlier statistics.

  P3  Is the leading eigenvalue complex broadly, or was 3/3 a small sample? Reported as a
      fraction over all prompts, with the count of oscillatory modes.

  P4  How long does the rotating mode live? turns-to-1% = ln(0.01)/ln(|lambda|) * |arg|/2pi.
      D55 got ~4 turns on three prompts. If that holds broadly it bounds any scheme that hopes
      to carry information around the loop by rotation, because the carrier decays.

  P5  FLOOR. Does rho differ between regimes at all? D115 found contraction tracks task template
      rather than difficulty; this asks whether it tracks the regime label.

INSTRUMENT. `scripts/jacobian_spectrum.py`, validated locally against an EXACT dense
eigendecomposition at mini width: matrix-free ARPACK reproduced the exact top-6 magnitudes to
4.3e-07 and agreed on the complex/real verdict. At 5280 dimensions no such reference exists, so
that check had to be made where it was possible. Forward-mode AD needs the MATH SDPA backend;
the reverse fallback is exact for the spectrum because eig(J^T) = eig(J).
"""

from __future__ import annotations

import json
import math
import os
import random as _r
import subprocess
import sys
import time

# numpy/torch are imported inside main() on the cloud image; the inlined instrument below
# only touches them at call time, after main() has installed and imported them.

SEED = 20260812
MARKER = "A"
SEQS = ("1 6 0 1 7 7 8 1", "3 5 5 3 6 1 5 2", "2 4 4 1 7 5 0 9")
ROT_NOUNS = ("symbol", "symptom")
SET_NOUNS = ("element", "token")
N_CENSUS = 2
N_EIGS = 16
WARMUP = 32              # unrolls before the Jacobian is taken: at the settled end
TOL = 1e-5
MAXITER = 600
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 7000


def prompt_for(word, seq, mk=MARKER):
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def census_items():
    rng = _r.Random(SEED)
    out = []
    for _ in range(N_CENSUS):
        a = rng.randrange(1, 9)
        xs = rng.sample(range(1, 100), 3)
        bits = [rng.randrange(2) for _ in range(9)]
        d = rng.randrange(10)
        out += [
            {"family": "echo_digit", "prompt": f"Repeat this number exactly.\nNumber: {d}"},
            {"family": "add1", "prompt": f"What is {a} + 1?"},
            {"family": "sort_min",
             "prompt": f"What is the smallest of these numbers: {xs[0]}, {xs[1]}, {xs[2]}?"},
            {"family": "count_mod3",
             "prompt": ("Count how many ones are in this sequence, then give the remainder "
                        f"when divided by 3.\nSequence: {' '.join(map(str, bits))}")},
        ]
    return out


def build_items():
    out = []
    for n in ROT_NOUNS + SET_NOUNS:
        for i, s in enumerate(SEQS):
            out.append({"kind": "regime", "label": n, "item": i,
                        "regime": "rotating" if n in ROT_NOUNS else "settling",
                        "prompt": prompt_for(n, s)})
    for i, it in enumerate(census_items()):
        out.append({"kind": "census", "label": it["family"], "item": i,
                    "regime": "unlabelled", "prompt": it["prompt"]})
    return out


def make_matvec(step_fn, hs, prefer_forward=True):  # noqa: C901
    """v -> J v at the last position. Returns (matvec, mode)."""
    dim = hs.shape[-1]

    def _fwd(v):
        vt = torch.zeros_like(hs)
        vt[0, -1, :] = torch.as_tensor(np.asarray(v, dtype=np.float32),
                                       device=hs.device, dtype=hs.dtype)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            _, jv = torch.func.jvp(step_fn, (hs,), (vt,))
        return jv[0, -1, :]

    def _rev(v):
        u = torch.zeros_like(hs)
        u[0, -1, :] = torch.as_tensor(np.asarray(v, dtype=np.float32),
                                      device=hs.device, dtype=hs.dtype)
        x = hs.detach().clone().requires_grad_(True)
        y = step_fn(x)
        (g,) = torch.autograd.grad(y, x, grad_outputs=u)
        return g[0, -1, :]

    mode = "forward"
    if prefer_forward:
        try:
            _fwd(np.zeros(dim, dtype=np.float32))
        except Exception:
            mode = "reverse"
    else:
        mode = "reverse"
    fn = _fwd if mode == "forward" else _rev

    def matvec(v):
        return fn(v).detach().float().cpu().numpy().astype(np.float64)

    return matvec, mode


def spectrum(matvec, dim, k=12, tol=1e-6, maxiter=600):
    from scipy.sparse.linalg import LinearOperator, eigs
    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    vals = eigs(op, k=k, which="LM", return_eigenvectors=False, tol=tol, maxiter=maxiter)
    return vals[np.argsort(-np.abs(vals))]


def summarise(vals):
    osc = vals[np.abs(vals.imag) > 1e-8]
    lead = vals[0]
    out = {
        "rho": float(np.abs(lead)),
        "lead_real": bool(abs(lead.imag) < 1e-8),
        "n_osc": int(len(osc)),
        "n_eigs": int(len(vals)),
    }
    if len(osc):
        a = float(abs(np.angle(osc[0])))
        out["arg_osc"] = a
        out["rho_osc"] = float(np.abs(osc[0]))
        out["period_unrolls"] = (2 * math.pi / a) if a > 1e-12 else None
        # how many turns the rotating mode survives before contraction kills it
        r = float(np.abs(osc[0]))
        out["turns_to_1pct"] = ((math.log(0.01) / math.log(r)) * a / (2 * math.pi)
                                if 0 < r < 1 else None)
    return out


# --------------------------------------------------------------------------- local gate


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.join(OUTDIR, "jacspec.json")
    mnt = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"results -> {out_path}; weights mount -> {mnt}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("pip install 'scipy>=1.11'")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # The inlined instrument lives at MODULE level, so it resolves `np` and `torch` as module
    # globals -- but these imports are locals of main(). Bind them explicitly. The local dry run
    # (scripts/a53_local.py) caught this, and a missing `import math`, before any GPU was spent;
    # both would have raised NameError after the first spectrum, which is A47's failure mode.
    globals()["np"], globals()["torch"] = np, torch

    # The instrument is INLINED below rather than imported from scripts/. It is duplicated
    # from scripts/jacobian_spectrum.py, which is where it was validated against an exact dense
    # eigendecomposition -- but a kernel that imports across a path which only exists after
    # clone-and-chdir is a path nobody has executed, and that is precisely the class of thing
    # that cost A47 77 minutes. Every other kernel here is self-contained; so is this one.

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tok = AutoTokenizer.from_pretrained(mnt)
    cfg = AutoConfig.from_pretrained(mnt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mnt, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    def encode(body):
        text = tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)

    def settled_state(ids):
        """Run WARMUP unrolls and return the state plus the block's own auxiliary args."""
        grab = {}
        orig = model.core_block_forward

        def wrapped(x, emb, freqs, mask, pkv, blk, step):
            if "aux" not in grab:
                grab["aux"] = (emb, freqs, blk.detach().clone(), step)
            out = orig(x, emb, freqs, mask, pkv, blk, step)
            grab["h"] = out[0].detach().clone()
            return out

        model.core_block_forward = wrapped
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=WARMUP)
        finally:
            model.core_block_forward = orig
        return grab["h"], grab["aux"]

    items = build_items()
    print(f"{len(items)} prompts: "
          f"{sum(1 for i in items if i['regime']=='rotating')} rotating, "
          f"{sum(1 for i in items if i['regime']=='settling')} settling, "
          f"{sum(1 for i in items if i['regime']=='unlabelled')} census", flush=True)

    t0, rows = time.time(), []
    for n_it, it in enumerate(items):
        try:
            ids = encode(it["prompt"])
            hs, aux = settled_state(ids)
            emb, freqs, blk, step = aux

            def step_fn(x, _e=emb, _f=freqs, _b=blk, _s=step):
                o, _ = model.core_block_forward(x, _e, _f, None, None, _b.clone(), _s)
                return o

            matvec, mode = make_matvec(step_fn, hs)
            vals = spectrum(matvec, hs.shape[-1], k=N_EIGS, tol=TOL, maxiter=MAXITER)
            rec = {**{k: it[k] for k in ("kind", "label", "item", "regime")},
                   "n_tokens": int(ids.shape[1]), "ad_mode": mode, "ok": True,
                   "eigs": [[float(v.real), float(v.imag)] for v in vals],
                   **summarise(vals)}
        except Exception as exc:  # noqa: BLE001
            rec = {**{k: it[k] for k in ("kind", "label", "item", "regime")},
                   "ok": False, "why": f"{type(exc).__name__}: {exc}"}
        rows.append(rec)
        print(json.dumps(rec)[:300], flush=True)

        # BANK EVERY ITEM. A47 lost 77 minutes of compute to a scoring bug that ran before
        # anything was written (directions.md Q.1). Nothing here may cost a run again.
        with open(out_path, "w") as f:
            json.dump({"rows": rows, "warmup": WARMUP, "n_eigs": N_EIGS, "seed": SEED,
                       "elapsed_s": time.time() - t0}, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banked and stopping cleanly", flush=True)
            break

    print(f"\nBANKED {len(rows)} rows to {out_path}", flush=True)
    try:
        _score(rows)
    except Exception as exc:  # noqa: BLE001
        print(f"SCORING FAILED ({type(exc).__name__}: {exc}) -- data is banked, "
              f"re-score offline.", flush=True)
    return 0


def _score(rows):
    """Printing only. The data is already on disk before this runs."""
    import statistics as st
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        print("NO USABLE ROWS", flush=True)
        return

    print("\n=== P1 GATE: contraction in the range four methods have measured ===", flush=True)
    rho = [r["rho"] for r in ok]
    med = st.median(rho)
    print(f"  median rho {med:.4f}  (min {min(rho):.4f}, max {max(rho):.4f}, n={len(rho)})",
          flush=True)
    print(f"  {'OK' if 0.75 <= med <= 0.90 else 'GATE FAILED -- do not read below'}", flush=True)

    print("\n=== P3: is the leading eigenvalue complex broadly? ===", flush=True)
    cx = sum(1 for r in ok if not r.get("lead_real", True))
    print(f"  leading eigenvalue complex in {cx}/{len(ok)} prompts", flush=True)
    nosc = [r.get("n_osc", 0) for r in ok]
    print(f"  oscillatory modes among the top {ok[0].get('n_eigs')}: "
          f"median {st.median(nosc):.1f}, range {min(nosc)}-{max(nosc)}", flush=True)

    print("\n=== P2 PRIMARY: does the implied period differ by regime? ===", flush=True)
    for arm in ("rotating", "settling", "unlabelled"):
        v = [r for r in ok if r["regime"] == arm and r.get("period_unrolls")]
        if not v:
            print(f"  {arm:11s} no oscillatory mode in any prompt", flush=True)
            continue
        per = [r["period_unrolls"] for r in v]
        near6 = sum(1 for p in per if 5.0 <= p <= 7.0)
        print(f"  {arm:11s} n={len(v):2d}  median period {st.median(per):.2f} unrolls  "
              f"range {min(per):.2f}-{max(per):.2f}  in 5-7: {near6}/{len(v)}", flush=True)
    rot = [r["period_unrolls"] for r in ok
           if r["regime"] == "rotating" and r.get("period_unrolls")]
    set_ = [r["period_unrolls"] for r in ok
            if r["regime"] == "settling" and r.get("period_unrolls")]
    if rot and set_:
        print(f"  separation of medians: {abs(st.median(rot) - st.median(set_)):.3f} unrolls",
              flush=True)
        print("  P2(a) same phenomenon: rotating near 6, settling not. "
              "P2(b) separate lines: both arms alike.", flush=True)

    print("\n=== P4: how long does the rotating mode survive? ===", flush=True)
    tt = [r["turns_to_1pct"] for r in ok if r.get("turns_to_1pct")]
    if tt:
        print(f"  turns to 1% amplitude: median {st.median(tt):.2f}, "
              f"range {min(tt):.2f}-{max(tt):.2f}  (D55 got ~4 on three prompts)", flush=True)

    print("\n=== P5 FLOOR: does rho itself track the regime label? ===", flush=True)
    for arm in ("rotating", "settling", "unlabelled"):
        v = [r["rho"] for r in ok if r["regime"] == arm]
        if v:
            print(f"  {arm:11s} n={len(v):2d}  median rho {st.median(v):.4f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
