"""A53 dry run: the whole kernel path on the mini Raven, before any GPU slot is spent.

WHY THIS EXISTS. A47 burned 77 minutes of GPU and returned nothing because a scoring block that
had never been executed raised IndexError after the compute finished (directions.md Q.1). The
lesson was not "read more carefully" -- it was that the scoring section had never been run at all.
So this runs A53's *entire* path locally: prompt construction, the settled-state capture, the
matrix-free spectrum, the summary fields, and the kernel's own `_score` over the real record
shape. The only things it cannot exercise are the cloud scaffolding and the 3.5B weights.

Token ids are random because the mini model has no tokeniser. Nothing in the control flow
depends on which tokens they are; the numbers it prints are not results.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from jacobian_spectrum import build_mini  # noqa: E402

KERNEL = ROOT / "scratch" / "ds_jacspec" / "job.py"


def load_kernel():
    spec = importlib.util.spec_from_file_location("a53_job", KERNEL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["a53_job"] = mod
    spec.loader.exec_module(mod)     # module level is stdlib-only by construction
    return mod


def main() -> int:
    job = load_kernel()
    # Use the KERNEL's own inlined instrument, not the one in scripts/. The kernel is what
    # runs on the GPU; testing a different copy tests the wrong code.
    import numpy as _np
    job.np, job.torch = _np, torch          # main() binds these as locals on the cloud image;
    make_matvec, spectrum = job.make_matvec, job.spectrum   # module-level defs need them global
    summarise = job.summarise
    items = job.build_items()
    print(f"prompt set: {len(items)} items -- "
          f"{sum(1 for i in items if i['regime']=='rotating')} rotating, "
          f"{sum(1 for i in items if i['regime']=='settling')} settling, "
          f"{sum(1 for i in items if i['regime']=='unlabelled')} census")
    assert all(k in items[0] for k in ("kind", "label", "item", "regime", "prompt"))

    model, cfg = build_mini()
    rng = torch.Generator().manual_seed(0)

    def settled_state(ids, warmup=8):
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
                torch.manual_seed(job.SEED)
                model(input_ids=ids, num_steps=warmup)
        finally:
            model.core_block_forward = orig
        return grab["h"], grab["aux"]

    t0, rows = time.time(), []
    for it in items:
        n = 20 + (len(it["prompt"]) % 12)          # vary length, as real prompts do
        ids = torch.randint(0, 500, (1, n), generator=rng)
        hs, aux = settled_state(ids)
        emb, freqs, blk, step = aux

        def step_fn(x, _e=emb, _f=freqs, _b=blk, _s=step):
            o, _ = model.core_block_forward(x, _e, _f, None, None, _b.clone(), _s)
            return o

        matvec, mode = make_matvec(step_fn, hs)
        vals = spectrum(matvec, hs.shape[-1], k=min(job.N_EIGS, hs.shape[-1] - 2))
        rows.append({**{k: it[k] for k in ("kind", "label", "item", "regime")},
                     "n_tokens": int(ids.shape[1]), "ad_mode": mode, "ok": True,
                     "eigs": [[float(v.real), float(v.imag)] for v in vals],
                     **summarise(vals)})

    print(f"\ncomputed {len(rows)} spectra in {time.time()-t0:.0f}s "
          f"(AD mode: {rows[0]['ad_mode']})")

    # the fields _score reads must all be present
    need = ("rho", "lead_real", "n_osc", "n_eigs", "regime")
    missing = {k for r in rows for k in need if k not in r}
    print(f"required fields present: {'yes' if not missing else 'MISSING ' + str(missing)}")

    print("\n--- running the kernel's own _score over these rows ---")
    job._score(rows)

    print("\n--- and over the degenerate cases that break scoring blocks ---")
    job._score([{**rows[0], "n_osc": 0, "period_unrolls": None, "turns_to_1pct": None}])
    job._score([{**rows[0], "ok": False, "why": "simulated failure"}])
    job._score([])
    print("\nVERDICT: full A53 path runs end to end, including scoring and its degenerate cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
