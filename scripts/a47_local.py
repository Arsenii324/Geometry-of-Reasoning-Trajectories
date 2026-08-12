"""A47 reproduced locally on the mini Raven, to find why the third run died undiagnosed.

BACKGROUND. `scratch/ds_h0inject/job.py` (A47) failed three times. The first two were diagnosed:
a donor-length gate that could never pass dropped 18 of 18 items, the kernel hit its own
"NO USABLE ITEMS" guard and returned 1, and DataSphere reported ERROR with no output. That gate
was removed. The third run reached 4/18 recipients at 542 s -- so it ran real compute and died
later -- and was **never diagnosed**, deliberately, under a time commitment on submission day
(directions.md §Q).

WHAT THIS SCRIPT IS. The same control flow -- the `initialize_state` monkeypatch, the
`core_block[-1]` hook, `coda_head`, and the per-item loop with its ~17 orbits -- against the real
`RavenForCausalLM` classes at 27.7M params, locally, no GPU, no weights download. Token ids are
random because the mini model has no tokeniser; nothing in the failure hypothesis depends on
which tokens they are.

WHAT IT CAN AND CANNOT FIND. It can find: hook leaks, state accumulating across items, shape
errors on later items, unbounded memory, and whether the P1 gate is even reachable. It cannot
find: CUDA OOM, DataSphere wall limits, or anything specific to the 3.5B weights. If everything
here passes, the remaining hypotheses are the platform ones, and that is itself the answer.

WHAT IT FOUND -- see the verdict printed at the end, and the note added to directions.md §Q.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import resource
import sys
import time
import types
import pathlib

import numpy as np
import torch

SRC = pathlib.Path(
    "/Users/a2mogus/build-projs/huginn-load/01-huginn-recurrent-depth_2502.05171"
    "/code/recurrent-pretraining/recpre"
)
DEPTH = 48          # as in the kernel
N_ITEMS = 18        # the number the kernel actually built
SEED = 20260811
ALT_SEED = 76543
TOPK = 5


def load_real_classes():
    pkg = types.ModuleType("recpre")
    pkg.__path__ = [str(SRC)]
    sys.modules["recpre"] = pkg

    def load(name):
        spec = importlib.util.spec_from_file_location(f"recpre.{name}", SRC / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"recpre.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    return load("raven_config_minimal"), load("raven_modeling_minimal")


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main() -> int:
    rc, rm = load_real_classes()
    cfg = rc.RavenConfig(n_embd=64, n_heads=4, n_layers=8, block_size=256,
                         vocab_size=512, effective_expected_depth=4, mean_recurrence=4)
    model = rm.RavenForCausalLM(cfg).eval()
    print(f"mini RavenForCausalLM {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
          f"DEPTH={DEPTH}, items={N_ITEMS}")

    core_last = model.transformer.core_block[-1]
    orig_init = model.initialize_state
    DONOR = {"v": None}
    CAPTURED = {}

    def patched_init(input_embeds, scale: float = 1.0):
        if DONOR["v"] is not None:
            CAPTURED["h0"] = DONOR["v"].detach().clone()
            return DONOR["v"].clone()
        h = orig_init(input_embeds, scale)
        CAPTURED["h0"] = h.detach().clone()
        return h

    model.initialize_state = patched_init

    def coda_head(h, freqs):
        x = model.transformer.ln_f(h)
        bi = torch.tensor(0, dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        return model.lm_head(model.transformer.ln_f(x))

    def orbit(ids, gold_ids, donor=None, seed=SEED):
        n = ids.shape[1]
        freqs = model.freqs_cis[:, :n]
        ranks = {g: [] for g in gold_ids}
        traj, tops = [], []
        core_last._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                h = o.detach()[0, -1]
                traj.append(h.float().cpu().numpy().copy())
                row = torch.log_softmax(
                    coda_head(o.detach(), freqs).float()[0, n - 1], dim=-1)
                for g in gold_ids:
                    ranks[g].append(int((row > row[g]).sum().item()) + 1)
                if len(traj) in (1, 2, 4, 8, 16, 32, DEPTH):
                    lp, ix = torch.topk(row, TOPK)
                    tops.append([len(traj), [[int(j), round(float(p), 3)]
                                             for p, j in zip(lp.tolist(), ix.tolist())]])

        DONOR["v"] = donor
        CAPTURED.pop("h0", None)
        hh = core_last.register_forward_hook(hook)
        try:
            with torch.no_grad():
                torch.manual_seed(seed)
                model(input_ids=ids, num_steps=DEPTH)
        finally:
            hh.remove()
            DONOR["v"] = None
        return {"ranks": {str(k): v for k, v in ranks.items()}, "tops": tops,
                "traj": np.array(traj, dtype=np.float32), "h0": CAPTURED.get("h0")}

    # --- items: three "families", six each, lengths deliberately UNEQUAL as in the real run
    rng = np.random.default_rng(0)
    fams = ["echo_digit", "add1", "sub1"]
    lens = {"echo_digit": 49, "add1": 40, "sub1": 40}
    test = [{"family": f, "item": i,
             "ids": torch.tensor(rng.integers(0, 500, (1, lens[f])), dtype=torch.long),
             "gold": int(rng.integers(0, 500))}
            for f in fams for i in range(N_ITEMS // 3)]
    print(f"{len(test)} recipients, lengths {sorted({int(t['ids'].shape[1]) for t in test})}")

    t0 = time.time()
    rows, p1_fail, first_err = [], [], None
    hooks_seen, rss0 = [], rss_mb()

    for n_it, it in enumerate(test):
        try:
            ids = it["ids"]
            g_self = it["gold"]
            same = [x for x in test if x["family"] == it["family"] and x["item"] != it["item"]]
            xfam = [x for x in test if x["family"] != it["family"]]
            d_same, d_xfam = same[0], xfam[0]
            g_don = d_same["gold"]
            gold_ids = [g_self] if g_don == g_self else [g_self, g_don]

            base = orbit(ids, gold_ids)
            h0_own = base["h0"]
            don_same = orbit(d_same["ids"], [g_don])
            don_xfam = orbit(d_xfam["ids"], [g_don])

            def tile(vec):
                return torch.tensor(vec, dtype=torch.float32).view(1, 1, -1).repeat(
                    1, ids.shape[1], 1)

            DONOR["v"] = None
            torch.manual_seed(ALT_SEED)
            with torch.no_grad():
                model(input_ids=ids, num_steps=1)
            h0_alt = CAPTURED.get("h0")

            star_same = tile(don_same["traj"][-1])
            n_h0, n_star = float(h0_own.norm()), float(star_same.norm())
            arms = {
                "own_h0": h0_own,
                "redraw_h0": h0_alt,
                "mid4_same": tile(don_same["traj"][3]),
                "star_same": star_same,
                "star_xfam": tile(don_xfam["traj"][-1]),
                "star_scaled": star_same * (n_h0 / max(n_star, 1e-9)),
            }
            mix_stmt = h0_own.clone()
            mix_stmt[:, -1, :] = h0_alt[:, -1, :]
            mix_last = h0_alt.clone()
            mix_last[:, -1, :] = h0_own[:, -1, :]
            arms["mix_lastpos_redrawn"] = mix_stmt
            arms["mix_stmtpos_redrawn"] = mix_last

            for sd in (SEED, ALT_SEED, 11111, 22222, 33333, 44444):
                orbit(ids, gold_ids, donor=None, seed=sd)

            for name, donor in arms.items():
                o = orbit(ids, gold_ids, donor=donor)
                if name == "own_h0":
                    d = max(abs(a - b) for a, b in
                            zip(o["ranks"][str(g_self)], base["ranks"][str(g_self)]))
                    if d != 0:
                        p1_fail.append((n_it, d))
            rows.append(1)
        except Exception as exc:  # noqa: BLE001
            if first_err is None:
                first_err = f"item {n_it}: {type(exc).__name__}: {exc}"
            rows.append(0)

        hooks_seen.append(len(core_last._forward_hooks))
        if n_it % 4 == 0 or n_it == len(test) - 1:
            gc.collect()
            print(f"  item {n_it:>2}/{len(test)}  {time.time()-t0:>6.0f}s  "
                  f"rss {rss_mb():>6.0f}MB  live hooks {len(core_last._forward_hooks)}")

    print("\n" + "=" * 74)
    print(f"completed {sum(rows)}/{len(test)} items in {time.time()-t0:.0f}s")
    print(f"RSS {rss0:.0f} -> {rss_mb():.0f} MB   (growth {rss_mb()-rss0:+.0f} MB)")
    print(f"max live forward hooks at any point: {max(hooks_seen)}  (must stay 0 between orbits)")
    print(f"P1 (own h0 reproduces the baseline rank curve exactly): "
          f"{'FAIL on items ' + str(p1_fail) if p1_fail else 'PASS on every item'}")
    print(f"first exception: {first_err or 'none'}")
    bad = int(bool(p1_fail) or first_err is not None or max(hooks_seen) > 0
              or sum(rows) != len(test))
    print("VERDICT:", "control flow is sound at mini scale" if not bad else "REPRODUCED A FAULT")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
