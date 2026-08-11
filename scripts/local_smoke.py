"""Run the real Huginn class at toy size, locally, before spending a GPU slot.

WHY. Two kernels died on 2026-08-11 for reasons a local run would have caught in seconds, each
after the weights had loaded on a remote GPU:

  * **A47** monkeypatched `model.transformer.initialize_state`. `transformer` is a `ModuleDict`
    with keys `[adapter, coda, core_block, ln_f, prelude, wte]`; `initialize_state` is a method
    of `RavenForCausalLM`. AttributeError on the first model-touching line.
  * **A41** concatenated a fixed-length `e` onto a sequence that **grows** during generation.
    The `cat` shape-mismatches at the second generated token.

`model_attr_check.py` catches the first class (names). It cannot catch the second (shapes),
because a shape error is only visible when tensors actually flow. This runs them.

WHAT IT IS. `raven_modeling_minimal.py` and `raven_config_minimal.py` ship in the released
repo, so the **real classes** can be instantiated at toy dimensions with random weights --
**27.7M parameters, builds in about a second, no download, no MPS, no OOM risk.** Every code
path a kernel touches (prelude, adapter, `core_block`, coda, `ln_f`, the unroll loop, the KV
cache) is the genuine one; only the sizes are small.

WHAT IT CHECKS -- the four intervention patterns this project's kernels actually use, each one
a place a kernel has broken or could:

  1. a forward hook on `core_block[-1]` fires exactly `num_steps` times
  2. h0 injection by monkeypatching `initialize_state` -- **on the right object** (A47)
  3. `e`-patching at every unroll at fixed sequence length (A24/A32/A39's pattern)
  4. `e`-patching **during generation**, where the sequence grows (A41's failure), with the
     naive fixed-length version shown failing and the correct version shown working

USE. Run it after writing a kernel and before `preflight --launch`. If a pattern you need is
not here, add it here first and make it pass, then copy it into the kernel -- that is the
point: kernels should copy from a verified reference rather than invent an intervention.

    uv run python scripts/local_smoke.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

SRC = pathlib.Path(
    "/Users/a2mogus/build-projs/huginn-load/01-huginn-recurrent-depth_2502.05171"
    "/code/recurrent-pretraining/recpre"
)


def load_real_classes():
    """Import the released modeling file as a package so its relative imports resolve."""
    if not SRC.exists():
        raise SystemExit(f"released source not found at {SRC}")
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


def build(n_embd=64, n_heads=4, n_layers=8, vocab=512):
    rc, rm = load_real_classes()
    cfg = rc.RavenConfig(n_embd=n_embd, n_heads=n_heads, n_layers=n_layers,
                         block_size=256, vocab_size=vocab,
                         effective_expected_depth=4, mean_recurrence=4)
    return rm.RavenForCausalLM(cfg), cfg


def main() -> int:
    import torch

    model, cfg = build()
    d_model = cfg.n_embd
    n_par = sum(p.numel() for p in model.parameters())
    print(f"built real RavenForCausalLM at toy size: {n_par / 1e6:.1f}M params, "
          f"n_embd={d_model}")
    print(f"transformer keys: {sorted(model.transformer.keys())}\n")
    ids = torch.randint(0, 500, (1, 12))
    bad = 0

    # 1. the hook every kernel here uses
    seen = []
    h = model.transformer.core_block[-1].register_forward_hook(
        lambda _m, _i, o: seen.append(o.detach()))
    with torch.no_grad():
        model(input_ids=ids, num_steps=4)
    h.remove()
    ok = len(seen) == 4 and seen[0].shape[1] == ids.shape[1]
    print(f"  {'ok  ' if ok else 'FAIL'} hook on core_block[-1] fired {len(seen)}x "
          f"(expect 4), state {tuple(seen[0].shape)}")
    bad += not ok

    # 2. h0 injection -- A47's bug is that this attribute is on the MODEL, not the ModuleDict
    on_model = hasattr(model, "initialize_state")
    on_tf = hasattr(model.transformer, "initialize_state")
    print(f"  {'ok  ' if (on_model and not on_tf) else 'FAIL'} initialize_state is on the "
          f"model ({on_model}), not on transformer ({on_tf})  <- A47 patched the wrong one")
    bad += not (on_model and not on_tf)

    orig_init = model.initialize_state
    donor = {}

    def patched_init(input_embeds, scale: float = 1.0):
        s = orig_init(input_embeds, scale)
        donor["shape"] = tuple(s.shape)
        return s

    model.initialize_state = patched_init
    with torch.no_grad():
        model(input_ids=ids, num_steps=3)
    model.initialize_state = orig_init
    ok = donor.get("shape", (0,))[1] == ids.shape[1]
    print(f"  {'ok  ' if ok else 'FAIL'} monkeypatched initialize_state ran, h0 shape "
          f"{donor.get('shape')}")
    bad += not ok

    # 3. e-patching at fixed length (A24/A32/A39)
    grabbed = {}

    def grab(_m, inp):
        grabbed.setdefault("e", inp[0][..., d_model:].detach().clone())

    hh = model.transformer.adapter.register_forward_pre_hook(grab)
    with torch.no_grad():
        model(input_ids=ids, num_steps=1)
    hh.remove()
    e = grabbed["e"]

    def patch(donor_e):
        def pre(_m, inp):
            return (torch.cat([inp[0][..., :d_model], donor_e], dim=-1),)
        return pre

    hh = model.transformer.adapter.register_forward_pre_hook(patch(e))
    try:
        with torch.no_grad():
            model(input_ids=ids, num_steps=3)
        print(f"  ok   e-patch at fixed length, donor e {tuple(e.shape)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL e-patch at fixed length: {type(exc).__name__}: {exc}")
        bad += 1
    finally:
        hh.remove()

    # 4. e-patching during GENERATION -- A41 died here. Naive must fail, correct must pass.
    grew = torch.cat([ids, torch.randint(0, 500, (1, 1))], dim=1)
    hh = model.transformer.adapter.register_forward_pre_hook(patch(e))
    naive_failed = False
    try:
        with torch.no_grad():
            model(input_ids=grew, num_steps=2)
    except Exception:  # noqa: BLE001
        naive_failed = True
    finally:
        hh.remove()
    verdict = ("raises as it must" if naive_failed
               else "silently did NOT raise -- A41 would not have been caught")
    print(f"  {'ok  ' if naive_failed else 'FAIL'} a fixed-length e on a GROWN sequence "
          f"{verdict}  <- A41's bug")
    bad += not naive_failed

    grabbed.clear()
    hh = model.transformer.adapter.register_forward_pre_hook(grab)
    with torch.no_grad():
        model(input_ids=grew, num_steps=1)
    hh.remove()
    hh = model.transformer.adapter.register_forward_pre_hook(patch(grabbed["e"]))
    try:
        with torch.no_grad():
            model(input_ids=grew, num_steps=2)
        print("  ok   e recomputed for the grown sequence works  <- A41's fix")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL recomputed e still breaks: {type(exc).__name__}: {exc}")
        bad += 1
    finally:
        hh.remove()

    print(f"\n{'SMOKE FAILED' if bad else 'SMOKE PASSED'} ({bad} failing)")
    print("Sizes are toy; every code path is the released one. Shape and attribute errors "
          "surface here; numerical results do not.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
