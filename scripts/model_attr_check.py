"""Do a kernel's `model.<...>` attribute paths actually exist on Huginn? Checked without a GPU.

WHY IT EXISTS. A47 died on its first model-touching line. It patched
`model.transformer.initialize_state`, and `initialize_state` is **not on `transformer`** -- the
source builds `self.transformer = torch.nn.ModuleDict(wte=, adapter=, core_block=, coda=,
ln_f=)` (`raven_modeling_minimal.py:562-569`, five keys) and defines `initialize_state` on
`RavenForCausalLM` itself (`:804`, called as `self.initialize_state(...)` at `:738`). One wrong
attribute path, one AttributeError, a whole GPU run lost with no output written.

That is the second structural failure of this shape today: A41 concatenated a fixed-length `e`
onto a sequence that grows during generation. **Neither needed a GPU to catch, and neither was
caught, because our preflight checks items and imports but never touches the model surface.**

Weights are 7 GB and cannot be loaded locally, so the check is against a **declared attribute
map read from the released source**, not against a live model. That is enough for this failure
class, which is entirely about whether a name exists.

    uv run python scripts/model_attr_check.py scratch/ds_h0inject/job.py
"""

from __future__ import annotations

import pathlib
import re
import sys

# From tomg-group-umd/huginn-0125's raven_modeling_minimal.py, line numbers cited so this can
# be re-verified rather than trusted.
TRANSFORMER_KEYS = {"wte", "prelude", "adapter", "core_block", "coda", "ln_f"}   # :562-571,
#   read from the ModuleDict construction verbatim. My first version of this map came from a
#   grep whose pattern I chose, and it omitted `prelude` -- flagging ds_eigen as broken when
#   the checker was the broken one. RC1: I measured a proxy for the thing.
MODEL_METHODS = {                                                            # on RavenForCausalLM
    "initialize_state",        # :804  -- NOT on .transformer; this is what killed A47
    "iterate_forward",         # :736
    "randomized_iteration_sampler",
    "forward", "generate", "config", "device", "lm_head", "freqs_cis", "transformer",
    "emb_scale", "to", "eval", "train", "parameters", "named_parameters", "state_dict",
}

PATH = re.compile(r"\bmodel\.((?:[A-Za-z_][A-Za-z0-9_]*)(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")


# filenames, not attribute access: `model.safetensors.index.json` is a string literal
FILE_EXT = ("json", "py", "safetensors", "bin", "txt", "yaml", "yml", "pt", "npy")


def check(text: str):
    bad, ok = [], []
    for m in PATH.finditer(text):
        parts = m.group(1).split(".")
        if parts[-1] in FILE_EXT:
            continue
        head = parts[0]
        if head == "transformer" and len(parts) > 1:
            if parts[1] not in TRANSFORMER_KEYS:
                bad.append((m.group(0),
                            f"`transformer` is a ModuleDict with keys {sorted(TRANSFORMER_KEYS)}; "
                            f"`{parts[1]}` is not one of them"
                            + (" -- it is a method of RavenForCausalLM, so use "
                               f"`model.{parts[1]}`" if parts[1] in MODEL_METHODS else "")))
            else:
                ok.append(m.group(0))
        elif head in MODEL_METHODS:
            ok.append(m.group(0))
        else:
            bad.append((m.group(0), f"`{head}` is not a known attribute of RavenForCausalLM"))
    return ok, bad


def main() -> int:
    targets = [pathlib.Path(a) for a in sys.argv[1:]] or sorted(
        pathlib.Path("scratch").glob("ds_*/job.py"))
    fails = 0
    for f in targets:
        if not f.exists():
            continue
        ok, bad = check(f.read_text(errors="ignore"))
        if bad:
            fails += 1
            print(f"\n{f}")
            for expr, why in dict.fromkeys(bad):
                print(f"  FAIL  {expr}\n        {why}")
        else:
            print(f"ok    {f}  ({len(set(ok))} distinct model paths)")
    print(f"\n{fails} file(s) with unresolved model attribute paths")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
