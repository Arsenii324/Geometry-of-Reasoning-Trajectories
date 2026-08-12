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
# THIS MAP WAS INCOMPLETE TWICE, AND BOTH TIMES IT FAILED A GOOD KERNEL.
# First it omitted `prelude` and flagged ds_eigen. Then on 2026-08-12 it omitted
# `core_block_forward` and flagged ds_jacspec -- a kernel whose full path had already been run
# locally against the real class. Same failure, second instance: the checker was the broken one.
# It is no longer hand-listed. Regenerate with, and paste the result below:
#
#   .venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); \
#     from jacobian_spectrum import build_mini; m,_=build_mini(); \
#     print(sorted(n for n,v in vars(type(m)).items() if callable(v) and not n.startswith('__')))"
#
MODEL_METHODS = {                                    # on RavenForCausalLM, verified 2026-08-12
    "initialize_state",              # :804  -- NOT on .transformer; this is what killed A47
    "iterate_forward",               # :727  (this map said :736; corrected)
    "core_block_forward",            # :764  -- omitted until 2026-08-12
    "iterate_one_step",              # :842
    "predict_from_latents",          # :870
    "embed_inputs",                  # :902
    "forward_with_adaptive_compute", # :999  -- an inference mode nobody here has used
    "generate_with_adaptive_compute",# :1552
    "generate_minimal",              # :1166
    "generate_speculative",          # :1751
    "generate_diffusion_style",      # :1235
    "randomized_iteration_sampler",  # :784
    "get_stats",                     # :1037
    "compile_mask",                  # :593
    "get_input_embeddings",          # :582
    "get_output_embeddings",         # :585
    "_maybe_inject_noise",           # :815  -- the unused test_time_noise API
    "_maybe_checkpoint_core_block",  # :1051
    "_precompute_freqs_cis",         # :588
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
