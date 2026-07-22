"""Hard-won constants for the Huginn extraction pipeline. Do NOT edit values blindly.

Every value here was fixed by debugging the MVP notebook (notebooks/01_mvp_h2.ipynb) —
treat them as the "as-is" reproducibility guarantee.

GOTCHAS (kept verbatim from the notebook):
- transformers must be 4.50–4.53. Outside the window the custom modeling code breaks:
    4.49-  : no `device` in _get_initial_cache_position
    4.54+  : key_cache became a read-only property
    5.x    : tied-weights loading expects a dict
  Pin 4.53.3 + MODEL_REVISION — both mandatory.
- num_steps is a plain int, NOT a torch.tensor (len() of a 0-d tensor crashes forward).
- Hook `model.transformer.core_block[-1]`; `_forward_hooks.clear()` + try/finally, else a
  hook left by a crashed run doubles the captures.
- h_0 is a RANDOM init (torch.manual_seed before forward): it is an outlier, so winding is
  measured with a burn-in that drops the initial arc.
- For a trajectory use forward, NOT generate.
"""

from __future__ import annotations

import os

MODEL_ID = os.environ.get("HUGINN_MODEL_ID", "tomg-group-umd/huginn-0125")
# MODEL_REVISION only pins anything when MODEL_ID is a Hub repo id. If
# HUGINN_MODEL_ID is overridden to a local path (e.g. a Kaggle dataset mount),
# from_pretrained's `revision=` kwarg is silently ignored for local paths — you
# get whatever code/weights are on disk, unpinned, no error. Check the local
# copy matches this revision yourself before trusting results from that path.
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"  # freeze the remote code
TRANSFORMERS_VERSION = "4.53.3"  # working window is ONLY 4.50–4.53

HIDDEN_DIM = 5280           # dim of every h_t captured from core_block
DEFAULT_NUM_STEPS = 64      # recurrent unrolls r (the MVP budget)
