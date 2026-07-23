"""Load the Huginn recurrent-depth transformer for inference.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/00_smoke_extract.ipynb, cell-2).
    Retry/low_cpu_mem_usage hardening added 2026-07-23.
TASK: Load Huginn (custom architecture, trust_remote_code=True) at the pinned
    revision onto the target device/dtype and return (model, tokenizer).
I/O: (device, dtype) -> (model, tok).

GOTCHA: transformers must be 4.50–4.53 and the revision must be pinned — see
    traj_geom.constants. Per-unroll hidden states are NOT exposed via
    output_hidden_states; capture them by hooking core_block (see hook.py).

GOTCHA: a transient network hiccup during the initial (multi-GB) weight
    download would otherwise kill an entire paid/quota-limited GPU session
    for a reason that has nothing to do with the experiment itself — retried
    with backoff below rather than left to crash on the first attempt.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from traj_geom.constants import MODEL_ID, MODEL_REVISION

_LOAD_RETRIES = 3
_LOAD_RETRY_BACKOFF_S = 10.0


def load_huginn(
    device: str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> tuple[Any, Any]:
    """Load the Huginn model and tokenizer at the pinned revision.

    Retries the (large, network-dependent) weight download up to
    ``_LOAD_RETRIES`` times with linear backoff before giving up — a GPU
    session is far more expensive than a few seconds of retry delay, so a
    transient connection error on attempt 1 should not waste the whole run.

    Args:
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        dtype: Compute dtype (default ``torch.bfloat16``).

    Returns:
        ``(model, tokenizer)`` — model moved to ``device`` and set to eval mode.

    Raises:
        The last attempt's exception, unmodified, if every retry fails —
        loud failure after retries are exhausted, not a silent fallback.
    """
    model = None
    for attempt in range(1, _LOAD_RETRIES + 1):
        try:
            model = (
                AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # stream weights to target dtype/device directly,
                )  # instead of materialising a full fp32 copy on CPU first
                .to(device)
                .eval()
            )
            break
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure here
            # (network, disk, HF rate limit) is worth one retry before giving up.
            if attempt == _LOAD_RETRIES:
                raise
            print(
                f"load_huginn: attempt {attempt}/{_LOAD_RETRIES} failed ({e!r}), "
                f"retrying in {_LOAD_RETRY_BACKOFF_S:.0f}s..."
            )
            time.sleep(_LOAD_RETRY_BACKOFF_S)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    return model, tok
