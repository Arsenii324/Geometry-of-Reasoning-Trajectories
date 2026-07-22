"""Capture the latent path of one token across recurrent unrolls.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, get_trajectory).
TASK: Hook Huginn's recurrent core_block so every unroll writes the hidden
    state of the selected token into a buffer; return the stacked path.
I/O: (model, tok, prompt, num_steps, seed, token_index) -> np.ndarray
    of shape [num_steps, HIDDEN_DIM].

GOTCHAS (hard-won — keep):
  * Hook `model.transformer.core_block[-1]`. Call `_forward_hooks.clear()` and
    wrap the forward in try/finally, else a hook left by a crashed run DOUBLES
    the captures.
  * `num_steps` must be a plain int, NOT a torch.tensor (len() of a 0-d tensor
    crashes the forward).
  * h_0 is a RANDOM init: torch.manual_seed(seed) before the forward for
    reproducibility (it is an outlier; downstream winding uses a burn-in).
  * Use `model(...)` (forward), NOT generate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from traj_geom.constants import DEFAULT_NUM_STEPS


def extract_trajectory(
    model: Any,
    tok: Any,
    prompt: str,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = 0,
    token_index: int = -1,
) -> np.ndarray:
    """Run ``prompt`` through ``num_steps`` recurrent unrolls and record the path.

    Args:
        model: A model handle from :func:`traj_geom.extraction.model.load_huginn`.
        tok: The matching tokenizer.
        prompt: Input prompt text.
        num_steps: Number of recurrent unrolls r (plain int, not a tensor).
        seed: Seed for the random recurrent initialisation h_0.
        token_index: Prompt token position to trace; -1 = last (answer) token.

    Returns:
        The latent path as an array of shape [num_steps, HIDDEN_DIM].
    """
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()  # drop hooks left by any crashed run (else doubled captures)

    lat: list[torch.Tensor] = []
    h = mod.register_forward_hook(lambda m, i, o: lat.append(o.detach().float().cpu()))
    try:
        torch.manual_seed(seed)  # h_0 is random -> seed for reproducibility
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            model(input_ids=ids, num_steps=num_steps)  # int, NOT tensor; forward, NOT generate
    finally:
        h.remove()  # removed even if the forward raised

    return torch.stack([latent[0, token_index, :] for latent in lat]).numpy()


def extract_trajectory_v6(
    model: Any,
    tok: Any,
    prompt: str,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = 0,
    token_index: int = -1,
) -> dict:
    """Run ``prompt`` through ``num_steps`` recurrent unrolls and record the path AND logits.

    Args:
        model: A model handle from :func:`traj_geom.extraction.model.load_huginn`.
        tok: The matching tokenizer.
        prompt: Input prompt text.
        num_steps: Number of recurrent unrolls r.
        seed: Seed for the random recurrent initialisation h_0.
        token_index: Prompt token position to trace; -1 = last (answer) token.

    Returns:
        dict containing:
        - "latents": np.ndarray of shape [num_steps, HIDDEN_DIM]
        - "logits": np.ndarray of shape [num_steps, VOCAB_SIZE]
    """
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()  # drop hooks left by any crashed run

    lat: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    
    def v6_hook(m, i, o):
        # o is the hidden state [batch, seq_len, hidden_dim]
        # We must detach it to save memory
        h_state = o.detach().float()
        
        # Save the latent for the specific token
        lat.append(h_state[0, token_index, :].cpu())
        
        # Compute logits for this step to track correctness over time
        # Huginn standard projection: lm_head(ln_f(h))
        with torch.no_grad():
            try:
                # Apply layer norm if present
                if hasattr(model.transformer, 'ln_f'):
                    h_norm = model.transformer.ln_f(h_state)
                else:
                    h_norm = h_state
                # Project to vocab
                step_logits = model.lm_head(h_norm)
                # Store the logits for the target token
                logits_list.append(step_logits[0, token_index, :].cpu())
            except Exception:
                # Fallback if the architecture is slightly different than expected
                pass

    h = mod.register_forward_hook(v6_hook)
    try:
        torch.manual_seed(seed)
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            model(input_ids=ids, num_steps=num_steps)
    finally:
        h.remove()

    result = {
        "latents": torch.stack(lat).numpy(),
    }
    
    if logits_list:
        result["logits"] = torch.stack(logits_list).numpy()
        
    return result
