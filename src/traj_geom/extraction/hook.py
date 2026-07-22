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
  * If you ever want per-step LOGITS (return_logits=True): the real per-token
    readout path is `core_block -> coda (2 layers) -> ln_f -> lm_head`
    (verified against `RavenForCausalLM.forward()` in
    `raven_modeling_minimal.py`). Hooking `core_block[-1]` alone gives you the
    state *before* coda — running `ln_f` then `lm_head` straight on that,
    skipping coda, produces logits that never correspond to anything the real
    model predicts, at any step, including the last. `_replicate_coda_head`
    below runs the actual coda blocks first; `validate_logits=True` (default)
    checks the reconstruction against a real forward() call's own logits on
    every use, since this has never been confirmed on real hardware — do not
    set it False until you've seen it pass at least once on your setup.
"""

from __future__ import annotations

from typing import Any

import torch

from traj_geom.constants import DEFAULT_NUM_STEPS


def _replicate_coda_head(
    model: Any, h_state: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    """Run the model's own coda -> ln_f -> lm_head tail on an intermediate state.

    Mirrors `RavenForCausalLM.forward()` exactly (block_idx counts down from
    -1, no attention mask / KV-cache needed for this one-shot, non-generation
    use). Do not shortcut this by calling `ln_f`/`lm_head` directly on
    `h_state` — that skips the two coda layers and gives meaningless logits.
    """
    x = h_state
    block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
    for block in model.transformer.coda:
        block_idx -= 1
        x = block(x, freqs_cis, block_idx, None, None)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


def extract_trajectory(
    model: Any,
    tok: Any,
    prompt: str,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = 0,
    token_index: int = -1,
    return_logits: bool = False,
    validate_logits: bool = True,
):
    """Run ``prompt`` through ``num_steps`` recurrent unrolls and record the path.

    Args:
        model: A model handle from :func:`traj_geom.extraction.model.load_huginn`.
        tok: The matching tokenizer.
        prompt: Input prompt text.
        num_steps: Number of recurrent unrolls r (plain int, not a tensor).
        seed: Seed for the random recurrent initialisation h_0.
        token_index: Prompt token position to trace; -1 = last (answer) token.
        return_logits: If True, also reconstruct and return the logits at
            each unroll (via the real coda/ln_f/lm_head tail, see GOTCHAS).
        validate_logits: If True and return_logits is True, cross-checks the
            final unroll's reconstructed logits against a genuine forward()
            call's own output, raising RuntimeError on disagreement rather
            than silently returning wrong numbers. Costs one extra forward
            pass. See GOTCHAS before ever setting this False.

    Returns:
        If return_logits is False, returns np.ndarray of shape [num_steps, HIDDEN_DIM].
        If return_logits is True, returns a dict with "latents" and "logits" —
        "logits" is only present if every step's reconstruction succeeded.
    """
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()  # drop hooks left by any crashed run (else doubled captures)

    lat: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []

    torch.manual_seed(seed)  # h_0 is random -> seed for reproducibility
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    freqs_cis = model.freqs_cis[:, : ids.shape[1]] if return_logits else None

    def tracking_hook(m, i, o):
        if not return_logits:
            lat.append(o.detach().float().cpu())
            return

        # KEEP native model dtype for the coda/head math; only cast to float32
        # once we're down to the single position we actually want, for numpy.
        h_state = o.detach()
        lat.append(h_state[0, token_index, :].cpu().float())

        with torch.no_grad():
            # Deliberately no try/except: a failed step here means the
            # reconstruction is wrong for this trajectory, and a loud crash
            # is far better than silently shortening logits_list, which
            # would misalign every later step against the wrong unroll index.
            step_logits = _replicate_coda_head(model, h_state, freqs_cis)
            logits_list.append(step_logits[0, token_index, :].cpu().float())

    h = mod.register_forward_hook(tracking_hook)
    try:
        with torch.no_grad():
            # int, NOT tensor; forward, NOT generate
            out = model(input_ids=ids, num_steps=num_steps)
    finally:
        h.remove()  # removed even if the forward raised

    if not return_logits:
        return torch.stack([latent[0, token_index, :] for latent in lat]).numpy()

    if validate_logits and logits_list:
        real_final = out.logits[0, token_index, :].float().cpu()
        recon_final = logits_list[-1]
        if not torch.allclose(real_final, recon_final, atol=1e-2, rtol=1e-2):
            diff = (real_final - recon_final).abs().max().item()
            raise RuntimeError(
                "extract_trajectory(return_logits=True): the reconstructed "
                "final-step logits do not match the model's own real "
                "forward() output for this prompt (max abs diff "
                f"{diff:.4g}). The coda replication in _replicate_coda_head "
                "does not match this model/revision's actual architecture — "
                "do not trust any logits from this call until this is fixed."
            )

    result = {"latents": torch.stack(lat).numpy()}
    if logits_list:
        result["logits"] = torch.stack(logits_list).numpy()
    return result
