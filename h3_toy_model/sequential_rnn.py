"""Recurrent-OVER-TIME counterpart to TinyRecursiveModel (which is recurrent-
over-DEPTH, like Huginn: full context re-available at every unroll step, h_0
carries no task information). This model instead processes one token per
step -- h_{t+1} = f(h_t, x_t) -- so the running count genuinely must be
carried forward IN h, matching the architecture the Contraction Bottleneck
Theorem's own worked example (contraction_proof.md sec 5) implicitly assumes.

STATUS: new 2026-07-18. Built to test a hypothesis from h3_results/FINDINGS.md:
that H3's mechanism doesn't bite on recurrent-depth architectures because
task info is never encoded via h_0-dependence in the first place. This is the
architecture where it should, if the theorem's applied claim is right.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SequentialCountingRNN(nn.Module):
    """Elman-style RNN: h_{t+1} = tanh(W_h h_t + W_x x_t + b), one input
    token consumed per recurrent step. A linear head reads the running count
    off the FINAL hidden state only (after the whole sequence is consumed) --
    same evaluation protocol as TinyRecursiveModel, so results are comparable.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_x = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.prediction_head = nn.Linear(hidden_dim, num_classes)

    def step(self, h_prev: torch.Tensor, x_embed: torch.Tensor) -> torch.Tensor:
        """One token's update: h_{t+1} = tanh(W_h h_t + W_x x_t)."""
        return torch.tanh(self.W_h(h_prev) + self.W_x(x_embed))

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume input_ids one token per recurrent step (num_steps = seq_len,
        not a free parameter like the depth-recurrent model -- that's the point).

        Returns:
            logits: [batch, num_classes] from the final hidden state.
            states: [seq_len, batch, hidden_dim] for trajectory logging.
        """
        batch_size, seq_len = input_ids.shape
        x_embeds = self.embedding(input_ids)  # [batch, seq_len, hidden_dim]
        h = torch.zeros(batch_size, self.hidden_dim, device=input_ids.device)
        states = []
        for t in range(seq_len):
            h = self.step(h, x_embeds[:, t, :])
            states.append(h)
        states_tensor = torch.stack(states, dim=0)
        logits = self.prediction_head(h)
        return logits, states_tensor


def jacobian_reg_loss(
    model: SequentialCountingRNN, h: torch.Tensor, x_embed: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    """Same finite-difference local-Lipschitz proxy as
    train_tiny_recursive.py::compute_jacobian_reg_loss, adapted to this
    model's step() signature (h, x_embed) instead of (h, context) -- the
    formula is identical, only the call target differs.
    """
    delta = torch.randn_like(h)
    delta = delta / torch.linalg.norm(delta, dim=-1, keepdim=True) * eps
    h_next = model.step(h, x_embed)
    h_perturbed_next = model.step(h + delta, x_embed)
    diff_next = torch.linalg.norm(h_perturbed_next - h_next, dim=-1)
    diff_prev = torch.linalg.norm(delta, dim=-1)
    return torch.mean(diff_next / (diff_prev + 1e-12))
