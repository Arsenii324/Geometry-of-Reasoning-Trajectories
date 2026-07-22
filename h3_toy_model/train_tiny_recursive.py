import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional
import os

from synthetic_tasks import get_extrapolation_datasets


class TinyRecursiveModel(nn.Module):
    """
    A minimal weight-tied recurrent model (similar in spirit to TRM) 
    designed to test unrolling dynamics and contraction regularizations on synthetic tasks.
    """
    def __init__(self, vocab_size: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 2. Adapter (maps concatenated recurrent state and token embedding back to hidden_dim)
        self.adapter = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        
        # 3. Recurrent core layer (weight-tied MLP or transformer-like block)
        self.core_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 4. Final prediction head
        self.prediction_head = nn.Linear(hidden_dim, num_classes)
        
    def step(self, h_prev: torch.Tensor, x_embed: torch.Tensor) -> torch.Tensor:
        """
        Executes a single recurrent reasoning step.
        h_{t+1} = Core(Adapter([h_t; x_embed]))
        """
        # Concatenate and pass through adapter
        mixed = torch.cat([h_prev, x_embed], dim=-1)
        adapter_out = self.adapter(mixed)
        
        # Residual connection over the core layer
        h_next = self.core_layer(adapter_out) + adapter_out
        return h_next

    def forward(self, input_ids: torch.Tensor, num_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unrolls the recurrent block for `num_steps` over the input sequence representations.
        For simple synthetic tasks, we average or sum token embeddings as the context representation,
        then iterate a latent state h_t for num_steps.
        
        Returns:
            - logits: [batch_size, num_classes]
            - states: [num_steps, batch_size, hidden_dim] (for trajectory logging)
        """
        batch_size, seq_len = input_ids.shape
        
        # Extract token embeddings and pool them (mean) to represent context
        x_embeds = self.embedding(input_ids)  # [batch_size, seq_len, hidden_dim]
        context = torch.mean(x_embeds, dim=1)  # [batch_size, hidden_dim]
        
        # Initialize recurrent state h_0 to random/zeros
        h = torch.zeros((batch_size, self.hidden_dim), device=input_ids.device)
        
        states = []
        for _ in range(num_steps):
            h = self.step(h, context)
            states.append(h)
            
        states_tensor = torch.stack(states, dim=0)  # [num_steps, batch_size, hidden_dim]
        logits = self.prediction_head(h)
        return logits, states_tensor


def compute_jacobian_reg_loss(
    model: TinyRecursiveModel, 
    h: torch.Tensor, 
    context: torch.Tensor, 
    eps: float = 1e-4
) -> torch.Tensor:
    """
    Computes a finite-difference approximation of the local Jacobian norm:
    J_reg = ||f(h + delta, context) - f(h, context)||_2 / ||delta||_2
    
    If J_reg is regularized to be strictly less than 1, we force the map
    to behave as a strict contraction (testing H3).
    """
    # Generate random perturbations delta of size eps
    delta = torch.randn_like(h)
    delta = delta / torch.linalg.norm(delta, dim=-1, keepdim=True) * eps
    
    # Evaluate recurrent map on h and perturbed h + delta
    h_next = model.step(h, context)
    h_perturbed_next = model.step(h + delta, context)
    
    # Calculate local expansion ratio
    diff_next = torch.linalg.norm(h_perturbed_next - h_next, dim=-1)
    diff_prev = torch.linalg.norm(delta, dim=-1)
    
    # Expansion norm (ratio)
    local_lipschitz = diff_next / (diff_prev + 1e-12)
    
    # Penalize local lipschitz constants greater than target contractive rate
    # E.g., we want to minimize this value to enforce contraction
    return torch.mean(local_lipschitz)


def train_and_evaluate(
    task_name: str = "counting",
    hidden_dim: int = 64,
    num_steps: int = 16,
    num_epochs: int = 15,
    beta_regularization: float = 0.0,  # Set to > 0.0 (e.g. 0.5) to enforce contraction (STARS proxy)
    seed: Optional[int] = None,  # fixes torch+numpy RNG; None preserves old (unseeded) behavior
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"\n================= Training TRM on: {task_name.upper()} =================")
    print(f"Regularization Factor (beta): {beta_regularization} (0 = unconstrained, >0 = forced contraction)")

    # Generate train and extrapolation test sets
    train_len, test_len = 15, 45
    train_ds, test_ds = get_extrapolation_datasets(
        task_name, train_len=train_len, test_len=test_len, num_train=4000, num_test=800
    )
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # Task specific config
    vocab_size = 2
    num_classes = 128 if task_name == "counting" else (2 if task_name == "parity" else 3)
    
    model = TinyRecursiveModel(vocab_size=vocab_size, hidden_dim=hidden_dim, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_reg = 0.0
        
        for input_ids, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            logits, states = model(input_ids, num_steps=num_steps)
            ce_loss = criterion(logits, labels)
            
            # Calculate Jacobian regularization (finite-difference approximation)
            # We sample a random step to regularize (e.g. step T//2)
            step_idx = num_steps // 2
            h_state = states[step_idx]
            
            # Reconstruct context
            x_embeds = model.embedding(input_ids)
            context = torch.mean(x_embeds, dim=1)
            
            reg_loss = compute_jacobian_reg_loss(model, h_state, context)
            
            total_loss = ce_loss + beta_regularization * reg_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_ce += ce_loss.item()
            epoch_reg += reg_loss.item()
            
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Loss: {epoch_loss/len(train_loader):.4f} | CE: {epoch_ce/len(train_loader):.4f} | JSR Proxy: {epoch_reg/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    
    # 1. In-distribution evaluation (same length as training)
    in_dist_ds, _ = get_extrapolation_datasets(task_name, train_len=train_len, test_len=train_len, num_train=10, num_test=500)
    in_dist_loader = DataLoader(in_dist_ds, batch_size=64, shuffle=False)
    
    correct_in = 0
    total_in = 0
    with torch.no_grad():
        for input_ids, labels in in_dist_loader:
            logits, _ = model(input_ids, num_steps=num_steps)
            preds = torch.argmax(logits, dim=-1)
            correct_in += (preds == labels).sum().item()
            total_in += labels.size(0)
            
    in_acc = (correct_in / total_in) * 100
    
    # 2. Extrapolation evaluation (length tripled!)
    # We increase unroll steps proportionally to sequence length increase
    extrap_steps = int(num_steps * (test_len / train_len))
    
    correct_ex = 0
    total_ex = 0
    with torch.no_grad():
        for input_ids, labels in test_loader:
            logits, _ = model(input_ids, num_steps=extrap_steps)
            preds = torch.argmax(logits, dim=-1)
            correct_ex += (preds == labels).sum().item()
            total_ex += labels.size(0)
            
    ex_acc = (correct_ex / total_ex) * 100
    
    print("\n--- Evaluation Results ---")
    print(f"In-Distribution Accuracy (Length {train_len}): {in_acc:.2f}%")
    print(f"Extrapolation Accuracy (Length {test_len}, Steps {extrap_steps}): {ex_acc:.2f}%")
    return in_acc, ex_acc, model


if __name__ == "__main__":
    # Example execution (unregularized). seed=0 makes this reproducible --
    # the original unseeded version gave 100%/100%/90% in-distribution
    # accuracy across 3 repeats of the beta=5.0 condition (2026-07-17 audit).
    train_and_evaluate(task_name="counting", num_epochs=5, beta_regularization=0.0, seed=0)

    # Example execution (highly regularized - forced contraction)
    train_and_evaluate(task_name="counting", num_epochs=5, beta_regularization=5.0, seed=0)
