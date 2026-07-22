"""H3 validation on the recurrent-over-TIME architecture (SequentialCountingRNN),
run identically to h3_validation.py's recurrent-over-DEPTH sweep for a direct
comparison. Same task (counting), same train/test lengths, same beta grid,
same seeds, same metrics (exact-match accuracy, linear-probe R^2 on the final
hidden state, Spearman(prediction, true count), measured rho(d_h R) via
power iteration) -- only the architecture differs.

Prediction being tested: H3's mechanism (contraction erases dependence on
h_0, and Section 5 of contraction_proof.md applies this to counting/FSA
state-tracking) should bite HERE, unlike on the depth-recurrent model, because
here the running count genuinely only exists via h_t carrying it forward --
there's no re-injected context to fall back on.

Run: .venv/bin/python sequential_h3_validation.py
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from sequential_rnn import SequentialCountingRNN, jacobian_reg_loss
from spectral import spectral_norm_at_point
from synthetic_tasks import SequenceCountingDataset
from torch.utils.data import DataLoader

BETAS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
SEEDS = (0, 1, 2)
HIDDEN_DIM = 64
NUM_EPOCHS = 8
TRAIN_LEN, TEST_LEN = 15, 45
NUM_CLASSES = 128
N_RHO_SAMPLES = 12
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "h3_results", "sequential_sweep.csv")


@dataclass
class RunResult:
    beta: float
    seed: int
    in_acc: float
    ex_acc: float
    spearman_pred_vs_true: float
    spearman_p: float
    probe_r2: float
    rho_mean: float
    rho_min: float
    rho_max: float
    rho_std: float
    train_seconds: float = field(default=0.0)


def train_and_evaluate(beta: float, seed: int) -> tuple[float, float, SequentialCountingRNN]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = SequenceCountingDataset(4000, TRAIN_LEN, num_classes=NUM_CLASSES)
    test_ds = SequenceCountingDataset(800, TEST_LEN, num_classes=NUM_CLASSES)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = SequentialCountingRNN(vocab_size=2, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(NUM_EPOCHS):
        for input_ids, labels in train_loader:
            optimizer.zero_grad()
            logits, states = model(input_ids)
            ce_loss = criterion(logits, labels)

            # Regularize at a step partway through the sequence, same
            # "sample the middle" convention as the depth-recurrent baseline.
            t_star = TRAIN_LEN // 2
            h_state = states[t_star - 1] if t_star > 0 else torch.zeros(input_ids.shape[0], HIDDEN_DIM)
            x_embed_t = model.embedding(input_ids)[:, t_star, :]
            reg_loss = jacobian_reg_loss(model, h_state, x_embed_t)

            total_loss = ce_loss + beta * reg_loss
            total_loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        in_ids = torch.tensor(SequenceCountingDataset(500, TRAIN_LEN, num_classes=NUM_CLASSES).sequences, dtype=torch.long)
        in_labels_ds = SequenceCountingDataset(500, TRAIN_LEN, num_classes=NUM_CLASSES)
        in_logits, _ = model(torch.tensor(in_labels_ds.sequences, dtype=torch.long))
        in_preds = torch.argmax(in_logits, dim=-1).numpy()
        in_acc = float((in_preds == in_labels_ds.labels).mean() * 100)

        correct_ex, total_ex = 0, 0
        for input_ids, labels in test_loader:
            logits, _ = model(input_ids)
            preds = torch.argmax(logits, dim=-1)
            correct_ex += (preds == labels).sum().item()
            total_ex += labels.size(0)
        ex_acc = (correct_ex / total_ex) * 100

    return in_acc, ex_acc, model


def measure_rho(model: SequentialCountingRNN, n_samples: int, seed: int) -> np.ndarray:
    """Sample (h, x_embed) pairs from real forward passes -- both train- and
    test-length sequences, at a random token position within each -- and
    estimate the local operator norm of step() at each.
    """
    model.eval()
    rhos = []
    for i in range(n_samples):
        length = TRAIN_LEN if i % 2 == 0 else TEST_LEN
        g = torch.Generator().manual_seed(seed + 2000 + i)
        ids = torch.randint(0, 2, (1, length), generator=g)
        with torch.no_grad():
            x_embeds = model.embedding(ids)[0]  # [length, hidden_dim]
            h = torch.zeros(model.hidden_dim)
            t_star = int(torch.randint(0, length, (1,), generator=g).item())
            for t in range(t_star):
                h = model.step(h.unsqueeze(0), x_embeds[t].unsqueeze(0))[0]
        h = h.detach()
        x_t = x_embeds[t_star].detach()

        def step_fn(hh: torch.Tensor, xt=x_t) -> torch.Tensor:
            return model.step(hh.unsqueeze(0), xt.unsqueeze(0))[0]

        rhos.append(spectral_norm_at_point(step_fn, h, n_iter=25))
    return np.array(rhos)


def linear_probe_r2(model: SequentialCountingRNN, seed: int) -> float:
    """Same closed-form ridge-regression probe as h3_validation.py, adapted:
    forward() here takes only input_ids (no num_steps -- it's implied by
    seq_len), everything else identical.
    """
    model.eval()
    n_probe_train, n_probe_test = 500, 300

    def collect(n: int, np_seed: int) -> tuple[np.ndarray, np.ndarray]:
        np.random.seed(np_seed)
        ds = SequenceCountingDataset(n, TEST_LEN, num_classes=NUM_CLASSES)
        with torch.no_grad():
            ids = torch.tensor(ds.sequences, dtype=torch.long)
            _, states = model(ids)
            h_final = states[-1].numpy()
        return h_final, ds.sequences.sum(axis=1).astype(float)

    h_train, y_train = collect(n_probe_train, seed + 3001)
    h_test, y_test = collect(n_probe_test, seed + 3002)

    lam = 1.0
    X = np.hstack([h_train, np.ones((h_train.shape[0], 1))])
    Xt = np.hstack([h_test, np.ones((h_test.shape[0], 1))])
    A = X.T @ X + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ y_train)
    pred = Xt @ w
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def run_one(beta: float, seed: int) -> RunResult:
    t0 = time.time()
    in_acc, ex_acc, model = train_and_evaluate(beta, seed)
    train_seconds = time.time() - t0

    test_ds = SequenceCountingDataset(500, TEST_LEN, num_classes=NUM_CLASSES)
    with torch.no_grad():
        ids = torch.tensor(test_ds.sequences, dtype=torch.long)
        logits, _ = model(ids)
        preds = torch.argmax(logits, dim=-1).numpy()
    true_counts = test_ds.sequences.sum(axis=1)
    rho_s, rho_p = spearmanr(preds, true_counts)

    probe_r2 = linear_probe_r2(model, seed)
    rhos = measure_rho(model, N_RHO_SAMPLES, seed)

    return RunResult(
        beta=beta, seed=seed, in_acc=in_acc, ex_acc=ex_acc,
        spearman_pred_vs_true=float(rho_s), spearman_p=float(rho_p),
        probe_r2=float(probe_r2),
        rho_mean=float(rhos.mean()), rho_min=float(rhos.min()),
        rho_max=float(rhos.max()), rho_std=float(rhos.std()),
        train_seconds=train_seconds,
    )


def main() -> None:
    print(f"Sequential (recurrent-over-time) RNN H3 sweep: {len(BETAS)} betas x {len(SEEDS)} seeds "
          f"= {len(BETAS) * len(SEEDS)} runs ({NUM_EPOCHS} epochs each)...\n")
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    results: list[RunResult] = []
    for beta in BETAS:
        for seed in SEEDS:
            r = run_one(beta, seed)
            results.append(r)
            print(
                f"beta={beta:5.1f} seed={seed} | in_acc={r.in_acc:6.2f}% ex_acc={r.ex_acc:6.2f}% "
                f"| pred~true rho={r.spearman_pred_vs_true:+.3f} | probe R2={r.probe_r2:+.3f} "
                f"| rho(dR) mean={r.rho_mean:.3f} [{r.rho_min:.3f},{r.rho_max:.3f}] | {r.train_seconds:.1f}s"
            )

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(RunResult.__dataclass_fields__.keys()))
        for r in results:
            w.writerow([getattr(r, k) for k in RunResult.__dataclass_fields__])
    print(f"\nSaved {RESULTS_CSV}")


if __name__ == "__main__":
    main()
