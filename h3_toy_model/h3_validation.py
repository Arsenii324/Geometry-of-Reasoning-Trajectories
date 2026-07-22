"""Rigorous empirical validation of H3 (the Contraction Bottleneck Theorem) on
the toy TinyRecursiveModel -- the one model in this whole project that can
actually be run without a GPU or Huginn access.

WHY THIS EXISTS: the original presentation.md claim ("100% unconstrained vs
90% contracting in-distribution; 0% extrapolation for both") only ever
compared two beta_regularization values (0 vs 5.0), never actually measured
the model's real local contraction factor rho(d_h R), and used
compute_jacobian_reg_loss (a single-sample finite-difference proxy) purely as
a *training loss term*, never as a post-hoc diagnostic. Nobody had verified
that beta_regularization actually controls rho in a predictable way, nor
tested H3's actual causal claim (lower rho -> worse state-holding) with more
than two beta values.

A CONFOUND FOUND AND CONTROLLED FOR HERE: the original extrapolation-accuracy
metric (exact classification match at length 45) is confounded by output
*label-space* extrapolation, not just internal state-tracking -- verified
empirically: with train_len=15, num_train=4000, the training label
distribution is Binomial(15, 0.3) truncated at 0..11, and at test_len=45 the
true-label distribution is Binomial(45, 0.3), 72.12% of whose mass falls on
label values (12-45) that NEVER appear in the training set at all. A model
with a perfect internal counter would *still* score ~0% exact-match here,
because its classifier head was never trained to produce those classes. The
original presentation.md already flagged this qualitatively ("due to
classification label shifts") -- this file adds the exact number and, more
importantly, two additional metrics that are NOT confounded this way:
  1. Spearman(predicted_class, true_count) on the extrapolation set -- does
     the model's output still track magnitude even when it can't hit the
     exact (unseen) class?
  2. A closed-form linear-probe R^2: freeze the trained model, fit ridge
     regression from its final hidden state h_T to the true count on a
     probe-train split of length-45 sequences, evaluate R^2 on held-out
     probe-test sequences. This tests whether count information is linearly
     decodable from the internal state at all, independent of whether the
     original (never-exposed-to-those-classes) softmax head can read it out.

Run: .venv/bin/python h3_validation.py
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import spearmanr

from spectral import spectral_norm_at_point
from synthetic_tasks import SequenceCountingDataset
from train_tiny_recursive import TinyRecursiveModel, train_and_evaluate

BETAS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
SEEDS = (0, 1, 2)
TASK = "counting"
NUM_EPOCHS = 8
NUM_STEPS = 16
TRAIN_LEN, TEST_LEN = 15, 45
N_RHO_SAMPLES = 12  # hidden states sampled per model for the rho estimate
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "h3_results", "h3_sweep.csv")


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


def label_distribution_confound_check() -> tuple[float, float]:
    """Quantify the output-label-space extrapolation confound (see module docstring)."""
    train_ds = SequenceCountingDataset(4000, TRAIN_LEN)
    test_ds = SequenceCountingDataset(800, TEST_LEN)
    train_labels = set(train_ds.labels.tolist())
    unseen_mask = ~np.isin(test_ds.labels, list(train_labels))
    frac_unseen = float(unseen_mask.mean())
    max_train_label = int(train_ds.labels.max())
    return frac_unseen, max_train_label


def measure_rho_along_trajectories(
    model: TinyRecursiveModel, num_steps: int, n_samples: int, seed: int
) -> np.ndarray:
    """Sample hidden states from real forward passes (both train- and
    test-length inputs) and estimate the local operator norm at each.

    Sampling from real trajectories (not just h=0) matters: the recurrent map
    is nonlinear, so its local Lipschitz constant genuinely varies across the
    state space, and h=0 alone (the fixed starting point every rollout
    shares) is not necessarily representative of where the trajectory
    actually spends its time.
    """
    model.eval()
    rhos = []
    for i in range(n_samples):
        length = TRAIN_LEN if i % 2 == 0 else TEST_LEN
        g2 = torch.Generator().manual_seed(seed + 2000 + i)
        ids = torch.randint(0, 2, (1, length), generator=g2)
        with torch.no_grad():
            x_embeds = model.embedding(ids)
            context = torch.mean(x_embeds, dim=1)[0].detach()
            h = torch.zeros(model.hidden_dim)
            t_star = int(torch.randint(0, num_steps, (1,), generator=g2).item())
            for _ in range(t_star):
                h = model.step(h.unsqueeze(0), context.unsqueeze(0))[0]
        h = h.detach()

        def step_fn(hh: torch.Tensor, ctx=context) -> torch.Tensor:
            return model.step(hh.unsqueeze(0), ctx.unsqueeze(0))[0]

        rho = spectral_norm_at_point(step_fn, h, n_iter=25)
        rhos.append(rho)
    return np.array(rhos)


def linear_probe_r2(model: TinyRecursiveModel, num_steps_extrap: int, seed: int) -> float:
    """Closed-form ridge-regression probe: final hidden state h_T -> true count,
    fit on one split of length-45 sequences, evaluated (R^2) on a held-out
    split. Tests decodability of count information independent of the
    (never-trained-on-those-classes) softmax head.
    """
    model.eval()
    n_probe_train, n_probe_test = 500, 300

    def collect(n: int, np_seed: int) -> tuple[np.ndarray, np.ndarray]:
        # SequenceCountingDataset draws its own randomness at construction time
        # (np.random.choice, not seeded per-call) -- reseed numpy immediately
        # before constructing it so probe-train/probe-test splits are
        # reproducible and (with different seeds) disjoint in expectation.
        np.random.seed(np_seed)
        ds = SequenceCountingDataset(n, TEST_LEN)
        with torch.no_grad():
            ids = torch.tensor(ds.sequences, dtype=torch.long)
            _, states = model(ids, num_steps=num_steps_extrap)
            h_final = states[-1].numpy()  # [n, hidden_dim]
        return h_final, ds.sequences.sum(axis=1).astype(float)  # unclipped true count

    h_train, y_train = collect(n_probe_train, seed + 3001)
    h_test, y_test = collect(n_probe_test, seed + 3002)

    # Ridge regression, closed form: w = (X^T X + lambda I)^-1 X^T y
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
    in_acc, ex_acc, model = train_and_evaluate(
        task_name=TASK, num_steps=NUM_STEPS, num_epochs=NUM_EPOCHS,
        beta_regularization=beta, seed=seed,
    )
    train_seconds = time.time() - t0

    # Predicted-vs-true magnitude tracking on the extrapolation set.
    extrap_steps = int(NUM_STEPS * (TEST_LEN / TRAIN_LEN))
    test_ds = SequenceCountingDataset(500, TEST_LEN)
    with torch.no_grad():
        ids = torch.tensor(test_ds.sequences, dtype=torch.long)
        logits, _ = model(ids, num_steps=extrap_steps)
        preds = torch.argmax(logits, dim=-1).numpy()
    true_counts = test_ds.sequences.sum(axis=1)  # unclipped true count, not the clipped label
    rho_s, rho_p = spearmanr(preds, true_counts)

    probe_r2 = linear_probe_r2(model, extrap_steps, seed)
    rhos = measure_rho_along_trajectories(model, NUM_STEPS, N_RHO_SAMPLES, seed)

    return RunResult(
        beta=beta, seed=seed, in_acc=in_acc, ex_acc=ex_acc,
        spearman_pred_vs_true=float(rho_s), spearman_p=float(rho_p),
        probe_r2=float(probe_r2),
        rho_mean=float(rhos.mean()), rho_min=float(rhos.min()),
        rho_max=float(rhos.max()), rho_std=float(rhos.std()),
        train_seconds=train_seconds,
    )


def main() -> None:
    frac_unseen, max_train_label = label_distribution_confound_check()
    print(f"Label-space confound check: {frac_unseen:.2%} of test(len=45) true labels "
          f"never appear in train(len=15) (max train label = {max_train_label}).")
    print(f"Sweeping {len(BETAS)} betas x {len(SEEDS)} seeds = {len(BETAS) * len(SEEDS)} runs "
          f"({NUM_EPOCHS} epochs each)...\n")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    results: list[RunResult] = []
    for beta in BETAS:
        for seed in SEEDS:
            r = run_one(beta, seed)
            results.append(r)
            print(
                f"beta={beta:5.1f} seed={seed} | in_acc={r.in_acc:6.2f}% ex_acc={r.ex_acc:6.2f}% "
                f"| pred~true rho={r.spearman_pred_vs_true:+.3f} (p={r.spearman_p:.3f}) "
                f"| probe R2={r.probe_r2:+.3f} | rho(dR) mean={r.rho_mean:.3f} "
                f"[{r.rho_min:.3f},{r.rho_max:.3f}] | {r.train_seconds:.1f}s"
            )

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(RunResult.__dataclass_fields__.keys()))
        for r in results:
            w.writerow([getattr(r, k) for k in RunResult.__dataclass_fields__])
    print(f"\nSaved {RESULTS_CSV}")


if __name__ == "__main__":
    main()
