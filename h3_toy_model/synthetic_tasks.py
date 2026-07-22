
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceCountingDataset(Dataset):
    """
    Synthetic counting task.
    Sequence contains tokens '0' (filler) and '1' (target).
    The label is the total number of '1's in the sequence.
    Trains on short lengths, tests on long lengths to evaluate length extrapolation.
    """
    def __init__(self, num_samples: int, seq_length: int, num_classes: int = 128):
        super().__init__()
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_classes = num_classes

        # Generate random sequences of 0 and 1
        # Probability of target '1' is 0.3
        self.sequences = np.random.choice([0, 1], size=(num_samples, seq_length), p=[0.7, 0.3])
        self.labels = np.sum(self.sequences == 1, axis=1)

        # Clip labels to num_classes - 1
        self.labels = np.clip(self.labels, 0, num_classes - 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class ParityTrackingDataset(Dataset):
    """
    Synthetic parity task.
    Sequence is binary. Output label is 1 if the sum of elements is odd, 0 if even.
    This tests cyclic (looping) latent trajectories since state transitions alternate between 0 and 1.
    """
    def __init__(self, num_samples: int, seq_length: int):
        super().__init__()
        self.num_samples = num_samples
        self.seq_length = seq_length

        self.sequences = np.random.choice([0, 1], size=(num_samples, seq_length), p=[0.5, 0.5])
        self.labels = np.sum(self.sequences, axis=1) % 2

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class FSATrackingDataset(Dataset):
    """
    Finite State Automata (FSA) state-tracking task.
    We define a simple FSA with 3 states (S0, S1, S2) and transition rules under inputs 'a' (0) and 'b' (1).
    Rules:
      S0 -- 0 -> S1,  S0 -- 1 -> S0
      S1 -- 0 -> S2,  S1 -- 1 -> S0
      S2 -- 0 -> S2,  S2 -- 1 -> S0
    Tests if the model can track discrete state transitions.
    """
    def __init__(self, num_samples: int, seq_length: int):
        super().__init__()
        self.num_samples = num_samples
        self.seq_length = seq_length

        # Generate random inputs (0 and 1)
        self.sequences = np.random.choice([0, 1], size=(num_samples, seq_length), p=[0.5, 0.5])
        self.labels = []

        for seq in self.sequences:
            state = 0  # Initial state S0
            for char in seq:
                if state == 0:
                    state = 1 if char == 0 else 0
                elif state == 1:
                    state = 2 if char == 0 else 0
                elif state == 2:
                    state = 2 if char == 0 else 0
            self.labels.append(state)

        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# Simple helper to generate train/test splits for length extrapolation
def get_extrapolation_datasets(
    task_name: str,
    num_train: int = 5000,
    num_test: int = 1000,
    train_len: int = 15,
    test_len: int = 40
) -> tuple[Dataset, Dataset]:
    """
    Generates training data on short sequence lengths and test data on longer lengths
    to analyze extrapolation failure/success modes under contraction constraints.
    """
    if task_name == "counting":
        train_ds = SequenceCountingDataset(num_train, train_len)
        test_ds = SequenceCountingDataset(num_test, test_len)
    elif task_name == "parity":
        train_ds = ParityTrackingDataset(num_train, train_len)
        test_ds = ParityTrackingDataset(num_test, test_len)
    elif task_name == "fsa":
        train_ds = FSATrackingDataset(num_train, train_len)
        test_ds = FSATrackingDataset(num_test, test_len)
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    return train_ds, test_ds
