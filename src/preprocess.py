"""
Data preprocessing for DCQ experiment.
Generates synthetic language data for training and evaluation.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Tuple

def get_synthetic_language_data(vocab_size: int = 1000,  
                               seq_len: int = 32,  
                               n_samples: int = 5000,  
                               pattern: str = "uniform") -> Tuple[TensorDataset, TensorDataset]:
    """Create two splits of synthetic token sequences.
    pattern="uniform"  ‑> each token equally likely
    pattern="bursty"   ‑> Zipf-like distribution (to test robustness)
    """
    if pattern == "uniform":
        probs = np.ones(vocab_size) / vocab_size
    else:  # bursty / Zipf
        ranks = np.arange(1, vocab_size + 1)
        probs = 1. / ranks
        probs = probs / probs.sum()
    
    tokens = np.random.choice(vocab_size, size=(n_samples, seq_len + 1), p=probs).astype(np.int64)
    x = torch.from_numpy(tokens[:, :-1])
    y = torch.from_numpy(tokens[:, 1:])
    
    split = int(0.9 * n_samples)
    train_ds = TensorDataset(x[:split], y[:split])
    val_ds   = TensorDataset(x[split:], y[split:])
    return train_ds, val_ds
