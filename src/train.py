"""
Training utilities for DCQ experiment.
Contains model definitions and training functions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def gumbel_noise(t: torch.Tensor) -> torch.Tensor:
    """Sample Gumbel(0,1) noise – same shape as *t*."""
    u = torch.rand_like(t)
    return -torch.log(-torch.log(u + 1e-9) + 1e-9)

def gumbel_sinkhorn(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """A tiny stand-in: row-wise relaxed one-hot via softmax((logits+g)/tau)."""
    g = gumbel_noise(logits)
    return F.softmax((logits + g) / tau, dim=-1)

class DCQLinear(nn.Module):
    """Differentiable-Codebook-Quantised fully-connected layer."""
    def __init__(self, in_f: int, out_f: int,
                 cg_size: int = 16, cl_size: int = 8,
                 tau_init: float = 5.0):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.in_features, self.out_features = in_f, out_f  # For LoRA compatibility
        self.cg_size, self.cl_size = cg_size, cl_size
        self.CG = nn.Parameter(torch.randn(cg_size, dtype=torch.float32) * 0.02)
        self.CL = nn.Parameter(torch.randn(out_f, cl_size, dtype=torch.float32) * 0.02)
        self.idx_i = nn.Parameter(torch.randn(out_f, in_f, cg_size) * 0.01)
        self.idx_j = nn.Parameter(torch.randn(out_f, in_f, cl_size) * 0.01)
        self.register_buffer("tau", torch.tensor(tau_init))

    def forward(self, x):
        B = x.size(0)
        P_i = gumbel_sinkhorn(self.idx_i, self.tau.item())
        P_j = gumbel_sinkhorn(self.idx_j, self.tau.item())
        W_global = torch.einsum('oik,k->oi', P_i, self.CG)
        W_local  = torch.einsum('oij,oj->oi', P_j, self.CL)
        W = W_global + W_local
        return F.linear(x, W)

    def harden(self):
        """Turn relaxed probabilities into hard argmax indices (inference)."""
        with torch.no_grad():
            hard_i = self.idx_i.argmax(-1)
            hard_j = self.idx_j.argmax(-1)
            W = self.CG[hard_i] + self.CL[torch.arange(self.out_f).unsqueeze(1), hard_j]
        return W

    def utilisation_penalty(self, min_rate: float = 0.01) -> torch.Tensor:
        P_i = F.softmax(self.idx_i, dim=-1)
        util = P_i.mean(dim=(0, 1))
        return F.relu(min_rate - util).mean()

class ToyLM(nn.Module):
    def __init__(self, vocab: int, embed: int, hidden: int, dcq: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        if dcq:
            self.fc = DCQLinear(embed, vocab)
        else:
            self.fc = nn.Linear(embed, vocab)

    def forward(self, x):
        B, T = x.shape
        emb = self.embed(x)  # (B, T, embed)
        emb_flat = emb.view(-1, emb.size(-1))  # (B*T, embed)
        logits = self.fc(emb_flat)  # (B*T, vocab)
        return logits.view(B, T, -1)  # (B, T, vocab)

class LoRALinear(nn.Module):
    """Simple, standalone LoRA wrapper for nn.Linear."""
    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: int = 16):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(r, base_layer.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base_layer.out_features, r))
        self.scaling = alpha / r
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(x, self.B @ self.A) * self.scaling
        return base_out + lora_out

def inject_lora(model: ToyLM, r: int = 4):
    """Replace the *fc* layer with a LoRA-decorated version."""
    model.fc = LoRALinear(model.fc, r=r)
    return model

def run_epoch(model: nn.Module, data_loader: DataLoader, optim=None,  
              penalty_beta: float = 0.0, device: str = "cpu") -> float:
    """If *optim* is None → evaluation mode (no grad).  Returns average NLL."""
    training = optim is not None
    total_loss, n_tokens = 0.0, 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        if training:
            if isinstance(model.fc, DCQLinear):
                ent = (-F.softmax(model.fc.idx_i, -1) * F.log_softmax(model.fc.idx_i, -1)).sum() / model.fc.idx_i.numel()
                util_pen = model.fc.utilisation_penalty()
                loss = loss + penalty_beta * ent + 0.1 * util_pen
            optim.zero_grad()
            loss.backward()
            optim.step()
        total_loss += loss.item() * x.size(0) * x.size(1)
        n_tokens += x.size(0) * x.size(1)
    return math.exp(total_loss / n_tokens)
