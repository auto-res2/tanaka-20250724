"""
Evaluation utilities for DCQ experiment.
Contains experiment implementations and plotting functions.
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List

from train import ToyLM, inject_lora, run_epoch, DCQLinear
from preprocess import get_synthetic_language_data

def experiment_1(device: str = "cpu") -> Dict[str, List[float]]:
    """Parameter gap closure experiment."""
    print("\n===== Experiment 1: Parameter-Gap Closure (toy) =====")
    train_ds, val_ds = get_synthetic_language_data(pattern="uniform")
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=64)

    model_fp = ToyLM(vocab=1000, embed=128, hidden=256, dcq=False).to(device)
    opt_fp   = torch.optim.AdamW(model_fp.parameters(), lr=2e-3)
    ppl_fp_hist = []
    for epoch in range(3):
        _ = run_epoch(model_fp, train_dl, opt_fp, device=device)
        ppl = run_epoch(model_fp, val_dl, device=device)
        ppl_fp_hist.append(ppl)
        print(f"Baseline epoch {epoch+1}: PPL={ppl:.2f}")

    model_dcq = ToyLM(vocab=1000, embed=128, hidden=256, dcq=True).to(device)
    opt_dcq   = torch.optim.AdamW(model_dcq.parameters(), lr=2e-3)
    ppl_dcq_hist = []
    for epoch in range(3):
        if isinstance(model_dcq.fc, DCQLinear):
            model_dcq.fc.tau.mul_(0.5)
        _ = run_epoch(model_dcq, train_dl, opt_dcq, penalty_beta=0.02, device=device)
        ppl = run_epoch(model_dcq, val_dl, device=device)
        ppl_dcq_hist.append(ppl)
        print(f"DCQ epoch {epoch+1}: PPL={ppl:.2f}; tau={model_dcq.fc.tau.item():.2f}")

    def trainable_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainables – baseline: {trainable_params(model_fp):,}")
    print(f"Trainables – DCQ     : {trainable_params(model_dcq):,}")

    plt.figure()
    plt.plot(ppl_fp_hist, label="fp32")
    plt.plot(ppl_dcq_hist, label="DCQ")
    plt.xlabel("Epoch")
    plt.ylabel("Validation PPL")
    plt.title("Experiment 1: PPL curves")
    plt.legend()
    os.makedirs(".research/iteration1/images", exist_ok=True)
    plt.savefig(".research/iteration1/images/training_loss_dcq.pdf", bbox_inches="tight")
    plt.close()
    print("Saved figure: .research/iteration1/images/training_loss_dcq.pdf")

    return {"ppl_fp": ppl_fp_hist, "ppl_dcq": ppl_dcq_hist}

def experiment_2(device: str = "cpu") -> None:
    """LoRA / adapter friendliness experiment."""
    print("\n===== Experiment 2: LoRA Friendliness (toy) =====")
    train_ds, val_ds = get_synthetic_language_data(pattern="bursty")
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=64)

    baseline = ToyLM(1000, 128, 256, dcq=False).to(device)
    baseline = inject_lora(baseline, r=4)
    optim_base = torch.optim.AdamW(filter(lambda p: p.requires_grad, baseline.parameters()), lr=3e-3)

    dcq_model = ToyLM(1000, 128, 256, dcq=True).to(device)
    for p in dcq_model.parameters():
        p.requires_grad = False
    dcq_model = inject_lora(dcq_model, r=4)
    optim_dcq = torch.optim.AdamW(filter(lambda p: p.requires_grad, dcq_model.parameters()), lr=3e-3)

    def finetune(model, opt):
        for _ in range(2):
            run_epoch(model, train_dl, opt, device=device)
        return run_epoch(model, val_dl, device=device)

    ppl_base = finetune(baseline, optim_base)
    ppl_dcq  = finetune(dcq_model, optim_dcq)
    print(f"LoRA-fp32   PPL={ppl_base:.2f}")
    print(f"LoRA-on-DCQ PPL={ppl_dcq :.2f}")

def experiment_3(device: str = "cpu") -> None:
    """Federated fine-tuning simulation."""
    print("\n===== Experiment 3: Federated Fine-Tuning Simulation (toy) =====")
    base_model = ToyLM(1000, 128, 256, dcq=True).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    base_model = inject_lora(base_model, r=4)

    n_clients = 20
    seq_len   = 16
    uploads, ppl_hist = [], []

    global_lora_state = {n: p.detach().clone() for n, p in base_model.named_parameters() if p.requires_grad}

    for rnd in range(5):
        deltas = []
        for cid in range(n_clients):
            torch.manual_seed(cid + rnd * 100)
            x = torch.randint(0, 1000, (32, seq_len), dtype=torch.long).to(device)
            y = torch.randint(0, 1000, (32,),        dtype=torch.long).to(device)
            client_model = ToyLM(1000, 128, 256, dcq=True).to(device)
            for p in client_model.parameters():
                p.requires_grad = False
            client_model = inject_lora(client_model, r=4)
            with torch.no_grad():
                for n, p in client_model.named_parameters():
                    if n in global_lora_state:
                        p.copy_(global_lora_state[n])
            delta = simulate_client_training(client_model, x, y)
            n_bytes = sum(v.numel() * 2 for v in delta.values())
            uploads.append(n_bytes / 1e6)
            deltas.append(delta)
        for k in global_lora_state.keys():
            global_lora_state[k] = torch.stack([d[k] for d in deltas]).mean(0)
        with torch.no_grad():
            test_x = torch.randint(0, 1000, (256, seq_len), dtype=torch.long).to(device)
            test_y = torch.randint(0, 1000, (256,),        dtype=torch.long).to(device)
            eval_model = ToyLM(1000, 128, 256, dcq=True).to(device)
            for p in eval_model.parameters():
                p.requires_grad = False
            eval_model = inject_lora(eval_model, r=4)
            for n, p in eval_model.named_parameters():
                if n in global_lora_state:
                    p.copy_(global_lora_state[n])
            out = eval_model(test_x)  # (B, T, vocab)
            test_y_seq = test_y.unsqueeze(1).repeat(1, test_x.size(1))  # (B, T)
            ppl = math.exp(F.cross_entropy(out.view(-1, out.size(-1)), test_y_seq.reshape(-1)).item())
            ppl_hist.append(ppl)
            print(f"Round {rnd+1}: global PPL={ppl:.2f}, mean upload={np.mean(uploads[-n_clients:]):.2f} MB")

    plt.figure()
    plt.plot(np.cumsum(uploads), label="Cumulative upload MB")
    plt.xlabel("Client uploads (sequential)")
    plt.ylabel("MB")
    plt.title("Federated traffic accumulation (toy)")
    plt.savefig(".research/iteration1/images/bandwidth_vs_round.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(ppl_hist, marker="o")
    plt.xlabel("Federated round")
    plt.ylabel("PPL")
    plt.title("PPL vs round (toy FL)")
    plt.savefig(".research/iteration1/images/perplexity_vs_round.pdf", bbox_inches="tight")
    plt.close()

    print("Saved figures: bandwidth_vs_round.pdf, perplexity_vs_round.pdf")

def simulate_client_training(model: ToyLM, data: torch.Tensor, target: torch.Tensor,  
                             lr: float = 1e-2, steps: int = 1) -> Dict[str, torch.Tensor]:
    """Local client optimisation on LoRA weights only; returns parameter delta."""
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for _ in range(steps):
        out = model(data)  # (B, T, vocab)
        target_seq = target.unsqueeze(1).repeat(1, data.size(1))  # (B, T)
        loss = F.cross_entropy(out.view(-1, out.size(-1)), target_seq.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    delta = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            delta[n] = p.detach().cpu()
    return delta
