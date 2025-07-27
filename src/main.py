"""
Main experiment script for DCQ (Differentiable Codebook Quantisation).
Orchestrates the entire experiment pipeline from data preprocessing to evaluation.
"""

import os
import random
import json
import torch
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def test():
    """Quick functionality test – executed by CI."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running quick test on device: {device}")
    
    from evaluate import experiment_1, experiment_2, experiment_3
    
    exp1 = experiment_1(device)
    experiment_2(device)
    experiment_3(device)
    
    assert os.path.exists(".research/iteration1/images/training_loss_dcq.pdf"), "Plot missing!"
    assert len(exp1["ppl_fp"]) == 3 and len(exp1["ppl_dcq"]) == 3
    print("All tests passed ✔")

def main():
    """Main experiment execution."""
    print("=" * 60)
    print("DCQ (Differentiable Codebook Quantisation) Experiment")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    from evaluate import experiment_1, experiment_2, experiment_3
    
    print("\nStarting DCQ experiments...")
    
    exp1_results = experiment_1(device)
    experiment_2(device)
    experiment_3(device)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Device used: {device}")
    print(f"Final baseline PPL: {exp1_results['ppl_fp'][-1]:.2f}")
    print(f"Final DCQ PPL: {exp1_results['ppl_dcq'][-1]:.2f}")
    print(f"PPL difference: {exp1_results['ppl_dcq'][-1] - exp1_results['ppl_fp'][-1]:.2f}")
    
    image_dir = ".research/iteration1/images"
    pdf_files = [f for f in os.listdir(image_dir) if f.endswith('.pdf')]
    print(f"\nGenerated {len(pdf_files)} PDF plots:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    research_history = {
        "status_enum": "stopped",
        "experiment_completed": True,
        "device_used": device,
        "final_baseline_ppl": exp1_results['ppl_fp'][-1],
        "final_dcq_ppl": exp1_results['ppl_dcq'][-1],
        "pdf_files_generated": pdf_files,
        "timestamp": "2025-07-27"
    }
    
    with open(".research/research_history.json", "w") as f:
        json.dump(research_history, f, indent=2)
    
    print(f"\nExperiment completed successfully!")
    print(f"Status set to: stopped")
    print(f"Results saved in: {image_dir}")

if __name__ == "__main__":
    test()
    main()
