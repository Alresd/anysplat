#!/usr/bin/env python3
"""
Test script for RE10K dataset using AnySplat
Automatically downloads model if checkpoint doesn't exist
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def download_model():
    """Download model from Hugging Face if checkpoint doesn't exist"""
    from src.model.model.anysplat import AnySplat

    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "re10k.ckpt"

    if not checkpoint_path.exists():
        print("Downloading model from Hugging Face...")
        try:
            model = AnySplat.from_pretrained("lhjiang/anysplat")
            # Save the model checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
            return str(checkpoint_path)
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Using Hugging Face model directly...")
            return None
    else:
        print(f"Using existing checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

def run_test():
    """Run the test with appropriate configuration"""
    checkpoint_path = download_model()

    # Base command
    cmd_parts = [
        "python", "-m", "src.main",
        "+experiment=re10k",
        "mode=test",
        "dataset/view_sampler=evaluation",
        "dataset.view_sampler.num_context_views=6",
        "dataset.view_sampler.index_path=assets/re10k_evaluation/re10k_ctx_6v_tgt_8v_n50.json",
        "test.compute_scores=true"
    ]

    # Add checkpoint if available
    if checkpoint_path:
        cmd_parts.append(f"checkpointing.load={checkpoint_path}")

    # Set CUDA device
    cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device

    cmd = " ".join(cmd_parts)
    print(f"Running command: {cmd}")
    print(f"Using CUDA device: {cuda_device}")

    # Execute the command
    import subprocess
    result = subprocess.run(cmd, shell=True, env=env)
    return result.returncode

if __name__ == "__main__":
    exit_code = run_test()
    sys.exit(exit_code)