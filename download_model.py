#!/usr/bin/env python3
"""
Download AnySplat model from Hugging Face
This script downloads the pretrained model and saves it as a checkpoint file
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def download_anysplat_model(checkpoint_path: str = "./checkpoints/anysplat.ckpt"):
    """Download AnySplat model from Hugging Face and save as checkpoint"""
    from src.model.model.anysplat import AnySplat

    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        print(f"Checkpoint already exists at {checkpoint_path}")
        return str(checkpoint_path)

    print("Downloading AnySplat model from Hugging Face...")
    try:
        # Download model from HuggingFace
        model = AnySplat.from_pretrained("lhjiang/anysplat")

        # Save the model checkpoint in a format compatible with the training code
        state_dict = model.state_dict()

        # Create a checkpoint dict that matches the expected format
        checkpoint = {
            'state_dict': state_dict,
            'model_name': 'anysplat',
            'source': 'huggingface:lhjiang/anysplat'
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")

        return str(checkpoint_path)

    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def download_pretrained_model():
    """Download the pretrained model that can be used for testing"""
    print("Downloading pretrained model for testing...")

    # Try to download the model with different checkpoint names
    checkpoint_options = [
        "./checkpoints/anysplat.ckpt",
        "./checkpoints/re10k.ckpt",
        "./checkpoints/re10k-256x448-unet_volsplat/checkpoints/epoch_2148-step_58000.ckpt"
    ]

    for checkpoint_path in checkpoint_options:
        result = download_anysplat_model(checkpoint_path)
        if result:
            print(f"Successfully downloaded model to: {result}")
            return result

    print("Failed to download model")
    return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download AnySplat model")
    parser.add_argument("--checkpoint_path", type=str,
                       default="./checkpoints/anysplat.ckpt",
                       help="Path to save the checkpoint")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # No arguments provided, download with default options
        download_pretrained_model()
    else:
        download_anysplat_model(args.checkpoint_path)