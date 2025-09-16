#!/usr/bin/env python3
"""
Convert HuggingFace model to checkpoint format compatible with AnySplat training code
"""

import os
import sys
import torch
from pathlib import Path
import json

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

def convert_hf_to_checkpoint(hf_model_dir="./checkpoints/anysplat_hf",
                           output_path="./checkpoints/anysplat.ckpt"):
    """Convert HuggingFace model to checkpoint format"""

    hf_dir = Path(hf_model_dir)
    if not hf_dir.exists():
        print(f"HuggingFace model directory not found: {hf_dir}")
        return False

    # Look for model files
    safetensors_file = hf_dir / "model.safetensors"
    pytorch_file = hf_dir / "pytorch_model.bin"
    config_file = hf_dir / "config.json"

    if safetensors_file.exists() and HAS_SAFETENSORS:
        print(f"Loading from safetensors: {safetensors_file}")
        state_dict = {}
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    elif pytorch_file.exists():
        print(f"Loading from pytorch: {pytorch_file}")
        state_dict = torch.load(pytorch_file, map_location='cpu')
    elif safetensors_file.exists() and not HAS_SAFETENSORS:
        print("Found safetensors file but safetensors library not installed")
        print("Install with: pip install safetensors")
        return False
    else:
        print("No model file found (model.safetensors or pytorch_model.bin)")
        return False

    # Load config if available
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    # Create checkpoint in the format expected by AnySplat training code
    checkpoint = {
        'state_dict': state_dict,
        'epoch': 0,
        'global_step': 0,
        'pytorch-lightning_version': '2.0.0',
        'state_dict_keys': list(state_dict.keys()),
        'hyper_parameters': config,
        'model_name': 'anysplat',
        'source': 'huggingface:lhjiang/anysplat'
    }

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    print(f"Saving checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)

    # Verify the checkpoint
    try:
        loaded = torch.load(output_path, map_location='cpu')
        print(f"Checkpoint saved successfully!")
        print(f"Keys in checkpoint: {list(loaded.keys())}")
        print(f"Model parameters: {len(loaded['state_dict'])}")
        return True
    except Exception as e:
        print(f"Error verifying checkpoint: {e}")
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert HuggingFace model to checkpoint")
    parser.add_argument("--hf_model_dir", type=str, default="./checkpoints/anysplat_hf",
                       help="Path to HuggingFace model directory")
    parser.add_argument("--output_path", type=str, default="./checkpoints/anysplat.ckpt",
                       help="Path for output checkpoint")

    args = parser.parse_args()

    success = convert_hf_to_checkpoint(args.hf_model_dir, args.output_path)
    if success:
        print(f"Conversion successful! You can now use: {args.output_path}")
    else:
        print("Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()