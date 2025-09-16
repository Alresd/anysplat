#!/usr/bin/env python3
"""
Simple model download script for AnySplat
Downloads the model using huggingface_hub directly without complex imports
"""

import os
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import json

def download_anysplat_simple(output_dir="./checkpoints"):
    """Simple download using huggingface_hub"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_repo = "lhjiang/anysplat"

    print(f"Downloading AnySplat model from {model_repo}...")

    try:
        # Download the entire model directory
        local_dir = output_dir / "anysplat_hf"
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

        print(f"Model downloaded to {local_dir}")

        # Create a simple checkpoint reference
        checkpoint_info = {
            "model_path": str(local_dir),
            "repo_id": model_repo,
            "download_time": str(torch.tensor(0)),  # placeholder
        }

        info_file = output_dir / "anysplat_info.json"
        with open(info_file, "w") as f:
            json.dump(checkpoint_info, f, indent=2)

        print(f"Model info saved to {info_file}")
        print("You can now use the model for inference!")

        return str(local_dir)

    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download AnySplat model")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Directory to save the model")

    args = parser.parse_args()

    result = download_anysplat_simple(args.output_dir)
    if result:
        print(f"Successfully downloaded model to: {result}")
    else:
        print("Failed to download model")

if __name__ == "__main__":
    main()