#!/usr/bin/env python3
"""
Fix ScanNet test by converting HuggingFace model to proper checkpoint format
"""

import os
import sys
import torch
from pathlib import Path
import json

def convert_hf_model():
    """Convert HuggingFace model to AnySplat checkpoint format"""

    hf_dir = Path("./checkpoints/anysplat_hf")
    output_path = Path("./checkpoints/anysplat.ckpt")

    if not hf_dir.exists():
        print(f"❌ HuggingFace model directory not found: {hf_dir}")
        print("Please run: python download_model_simple.py")
        return False

    print(f"📁 Found HuggingFace model directory: {hf_dir}")

    # Look for model files
    safetensors_file = hf_dir / "model.safetensors"
    pytorch_file = hf_dir / "pytorch_model.bin"
    config_file = hf_dir / "config.json"

    # Try different ways to load the model
    state_dict = None

    if pytorch_file.exists():
        print(f"📦 Loading from pytorch_model.bin...")
        try:
            state_dict = torch.load(pytorch_file, map_location='cpu')
            print("✅ Successfully loaded pytorch_model.bin")
        except Exception as e:
            print(f"❌ Error loading pytorch_model.bin: {e}")

    if state_dict is None and safetensors_file.exists():
        print(f"📦 Trying to load from model.safetensors...")
        try:
            # Try using safetensors
            from safetensors import safe_open
            state_dict = {}
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            print("✅ Successfully loaded model.safetensors")
        except ImportError:
            print("⚠️  safetensors not installed. Install with: pip install safetensors")
            # Try alternative method using transformers
            try:
                print("📦 Trying alternative loading method...")
                # Add the parent directory to sys.path temporarily
                sys.path.insert(0, str(Path(__file__).parent))
                from src.model.model.anysplat import AnySplat
                model = AnySplat.from_pretrained("lhjiang/anysplat")
                state_dict = model.state_dict()
                print("✅ Successfully loaded via AnySplat.from_pretrained")
                # Remove the path we added
                sys.path.pop(0)
            except Exception as e:
                print(f"❌ Alternative loading failed: {e}")
        except Exception as e:
            print(f"❌ Error loading safetensors: {e}")

    if state_dict is None:
        print("❌ Could not load model from any format")
        return False

    # Load config
    config = {}
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print("✅ Loaded config.json")
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")

    # Create Lightning-compatible checkpoint
    checkpoint = {
        'state_dict': state_dict,
        'epoch': 0,
        'global_step': 0,
        'pytorch-lightning_version': '2.0.0',
        'hyper_parameters': config,
        'model_name': 'anysplat',
        'source': 'huggingface:lhjiang/anysplat',
        'optimizer_states': [],
        'lr_schedulers': []
    }

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    print(f"💾 Saving checkpoint to: {output_path}")
    try:
        torch.save(checkpoint, output_path)
        print("✅ Checkpoint saved successfully!")

        # Verify the checkpoint
        loaded = torch.load(output_path, map_location='cpu')
        print(f"📊 Checkpoint contains {len(loaded['state_dict'])} parameters")
        return True

    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        return False

def main():
    print("🔧 Converting HuggingFace model to AnySplat checkpoint format...")
    success = convert_hf_model()

    if success:
        print("\n✅ SUCCESS! Model conversion completed.")
        print("🚀 You can now run: ./test_scannet.sh")
    else:
        print("\n❌ FAILED! Model conversion failed.")
        print("🔍 Troubleshooting steps:")
        print("1. Make sure the model was downloaded: python download_model_simple.py")
        print("2. Install safetensors: pip install safetensors")
        print("3. Check if jaxtyping and beartype are installed: pip install jaxtyping beartype")

if __name__ == "__main__":
    main()