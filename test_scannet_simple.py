#!/usr/bin/env python3
"""
Simple ScanNet test script with automatic model download
This script directly uses the AnySplat model from HuggingFace without complex configuration
"""

import os
import sys
import torch
from pathlib import Path
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_scannet_simple(data_dir: str, output_dir: str = "outputs/scannet_simple"):
    """Simple test function using AnySplat directly"""
    from src.model.model.anysplat import AnySplat
    from src.utils.image import process_image
    import numpy as np
    from PIL import Image
    import torchvision.transforms as tf

    # Load model
    print("Loading AnySplat model from Hugging Face...")
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"Model loaded on {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load sample images from ScanNet data
    scene_dir = Path(data_dir)
    if not scene_dir.exists():
        print(f"Data directory {data_dir} does not exist!")
        return

    # Look for the first scene directory
    scene_paths = [p for p in scene_dir.iterdir() if p.is_dir()]
    if not scene_paths:
        print(f"No scene directories found in {data_dir}")
        return

    test_scene = scene_paths[0]
    color_dir = test_scene / "color"

    if not color_dir.exists():
        print(f"No color directory found in {test_scene}")
        return

    # Get image files
    image_files = sorted([f for f in color_dir.glob("*.jpg")])[:10]  # Take first 10 images
    if len(image_files) < 3:
        print(f"Not enough images found in {color_dir}")
        return

    print(f"Found {len(image_files)} images in {test_scene}")

    # Load and process images
    to_tensor = tf.ToTensor()
    images = []
    for img_path in image_files[:6]:  # Use first 6 as context
        try:
            img = Image.open(img_path)
            img = img.resize((448, 224))  # Resize to model input size
            img_tensor = to_tensor(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    if len(images) < 3:
        print("Not enough valid images loaded")
        return

    # Stack images and prepare for model
    images = torch.stack(images[:6], dim=0).unsqueeze(0).to(device)  # [1, N, 3, H, W]
    images = (images + 1) * 0.5  # Normalize to [0, 1]

    print(f"Input images shape: {images.shape}")

    try:
        # Run inference
        print("Running inference...")
        with torch.no_grad():
            gaussians, pred_context_pose = model.inference(images)

        print("Inference completed successfully!")
        print(f"Generated {gaussians.shape[1] if len(gaussians.shape) > 1 else 'N/A'} Gaussians")

        # Save some basic information
        result_file = Path(output_dir) / "test_result.txt"
        with open(result_file, "w") as f:
            f.write(f"ScanNet Test Results\n")
            f.write(f"Scene: {test_scene.name}\n")
            f.write(f"Input images: {len(images[0])}\n")
            f.write(f"Gaussians shape: {gaussians.shape}\n")
            f.write(f"Device: {device}\n")

        print(f"Results saved to {result_file}")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Simple ScanNet test with AnySplat")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to ScanNet test data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/scannet_simple",
                       help="Output directory for results")

    args = parser.parse_args()

    test_scannet_simple(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()