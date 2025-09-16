#!/usr/bin/env python3
"""
Simple test script for AnySplat using HuggingFace model
This bypasses the complex training framework and directly tests model inference
"""
import torch
import numpy as np
import sys
import os
sys.path.append('.')

from src.model.model.anysplat import AnySplat
from src.utils.image import process_image

def test_anysplat():
    print("Loading AnySplat model from HuggingFace...")

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    model = model.to(device)
    model.eval()

    # Disable gradients for inference
    for param in model.parameters():
        param.requires_grad = False

    print(f"Model loaded successfully on {device}")

    # Create dummy input images (3 views, 448x448, RGB)
    # In a real scenario, you would load actual images using process_image()
    dummy_images = torch.randn(1, 3, 3, 448, 448).to(device)  # [batch, views, channels, height, width]

    print("Running inference...")

    try:
        # Run inference
        with torch.no_grad():
            gaussians, pred_context_pose = model.inference((dummy_images + 1) * 0.5)

        print("✓ Inference successful!")
        print(f"Gaussians shape: {gaussians.shape if hasattr(gaussians, 'shape') else type(gaussians)}")
        print(f"Pose prediction keys: {pred_context_pose.keys()}")
        print(f"Extrinsic shape: {pred_context_pose['extrinsic'].shape}")
        print(f"Intrinsic shape: {pred_context_pose['intrinsic'].shape}")

        return True

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_anysplat()
    if success:
        print("\n🎉 AnySplat model test completed successfully!")
    else:
        print("\n❌ AnySplat model test failed!")
        sys.exit(1)