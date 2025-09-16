#!/usr/bin/env python3
"""
Fix torch_scatter dependency by installing or providing fallback
"""

import sys
import subprocess

def install_torch_scatter():
    """Try to install torch_scatter"""
    print("Attempting to install torch_scatter...")

    # Get PyTorch version
    import torch
    torch_version = torch.__version__
    cuda_version = torch.version.cuda

    print(f"PyTorch version: {torch_version}")
    print(f"CUDA version: {cuda_version}")

    # Try different installation methods
    install_commands = [
        # Method 1: pip install with index
        "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html",
        # Method 2: conda install
        "conda install pytorch-scatter -c pyg",
        # Method 3: pip install without constraints
        "pip install torch-scatter",
    ]

    for cmd in install_commands:
        print(f"Trying: {cmd}")
        try:
            subprocess.check_call(cmd.split())
            print("✅ Successfully installed torch_scatter!")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Failed: {e}")
            continue

    print("❌ Could not install torch_scatter")
    return False

def test_torch_scatter():
    """Test if torch_scatter works"""
    try:
        from torch_scatter import scatter_add, scatter_max
        import torch

        # Simple test
        src = torch.randn(6, 3)
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        result = scatter_add(src, index, dim=0)
        print(f"✅ torch_scatter works! Result shape: {result.shape}")
        return True
    except ImportError as e:
        print(f"❌ torch_scatter import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ torch_scatter test failed: {e}")
        return False

def check_environment():
    """Check current environment"""
    import torch
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

def main():
    print("🔧 Fixing torch_scatter dependency...")

    check_environment()

    # First try to test if it already works
    if test_torch_scatter():
        print("✅ torch_scatter is already working!")
        return

    # Try to install
    if install_torch_scatter():
        if test_torch_scatter():
            print("✅ torch_scatter installation successful!")
        else:
            print("❌ torch_scatter installed but not working properly")

    print("\n📋 If torch_scatter still doesn't work, you have these options:")
    print("1. Try manual installation:")
    print("   pip install torch-scatter")
    print("2. Use our native implementation (already configured)")
    print("3. The code will automatically fall back to PyTorch native ops")
    print("\n🚀 You can now try running: ./test_scannet.sh")

if __name__ == "__main__":
    main()