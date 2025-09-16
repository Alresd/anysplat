# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AnySplat is a feed-forward 3D Gaussian Splatting system that reconstructs 3D scenes from uncalibrated multi-view images. The codebase implements a transformer-based architecture with three main components:

- **Geometry Encoder**: Processes uncalibrated input images
- **Three Decoder Heads**:
  - F_G: Predicts Gaussian parameters (μ, σ, r, s, c)
  - F_D: Predicts depth maps
  - F_C: Predicts camera poses
- **Differentiable Voxelization**: Converts pixel-wise Gaussians to voxelized 3D Gaussians

## Key Commands

### Environment Setup
```bash
# Create conda environment
conda create -y -n anysplat python=3.10
conda activate anysplat
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Training
```bash
# Single node training
python src/main.py +experiment=dl3dv trainer.num_nodes=1

# Multi-node training
export GPU_NUM=8
export NUM_NODES=2
torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$GPU_NUM \
  --rdzv_id=test \
  --rdzv_backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m src.main +experiment=multi-dataset +hydra.job.config.store_config=false
```

### Evaluation
```bash
# Novel View Synthesis evaluation
python src/eval_nvs.py --data_dir ...

# Pose Estimation evaluation
python src/eval_pose.py --co3d_dir ... --co3d_anno_dir ...
```

### Inference and Demo
```bash
# Run Gradio demo
python demo_gradio.py

# Post-optimization
python src/post_opt/simple_trainer.py default --data_dir ...
```

### Testing
```bash
# ScanNet testing
./test_scannet.sh
python test_scannet_direct.py
```

## Architecture Overview

### Core Source Structure
- `src/model/`: Core model implementations
  - `encoder/`: Transformer-based geometry encoders (AnySplat encoder, CRoCo backbone)
  - `decoder/`: Splatting decoders with CUDA implementations
  - `model/anysplat.py`: Main model class with HuggingFace integration
- `src/dataset/`: Dataset loaders for CO3D, DL3DV, ScanNet++
- `src/loss/`: Loss functions for training
- `src/geometry/`: Geometric utilities and transformations
- `src/visualization/`: Rendering and visualization tools

### Configuration System
The project uses Hydra for configuration management:
- `config/main.yaml`: Main configuration entry point
- `config/model/`: Model-specific configurations (encoder, decoder)
- `config/dataset/`: Dataset configurations
- `config/loss/`: Loss function configurations

### Key Model Components
1. **AnySplat Main Model** (`src/model/model/anysplat.py`): HuggingFace-compatible main model
2. **Encoder** (`src/model/encoder/anysplat.py`): Transformer-based geometry encoder
3. **Decoder** (`src/model/decoder/decoder_splatting_cuda.py`): CUDA-optimized splatting decoder
4. **Gaussian Adapter** (`src/model/encoder/common/gaussian_adapter.py`): Converts features to Gaussian parameters

### Training Framework
- Uses PyTorch Lightning for training orchestration
- Supports multi-GPU and multi-node distributed training
- WandB integration for experiment tracking
- Configurable view sampling strategies for different datasets

### Key Dependencies
- PyTorch 2.2.0+ with CUDA 12.1+
- PyTorch3D (included as submodule)
- gsplat for Gaussian splatting operations
- Hydra for configuration management
- WandB for experiment tracking
- HuggingFace Hub for model distribution

## Important Implementation Notes

- The model inherits from `huggingface_hub.PyTorchModelHubMixin` for easy model sharing
- CUDA kernels are used extensively for Gaussian splatting operations
- The differentiable voxelization module is a core innovation for handling arbitrary view counts
- Camera pose prediction and depth estimation are jointly optimized with Gaussian parameters
- The codebase supports three main datasets: CO3Dv2, DL3DV, and ScanNet++, each with different view sampling strategies