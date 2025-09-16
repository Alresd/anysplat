# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AnySplat: Feed-forward 3D Gaussian Splatting

This is a research codebase for AnySplat, a transformer-based architecture for feed-forward 3D Gaussian Splatting from unconstrained views. The system predicts Gaussian parameters, depth maps, and camera poses from uncalibrated input images.

## Key Commands

### Environment Setup
```bash
conda create -y -n anysplat python=3.10
conda activate anysplat
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Training
```bash
# Single node training
python src/main.py +experiment=dl3dv trainer.num_nodes=1

# ScanNet training
./train_scannet.sh

# ScanNet manual training
python src/main.py +experiment=scannet trainer.num_nodes=1 'dataset.roots'='["/tmp/scannet"]'

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

# Pose estimation evaluation
python src/eval_pose.py --co3d_dir ... --co3d_anno_dir ...
```

### Demo
```bash
# Launch Gradio demo interface
python demo_gradio.py

# Inference script
python inference.py
```

### Testing
```bash
# Quick test script with automatic model download
./test_model.sh

# ScanNet testing
./test_scannet.sh

# Manual test command (modify paths as needed)
CUDA_VISIBLE_DEVICES=6 python -m src.main +experiment=re10k \
checkpointing.load=./checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/re10k_evaluation/re10k_ctx_6v_tgt_8v_n50.json \
test.compute_scores=true

# ScanNet manual test command
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=scannet \
data_loader.train.batch_size=1 \
'dataset.roots'='["/tmp/scannet"]' \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/scannet_index.json \
mode=test \
test.compute_scores=true \
checkpointing.pretrained_model=checkpoints/re10k.ckpt \
output_dir=outputs/scannet-256x256

# Download model programmatically
python test_re10k.py
```

### Code Quality
```bash
# Linting with ruff (configured in requirements.txt)
ruff check src/

# Code formatting with black
black src/
```

## Architecture Overview

### Core Components

1. **Model Architecture** (`src/model/`):
   - **Encoder** (`src/model/encoder/`): Transformer-based geometry encoder with multiple backbone options (CroCo, DINO, ResNet)
   - **Decoder** (`src/model/decoder/`): CUDA-based splatting decoder for rendering
   - **Heads** (`src/model/encoder/heads/`): Three decoder heads (F_G, F_D, F_C) for Gaussian parameters, depth, and camera poses

2. **Dataset System** (`src/dataset/`):
   - Supports multiple datasets: CO3Dv2, DL3DV, ScanNet++, ScanNet
   - View samplers for different training strategies: all, arbitrary, bounded, evaluation, rank
   - Data shims for augmentation, cropping, geometry processing, loading, normalization, patching

3. **Loss Functions** (`src/loss/`):
   - Multiple loss types: MSE, SSIM, LPIPS, depth consistency, normal consistency, Huber, opacity
   - Configurable loss combinations via YAML configuration

4. **Configuration System**:
   - Hydra-based configuration with hierarchical YAML files in `config/`
   - Experiment configs (`config/experiment/`): dl3dv, co3d, scannetpp, scannet, re10k, multi-dataset
   - Model configs (`config/model/`): encoder and decoder configurations
   - Dataset configs (`config/dataset/`): dataset-specific and view sampler configurations

### Key Dependencies

- PyTorch 2.2.0 with CUDA 12.1
- Lightning for training framework
- Hydra for configuration management
- gsplat for 3D Gaussian splatting operations
- xformers for efficient transformer operations
- PyTorch3D for 3D geometry operations

### Data Flow

1. Input: Set of uncalibrated images
2. Encoder: Transformer processes images to extract features
3. Heads: Three decoder heads predict:
   - Gaussian parameters (μ, σ, r, s, c)
   - Depth maps (D)
   - Camera poses (p)
4. Voxelization: Differentiable voxelization converts pixel-wise to pre-voxel 3D Gaussians
5. Rendering: Multi-view images and depth maps rendered from voxelized Gaussians
6. Supervision: RGB loss and geometry losses using pseudo-geometry priors from VGGT

### Training Configuration

- Uses wandb for experiment tracking
- Supports multi-node distributed training
- Checkpointing every 5000 steps
- Gradient clipping at 0.5
- Learning rate: 1.5e-4 with 2000 warmup steps
- Backbone learning rate multiplier: 0.1

### File Organization

- `src/main.py`: Main training entry point with Hydra configuration
- `inference.py`: Standalone inference script
- `demo_gradio.py`: Gradio web interface
- `test_model.sh`: Quick test script with automatic model download
- `test_scannet.sh`: ScanNet testing script
- `train_scannet.sh`: ScanNet training script
- `test_re10k.py`: Model download and checkpoint management script
- `src/eval_nvs.py`, `src/eval_pose.py`: Evaluation scripts for novel view synthesis and pose estimation
- `src/dataset/dataset_scannet.py`: ScanNet dataset implementation
- `src/misc/`: Utility functions for image I/O, camera utils, benchmarking, logging
- `src/geometry/`: 3D geometry operations and camera embeddings
- `src/post_opt/`: Post-optimization scripts for refining results
- `assets/scannet_index.json`: ScanNet evaluation index configuration
- `examples/`: Example input data and usage demonstrations

## Development Notes

- The codebase uses beartype and jaxtyping for runtime type checking
- Configuration is managed through Hydra with hierarchical YAML files in `config/`
- The model can be loaded from Hugging Face Hub (`lhjiang/anysplat`)
- Supports both training from scratch and loading pretrained checkpoints
- Uses Lightning for distributed training and checkpointing
- Code quality tools: ruff for linting, black for formatting
- Custom PyTorch3D and gsplat installations from specific sources (see requirements.txt)

### Quick Development Workflow

1. Environment setup: `conda create -y -n anysplat python=3.10 && conda activate anysplat`
2. Install dependencies: `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt`
3. Quick test: `./test_model.sh`
4. Code quality: `ruff check src/ && black src/`
5. Train: `python src/main.py +experiment=dl3dv trainer.num_nodes=1`