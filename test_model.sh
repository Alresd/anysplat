#!/bin/bash

# Activate conda environment
conda activate /mnt/pfs/users/chaojun.ni/wangweijie_mnt/wangweijie/miniconda3/envs/depthsplat

# Check if checkpoint exists, if not download model using Python script
if [ ! -f "./checkpoints/re10k.ckpt" ]; then
    echo "Checkpoint not found, downloading model..."
    python test_re10k.py
else
    # Run test with existing checkpoint
    CUDA_VISIBLE_DEVICES=6 python -m src.main +experiment=re10k \
    checkpointing.load=./checkpoints/re10k.ckpt \
    mode=test \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=6 \
    dataset.view_sampler.index_path=assets/re10k_evaluation/re10k_ctx_6v_tgt_8v_n50.json \
    test.compute_scores=true
fi 