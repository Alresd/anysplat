#!/bin/bash

# Test script for ScanNet dataset using AnySplat
# This script follows the pattern provided by the user

echo "Using HuggingFace model: lhjiang/anysplat"

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=scannet \
data_loader.train.batch_size=1 \
'dataset.scannet.roots'='["/tmp/scannet"]' \
dataset.scannet.view_sampler.num_context_views=6 \
dataset.scannet.view_sampler.num_target_views=2 \
trainer.max_steps=150000 \
mode=test \
test.save_video=false \
test.save_image=false \
wandb.mode=disabled \
model.encoder.pretrained_weights="lhjiang/anysplat" \
test.output_path=outputs/scannet-256x256 \
hydra.run.dir=outputs/scannet-256x256