#!/bin/bash

# Test script for ScanNet dataset using AnySplat
# This script follows the pattern provided by the user

# Check if model exists, if not create a dummy one for testing
if [ ! -f "./checkpoints/anysplat.ckpt" ] && [ ! -f "./checkpoints/re10k.ckpt" ] && [ ! -d "./checkpoints/anysplat_hf" ]; then
    echo "No pretrained model found, creating dummy checkpoint for config testing..."
    mkdir -p checkpoints
    touch checkpoints/anysplat.ckpt
fi

# Use the downloaded model if the specified checkpoint doesn't exist
CHECKPOINT_PATH="checkpoints/re10k-256x448-unet_volsplat/checkpoints/epoch_2148-step_58000.ckpt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    if [ -f "./checkpoints/anysplat.ckpt" ]; then
        CHECKPOINT_PATH="checkpoints/anysplat.ckpt"
    elif [ -f "./checkpoints/re10k.ckpt" ]; then
        CHECKPOINT_PATH="checkpoints/re10k.ckpt"
    elif [ -d "./checkpoints/anysplat_hf" ]; then
        # Convert HuggingFace model to checkpoint format
        echo "Found HuggingFace model directory. Converting to checkpoint format..."
        python convert_hf_to_checkpoint.py
        if [ -f "./checkpoints/anysplat.ckpt" ]; then
            CHECKPOINT_PATH="checkpoints/anysplat.ckpt"
            echo "Successfully converted HuggingFace model to checkpoint format"
        else
            echo "Failed to convert HuggingFace model"
            exit 1
        fi
    else
        echo "No checkpoint found! Please download the model first."
        exit 1
    fi
fi

echo "Using checkpoint: $CHECKPOINT_PATH"

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=scannet \
data_loader.train.batch_size=1 \
'dataset.scannet.roots'='["/tmp/scannet"]' \
dataset.scannet.view_sampler.num_context_views=6 \
dataset.scannet.view_sampler.num_target_views=2 \
trainer.max_steps=150000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
mode=test \
test.save_video=false \
test.save_depth_concat_img=false \
test.save_image=false \
test.save_gt_image=false \
test.save_input_images=false \
test.save_video=false \
test.save_gaussian=false \
checkpointing.load=$CHECKPOINT_PATH \
test.output_path=outputs/scannet-256x256 \
hydra.run.dir=outputs/scannet-256x256