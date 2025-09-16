#!/bin/bash

# Training script for ScanNet dataset using AnySplat

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.main +experiment=scannet \
data_loader.train.batch_size=1 \
'dataset.roots'='["/tmp/scannet"]' \
trainer.max_steps=150000 \
trainer.val_check_interval=0.9 \
train.eval_model_every_n_val=40 \
checkpointing.every_n_train_steps=2000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
output_dir=checkpoints/scannet-256x256-anysplat \
wandb.project=AnySplat_ScanNet