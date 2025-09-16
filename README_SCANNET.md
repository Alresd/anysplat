# ScanNet 测试指南

## 问题解决

您遇到的 `jaxtyping` 模块缺失错误是因为环境中缺少一些依赖。以下是解决方案：

## 方案1：安装缺失依赖（推荐）

```bash
# 安装缺失的依赖
pip install jaxtyping beartype

# 然后运行下载脚本
python download_model.py

# 或者直接运行ScanNet测试
./test_scannet.sh
```

## 方案2：使用简化的下载脚本

```bash
# 使用不依赖复杂导入的简化下载脚本
python download_model_simple.py

# 这会下载模型到 ./checkpoints/anysplat_hf/ 目录
```

## 方案3：手动下载模型

如果自动下载有问题，您可以：

1. 访问 https://huggingface.co/lhjiang/anysplat
2. 手动下载模型文件
3. 放置到 `./checkpoints/` 目录下

## 方案4：使用现有评估脚本

AnySplat已经有内置的评估脚本可以自动下载模型：

```bash
# 使用内置的评估脚本（会自动下载模型）
python src/eval_nvs.py --data_dir /path/to/your/scannet/data --output_path outputs/scannet_nvs
```

## ScanNet数据格式

确保您的ScanNet数据按以下结构组织：

```
/tmp/scannet/
├── test/
│   ├── scene0000_00/
│   │   ├── color/
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   ├── depth/（可选）
│   │   │   ├── 0.png
│   │   │   └── ...
│   │   ├── extrinsics.npy
│   │   └── intrinsic/
│   │       └── intrinsic_color.txt
│   └── ...
└── test_idx.txt
```

## 快速测试命令

一旦依赖解决，您可以使用：

```bash
# 原始的测试命令（按您要求的格式）
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=scannet \
data_loader.train.batch_size=1 \
'dataset.roots'='["/tmp/scannet"]' \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/scannet_index.json \
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
test.save_gaussian=false \
checkpointing.pretrained_model=checkpoints/anysplat.ckpt \
output_dir=outputs/scannet-256x256
```

## 解决依赖问题的完整步骤

```bash
# 1. 确保在正确的conda环境中
conda activate anysplat

# 2. 安装缺失的依赖
pip install jaxtyping beartype

# 3. 下载模型
python download_model.py

# 4. 运行ScanNet测试
./test_scannet.sh
```

这样应该能解决您的模型下载和测试问题！