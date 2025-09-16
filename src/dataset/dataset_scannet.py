import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Union
import os
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler, ViewSamplerEvaluation
from ..evaluation.evaluation_index_generator import IndexEntry
from ..geometry.projection import get_fov


@dataclass
class DatasetScannetCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    load_depth: bool = True
    near: float = 0.5
    far: float = 15.0


@dataclass
class DatasetScannetCfgWrapper:
    scannet: DatasetScannetCfg


class DatasetScannet(Dataset):
    cfg: DatasetScannetCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]

    def __init__(
        self,
        cfg: DatasetScannetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Collect chunks.
        self.chunks = []

        print('-'*20 + f'data root: {cfg.roots[0]}')
        if self.data_stage != 'test':
            for root in cfg.roots:
                root = root / self.data_stage
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.is_dir()]
                )
                self.chunks.extend(root_chunks)
        else:
            # For test stage, use index to get specific scenes
            root = cfg.roots[0] / self.data_stage
            self.chunks = sorted(
                [root / path for path in self.index]
            )

        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __getitem__(self, idx):
        # Handle tuple input for test stage
        if isinstance(idx, tuple):
            idx, _, _ = idx

        path = self.chunks[idx]
        scene = str(path).split('/')[-1]
        if not os.path.exists(path):
            path = Path(str(path)[:-2])

        # Load data following BEV-Splat approach
        imshape = self.to_tensor(Image.open(os.path.join(path, 'color', '0.jpg'))).shape
        extrinsics = torch.from_numpy(np.load(os.path.join(path, 'extrinsics.npy'))).float()
        intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_color.txt'))\
                                    [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()

        # Sample views using view_sampler (following BEV-Splat approach)
        context_index, target_indices = self.view_sampler.sample(
            scene,
            extrinsics,
            intrinsics,
            path=path,
        )

        # Normalize intrinsics by image dimensions
        intrinsics_norm = intrinsics.clone()
        intrinsics_norm[:, 0] /= imshape[2]  # width
        intrinsics_norm[:, 1] /= imshape[1]  # height

        example = {'scene': scene}

        # Load context images and depths
        context_images = []
        context_depths = []
        for idx_val in context_indices:
            idx_int = int(idx_val.item())
            img_path = os.path.join(path, 'color', f'{idx_int}.jpg')
            if not os.path.exists(img_path):
                # Try with zero padding
                img_path = os.path.join(path, 'color', f'{idx_int:06d}.jpg')

            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = self.to_tensor(img.resize((640, 480)))
                context_images.append(img[None])
            else:
                # Create dummy image
                context_images.append(torch.zeros(1, 3, 480, 640))

            if self.cfg.load_depth:
                depth_path = os.path.join(path, 'depth', f'{idx_int}.png')
                if not os.path.exists(depth_path):
                    depth_path = os.path.join(path, 'depth', f'{idx_int:06d}.png')

                if os.path.exists(depth_path):
                    depth_img = Image.open(depth_path)
                    depth_img = (np.asarray(depth_img.resize((640, 480))) / 1000.0).astype(np.float32)
                    depth_img = self.to_tensor(depth_img)
                    context_depths.append(depth_img[None])
                else:
                    context_depths.append(torch.zeros(1, 1, 480, 640))

        context_images = torch.cat(context_images)
        if self.cfg.load_depth:
            context_depths = torch.cat(context_depths)

        # Prepare context data
        context_content = {
            "extrinsics": extrinsics[context_indices],
            "intrinsics": intrinsics_norm[context_indices],
            "image": context_images,
            "near": self.get_bound("near", len(context_indices)),
            "far": self.get_bound("far", len(context_indices)),
            "index": context_indices,
        }

        if self.cfg.load_depth:
            context_content['depth'] = context_depths

        example['context'] = context_content

        # Load target images and depths
        target_images = []
        target_depths = []
        for idx_val in target_indices:
            idx_int = int(idx_val.item())
            img_path = os.path.join(path, 'color', f'{idx_int}.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(path, 'color', f'{idx_int:06d}.jpg')

            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = self.to_tensor(img.resize((640, 480)))
                target_images.append(img[None])
            else:
                target_images.append(torch.zeros(1, 3, 480, 640))

            if self.cfg.load_depth:
                depth_path = os.path.join(path, 'depth', f'{idx_int}.png')
                if not os.path.exists(depth_path):
                    depth_path = os.path.join(path, 'depth', f'{idx_int:06d}.png')

                if os.path.exists(depth_path):
                    depth_img = Image.open(depth_path)
                    depth_img = (np.asarray(depth_img.resize((640, 480))) / 1000.0).astype(np.float32)
                    depth_img = self.to_tensor(depth_img)
                    target_depths.append(depth_img[None])
                else:
                    target_depths.append(torch.zeros(1, 1, 480, 640))

        target_images = torch.cat(target_images)
        if self.cfg.load_depth:
            target_depths = torch.cat(target_depths)

        # Prepare target data
        target_content = {
            "extrinsics": extrinsics[target_indices],
            "intrinsics": intrinsics_norm[target_indices],
            "image": target_images,
            "near": self.get_bound("near", len(target_indices)),
            "far": self.get_bound("far", len(target_indices)),
            "index": target_indices,
        }

        if self.cfg.load_depth:
            target_content['depth'] = target_depths

        example["target"] = target_content

        # Apply augmentation and normalization
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        example = apply_crop_shim(example, tuple(self.cfg.input_image_shape))

        return example

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self.cfg, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Union[Path, IndexEntry]]:
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")

        merged_index = {}
        for data_stage in data_stages:
            for root in self.cfg.roots:
                index_file = root / f'{data_stage}_idx.txt'
                if index_file.exists():
                    with open(index_file, 'r') as f:
                        index = f.read().strip().split('\n')
                    try:
                        index.remove('')
                    except:
                        pass
                    index = {x: Path(root / data_stage / x) for x in index if x}
                else:
                    # Fallback: use all directories in the data_stage folder
                    stage_dir = root / data_stage
                    if stage_dir.exists():
                        index = {d.name: d for d in stage_dir.iterdir() if d.is_dir()}
                    else:
                        index = {}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}

        if isinstance(self.view_sampler, ViewSamplerEvaluation):
            # Filter by evaluation index if available
            if hasattr(self.view_sampler, 'index') and self.view_sampler.index:
                eval_scenes = {k.split('_')[0] if '_' in k else k for k in self.view_sampler.index.keys()}
                merged_index = {k: v for k, v in merged_index.items() if k in eval_scenes}

        return merged_index

    def __len__(self) -> int:
        return len(self.index.keys())