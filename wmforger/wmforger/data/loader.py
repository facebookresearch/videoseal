# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import ToTensor
from torchvision.datasets.folder import is_image_file

from ..utils.dist import is_dist_avail_and_initialized


def get_image_paths(path):
    cache_dir = '.cache'
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = path.replace('/', '_') + '.json'
    cache_file = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            paths = json.load(f)
    else:
        paths = []
        for root, _, files in os.walk(path):
            for filename in files:
                if is_image_file(filename):
                    paths.append(os.path.join(root, filename))
        paths = sorted(paths)
        with open(cache_file, 'w') as f:
            json.dump(paths, f)
    return paths


class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path, transform=None, mask_transform=None):
        # assuming 'path' is a folder of image files path and
        # 'annotation_path' is the base path for corresponding annotation json files
        self.samples = get_image_paths(path)
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = ToTensor()(img)

        if self.transform:
            img = self.transform(img)

        # Get MASKS
        mask = torch.ones_like(img[0:1, ...])

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        
        return img, mask

    def __len__(self):
        return len(self.samples)


def custom_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])

    images, masks = zip(*batch)
    images = torch.stack(images)

    # Find the maximum number of masks in any single image
    max_masks = max(mask.shape[0] for mask in masks)
    if max_masks == 1:
        masks = torch.stack(masks)
        return images, masks

    # Pad each mask tensor to have 'max_masks' masks and add the inverse mask
    padded_masks = []
    for mask in masks:
        # Calculate the union of all masks in this image
        # Assuming mask is of shape [num_masks, H, W]
        union_mask = torch.max(mask, dim=0).values

        # Calculate the inverse of the union mask
        inverse_mask = ~union_mask

        # Pad the mask tensor to have 'max_masks' masks
        pad_size = max_masks - mask.shape[0]
        if pad_size > 0:
            padded_mask = F.pad(mask, pad=(
                0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
        else:
            padded_mask = mask

        # Append the inverse mask to the padded mask tensor
        # padded_mask = torch.cat([padded_mask, inverse_mask.unsqueeze(0)], dim=0)

        padded_masks.append(padded_mask)

    # Stack the padded masks
    masks = torch.stack(padded_masks)

    return images, masks


def get_dataloader_segmentation(
    data_dir: str,
    ann_file: str,
    transform: callable,
    mask_transform: callable,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8,
) -> DataLoader:
    dataset = ImageFolder(path=data_dir, transform=transform, mask_transform=mask_transform)

    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)

    return dataloader
