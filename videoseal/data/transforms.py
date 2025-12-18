# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torchvision import transforms

default_transform = transforms.Compose([
    transforms.ToTensor(),
])


class RGB2YUV(nn.Module):
    def __init__(self):
        super(RGB2YUV, self).__init__()
        self.register_buffer('M', torch.tensor([[0.299, 0.587, 0.114],
                                                [-0.14713, -0.28886, 0.436],
                                                [0.615, -0.51499, -0.10001]], dtype=torch.float32)
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # b h w c
        yuv = torch.matmul(x, self.M.T)
        yuv = yuv.permute(0, 3, 1, 2).contiguous()
        return yuv


class YUV2RGB(nn.Module):
    def __init__(self):
        super(YUV2RGB, self).__init__()
        M = torch.tensor([[1.0, 0.0, 1.13983],
                          [1.0, -0.39465, -0.58060],
                          [1.0, 2.03211, 0.0]], dtype=torch.float32)
        self.M = nn.Parameter(M, requires_grad=False)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # b h w c
        rgb = torch.matmul(x, self.M.T)
        rgb = rgb.permute(0, 3, 1, 2).contiguous()
        return rgb


def rgb_to_yuv(img):
    M = torch.tensor([[0.299, 0.587, 0.114],
                      [-0.14713, -0.28886, 0.436],
                      [0.615, -0.51499, -0.10001]], dtype=torch.float32).to(img.device)
    img = img.permute(0, 2, 3, 1).contiguous()  # b h w c
    yuv = torch.matmul(img, M.T)
    yuv = yuv.permute(0, 3, 1, 2).contiguous()
    return yuv


def yuv_to_rgb(img):
    M = torch.tensor([[1.0, 0.0, 1.13983],
                      [1.0, -0.39465, -0.58060],
                      [1.0, 2.03211, 0.0]], dtype=torch.float32).to(img.device)
    img = img.permute(0, 2, 3, 1).contiguous()  # b h w c
    rgb = torch.matmul(img, M.T)
    rgb = rgb.permute(0, 3, 1, 2).contiguous()
    return rgb


def get_transforms(
    img_size: int,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
):
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    return train_transform, val_transform


def get_resize_transform(img_size, resize_only=True):
    if resize_only:  # makes more sense for pre-training
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
        ])
    return transform, transform


def get_transforms_segmentation(
    img_size: int,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
):
    """
        Get transforms for segmentation task.
        Important: No random geometry transformations must be applied for the mask to be valid.
        Args:
            img_size: int: size of the image
            brightness: float: brightness factor
            contrast: float: contrast factor
            saturation: float: saturation factor
            hue: float: hue factor
        Returns:
            train_transform: transforms.Compose: transforms for training set
            train_mask_transform: transforms.Compose: transforms for mask in training set
            val_transform: transforms.Compose: transforms for validation set
    """
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.ToTensor(),
    ])
    train_mask_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    val_mask_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size)
    ])
    return train_transform, train_mask_transform, val_transform, val_mask_transform
