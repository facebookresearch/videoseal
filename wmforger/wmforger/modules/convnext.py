# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from .common import LayerNorm, GRN
from .vit import TemporalBlock


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., temporal_attention=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.temp_block = None
        if temporal_attention:
            self.temp_block = TemporalBlock(dim, num_heads=dim//48, use_rel_pos=True, video_len=32)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous() # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        if self.temp_block is not None:
            x = self.temp_block(x)

        x = x.permute(0, 3, 1, 2).contiguous() # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvnextConv2p1dWrapper(nn.Module):
    def __init__(self, *args, temporal_kernel_size=3, **kwargs):
        """
        Wrapper for 2D convolution then optional temporal convolution to handle 4D input tensors.
        Allows to keep 2D convolution unchanged, then add a temporal convolution.
        Args:
            *args: Arguments for nn.Conv2d.
            **kwargs: Keyword arguments for nn.Conv2d.
        """
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        assert temporal_kernel_size % 2 == 1
        self.temp_conv = nn.Conv3d(
            args[1], args[1], # in_channels, out_channels
            kernel_size=(temporal_kernel_size, 1, 1), 
            padding=(temporal_kernel_size // 2, 0, 0), 
            bias=False
        )

    def forward(self, x):
        assert len(x.shape) == 4
        x = self.conv(x)
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4) # change [B, C, H, W] to [1, C, T, H, W]
        x = self.temp_conv(x)
        x = x.permute(0, 2, 1, 3, 4).squeeze(0)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_bits (int): Number of bits.
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(
        self, in_chans=3,
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
        drop_path_rate=0., head_init_scale=1., 
        temporal_convs=False, temporal_attention=False
    ) -> None:
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        op_ = ConvnextConv2p1dWrapper if temporal_convs else nn.Conv2d
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    op_(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(
                    dim=dims[i], drop_path=dp_rates[cur + j], 
                    temporal_attention=temporal_attention
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_bits)

        # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for ii, (down, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            x = down(x)
            x = stage(x)
        return x # no average pooling, (N, C, H, W)

    def forward(self, x):
        x = self.forward_features(x)
        return x

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


if __name__ == '__main__':
    model = convnextv2_tiny()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
    print(model)
    print("ConvNeXtV2 model created successfully.")

    # try scripting
    scripted_model = torch.jit.script(model)
    print(scripted_model)