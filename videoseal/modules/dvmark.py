# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class DVMarkEncoder(nn.Module):

    def __init__(self, num_bits):
        super(DVMarkEncoder, self).__init__()
        self.num_bits = num_bits

        layers = []
        in_channels = 3
        for i in range(4):
            layers.append(nn.Conv3d(in_channels, 64, kernel_size=(1 if i < 3 else 3, 3, 3), padding=(0 if i < 3 else 1, 1, 1)))
            layers.append(nn.ReLU())
            in_channels = 64
        self.transform_layer = nn.Sequential(*layers)

        layers = []
        in_channels = 64 + num_bits
        out_channels = 256
        for i in range(3):
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels = 128
        self.emb_layer1 = nn.Sequential(*layers)

        self.avg_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        layers = []
        in_channels = 128 + num_bits
        out_channels = 512
        for i in range(3):
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels = 256
        self.emb_layer2 = nn.Sequential(*layers)


        layers = []
        in_channels = 128 + 256
        out_channels = 256
        for i in range(3):
            if i == 2:
                out_channels = 3
                layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels = 128
        self.emb_layer3 = nn.Sequential(*layers)

        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):
        imgs = imgs.unsqueeze(0).permute(0, 2, 1, 3, 4)

        # imgs.shape == [B, C, T, H, W]
        # msgs.shape == [B, num_bits]
        B, C, T, H, W = imgs.shape
        msgs = msgs.permute(1, 0)
        msgs_fullsize = msgs.view(1, self.num_bits, T, 1, 1).expand(-1, -1, -1, H, W)
        msgs_halfsize = msgs.view(1, self.num_bits, T, 1, 1).expand(-1, -1, -1, H//2, W//2)

        x = self.transform_layer(imgs)
        x = torch.cat([x, msgs_fullsize], 1)
        x_skip = self.emb_layer1(x)

        x = self.avg_pool(x_skip)
        x = torch.cat([x, msgs_halfsize], 1)
        x = self.emb_layer2(x)

        x = torch.nn.functional.upsample_bilinear(x.permute(0, 2, 1, 3, 4).view(B * T, 256, H//2, W//2), size=(H, W)).view(B, T, 256, H, W).permute(0, 2, 1, 3, 4)
        x = torch.cat([x_skip, x], 1)
        x = self.emb_layer3(x)

        x = self.tanh(x)
        x = x.permute(0, 2, 1, 3, 4).squeeze(0)
        return x


class DVMarkDecoder(nn.Module):

    def __init__(self, num_bits):
        super(DVMarkDecoder, self).__init__()
        self.num_bits = num_bits

        layers = {}
        in_channels = 3
        for i, out_channels in enumerate([128, 128, 256, self.num_bits + 1]):
            layers[f"layer{i + 1}"] = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            in_channels = out_channels
        self.layers = nn.ModuleDict(layers)

        self.avg_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4)
        x = self.layers["layer1"](x)
        x = self.relu(x)
        x = self.layers["layer2"](x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.layers["layer3"](x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.layers["layer4"](x)
        x = x.mean(dim=[3, 4])
        x = x.permute(0, 2, 1).squeeze(0)
        return x
