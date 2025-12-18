# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.jnd import JND


class JNDLoss(nn.Module):
    def __init__(self,
        loss_type = 0
    ):
        super(JNDLoss, self).__init__()
        self.jnd = JND()
        self.mse = nn.MSELoss()
        self.loss_type = loss_type
    
    def forward(
        self, 
        imgs: torch.Tensor,
        imgs_w: torch.Tensor,
    ):
        jnds = self.jnd.heatmaps(imgs)  # b 1 h w
        deltas = imgs_w - imgs  # b c h w
        if self.loss_type == 0:
            loss = self.mse(1.0 * deltas.abs(), jnds)
            return loss
        else:
            raise ValueError(f"Loss type {self.loss_type} not supported. Use 0 or 1")
