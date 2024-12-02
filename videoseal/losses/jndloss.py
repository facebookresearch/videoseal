
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.jnd import JND


class JNDLoss(nn.Module):
    def __init__(self, 
        preprocess = lambda x: x,
        loss_type = 0
    ):
        super(JNDLoss, self).__init__()
        self.jnd = JND(preprocess)
        self.mse = nn.MSELoss()
        self.loss_type = loss_type
        self.rgbs = nn.Parameter(
            # torch.tensor([0.5, 0.5, 0.05]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            torch.tensor([0.299, 0.587, 0.114]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )  # 1 c 1 1
    
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
        elif self.loss_type == 1:
            jnd_max = 0.3
            loss =  self.mse(1.0 * deltas * (jnd_max - jnds), torch.zeros_like(deltas))
            return loss
        elif self.loss_type == 2:
            jnd_max = 0.1
            # ≈ jnd_max when jnds ≈ 0, = 0 when jnds > jnd_max
            jnds = jnd_max - torch.clamp(jnds, 0, jnd_max)  
            # penalize more green and red than blue
            jnds = self.rgbs * jnds  # 1 c 1 1 * b c h w -> b c h w
            loss =  self.mse(deltas * jnds / jnd_max, torch.zeros_like(deltas))
            return loss
        elif self.loss_type == 3:
            jnd_max = 1.0
            # ≈ jnd_max when jnds ≈ 0, =0 when jnds > jnd_max
            jnds = (1 / self.rgbs) * jnds  # 1 c 1 1 * b c h w -> b c h w
            jnds = jnd_max - torch.clamp(jnds, 0, jnd_max)  
            # penalize more green and red than blue
            loss =  self.mse(1e1 * deltas * jnds, torch.zeros_like(deltas))
            return loss
        elif self.loss_type == 4:
            jnd_max = 0.2
            # ≈ jnd_max when jnds ≈ 0, = 0 when jnds > jnd_max
            jnds = jnd_max - torch.clamp(jnds, 0, jnd_max)  
            # penalize more green and red than blue
            # jnds = self.rgbs * jnds  # 1 c 1 1 * b c h w -> b c h w
            loss =  self.mse(1e1 * deltas * jnds, torch.zeros_like(deltas))
            return loss
        elif self.loss_type == 5:
            # Avoid division by zero and stabilize training
            epsilon = 1e-2
            scaled_jnds = 1 / (jnds + epsilon)  # Inverse JND scaling
            # Calculate the weighted loss
            loss = self.mse(deltas * scaled_jnds, torch.zeros_like(deltas))
            return loss
        else:
            raise ValueError(f"Loss type {self.loss_type} not supported. Use 0 or 1")

