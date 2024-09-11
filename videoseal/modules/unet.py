
from functools import partial
import einops

import torch
from torch import nn
import torch.nn.functional as F

from .common import ChanRMSNorm, Upsample, Downsample

# https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
# https://github.com/milesial/Pytorch-UNet/blob/master/train.py



class ResnetBlock(nn.Module):
    """Conv Norm Act * 2"""

    def __init__(self, in_channels, out_channels, act_layer, norm_layer, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            act_layer(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer()
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.double_conv(x) + self.res_conv(x)


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer, norm_layer, upsampling_type='bilinear'):
        super().__init__()
        self.up = Upsample(upsampling_type, in_channels, out_channels, 2, act_layer)
        self.conv = ResnetBlock(out_channels, out_channels, act_layer, norm_layer)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer, norm_layer, upsampling_type='bilinear'):
        super().__init__()
        if upsampling_type == 'bilinear':
            self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.down = Downsample(in_channels, out_channels, act_layer)
        self.conv = ResnetBlock(out_channels, out_channels, act_layer, norm_layer)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)


class BottleNeck(nn.Module):
    def __init__(self,
        num_blocks: int, 
        channels_in: int,
        channels_out: int, 
        act_layer: nn.Module,
        norm_layer: nn.Module,
        *args, **kwargs
    ) -> None:
        super(BottleNeck, self).__init__()
        model = []
        for _ in range(num_blocks):
            model += [ResnetBlock(channels_in, channels_out, act_layer, norm_layer)]
            channels_in = channels_out
        self.model = nn.Sequential(*model)
        
    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        return self.model(x)  # b c+c' h w -> b c h w


class UNetMsg(nn.Module):
    def __init__(self, 
        msg_processor: nn.Module,
        in_channels: int,
        out_channels: int,
        z_channels: int,
        num_blocks: int,
        activation: str,
        normalization: str,
        z_channels_mults: tuple[int],
        upsampling_type: str = 'bilinear',
        downsampling_type: str = 'bilinear',
        last_tanh: bool = True,
        zero_init: bool = False,
        bw: bool = False,
    ):
        super(UNetMsg, self).__init__()
        self.msg_processor = msg_processor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.num_blocks = num_blocks
        self.z_channels_mults = z_channels_mults
        self.last_tanh = last_tanh
        self.bw = bw
        if self.bw:
            out_channels = 1
            self.out_channels = 1
        self.connect_scale = 2 ** -0.5

        # Set the normalization layer
        if normalization == "batch":
            norm_layer = nn.BatchNorm2d
        elif normalization == "group":
            norm_layer = partial(nn.GroupNorm, num_groups=8)
        elif normalization == "layer":
            norm_layer = nn.LayerNorm
        elif normalization == "rms":
            norm_layer = ChanRMSNorm
        else:
            raise NotImplementedError
    
        # Set the activation layer
        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "leakyrelu":
            act_layer = partial(nn.LeakyReLU, negative_slope=0.2)
        elif activation == "gelu":
            act_layer = nn.GELU
        elif activation == "silu":
            act_layer = nn.SiLU
        else:
            raise NotImplementedError

        # Calculate the z_channels for each layer based on z_channels_mults
        z_channels = [self.z_channels * m for m in self.z_channels_mults]

        # Initial convolution
        self.inc = ResnetBlock(in_channels, z_channels[0], act_layer, norm_layer)

        # Downward path
        self.downs = nn.ModuleList()
        for ii in range(len(z_channels) - 1):
            self.downs.append(DBlock(z_channels[ii], z_channels[ii + 1], act_layer, norm_layer, downsampling_type))

        # Message mixing and middle blocks
        z_channels[-1] = z_channels[-1] + self.msg_processor.hidden_size
        self.bottleneck = BottleNeck(num_blocks, z_channels[-1], z_channels[-1], act_layer, norm_layer)

        # Upward path
        self.ups = nn.ModuleList()
        for ii in reversed(range(len(z_channels) - 1)):
            self.ups.append(UBlock(2 * z_channels[ii + 1], z_channels[ii], act_layer, norm_layer, upsampling_type))

        # Final output convolution
        self.outc = nn.Conv2d(z_channels[0], out_channels, 1)
        if zero_init:
            self.zero_init_(self.outc)

    def forward(self, 
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ):
        # Initial convolution
        x1 = self.inc(imgs)
        hiddens = [x1]
        
        # Downward path
        for dblock in self.downs:
            hiddens.append(dblock(hiddens[-1]))  # b d h w -> b d' h/2 w/2

        # Middle path
        hiddens.append(self.msg_processor(hiddens.pop(), msgs))  # b c+c' h w
        x = self.bottleneck(hiddens[-1])

        # Upward path
        concat_skip_connect = lambda x: torch.cat((x, hiddens.pop() * self.connect_scale), dim = 1)
        for ublock in self.ups:
            x = concat_skip_connect(x)  # b d h w -> b 2d h w
            x = ublock(x)  # b d h w

        # Output layer
        logits = self.outc(x)
        if self.last_tanh:
            logits = torch.tanh(logits)
        if self.bw:
            logits = logits.repeat(1, 3, 1, 1)
            # rgbs = torch.tensor([0.299, 0.587, 0.114]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # logits = logits * rgbs.to(logits.device)
        return logits

    def use_checkpointing(self):
        # Apply checkpointing to save memory during training
        self.inc = torch.utils.checkpoint(self.inc)
        for ii in range(len(self.downs)):
            self.downs[ii] = torch.utils.checkpoint(self.downs[ii])
        for ii in range(len(self.ups)):
            self.ups[ii] = torch.utils.checkpoint(self.ups[ii])
        self.outc = torch.utils.checkpoint(self.outc)

    def zero_init_(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        return m