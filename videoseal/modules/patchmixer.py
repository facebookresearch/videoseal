"""
Test with:
    python -m videoseal.modules.patchmixer
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_activation, get_normalization


class Affine(nn.Module):
    def __init__(self, dim: int, init=1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim).view(-1, 1, 1))
        self.bias = nn.Parameter(init * torch.zeros(dim).view(-1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(self.bias, self.weight, x)


class MatmulConv2d(nn.Conv2d):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        bs, ch, h, w = image.shape
        if self.groups > 1:
            weight = self.weight.view(self.groups, -1, self.weight.shape[1])
            image = image.view(-1, self.groups, ch // self.groups, h * w)

            image = torch.matmul(weight, image).view(bs, -1, h, w)
        else:
            # use matmul for this case to keep a contiguous output
            image = torch.matmul(self.weight.squeeze(), image.view(bs, ch, h * w)).view(
                bs, -1, h, w
            )
        if self.bias is not None:
            image = image + self.bias.view(-1, 1, 1)
        return image


class InvertedResidualMLP(nn.Module):
    def __init__(self, dim, act_layer, norm_layer, mlp_ratio, layerscale_init, with_conv=False, kernel_size=7):
        super().__init__()
        hidden_dim = int(mlp_ratio * dim)
        if with_conv:
            padding = kernel_size // 2
            self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=True)
        else:
            self.conv1 = nn.Identity()
        self.norm = norm_layer(dim)
        self.fc1 = MatmulConv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.act = act_layer()
        self.fc2 = MatmulConv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.layerscale = Affine(dim, layerscale_init)

    def forward(self, x):
        z = self.conv1(x)
        z = self.norm(z)
        z = self.fc1(z)
        z = self.act(z)
        z = self.fc2(z)
        return self.layerscale(z) + x


class Encoder(nn.Module):
    def __init__(self, channels, act_layer, norm_layer, mlp_ratio, layerscale_init):
        super(Encoder, self).__init__()
        layerscale_init *= 1 / math.sqrt(len(channels))
        downs = [
            nn.PixelUnshuffle(4),
            nn.Conv2d(channels[0] * 16, channels[1], kernel_size=3, padding=1)
        ]
        last_chan = channels[1]
        for next_chan in channels[2:]:
            downs += [
                InvertedResidualMLP(
                    last_chan, act_layer, norm_layer, 
                    mlp_ratio, layerscale_init
                ),
                nn.PixelUnshuffle(2),
                MatmulConv2d(last_chan * 4, next_chan, kernel_size=1)
            ]
            last_chan = next_chan
        self.downs = nn.Sequential(*downs)

    def forward(self, x):
        return self.downs(x)


class Decoder(nn.Module):
    def __init__(self, channels, act_layer, norm_layer, mlp_ratio, layerscale_init):
        super(Decoder, self).__init__()
        layerscale_init *= 1 / math.sqrt(len(channels)) 
        ups = []
        last_chan = channels[0]
        for next_chan in channels[1:-1]:
            ups += [
                MatmulConv2d(last_chan, next_chan * 4, kernel_size=1),
                nn.PixelShuffle(2),
                InvertedResidualMLP(
                    next_chan, act_layer, norm_layer, 
                    mlp_ratio, layerscale_init)
            ]
            last_chan = next_chan
        ups += [
            nn.Conv2d(last_chan, channels[-1] * 16, kernel_size=3, padding=1),
            nn.PixelShuffle(4),
        ]
        self.ups = nn.Sequential(*ups)
        
    def forward(self, x):
        return self.ups(x)


class Bottleneck(nn.Module):
    def __init__(self, dim, num_blocks, act_layer, norm_layer, mlp_ratio, layerscale_init):
        super(Bottleneck, self).__init__()
        self.blocks = nn.Sequential(*[
            InvertedResidualMLP(
                dim, act_layer, norm_layer, 
                mlp_ratio, layerscale_init, 
                with_conv=True, kernel_size=7
            ) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        return self.blocks(x)


class PatchmixerMsg(nn.Module):
    def __init__(self,
        msg_processor: nn.Module,
        in_channels: int,
        out_channels: int,
        z_channels: int,
        z_channels_mults: tuple[float],
        num_blocks: int,
        activation: str,
        normalization: str,
        mlp_ratio: float = 4.0,
        layerscale_init: float = 1e-5,
        last_tanh: bool = True,
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            z_channels (int): Number of base channels in the latent space.
            z_channels_mults (tuple[float]): Multipliers for the latent space channels during downsampling/upsampling.
            num_blocks (int): Number of residual blocks in the bottleneck.
            activation (str): Activation function. E.g. 'relu', 'gelu'.
            normalization (str): Normalization layer. E.g. 'batch', 'layer'.
            mlp_ratio (float): Multiplier for the hidden dimension in the MLP. Default: 4.0.
            layerscale_init (float): Initial value for the layer scale. Default: 1e-5.
        """
        super(PatchmixerMsg, self).__init__()

        self.msg_processor = msg_processor
        self.last_tanh = last_tanh

        norm_layer = get_normalization(normalization)
        act_layer = get_activation(activation)

        z_channels = [int(z_channels * m) for m in z_channels_mults]
        self.encoder = Encoder(
            [in_channels] + z_channels, 
            act_layer, norm_layer, 
            mlp_ratio, layerscale_init
        )
        z_channels[-1] += msg_processor.hidden_size
        self.bottleneck = Bottleneck(
            z_channels[-1], num_blocks,
            act_layer, norm_layer,
            mlp_ratio, layerscale_init
        )        
        self.decoder = Decoder(
            z_channels[::-1] + [out_channels], 
            act_layer, norm_layer, 
            mlp_ratio, layerscale_init
        )
        self.last_layer = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, msgs):
        x = self.encoder(x)
        x = self.msg_processor(x, msgs)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.last_layer(x)
        if self.last_tanh:
            x = torch.tanh(x)
        return x


if __name__ == '__main__':
    
    from ..modules.msg_processor import MsgProcessor

    for nbits in [16, 64, 128]:
        msg_processor = MsgProcessor(nbits=nbits, hidden_size=nbits)
        model = PatchmixerMsg(
            msg_processor=msg_processor,
            in_channels=3,
            out_channels=3,
            z_channels=256,
            z_channels_mults=(1, 1, 2),
            num_blocks=4,
            activation='relu',
            normalization='layer',
            mlp_ratio=4.0,
            layerscale_init=1e-1,
            last_tanh=True,
        )
        # number of parameters
        # print(model)
        print(nbits)
        print(f'embedder: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M parameters')
        x = torch.randn(2, 3, 256, 256)
        msgs = msg_processor.get_random_msg(2)
        y = model(x, msgs)
        print(y.shape)
    
    for in_channels, out_channels in [(1, 1), (3, 1), (3, 3)]:
        msg_processor = MsgProcessor(nbits=64, hidden_size=64)
        model = PatchmixerMsg(
            msg_processor=msg_processor,
            in_channels=in_channels,
            out_channels=out_channels,
            z_channels=256,
            z_channels_mults=(1, 1, 2),
            num_blocks=4,
            activation='relu',
            normalization='batch',
            mlp_ratio=4.0,
            layerscale_init=1e-5,
            last_tanh=True,
        )
        # number of parameters
        # print(model)
        print(f'{in_channels} -> {out_channels}')
        print(f'embedder: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M parameters')
        x = torch.randn(2, in_channels, 256, 256)
        msgs = msg_processor.get_random_msg(2)
        y = model(x, msgs)
        print(y.shape)