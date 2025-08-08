# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py

import functools
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

from ..data.transforms import RGB2YUV


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=32, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = lambda ndf: nn.GroupNorm(4, ndf)
            # norm_layer = nn.BatchNorm2d()
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.GroupNorm
            # use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.GroupNorm
            # use_bias = norm_layer != nn.BatchNorm2d

        self.input_nc = input_nc
        self.rgb2yuv = RGB2YUV()

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # If input has 3 channels but the model is for 1 channel, only use the first channel
        if self.input_nc == 1 and input.shape[1] == 3:
            input = self.rgb2yuv(input)[:, 0:1]
        return self.main(input)


class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


class BlurBlock(nn.Module):
    def __init__(self, kernel: Tuple[int] = (1, 3, 3, 1)):
        """Initializes the blur block.

        Args:
            kernel -> Tuple[int]: The kernel size.
        """
        super().__init__()

        self.kernel_size = len(kernel)

        kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False)
        kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.kernel = kernel

    def calc_same_pad(self, i: int, k: int, s: int) -> int:
        """Calculates the same padding for the BlurBlock.

        Args:
            i -> int: Input size.
            k -> int: Kernel size.
            s -> int: Stride.

        Returns:
            pad -> int: The padding.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            out -> torch.Tensor: The output tensor.
        """
        ic, ih, iw = x.size()[-3:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size, s=2)
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size, s=2)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        kernel_on_device = self.kernel.to(x.device)
        weight = kernel_on_device.expand(ic, -1, -1, -1)

        out = F.conv2d(input=x, weight=weight, stride=2, groups=x.shape[1])
        return out


class Conv2dSame(nn.Conv2d):
    """Convolution wrapper for 2D convolutions using `SAME` padding."""

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """Calculate padding such that the output has the same height/width when stride=1.

        Args:
            i -> int: Input size.
            k -> int: Kernel size.
            s -> int: Stride size.
            d -> int: Dilation rate.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolution applying explicit `same` padding.

        Args:
            x -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return super().forward(x)


class NLayerDiscriminatorV2(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 128,
        num_stages: int = 3,
        activation_fn: str = "leaky_relu",
        blur_resample: bool = True,
        blur_kernel_size: int = 4,
        use_maxpool: bool = False,
    ):
        """Initializes the NLayerDiscriminatorV2. 
        Discriminator taken from the MaskBit paper https://arxiv.org/abs/2409.16211

        Args:
            num_channels -> int: The number of input channels.
            hidden_channels -> int: The number of hidden channels.
            num_stages -> int: The number of stages.
            activation_fn -> str: The activation function.
            blur_resample -> bool: Whether to use blur resampling.
            blur_kernel_size -> int: The blur kernel size.
        """
        super().__init__()
        assert num_stages > 0, "Discriminator cannot have 0 stages"
        assert (not blur_resample) or (
            blur_kernel_size >= 3 and blur_kernel_size <= 5
        ), "Blur kernel size must be in [3,5] when sampling]"
        self.input_nc = num_channels
        self.rgb2yuv = RGB2YUV()
        in_channel_mult = (1,) + tuple(map(lambda t: 2**t, range(num_stages)))
        init_kernel_size = 5
        if activation_fn == "leaky_relu":
            activation = functools.partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            activation = nn.SiLU

        sequence = [
            Conv2dSame(num_channels, hidden_channels, kernel_size=init_kernel_size),
            activation(),
        ]

        BLUR_KERNEL_MAP = {
            3: (1, 2, 1),
            4: (1, 3, 3, 1),
            5: (1, 4, 6, 4, 1),
        }

        for i_level in range(num_stages):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]
            sequence += [
                Conv2dSame(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                ),
                (
                    nn.AvgPool2d(kernel_size=2, stride=2)
                    if not blur_resample
                    else BlurBlock(BLUR_KERNEL_MAP[blur_kernel_size])
                ),
                nn.GroupNorm(32, out_channels),
                activation(),
            ]
        if use_maxpool:
            sequence += [nn.AdaptiveMaxPool2d((16, 16))]

        sequence += [
            Conv2dSame(out_channels, out_channels, 1),
            activation(),
            Conv2dSame(out_channels, 1, kernel_size=5),
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            output -> torch.Tensor: The output tensor.
        """
        # If input has 3 channels but the model is for 1 channel, only use the first channel
        if self.input_nc == 1 and x.shape[1] == 3:
            x = self.rgb2yuv(x)[:, 0:1]
        return self.main(x)


class MultiscaleDisc(nn.Module):
    def __init__(
        self,
        disc_scales: int,
        version: str = "v1",
        in_channels: int = 3,
        num_layers: int = 3,
        use_actnorm: bool = False,
    ):
        super().__init__()
        self.discriminators = nn.ModuleDict(
            {
                str(2**ii): build_discriminator(
                    scales=1,
                    version=version,
                    in_channels=in_channels,
                    num_layers=num_layers,
                    use_actnorm=use_actnorm,
                )
                for ii in range(disc_scales)
            }
        )

    def forward(self, inputs):
        logits = []
        for scale, disc in self.discriminators.items():
            if scale == "1":
                resized_inputs = inputs
            else:
                resized_inputs = F.interpolate(
                    inputs,
                    scale_factor=1 / int(scale),
                    mode="bilinear",
                    align_corners=False,
                )
            logits.append(disc(resized_inputs).reshape(inputs.size(0), -1))
        logits = torch.cat(logits, dim=1)
        return logits


def build_discriminator(
    scales: int = 1,
    version: str = "v1",
    in_channels: int = 3,
    num_layers: int = 3,
    use_actnorm: bool = False,
) -> nn.Module:
    """
    Choose which version of the discriminator to use.
    Args:
        version: (str) The version of the discriminator to use.
            Options: "v1", "v2". v2 is MaskBit
        in_channels: (int) The number of input channels for the discriminator.
        num_layers: (int) The number of layers in the discriminator.
        use_actnorm: (bool) Whether to use ActNorm in the discriminator.
    """
    if scales == 1:
        if version == "v1":
            return NLayerDiscriminator(
                    input_nc=in_channels,
                    n_layers=num_layers,
                    use_actnorm=use_actnorm
            ).apply(weights_init)
        elif version == "v2":
            return NLayerDiscriminatorV2(
                num_channels=in_channels,
                num_stages=num_layers,
            )
        else:
            raise ValueError(f"Unknown discriminator version: {version}")
    else:
        return MultiscaleDisc(scales)
