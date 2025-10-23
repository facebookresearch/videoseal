# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from wmforger.modules.convnext import ConvNeXtV2
from wmforger.modules.pixel_decoder import PixelDecoder


class Extractor(nn.Module):
    """
    Abstract class for watermark detection.
    """

    def __init__(self) -> None:
        super(Extractor, self).__init__()

    def preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        return imgs * 2 - 1
    
    def postprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        return (imgs + 1) / 2

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The predicted masks and/or messages.
        """
        return ...


class ConvnextExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """

    def __init__(
        self,
        convnext: ConvNeXtV2,
        pixel_decoder: PixelDecoder,
    ) -> None:
        super(ConvnextExtractor, self).__init__()
        self.convnext = convnext
        self.pixel_decoder = pixel_decoder

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            masks: (torch.Tensor) Batched masks with shape Bx(1+nbits)xHxW
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        latents = self.convnext(imgs)  # b c h/f w/f
        masks = self.pixel_decoder(latents)
        return masks


def build_extractor(name, cfg, img_size, nbits):
    assert name.startswith('convnext')
    # updates some cfg
    cfg.pixel_decoder.nbits = nbits
    # build the encoder, decoder and msg processor
    convnext = ConvNeXtV2(**cfg.encoder)
    pixel_decoder = PixelDecoder(**cfg.pixel_decoder)
    extractor = ConvnextExtractor(convnext, pixel_decoder)
    
    return extractor
