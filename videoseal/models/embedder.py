# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from ..modules.hidden import HiddenEncoder
from ..modules.msg_processor import MsgProcessor
from ..modules.unet import UNetMsg
from ..modules.vae import VAEDecoder, VAEEncoder
from ..modules.dvmark import DVMarkEncoder


class Embedder(nn.Module):
    """
    Abstract class for watermark embedding.
    """

    def __init__(self) -> None:
        super(Embedder, self).__init__()
        self.preprocess = lambda x: x * 2 - 1
        self.yuv = False  # used by WAM to know if the model should take YUV images

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        """
        Generate a random message
        """
        return None

    def get_last_layer(self) -> torch.Tensor:
        return None

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        return None


class VAEEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
        self,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        msg_processor: MsgProcessor
    ) -> None:
        super(VAEEmbedder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.msg_processor = msg_processor

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)  # b x k

    def get_last_layer(self) -> torch.Tensor:
        last_layer = self.decoder.conv_out.weight
        return last_layer

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        latents = self.encoder(imgs)
        latents_w = self.msg_processor(latents, msgs)
        imgs_w = self.decoder(latents_w)
        return imgs_w


class PatchmixerEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
        self,
        patchmixer: nn.Module,
        msg_processor: MsgProcessor
    ) -> None:
        super(PatchmixerEmbedder, self).__init__()
        self.patchmixer = patchmixer
        self.msg_processor = msg_processor

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)  # b x k

    def get_last_layer(self) -> torch.Tensor:
        last_layer = self.patchmixer.last_layer.weight
        return last_layer

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        imgs_w = self.patchmixer(imgs, msgs)
        return imgs_w


class UnetEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
        self,
        unet: nn.Module,
        msg_processor: MsgProcessor
    ) -> None:
        super(UnetEmbedder, self).__init__()
        self.unet = unet
        self.msg_processor = msg_processor

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)  # b x k

    def get_last_layer(self) -> torch.Tensor:
        last_layer = self.unet.outc.weight
        return last_layer

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        imgs_w = self.unet(imgs, msgs)
        return imgs_w


class HiddenEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
        self,
        hidden_encoder: HiddenEncoder
    ) -> None:
        super(HiddenEmbedder, self).__init__()
        self.hidden_encoder = hidden_encoder

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        nbits = self.hidden_encoder.num_bits
        return torch.randint(0, 2, (bsz, nbits))

    def get_last_layer(self) -> torch.Tensor:
        last_layer = self.hidden_encoder.final_layer.weight
        return last_layer

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        msgs = 2 * msgs.float() - 1
        imgs = self.preprocess(imgs)
        imgs_w = self.hidden_encoder(imgs, msgs)
        return imgs_w


class DVMarkEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
        self,
        unet: nn.Module,
    ) -> None:
        super(DVMarkEmbedder, self).__init__()
        self.unet = unet

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        nbits = self.unet.num_bits
        return torch.randint(0, 2, (bsz, nbits))

    def get_last_layer(self) -> torch.Tensor:
        last_layer = self.unet.emb_layer3[-1].weight
        return last_layer

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        imgs_w = self.unet(imgs, msgs)
        return imgs_w


def build_embedder(name, cfg, nbits, hidden_size_multiplier=2):
    hidden_size = int(nbits * hidden_size_multiplier)
    if name.startswith('vae'):
        # updates some cfg
        cfg.msg_processor.nbits = nbits
        cfg.msg_processor.hidden_size = hidden_size
        cfg.decoder.z_channels = hidden_size + cfg.encoder.z_channels
        # build the encoder, decoder and msg processor
        encoder = VAEEncoder(**cfg.encoder)
        msg_processor = MsgProcessor(**cfg.msg_processor)
        decoder = VAEDecoder(**cfg.decoder)
        embedder = VAEEmbedder(encoder, decoder, msg_processor)
    elif name.startswith('unet'):
        # updates some cfg
        cfg.msg_processor.nbits = nbits
        cfg.msg_processor.hidden_size = hidden_size
        # build the encoder, decoder and msg processor
        msg_processor = MsgProcessor(**cfg.msg_processor)
        unet = UNetMsg(msg_processor=msg_processor, **cfg.unet)
        embedder = UnetEmbedder(unet, msg_processor)
    elif name.startswith('hidden'):
        # updates some cfg
        cfg.num_bits = nbits
        # build the encoder, decoder and msg processor
        hidden_encoder = HiddenEncoder(**cfg)
        embedder = HiddenEmbedder(hidden_encoder)
    elif name.startswith('patchmixer'):
        # updates some cfg
        cfg.msg_processor.nbits = nbits
        cfg.msg_processor.hidden_size = nbits
        # build the encoder, decoder and msg processor
        msg_processor = MsgProcessor(**cfg.msg_processor)
        patchmixer = PatchmixerMsg(msg_processor=msg_processor, **cfg.patchmixer)
        embedder = PatchmixerEmbedder(patchmixer, msg_processor)
    elif name.startswith('dvmark'):
        embedder = DVMarkEmbedder(DVMarkEncoder(nbits))
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    embedder.yuv = True if 'yuv' in name else False
    return embedder
