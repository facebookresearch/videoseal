
import torch
from torch import nn
from videoseal.data.transforms import rgb_to_yuv, yuv_to_rgb
from videoseal.modules.hidden import HiddenEncoder
from videoseal.modules.msg_processor import MsgProcessor
from videoseal.modules.unet import UNetMsg
from videoseal.modules.vae import VAEDecoder, VAEEncoder


class Embedder(nn.Module):
    """
    Abstract class for watermark embedding.
    """

    def __init__(self) -> None:
        super(Embedder, self).__init__()

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        """
        Generate a random message
        """
        return ...

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
        return ...


class VAEEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
        self,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        msg_processor: MsgProcessor,
        yuv: bool = False
    ) -> None:
        super(VAEEmbedder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.msg_processor = msg_processor
        self.yuv = yuv

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)  # b x k

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
        if self.yuv:
            imgs = rgb_to_yuv(imgs)
        latents = self.encoder(imgs)
        latents_w = self.msg_processor(latents, msgs)
        imgs_w = self.decoder(latents_w)
        if self.yuv:
            imgs_w = yuv_to_rgb(imgs_w)
        return imgs_w


class UnetEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
        self,
        unet: nn.Module,
        msg_processor: MsgProcessor,
        yuv: bool = False
    ) -> None:
        super(UnetEmbedder, self).__init__()
        self.unet = unet
        self.msg_processor = msg_processor
        self.yuv = yuv

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)  # b x k

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
        if self.yuv:
            imgs = rgb_to_yuv(imgs)
        imgs_w = self.unet(imgs, msgs)
        if self.yuv:
            imgs_w = yuv_to_rgb(imgs_w)
        return imgs_w


class HiddenEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
        self,
        hidden_encoder: HiddenEncoder,
    ) -> None:
        super(HiddenEmbedder, self).__init__()
        self.hidden_encoder = hidden_encoder

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        nbits = self.hidden_encoder.num_bits
        return torch.randint(0, 2, (bsz, nbits))

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
        return self.hidden_encoder(imgs, msgs)


def build_embedder(name, cfg, nbits):
    if name.startswith('vae'):
        # updates some cfg
        cfg.msg_processor.nbits = nbits
        cfg.msg_processor.hidden_size = nbits * 2
        cfg.decoder.z_channels = (nbits * 2) + cfg.encoder.z_channels
        # build the encoder, decoder and msg processor
        encoder = VAEEncoder(**cfg.encoder)
        msg_processor = MsgProcessor(**cfg.msg_processor)
        decoder = VAEDecoder(**cfg.decoder)
        yuv = cfg.get('yuv', False)
        embedder = VAEEmbedder(encoder, decoder, msg_processor, yuv)
    elif name.startswith('unet'):
        # updates some cfg
        cfg.msg_processor.nbits = nbits
        cfg.msg_processor.hidden_size = nbits * 2
        # build the encoder, decoder and msg processor
        msg_processor = MsgProcessor(**cfg.msg_processor)
        unet = UNetMsg(msg_processor=msg_processor, **cfg.unet)
        yuv = cfg.get('yuv', False)
        embedder = UnetEmbedder(unet, msg_processor, yuv)
    elif name.startswith('hidden'):
        # updates some cfg
        cfg.num_bits = nbits
        # build the encoder, decoder and msg processor
        hidden_encoder = HiddenEncoder(**cfg)
        embedder = HiddenEmbedder(hidden_encoder)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    return embedder
