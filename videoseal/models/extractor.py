
import torch
from torch import nn
from videoseal.modules.convnext import ConvNeXtV2
from videoseal.modules.hidden import HiddenDecoder
from videoseal.modules.pixel_decoder import PixelDecoder
from videoseal.modules.vit import ImageEncoderViT


class Extractor(nn.Module):
    """
    Abstract class for watermark detection.
    """

    def __init__(self) -> None:
        super(Extractor, self).__init__()

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


class SegmentationExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        pixel_decoder: PixelDecoder,
    ) -> None:
        super(SegmentationExtractor, self).__init__()
        self.image_encoder = image_encoder
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
        latents = self.image_encoder(imgs)
        masks = self.pixel_decoder(latents)

        return masks


class DinoExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """

    def __init__(
        self,
        image_encoder: str,
        hook_indices: list[int],
        pixel_decoder: PixelDecoder,
    ) -> None:
        super(DinoExtractor, self).__init__()
        assert image_encoder in ['dinov2_vits14', 'dinov2_vitb14']
        # vits 384, vitb 768
        self.image_encoder = torch.hub.load(
            'facebookresearch/dinov2', image_encoder)
        self.image_encoder.mask_token = None
        self.hook_indices = hook_indices
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
        latents = self.image_encoder.get_intermediate_layers(
            imgs,
            reshape=True, n=self.hook_indices
        )  # 4 x b c h/f w/f
        latents = torch.cat(latents, dim=1)  # 4 b c h/f w/f -> b 4c h/f w/f
        masks = self.pixel_decoder(latents)

        return masks


class HiddenExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """

    def __init__(
        self,
        hidden_decoder: HiddenDecoder,
    ) -> None:
        super(HiddenExtractor, self).__init__()
        self.hidden_decoder = hidden_decoder

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
        masks = self.hidden_decoder(imgs)
        return masks


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
        latents = self.convnext(imgs)  # b c h/f w/f
        masks = self.pixel_decoder(latents)
        return masks


def build_extractor(name, cfg, img_size, nbits):
    if name.startswith('sam'):
        cfg.encoder.img_size = img_size
        cfg.pixel_decoder.nbits = nbits
        image_encoder = ImageEncoderViT(**cfg.encoder)
        pixel_decoder = PixelDecoder(**cfg.pixel_decoder)
        extractor = SegmentationExtractor(
            image_encoder=image_encoder, pixel_decoder=pixel_decoder)
    elif name.startswith('dino2'):
        image_encoder = cfg.encoder.name
        hook_indices = cfg.encoder.hook_indices
        pixel_decoder = PixelDecoder(**cfg.pixel_decoder)
        extractor = DinoExtractor(image_encoder, hook_indices, pixel_decoder)
    elif name.startswith('hidden'):
        # updates some cfg
        cfg.num_bits = nbits
        # build the encoder, decoder and msg processor
        hidden_decoder = HiddenDecoder(**cfg)
        extractor = HiddenExtractor(hidden_decoder)
    elif name.startswith('convnext'):
        # updates some cfg
        cfg.num_bits = nbits
        # build the encoder, decoder and msg processor
        convnext = ConvNeXtV2(**cfg.encoder)
        pixel_decoder = PixelDecoder(**cfg.pixel_decoder)
        extractor = ConvnextExtractor(convnext, pixel_decoder)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    return extractor
