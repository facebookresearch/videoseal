"""
Test with:
    python -m videoseal.models.wam
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..augmentation.augmenter import Augmenter
from ..data.transforms import RGB2YUV
from ..modules.jnd import JND
from .blender import Blender
from .embedder import Embedder
from .extractor import Extractor


class Wam(nn.Module):

    @property
    def device(self):
        """Return the device of the model."""
        return next(self.parameters()).device

    def __init__(
        self,
        embedder: Embedder,
        detector: Extractor,
        augmenter: Augmenter,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        clamp: bool = True,
        img_size: int = 256,
        blending_method: str = "additive",
    ) -> None:
        """
        WAM (watermark-anything models) model that combines an embedder, a detector, and an augmenter.
        Embeds a message into an image and detects it as a mask.

        Arguments:
            embedder: The watermark embedder
            detector: The watermark detector
            augmenter: The image augmenter
            attenuation: The JND model to attenuate the watermark distortion
            scaling_w: The scaling factor for the watermark
            scaling_i: The scaling factor for the image
            img_size: The size at which the images are processed
            clamp: Whether to clamp the output images to [0, 1]
        """
        super().__init__()
        # modules
        self.embedder = embedder
        self.detector = detector
        self.augmenter = augmenter
        # image format
        self.img_size = img_size
        self.rgb2yuv = RGB2YUV()
        # blending
        assert blending_method in Blender.AVAILABLE_BLENDING_METHODS
        self.blender = Blender(scaling_i, scaling_w, blending_method)
        self.attenuation = attenuation
        self.clamp = clamp

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.embedder.get_random_msg(bsz, nb_repetitions)  # b x k

    def forward(
        self,
        imgs: torch.Tensor,
        masks: torch.Tensor,
        msgs: torch.Tensor = None,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    ) -> dict:
        """
        Does the full forward pass of the WAM model (used for training).
        (1) Generates watermarked images from the input images and messages.
        (2) Augments the watermarked images.
        (3) Detects the watermark in the augmented images.
        """
        # optionally create message
        if msgs is None:
            msgs = self.get_random_msg(imgs.shape[0])  # b x k
            msgs = msgs.to(imgs.device)

        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                     **interpolation)

        # generate watermarked images
        if self.embedder.yuv:  # take y channel only
            preds_w = self.embedder(self.rgb2yuv(imgs_res)[:, 0:1], msgs)
        else:
            preds_w = self.embedder(imgs_res, msgs)

        # interpolate back
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            preds_w = F.interpolate(preds_w, size=imgs.shape[-2:],
                                    **interpolation)
        preds_w = preds_w.to(imgs.device)
        imgs_w = self.blender(imgs, preds_w)

        # apply attenuation and clamp
        if self.attenuation is not None:
            self.attenuation.to(imgs.device)
            imgs_w = self.attenuation(imgs, imgs_w)
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)
        # augment
        imgs_aug, masks, selected_aug = self.augmenter(
            imgs_w, imgs, masks, is_video=False, do_resize=False)

        # interpolate back
        if imgs_aug.shape[-2:] != (self.img_size, self.img_size):
            imgs_aug = F.interpolate(imgs_aug, size=(self.img_size, self.img_size),
                                        **interpolation)
            
        # detect watermark
        preds = self.detector(imgs_aug)
        # create and return outputs
        outputs = {
            "msgs": msgs,  # original messages: b k
            "masks": masks,  # augmented masks: b 1 h w
            "preds_w": preds_w,  # predicted watermarks: b c h w
            "imgs_w": imgs_w,  # watermarked images: b c h w
            "imgs_aug": imgs_aug,  # augmented images: b c h w
            "preds": preds,  # predicted masks and/or messages: b (1+nbits) h w
            "selected_aug": selected_aug,  # selected augmentation
        }
        return outputs

    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor = None,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
        lowres_attenuation: bool = False,
    ) -> dict:
        """
        Generates watermarked images from the input images and messages (used for inference).
        Images may be arbitrarily sized.
        Args:
            imgs (torch.Tensor): Batched images with shape BxCxHxW.
            msgs (torch.Tensor): Optional messages with shape BxK.
            interpolation (dict): Interpolation parameters.
            lowres_attenuation (bool): Whether to attenuate the watermark at low resolution,
                which is more memory efficient for high-resolution images.
        Returns:
            dict: A dictionary with the following keys:
                - msgs (torch.Tensor): Original messages with shape BxK.
                - preds_w (torch.Tensor): Predicted watermarks with shape BxCxHxW.
                - imgs_w (torch.Tensor): Watermarked images with shape BxCxHxW.
        """
        # optionally create message
        if msgs is None:
            msgs = self.get_random_msg(imgs.shape[0])  # b x k

        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                     **interpolation)
        imgs_res = imgs_res.to(self.device)

        # generate watermarked images
        if self.embedder.yuv:  # take y channel only
            preds_w = self.embedder(
                self.rgb2yuv(imgs_res)[:, 0:1],
                msgs.to(self.device)
            )
        else:
            preds_w = self.embedder(imgs_res, msgs.to(self.device))

        # attenuate at low resolution if needed
        if self.attenuation is not None and lowres_attenuation:
            self.attenuation.to(imgs_res.device)
            hmaps = self.attenuation.heatmaps(imgs_res)
            preds_w = hmaps * preds_w

        # interpolate back
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            preds_w = F.interpolate(preds_w, size=imgs.shape[-2:],
                                    **interpolation)
        preds_w = preds_w.to(imgs.device)
        
        # apply attenuation
        if self.attenuation is not None and not lowres_attenuation:
            self.attenuation.to(imgs.device)
            hmaps = self.attenuation.heatmaps(imgs)
            preds_w = hmaps * preds_w

        # blend and clamp
        imgs_w = self.blender(imgs, preds_w)
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)

        outputs = {
            "msgs": msgs,  # original messages: b k
            "preds_w": preds_w,  # predicted watermarks: b c h w
            "imgs_w": imgs_w,  # watermarked images: b c h w
        }
        return outputs

    def detect(
        self,
        imgs: torch.Tensor,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    ) -> dict:
        """
        Performs the forward pass of the detector only (used at inference).
        Rescales the input images to 256x256 pixels and then computes the mask and the message.
        Args:
            imgs (torch.Tensor): Batched images with shape BxCxHxW.
        Returns:
            dict: A dictionary with the following keys:
                - preds (torch.Tensor): Predicted masks and/or messages with shape Bx(1+nbits)xHxW.
        """

        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                        **interpolation)
        imgs_res = imgs_res.to(self.device)

        # detect watermark
        preds = self.detector(imgs_res).to(imgs.device)

        outputs = {
            "preds": preds,  # predicted masks and/or messages: b (1+nbits) h w
        }
        return outputs


if __name__ == "__main__":

    from functools import partial

    import torch

    from videoseal.models.embedder import Embedder
    from videoseal.models.extractor import Extractor
    from videoseal.modules.msg_processor import MsgProcessor
    from videoseal.modules.pixel_decoder import PixelDecoder
    from videoseal.modules.vae import VAEDecoder, VAEEncoder
    from videoseal.modules.vit import ImageEncoderViT

    nbits = 0
    ch = 64
    ch_mult = (1, 2, 2)
    msg_emb_ch = 4 + 2*nbits

    print("\ntest for the embedder model\n")

    # test the embedder model
    encoder = VAEEncoder(
        ch=ch,
        out_ch=3,
        ch_mult=ch_mult,
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        in_channels=3,
        resolution=64,
        z_channels=4,
        double_z=False
    )
    msg_processor = MsgProcessor(
        nbits=nbits,
        hidden_size=msg_emb_ch,
        msg_processor_type="concat",
    )
    decoder = VAEDecoder(
        ch=ch,
        out_ch=3,
        ch_mult=ch_mult,
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        in_channels=3,
        resolution=64,
        z_channels=4 + msg_emb_ch,
    )

    # build the model
    model = Embedder(encoder, decoder, msg_processor)
    print(model)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M parameters')
    print(
        f'encoder: {sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')
    print(
        f'decoder: {sum(p.numel() for p in decoder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')
    print(
        f'msg_processor: {sum(p.numel() for p in msg_processor.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # test the model
    imgs = torch.randn(2, 3, 256, 256)
    msgs = torch.randint(0, 2, (2, nbits))
    out = model(imgs, msgs)
    print(out.shape)

    print("\ntest for the detector model\n")

    image_size = 256
    vit_patch_size = 16

    model = 'tiny'
    if model == 'base':
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
    elif model == 'small':
        encoder_embed_dim = 384
        encoder_depth = 12
        encoder_num_heads = 6
    elif model == 'tiny':
        encoder_embed_dim = 192
        encoder_depth = 12
        encoder_num_heads = 3

    encoder_global_attn_indexes = [2, 5, 8, 11]
    out_chans = 512

    image_embedding_size = image_size // vit_patch_size
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        # window_size = 14,
        out_chans=out_chans,
    )
    pixel_decoder = PixelDecoder(
        embed_dim=out_chans
    )

    detector = Extractor(
        image_encoder=image_encoder,
        pixel_decoder=pixel_decoder
    )

    print(detector)
    print(f'{sum(p.numel() for p in detector.parameters() if p.requires_grad) / 1e6:.1f}M parameters')
    print(
        f'image_encoder: {sum(p.numel() for p in image_encoder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')
    print(
        f'pixel_decoder: {sum(p.numel() for p in pixel_decoder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')
