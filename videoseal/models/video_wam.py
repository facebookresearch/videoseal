"""
Test with:
    python -m videoseal.models.video_wam
"""

import random

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from videoseal.augmentation.augmenter import Augmenter
from videoseal.modules.jnd import JND
from videoseal.models.embedder import Embedder
from videoseal.models.extractor import Extractor

class VideoWam(nn.Module):
    wm_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        embedder: Embedder,
        detector: Extractor,
        augmenter: Augmenter,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        img_size: int = 256,
        chunk_size: int = 8,
        step_size: int = 4,
        device: str = device,
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
        """
        super().__init__()
        # modules
        self.embedder = embedder
        self.detector = detector
        self.augmenter = augmenter
        self.attenuation = attenuation
        # scalings
        self.scaling_w = scaling_w
        self.scaling_i = scaling_i
        # video settings
        self.chunk_size = chunk_size  # encode 8 imgs at a time
        self.step_size = step_size  # propagate the wm to 4 next imgs
        self.resize_to = transforms.Resize(img_size, antialias=True)
        # device
        self.device = device

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.embedder.get_random_msg(bsz, nb_repetitions)  # b x k

    @torch.no_grad()
    def embed_inference(
        self,
        imgs: torch.Tensor,
        msg: torch.Tensor = None,
    ):
        """ 
        Does the forward pass of the encoder only.
        Rescale the watermark signal by a JND (just noticeable difference heatmap) that says where pixel can be changed without being noticed.
        The watermark signal is computed on the image downsampled to 256x... pixels, and then upsampled to the original size.
        The watermark signal is computed every step_size imgs and propagated to the next step_size imgs.

        Args:
            imgs: (torch.Tensor) Batched images with shape FxCxHxW
            msg: (torch.Tensor) Batched messages with shape 1xL
        """
        if msg is None:
            msg = self.get_random_msg()

        # encode by chunk of 8 imgs, propagate the wm to 4 next imgs
        chunk_size = self.chunk_size  # n
        step_size = self.step_size
        msg = msg.repeat(chunk_size, 1).to(self.device) # 1 k -> n k

        # initialize watermarked imgs
        imgs_w = torch.zeros_like(imgs) # f 3 h w

        for ii in range(0, len(imgs[::step_size]), chunk_size):
            nimgs_in_ck = min(chunk_size, len(imgs[::step_size]) - ii)
            start = ii*step_size
            end = start + nimgs_in_ck * step_size
            all_imgs_in_ck = imgs[start : end, ...].to(self.device) # f 3 h w

            # choose one frame every step_size
            imgs_in_ck = all_imgs_in_ck[::step_size] # n 3 h w
            # downsampling with fixed short edge
            imgs_in_ck = self.resize_to(imgs_in_ck) # n 3 wm_h wm_w
            # deal with last chunk that may have less than chunk_size frames
            if nimgs_in_ck < chunk_size:  
                msg = msg[:nimgs_in_ck]
            
            # get deltas for the chunk, and repeat them for each frame in the chunk
            deltas_in_ck = self.embedder(imgs_in_ck, msg) # n 3 wm_h wm_w
            deltas_in_ck = torch.repeat_interleave(deltas_in_ck, step_size, dim=0) # f 3 wm_h wm_w
            deltas_in_ck = deltas_in_ck[:len(all_imgs_in_ck)] # at the end of video there might be more deltas than needed
            
            # upsampling
            deltas_in_ck = nn.functional.interpolate(deltas_in_ck, size=imgs.shape[-2:], mode='bilinear', align_corners=True)
            
            # create watermarked imgs
            all_imgs_in_ck_w = self.scaling_i * all_imgs_in_ck + self.scaling_w * deltas_in_ck
            if self.attenuation is not None:
                all_imgs_in_ck_w = self.attenuation(all_imgs_in_ck, all_imgs_in_ck_w)
            imgs_w[start : end, ...] = all_imgs_in_ck_w.cpu() # n 3 h w

        return imgs_w

    @torch.no_grad()
    def detect_inference(
        self,
        imgs: torch.Tensor,
    ):
        """
        ...
        
        Args:
            imgs: (torch.Tensor) Batched images with shape FxCxHxW
        """
        ...


if __name__ == "__main__":
    pass