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
from videoseal.models.embedder import Embedder
from videoseal.models.extractor import Extractor
from videoseal.models.wam import Wam
from videoseal.modules.jnd import JND


class VideoWam(Wam):
    """
    A video watermarking model that extends the Wam class.
    This model combines an embedder, a detector, and an augmenter to embed watermarks into videos.
    It also includes optional attenuation and scaling parameters to control the strength of the watermark.
    Attributes:
        embedder (Embedder): The watermark embedder.
        detector (Extractor): The watermark detector.
        augmenter (Augmenter): The image augmenter.
        attenuation (JND, optional): The JND model to attenuate the watermark distortion. Defaults to None.
        scaling_w (float, optional): The scaling factor for the watermark. Defaults to 1.0.
        scaling_i (float, optional): The scaling factor for the image. Defaults to 1.0.
        chunk_size (int, optional): The number of frames to encode at a time. Defaults to 8.
        step_size (int, optional): The number of frames to propagate the watermark to. Defaults to 4.
        img_size (int, optional): The size of the images to resize to. Defaults to 256.
    """

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
    ) -> None:
        """
        Initializes the VideoWam model.
        Args:
            embedder (Embedder): The watermark embedder.
            detector (Extractor): The watermark detector.
            augmenter (Augmenter): The image augmenter.
            attenuation (JND, optional): The JND model to attenuate the watermark distortion. Defaults to None.
            scaling_w (float, optional): The scaling factor for the watermark. Defaults to 1.0.
            scaling_i (float, optional): The scaling factor for the image. Defaults to 1.0.
            img_size (int, optional): The size of the images to resize to. Defaults to 256.
            chunk_size (int, optional): The number of frames to encode at a time. Defaults to 8.
            step_size (int, optional): The number of frames to propagate the watermark to. Defaults to 4.
        """
        super().__init__(
            embedder=embedder,
            detector=detector,
            augmenter=augmenter,
            attenuation=attenuation,
            scaling_w=scaling_w,
            scaling_i=scaling_i,
        )
        # video settings
        self.chunk_size = chunk_size  # encode 8 frames at a time
        self.step_size = step_size  # propagate the wm to 4 next frames
        self.resize_to = transforms.Resize(
            (img_size, img_size), antialias=True)

    def forward_video(self, frames: torch.Tensor,
                      msg: torch.Tensor = None,
                      ):
        raise NotImplementedError

    @torch.no_grad()
    def embed_inference(
        self,
        frames: torch.Tensor,
        msg: torch.Tensor = None,
    ) -> torch.Tensor:
        """ 
        Does the forward pass of the encoder only.
        Rescale the watermark signal by a JND (just noticeable difference heatmap) that says where pixel can be changed without being noticed.
        The watermark signal is computed on the image downsampled to 256x... pixels, and then upsampled to the original size.
        The watermark signal is computed every step_size frames and propagated to the next step_size frames.

        Args:
            frames: (torch.Tensor) Batched images with shape FxCxHxW
            msg: (torch.Tensor) Batched messages with shape 1xL

        Returns:
            frames_w: (torch.Tensor) Batched watermarked images with shape FxCxHxW
        """
        if msg is None:
            msg = self.get_random_msg()

        # encode by chunk of 8 frames, propagate the wm to 4 next frames
        chunk_size = self.chunk_size  # n
        step_size = self.step_size
        msg = msg.repeat(chunk_size, 1).to(frames.device)  # 1 k -> n k

        # initialize watermarked frames
        frames_w = torch.zeros_like(frames)  # f 3 h w

        for ii in range(0, len(frames[::step_size]), chunk_size):
            nframes_in_ck = min(chunk_size, len(frames[::step_size]) - ii)
            start = ii*step_size
            end = start + nframes_in_ck * step_size
            all_frames_in_ck = frames[start: end, ...]  # f 3 h w

            # choose one frame every step_size
            frames_in_ck = all_frames_in_ck[::step_size]  # n 3 h w
            # downsampling with fixed short edge
            frames_in_ck = self.resize_to(frames_in_ck)  # n 3 wm_h wm_w
            # deal with last chunk that may have less than chunk_size frames
            if nframes_in_ck < chunk_size:
                msg = msg[:nframes_in_ck]

            # get deltas for the chunk, and repeat them for each frame in the chunk
            deltas_in_ck = self.embedder(frames_in_ck, msg)  # n 3 wm_h wm_w
            deltas_in_ck = torch.repeat_interleave(
                deltas_in_ck, step_size, dim=0)  # f 3 wm_h wm_w
            # at the end of video there might be more deltas than needed
            deltas_in_ck = deltas_in_ck[:len(all_frames_in_ck)]

            # upsampling
            deltas_in_ck = nn.functional.interpolate(
                deltas_in_ck, size=frames.shape[-2:], mode='bilinear', align_corners=True)

            # create watermarked frames
            all_frames_in_ck_w = self.scaling_i * \
                all_frames_in_ck + self.scaling_w * deltas_in_ck
            if self.attenuation is not None:
                all_frames_in_ck_w = self.attenuation(
                    all_frames_in_ck, all_frames_in_ck_w)
            # all_frames_in_ck = all_frames_in_ck.cpu()  # move to cpu to save gpu memory
            frames_w[start: end, ...] = all_frames_in_ck_w  # n 3 h w

        return frames_w

    @torch.no_grad()
    def detect_inference(
        self,
        frames: torch.Tensor,
        aggregation: str = "avg",
    ) -> torch.Tensor:
        """
        Does the forward pass of the detector only.
        Rescale the image to 256x... pixels, and then compute the mask and the message.

        Args:
            frames: (torch.Tensor) Batched images with shape FxCxHxW
        """
        frames = self.resize_to(frames)
        chunksize = 16  # n
        all_preds = []
        for ii in range(0, len(frames), chunksize):
            nframes_in_ck = min(chunksize, len(frames) - ii)
            preds = self.detector(
                frames[ii:ii+nframes_in_ck]
            )
            all_preds.append(preds)  # n k ..
        preds = torch.cat(all_preds, dim=0)  # f k ..
        mask_preds = preds[:, 0:1]  # binary detection bit (not used for now)
        bit_preds = preds[:, 1:]  # b k ..

        if aggregation is None:
            decoded_msg = bit_preds
        elif aggregation == "avg":
            decoded_msg = bit_preds.mean(dim=0)
        elif aggregation == "weighted_avg":
            decoded_msg = (bit_preds * bit_preds.abs()).mean(dim=0)  # b k -> k
        msg = (decoded_msg > 0).squeeze()
        return msg


if __name__ == "__main__":
    pass
