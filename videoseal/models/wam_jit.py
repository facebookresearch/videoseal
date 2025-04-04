# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains TorchScript-compatible versions of the Wam and VideoWam classes.
Run with:
    python -m videoseal.models.wam_jit
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..data.transforms import RGB2YUV
from ..modules.jnd import JND
from .embedder import Embedder
from .extractor import Extractor


class Blender(nn.Module):
    def __init__(self, scaling_i, scaling_w):
        super(Blender, self).__init__()
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

    def forward(self, imgs: torch.Tensor, preds_w: torch.Tensor) -> torch.Tensor:
        """
        Simple additive blending.
        
        Parameters:
            imgs (torch.Tensor): The original image batch tensor
            preds_w (torch.Tensor): The watermark batch tensor
            
        Returns:
            torch.Tensor: Blended output as scaling_i * imgs + scaling_w * preds_w
        """
        return self.scaling_i * imgs + self.scaling_w * preds_w


class WamJIT(nn.Module):
    """
    TorchScript-compatible version of Wam and VideoWam classes.
    This model is optimized for inference and doesn't include augmentation functionality.
    """

    def __init__(
        self,
        embedder: Embedder,
        detector: Extractor,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        img_size: int = 256,
        clamp: bool = True,
        chunk_size: int = 16,
        step_size: int = 4,
        video_mode: str = "repeat",
        do_attenuation: bool = True,
        lowres_attenuation: bool = True,
    ) -> None:
        """
        Initialize the WamJIT model.

        Args:
            embedder (Embedder): The watermark embedder
            detector (Extractor): The watermark detector
            attenuation (JND, optional): The JND model to attenuate the watermark distortion. Defaults to None.
            scaling_w (float, optional): The scaling factor for the watermark. Defaults to 1.0.
            scaling_i (float, optional): The scaling factor for the image. Defaults to 1.0.
            img_size (int, optional): The size at which the images are processed. Defaults to 256.
            clamp (bool, optional): Whether to clamp the output images to [0, 1]. Defaults to True.
            chunk_size (int, optional): The number of frames/imgs to encode at a time for videos. Defaults to 16.
            step_size (int, optional): The number of frames/imgs to propagate the watermark to. Defaults to 4.
            video_mode (str, optional): The mode to use for video watermarking. Defaults to "repeat".
            lowres_attenuation (bool, optional): Whether to attenuate at low resolution. Defaults to True.
        """
        super().__init__()
        # modules
        self.embedder = embedder
        self.detector = detector
        # image format
        self.img_size = img_size
        self.rgb2yuv = RGB2YUV()
        # blending - using Blender instead of Blender
        self.blender = Blender(scaling_i, scaling_w)
        self.clamp = clamp
        # video settings
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.video_mode = video_mode
        # attenuation settings
        self.attenuation = attenuation
        self.do_attenuation = do_attenuation
        self.lowres_attenuation = lowres_attenuation
        assert do_attenuation == (attenuation is not None), "Attenuation must be set if do_attenuation is True"

    def forward(
        self,
        imgs: torch.Tensor,  # [b, c, h, w] or [frames, c, h, w]
        msgs: torch.Tensor,
        is_video: bool = False,
        aggregate: bool = False,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ) -> tuple:
        """
        Forward pass that embeds a message and then detects it.
        
        Args:
            imgs: Input images or video frames
            msgs: Optional messages to embed
            is_video: Whether the input is a video
            aggregate: Aggregate the detection results into a single message. Only for video
            mode: Interpolation mode
            align_corners: Whether to align corners in interpolation
            antialias: Whether to use antialiasing in interpolation
            
        Returns:
            tuple: (watermarked_imgs, detected_msgs)
        """
        # Embed the message
        imgs_w = self.embed(imgs, msgs, is_video, mode, align_corners, antialias)
        
        # Detect the message
        if aggregate:
            assert is_video, "Aggregation is only supported for videos"
            preds = self.detect_video_and_aggregate(imgs_w, "avg", mode, align_corners, antialias)
        else:
            preds = self.detect(imgs_w, is_video, mode, align_corners, antialias)
        
        return imgs_w, preds

    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
        is_video: bool = False,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        """
        Embed a message into images or video frames.
        
        Args:
            imgs: Input images or video frames
            msgs: Optional messages to embed
            is_video: Whether the input is a video
            mode: Interpolation mode
            align_corners: Whether to align corners in interpolation
            antialias: Whether to use antialiasing in interpolation
            
        Returns:
            dict: A dictionary containing the watermarked images and embedded messages
        """
        if is_video:
            return self.embed_video(imgs, msgs, mode, align_corners, antialias)
        else:
            return self.embed_img(imgs, msgs, mode, align_corners, antialias)

    def detect(
        self,
        imgs: torch.Tensor,
        is_video: bool = False,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        """
        Detect messages from watermarked images or video frames.
        
        Args:
            imgs: Input images or video frames
            is_video: Whether the input is a video
            mode: Interpolation mode
            align_corners: Whether to align corners in interpolation
            antialias: Whether to use antialiasing in interpolation
            
        Returns:
            dict: A dictionary containing the detected messages
        """
        if is_video:
            return self.detect_video(imgs, mode, align_corners, antialias)
        else:
            return self.detect_img(imgs, mode, align_corners, antialias)

    def embed_img(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        """
        Generates watermarked images from the input images and messages (used for inference).
        Images may be arbitrarily sized.
        
        Args:
            imgs (torch.Tensor): Batched images with shape BxCxHxW.
            msgs (torch.Tensor): Optional messages with shape BxK.
            mode: Interpolation mode
            align_corners: Whether to align corners in interpolation
            antialias: Whether to use antialiasing in interpolation
            
        Returns:
            dict: A dictionary with the watermarked images and embedded messages
        """
        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                     mode=mode, align_corners=align_corners, 
                                     antialias=antialias)
        
        # generate watermarked images
        if self.embedder.yuv:  # take y channel only
            preds_w = self.embedder(
                self.rgb2yuv(imgs_res)[:, 0:1],
                msgs
            )
        else:
            preds_w = self.embedder(imgs_res, msgs)

        # attenuate at low resolution if needed
        if self.do_attenuation and self.lowres_attenuation:
            hmaps = self.attenuation.heatmaps(imgs_res)
            preds_w = hmaps * preds_w

        # interpolate back
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            preds_w = F.interpolate(preds_w, size=imgs.shape[-2:],
                                    mode=mode, align_corners=align_corners, 
                                    antialias=antialias)
        
        # apply attenuation
        if self.do_attenuation and not self.lowres_attenuation:
            hmaps = self.attenuation.heatmaps(imgs)
            preds_w = hmaps * preds_w

        # blend and clamp
        imgs_w = self.blender(imgs, preds_w)
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)

        return imgs_w

    def embed_video(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        """
        Generates watermarked videos from the input images and messages (used for inference).
        Videos may be arbitrarily sized.
        
        Args:
            imgs (torch.Tensor): Video frames with shape FxCxHxW.
            msgs (torch.Tensor): Optional messages with shape 1xK.
            mode: Interpolation mode
            align_corners: Whether to align corners in interpolation
            antialias: Whether to use antialiasing in interpolation
            
        Returns:
            dict: A dictionary with the watermarked frames and embedded messages
        """
        assert msgs.shape[0] == 1, "Message should be unique"
        msgs = msgs.repeat(self.chunk_size, 1)  # 1 k -> n k

        # encode by chunk of cksz imgs, propagate the wm to spsz next imgs
        chunk_size = self.chunk_size  # n=cksz
        step_size = self.step_size  # spsz

        # initialize watermarked imgs
        imgs_w = torch.zeros_like(imgs)  # f 3 h w

        # chunking is necessary to avoid memory issues (when too many frames)
        for ii in range(0, len(imgs[::step_size]), chunk_size):
            nimgs_in_ck = min(chunk_size, len(imgs[::step_size]) - ii)
            start = ii * step_size
            end = start + nimgs_in_ck * step_size
            all_imgs_in_ck = imgs[start: end, ...]  # f 3 h w

            # deal with last chunk that may have less than chunk_size imgs
            if nimgs_in_ck < chunk_size:
                msgs_chunk = msgs[:nimgs_in_ck]
            else:
                msgs_chunk = msgs  # Use full msgs

            # interpolate
            all_imgs_res = all_imgs_in_ck.clone()
            if all_imgs_res.shape[-2:] != (self.img_size, self.img_size):
                all_imgs_res = F.interpolate(all_imgs_res, size=(self.img_size, self.img_size),
                                            mode=mode, align_corners=align_corners, 
                                            antialias=antialias)

            # choose one frame every step_size
            imgs_res = all_imgs_res[::step_size]  # n 3 h w

            # get deltas for the chunk, and repeat them for each frame in the chunk
            if self.embedder.yuv:  # take y channel only
                imgs_res_yuv = self.rgb2yuv(imgs_res)[:, 0:1]
                preds_w = self.embedder(imgs_res_yuv, msgs_chunk)
            else:
                preds_w = self.embedder(imgs_res, msgs_chunk)
            
            # use _apply_video_mode to expand predictions based on video_mode
            preds_w = self._apply_video_mode(preds_w, len(all_imgs_in_ck), step_size, self.video_mode)

            # attenuate at low resolution if needed
            if self.attenuation is not None and self.lowres_attenuation:
                hmaps = self.attenuation.heatmaps(all_imgs_res)
                preds_w = hmaps * preds_w
            
            # interpolate back
            if all_imgs_in_ck.shape[-2:] != (self.img_size, self.img_size):
                preds_w = F.interpolate(preds_w, size=all_imgs_in_ck.shape[-2:],
                                        mode=mode, align_corners=align_corners, 
                                        antialias=antialias)

            # attenuate at full resolution if needed
            if self.attenuation is not None and not self.lowres_attenuation:
                hmaps = self.attenuation.heatmaps(all_imgs_in_ck)
                preds_w = hmaps * preds_w

            # blend
            all_imgs_in_ck_w = self.blender(all_imgs_in_ck, preds_w)
            imgs_w[start: end, ...] = all_imgs_in_ck_w  # n 3 h w

        # clamp
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)

        return imgs_w

    def _apply_video_mode(self, preds_w: torch.Tensor, total_frames: int, step_size: int, video_mode: str) -> torch.Tensor:
        """
        Applies the selected video mode to expand predictions across frames.
        
        Args:
            preds_w (torch.Tensor): Predictions for key frames [n, c, h, w]
            total_frames (int): Total number of frames to generate
            step_size (int): Number of frames between key frames
            video_mode (str): The video mode to use. Can be one of "alternate", "repeat", "interpolate"
            
        Returns:
            torch.Tensor: Expanded predictions [total_frames, c, h, w]
        """
        if video_mode == "repeat":
            # repeat each prediction for step_size frames
            preds_w = torch.repeat_interleave(preds_w, step_size, dim=0)  # f c h w
        elif video_mode == "alternate":
            # create a tensor of zeros and place predictions at intervals of step_size
            full_size = (total_frames,) + preds_w.shape[1:]  # f c h w
            full_preds = torch.zeros(full_size, device=preds_w.device)  # f c h w
            full_preds[::step_size] = preds_w  # place preds_w [n c h w] every step_size frames
            preds_w = full_preds  # f c h w
        elif video_mode == "interpolate":
            # interpolate between predictions
            full_size = (total_frames,) + preds_w.shape[1:]  # f c h w
            full_preds = torch.zeros(full_size, device=preds_w.device)  # f c h w
            # interpolation factors
            alpha = 1 - torch.linspace(0, 1, steps=step_size, device=preds_w.device)  # step_size
            alpha = alpha.repeat((total_frames-1) // step_size).view(-1, 1, 1, 1)  # (f-1)//step 1 1 1 1
            # key frames and shifted key frames
            start_frames = torch.repeat_interleave(preds_w[:-1], step_size, dim=0)  # (f-1)//step c h w
            end_frames = torch.repeat_interleave(preds_w[1:], step_size, dim=0)  # (f-1)//step c h w
            # interpolate between key frames and shifted
            interpolated_preds = alpha * start_frames + (1-alpha) * end_frames  # (f-1)//step c h w
            # fill the rest of the frames with the last ones
            last_start = len(interpolated_preds)
            full_preds[:last_start] = interpolated_preds
            full_preds[last_start:] = preds_w[-1]  # use last prediction for remaining frames
            preds_w = full_preds  # f c h w
        
        return preds_w[:total_frames]  # f c h w

    def detect_img(
        self,
        imgs: torch.Tensor,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        """
        Performs the forward pass of the detector only (used at inference).
        Rescales the input images to 256x256 pixels and then computes the mask and the message.
        
        Args:
            imgs (torch.Tensor): Batched images with shape BxCxHxW.
            mode: Interpolation mode
            align_corners: Whether to align corners in interpolation
            antialias: Whether to use antialiasing in interpolation
            
        Returns:
            dict: A dictionary containing the detected messages
        """
        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                     mode=mode, align_corners=align_corners, 
                                     antialias=antialias)
        
        # detect watermark
        preds = self.detector(imgs_res)

        return preds

    def detect_video(
        self,
        imgs: torch.Tensor,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        """
        Performs the forward pass of the detector only for videos.
        Rescales the input images to 256x256 pixels and then computes the mask and the message.
        
        Args:
            imgs (torch.Tensor): Video frames with shape FxCxHxW.
            mode: Interpolation mode
            align_corners: Whether to align corners in interpolation
            antialias: Whether to use antialiasing in interpolation
            
        Returns:
            dict: A dictionary containing the detected messages
        """
        all_preds = []
        for ii in range(0, len(imgs), self.chunk_size):
            nimgs_in_ck = min(self.chunk_size, len(imgs) - ii)
            preds = self.detect_img(
                imgs[ii:ii+nimgs_in_ck], 
                mode, 
                align_corners, 
                antialias
            )
            all_preds.append(preds)  # n k ..
        preds = torch.cat(all_preds, dim=0)  # f k ..
        return preds

    def detect_video_and_aggregate(
        self,
        imgs: torch.Tensor,
        aggregation: str = "avg",
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = False,
    ) -> torch.Tensor:
        """
        Detects the message in a video and aggregates the predictions across frames.
        
        Args:
            imgs (torch.Tensor): Video frames with shape FxCxHxW.
            aggregation (str, optional): Aggregation method. Defaults to "avg".
            mode: Interpolation mode
            align_corners: Whether to align corners in interpolation
            antialias: Whether to use antialiasing in interpolation
            
        Returns:
            torch.Tensor: Aggregated binary message
        """
        preds = self.detect_video(imgs, mode, align_corners, antialias)
        mask_preds = preds[:, 0:1]  # binary detection bit (not used for now)
        bit_preds = preds[:, 1:]  # f k .., must <0 for bit 0 and >0 for bit 1
        
        if aggregation is None:
            decoded_msg = bit_preds
        elif aggregation == "avg":
            decoded_msg = bit_preds.mean(dim=0)
        elif aggregation == "squared_avg":
            decoded_msg = (bit_preds * bit_preds.abs()).mean(dim=0)  # f k -> k
        elif aggregation == "l1norm_avg":
            frame_weights = torch.norm(bit_preds, p=1, dim=1).unsqueeze(1)  # f 1
            decoded_msg = (bit_preds * frame_weights).mean(dim=0)  # f k -> k
        elif aggregation == "l2norm_avg":
            frame_weights = torch.norm(bit_preds, p=2, dim=1).unsqueeze(1)  # f 1
            decoded_msg = (bit_preds * frame_weights).mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        msg = (decoded_msg > 0).squeeze().unsqueeze(0)  # 1 k
        return msg

def test_model_jit():
    import os
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    import matplotlib.pyplot as plt

    from videoseal.evals.full import setup_model_from_checkpoint
    from videoseal.models.wam_jit import WamJIT

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    file = "/private/home/pfz/_images/tahiti.png"
    imgs = Image.open(file, "r").convert("RGB")  # keep only rgb channels
    imgs = to_tensor(imgs).unsqueeze(0).float()

    # Generate random message
    msgs = torch.randint(0, 2, (1, 256))

    # Move to device
    imgs = imgs.to(device)
    msgs = msgs.to(device)

    # Load model from checkpoint
    ckpt = "/checkpoint/pfz/2025_logs/0306_vseal_ydisc_release_bis/_nbits=256/checkpoint600.pth"

    # Setup model from checkpoint and configure
    wam = setup_model_from_checkpoint(ckpt)
    wam.round = False
    wam.blender.scaling_w = 0.2
    wam.eval()
    wam.to("cpu")

    # Create JIT version
    model_jit = WamJIT(
        wam.embedder, 
        wam.detector, 
        wam.attenuation, 
        scaling_w=0.2, 
        scaling_i=1.0
    )
    model_jit.eval()
    model_jit.to("cpu")

    # Script the model
    print("Converting model to TorchScript...")
    model_jit = torch.jit.script(model_jit)
    model_jit.save(f"{output_dir}/y_256b_img.pt")
    print(f"Saved model to {output_dir}/y_256b_img.pt")
    print(f"Size of the model: {os.path.getsize(f'{output_dir}/y_256b_img.pt') / 1e6:.2f} MB")

    # Test - Compare original and JIT models
    print("Testing model...")
    wam.to(device)
    model_jit = torch.jit.load(f"{output_dir}/y_256b_img.pt")
    model_jit.to(device)

    # Run original and jit model
    imgs_w = wam.embed(imgs, msgs, lowres_attenuation=True)["imgs_w"]
    imgs_w_jit = model_jit.embed(imgs, msgs)

    # Compare results
    diff = (imgs_w - imgs_w_jit).abs()
    print(f"Mean absolute difference: {diff.mean().item()}")
    print(f"Max absolute difference: {diff.max().item()}")

    if diff.mean().item() < 1e-5:
        print("✅ Test passed: JIT model produces the same outputs as the original model")
    else:
        print("❌ Test failed: JIT model produces different outputs than the original model")

    # plot the diff between the original and JIT model
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(imgs_w_jit.detach().squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.title("jitted")
    plt.subplot(1, 3, 2)
    plt.imshow(imgs_w.detach().squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.title("original")
    plt.subplot(1, 3, 3)
    diff = (imgs_w - imgs_w_jit).abs() * 25
    plt.imshow(diff.detach().squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.title("diff")
    plt.show()

    # compare detection results
    preds = wam.detect(imgs_w, is_video=False)["preds"]
    preds_jit = model_jit.detect(imgs_w, is_video=False)

    # orint diff between preds_jit and preds
    diff = (preds - preds_jit).abs()
    print(f"Mean absolute difference in predictions: {diff.mean().item()}")
    print(f"Max absolute difference in predictions: {diff.max().item()}")
    if diff.mean().item() < 1e-5:
        print("✅ Test passed: JIT model produces the same predictions as the original model")
    else:
        print("❌ Test failed: JIT model produces different predictions than the original model")


if __name__ == "__main__":
    test_model_jit()
