# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F

from ..augmentation.augmenter import Augmenter
from .embedder import Embedder
from .extractor import Extractor
from .wam import Wam
from ..modules.jnd import JND


class Videoseal(Wam):
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
        chunk_size (int, optional): The number of frames/imgs to encode at a time. Defaults to 8.
        step_size (int, optional): The number of frames/imgs to propagate the watermark to. Defaults to 4.
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
        clamp: bool = True,
        chunk_size: int = 8,
        step_size: int = 4,
        blending_method: str = "additive",
        video_mode: str = "repeat",
        lowres_attenuation: bool = False,
    ) -> None:
        """
        Initializes the Videoseal model.
        Args:
            embedder (Embedder): The watermark embedder.
            detector (Extractor): The watermark detector.
            augmenter (Augmenter): The image augmenter.
            attenuation (JND, optional): The JND model to attenuate the watermark distortion. Defaults to None.
            scaling_w (float, optional): The scaling factor for the watermark. Defaults to 1.0.
            scaling_i (float, optional): The scaling factor for the image. Defaults to 1.0.
            img_size (int, optional): The size of the frame to resize to intermediately while generating the watermark then upscale, the final video / image size is kept the same. Defaults to 256.
            chunk_size (int, optional): The number of frames/imgs to encode at a time. Defaults to 8.
            step_size (int, optional): The number of frames/imgs to propagate the watermark to. Defaults to 4.
            video_mode (str, optional): The mode to use for video watermarking. Can be one of "alternate", "repeat", "interpolate".
        """
        super().__init__(
            embedder=embedder,
            detector=detector,
            augmenter=augmenter,
            attenuation=attenuation,
            scaling_w=scaling_w,
            scaling_i=scaling_i,
            img_size=img_size,
            clamp=clamp,
            blending_method=blending_method
        )
        # video settings
        self.chunk_size = chunk_size  # encode 8 frames/imgs at a time
        self.step_size = step_size  # propagate the wm to 4 next frame/img
        self.video_mode = video_mode  # repeat, alternate or interpolate
        self.lowres_attenuation = lowres_attenuation

    @staticmethod
    def _apply_video_mode(preds_w: torch.Tensor, total_frames: int, step_size: int, video_mode: str) -> torch.Tensor:
        """
        applies the selected video mode to expand predictions across frames
        args:
            preds_w (torch.Tensor): predictions for key frames [n, c, h, w]
            total_frames (int): total number of frames to generate
            step_size (int): number of frames between key frames
            video_mode (str): the video mode to use. can be one of "alternate", "repeat", "interpolate"
        returns:
            torch.Tensor: expanded predictions [total_frames, c, h, w]
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
            alpha = alpha.repeat((total_frames-1) // step_size).view(-1, 1, 1, 1)  # (f-1)//step 1 1 1
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

    def forward(
        self,
        # [b, c, h, w] for batch of images or [b, frames, c, h, w] / [frames, c, h, w] for batch of videos
        imgs: torch.Tensor,
        masks: torch.Tensor,
        msgs: torch.Tensor = None,
        is_video: bool = True,
    ) -> dict:
        """
        Does the full forward pass of the WAM model (used for training).
        (1) Generates watermarked images from the input images and messages.
        (2) Augments the watermarked images.
        (3) Detects the watermark in the augmented images.
        Falls back to the parent class for batch of images.
        """
        assert not (is_video and len(imgs.shape) not in [4, 5]), \
            "If is_video is True, input shape should be [b, frames, c, h, w] or [frames, c, h, w]"
        assert not (not is_video and len(imgs.shape) != 4), \
            "If is_video is False, input shape should be [b, c, h, w]"

        if not is_video:
            # fallback on parent class for batch of images
            return super().forward(imgs, masks, msgs)

        if len(imgs.shape) == 5:
            # batch of videos, where each video is a sequence of frames (images)
            # imgs shape: [b, frames, c, h, w], where b is the batch size, frames is the number of frames in each video
            outputs = []
            for i in range(imgs.shape[0]):
                video_frames = imgs[i]  # [frames, c, h, w]
                video_masks = masks[i] if masks is not None else None
                video_msgs = msgs[i] if msgs is not None else None
                output = self.video_forward(
                    video_frames, video_masks, video_msgs)
                outputs.append(output)
            return outputs
        elif len(imgs.shape) == 4:
            # single video, represented as a sequence of frames (images)
            # imgs shape: [frames, c, h, w], where frames is the number of frames in the video
            return self.video_forward(imgs, masks, msgs)
        else:
            raise ValueError("Invalid input shape")

    def video_forward(
        self,
        imgs: torch.Tensor,  # [frames, c, h, w] for a single video
        masks: torch.Tensor,
        msgs: torch.Tensor = None,  # 1 message per video
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    ) -> dict:
        """
        Generate watermarked video from the input video imgs.
        """
        # create message 1 message per video but repeat for all frames
        # we need this to calcualte the loss
        if msgs is None:
            msgs = self.get_random_msg()  # 1 x k
        else:
            assert msgs.shape[0] == 1, "Message should be unique"
        msgs = msgs.to(imgs.device)
        msgs = msgs.expand(len(imgs), -1)  # frames k

        # interpolate to processing size
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                    **interpolation)

        # generate watermarked images
        if self.embedder.yuv:  # take y channel only
            key_frame_preds = self.embedder(
                self.rgb2yuv(imgs_res)[:, 0:1][::self.step_size], 
                msgs[::self.step_size]
            )
        else:
            key_frame_preds = self.embedder(imgs_res[::self.step_size], msgs[::self.step_size])

        # apply video mode to expand predictions across all frames
        preds_w = self._apply_video_mode(key_frame_preds, len(imgs_res), self.step_size, self.video_mode)

        # Handle attenuation based on the lowres_attenuation flag
        if self.lowres_attenuation and self.attenuation is not None:
            # Apply attenuation at low resolution
            self.attenuation.to(imgs_res.device)
            hmaps = self.attenuation.heatmaps(imgs_res)
            preds_w = hmaps * preds_w

            # interpolate predictions back to original size
            if imgs.shape[-2:] != (self.img_size, self.img_size):
                preds_w = F.interpolate(preds_w, size=imgs.shape[-2:],
                                        **interpolation)
            preds_w = preds_w.to(imgs.device)
            
            # blend with no additional attenuation
            imgs_w = self.blender(imgs, preds_w)  # frames c h w
        else:
            # interpolate predictions back to original size
            if imgs.shape[-2:] != (self.img_size, self.img_size):
                preds_w = F.interpolate(preds_w, size=imgs.shape[-2:],
                                        **interpolation)
            preds_w = preds_w.to(imgs.device)
            
            # blend
            imgs_w = self.blender(imgs, preds_w)  # frames c h w

            # apply attenuation at full resolution
            if self.attenuation is not None:
                self.attenuation.to(imgs.device)
                imgs_w = self.attenuation(imgs, imgs_w)

        # always apply clamping
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)

        # augment
        imgs_aug, masks, selected_aug = self.augmenter(
            imgs_w, imgs, masks, is_video=True, do_resize=False)

        # interpolate augmented images to processing size for detection
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_aug = F.interpolate(imgs_aug, size=(self.img_size, self.img_size),
                                    **interpolation)

        # detect watermark
        preds = self.detector(imgs_aug)

        # create and return outputs
        outputs = {
            # message per video but repeated for batchsize: b x k
            "msgs": msgs.expand(imgs.shape[0], -1),
            "masks": masks,  # augmented masks: frames 1 h w
            "imgs_w": imgs_w,  # watermarked imgs: frames c h w
            "imgs_aug": imgs_aug,  # augmented imgs: frames c h w
            "preds": preds,  # predicted message: 1 (1+nbits) h w
            "selected_aug": selected_aug,  # selected augmentation
        }
        return outputs

    @torch.no_grad()
    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor = None,
        is_video: bool = True,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
        lowres_attenuation: bool = False,
    ) -> dict:
        """ 
        Generates watermarked videos from the input images and messages (used for inference).
        Videos may be arbitrarily sized.
        """
        """ 
        Generates watermarked videos from the input images and messages (used for inference).
        Videos may be arbitrarily sized.
        """
        if not is_video:
            # fallback on parent class for batch of images
            return super().embed(imgs, msgs, interpolation, lowres_attenuation)
        if msgs is None:
            msgs = self.get_random_msg()  # 1 x k
        else:
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
                msgs = msgs[:nimgs_in_ck]

            # interpolate
            all_imgs_res = all_imgs_in_ck.clone()
            if all_imgs_res.shape[-2:] != (self.img_size, self.img_size):
                all_imgs_res = F.interpolate(all_imgs_res, size=(self.img_size, self.img_size),
                                            **interpolation)
            all_imgs_res = all_imgs_res.to(self.device)

            # choose one frame every step_size
            imgs_res = all_imgs_res[::step_size]  # n 3 h w

            # get deltas for the chunk, and repeat them for each frame in the chunk
            if self.embedder.yuv:  # take y channel only
                imgs_res = self.rgb2yuv(imgs_res)[:, 0:1]
            preds_w = self.embedder(imgs_res, msgs.to(self.device))
            
            # use _apply_video_mode to expand predictions based on video_mode
            preds_w = self._apply_video_mode(preds_w, len(all_imgs_in_ck), step_size, self.video_mode)

            # attenuate at low resolution if needed
            if self.attenuation is not None and lowres_attenuation:
                self.attenuation.to(all_imgs_res.device)
                hmaps = self.attenuation.heatmaps(all_imgs_res)
                preds_w = hmaps * preds_w
            
            # interpolate back
            preds_w = preds_w.to(imgs.device)
            if all_imgs_in_ck.shape[-2:] != (self.img_size, self.img_size):
                preds_w = F.interpolate(preds_w, size=all_imgs_in_ck.shape[-2:],
                                        **interpolation)

            # attenuate at full resolution if needed
            if self.attenuation is not None and not lowres_attenuation:
                self.attenuation.to(all_imgs_in_ck.device)
                hmaps = self.attenuation.heatmaps(all_imgs_in_ck)
                preds_w = hmaps * preds_w

            # blend
            all_imgs_in_ck_w = self.blender(all_imgs_in_ck, preds_w)
            imgs_w[start: end, ...] = all_imgs_in_ck_w  # n 3 h w

        # clamp
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)

        outputs = {
            "imgs_w": imgs_w,  # watermarked imgs: f 3 h w
            "msgs": msgs[0:1].repeat(len(imgs), 1),  # original messages: f k
        }
        return outputs
    
    @torch.no_grad()
    def detect(
        self,
        imgs: torch.Tensor,
        is_video: bool = True,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    ) -> dict:
        """
        Performs the forward pass of the detector only.
        Rescales the input images to 256x... pixels and then computes the mask and the message.
        Args:
            imgs (torch.Tensor): Batched images with shape FxCxHxW, where F is the number of frames,
                                    C is the number of channels, H is the height, and W is the width.
        Returns:
            dict: The output predictions.
                - torch.Tensor: Predictions for each frame with shape Fx(K+1),
                                where K is the length of the binary message. The first column represents
                                the probability of the detection bit, and the remaining columns represent
                                the probabilities of each bit in the message.
        """
        if not is_video:
            # fallback on parent class for batch of images
            return super().detect(imgs)
        all_preds = []
        for ii in range(0, len(imgs), self.chunk_size):
            nimgs_in_ck = min(self.chunk_size, len(imgs) - ii)
            outputs = super().detect(
                imgs[ii:ii+nimgs_in_ck], 
                interpolation
            )
            preds = outputs["preds"]
            all_preds.append(preds)  # n k ..
        preds = torch.cat(all_preds, dim=0)  # f k ..
        outputs = {
            "preds": preds,  # predicted masks and/or messages: f (1+nbits) h w
        }
        return outputs

    def extract_message(
        self,
        imgs: torch.Tensor,
        aggregation: str = "avg",
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": False},
    ) -> torch.Tensor:
        """
        Detects the message in a video and aggregates the predictions across frames.
        This method is mainly used for downstream inference to simplify the interface.
        If you want to obtain normal probabilities, use `video_detect` instead.
        Args:
            imgs (torch.Tensor): Batched images with shape FxCxHxW, where F is the number of frames,
                    C is the number of channels, H is the height, and W is the width.
            aggregation (str, optional): Aggregation method. Can be one of "avg",
                "weighted_avg", or None. Defaults to "avg".
        Returns:
            torch.Tensor: Aggregated binary message with shape K,
                where K is the length of the message.
        Note:
            If aggregation is None, returns the predictions for each frame without aggregation.
        """
        outputs = self.detect(imgs, is_video=True, interpolation=interpolation)
        preds = outputs["preds"]
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
        msg = (decoded_msg > 0).squeeze().unsqueeze(0)  # 1 k
        return msg


if __name__ == "__main__":
    pass
