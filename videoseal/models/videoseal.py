"""
Test with:
    python -m videoseal.models.videoseal
"""

import torch

from videoseal.augmentation.augmenter import Augmenter
from videoseal.models.embedder import Embedder
from videoseal.models.extractor import Extractor
from videoseal.models.wam import Wam
from videoseal.modules.jnd import JND


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
        blending_method: str = "additive"
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

    def video_embedder(
        self,
        imgs: torch.Tensor,
        msg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generates deltas one every step_size frames, then repeats for the next step_size frames.
        """
        # TODO: deal with the case where the embedder predicts images instead of deltas
        msg = msg.repeat(len(imgs) // self.step_size, 1)  # n k
        preds_w = self.embedder(imgs[::self.step_size], msg)  # n 3 h w
        preds_w = torch.repeat_interleave(
            preds_w, self.step_size, dim=0)
        return preds_w[:len(imgs)]  # f 3 h w

    def video_forward(
        self,
        imgs: torch.Tensor,  # [frames, c, h, w] for a single video
        masks: torch.Tensor,
        msgs: torch.Tensor = None,  # 1 message per video
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
        # generate watermarked images
        if self.embedder.yuv:  # take y channel only
            preds_w = self.video_embedder(self.rgb2yuv(imgs)[:, 0:1], msgs)
        else:
            preds_w = self.video_embedder(imgs, msgs)
        imgs_w = self.blend(imgs, preds_w)  # frames c h w
        # augment
        imgs_aug, masks, selected_aug = self.augmenter(
            imgs_w, imgs, masks, is_video=True)
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
    ) -> dict:
        """ 
        Generates watermarked videos from the input images and messages (used for inference).
        Videos may be arbitrarily sized.
        """
        if not is_video:
            # fallback on parent class for batch of images
            return super().embed(imgs, msgs)
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

            # choose one frame every step_size
            imgs_in_ck = all_imgs_in_ck[::step_size]  # n 3 h w

            # deal with last chunk that may have less than chunk_size imgs
            if nimgs_in_ck < chunk_size:
                msgs = msgs[:nimgs_in_ck]

            # get deltas for the chunk, and repeat them for each frame in the chunk
            outputs = super().embed(imgs_in_ck, msgs)  # n 3 h w
            deltas_in_ck = outputs["preds_w"]  # n 3 h w
            deltas_in_ck = torch.repeat_interleave(
                deltas_in_ck, step_size, dim=0)  # f 3 h w

            # at the end of video there might be more deltas than needed
            deltas_in_ck = deltas_in_ck[:len(all_imgs_in_ck)]

            # create watermarked imgs
            all_imgs_in_ck_w = self.blend(all_imgs_in_ck, deltas_in_ck)
            imgs_w[start: end, ...] = all_imgs_in_ck_w  # n 3 h w

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
    ) -> dict:
        """
        Performs the forward pass of the detector only.
        Rescales the input images to 256x... pixels and then computes the mask and the message.
        Args:
            imgs (torch.Tensor): Batched images with shape FxCxHxW, where F is the number of frames,
                                    C is the number of channels, H is the height, and W is the width. 
                                    if shape CxHxW is passed automatically will be considered as img
        Returns:
            dict: The output predictions.
                - torch.Tensor: Predictions for each frame with shape Fx(K+1),
                                where K is the length of the binary message. The first column represents
                                the probability of the detection bit, and the remaining columns represent
                                the probabilities of each bit in the message.
        """
        if not is_video or len(imgs.shape) == 3:
            # fallback on parent class for batch of images
            return super().detect(imgs)
        all_preds = []
        for ii in range(0, len(imgs), self.chunk_size):
            nimgs_in_ck = min(self.chunk_size, len(imgs) - ii)
            outputs = super().detect(
                imgs[ii:ii+nimgs_in_ck]
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
        aggregation: str = "avg"
    ) -> torch.Tensor:
        """
        Detects the message in a video and aggregates the predictions across frames.
        This method is mainly used for downstream inference to simplify the interface.
        If you want to obtain normal probabilities, use `video_detect` instead.
        Args:
            imgs (torch.Tensor): Batched images with shape FxCxHxW, where F is the number of frames,
                    C is the number of channels, H is the height, and W is the width.
            aggregation (str, optional): Aggregation method. Can be one of "avg",
                "squared_avg", etc. or None. Defaults to "avg".
        Returns:
            torch.Tensor: Aggregated binary message with shape k,
                where k is the length of the message.
        Note:
            If aggregation is None, returns the predictions for each frame without aggregation.
        """
        outputs = self.detect(imgs, is_video=True)
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
        msg = (decoded_msg > 0).squeeze().unsqueeze(0).to(int)  # 1 k
        return msg


if __name__ == "__main__":
    pass
