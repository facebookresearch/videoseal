import base64
import io
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO, Tuple

import numpy as np
import torch
import triton_python_backend_utils as pb_utils  # type: ignore
from decord import VideoReader, cpu

from videoseal.demo.io import DemoVideoBatchReader, DemoVideoWriter
from videoseal.demo.triton import TritonPythonModelBase
from videoseal.demo.utils import (
    decode_message_tensor,
    encode_message_tensor,
    load_video_wam,
)


class TritonPythonModel(TritonPythonModelBase):
    def initialize(self, args):
        super().initialize(args)

        checkpoints_dir = Path(
            args["model_repository"],
            args["model_version"],
            "checkpoints",
        )
        config_path = checkpoints_dir / "model.yaml"
        checkpoint_path = checkpoints_dir / "model.pt"

        self.wam, self.config = load_video_wam(config_path, checkpoint_path)
        self.wam.eval()
        self.wam.to(self.device)

        self.key_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]), "key")[
                "data_type"
            ]
        )
        self.message_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "message"
            )["data_type"]
        )
        self.watermarked_video_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "watermarked_video"
            )["data_type"]
        )
        self.xray_video_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "xray_video"
            )["data_type"]
        )

    def execute(self, requests):
        return [self._process_request(request) for request in requests]

    @torch.inference_mode()
    @torch.no_grad()
    def _process_request(self, request):
        n_message_bits = self.config.args.nbits
        video_b64 = self.get_singleton_input(request, "video")
        key = self.get_singleton_input(request, "key") or self._random_binary_string(
            28
        ).encode("utf-8")
        message = self.get_singleton_input(request, "message") or (
            self._random_binary_string(n_message_bits - len(key)).encode("utf-8")
        )
        num_pixels_per_batch = self.get_singleton_input(
            request, "num_pixels_per_batch"
        ) or (8 * 1024 * 1024)
        num_video_reader_threads = (
            self.get_singleton_input(request, "num_video_reader_threads") or 8
        )
        if (
            video_b64 is None
            or not isinstance(key, bytes)
            or not isinstance(message, bytes)
            or not isinstance(num_pixels_per_batch, int)
            or not isinstance(num_video_reader_threads, int)
        ):
            return

        message_tensor = encode_message_tensor(
            key.decode("utf-8"), message.decode("utf-8"), nbits=n_message_bits
        )

        with ExitStack() as stack:
            video_io = stack.push(io.BytesIO(base64.b64decode(video_b64)))
            watermarked_video_io = stack.push(io.BytesIO())
            xray_video_io = stack.push(io.BytesIO())

            self._write_output_videos(
                video_io,
                watermarked_video_io,
                xray_video_io,
                message_tensor,
                num_pixels_per_batch,
                num_video_reader_threads,
            )

            watermarked_video_b64 = base64.b64encode(
                watermarked_video_io.getvalue()
            ).decode("utf-8")
            xray_video_b64 = base64.b64encode(xray_video_io.getvalue()).decode("utf-8")

        output_key, output_message = decode_message_tensor(message_tensor)

        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "watermarked_video",
                    np.array(
                        [watermarked_video_b64], dtype=self.watermarked_video_dtype
                    ),
                ),
                pb_utils.Tensor(
                    "xray_video",
                    np.array([xray_video_b64], dtype=self.xray_video_dtype),
                ),
                pb_utils.Tensor(
                    "key",
                    np.array([output_key], dtype=self.key_dtype),
                ),
                pb_utils.Tensor(
                    "message",
                    np.array([output_message], dtype=self.message_dtype),
                ),
            ]
        )

    def _write_output_videos(
        self,
        video_io: BinaryIO,
        watermarked_video_io: BinaryIO,
        xray_video_io: BinaryIO,
        message: torch.Tensor,
        num_pixels_per_batch: int,
        num_video_reader_threads: int,
    ):
        video_reader = VideoReader(
            video_io, num_threads=num_video_reader_threads, ctx=cpu()
        )
        fps = math.floor(video_reader.get_avg_fps())
        height, width = video_reader[0].shape[:2]
        batch_size = math.floor(num_pixels_per_batch / (height * width))
        with ExitStack() as stack:
            watermarked_video_writer = stack.push(
                DemoVideoWriter(
                    watermarked_video_io, fps=fps, width=width, height=height
                )
            )
            xray_video_writer = stack.push(
                DemoVideoWriter(xray_video_io, fps=fps, width=width, height=height)
            )
            for frames in DemoVideoBatchReader(video_reader, batch_size=batch_size):
                frames = torch.from_numpy(frames).to(self.device)
                watermarked_frames, xray_frames = self._embed(frames, message)
                watermarked_video_writer.write(watermarked_frames.cpu().numpy())
                xray_video_writer.write(xray_frames.cpu().numpy())

    def _embed(
        self, frames: torch.Tensor, message: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_frames = frames.float().permute(0, 3, 1, 2) / 255.0
        outputs = self.wam.embed(
            normalized_frames,
            msgs=message.unsqueeze(0),
            is_video=True,
        )
        normalized_watermarked_frames = outputs["imgs_w"]
        normalized_xray_frames = self._get_xray(
            normalized_frames, normalized_watermarked_frames
        )
        return (
            (normalized_watermarked_frames.permute(0, 2, 3, 1) * 255).to(torch.uint8),
            (normalized_xray_frames.permute(0, 2, 3, 1) * 255).to(torch.uint8),
        )

    def _get_xray(
        self, vid: torch.Tensor, watermarked_vid: torch.Tensor
    ) -> torch.Tensor:
        # Compute min and max values, reshape, and normalize
        vid_xray = torch.abs(watermarked_vid - vid)
        min_vals = (
            vid_xray.view(vid_xray.shape[0], vid_xray.shape[1], -1)
            .min(dim=2, keepdim=True)[0]
            .view(vid_xray.shape[0], vid_xray.shape[1], 1, 1)
        )
        max_vals = (
            vid_xray.view(vid_xray.shape[0], vid_xray.shape[1], -1)
            .max(dim=2, keepdim=True)[0]
            .view(vid_xray.shape[0], vid_xray.shape[1], 1, 1)
        )
        # Normalize in-place to save memory / time
        return vid_xray.subtract_(min_vals).divide_(max_vals.subtract_(min_vals))

    def _random_binary_string(self, length: int) -> str:
        return "".join([str(bit) for bit in np.random.randint(0, 2, length)])
