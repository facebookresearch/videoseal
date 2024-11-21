import base64
import io
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import triton_python_backend_utils as pb_utils  # type: ignore
from decord import VideoReader, cpu

from videoseal.demo.io import DemoVideoBatchReader
from videoseal.demo.triton import TritonPythonModelBase
from videoseal.demo.utils import (
    decode_message_tensor,
    encode_message_tensor,
    load_video_wam,
)
from videoseal.evals.metrics import bit_accuracy


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

        self.wam, _ = load_video_wam(config_path, checkpoint_path)
        self.wam.eval()
        self.wam.to(self.device)

        self.bit_accuracy_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "bit_accuracy"
            )["data_type"]
        )
        self.message_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "message"
            )["data_type"]
        )

    def execute(self, requests):
        return [self._process_request(request) for request in requests]

    @torch.inference_mode()
    @torch.no_grad()
    def _process_request(self, request):
        video_b64 = self.get_singleton_input(request, "video")
        key = self.get_singleton_input(request, "key")
        num_pixels_per_batch = self.get_singleton_input(
            request, "num_pixels_per_batch"
        ) or (8 * 1024 * 1024)
        num_video_reader_threads = (
            self.get_singleton_input(request, "num_video_reader_threads") or 8
        )
        if (
            video_b64 is None
            or not isinstance(num_pixels_per_batch, int)
            or not isinstance(num_video_reader_threads, int)
            or not isinstance(key, bytes)
        ):
            return

        key_str = key.decode("utf-8")
        key = (
            encode_message_tensor(key_str, "", nbits=len(key_str))
            .to(self.device)
            .unsqueeze(0)
        )

        with ExitStack() as stack:
            video_io = stack.push(io.BytesIO(base64.b64decode(video_b64)))
            video_reader = VideoReader(
                video_io, num_threads=num_video_reader_threads, ctx=cpu()
            )
            height, width = video_reader[0].shape[:2]
            batch_size = math.floor(num_pixels_per_batch / (height * width))

            bit_accuracies = []
            message_preds = []
            for frames in DemoVideoBatchReader(video_reader, batch_size=batch_size):
                frames = torch.from_numpy(frames).to(self.device)
                bit_accuracy, message_pred = self._detect(frames, key)
                bit_accuracies.append(bit_accuracy)
                message_preds.append(message_pred)

        bit_accuracy = np.mean(bit_accuracies)
        _, message = decode_message_tensor(
            torch.stack(message_preds).nanmean(0).round().int(), nkeybits=0
        )

        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "bit_accuracy",
                    np.array([bit_accuracy], dtype=self.bit_accuracy_dtype),
                ),
                pb_utils.Tensor(
                    "message",
                    np.array([message], dtype=self.message_dtype),
                ),
            ]
        )

    def _detect(
        self, frames: torch.Tensor, key: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        normalized_frames = frames.float().permute(0, 3, 1, 2) / 255.0
        outputs = self.wam.detect(normalized_frames, is_video=True)
        preds = outputs["preds"]
        key_preds = preds[:, 1 : (key.shape[1] + 1)]
        message_preds = preds[:, (key.shape[1] + 1) :]
        keys = key.repeat(len(frames), 1)
        return (
            bit_accuracy(key_preds, keys).nanmean().item(),
            (message_preds > 0).float().nanmean(dim=0),
        )
