import base64
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils  # type: ignore

from videoseal.data.datasets import VideoDataset
from videoseal.demo.triton import TritonPythonModelBase
from videoseal.demo.utils import load_video_wam
from videoseal.utils.display import get_fps, save_vid


class TritonPythonModel(TritonPythonModelBase):
    def initialize(self, args):
        super().initialize(args)

        checkpoints_dir = Path(
            args["model_repository"],
            args["model_version"],
            "checkpoints",
        )
        config_path = checkpoints_dir / "unet.yaml"
        checkpoint_path = checkpoints_dir / "unet.pt"

        self.wam = load_video_wam(config_path, checkpoint_path)
        self.wam.eval()
        self.wam.to(self.device)

        self.watermarked_video_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "watermarked_video"
            )["data_type"]
        )

    def execute(self, requests):
        return [self._process_request(request) for request in requests]

    @torch.inference_mode()
    @torch.no_grad()
    def _process_request(self, request):
        video_b64 = self.get_singleton_input(request, "video")
        if video_b64 is None:
            return

        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
            temp_video.write(base64.b64decode(video_b64))
            temp_video.seek(0)
            fps, _ = get_fps(temp_video.name)
            vid, _ = VideoDataset.load_full_video_decord(temp_video.name)
            if isinstance(vid, list):
                return

        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_watermarked_video:
            outputs = self.wam.embed(vid, is_video=True)
            imgs_w = outputs["imgs_w"].cpu()  # type: ignore TODO: Fix VideoWam.embed type hints
            save_vid(imgs_w, temp_watermarked_video.name, math.floor(fps))
            temp_watermarked_video.seek(0)
            watermarked_video_b64 = base64.b64encode(
                temp_watermarked_video.read()
            ).decode("utf-8")
            return pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "watermarked_video",
                        np.array(
                            [watermarked_video_b64], dtype=self.watermarked_video_dtype
                        ),
                    )
                ]
            )
