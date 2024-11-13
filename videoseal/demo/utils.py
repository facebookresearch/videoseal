from pathlib import Path
from typing import Tuple

import torch
from omegaconf import OmegaConf

from videoseal.evals.full import VideoWamConfig, setup_model
from videoseal.models.video_wam import VideoWam


def load_video_wam(
    config_path: Path, checkpoint_path: Path
) -> Tuple[VideoWam, VideoWamConfig]:
    config: VideoWamConfig = OmegaConf.structured(OmegaConf.load(config_path))
    return (setup_model(config, checkpoint_path), config)


def encode_message_tensor(key: str, message: str, nbits: int = 64) -> torch.Tensor:
    combined = key + message
    if len(combined) > nbits:
        raise ValueError(
            f"The key and message encoded to {len(combined)} bits. Only {nbits} bits are allowed."
        )
    combined += "0" * (nbits - len(combined))
    return torch.tensor([int(bit) for bit in combined])


def decode_message_tensor(bits: torch.Tensor, nkeybits: int = 28) -> Tuple[str, str]:
    bits_np = bits.cpu().numpy()
    key = "".join([str(bit) for bit in bits_np[:nkeybits]])
    message = "".join([str(bit) for bit in bits_np[nkeybits:]])
    return (key, message)
