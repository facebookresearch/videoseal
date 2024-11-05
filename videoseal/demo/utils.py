from pathlib import Path

from omegaconf import OmegaConf

from videoseal.evals.full import VideoWamConfig, setup_model
from videoseal.models.video_wam import VideoWam


def load_video_wam(config_path: Path, checkpoint_path: Path) -> VideoWam:
    config: VideoWamConfig = OmegaConf.structured(OmegaConf.load(config_path))
    return setup_model(config, checkpoint_path)
