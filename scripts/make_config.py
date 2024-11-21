import argparse
from pathlib import Path
from videoseal.evals.full import get_config_from_checkpoint
from omegaconf import OmegaConf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="make_config", description="Make model config from checkpoint path."
    )
    parser.add_argument("checkpoint_path")
    parser.add_argument("-o", "--output-path", default="model.yaml")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    output_path = Path(args.output_path)
    config = get_config_from_checkpoint(checkpoint_path)
    with open(output_path, 'w') as file:
      file.write(OmegaConf.to_yaml(config))