# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import requests
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import omegaconf
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf

import videoseal.utils.dist as udist
from videoseal.augmentation.augmenter import get_dummy_augmenter
from videoseal.data.datasets import CocoImageIDWrapper, ImageFolder, VideoDataset, SimpleVideoDataset
from videoseal.models import Videoseal, build_embedder, build_extractor, build_baseline
from videoseal.modules.jnd import JND

DEFAULT_CARD = 'videoseal_1.0'


@dataclass
class SubModelConfig:
    """Configuration for a sub-model."""
    model: str
    params: DictConfig


@dataclass
class VideosealConfig:
    """Configuration for a Video Seal model."""
    args: DictConfig
    embedder: SubModelConfig
    extractor: SubModelConfig


def resolve_config_path(cfg_path):
    # Resolve the config path in the following order:
    # 1. Search from the working directory
    # 2. Search from the package directory (for pip-installed packages)
    if not Path(cfg_path).is_file():
        # Use parents[1] to resolve relative to the videoseal package root
        cfg_path = Path(__file__).parents[1].joinpath(cfg_path)

    return cfg_path


def get_config_from_checkpoint(ckpt_path: Path) -> VideosealConfig:
    """
    Load configuration from a checkpoint file.

    Args:
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    VideosealConfig: Loaded configuration.
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    args = checkpoint['args']
    args = OmegaConf.create(args)

    if not isinstance(args, DictConfig):
        raise Exception("Expected logfile to contain params dictionary.")

    # Load sub-model configurations
    embedder_cfg_path = resolve_config_path(args.embedder_config)
    embedder_cfg = OmegaConf.load(embedder_cfg_path)
    extractor_cfg_path = resolve_config_path(args.extractor_config)
    extractor_cfg = OmegaConf.load(extractor_cfg_path)

    # Create sub-model configurations
    embedder_model = args.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[embedder_model]
    extractor_model = args.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[extractor_model]

    return VideosealConfig(
        args=args,
        embedder=SubModelConfig(model=embedder_model, params=embedder_params),
        extractor=SubModelConfig(model=extractor_model, params=extractor_params),
    )


def setup_model(config: VideosealConfig, ckpt_path: Path) -> Videoseal:
    """
    Set up a Video Seal model from a configuration and checkpoint file.

    Args:
    config (VideosealConfig): Model configuration.
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    Videoseal: Loaded model.
    """
    args = config.args

    # prepare some args for backward compatibility
    if "img_size_proc" in args:
        args.img_size = args.img_size_proc
    else:
        args.img_size = args.img_size_extractor

    if "hidden_size_multiplier" in args:
        args.hidden_size_multiplier = args.hidden_size_multiplier
    else:
        args.hidden_size_multiplier = 2

    # Handle backward compatibility for videowam_* parameter names (from 1210-oss branch)
    # If old parameter names exist (videowam_chunk_size, videowam_step_size), 
    # use them to set the new names (videoseal_chunk_size, videoseal_step_size)
    if "videowam_chunk_size" in args and "videoseal_chunk_size" not in args:
        args.videoseal_chunk_size = args.videowam_chunk_size
    if "videowam_step_size" in args and "videoseal_step_size" not in args:
        args.videoseal_step_size = args.videowam_step_size

    # Build models
    embedder = build_embedder(config.embedder.model, config.embedder.params, args.nbits, args.hidden_size_multiplier)
    extractor = build_extractor(config.extractor.model, config.extractor.params, args.img_size, args.nbits)
    augmenter = get_dummy_augmenter()  # does nothing

    # Build attenuation
    if args.attenuation.lower().startswith("jnd"):
        attenuation_cfg_path = resolve_config_path(args.attenuation_config)
        attenuation_cfg = omegaconf.OmegaConf.load(attenuation_cfg_path)
        attenuation = JND(**attenuation_cfg[args.attenuation])
    else:
        attenuation = None

    # Build the complete model
    wam = Videoseal(
        embedder,
        extractor,
        augmenter,
        attenuation=attenuation,
        scaling_w=args.scaling_w,
        scaling_i=args.scaling_i,
        img_size=args.img_size,
        chunk_size=args.videoseal_chunk_size,
        step_size=args.videoseal_step_size
    )

    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        msg = wam.load_state_dict(checkpoint['model'], strict=False)
        print(f"Model loaded successfully from {ckpt_path} with message: {msg}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    return wam

def setup_model_from_checkpoint(ckpt_path: str) -> Videoseal:
    """
    # Example usage
    ckpt_path = '/path/to/videoseal/checkpoint.pth'
    wam = setup_model_from_checkpoint(ckpt_path)

    or 
    ckpt_path = 'baseline/wam'
    wam = setup_model_from_checkpoint(ckpt_path)
    """
    # load baselines. Should be in the format of "baseline/{method}"
    if "baseline" in ckpt_path:
        method = ckpt_path.split('/')[-1]
        return build_baseline(method)
    # load model card - check if it's a model card name (not a path)
    # Model cards are YAML files in videoseal/cards/ directory
    elif not ckpt_path.endswith('.pth') and '/' not in ckpt_path:
        # This is likely a model card name (e.g., 'videoseal', 'chunkyseal', 'pixelseal')
        return setup_model_from_model_card(ckpt_path)
    # load videoseal ckpts
    else:
        config = get_config_from_checkpoint(ckpt_path)
        return setup_model(config, ckpt_path)


def setup_model_from_model_card(model_card: Path | str) -> Videoseal:
    """
    Set up a Video Seal model from a model card YAML file.
    Args:
        model_card (Path | str): Path to the model card YAML file or name of the model card.
    Returns:
        Videoseal: Loaded model.
    """
    # Use package-relative path for pip-installed packages
    cards_dir = Path(__file__).parent.parent / "cards"
    if model_card == 'videoseal':
        model_card = DEFAULT_CARD

    if isinstance(model_card, str):
        available_cards = [card.stem for card in cards_dir.glob('*.yaml')]
        if model_card not in available_cards:
            print(f"Available model cards: {', '.join(available_cards)}")
            raise FileNotFoundError(f"Model card '{model_card}' not found in {cards_dir}")
        model_card_path = cards_dir / f'{model_card}.yaml'
    elif isinstance(model_card, Path):
        if not model_card.exists():
            print(f"Available model cards: {', '.join([card.stem for card in cards_dir.glob('*.yaml')])}")
            raise FileNotFoundError(f"Model card file '{model_card}' not found")
        model_card_path = model_card
    else:
        raise TypeError("Model card must be a string or a Path object")

    with open(model_card_path, 'r') as file:
        config = OmegaConf.load(file)
    
    if Path(config.checkpoint_path).is_file():
        ckpt_path = Path(config.checkpoint_path)

    elif str(config.checkpoint_path).startswith("https://huggingface.co/facebook/video_seal/"):
        # Extract the filename from the URL
        import os
        checkpoint_url = str(config.checkpoint_path)
        
        # Handle URLs with or without 'resolve/main'
        if "/resolve/" in checkpoint_url:
            fname = os.path.basename(checkpoint_url.split("/resolve/", 1)[1])  # Extract after 'resolve/<branch>/'
        else:
            fname = os.path.basename(checkpoint_url)  # Extract the filename directly
        
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError:
            print(
                f"The model path {config.checkpoint_path} seems to be a direct HF path, "
                "but you do not have `huggingface_hub` installed. Install it with "
                "`pip install huggingface_hub` to use this feature."
            )
            raise
        
        # Download the checkpoint
        ckpt_path = hf_hub_download(
            repo_id="facebook/video_seal",  # The repository ID
            filename=fname  # Dynamically determined filename
        )

    elif is_url(config.checkpoint_path):
        if udist.is_dist_avail_and_initialized():
            # download only on the main process
            if udist.is_main_process():
                ckpt_path = maybe_download_checkpoint(config.checkpoint_path)
            dist.barrier()

        ckpt_path = maybe_download_checkpoint(config.checkpoint_path)
    else:
        raise RuntimeError(f"Path or uri {config.checkpoint_path} is unknown or does not exist")

    return setup_model(config, ckpt_path)


def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def maybe_download_checkpoint(url):
    try:
        # Extract path components to create unique filename
        # For URLs like .../videoseal/chunkyseal/checkpoint.pth, use chunkyseal_checkpoint.pth
        path_parts = urlparse(url).path.split("/")
        if len(path_parts) >= 2:
            # Use parent directory + filename to make it unique
            basename = f"{path_parts[-2]}_{path_parts[-1]}"
        else:
            basename = path_parts[-1]
        os.makedirs("ckpts", exist_ok=True)
        filename = os.path.abspath(os.path.join("ckpts", basename))

        if os.path.exists(filename):
            print(f"File {filename} exists, skipping download")
            return filename

        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File {url} downloaded successfully to {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

    return filename


def setup_dataset(args):
    try:
        dataset_config = omegaconf.OmegaConf.load(f"configs/datasets/{args.dataset}.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset configuration not found: {args.dataset}")
    if args.is_video:
        # Simple video dataset, intended for inference only
        if hasattr(args, "simple_video_dataset") and args.simple_video_dataset:
            dataset = SimpleVideoDataset(
                dataset_config.val_dir,
                args.short_edge_size
            )
        # Video dataset, with optional masks, intended for training
        else:
            dataset = VideoDataset(
                folder_paths = [dataset_config.val_dir],
                transform = None,
                output_resolution = args.short_edge_size,
                num_workers = 0,
                subsample_frames = False,
            )
        print(f"Video dataset loaded from {dataset_config.val_dir}")
    else:
        # Image dataset
        resize_short_edge = None
        if args.short_edge_size > 0:
            resize_short_edge = transforms.Resize(args.short_edge_size)
        if dataset_config.val_annotation_file:
            # COCO dataset, with masks
            dataset = CocoImageIDWrapper(
                root = dataset_config.val_dir,
                annFile = dataset_config.val_annotation_file,
                transform = resize_short_edge, 
                mask_transform = resize_short_edge
            )
        else:
            # ImageFolder dataset
            dataset = ImageFolder(
                path = dataset_config.val_dir,
                transform = resize_short_edge
            )  
        print(f"Image dataset loaded from {dataset_config.val_dir}")
    return dataset
