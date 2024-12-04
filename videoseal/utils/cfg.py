# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import omegaconf
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf

from videoseal.augmentation import get_validation_augs
from videoseal.augmentation.augmenter import get_dummy_augmenter
from videoseal.data.datasets import (CocoImageIDWrapper, ImageFolder,
                                     VideoDataset)
from videoseal.models import Videoseal, build_embedder, build_extractor
from videoseal.modules.jnd import JND
from videoseal.utils.data import Modalities, parse_dataset_params

omegaconf.OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
omegaconf.OmegaConf.register_new_resolver("add", lambda x, y: x + y)

@dataclass
class SubModelConfig:
    """Configuration for a sub-model."""
    model: str
    params: DictConfig


@dataclass
class VideosealConfig:
    """Configuration for a VideoWAM model."""
    args: DictConfig
    embedder: SubModelConfig
    extractor: SubModelConfig


def get_config_from_checkpoint(ckpt_path: Path) -> VideosealConfig:
    """
    Load configuration from a checkpoint file.

    Args:
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    VideosealConfig: Loaded configuration.
    """
    exp_dir, exp_name = os.path.dirname(ckpt_path).rsplit('/', 1)
    logfile_path = os.path.join(exp_dir, 'logs', exp_name + '.stdout')

    with open(logfile_path, 'r') as file:
        for line in file:
            if '__log__:' in line:
                params = json.loads(line.split('__log__:')[1].strip())
                break

    args = OmegaConf.create(params)
    if not isinstance(args, DictConfig):
        raise Exception("Expected logfile to contain params dictionary.")

    # Load sub-model configurations
    embedder_cfg = OmegaConf.load(args.embedder_config)
    extractor_cfg = OmegaConf.load(args.extractor_config)

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
    Set up a VideoWAM model from a configuration and checkpoint file.

    Args:
    config (VideosealConfig): Model configuration.
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    Videoseal: Loaded model.
    """
    args = config.args

    # Build models
    embedder = build_embedder(config.embedder.model, config.embedder.params, args.nbits)
    extractor = build_extractor(config.extractor.model, config.extractor.params, args.img_size_extractor, args.nbits)
    augmenter = get_dummy_augmenter()  # does nothing

    # Build attenuation
    attenuation = None
    if args.attenuation.lower() != "none":
        attenuation_cfg = OmegaConf.load(args.attenuation_config)
        attenuation = JND(**attenuation_cfg[args.attenuation])

    # Build the complete model
    wam = Videoseal(
        embedder,
        extractor,
        augmenter,
        attenuation=attenuation,
        scaling_w=args.scaling_w,
        scaling_i=args.scaling_i,
        img_size=args.img_size,
        chunk_size=args.videowam_chunk_size,
        step_size=args.videowam_step_size
    )

    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        msg = wam.load_state_dict(checkpoint['model'], strict=False)
        print(f"Model loaded successfully from {ckpt_path} with message: {msg}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    return wam


def setup_model_from_checkpoint(ckpt_path: Path) -> Videoseal:
    """
    Set up a VideoWAM model from a checkpoint file.

    Args:
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    Videoseal: Loaded model.
    """
    config = get_config_from_checkpoint(ckpt_path)
    return setup_model(config, ckpt_path)

def setup_model_from_model_card(model_card: Path | str) -> Videoseal:
    """
    Set up a VideoWAM model from a model card YAML file.


    Args:
    model_card (Path | str): Path to the model card YAML file or name of the model card.


    Returns:
    Videoseal: Loaded model.
    """

    # Get the path of the videoseal package
    videoseal_path = Path(importlib.util.find_spec('videoseal').origin).parent

    # Define the cards directory as a subdirectory of the videoseal package
    cards_dir = videoseal_path / 'cards'

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

    elif str(config.checkpoint_path).startswith("https://huggingface.co/facebook/videoseal/"):
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
            repo_id="facebook/videoseal",  # The repository ID
            filename=fname  # Dynamically determined filename
        )
        
    else:
        raise RuntimeError(f"Path or uri {config.checkpoint_path} is unknown or does not exist")

    return setup_model(config, ckpt_path)

def setup_dataset(args):
    try:
        dataset_config = omegaconf.OmegaConf.load(f"configs/datasets/{args.dataset}.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset configuration not found: {args.dataset}")
    if args.is_video:
        # Video dataset, with optional masks
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