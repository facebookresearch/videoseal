"""
Example usage (cluster 2 gpus):
    torchrun --nproc_per_node=2 train.py --local_rank 0
Example usage (cluster 1 gpu):
    torchrun train.py --debug_slurm
    For eval ful only:
        torchrun train.py --debug_slurm --only_eval True --output_dir output/

Example:  decoding only, hidden like
    torchrun --nproc_per_node=2 train.py --local_rank 0 --nbits 32 --saveimg_freq 1 --lambda_i 0 --lambda_det 0 --lambda_dec 1 --lambda_d 0  --img_size 128 --img_size_extractor 128 --embedder_model hidden --extractor_model hidden

Args inventory:
    --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=100,warmup_lr_init=1e-6,warmup_t=5
    --optimizer Lamb,lr=1e-3
    --train_dir /checkpoint/pfz/projects/watermarking/data/coco_1k_orig
    --resume_from /checkpoint/pfz/2024_logs/0611_segmark_lpipsmse/_lambda_d=0.25_lambda_i=0.5_scaling_w=0.4/checkpoint050.pth
    --extractor_model dino2s_indices=11_upscale=1_14 --img_size 336 --batch_size 8 --extractor_config configs/extractor_dinos.yaml
    --embedder_model vae_sd --embedder_config configs/embedder_sd.yaml
    --local_rank 0  --only_eval True --scaling_w 0.4 --embedder_model vae_small --extractor_model sam_small --augmentation_config configs/all_augs.yaml --resume_from /checkpoint/pfz/2024_logs/0708_segmark_bigger_vae/_scaling_w=0.4_embedder_model=vae_small_extractor_model=sam_small/checkpoint.pth
    --local_rank 0  --only_eval True --scaling_w 2.0 --scaling_i 1.0 --nbits 16 --lambda_dec 6.0 --lambda_det 1.0 --lambda_d 0.0 --lambda_i 0.0 --perceptual_loss none --seed 0 --scheduler none --optimizer AdamW,lr=1e-5 --epochs 50 --batch_size_eval 32 --batch_size 16 --img_size 256 --attenuation jnd_1_3 --resume_from /checkpoint/pfz/2024_logs/0708_segmark_bigger_vae/_scaling_w=0.4_embedder_model=vae_small_extractor_model=sam_small/checkpoint.pth --embedder_model vae_small --extractor_model sam_small --augmentation_config configs/all_augs.yaml

"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from typing import List

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CocoDetection
from torchvision.utils import save_image

import videoseal.utils as utils
import videoseal.utils.dist as udist
import videoseal.utils.logger as ulogger
import videoseal.utils.optim as uoptim
from videoseal.augmentation.augmenter import Augmenter
from videoseal.augmentation.geometric import (Crop, HorizontalFlip, Identity,
                                              Perspective, Resize, Rotate)
from videoseal.augmentation.valuemetric import (JPEG, Brightness, Contrast,
                                                GaussianBlur, Hue,
                                                MedianFilter, Saturation)
from videoseal.data.loader import (get_dataloader, get_dataloader_segmentation,
                                   get_video_dataloader)
from videoseal.data.metrics import (accuracy, bit_accuracy,
                                    bit_accuracy_inference, iou, psnr)
from videoseal.data.transforms import (get_transforms,
                                       get_transforms_segmentation,
                                       normalize_img, unnormalize_img,
                                       unstd_img)
from videoseal.losses.detperceptual import LPIPSWithDiscriminator
from videoseal.models import Wam, build_embedder, build_extractor
from videoseal.modules.jnd import JND
from videoseal.utils.image import create_diff_img, detect_wm_hm

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


modality_to_datasets = {
    "image": ["coco"],
    "video": ["sa-v"]
}


def get_dataset_parser(parser):
    """
    Adds dataset-related arguments to the parser.
    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.
    """
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument("--modality", type=str, default="image",
                       choices=["image", "video"], help="Modality of the dataset. Options: 'image', 'video'")
    group.add_argument("--dataset", type=str, default="coco",
                       choices=[j for i in modality_to_datasets.values() for j in i], help="Name of the dataset. Options: 'coco', 'sa-v'. If provided, will override explicit directory paths.")
    group.add_argument("--train_dir", type=str, default=None,
                       help="Path to the training directory. Required if --dataset is not provided.")
    group.add_argument("--train_annotation_file", type=str, default=None,
                       help="Path to the training annotation file. Required for image modality if --dataset is not provided.")
    group.add_argument("--val_dir", type=str, default=None,
                       help="Path to the validation directory. Required if --dataset is not provided.")
    group.add_argument("--val_annotation_file", type=str, default=None,
                       help="Path to the validation annotation file. Required for image modality if --dataset is not provided.")
    return parser


def parse_dataset_params(params):
    """
    Parses the dataset parameters and loads the dataset configuration.

    Logic:
    1. If explicit directory paths are provided (--train_dir, --val_dir, etc.), use those.
    2. If a dataset name is provided (--dataset), load the corresponding configuration from configs/datasets/<dataset_name>.yaml.
    3. If neither explicit directory paths nor a dataset name is provided, raise an error.

    Args:
        params (argparse.Namespace): The parsed command-line arguments.


    Returns:
        omegaconf.DictConfig: The parsed and merged dataset configuration.
    """
    assert params.dataset in modality_to_datasets[
        params.modality], f"Invalid dataset '{params.dataset}' for modality '{params.modality}'"

    if params.train_dir is not None and params.val_dir is not None:
        # Use explicit directory paths
        print("Warning: Using explicitly provided train and val directories. Ignoring dataset name.")
        params_dict = vars(params)
    elif params.dataset is not None:
        # Load dataset configuration
        dataset_cfg = omegaconf.OmegaConf.load(
            f"configs/datasets/{params.dataset}.yaml")
        params_dict = vars(params)
        # Convert params_dict to OmegaConf object
        params_omega = omegaconf.OmegaConf.create(params_dict)
        # Merge params with dataset_cfg
        merged_cfg = omegaconf.OmegaConf.merge(params_omega, dataset_cfg)
        return merged_cfg
    else:
        # Raise an error if neither explicit directory paths nor a dataset name is provided
        raise ValueError(
            "Either provide dataset name or explicit train and val directories")

    if params_dict['modality'] == "image":
        # Check that annotation files are provided for image modality
        assert params_dict['train_annotation_file'] is not None and params_dict['val_annotation_file'] is not None, \
            "Annotation files are required for image modality"

    # Convert params_dict to OmegaConf object
    params_omega = omegaconf.OmegaConf.create(params_dict)
    return params_omega


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    # Dataset #
    parser = get_dataset_parser(parser)

    aa("--output_dir", type=str, default="output/",
       help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Config paths')
    aa("--embedder_config", type=str, default="configs/embedder.yaml",
       help="Path to the embedder config file")
    aa("--augmentation_config", type=str, default="configs/all_augs.yaml",
       help="Path to the augmentation config file")
    aa("--extractor_config", type=str, default="configs/extractor.yaml",
       help="Path to the extractor config file")
    aa("--attenuation_config", type=str, default="configs/attenuation.yaml",
       help="Path to the attenuation config file")
    aa("--embedder_model", type=str, default=None,
       help="Name of the extractor model")
    aa("--extractor_model", type=str, default=None,
       help="Name of the extractor model")

    group = parser.add_argument_group('Image and watermark parameters')
    aa("--nbits", type=int, default=32,
       help="Number of bits used to generate the message. If 0, no message is used.")
    aa("--img_size", type=int, default=256, help="Size of the input images")
    aa("--img_size_extractor", type=int,
       default=256, help="Images are resized to this size before being fed to the extractor")
    aa("--attenuation", type=str, default="None", help="Attenuation model to use")
    aa("--scaling_w", type=float, default=0.2,
       help="Scaling factor for the watermark in the embedder model")
    aa("--scaling_w_schedule", type=str, default=None,
       help="Scaling factor for the watermark in the embedder model")
    aa("--scaling_i", type=float, default=1.0,
       help="Scaling factor for the image in the embedder model")
    aa("--threshold_mask", type=float, default=0.6,
       help="Threshold for the mask prediction using heatmap only (default: 0.7)")

    group = parser.add_argument_group('Optimizer parameters')
    aa("--optimizer", type=str, default="AdamW,lr=1e-4",
       help="Optimizer (default: AdamW,lr=1e-4)")
    aa("--optimizer_d", type=str, default=None,
       help="Discriminator optimizer. If None uses the same params (default: None)")
    aa("--scheduler", type=str, default="None", help="Scheduler (default: None)")
    aa('--epochs', default=100, type=int, help='Number of total epochs to run')
    aa('--batch_size', default=16, type=int, help='Batch size')
    aa('--batch_size_eval', default=64, type=int, help='Batch size for evaluation')
    aa('--temperature', default=1.0, type=float,
       help='Temperature for the mask loss')
    aa('--workers', default=8, type=int, help='Number of data loading workers')
    aa('--resume_from', default=None, type=str,
       help='Path to the checkpoint to resume from')

    group = parser.add_argument_group('Losses parameters')
    aa('--lambda_det', default=0.0, type=float,
       help='Weight for the watermark detection loss')
    aa('--lambda_dec', default=4.0, type=float,
       help='Weight for the watermark decoding loss')
    aa('--lambda_i', default=1.0, type=float, help='Weight for the image loss')
    aa('--lambda_d', default=0.5, type=float,
       help='Weight for the discriminator loss')
    aa('--balanced', type=utils.bool_inst, default=True,
       help='If True, the weights of the losses are balanced')
    aa('--total_gnorm', default=0.0, type=float,
       help='Total norm for the adaptive weights. If 0, uses the norm of the biggest weight.')
    aa('--perceptual_loss', default='lpips', type=str,
       help='Perceptual loss to use. "lpips", "watson_vgg" or "watson_fft"')
    aa('--disc_start', default=0, type=float,
       help='Weight for the discriminator loss')
    aa('--disc_num_layers', default=2, type=int,
       help='Number of layers for the discriminator')

    group = parser.add_argument_group('Misc.')
    aa('--only_eval', type=utils.bool_inst,
       default=False, help='If True, only runs evaluate')
    aa('--eval_freq', default=5, type=int, help='Frequency for evaluation')
    aa('--full_eval_freq', default=50, type=int,
       help='Frequency for full evaluation')
    aa('--saveimg_freq', default=5, type=int, help='Frequency for saving images')
    aa('--saveckpt_freq', default=50, type=int, help='Frequency for saving ckpts')
    aa('--seed', default=0, type=int, help='Random seed')

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)

    return parser


def main(params):

    # Load Dataset Params
    params = parse_dataset_params(params)

    # Distributed mode
    udist.init_distributed_mode(params)

    # Set seeds for reproductibility
    seed = params.seed + udist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if params.distributed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(omegaconf.OmegaConf.to_yaml(params)))

    # Copy the config files to the output dir
    if udist.is_main_process():
        os.makedirs(os.path.join(params.output_dir, 'configs'), exist_ok=True)
        os.system(
            f'cp {params.embedder_config} {params.output_dir}/configs/embedder.yaml')
        os.system(
            f'cp {params.augmentation_config} {params.output_dir}/configs/augs.yaml')
        os.system(
            f'cp {params.extractor_config} {params.output_dir}/configs/extractor.yaml')

    # Build the embedder model
    embedder_cfg = omegaconf.OmegaConf.load(params.embedder_config)
    params.embedder_model = params.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[params.embedder_model]
    embedder = build_embedder(params.embedder_model,
                              embedder_params, params.nbits)
    # print(embedder)
    print(
        f'embedder: {sum(p.numel() for p in embedder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # build the augmenter
    augmenter_cfg = omegaconf.OmegaConf.load(params.augmentation_config)
    augmenter = Augmenter(
        **augmenter_cfg
    )
    print(f'augmenter: {augmenter}')

    # Build the extractor model
    extractor_cfg = omegaconf.OmegaConf.load(params.extractor_config)
    params.extractor_model = params.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[params.extractor_model]
    extractor = build_extractor(
        params.extractor_model, extractor_params, params.img_size_extractor, params.nbits)
    print(
        f'extractor: {sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # build attenuation
    if params.attenuation.lower() != "none":
        attenuation_cfg = omegaconf.OmegaConf.load(params.attenuation_config)
        attenuation = JND(**attenuation_cfg[params.attenuation],
                          preprocess=unnormalize_img, postprocess=normalize_img)
    else:
        attenuation = None
    print(f'attenuation: {attenuation}')

    # build the complete model
    wam = Wam(embedder, extractor, augmenter, attenuation,
              params.scaling_w, params.scaling_i)
    wam.to(device)
    # print(wam)

    # build losses
    image_detection_loss = LPIPSWithDiscriminator(
        balanced=params.balanced, total_norm=params.total_gnorm,
        disc_weight=params.lambda_d, percep_weight=params.lambda_i,
        detect_weight=params.lambda_det, decode_weight=params.lambda_dec,
        disc_start=params.disc_start, disc_num_layers=params.disc_num_layers,
        percep_loss=params.perceptual_loss
    ).to(device)
    print(image_detection_loss)
    # print(f"discriminator: {sum(p.numel() for p in image_detection_loss.discriminator.parameters() if p.requires_grad) / 1e3:.1f}K parameters")

    # Build the scaling schedule
    if params.scaling_w_schedule is not None:
        scaling_w_schedule = uoptim.parse_params(params.scaling_w_schedule)
        scaling_scheduler = uoptim.ScalingScheduler(
            obj=wam, attribute="scaling_w", scaling_o=params.scaling_w,
            **scaling_w_schedule
        )
    else:
        scaling_scheduler = None

    # Build optimizer and scheduler
    optim_params = uoptim.parse_params(params.optimizer)
    optimizer = uoptim.build_optimizer(
        model_params=list(embedder.parameters()) +
        list(extractor.parameters()),
        **optim_params
    )
    scheduler_params = uoptim.parse_params(params.scheduler)
    scheduler = uoptim.build_lr_scheduler(
        optimizer=optimizer, **scheduler_params)
    print('optimizer: %s' % optimizer)
    print('scheduler: %s' % scheduler)

    # discriminator optimizer
    optim_params_d = uoptim.parse_params(
        params.optimizer) if params.optimizer_d is None else uoptim.parse_params(params.optimizer_d)
    optimizer_d = uoptim.build_optimizer(
        model_params=[*image_detection_loss.discriminator.parameters()],
        **optim_params_d
    )
    print('optimizer_d: %s' % optimizer_d)

    # Data loaders
    train_transform, train_mask_transform, val_transform, val_mask_transform = get_transforms_segmentation(
        params.img_size)

    if params.modality == "image":
        train_loader = get_dataloader_segmentation(params.train_dir, params.train_annotation_file,
                                                   transform=train_transform, mask_transform=train_mask_transform,
                                                   batch_size=params.batch_size,
                                                   num_workers=params.workers, shuffle=True)
        val_loader = get_dataloader_segmentation(params.val_dir, params.val_annotation_file,
                                                 transform=val_transform, mask_transform=val_mask_transform,
                                                 batch_size=params.batch_size_eval,
                                                 num_workers=params.workers, shuffle=False, random_nb_object=False)
    elif params.modality == "video":
        train_loader = get_video_dataloader(params.train_dir, batch_size=params.batch_size,
                                            num_workers=params.workers, transform=train_transform,
                                            mask_transform=train_mask_transform,
                                            output_resolution=(
                                                params.img_size, params.img_size),
                                            flatten_clips_to_frames=True)
        val_loader = get_video_dataloader(params.val_dir, batch_size=params.batch_size,
                                          num_workers=params.workers, transform=val_transform,
                                          mask_transform=val_mask_transform,
                                          output_resolution=(
                                              params.img_size, params.img_size),
                                          flatten_clips_to_frames=True)
    else:
        raise ValueError(
            f"Invalid modality: {params.modality}. Supported modalities are 'image' and 'video'.")

    # optionally resume training
    if params.resume_from is not None:
        uoptim.restart_from_checkpoint(
            params.resume_from,
            model=wam,
        )
    to_restore = {"epoch": 0}
    uoptim.restart_from_checkpoint(
        os.path.join(params.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=wam,
        optimizer=optimizer,
        optimizer_d=optimizer_d,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']
    for param_group in optimizer_d.param_groups:
        param_group['lr'] = optim_params_d['lr']
    optimizers = [optimizer, optimizer_d]

    # specific thing to do if distributed training
    if params.distributed:
        wam_ddp = nn.parallel.DistributedDataParallel(
            wam, device_ids=[params.local_rank])
        image_detection_loss.discriminator = nn.parallel.DistributedDataParallel(
            image_detection_loss.discriminator, device_ids=[params.local_rank]
        )
    else:
        wam_ddp = wam

    # setup for validation
    validation_augs = [
        (Identity,          [0]),  # No parameters needed for identity
        (HorizontalFlip,    [0]),  # No parameters needed for flip
        (Rotate,            [10, 30, 45, 90]),  # (min_angle, max_angle)
        (Resize,            [0.5, 0.75]),  # size ratio
        (Crop,              [0.5, 0.75]),  # size ratio
        (Perspective,       [0.2, 0.5, 0.8]),  # distortion_scale
        (Brightness,        [0.5, 1.5]),
        (Contrast,          [0.5, 1.5]),
        (Saturation,        [0.5, 1.5]),
        (Hue,               [-0.5, -0.25, 0.25, 0.5]),
        (JPEG,              [40, 60, 80]),
        (GaussianBlur,      [3, 5, 9, 17]),
        (MedianFilter,      [3, 5, 9, 17]),
    ]  # augs evaluated every full_eval_freq
    validation_augs_subset = [
        (Identity,          [0]),  # No parameters needed for identity
        (Brightness,        [0.5]),
        (Crop,              [0.75]),  # size ratio
        (JPEG,              [60]),
    ]  # augs evaluated every eval_freq
    dummy_img = torch.ones(3, params.img_size, params.img_size)
    validation_masks = augmenter.mask_embedder.sample_representative_masks(
        dummy_img)  # n 1 h w, full of ones or random masks depending on config

    # evaluation only
    if params.only_eval:
        val_stats = eval_one_epoch(wam, val_loader, image_detection_loss,
                                   0, validation_augs, validation_masks, params)
        if udist.is_main_process():
            with open(os.path.join(params.output_dir, 'log_only_eval.txt'), 'a') as f:
                f.write(json.dumps(val_stats) + "\n")
        return

    # start training
    print('training...')
    start_time = time.time()
    for epoch in range(start_epoch, params.epochs):
        log_stats = {'epoch': epoch}

        if scheduler is not None:
            scheduler.step(epoch)
        if scaling_scheduler is not None:
            scaling_scheduler.step(epoch)

        if params.distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            wam_ddp, optimizers, train_loader, image_detection_loss, epoch, params)
        log_stats = {**log_stats, **
                     {f'train_{k}': v for k, v in train_stats.items()}}

        if epoch % params.eval_freq == 0:
            augs = validation_augs if epoch % params.full_eval_freq == 0 else validation_augs_subset
            val_stats = eval_one_epoch(wam, val_loader, image_detection_loss,
                                       epoch, augs, validation_masks, params)
            # val_stats = eval_one_epoch(wam_ddp, val_loader, image_detection_loss, epoch, params)
            log_stats = {**log_stats, **
                         {f'val_{k}': v for k, v in val_stats.items()}}
        if udist.is_main_process():
            with open(os.path.join(params.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")

        save_dict = {
            'epoch': epoch + 1,
            'model': wam.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
        }
        udist.save_on_master(save_dict, os.path.join(
            params.output_dir, 'checkpoint.pth'))
        if params.saveckpt_freq and epoch % params.saveckpt_freq == 0:
            udist.save_on_master(save_dict, os.path.join(
                params.output_dir, f'checkpoint{epoch:03}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))


def train_one_epoch(
    wam: Wam,
    optimizers: List[torch.optim.Optimizer],
    train_loader: torch.utils.data.DataLoader,
    image_detection_loss: LPIPSWithDiscriminator,
    epoch: int,
    params: argparse.Namespace,
):
    wam.train()

    header = 'Train - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    for it, batch_items in enumerate(metric_logger.log_every(train_loader, 10, header)):

        if len(batch_items) == 3:
            imgs, masks, frames_positions = batch_items
        elif len(batch_items) == 2:
            imgs, masks = batch_items

        # masks are only used if segm_proba > 0
        imgs = imgs.to(device, non_blocking=True)

        # forward
        outputs = wam(imgs, masks)
        outputs["preds"] /= params.temperature

        if params.embedder_model.startswith("vae"):
            last_layer = wam.module.embedder.decoder.conv_out.weight if params.distributed else wam.embedder.decoder.conv_out.weight
        elif params.embedder_model.startswith("unet"):
            last_layer = wam.module.embedder.unet.outc.weight if params.distributed else wam.embedder.unet.outc.weight
        elif params.embedder_model.startswith("hidden"):
            last_layer = wam.module.embedder.hidden_encoder.final_layer.weight if params.distributed else wam.embedder.hidden.hidden_encoder.final_layer.weight
        else:
            last_layer = None
            # imgs.requires_grad = True
            # last_layer = imgs

        for optimizer_idx in [1, 0]:
            # index 1 for discriminator, 0 for embedder/extractor
            loss, logs = image_detection_loss(
                imgs, outputs["imgs_w"],
                outputs["masks"], outputs["msgs"], outputs["preds"],
                optimizer_idx, epoch,
                last_layer=last_layer,
            )
            optimizers[optimizer_idx].zero_grad()
            loss.backward()
            optimizers[optimizer_idx].step()

        # log stats
        log_stats = {
            **logs,
            'psnr': psnr(outputs["imgs_w"], imgs).mean().item(),
            'lr': optimizers[0].param_groups[0]['lr'],
        }
        bit_preds = outputs["preds"][:, 1:]  # b k h w
        mask_preds = outputs["preds"][:, 0:1]  # b 1 h w

        # bit accuracy
        if params.nbits > 0:
            bit_accuracy_ = bit_accuracy(
                bit_preds,  # b k h w
                outputs["msgs"],  # b k
                outputs["masks"]
            ).nanmean().item()
            log_stats['bit_acc'] = bit_accuracy_

        # localization metrics
        if params.lambda_det > 0:
            iou0 = iou(mask_preds, outputs["masks"], label=0).mean().item()
            iou1 = iou(mask_preds, outputs["masks"], label=1).mean().item()
            log_stats.update({
                f'acc': accuracy(mask_preds, outputs["masks"]).mean().item(),
                f'miou': (iou0 + iou1) / 2,
            })
        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        # save images
        if epoch % params.saveimg_freq == 0 and it == 0 and udist.is_main_process():
            # if epoch % params.saveimg_freq == 0 and it % 200 == 0 and udist.is_main_process():
            # save images and diff
            save_image(unnormalize_img(imgs),
                       os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_0_ori.png'), nrow=8)
            save_image(unnormalize_img(outputs["imgs_w"]),
                       os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_1_w.png'), nrow=8)
            save_image(create_diff_img(imgs, outputs["imgs_w"]),
                       # save_image(5 * unstd_img(params.scaling_w * outputs["deltas_w"]).abs(),
                       os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_2_diff.png'), nrow=8)
            save_image(unnormalize_img(outputs["imgs_aug"]),
                       os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_3_aug.png'), nrow=8)
            # save pred and target masks
            if params.lambda_det > 0:
                save_image(outputs["masks"],
                           os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_4_mask.png'), nrow=8)
                save_image(F.sigmoid(mask_preds / params.temperature),
                           os.path.join(params.output_dir, f'{epoch:03}_{it:03}_train_5_pred.png'), nrow=8)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_one_epoch(
    wam: Wam,
    val_loader: torch.utils.data.DataLoader,
    image_detection_loss: LPIPSWithDiscriminator,
    epoch: int,
    validation_augs: List,
    validation_masks: torch.Tensor,
    params: argparse.Namespace,
) -> dict:
    """ 
    Evaluate the model on the validation set, with different augmentations

    Args:
        wam (Wam): the model
        val_loader (torch.utils.data.DataLoader): the validation loader
        image_detection_loss (LPIPSWithDiscriminator): the loss function
        epoch (int): the current epoch
        validation_augs (List): list of augmentations to apply
        validation_masks (torch.Tensor): the validation masks, full of ones for now
        params (argparse.Namespace): the parameters
    """
    if torch.is_tensor(validation_masks):
        validation_masks = list(torch.unbind(validation_masks, dim=0))
    wam.eval()
    header = 'Val Full - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    aug_metrics = {}
    for it, (imgs, masks) in enumerate(metric_logger.log_every(val_loader, 10, header)):

        if it * params.batch_size_eval >= 100:
            break

        imgs = imgs.to(device, non_blocking=True)
        msgs = wam.get_random_msg(imgs.shape[0])  # b x k
        msgs = msgs.to(imgs.device)

        # generate watermarked images
        deltas_w = wam.embedder(imgs, msgs)
        imgs_w = wam.scaling_i * imgs + wam.scaling_w * deltas_w

        # attenuate
        if wam.attenuation is not None:
            imgs_w = wam.attenuation(imgs, imgs_w)

        for mask_id, masks in enumerate(validation_masks):
            # watermark masking
            masks = masks.to(imgs.device, non_blocking=True)  # 1 h w
            if len(masks.shape) < 4:
                masks = masks.unsqueeze(0).repeat(
                    imgs_w.shape[0], 1, 1, 1)  # b 1 h w
            imgs_masked = imgs_w * masks + imgs * (1 - masks)

            for transform, strengths in validation_augs:
                # Create an instance of the transformation
                transform_instance = transform()

                for strength in strengths:
                    do_resize = True  # hardcode for now, might need to change
                    if not do_resize:
                        imgs_aug, masks_aug = transform_instance(
                            imgs_masked, masks, strength)
                    else:
                        # h, w = imgs_w.shape[-2:]
                        h, w = params.img_size_extractor, params.img_size_extractor
                        imgs_aug, masks_aug = transform_instance(
                            imgs_masked, masks, strength)
                        if imgs_aug.shape[-2:] != (h, w):
                            imgs_aug = nn.functional.interpolate(imgs_aug, size=(
                                h, w), mode='bilinear', align_corners=False, antialias=True)
                            masks_aug = nn.functional.interpolate(masks_aug, size=(
                                h, w), mode='bilinear', align_corners=False, antialias=True)
                    selected_aug = str(
                        transform.__name__).lower() + '_' + str(strength)

                    # extract watermark
                    preds = wam.detector(imgs_aug)
                    mask_preds = preds[:, 0:1]  # b 1 ...
                    bit_preds = preds[:, 1:]  # b k ...

                    log_stats = {}
                    if params.nbits > 0:
                        bit_accuracy_ = bit_accuracy(
                            bit_preds,
                            msgs,
                            masks_aug
                        ).nanmean().item()

                    if params.nbits > 0:
                        log_stats[f'bit_acc'] = bit_accuracy_

                    if params.lambda_det > 0:
                        iou0 = iou(mask_preds, masks, label=0).mean().item()
                        iou1 = iou(mask_preds, masks, label=1).mean().item()
                        log_stats.update({
                            f'acc': accuracy(mask_preds, masks).mean().item(),
                            f'miou': (iou0 + iou1) / 2,
                        })

                    current_key = f"mask={mask_id}_aug={selected_aug}"
                    log_stats = {f"{k}_{current_key}": v for k,
                                 v in log_stats.items()}

                    # save stats of the current augmentation
                    aug_metrics = {**aug_metrics, **log_stats}

                    # save some of the images
                    if (epoch % params.saveimg_freq == 0 or params.only_eval) and it == 0 and udist.is_main_process():
                        save_image(unnormalize_img(imgs),
                                   os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_0_ori.png'), nrow=8)
                        save_image(unnormalize_img(imgs_w),
                                   os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_1_w.png'), nrow=8)
                        save_image(create_diff_img(imgs, imgs_w),
                                   os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_2_diff.png'), nrow=8)
                        save_image(unnormalize_img(imgs_aug),
                                   os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_3_aug.png'), nrow=8)
                        if params.lambda_det > 0:
                            save_image(masks,
                                       os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_4_mask.png'), nrow=8)
                            save_image(F.sigmoid(mask_preds / params.temperature),
                                       os.path.join(params.output_dir, f'{epoch:03}_{it:03}_val_5_pred.png'), nrow=8)

        torch.cuda.synchronize()
        for name, loss in aug_metrics.items():
            # if name == 'bit_acc' and math.isnan(loss):
            #     continue
            # if name in ["decode_loss", "decode_scale"] and loss == -1:
            #     continue  # Skip this update or replace with a default value
            metric_logger.update(**{name: loss})

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('val'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
