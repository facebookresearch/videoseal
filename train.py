"""
Example usage (local):
    torchrun --nproc_per_node=2 train.py --local_rank 0
Example usage (cluster 1 gpu):
    torchrun train.py --debug_slurm
    For eval ful only:
        torchrun train.py --debug_slurm --double_w False --only_eval True --output_dir /checkpoint/tomsander/2406-segmark/0628_doublewm_vs_onewm_fixed/_lambda_w=1_double_w=false
Example usage (cluster 2 gpus):
    torchrun --nproc_per_node=2 train.py --local_rank 0 --debug_slurm

Args inventory:
    --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=100,warmup_lr_init=1e-6,warmup_t=5
    --optimizer Lamb,lr=1e-3
    --train_dir /checkpoint/pfz/projects/watermarking/data/coco_1k_orig
    --resume_from /checkpoint/pfz/2024_logs/0611_segmark_lpipsmse/_lambda_d=0.25_lambda_i=0.5_scaling_w=0.4/checkpoint050.pth
    --extractor_model dino2s_indices=11_upscale=1_14 --img_size 336 --batch_size 8 --extractor_config configs/extractor_dinos.yaml
    --embedder_model vae_sd --embedder_config configs/embedder_sd.yaml

torchrun --nproc_per_node=2 train.py  --local_rank 0  --only_eval True --scaling_w 0.4 --embedder_model vae_small --extractor_model sam_small --augmentation_config configs/all_augs.yaml --resume_from /checkpoint/pfz/2024_logs/0708_segmark_bigger_vae/_scaling_w=0.4_embedder_model=vae_small_extractor_model=sam_small/checkpoint.pth
torchrun --nproc_per_node=2 train.py  --local_rank 0  --only_eval True --scaling_w 2.0 --scaling_i 1.0 --nbits 16 --lambda_w2 6.0 --lambda_w 1.0 --lambda_d 0.0 --lambda_i 0.0 --perceptual_loss none --seed 0 --scheduler none --optimizer AdamW,lr=1e-5 --epochs 50 --batch_size_eval 32 --batch_size 16 --img_size 256 --attenuation jnd_1_3 --resume_from /checkpoint/pfz/2024_logs/0708_segmark_bigger_vae/_scaling_w=0.4_embedder_model=vae_small_extractor_model=sam_small/checkpoint.pth --embedder_model vae_small --extractor_model sam_small --augmentation_config configs/all_augs.yaml


    
torchrun --nproc_per_node=2 train.py --local_rank 0  \
    --img_size 128      --saveimg_freq 1  \
    --lambda_i 1.0 --lambda_w 0 --lambda_d 0.0  \
    --perceptual_loss watson_vgg --scaling_i 0 --embedder_tanh_out false 
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
from videoseal.data.loader import get_dataloader, get_dataloader_segmentation
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


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--train_dir", type=str, default="/datasets01/COCO/060817/train2014/")
    aa("--train_annotation_file", type=str,
       default="/datasets01/COCO/060817/annotations/instances_train2014.json")
    aa("--val_dir", type=str, default="/datasets01/COCO/060817/val2014/")
    aa("--val_annotation_file", type=str,
       default="/datasets01/COCO/060817/annotations/instances_val2014.json")
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
    aa("--nbits", type=int, default=16,
       help="Number of bits used to generate the message. If 0, no message is used.")
    aa("--img_size", type=int, default=256, help="Size of the input images")
    aa("--img_size_extractor", type=int,
       default=256, help="Size of the input images")
    aa("--attenuation", type=str, default="None", help="Attenuation model to use")
    aa("--scaling_w", type=float, default=0.2,
       help="Scaling factor for the watermark in the embedder model")
    aa("--scaling_w_schedule", type=str, default=None,
       help="Scaling factor for the watermark in the embedder model")
    aa("--scaling_i", type=float, default=1.0,
       help="Scaling factor for the image in the embedder model")
    aa("--double_w", type=utils.bool_inst, default=False,
       help="Use 2 watermarks instead of 1. Can not be used with a detection loss simultanuously.")
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
    aa('--lambda_w', default=1.0, type=float,
       help='Weight for the watermark detection loss')
    aa('--lambda_w2', default=4.0, type=float,
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
    aa('--saveimg_freq', default=5, type=int, help='Frequency for saving images')
    aa('--saveckpt_freq', default=50, type=int, help='Frequency for saving ckpts')
    aa('--seed', default=0, type=int, help='Random seed')

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)

    return parser


def main(params):
    # Incompatibility between multiw and detection
    if (params.double_w and params.lambda_w != 0):
        print("Incompatible parameters: double_w and lambda_w!=0, setting lambda_w to 0 ")
        params.lambda_w = 0

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
    print("__log__:{}".format(json.dumps(vars(params))))

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
              params.scaling_w, params.scaling_i, params.double_w)
    wam.to(device)
    # print(wam)

    # build losses
    image_detection_loss = LPIPSWithDiscriminator(
        balanced=params.balanced, total_norm=params.total_gnorm,
        disc_weight=params.lambda_d, percep_weight=params.lambda_i,
        detect_weight=params.lambda_w, decode_weight=params.lambda_w2,
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
    train_loader = get_dataloader_segmentation(params.train_dir, params.train_annotation_file,
                                               transform=train_transform, mask_transform=train_mask_transform,
                                               batch_size=params.batch_size,
                                               num_workers=params.workers, shuffle=True)
    val_loader = get_dataloader_segmentation(params.val_dir, params.val_annotation_file,
                                             transform=val_transform, mask_transform=val_mask_transform,
                                             batch_size=params.batch_size_eval,
                                             num_workers=params.workers, shuffle=False, random_nb_object=False)

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
    ]
    dummy_img = torch.ones(3, params.img_size, params.img_size)
    validation_masks = augmenter.mask_embedder.sample_representative_masks(
        dummy_img)  # 5 1 h w
    if udist.is_main_process():
        save_image(validation_masks, os.path.join(
            params.output_dir, 'validation_masks.png'))

    # evaluation only
    if params.only_eval:
        val_stats = eval_full(wam, val_loader, image_detection_loss,
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
        print(f'Epoch {epoch} - scaling_w: {wam.scaling_w}')

        if params.distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            wam_ddp, optimizers, train_loader, image_detection_loss, epoch, params)
        log_stats = {**log_stats, **
                     {f'train_{k}': v for k, v in train_stats.items()}}

        if epoch % params.eval_freq == 0:
            val_stats = eval_full(wam, val_loader, image_detection_loss,
                                  epoch, validation_augs, validation_masks, params)
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

    for it, (imgs, masks) in enumerate(metric_logger.log_every(train_loader, 10, header)):

        imgs = imgs.to(device, non_blocking=True)

        # forward
        outputs = wam(imgs, masks)
        outputs["preds"] /= params.temperature

        if params.embedder_model.startswith("vae"):
            last_layer = wam.module.embedder.decoder.conv_out.weight if params.distributed else wam.embedder.decoder.conv_out.weight
        elif params.embedder_model.startswith("unet_plus"):
            last_layer = wam.module.embedder.unet.outc.weight if params.distributed else wam.embedder.unet.outc.weight
        elif params.embedder_model.startswith("unet"):
            last_layer = wam.module.embedder.unet.model.up[
                2].weight if params.distributed else wam.embedder.unet.model.up[2].weight
        else:
            last_layer = None
            # imgs.requires_grad = True
            # last_layer = imgs

        for optimizer_idx in [1, 0]:
            loss, logs = image_detection_loss(
                imgs, outputs["imgs_w"],
                outputs["masks"], outputs["msgs"], outputs["preds"],
                optimizer_idx, epoch,
                last_layer=last_layer, msgs2=outputs["msgs2"]
            )
            optimizers[optimizer_idx].zero_grad()
            loss.backward()
            optimizers[optimizer_idx].step()

        # log stats
        if params.nbits > 0:
            bit_accuracy_ = bit_accuracy(
                outputs["preds"][:, 1:, :, :],
                outputs["msgs"],
                outputs["masks"]
            ).nanmean().item()

        mask_preds = outputs["preds"][:, 0:1, :, :]  # b 1 h w
        # Mask pred using only the heatmap
        mask_preds_hm, mask_preds_hm_dynamic = detect_wm_hm(
            outputs["preds"], outputs["msgs"], bit_accuracy_, params)
        log_stats = {**logs, 'psnr': psnr(outputs["imgs_w"], imgs).mean().item(
        ), 'lr': optimizers[0].param_groups[0]['lr'], 'avg_target': outputs["masks"].mean().item()}
        for method, mask in zip(["", "_hm", "_hm_dynamic"], [mask_preds, mask_preds_hm, mask_preds_hm_dynamic]):
            log_stats.update({
                f'acc{method}': accuracy(mask, outputs["masks"]).mean().item(),
                f'iou_0{method}': iou(mask, outputs["masks"], label=0).mean().item(),
                f'iou_1{method}': iou(mask, outputs["masks"], label=1).mean().item(),
                f'avg_pred{method}': mask.mean().item(),
                f'norm_avg{method}': torch.norm(mask, p=2).item(),
            })
            log_stats[f'miou{method}'] = (
                log_stats[f'iou_0{method}'] + log_stats[f'iou_1{method}']) / 2
        if params.nbits > 0:
            log_stats['bit_acc'] = bit_accuracy_
        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            # if name == 'bit_acc' and math.isnan(loss):
            #     continue
            # if name in ['decode_loss', 'decode_scale'] and loss==-1:
            #     continue  # Skip this update or replace with a default value
            metric_logger.update(**{name: loss})

        # save images
        # if epoch % params.saveimg_freq == 0 and it == 0 and udist.is_main_process():
        if epoch % params.saveimg_freq == 0 and it % 200 == 0 and udist.is_main_process():
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
    params: argparse.Namespace,
):
    # In July 2024, this function is not used
    wam.eval()
    header = 'Val - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = ulogger.MetricLogger(delimiter="  ")
    aug_metrics = {}
    for it, (imgs, masks) in enumerate(metric_logger.log_every(val_loader, 10, header)):
        imgs = imgs.to(device, non_blocking=True)

        # forward
        outputs = wam(imgs)
        outputs["preds"] /= params.temperature

        # compute loss
        loss, logs = image_detection_loss(
            imgs, outputs["imgs_w"],
            outputs["masks"], outputs["msgs"], outputs["preds"],
            0, epoch, None
        )
        if params.nbits > 0:
            bit_accuracy_ = bit_accuracy(
                outputs["preds"][:, 1:, :, :],
                outputs["msgs"],
                outputs["masks"]
            ).nanmean().item()
        # log stats
        mask_preds = outputs["preds"][:, 0:1, :, :]  # b 1 h w
        # Mask pred using only the heatmap
        mask_preds_hm, mask_preds_hm_dynamic = detect_wm_hm(
            outputs["preds"], outputs["msgs"], bit_accuracy_, params)
        selected_aug = outputs["selected_aug"]
        if selected_aug not in aug_metrics:
            aug_metrics[selected_aug] = ulogger.MetricLogger(delimiter="  ")
        log_stats = {**logs, 'psnr': psnr(outputs["imgs_w"], imgs).mean().item(
        ), 'lr': optimizers[0].param_groups[0]['lr'], 'avg_target': outputs["masks"].mean().item()}
        for method, mask in zip(["", "_hm", "_hm_dynamic"], [mask_preds, mask_preds_hm, mask_preds_hm_dynamic]):
            log_stats.update({
                f'acc{method}': accuracy(mask, outputs["masks"]).mean().item(),
                f'iou_0{method}': iou(mask, outputs["masks"], label=0).mean().item(),
                f'iou_1{method}': iou(mask, outputs["masks"], label=1).mean().item(),
                f'avg_pred{method}': mask.mean().item(),
                f'norm_avg{method}': torch.norm(mask, p=2).item(),
            })
            log_stats[f'miou{method}'] = (
                log_stats[f'iou_0{method}'] + log_stats[f'iou_1{method}']) / 2
        if params.nbits > 0:
            log_stats['bit_acc'] = bit_accuracy_
        aug_metrics[selected_aug].update(**log_stats)
        torch.cuda.synchronize()
        for name, loss in log_stats.items():
            if name == 'bit_acc' and math.isnan(loss):
                continue
            metric_logger.update(**{name: loss})
        # save images
        if epoch % params.saveimg_freq == 0 and it == 0 and udist.is_main_process():
            save_image(unnormalize_img(imgs), os.path.join(
                params.output_dir, f'{epoch:03}_{it:03}_val_ori.png'), nrow=8)
            save_image(unnormalize_img(outputs["imgs_w"]), os.path.join(
                params.output_dir, f'{epoch:03}_{it:03}_val_w.png'), nrow=8)

    for aug, logger in aug_metrics.items():
        logger.synchronize_between_processes()
        print(f"Averaged stats for selected_aug {aug}: {logger}")
        for k, meter in logger.meters.items():
            metric_logger.update(**{f"{k}_{aug}": meter.global_avg})
    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('val'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_full(
    wam: Wam,
    val_loader: torch.utils.data.DataLoader,
    image_detection_loss: LPIPSWithDiscriminator,
    epoch: int,
    validation_augs: List,
    validation_masks: List,
    params: argparse.Namespace,
):
    if torch.is_tensor(validation_masks):
        validation_masks = list(torch.unbind(validation_masks, dim=0))
    wam.eval()
    header = 'Val Full - Epoch: [{}/{}]'.format(epoch, params.epochs)
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    # to save
    tosave = {
        nb_wm: [f"mask={1}_aug={'crop_0.75'}", f"mask={2}_aug={'resize_0.75'}",
                f"mask={3}_aug={'brightness_1.5'}", f"mask={4}_aug={'jpeg_60'}", f"mask={5}_aug={'identity_0'}"]
        for nb_wm in ["1wm", "2wm"]}
    imgs_tosave = {"1wm": [], "2wm": []}

    aug_metrics = {}
    for it, (imgs, masks) in enumerate(metric_logger.log_every(val_loader, 10, header)):
        validation_masks_and_seg = validation_masks + [masks]
        # TODO: loaded masks are not used for evaluation at the moment

        if it * params.batch_size_eval >= 100:
            break

        imgs = imgs.to(device, non_blocking=True)
        msgs = wam.get_random_msg(imgs.shape[0])  # b x k
        msgs = msgs.to(imgs.device)
        msgs2 = wam.get_secong_msg(msgs)  # b x k

        # generate watermarked images
        deltas_w = wam.embedder(imgs, msgs)
        imgs_w = wam.scaling_i * imgs + wam.scaling_w * deltas_w

        # generate watermarked images
        deltas_w2 = wam.embedder(imgs, msgs2)
        imgs_w2 = wam.scaling_i * imgs + wam.scaling_w * deltas_w2

        # attenuate
        if wam.attenuation is not None:
            imgs_w = wam.attenuation(imgs, imgs_w)
            imgs_w2 = wam.attenuation(imgs, imgs_w2)

        for mask_id, masks in enumerate(validation_masks_and_seg):
            # watermark masking
            masks = masks.to(imgs.device, non_blocking=True)  # 1 h w
            if len(masks.shape) < 4:
                masks = masks.unsqueeze(0).repeat(
                    imgs_w.shape[0], 1, 1, 1)  # b 1 h w
            imgs_masked = imgs_w * masks + imgs * (1 - masks)
            imgs_2wm = imgs_w * masks + imgs_w2 * (1 - masks)

            for nb_wm, imgs_masked_ in [("1wm", imgs_masked)]:
                # for nb_wm, imgs_masked_ in [("1wm", imgs_masked), ("2wm", imgs_2wm)]:
                for transform, strengths in validation_augs:
                    # Create an instance of the transformation
                    transform_instance = transform()

                    for strength in strengths:
                        do_resize = True  # hardcode for now, might need to change
                        if not do_resize:
                            imgs_aug, masks_aug = transform_instance(
                                imgs_masked_, masks, strength)
                        else:
                            # h, w = imgs_w.shape[-2:]
                            h, w = params.img_size_extractor, params.img_size_extractor
                            imgs_aug, masks_aug = transform_instance(
                                imgs_masked_, masks, strength)
                            if imgs_aug.shape[-2:] != (h, w):
                                imgs_aug = nn.functional.interpolate(imgs_aug, size=(
                                    h, w), mode='bilinear', align_corners=False, antialias=True)
                                masks_aug = nn.functional.interpolate(masks_aug, size=(
                                    h, w), mode='bilinear', align_corners=False, antialias=True)
                        selected_aug = str(
                            transform.__name__).lower() + '_' + str(strength)

                        # detect watermark
                        preds = wam.detector(imgs_aug)
                        if params.nbits > 0:
                            bit_preds = preds[:, 1:, :, :]
                            bit_accuracy_ = bit_accuracy(
                                bit_preds,
                                msgs,
                                masks_aug
                            ).nanmean().item()
                        # Start with masks by using the first bit of the prediction
                        mask_preds = preds[:, 0:1, :, :]  # b 1 h w
                        # Threshold on bit accuracy
                        mask_preds_hm, mask_preds_hm_dynamic = detect_wm_hm(
                            preds, msgs, bit_accuracy_, params)
                        log_stats = {}
                        if params.nbits > 0:
                            log_stats[f'bit_acc_{nb_wm}'] = bit_accuracy_

                        # compute stats for the augmentation and strength
                        for method, mask_preds_ in [('', mask_preds)]:
                            # for method, mask_preds_ in [('', mask_preds), ('_hm', mask_preds_hm), ('_hm_dynamic', mask_preds_hm_dynamic)]:
                            log_stats.update({
                                f'acc{method}_{nb_wm}': accuracy(mask_preds_, masks_aug).mean().item(),
                                f'iou_0{method}_{nb_wm}': iou(mask_preds_, masks_aug, label=0).mean().item(),
                                f'iou_1{method}_{nb_wm}': iou(mask_preds_, masks_aug, label=1).mean().item(),
                                f'avg_pred{method}_{nb_wm}': mask_preds_.mean().item(),
                                f'avg_target{method}_{nb_wm}': masks_aug.mean().item(),
                                f'norm_avg{method}_{nb_wm}': torch.norm(mask_preds_, p=2).item(),
                            })
                            log_stats[f'miou{method}_{nb_wm}'] = (
                                log_stats[f'iou_0{method}_{nb_wm}'] + log_stats[f'iou_1{method}_{nb_wm}']) / 2
                            if params.nbits > 0:
                                for decode_method in ['semihard']:
                                    # for decode_method in ['hard', 'semihard', 'soft']:
                                    log_stats[f"bit_acc{method}_{nb_wm}_{decode_method}"] = bit_accuracy_inference(
                                        bit_preds,
                                        msgs,
                                        F.sigmoid(mask_preds_),  # b h w
                                        method=decode_method
                                    ).nanmean().item()
                        current_key = f"mask={mask_id}_aug={selected_aug}"
                        log_stats = {f"{k}_{current_key}": v for k,
                                     v in log_stats.items()}

                        # save stats of the current augmentation
                        aug_metrics = {**aug_metrics, **log_stats}

                        # save some of the images
                        if (epoch % params.saveimg_freq == 0 or params.only_eval) and udist.is_main_process():
                            if current_key in tosave[nb_wm]:
                                # consider 1 image per augmentation
                                idx = len(imgs_tosave[nb_wm]) // 6
                                imgs_tosave[nb_wm].append(
                                    unnormalize_img(imgs[idx].cpu()))
                                imgs_tosave[nb_wm].append(
                                    unnormalize_img(imgs_w[idx].cpu()))
                                imgs_tosave[nb_wm].append(
                                    unnormalize_img(imgs_aug[idx].cpu()))
                                imgs_tosave[nb_wm].append(
                                    masks_aug[idx].cpu().repeat(3, 1, 1))
                                imgs_tosave[nb_wm].append(
                                    F.sigmoid(mask_preds[idx]).cpu().repeat(3, 1, 1))
                                imgs_tosave[nb_wm].append(
                                    F.sigmoid(mask_preds_hm_dynamic[idx]).cpu().repeat(3, 1, 1))
                                tosave[nb_wm].remove(current_key)

        torch.cuda.synchronize()
        for name, loss in aug_metrics.items():
            if name == 'bit_acc' and math.isnan(loss):
                continue
            if name in ["decode_loss", "decode_scale"] and loss == -1:
                continue  # Skip this update or replace with a default value
            metric_logger.update(**{name: loss})

    # save images
    if (epoch % params.saveimg_freq == 0 or params.only_eval) and udist.is_main_process():
        aux = "" if not params.only_eval else "_only_eval"
        save_image(torch.stack(imgs_tosave["1wm"]), os.path.join(
            params.output_dir, f'{epoch:03}_val_full{aux}.png'), nrow=6)
        # save_image(torch.stack(imgs_tosave["2wm"]), os.path.join(params.output_dir, f'{epoch:03}_val_full_2wm{aux}.png'), nrow=6)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('val'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
