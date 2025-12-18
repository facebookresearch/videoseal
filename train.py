# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usage (cluster 2 gpus):
    torchrun --nproc_per_node=2 train.py --local_rank 0
Example usage (cluster 1 gpu):
    torchrun train.py --debug_slurm
    For eval full only:
        torchrun train.py --debug_slurm --only_eval True --output_dir output/

Put OMP_NUM_THREADS such that OMP_NUM_THREADS=(number of CPU threads)/(nproc per node) to remove warning messages
        
Examples:

    1/ image training

    OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 train.py --local_rank 0 \
        --video_dataset none --image_dataset sa-1b-full-resized --workers 8 \
        --extractor_model convnext_tiny --embedder_model unet_small2_yuv_quant --hidden_size_multiplier 1 --nbits 128 \
        --scaling_w_schedule Cosine,scaling_min=0.2,start_epoch=200,epochs=200 --scaling_w 1.0 --scaling_i 1.0 --attenuation jnd_1_1 \
        --epochs 601 --iter_per_epoch 1000 --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=601,warmup_lr_init=1e-8,warmup_t=20 --optimizer AdamW,lr=5e-4 \
        --lambda_dec 1.0 --lambda_d 0.1 --lambda_i 0.1 --perceptual_loss yuv  --num_augs 2 --augmentation_config configs/all_augs.yaml --disc_in_channels 1 --disc_start 50 

    2/ video finetuning
    
    OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 train.py --local_rank 0 \
        --video_dataset sa-v --image_dataset none --workers 0 --frames_per_clip 16 \
        --resume_from /path/to/ckpt/full/checkpoint.pth --resume_optimizer_state True --resume_disc True  \
        --videoseal_step_size 4 --lowres_attenuation True --img_size_proc 256 --img_size_val 768 --img_size 768 \
        --extractor_model convnext_tiny --embedder_model unet_small2_yuv_quant --hidden_size_multiplier 1 --nbits 128 \
        --scaling_w_schedule None --scaling_w 0.2 --scaling_i 1.0 --attenuation jnd_1_1 \
        --epochs 601 --iter_per_epoch 100 --scheduler None --optimizer AdamW,lr=1e-5 \
        --lambda_dec 1.0 --lambda_d 0.5 --lambda_i 0.1 --perceptual_loss yuv  --num_augs 2 --augmentation_config configs/all_augs.yaml --disc_in_channels 1 --disc_start 50
"""

import argparse
import datetime
import json
import os
import time
from typing import List

import numpy as np
import omegaconf

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
from torchvision.utils import save_image

import videoseal.utils as utils
import videoseal.utils.dist as udist
import videoseal.utils.logger as ulogger
import videoseal.utils.optim as uoptim
from videoseal.augmentation import (get_validation_augs,
                                    get_validation_augs_subset)
from videoseal.augmentation.augmenter import Augmenter
from videoseal.data.loader import (get_dataloader_segmentation,
                                   get_video_dataloader)
from videoseal.data.transforms import get_resize_transform
from videoseal.evals.metrics import accuracy, bit_accuracy, iou, psnr, ssim
from videoseal.losses.videosealloss import VideosealLoss
from videoseal.models import Videoseal, Wam, build_embedder, build_extractor
from videoseal.modules.jnd import JND
from videoseal.utils.data import Modalities, parse_dataset_params
from videoseal.utils.display import save_vid
from videoseal.utils.image import create_diff_img
from videoseal.utils.tensorboard import CustomTensorboardWriter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Dataset parameters')
    aa("--image_dataset", type=str, 
        help="Name of the image dataset.", default="sa-1b")
    aa("--video_dataset", type=str,
        help="Name of the video dataset.", default="sa-v")
    aa("--prop_img_vid", type=float, default=0.5,
        help="Percentage of images in the hybrid dataset 0.5 means for each 5 epochs of images 5 video epoch is made. Only applicable if both --image_dataset and --video_dataset are provided.")
    aa("--video_start", type=int, default=500,
        help="Number of epochs before starting video training")
    aa("--finetune_detector_start", type=int, default=1e6,
       help="Number of epochs afterwhich the generator is frozen and detector is finetuned")

    group = parser.add_argument_group('Experiments parameters')
    aa("--output_dir", type=str, default="output/",
       help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Embedder and extractor config')
    aa("--embedder_config", type=str, default="configs/embedder.yaml",
       help="Path to the embedder config file")
    aa("--extractor_config", type=str, default="configs/extractor.yaml",
       help="Path to the extractor config file")
    aa("--attenuation_config", type=str, default="configs/attenuation.yaml",
       help="Path to the attenuation config file")
    aa("--embedder_model", type=str, default="unet_small2",
       help="Name of the extractor model")
    aa("--extractor_model", type=str, default="sam_tiny",
       help="Name of the extractor model")

    group = parser.add_argument_group('Augmentation parameters')
    aa("--augmentation_config", type=str, default="configs/all_augs.yaml",
       help="Path to the augmentation config file")
    aa("--num_augs", type=int, default=1,
       help="Number of augmentations to apply")

    group = parser.add_argument_group('Image and watermark parameters')
    aa("--nbits", type=int, default=32,
       help="Number of bits used to generate the message. If 0, no message is used.")
    aa("--hidden_size_multiplier", type=float, default=2,
         help="Hidden size multiplier for the message processor")
    aa("--img_size", type=int, default=256,
       help="Size of the input images for data preprocessing, used at loading time for training.")
    aa("--img_size_val", type=int, default=256,
       help="Size of the input images for data preprocessing, used at loading time for validation.")
    aa("--img_size_proc", type=int, default=256, 
       help="Size of the input images for interpolation in the embedder/extractor models")
    aa("--resize_only", type=utils.bool_inst, default=False,
         help="If True, only resize the image no crop is applied at loading time (without preserving aspect ratio)")
    aa("--attenuation", type=str, default="None", help="Attenuation model to use")
    aa("--blending_method", type=str, default="additive",
       help="The blending method to use. Options include: additive, multiplicative ..etc see Blender Class for more")
    aa("--scaling_w", type=float, default=0.2,
       help="Scaling factor for the watermark in the embedder model")
    aa("--scaling_w_schedule", type=str, default=None,
       help="Scaling factor for the watermark in the embedder model. Ex: 'Linear,scaling_min=0.025,epochs=100,start_epoch=0'")
    aa("--scaling_i", type=float, default=1.0,
       help="Scaling factor for the image in the embedder model")
    # Videoseal parameters related how to do video watermarking inference
    aa("--videoseal_chunk_size", type=int, default=32,
       help="The number of frames to encode at a time.")
    aa("--videoseal_step_size", type=int, default=4,
       help="The number of frames to propagate the watermark to.")
    aa("--lowres_attenuation", type=utils.bool_inst, default=False,
       help="Apply attenuation at low resolution for high-res images (more memory efficient)")

    group = parser.add_argument_group('Optimizer parameters')
    aa("--optimizer", type=str, default="AdamW,lr=1e-4",
       help="Optimizer (default: AdamW,lr=1e-4)")
    aa("--optimizer_d", type=str, default=None,
       help="Discriminator optimizer. If None uses the same params (default: None)")
    aa("--scheduler", type=str, default="None",
       help="Scheduler (default: None)")
    aa('--epochs', default=600, type=int,
       help='Number of total epochs to run')
    aa('--iter_per_epoch', default=10000, type=int,
       help='Number of iterations per epoch, made for very large datasets')
    aa('--sleepwake', type=utils.bool_inst, default=False,
       help='If True and lambda_d > 0 then do epoch optimize 0 and epoch optimizer 1 otherwise optimize them simultaneously')
    aa('--iter_per_valid', default=None, type=int,
       help='Number of iterations per eval, made for very large eval datasets if None eval on all dataset')
    aa('--resume_from', default=None, type=str,
       help='Path to the checkpoint to resume from')
    aa('--resume_disc', type=utils.bool_inst, default=False,
       help='If True, also load discriminator weights when resuming from checkpoint')
    aa('--resume_optimizer_state', type=utils.bool_inst, default=False,
       help='If True, also load optimizer state when resuming from checkpoint')

    group = parser.add_argument_group('Losses parameters')
    aa('--temperature', default=1.0, type=float,
       help='Temperature for the mask loss')
    aa('--lambda_det', default=0.0, type=float,
       help='Weight for the watermark detection loss')
    aa('--lambda_dec', default=1.0, type=float,
       help='Weight for the watermark decoding loss')
    aa('--lambda_i', default=0.0, type=float, help='Weight for the image loss')
    aa('--lambda_d', default=0.1, type=float,
       help='Weight for the discriminator loss')
    aa('--balanced', type=utils.bool_inst, default=False,
       help='If True, the weights of the losses are balanced')
    aa('--total_gnorm', default=0.0, type=float,
       help='Total norm for the adaptive weights. If 0, uses the norm of the biggest weight.')
    aa('--perceptual_loss', default='mse', type=str,
       help='Perceptual loss to use. "lpips", "watson_vgg" or "watson_fft"')
    aa('--disc_start', default=0, type=float,
       help='Weight for the discriminator loss')
    aa('--disc_num_layers', default=2, type=int,
       help='Number of layers for the discriminator')
    aa('--disc_in_channels', default=3, type=int,
         help='Number of input channels for the discriminator')
    
    group = parser.add_argument_group('Loading parameters')
    aa('--batch_size', default=32, type=int, help='Batch size')
    aa('--batch_size_eval', default=32, type=int, help='Batch size for evaluation')
    aa('--batch_size_video', default=1, type=int, help='Batch size')
    aa('--batch_size_video_eval', default=1,
       type=int, help='Batch size for evaluation')
    aa('--workers', default=0, type=int, help='Number of data loading workers')
    aa('--frames_per_clip', default=32, type=int,
       help='Number of frames per clip for video datasets')
    aa('--frame_step', default=1, type=int,
       help='Step between frames for video datasets')
    aa('--num_clips', default=1, type=int,
       help='Number of clips per video for video datasets')

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

    # Set up TensorBoard writer, this custom one works only in main process
    tensorboard = CustomTensorboardWriter(
        log_dir=os.path.join(params.output_dir, "tensorboard"))

    # Load dataset params from config files
    parse_dataset_params(params)

    # Convert params to OmegaConf object
    params = omegaconf.OmegaConf.create(vars(params))

    # Distributed mode
    udist.init_distributed_mode(params)

    # Set seeds for reproductibility
    seed = params.seed + udist.get_rank()
    # seed = params.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if params.distributed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Print the arguments and add to tensorboard
    print("__git__:{}".format(utils.get_sha()))
    json_params = json.dumps(
        omegaconf.OmegaConf.to_container(params, resolve=True))
    print("__log__:{}".format(json_params))

    # Copy the config files to the output dir
    if udist.is_main_process():
        os.makedirs(os.path.join(params.output_dir, 'configs'), exist_ok=True)
        os.system(f'cp {params.embedder_config} {params.output_dir}/configs/embedder.yaml')
        os.system(f'cp {params.augmentation_config} {params.output_dir}/configs/augs.yaml')
        os.system(f'cp {params.extractor_config} {params.output_dir}/configs/extractor.yaml')

    # Build the embedder model
    embedder_cfg = omegaconf.OmegaConf.load(params.embedder_config)
    params.embedder_model = params.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[params.embedder_model]
    embedder = build_embedder(params.embedder_model, embedder_params, params.nbits, params.hidden_size_multiplier)
    print(embedder)
    print(f'embedder: {sum(p.numel() for p in embedder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # build the augmenter
    augmenter_cfg = omegaconf.OmegaConf.load(params.augmentation_config)
    augmenter_cfg.num_augs = params.num_augs
    augmenter = Augmenter(
        **augmenter_cfg,
    )
    print(f'augmenter: {augmenter}')

    # Build the extractor model
    extractor_cfg = omegaconf.OmegaConf.load(params.extractor_config)
    params.extractor_model = params.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[params.extractor_model]
    extractor = build_extractor(params.extractor_model, extractor_params, params.img_size_proc, params.nbits)
    print(f'extractor: {sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # build attenuation
    if params.attenuation.lower() != "none":
        attenuation_cfg = omegaconf.OmegaConf.load(params.attenuation_config)
        if params.attenuation.lower().startswith("jnd"):
            attenuation_cfg = omegaconf.OmegaConf.load(params.attenuation_config)
            attenuation = JND(**attenuation_cfg[params.attenuation]).to(device)
        else:
            attenuation = None
    else:
        attenuation = None
    print(f'attenuation: {attenuation}')

    # build the complete model
    wam = Videoseal(embedder, extractor, augmenter, attenuation,
                   params.scaling_w, params.scaling_i,
                   img_size=params.img_size_proc,
                   chunk_size=params.videoseal_chunk_size,
                   step_size=params.videoseal_step_size,
                   blending_method=params.blending_method,
                   lowres_attenuation=params.lowres_attenuation)
    wam = wam.to(device)
    # print(wam)

    # build losses
    image_detection_loss = VideosealLoss(
        balanced=params.balanced, total_norm=params.total_gnorm,
        disc_weight=params.lambda_d, percep_weight=params.lambda_i,
        detect_weight=params.lambda_det, decode_weight=params.lambda_dec,
        disc_start=params.disc_start, disc_num_layers=params.disc_num_layers, disc_in_channels=params.disc_in_channels,
        percep_loss=params.perceptual_loss,
    ).to(device)
    print(image_detection_loss)
    # print(f"discriminator: {sum(p.numel() for p in image_detection_loss.discriminator.parameters() if p.requires_grad) / 1e3:.1f}K parameters")

    # Build the scaling schedule. Default is none
    if params.scaling_w_schedule is not None:
        scaling_w_schedule = uoptim.parse_params(params.scaling_w_schedule)
        scaling_scheduler = uoptim.ScalingScheduler(
            obj=wam.blender, attribute="scaling_w", scaling_o=params.scaling_w,
            **scaling_w_schedule
        )
    else:
        scaling_scheduler = None

    # Build optimizer and scheduler
    model_params = list(embedder.parameters()) + list(extractor.parameters())
    optim_params = uoptim.parse_params(params.optimizer)
    optimizer = uoptim.build_optimizer(model_params, **optim_params)
    scheduler_params = uoptim.parse_params(params.scheduler)
    scheduler = uoptim.build_lr_scheduler(optimizer, **scheduler_params)
    print('optimizer: %s' % optimizer)
    print('scheduler: %s' % scheduler)

    # discriminator optimizer
    if params.optimizer_d is None:
        optim_params_d = uoptim.parse_params(params.optimizer) 
    else:
        optim_params_d = uoptim.parse_params(params.optimizer_d)
    optimizer_d = uoptim.build_optimizer(
        model_params=[*image_detection_loss.discriminator.parameters()],
        **optim_params_d
    )
    scheduler_d = uoptim.build_lr_scheduler(optimizer=optimizer_d, **scheduler_params)
    print('optimizer_d: %s' % optimizer_d)
    print('scheduler_d: %s' % scheduler_d)

    # Data loaders
    train_transform, train_mask_transform = get_resize_transform(params.img_size, resize_only=params.resize_only)
    val_transform, val_mask_transform = get_resize_transform(params.img_size_val)
    image_train_loader = image_val_loader = video_train_loader = video_val_loader = None

    # TODO: allow larger number of workers (params.workers)
    # Currently set = 0 monothread causes segfaults with video compression augmentation
    # tested: VideoDatasets performance doesn't really increase with more workers
    # tested: ImageDatasets performance increase with more workers
    if params.modality in [Modalities.IMAGE, Modalities.HYBRID]:

        image_train_loader = get_dataloader_segmentation(params.image_dataset_config.train_dir,
                                                         params.image_dataset_config.train_annotation_file,
                                                         transform=train_transform,
                                                         mask_transform=train_mask_transform,
                                                         batch_size=params.batch_size,
                                                         num_workers=params.workers, shuffle=True)
        image_val_loader = get_dataloader_segmentation(params.image_dataset_config.val_dir,
                                                       params.image_dataset_config.val_annotation_file,
                                                       transform=val_transform,
                                                       mask_transform=val_mask_transform,
                                                       batch_size=params.batch_size_eval,
                                                       num_workers=params.workers,
                                                       shuffle=False,
                                                       random_nb_object=False)
    if params.modality in [Modalities.VIDEO, Modalities.HYBRID]:
        # bsz_video = 1
        # print(f"video batch size: {bsz_video}")
        video_train_loader = get_video_dataloader(params.video_dataset_config.train_dir,
                                                  batch_size=params.batch_size_video,
                                                  num_workers=params.workers,
                                                  transform=train_transform,
                                                  mask_transform=train_mask_transform,
                                                  output_resolution=params.img_size,
                                                  frames_per_clip=params.frames_per_clip,
                                                  frame_step=params.frame_step,
                                                  # TODO: Find a smart way to shuffle while making cache efficient
                                                  shuffle=True,
                                                  num_clips=params.num_clips,
                                                  )
        video_val_loader = get_video_dataloader(params.video_dataset_config.val_dir,
                                                batch_size=params.batch_size_video_eval,
                                                num_workers=params.workers,
                                                transform=val_transform,
                                                mask_transform=val_mask_transform,
                                                output_resolution=params.img_size_val,
                                                frames_per_clip=params.frames_per_clip,
                                                # TODO: Find a smart way to shuffle while making cache efficient
                                                shuffle=False,
                                                frame_step=params.frame_step,
                                                num_clips=params.num_clips,
                                                )

    # optionally resume training
    if params.resume_from is not None:
        components_to_load = {'model': wam}
        if params.resume_disc:
            components_to_load['discriminator'] = image_detection_loss.discriminator
        if params.resume_optimizer_state:
            components_to_load['optimizer'] = optimizer
            components_to_load['optimizer_d'] = optimizer_d
        uoptim.restart_from_checkpoint(
            params.resume_from,
            **components_to_load
        )

    to_restore = {
        "epoch": 0,
    }
    uoptim.restart_from_checkpoint(
        os.path.join(params.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=wam,
        discriminator=image_detection_loss.discriminator,
        optimizer=optimizer,
        optimizer_d=optimizer_d,
        scheduler=scheduler,
        scheduler_d=scheduler_d
    )
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']
    for param_group in optimizer_d.param_groups:
        param_group['lr'] = optim_params_d['lr']
    optimizers = [optimizer, optimizer_d]

    # specific thing to do if distributed training
    if params.distributed:
        # if model has batch norm convert it to sync batchnorm in distributed mode
        wam = nn.SyncBatchNorm.convert_sync_batchnorm(wam)

        wam_ddp = nn.parallel.DistributedDataParallel(
            wam, device_ids=[params.local_rank])
        image_detection_loss.discriminator = nn.parallel.DistributedDataParallel(
            image_detection_loss.discriminator, device_ids=[params.local_rank])
        wam = wam_ddp.module
    else:
        wam_ddp = wam

    dummy_img = torch.ones(3, params.img_size_val, params.img_size_val)
    validation_masks = augmenter.mask_embedder.sample_representative_masks(
        dummy_img)  # n 1 h w, full of ones or random masks depending on config

    # evaluation only
    if params.only_eval and udist.is_main_process():
        # get data loaders
        val_loaders = ((Modalities.IMAGE, image_val_loader),
                       (Modalities.VIDEO, video_val_loader))

        for val_loader, modality in val_loaders:
            if val_loader is not None:
                augs = get_validation_augs(modality == Modalities.VIDEO)

                print(f"running eval on {modality} dataset.")
                val_stats = eval_one_epoch(wam, val_loader, modality, image_detection_loss,
                                           0, augs, validation_masks, params)
                with open(os.path.join(params.output_dir, f'log_only_{modality}_eval.txt'), 'a') as f:
                    f.write(json.dumps(val_stats) + "\n")
        return

    def get_modality(epoch, params):
        # Decide on the modality of this epoch either video or images
        if params.modality == Modalities.HYBRID:
            if epoch >= params.video_start:
                if np.random.random() < params.prop_img_vid:
                    epoch_modality = Modalities.IMAGE
                else:
                    epoch_modality = Modalities.VIDEO
            else:
                epoch_modality = Modalities.IMAGE
        else:
            epoch_modality = params.modality
        return epoch_modality
    modalities = [get_modality(epoch, params)
                  for epoch in range(params.epochs)]

    # start training
    print('training...')
    start_time = time.time()
    for epoch in range(start_epoch, params.epochs):

        # prepare modality and select loader
        epoch_modality = modalities[epoch]
        assert epoch_modality in [Modalities.IMAGE, Modalities.VIDEO]
        epoch_train_loader = video_train_loader if epoch_modality == Modalities.VIDEO else image_train_loader

        # scheduler
        if scheduler is not None:
            scheduler.step(epoch)
            scheduler_d.step(epoch)
        if scaling_scheduler is not None:
            scaling_scheduler.step(epoch)

        if params.distributed:
            epoch_train_loader.sampler.set_epoch(epoch)

        # prepare if freezing the generator and finetuning the detector
        if epoch >= params.finetune_detector_start:
            # remove the grads from embedder
            wam.embedder.requires_grad_(False)
            wam.embedder.eval()
            
            # rebuild DDP with unused parameters
            wam_ddp = nn.parallel.DistributedDataParallel(
                wam, device_ids=[params.local_rank], find_unused_parameters=True)

            # set to 0 the weights of the perceptual losses
            params.lambda_i = 0.0
            params.lambda_d = 0.0
            params.balanced = False
            image_detection_loss.percep_weight = 0.0
            image_detection_loss.disc_weight = 0.0
            image_detection_loss.balanced = False  # not supported here because embedder is frozen

        # train and log
        train_stats = train_one_epoch(wam_ddp, optimizers, epoch_train_loader, epoch_modality, image_detection_loss, epoch, params, tensorboard=tensorboard)
        log_stats = {
            'epoch': epoch, 'modality': epoch_modality, 
            **{f'train_{k}': v for k, v in train_stats.items()}
        }

        if epoch % params.eval_freq == 0:
            val_loaders = ((Modalities.IMAGE, image_val_loader),
                        (Modalities.VIDEO, video_val_loader))
            for val_modality, epoch_val_loader in val_loaders:
                if epoch_val_loader is not None:
                    if (epoch % params.full_eval_freq == 0 and epoch > 0) or (epoch == params.epochs-1):
                        augs = get_validation_augs(val_modality == Modalities.VIDEO)
                    else:
                        augs = get_validation_augs_subset(val_modality == Modalities.VIDEO)
                    val_stats = eval_one_epoch(wam, epoch_val_loader, val_modality, image_detection_loss,
                                            epoch, augs, validation_masks, params, tensorboard=tensorboard)
                    log_stats = {
                        **log_stats, **{f'val_{val_modality}_{k}': v for k, v in val_stats.items()}}

        if udist.is_main_process():
            with open(os.path.join(params.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")
        if udist.is_dist_avail_and_initialized():
            dist.barrier()  # Ensures all processes wait until the main node finishes validation

        print("Saving Checkpoint..")
        discrim_no_ddp = image_detection_loss.discriminator.module if params.distributed else image_detection_loss.discriminator
        save_dict = {
            'epoch': epoch + 1,
            'model': wam.state_dict(),
            'discriminator': discrim_no_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'scheduler_d': scheduler_d.state_dict() if scheduler_d is not None else None,
            'args': omegaconf.OmegaConf.to_yaml(params),
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
    epoch_modality: str,
    image_detection_loss: VideosealLoss,
    epoch: int,
    params: argparse.Namespace,
    tensorboard: CustomTensorboardWriter
) -> dict:
    is_video = (epoch_modality == Modalities.VIDEO)

    wam.train()

    header = f'Train - Epoch: [{epoch}/{params.epochs}] - Modality: {epoch_modality}'
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    for it, batch_items in enumerate(metric_logger.log_every(train_loader, 10, header)):
        if it >= params.iter_per_epoch:
            break

        # some data loaders return batch_data, masks, frames_positions as well
        batch_imgs, batch_masks = batch_items[0], batch_items[1]

        # videos are too big to have a batch of them
        # so we do batch accumulation with bsz = 1
        if len(batch_imgs.shape) == 5:
            accumulation_steps = batch_imgs.shape[0]
        elif len(batch_imgs.shape) == 4:
            accumulation_steps = 1
            batch_masks = batch_masks.unsqueeze(0)
            batch_imgs = batch_imgs.unsqueeze(0)

        if params.lambda_d == 0:  # no disc, optimize embedder/extractor only
            optimizer_ids_for_epoch = [0]
        else:
            if params.sleepwake:  # alternate
                optimizer_ids_for_epoch = [epoch % 2]
            else:  # both during the same epoch
                optimizer_ids_for_epoch = [1, 0]

        # reset the optimizer gradients before accum gradients
        for optimizer_idx in optimizer_ids_for_epoch:
            optimizers[optimizer_idx].zero_grad()

        # accumulate gradients
        for acc_it in range(accumulation_steps):

            imgs, masks = batch_imgs[acc_it], batch_masks[acc_it]
            imgs = imgs.to(device, non_blocking=True)

            # forward
            outputs = wam(imgs, masks, is_video=is_video)
            outputs["preds"] /= params.temperature

            # last layer is used for gradient scaling
            last_layer = wam.embedder.get_last_layer() if not params.distributed else wam.module.embedder.get_last_layer()

            # index 1 for discriminator, 0 for embedder/extractor
            for optimizer_idx in optimizer_ids_for_epoch:
                loss, logs = image_detection_loss(
                    imgs, outputs["imgs_w"],
                    outputs["masks"], outputs["msgs"], outputs["preds"],
                    optimizer_idx, epoch,
                    last_layer=last_layer,
                )
                # Scale loss for accumulation so lr is not affected
                loss = loss / accumulation_steps
                loss.backward()

            # log stats
            log_stats = {
                **logs,
                'psnr': psnr(outputs["imgs_w"], imgs).mean().item(),
                'ssim': ssim(outputs["imgs_w"], imgs).mean().item(),
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
            for name, value in log_stats.items():
                metric_logger.update(**{name: value})

            # save images on training
            if (epoch % params.saveimg_freq == 0) and it == acc_it == 0:
            # if (epoch % params.saveimg_freq == 0) and (it % 50) == 0:
                ori_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_{epoch_modality}_train_0_ori.png')
                wm_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_{epoch_modality}_train_1_wm.png')
                diff_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_{epoch_modality}_train_2_diff.png')
                aug_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_{epoch_modality}_train_3_aug_{outputs["selected_aug"]}.png')
                if udist.is_main_process():
                    save_image(imgs, ori_path, nrow=8)
                    tensorboard.add_images("TRAIN/IMAGES/orig", imgs, epoch)
                    save_image(outputs["imgs_w"], wm_path, nrow=8)
                    tensorboard.add_images(
                        "TRAIN/IMAGES/wmed", outputs["imgs_w"], epoch)
                    save_image(create_diff_img(
                        imgs, outputs["imgs_w"]), diff_path, nrow=8)
                    tensorboard.add_images("TRAIN/IMAGES/diff", create_diff_img(
                        imgs, outputs["imgs_w"]), epoch)
                    save_image(outputs["imgs_aug"], aug_path, nrow=8)
                    tensorboard.add_images(
                        "TRAIN/IMAGES/aug", outputs["imgs_aug"], epoch)

        # end accumulate gradients batches
        # add optimizer step
        for optimizer_idx in optimizer_ids_for_epoch:
            optimizers[optimizer_idx].step()

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    train_logs = {k: meter.global_avg for k,
                  meter in metric_logger.meters.items()}

    tensorboard.add_scalars("TRAIN/LOSS", train_logs, epoch)

    return train_logs


@ torch.no_grad()
def eval_one_epoch(
    wam: Wam,
    val_loader: torch.utils.data.DataLoader,
    epoch_modality: str,
    image_detection_loss: VideosealLoss,
    epoch: int,
    validation_augs: List,
    validation_masks: torch.Tensor,
    params: argparse.Namespace,
    tensorboard: CustomTensorboardWriter,
) -> dict:
    """
    Evaluate the model on the validation set, with different augmentations

    Args:
        wam (Wam): the model
        val_loader (torch.utils.data.DataLoader): the validation loader
        image_detection_loss (VideosealLoss): the loss function
        epoch (int): the current epoch
        validation_augs (List): list of augmentations to apply
        validation_masks (torch.Tensor): the validation masks, full of ones for now
        params (argparse.Namespace): the parameters
    """
    is_video = (epoch_modality == Modalities.VIDEO)
    if torch.is_tensor(validation_masks):
        validation_masks = list(torch.unbind(validation_masks, dim=0))

    wam.eval()

    header = f'Val - Epoch: [{epoch}/{params.epochs}] - Modality: {epoch_modality}'
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    for it, batch_items in enumerate(metric_logger.log_every(val_loader, 10, header)):
        if params.iter_per_valid is not None and it >= params.iter_per_valid:
            break

        # some data loaders return batch_data, masks, frames_positions as well
        batch_imgs, batch_masks = batch_items[0], batch_items[1]

        # videos are too big to have a batch of them
        # so we do batch accumulation with bsz = 1
        if len(batch_imgs.shape) == 5:  # b f c h w
            accumulation_steps = batch_imgs.shape[0]
        elif len(batch_imgs.shape) == 4:  # b c h w
            accumulation_steps = 1
            batch_masks = batch_masks.unsqueeze(0)  # 1 b 1 h w
            batch_imgs = batch_imgs.unsqueeze(0)  # 1 b c h w

        for acc_it in range(accumulation_steps):
            imgs, masks = batch_imgs[acc_it], batch_masks[acc_it]

            imgs = imgs.to(device)
            masks = masks.to(device)

            # forward embedder
            embed_time = time.time()
            outputs = wam.embed(imgs, is_video=is_video, lowres_attenuation=params.lowres_attenuation)
            embed_time = (time.time() - embed_time) / imgs.shape[0]
            msgs = outputs["msgs"].to(device)  # b k
            imgs_w = outputs["imgs_w"]  # b c h w

            if (epoch % params.saveimg_freq == 0) and it == acc_it == 0 and udist.is_main_process():
                base_name = os.path.join(
                    params.output_dir, f'{epoch:03}_{acc_it*it:03}_{epoch_modality}_val')
                ori_path = base_name + '_0_ori.png'
                wm_path = base_name + '_1_wm.png'
                diff_path = base_name + '_2_diff.png'
                save_image(imgs, ori_path, nrow=8)
                save_image(imgs_w, wm_path, nrow=8)
                save_image(create_diff_img(imgs, imgs_w), diff_path, nrow=8)
                tensorboard.add_images(
                    "VALID/IMAGES/orig", imgs, acc_it*it*epoch)
                tensorboard.add_images(
                    "VALID/IMAGES/wmed", imgs_w, acc_it*it*epoch)
                tensorboard.add_images(
                    "VALID/IMAGES/diff", create_diff_img(imgs, imgs_w), acc_it*it*epoch)

                if epoch_modality == Modalities.VIDEO:
                    fps = 24 // 1
                    ori_path = ori_path.replace(".png", ".mp4")
                    wm_path = wm_path.replace(".png", ".mp4")
                    diff_path = diff_path.replace(".png", ".mp4")
                    save_vid(imgs, ori_path, fps)
                    save_vid(imgs_w, wm_path, fps)
                    save_vid(create_diff_img(imgs, imgs_w), diff_path, fps)
                    tensorboard.add_video(
                        "VALID/VIDEOS/orig", imgs.unsqueeze(0), acc_it*it*epoch, fps)
                    tensorboard.add_video(
                        "VALID/VIDEOS/wmed", imgs_w.unsqueeze(0), acc_it*it*epoch, fps)
                    tensorboard.add_video(
                        "VALID/VIDEOS/diff", create_diff_img(imgs, imgs_w).unsqueeze(0), acc_it*it*epoch, fps)

            # quality metrics
            metrics = {}
            metrics['psnr'] = psnr(imgs_w, imgs).mean().item()
            metrics['ssim'] = ssim(imgs_w, imgs).mean().item()
            metrics['embed_time'] = embed_time
            torch.cuda.synchronize()
            metric_logger.update(**metrics)

            extract_times = []
            for mask_id, masks in enumerate(validation_masks):
                # watermark masking
                masks = masks.to(imgs.device)  # 1 h w
                if len(masks.shape) < 4:
                    masks = masks.unsqueeze(0).repeat(
                        imgs_w.shape[0], 1, 1, 1)  # b 1 h w
                imgs_masked = imgs_w * masks + imgs * (1 - masks)

                for transform_instance, strengths in validation_augs:

                    for strength in strengths:
                        imgs_aug, masks_aug = transform_instance(
                                imgs_masked, masks, strength)
                        selected_aug = str(transform_instance) + f"_{strength}"
                        selected_aug = selected_aug.replace(", ", "_")

                        # extract watermark
                        extract_time = time.time()
                        outputs = wam.detect(imgs_aug, is_video=is_video)
                        extract_time = time.time() - extract_time
                        extract_times.append(extract_time / imgs_aug.shape[0])
                        preds = outputs["preds"]
                        mask_preds = preds[:, 0:1]  # b 1 ...
                        bit_preds = preds[:, 1:]  # b k ...

                        aug_log_stats = {}
                        if params.nbits > 0:
                            bit_accuracy_ = bit_accuracy(
                                bit_preds,
                                msgs,
                                masks_aug
                            ).nanmean().item()

                        if params.nbits > 0:
                            aug_log_stats[f'bit_acc'] = bit_accuracy_

                        if params.lambda_det > 0:
                            iou0 = iou(mask_preds, masks,
                                       label=0).mean().item()
                            iou1 = iou(mask_preds, masks,
                                       label=1).mean().item()
                            aug_log_stats.update({
                                f'acc': accuracy(mask_preds, masks).mean().item(),
                                f'miou': (iou0 + iou1) / 2,
                            })

                        current_key = f"mask={mask_id}_aug={selected_aug}"
                        aug_log_stats = {f"{k}_{current_key}": v for k,
                                         v in aug_log_stats.items()}

                        torch.cuda.synchronize()
                        metric_logger.update(**aug_log_stats)

            metrics['extract_time'] = np.mean(extract_times)
            torch.cuda.synchronize()
            metric_logger.update(**metrics)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('val'), metric_logger)
    valid_logs = {k: meter.global_avg for k,
                  meter in metric_logger.meters.items()}
    tensorboard.add_scalars("VALID", valid_logs, epoch)
    return valid_logs


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
