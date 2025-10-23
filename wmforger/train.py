# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import os
import time
from typing import Tuple
from PIL import Image
import random
import numpy as np
import omegaconf

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

import wmforger.utils as utils
import wmforger.utils.dist as udist
import wmforger.utils.logger as ulogger
import wmforger.utils.optim as uoptim
from wmforger.augmentation.augmenter import Augmenter
from wmforger.data.loader import get_dataloader_segmentation
from wmforger.data.transforms import get_resize_transform
from wmforger.models import build_extractor
from wmforger.utils.data import Modalities, parse_dataset_params
from wmforger.utils.tensorboard import CustomTensorboardWriter
from wmforger.modules.watermark_generators import FFTWatermarkWaves, FFTWatermarkGaussian, FFTWatermarkLines

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def random_crop(batch, size):
    _, b, c, h, w = batch.shape
    ch = torch.randint(0, h - size + 1, (b,))
    cw = torch.randint(0, w - size + 1, (b,))
    cropped_batch = torch.empty((2, b, c, size, size), dtype=batch.dtype, device=batch.device)
    for i in range(b):
        cropped_batch[:, i] = batch[:, i, :, ch[i]:ch[i]+size, cw[i]:cw[i]+size]
    return cropped_batch



def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Dataset parameters')
    aa("--image_dataset", type=str,  help="Name of the image dataset.", default="sa-1b-full")
    aa("--video_dataset", type=str, help="not used",  default="none")

    group = parser.add_argument_group('Experiments parameters')
    aa("--output_dir", type=str, default="output/",
       help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Embedder and extractor config')
    aa("--embedder_list", type=str, default="artificial/fft_waves,artificial/fft_gaussian,artificial/fft_lines")
    aa("--embedder_list_valid", type=str, default="artificial/fft_waves")
    aa("--extractor_config", type=str, default="configs/extractor.yaml", help="Path to the extractor config file")
    aa("--extractor_model", type=str, default="convnext_tiny", help="Name of the extractor model")

    group = parser.add_argument_group('Augmentation parameters')
    aa("--augmentation_config", type=str, default="configs/all_augs_v3.yaml", help="Path to the augmentation config file")
    aa("--num_augs", type=int, default=2, help="Number of augmentations to apply")

    group = parser.add_argument_group('Image and watermark parameters')
    aa("--img_size", type=int, default=768,
       help="Size of the input images for data preprocessing, used at loading time for training.")
    aa("--img_size_val", type=int, default=768,
       help="Size of the input images for data preprocessing, used at loading time for validation.")
    aa("--img_size_proc", type=int, default=256, 
       help="Size of the input images for interpolation in the embedder/extractor models")
    aa("--resize_only", type=utils.bool_inst, default=False,
         help="If True, only resize the image no crop is applied at loading time (without preserving aspect ratio)")
    # VideoWam parameters related how to do video watermarking inference

    group = parser.add_argument_group('Optimizer parameters')
    aa("--optimizer", type=str, default="AdamW,lr=5e-5", help="Optimizer")
    aa("--scheduler", type=str, default="None", help="Scheduler (default: None)")
    aa('--epochs', default=60, type=int, help='Number of total epochs to run')
    aa('--iter_per_epoch', default=1000, type=int, help='Number of iterations per epoch, made for very large datasets')
    aa('--iter_per_valid', default=50, type=int, help='Number of iterations per eval, made for very large eval datasets if None eval on all dataset')
    aa('--resume_from', default=None, type=str, help='Path to the checkpoint to resume from')
    aa('--resume_optimizer_state', type=utils.bool_inst, default=False, help='If True, also load optimizer state when resuming from checkpoint')

    group = parser.add_argument_group('Losses parameters')
    aa('--loss_type', default='bt_nll', type=str, help='Loss to use.', choices=['bce', 'bt_nll'])
    aa('--grad_perturbation', type=utils.bool_inst, default=True)
    aa('--use_grad_sign_only', type=utils.bool_inst, default=False)
    aa('--use_rand_perturbation', type=utils.bool_inst, default=False)
    aa('--max_perturbation', default=0.09, type=float)
    aa('--min_perturbation', default=0.03, type=float)
    aa('--n_perturbation_steps', default=2, type=int)
    aa('--watermark_strength_contrasting', type=utils.bool_inst, default=False)
    aa('--strength_contrasting_single_watermark', type=utils.bool_inst, default=False)
    aa('--weak_alpha', default=0.5, type=float)
    aa('--strong_alpha', default=2, type=float)
    aa('--alpha_range', default=0.4, type=float)
    aa('--ramdomly_invert_watermark', type=utils.bool_inst, default=True)
    aa('--grad_matching', type=utils.bool_inst, default=False)
    aa('--grad_matching_weight', default=0.2, type=float)
    
    group = parser.add_argument_group('Loading parameters')
    aa('--batch_size', default=16, type=int, help='Batch size')
    aa('--batch_size_eval', default=16, type=int, help='Batch size for evaluation')
    aa('--workers', default=3, type=int, help='Number of data loading workers')

    group = parser.add_argument_group('Misc.')
    aa('--eval_freq', default=1, type=int, help='Frequency for evaluation')
    aa('--saveckpt_freq', default=2, type=int, help='Frequency for saving ckpts')
    aa('--seed', default=75427, type=int, help='Random seed')

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)

    return parser


def construct_loss(loss_type):
    if loss_type == "bce":
        def fc_(real_logits, wm_logits):
            return F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + \
                   F.binary_cross_entropy_with_logits(wm_logits, torch.zeros_like(wm_logits))
        return fc_
    elif loss_type == "bt_nll":
        # https://arxiv.org/pdf/2305.18290, Eq. (2)
        def fc_(real_logits, wm_logits):
            return F.binary_cross_entropy_with_logits(real_logits - wm_logits, torch.ones_like(real_logits))
        return fc_
    else:
        raise NotImplementedError(f"Loss {loss_type} is not implemented.")

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

    ###################################################################################################
    fft_wm_mapping = {
        "artificial/fft_waves": FFTWatermarkWaves,
        "artificial/fft_gaussian": FFTWatermarkGaussian,
        "artificial/fft_lines": FFTWatermarkLines,
    }

    # BUILD EMBEDDER based on the gpu index
    embedder_list = params.embedder_list.split(",")
    embedder_list_idx = udist.get_rank() % len(embedder_list)
    
    if embedder_list[embedder_list_idx].startswith("artificial"):
        assert embedder_list[embedder_list_idx] in fft_wm_mapping
        embedder = fft_wm_mapping[embedder_list[embedder_list_idx]]().to(device)
    else:
        raise NotImplementedError(f"Embedder {embedder_list[embedder_list_idx]} is not implemented.")
    
    for p in embedder.parameters():
        p.requires_grad_ = False
    embedder.eval()

    # BUILD validation EMBEDDER
    embedder_list_valid = params.embedder_list_valid.split(",")
    embedder_list_valid_idx = udist.get_rank() % len(embedder_list_valid)
    
    if embedder_list_valid[embedder_list_valid_idx].startswith("artificial"):
        assert embedder_list_valid[embedder_list_valid_idx] in fft_wm_mapping
        embedder_valid = fft_wm_mapping[embedder_list_valid[embedder_list_valid_idx]]().to(device)
    else:
        raise NotImplementedError(f"Embedder {embedder_list_valid[embedder_list_valid_idx]} is not implemented.")
    
    for p in embedder_valid.parameters():
        p.requires_grad_ = False
    embedder_valid.eval()

    ###################################################################################################

    # build the augmenter
    augmenter_cfg = omegaconf.OmegaConf.load(params.augmentation_config)
    augmenter_cfg.num_augs = params.num_augs
    augmenter = Augmenter(
        **augmenter_cfg,
    ).to(device)
    print(f'augmenter: {augmenter}')

    # Build the extractor model
    extractor_cfg = omegaconf.OmegaConf.load(params.extractor_config)
    params.extractor_model = params.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[params.extractor_model]
    extractor = build_extractor(params.extractor_model, extractor_params, params.img_size_proc, nbits=0).to(device)
    print(f'extractor: {sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6:.1f}M trainable parameters')

    loss_function = construct_loss(params.loss_type)

    # Build optimizer and scheduler
    model_params = list(extractor.parameters())
    optim_params = uoptim.parse_params(params.optimizer)
    optimizer = uoptim.build_optimizer(model_params, **optim_params)
    scheduler_params = uoptim.parse_params(params.scheduler)
    scheduler = uoptim.build_lr_scheduler(optimizer, **scheduler_params)
    print('optimizer: %s' % optimizer)
    print('scheduler: %s' % scheduler)

    # Data loaders
    train_transform, train_mask_transform = get_resize_transform(params.img_size, resize_only=params.resize_only)
    val_transform, val_mask_transform = get_resize_transform(params.img_size_val)
    train_loader = val_loader = None

    # TODO: allow larger number of workers (params.workers)
    # Currently set = 0 monothread causes segfaults with video compression augmentation
    # tested: VideoDatasets performance doesn't really increase with more workers
    # tested: ImageDatasets performance increase with more workers
    assert params.modality == Modalities.IMAGE
    train_loader = get_dataloader_segmentation(params.image_dataset_config.train_dir,
                                                params.image_dataset_config.train_annotation_file,
                                                transform=train_transform,
                                                mask_transform=train_mask_transform,
                                                batch_size=params.batch_size,
                                                num_workers=params.workers, shuffle=True)
    val_loader = get_dataloader_segmentation(params.image_dataset_config.val_dir,
                                                params.image_dataset_config.val_annotation_file,
                                                transform=val_transform,
                                                mask_transform=val_mask_transform,
                                                batch_size=params.batch_size_eval,
                                                num_workers=params.workers,
                                                shuffle=False)

    # optionally resume training
    if params.resume_from is not None:
        components_to_load = {'model': extractor}
        if params.resume_optimizer_state:
            components_to_load['optimizer'] = optimizer
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
        model=extractor,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']

    # specific thing to do if distributed training
    if params.distributed:
        # if model has batch norm convert it to sync batchnorm in distributed mode
        extractor = nn.SyncBatchNorm.convert_sync_batchnorm(extractor)

        extractor_ddp = nn.parallel.DistributedDataParallel(
            extractor, device_ids=[params.local_rank])
        extractor = extractor_ddp.module
    else:
        extractor_ddp = extractor


    # start training
    print('training...')
    start_time = time.time()
    for epoch in range(start_epoch, params.epochs):
        # scheduler
        if scheduler is not None:
            scheduler.step(epoch)

        if params.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train and log
        model_tuple = (embedder, extractor_ddp, augmenter)
        train_stats = train_one_epoch(model_tuple, optimizer, train_loader, loss_function, epoch, params, tensorboard=tensorboard)
        log_stats = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()}
        }

        if epoch % params.eval_freq == 0:
            model_tuple = (embedder_valid, extractor_ddp, augmenter)
            val_stats = eval_one_epoch(model_tuple, optimizer, val_loader, epoch, params, tensorboard=tensorboard)
            log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}

        if udist.is_main_process():
            with open(os.path.join(params.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")
        if udist.is_dist_avail_and_initialized():
            dist.barrier()  # Ensures all processes wait until the main node finishes validation

        print("Saving Checkpoint..")
        save_dict = {
            'epoch': epoch + 1,
            'model': extractor.state_dict(),
            'optimizer': optimizer.state_dict(),
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
    model_tuple: Tuple,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    loss_function,
    epoch: int,
    params: argparse.Namespace,
    tensorboard: CustomTensorboardWriter
) -> dict:
    embedder, extractor, augmenter = model_tuple

    extractor.train()
    header = f'Train - Epoch: [{epoch}/{params.epochs}]'
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    for it, batch_items in enumerate(metric_logger.log_every(train_loader, 10, header)):
        if it >= params.iter_per_epoch:
            break

        # some data loaders return batch_data, masks, frames_positions as well
        batch_imgs, batch_masks = batch_items[0], batch_items[1]

        # videos are too big to have a batch of them
        # so we do batch accumulation with bsz = 1
        if len(batch_imgs.shape) == 5:
            assert len(batch_imgs) == 1
        elif len(batch_imgs.shape) == 4:
            batch_masks = batch_masks.unsqueeze(0)
            batch_imgs = batch_imgs.unsqueeze(0)

        optimizer.zero_grad()

        imgs, masks = batch_imgs[0], batch_masks[0]
        imgs = imgs.to(device, non_blocking=True)

        with torch.no_grad():
            # forward
            outputs = embedder.embed(imgs, is_video=params.modality == Modalities.VIDEO)
            watermarked_images = outputs["imgs_w"]

            if params.ramdomly_invert_watermark:
                if random.random() < 0.5:
                    watermarked_images = (imgs - (watermarked_images - imgs)).clip(0, 1)

            joined_imgs = torch.cat([imgs, watermarked_images], 0)
            joined_imgs_aug, masks, _ = augmenter.augment(
                joined_imgs, torch.cat([masks] * 2, 0), is_video=params.modality == Modalities.VIDEO, do_resize=True)
            joined_imgs_aug = joined_imgs_aug.view(2, -1, *imgs.shape[1:])

            if params.img_size != params.img_size_proc:
                assert params.img_size > params.img_size_proc
                joined_imgs_aug = random_crop(joined_imgs_aug, params.img_size_proc)

        original_images_probs = extractor(joined_imgs_aug[0])
        if params.grad_matching:
            perturbation = torch.zeros_like(joined_imgs_aug[1])
            perturbation.requires_grad_(True)
            watermarked_images_probs = extractor(joined_imgs_aug[1] + perturbation)
        else:
            watermarked_images_probs = extractor(joined_imgs_aug[1])

        loss = loss_function(original_images_probs, watermarked_images_probs)

        accuracy = ((original_images_probs > 0).float().mean() + (watermarked_images_probs < 0).float().mean()) / 2.
        ranking = ((original_images_probs  - watermarked_images_probs) > 0).float().mean()

        # log stats
        log_stats = {
            'loss': loss.item(),
            'acc': accuracy.item(),
            'ranking': ranking.item(),
            'lr': optimizer.param_groups[0]['lr'],
        }

        if params.grad_matching:
            grad_ = torch.autograd.grad(watermarked_images_probs.mean(), perturbation, create_graph=True)
            watermark = joined_imgs_aug[1] - joined_imgs_aug[0]
            loss_2ndorder = (1-F.cosine_similarity(grad_[0].view(-1), -watermark.view(-1), dim=0)) * params.grad_matching_weight
            log_stats["loss_2ndorder"] = loss_2ndorder.item()
            loss += loss_2ndorder

        loss.backward()

        if params.watermark_strength_contrasting:
            watermark1 = watermark2 = watermarked_images - imgs
            if not params.strength_contrasting_single_watermark:
                # run the embedder for the second time with different watermark message
                with torch.no_grad():
                    outputs2 = embedder.embed(imgs, is_video=params.modality == Modalities.VIDEO)
                    watermarked_images2 = outputs2["imgs_w"]
                watermark2 = watermarked_images2 - imgs

            aplha1 = params.weak_alpha + random.random() * params.alpha_range - params.alpha_range / 2
            aplha2 = params.strong_alpha + random.random() * params.alpha_range - params.alpha_range / 2

            watermarked_images_weak = (imgs + aplha1 * watermark1).clip(0, 1)
            watermarked_images_strong = (imgs + aplha2 * watermark2).clip(0, 1)
            joined_imgs_contrasting = torch.stack([watermarked_images_weak, watermarked_images_strong], 0)

            if params.img_size != params.img_size_proc:
                assert params.img_size > params.img_size_proc
                joined_imgs_contrasting = random_crop(joined_imgs_contrasting, params.img_size_proc)

            weak_probs = extractor(joined_imgs_contrasting[0])
            strong_probs = extractor(joined_imgs_contrasting[1])

            loss_wm_contrasting = loss_function(weak_probs, strong_probs)
            loss_wm_contrasting.backward()
            log_stats["loss_wm_contrasting"] = loss_wm_contrasting.item()

        # do joint step for the main loss, grad matching loss, and the watermark_strength_contrasting, if enabled
        if params.grad_matching:
            # the norm of the gradients in the cosine loss can be small and produce inf/nan gradients, check for that
            if all([torch.isfinite(p.grad).all().item() for p in extractor.parameters()]):
                optimizer.step()
            else:
                print("WARINING: Some gradients are not finite! Skipping the update step!", flush=True)
        else:
            optimizer.step()

        if params.grad_perturbation:
            if params.use_rand_perturbation:
                perturbation = torch.rand_like(joined_imgs_aug[1]).mul_(2).sub_(1).mul_(params.min_perturbation)
            else:
                perturbation = torch.zeros_like(joined_imgs_aug[1])
            perturbation.requires_grad_(True)

            for _ in range(params.n_perturbation_steps):
                perturbation_loss = -extractor(joined_imgs_aug[1] + perturbation).mean()
                perturbation_loss.backward()
                perturbation_lr = random.random() * (params.max_perturbation - params.min_perturbation) + params.min_perturbation
                
                perturbation_vec = perturbation.grad.detach()
                if params.use_grad_sign_only:
                    perturbation_vec.sign_()
                perturbation_vec.mul_(-perturbation_lr)

                with torch.no_grad():
                    perturbation.add_(perturbation_vec)
                    perturbation.grad.zero_()

            perturbed_images = (joined_imgs_aug[1] + perturbation).clip(0, 1)

            optimizer.zero_grad()
            original_images_probs = extractor(joined_imgs_aug[0])
            watermarked_images_probs = extractor(perturbed_images)

            loss_gradpert = loss_function(original_images_probs, watermarked_images_probs)
            loss_gradpert.backward()
            log_stats["loss_gradpert"] = loss_gradpert.item()

            optimizer.step()

        torch.cuda.synchronize()
        for name, value in log_stats.items():
            metric_logger.update(**{name: value})

        if it % 200 == 199:
            inputs_ = Image.fromarray(np.concatenate(np.concatenate(joined_imgs_aug.permute(0, 1, 3, 4, 2).mul(255).to(torch.uint8).cpu().numpy()[:, :6], 1), 1))
            inputs_.save(os.path.join(params.output_dir, f'{epoch:03}_{it:03}_{udist.get_rank()}_input.png'))
            if params.grad_perturbation:
                inputs_ = Image.fromarray(np.concatenate(perturbed_images.permute(0, 2, 3, 1).mul(255).to(torch.uint8).cpu().numpy()[:6], 1))
                inputs_.save(os.path.join(params.output_dir, f'{epoch:03}_{it:03}_{udist.get_rank()}_perturbed.png'))
            if params.watermark_strength_contrasting:
                inputs_ = Image.fromarray(np.concatenate(np.concatenate(joined_imgs_contrasting.permute(0, 1, 3, 4, 2).mul(255).to(torch.uint8).cpu().numpy()[:, :6], 1), 1))
                inputs_.save(os.path.join(params.output_dir, f'{epoch:03}_{it:03}_{udist.get_rank()}_contrasting.png'))

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    train_logs = {k: meter.global_avg for k,
                  meter in metric_logger.meters.items()}

    tensorboard.add_scalars("TRAIN", train_logs, epoch)

    return train_logs


@torch.no_grad()
def eval_one_epoch(
    model_tuple: Tuple,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    params: argparse.Namespace,
    tensorboard: CustomTensorboardWriter
) -> dict:
    embedder, extractor, augmenter = model_tuple

    extractor.eval()
    header = f'Val - Epoch: [{epoch}/{params.epochs}]'
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    for it, batch_items in enumerate(metric_logger.log_every(train_loader, 10, header)):
        if params.iter_per_valid is not None and it >= params.iter_per_valid:
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


        optimizer.zero_grad()

        # accumulate gradients
        for acc_it in range(accumulation_steps):

            imgs, masks = batch_imgs[acc_it], batch_masks[acc_it]
            imgs = imgs.to(device, non_blocking=True)

            with torch.no_grad():
                # forward
                outputs = embedder.embed(imgs, is_video=params.modality == Modalities.VIDEO)
                watermarked_images = outputs["imgs_w"]

                joined_imgs = torch.cat([imgs, watermarked_images], 0)
                joined_imgs_aug, masks, _ = augmenter.augment(
                    joined_imgs, torch.cat([masks] * 2, 0), is_video=params.modality == Modalities.VIDEO, do_resize=True)
                joined_imgs_aug = joined_imgs_aug.view(2, -1, *imgs.shape[1:])

                if params.img_size_val != params.img_size_proc:
                    assert params.img_size_val > params.img_size_proc
                    joined_imgs_aug = random_crop(joined_imgs_aug, params.img_size_proc)

            original_images_probs = extractor(joined_imgs_aug[0])
            watermarked_images_probs = extractor(joined_imgs_aug[1])

            accuracy = ((original_images_probs > 0).float().mean() + (watermarked_images_probs < 0).float().mean()) / 2.
            ranking = ((original_images_probs  - watermarked_images_probs) > 0).float().mean()

            # log stats
            log_stats = {
                'acc': accuracy.item(),
                'ranking': ranking.item(),
            }

            torch.cuda.synchronize()
            for name, value in log_stats.items():
                metric_logger.update(**{name: value})

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
