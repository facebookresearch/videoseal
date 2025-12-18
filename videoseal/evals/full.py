# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example:

1/ Evaluate a checkpoint on images

python -m videoseal.evals.full \
    --checkpoint /path/to/your/ckpt \
    --lowres_attenuation True --scaling_w 0.2 \
    --dataset sa-1b-full-resized --is_video false --num_samples 10 --save_first 10 

2/ Evaluate a checkpoint on videos

python -m videoseal.evals.full \
    --checkpoint /path/to/your/ckpt \
    --lowres_attenuation True --scaling_w 0.2 \
    --dataset sa-v --is_video true --num_samples 1 --save_first 1

3/ Evaluate a baseline. Use the checkpoint path as the method name

--checkpoint baseline/wam


Arguments invertory:
    --dataset coco --is_video false \
    --dataset sa-v --is_video true --num_samples 1 \   
"""
    
import argparse
import os

import numpy as np
import omegaconf
import pandas as pd
import tqdm
from lpips import LPIPS

import torch
from torch.utils.data import Dataset, Subset
from torchvision.utils import save_image

from .metrics import vmaf_on_tensor, bit_accuracy, iou, accuracy, pvalue, capacity, psnr, ssim, msssim, bd_rate
from ..augmentation import get_validation_augs
from ..models import Videoseal
from ..modules.jnd import JND
from ..utils import Timer, bool_inst
from ..utils.display import save_vid
from ..utils.image import create_diff_img
from ..utils.cfg import setup_dataset, setup_model_from_checkpoint
from .metrics import accuracy, bit_accuracy, iou, vmaf_on_tensor

@torch.no_grad()
def evaluate(
    model: Videoseal,
    dataset: Dataset, 
    is_video: bool,
    output_dir: str,
    save_first: int = -1,
    num_frames: int = 24*3,
    video_aggregation: str = "avg",
    only_identity: bool = False,
    only_combined: bool = False,
    bdrate: bool = True,
    decoding: bool = True,
    detection: bool = False,
    interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    lowres_attenuation: bool = False,
    skip_image_metrics: bool = False,
):
    """
    Gives detailed evaluation metrics for a model on a given dataset.
    Args:
        model (Videoseal): The model to evaluate
        dataset (Dataset): The dataset to evaluate on
        is_video (bool): Whether the data is video
        output_dir (str): Directory to save the output images
        num_frames (int): Number of frames to evaluate for video quality and extraction (default: 24*3 i.e. 3seconds)
        video_aggregation (str): Aggregation method for detection of video frames
        only_identity (bool): Whether to only evaluate the identity augmentation
        only_combined (bool): Whether to only evaluate combined augmentations
        decoding (bool): Whether to evaluate decoding metrics (default: True)
        detection (bool): Whether to evaluate detection metrics (default: False)
        skip_image_metrics (bool): Whether to skip computing image quality metrics (default: False)
    """
    all_metrics = []
    validation_augs = get_validation_augs(is_video, only_identity, only_combined)
    timer = Timer()

    # create lpips
    lpips_loss = LPIPS(net="alex").eval() if not skip_image_metrics else None

    # save the metrics as csv
    metrics_path = os.path.join(output_dir, "metrics.csv")
    print(f"Saving metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
            
        for it, batch_items in enumerate(tqdm.tqdm(dataset)):
            # initialize metrics
            metrics = {}

            # some data loaders return batch_data, masks, frames_positions as well
            if batch_items is None:
                continue
            imgs, masks = batch_items[0], batch_items[1]
            if not is_video:
                imgs = imgs.unsqueeze(0)  # c h w -> 1 c h w
                masks = masks.unsqueeze(0) if isinstance(masks, torch.Tensor) else masks
            metrics['iteration'] = it
            metrics['t'] = imgs.shape[-4]
            metrics['h'] = imgs.shape[-2]
            metrics['w'] = imgs.shape[-1]

            # forward embedder, at any resolution
            # does cpu -> gpu -> cpu when gpu is available
            timer.start()
            outputs = model.embed(imgs, is_video=is_video, interpolation=interpolation, lowres_attenuation=lowres_attenuation)
            metrics['embed_time'] = timer.end()
            msgs = outputs["msgs"]  # b k
            imgs_w = outputs["imgs_w"]  # b c h w

            # cut frames
            imgs = imgs[:num_frames]  # f c h w
            msgs = msgs[:num_frames]  # f k
            imgs_w = imgs_w[:num_frames]  # f c h w
            masks = masks[:num_frames]  # f 1 h w

            # compute qualitative metrics
            if not skip_image_metrics:
                metrics['psnr'] = psnr(
                    imgs_w[:num_frames], 
                    imgs[:num_frames],
                    is_video).mean().item()
                metrics['ssim'] = ssim(
                    imgs_w[:num_frames], 
                    imgs[:num_frames]).mean().item()
                metrics['msssim'] = msssim(
                    imgs_w[:num_frames], 
                    imgs[:num_frames]).mean().item()
                metrics['lpips'] = lpips_loss(
                    2*imgs_w[:num_frames]-1, 
                    2*imgs[:num_frames]-1).mean().item()
                if is_video:
                    timer.start()
                    metrics['vmaf'] = vmaf_on_tensor(
                        imgs_w[:num_frames], imgs[:num_frames])
                    metrics['vmaf_time'] = timer.end()

                # bdrate
                if bdrate and is_video:
                    r1, vmaf1, r2, vmaf2 = [], [], [], []
                    for crf in [28, 34, 40, 46]:
                        vmaf_score, aux = vmaf_on_tensor(imgs, return_aux=True, crf=crf)
                        r1.append(aux['bps2'])
                        vmaf1.append(vmaf_score)
                        vmaf_score, aux = vmaf_on_tensor(imgs_w, return_aux=True, crf=crf)
                        r2.append(aux['bps2'])
                        vmaf2.append(vmaf_score)
                    metrics['r1'] = '_'.join(str(x) for x in r1)
                    metrics['vmaf1'] = '_'.join(str(x) for x in vmaf1)
                    metrics['r2'] = '_'.join(str(x) for x in r2)
                    metrics['vmaf2'] = '_'.join(str(x) for x in vmaf2)
                    metrics['bd_rate'] = bd_rate(r1, vmaf1, r2, vmaf2)

            # save images and videos
            if it < save_first:
                base_name = os.path.join(output_dir, f'{it:03}_val')
                ori_path = base_name + '_0_ori.png'
                wm_path = base_name + '_1_wm.png'
                diff_path = base_name + '_2_diff.png'
                save_image(imgs[:8], ori_path, nrow=8)
                save_image(imgs_w[:8], wm_path, nrow=8)
                save_image(create_diff_img(imgs[:8], imgs_w[:8]), diff_path, nrow=8)
                if is_video:
                    fps = 24 // 1
                    ori_path = ori_path.replace(".png", ".mp4")
                    wm_path = wm_path.replace(".png", ".mp4")
                    diff_path = diff_path.replace(".png", ".mp4")
                    # timer.start()
                    save_vid(imgs, ori_path, fps)
                    save_vid(imgs_w, wm_path, fps)
                    save_vid(create_diff_img(imgs, imgs_w), diff_path, fps)
                    # metrics['save_vid_time'] = timer.end()

            # extract watermark and evaluate robustness
            if detection or decoding:
                # masks, for now, are all ones
                masks = torch.ones_like(imgs[:, :1])  # b 1 h w
                imgs_masked = imgs_w * masks + imgs * (1 - masks)

                # extraction for different augmentations
                for validation_aug, strengths in validation_augs:

                    for strength in strengths:
                        imgs_aug, masks_aug = validation_aug(
                            imgs_masked, masks, strength)
                        selected_aug = str(validation_aug) + f"_{strength}"
                        selected_aug = selected_aug.replace(", ", "_")

                        # extract watermark
                        timer.start()
                        if is_video:
                            preds = model.extract_message(imgs_aug, video_aggregation, interpolation)  # 1 k
                            preds = torch.cat([torch.ones(preds.size(0), 1).to(preds.device), preds], dim=1)  # 1 1+k
                            outputs = {"preds": preds}
                            msgs = msgs[:1]  # 1 k
                        else:    
                            outputs = model.detect(imgs_aug, is_video=False)  # 1 k
                        timer.step()
                        preds = outputs["preds"]
                        mask_preds = preds[:, 0:1]  # b 1 ...
                        bit_preds = preds[:, 1:]  # b k ...

                        aug_log_stats = {}
                        if decoding:
                            aug_log_stats[f'bit_acc'] = bit_accuracy(
                                bit_preds, msgs, masks_aug).nanmean().item()
                            aug_log_stats[f'pvalue'] = pvalue(
                                bit_preds, msgs, masks_aug).nanmean().item()
                            aug_log_stats[f'log_pvalue'] = -np.log10(
                                aug_log_stats[f'pvalue']) if aug_log_stats[f'pvalue'] > 0 else -100
                            aug_log_stats[f'capacity'] = capacity(
                                bit_preds, msgs, masks_aug).nanmean().item()

                        if detection:
                            iou0 = iou(mask_preds, masks, label=0).mean().item()
                            iou1 = iou(mask_preds, masks, label=1).mean().item()
                            aug_log_stats.update({
                                f'acc': accuracy(mask_preds, masks).mean().item(),
                                f'miou': (iou0 + iou1) / 2,
                            })

                        current_key = f"{selected_aug}"
                        aug_log_stats = {f"{k}_{current_key}": v for k,
                                        v in aug_log_stats.items()}
                        metrics.update(aug_log_stats)
                metrics['extract_time'] = timer.avg_step()
            all_metrics.append(metrics)

            # save metrics
            if it == 0:
                f.write(','.join(metrics.keys()) + '\n')
            f.write(','.join(map(str, metrics.values())) + '\n')
            f.flush()
    return all_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation')

    group = parser.add_argument_group('Dataset')
    group.add_argument("--dataset", type=str, help="Name of the dataset.")
    group.add_argument('--is_video', type=bool_inst, default=False, 
                       help='Whether the data is video')
    group.add_argument('--short_edge_size', type=int, default=-1, 
                       help='Resizes the short edge of the image to this size at loading time, and keep the aspect ratio. If -1, no resizing.')
    group.add_argument('--num_frames', type=int, default=24*3, 
                       help='Number of frames to evaluate for video quality')
    group.add_argument('--num_samples', type=int, default=100, 
                          help='Number of samples to evaluate')
    group.add_argument('--video_aggregation', type=str, default="avg",
                            help='Aggregation method for detection of video frames')

    group = parser.add_argument_group('Model parameters to override. If not provided, the checkpoint values are used.')
    group.add_argument("--attenuation_config", type=str, default="configs/attenuation.yaml",
                        help="Path to the attenuation config file")
    group.add_argument("--attenuation", type=str, default=None,
                        help="Attenuation model to use")
    group.add_argument("--lowres_attenuation", type=bool_inst, default=False,
                        help="Whether to do attenuation at low resolution")
    group.add_argument("--img_size_proc", type=int, default=None, 
                        help="Size of the input images for interpolation in the embedder/extractor models")
    group.add_argument("--scaling_w", default=None,
                        help="Scaling factor for the watermark in the embedder model")
    group.add_argument('--videoseal_chunk_size', type=int, default=32, 
                        help='Number of frames to chunk during forward pass')
    group.add_argument('--videoseal_step_size', type=int, default=4,
                        help='The number of frames to propagate the watermark to')
    group.add_argument('--videoseal_mode', type=str, default='repeat', 
                        help='The inference mode for videos')

    group = parser.add_argument_group('Experiment')
    group.add_argument("--output_dir", type=str, default="outputs/", help="Output directory for logs and images (Default: /output)")
    group.add_argument('--save_first', type=int, default=-1, help='Number of images/videos to save')
    group.add_argument('--only_identity', type=bool_inst, default=False, help='Whether to only evaluate the identity augmentation')
    group.add_argument('--only_combined', type=bool_inst, default=False, help='Whether to only evaluate combined augmentations')
    group.add_argument('--bdrate', type=bool_inst, default=False, help='Whether to compute BD-rate')
    group.add_argument('--decoding', type=bool_inst, default=True, help='Whether to evaluate decoding metrics')
    group.add_argument('--detection', type=bool_inst, default=False, help='Whether to evaluate detection metrics')
    group.add_argument('--skip_image_metrics', type=bool_inst, default=False, help='Whether to skip computing image quality metrics')

    group = parser.add_argument_group('Interpolation')
    group.add_argument('--interpolation_mode', type=str, default='bilinear',
                      choices=['nearest', 'bilinear', 'bicubic', 'area'],
                      help='Interpolation mode for resizing')
    group.add_argument('--interpolation_align_corners', type=bool_inst, default=False,
                      help='Align corners for interpolation')
    group.add_argument('--interpolation_antialias', type=bool_inst, default=True,
                      help='Use antialiasing for interpolation')

    args = parser.parse_args()

    # change some of the params to accept None from json
    if args.scaling_w in ['None', 'none', -1]:
        args.scaling_w = None

    # Setup the model
    model = setup_model_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Override model parameters in args
    model.blender.scaling_w = float(args.scaling_w or model.blender.scaling_w)
    model.chunk_size = args.videoseal_chunk_size or model.chunk_size
    model.step_size = args.videoseal_step_size or model.step_size
    model.video_mode = args.videoseal_mode or model.mode
    model.img_size = args.img_size_proc or model.img_size

    # Setup the device
    avail_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device or avail_device
    model.to(device)

    # Override attenuation build
    if args.attenuation is not None:
        # should be on CPU to operate on high resolution videos
        if args.attenuation.lower().startswith("jnd"):
            attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)
            attenuation = JND(**attenuation_cfg[args.attenuation])
        else:
            attenuation = None
        model.attenuation = attenuation

    # Setup the dataset
    args.simple_video_dataset = True  # use simple video dataset for evaluation to speed up
    dataset = setup_dataset(args)
    dataset = Subset(dataset, range(args.num_samples))

    # evaluate the model, quality and extraction metrics
    os.makedirs(args.output_dir, exist_ok=True)
    interpolation = {
        "mode": args.interpolation_mode,
        "align_corners": args.interpolation_align_corners,
        "antialias": args.interpolation_antialias
    }
    metrics = evaluate(
        model = model,
        dataset = dataset, 
        is_video = args.is_video,
        output_dir = args.output_dir,
        save_first = args.save_first,
        num_frames = args.num_frames,
        video_aggregation = args.video_aggregation,
        only_identity = args.only_identity,
        only_combined = args.only_combined,
        bdrate = args.bdrate,
        decoding = args.decoding,
        detection = args.detection,
        interpolation = interpolation,
        lowres_attenuation = args.lowres_attenuation,
        skip_image_metrics = args.skip_image_metrics,
    )

    # Print mean
    pd.set_option('display.max_rows', None)
    to_remove = ['r1', 'r2', 'vmaf1', 'vmaf2']
    metrics = [{k: v for k, v in metric.items() if k not in to_remove} for metric in metrics]
    print(pd.DataFrame(metrics).mean())


if __name__ == '__main__':
    main()