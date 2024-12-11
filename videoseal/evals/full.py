# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the sav_dataset directory of this source tree.

"""
python -m videoseal.evals.full \
    --checkpoint /path/to/checkpoint.pth \
    --dataset coco --is_video false \
    --dataset sa-v --is_video true --num_samples 1 \
"""


import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import omegaconf
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset, Subset
from torchvision.utils import save_image

from videoseal.utils.cfg import (get_validation_augs, setup_dataset,
                                 setup_model_from_checkpoint)

from ..evals.metrics import bd_rate, psnr, ssim
from ..models import Videoseal
from ..modules.jnd import JND
from ..utils import Timer, bool_inst
from ..utils.display import save_vid
from ..utils.image import create_diff_img
from .metrics import accuracy, bit_accuracy, iou, vmaf_on_tensor


@torch.no_grad()
def evaluate(
    videoseal: Videoseal,
    dataset: Dataset, 
    is_video: bool,
    output_dir: str,
    save_first: int = -1,
    num_frames: int = 24*3,
    bdrate: bool = True,
    decoding: bool = True,
    detection: bool = False,
):
    """
    Gives detailed evaluation metrics for a model on a given dataset.
    Args:
        videoseal (Videoseal): The model to evaluate
        dataset (Dataset): The dataset to evaluate on
        is_video (bool): Whether the data is video
        output_dir (str): Directory to save the output images
        num_frames (int): Number of frames to evaluate for video quality and extraction (default: 24*3 i.e. 3seconds)
        decoding (bool): Whether to evaluate decoding metrics (default: True)
        detection (bool): Whether to evaluate detection metrics (default: False)
    """
    all_metrics = []
    validation_augs = get_validation_augs(is_video)
    timer = Timer()

    # save the metrics as csv
    metrics_path = os.path.join(output_dir, "metrics.csv")
    print(f"Saving metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
            
        for it, batch_items in enumerate(tqdm.tqdm(dataset)):
            # initialize metrics
            metrics = {}

            # some data loaders return batch_data, masks, frames_positions as well
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
            outputs = videoseal.embed(imgs, is_video=is_video)
            metrics['embed_time'] = timer.end()
            msgs = outputs["msgs"]  # b k
            imgs_w = outputs["imgs_w"]  # b c h w

            # cut frames
            imgs = imgs[:num_frames]  # f c h w
            msgs = msgs[:num_frames]  # f k
            imgs_w = imgs_w[:num_frames]  # f c h w
            masks = masks[:num_frames]  # f 1 h w

            # compute qualitative metrics
            metrics['psnr'] = psnr(
                imgs_w[:num_frames], 
                imgs[:num_frames]).mean().item()
            metrics['ssim'] = ssim(
                imgs_w[:num_frames], 
                imgs[:num_frames]).mean().item()
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
                    timer.start()
                    save_vid(imgs, ori_path, fps)
                    save_vid(imgs_w, wm_path, fps)
                    save_vid(imgs - imgs_w, diff_path, fps)
                    metrics['save_vid_time'] = timer.end()

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
                        outputs = videoseal.detect(imgs_aug, is_video=is_video)
                        timer.step()
                        preds = outputs["preds"]
                        mask_preds = preds[:, 0:1]  # b 1 ...
                        bit_preds = preds[:, 1:]  # b k ...

                        aug_log_stats = {}
                        if decoding:
                            bit_accuracy_ = bit_accuracy(
                                bit_preds,
                                msgs,
                                masks_aug
                            ).nanmean().item()
                            aug_log_stats[f'bit_acc'] = bit_accuracy_

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
    group.add_argument("--dataset", type=str, 
                       choices=["coco", "coco-stuff-blurred", "sa-v"], help="Name of the dataset.")
    group.add_argument('--is_video', type=bool_inst, default=False, 
                       help='Whether the data is video')
    group.add_argument('--short_edge_size', type=int, default=-1, 
                       help='Resizes the short edge of the image to this size at loading time, and keep the aspect ratio. If -1, no resizing.')
    group.add_argument('--num_frames', type=int, default=24*3, 
                       help='Number of frames to evaluate for video quality')
    group.add_argument('--num_samples', type=int, default=100, 
                          help='Number of samples to evaluate')
    
    group = parser.add_argument_group('Model parameters to override. If not provided, the checkpoint values are used.')
    group.add_argument("--attenuation_config", type=str, default="configs/attenuation.yaml",
       help="Path to the attenuation config file")
    group.add_argument("--attenuation", type=str, default="None",
                        help="Attenuation model to use")
    group.add_argument("--scaling_w", type=float, default=None,
                        help="Scaling factor for the watermark in the embedder model")
    group.add_argument('--videoseal_chunk_size', type=int, default=None, 
                        help='Number of frames to chunk during forward pass')
    group.add_argument('--videoseal_step_size', type=int, default=None,
                        help='The number of frames to propagate the watermark to')

    group = parser.add_argument_group('Experiment')
    group.add_argument("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    group.add_argument('--save_first', type=int, default=-1, help='Number of images/videos to save')
    group.add_argument('--bdrate', type=bool_inst, default=True, help='Whether to compute BD-rate')
    group.add_argument('--decoding', type=bool_inst, default=True, help='Whether to evaluate decoding metrics')
    group.add_argument('--detection', type=bool_inst, default=False, help='Whether to evaluate detection metrics')

    args = parser.parse_args()

    # Setup the model
    model = setup_model_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Override model parameters in args
    model.scaling_w = args.scaling_w or model.scaling_w
    model.chunk_size = args.videoseal_chunk_size or model.chunk_size
    model.step_size = args.videoseal_step_size or model.step_size

    # Setup the device
    avail_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device or avail_device
    model.to(device)

    # Override attenuation build
    # should be on CPU to operate on high resolution videos
    if args.attenuation.lower() != "none":
        attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)
        attenuation = JND(**attenuation_cfg[args.attenuation])
    else:
        attenuation = None
    model.attenuation = attenuation

    # Setup the dataset    
    dataset = setup_dataset(args)
    dataset = Subset(dataset, range(args.num_samples))

    # evaluate the model, quality and extraction metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = evaluate(
        videoseal = model, 
        dataset = dataset, 
        is_video = args.is_video,
        output_dir = args.output_dir,
        save_first = args.save_first,
        num_frames = args.num_frames,
        bdrate = args.bdrate,
        decoding = args.decoding,
        detection = args.detection,
    )

    # Print mean
    pd.set_option('display.max_rows', None)
    to_remove = ['r1', 'r2', 'vmaf1', 'vmaf2']
    metrics = [{k: v for k, v in metric.items() if k not in to_remove} for metric in metrics]
    print(pd.DataFrame(metrics).mean())


if __name__ == '__main__':
    main()