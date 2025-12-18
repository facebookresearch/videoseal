# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate a model with different step sizes for video watermark propagation.

python -m videoseal.evals.step_size_eval \
    --checkpoint videoseal_1.0  --lowres_attenuation true \
    --dataset sa-v --is_video true --num_samples 10 --save_first -1 --skip_image_metrics true  --only_combined true \
    --step_sizes 1 2 4 8 16 \
    --output_dir output/step_size_eval_combined
"""

import argparse
import os
import json

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
from .full import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model with different step sizes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--step_sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16], 
                        help='Step sizes to evaluate (default: [1, 2, 4, 8, 16])')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation')

    group = parser.add_argument_group('Dataset')
    group.add_argument("--dataset", type=str, help="Name of the dataset.")
    group.add_argument('--is_video', type=bool_inst, default=True, 
                       help='Whether the data is video (default: True)')
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
    group.add_argument('--videoseal_mode', type=str, default='repeat', 
                        help='The inference mode for videos')

    group = parser.add_argument_group('Experiment')
    group.add_argument("--output_dir", type=str, default="output/step_size_eval", 
                       help="Output directory for logs and images")
    group.add_argument('--save_first', type=int, default=-1, help='Number of images/videos to save')
    parser.add_argument('--only_identity', type=bool_inst, default=False, help='Whether to only evaluate the identity augmentation')
    parser.add_argument('--only_combined', type=bool_inst, default=False, help='Whether to only evaluate combined augmentations')
    parser.add_argument('--bdrate', type=bool_inst, default=False, help='Whether to compute BD-rate')
    parser.add_argument('--decoding', type=bool_inst, default=True, help='Whether to evaluate decoding metrics')
    parser.add_argument('--detection', type=bool_inst, default=False, help='Whether to evaluate detection metrics')
    parser.add_argument('--skip_image_metrics', type=bool_inst, default=False, help='Whether to skip computing image quality metrics')

    group = parser.add_argument_group('Interpolation')
    group.add_argument('--interpolation_mode', type=str, default='bilinear',
                      choices=['nearest', 'bilinear', 'bicubic', 'area'],
                      help='Interpolation mode for resizing')
    group.add_argument('--interpolation_align_corners', type=bool_inst, default=False,
                      help='Align corners for interpolation')
    group.add_argument('--interpolation_antialias', type=bool_inst, default=True,
                      help='Use antialiasing for interpolation')

    args = parser.parse_args()

    # Ensure we're evaluating videos
    if not args.is_video:
        print("Warning: step size evaluation is designed for videos. Setting is_video=True.")
        args.is_video = True

    # change some of the params to accept None from json
    if args.scaling_w in ['None', 'none', -1]:
        args.scaling_w = None

    # Setup the model
    model = setup_model_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Override model parameters in args
    model.blender.scaling_w = float(args.scaling_w or model.blender.scaling_w)
    model.chunk_size = args.videoseal_chunk_size or model.chunk_size
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

    # Create interpolation dict
    interpolation = {
        "mode": args.interpolation_mode,
        "align_corners": args.interpolation_align_corners,
        "antialias": args.interpolation_antialias
    }
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration for reference
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        config = vars(args).copy()
        # Convert non-serializable types to strings
        for k, v in config.items():
            if not isinstance(v, (int, float, str, bool, list, dict, type(None))):
                config[k] = str(v)
        json.dump(config, f, indent=2)
    
    # Run evaluation for each step size
    all_metrics = []
    
    for step_size in args.step_sizes:
        print(f"\n\n===== Evaluating with step_size={step_size} =====")
        
        # Create subdirectory for this step size
        step_output_dir = os.path.join(args.output_dir, f"step_{step_size}")
        os.makedirs(step_output_dir, exist_ok=True)
        
        # Set step size for the model
        model.step_size = step_size
        
        # Run evaluation
        metrics = evaluate(
            model=model,
            dataset=dataset, 
            is_video=args.is_video,
            output_dir=step_output_dir,
            save_first=args.save_first,
            num_frames=args.num_frames,
            video_aggregation=args.video_aggregation,
            only_identity=args.only_identity,
            only_combined=args.only_combined,
            bdrate=args.bdrate,
            decoding=args.decoding,
            detection=args.detection,
            interpolation=interpolation,
            lowres_attenuation=args.lowres_attenuation,
            skip_image_metrics=args.skip_image_metrics,
        )
        
        # Add step size to metrics
        for metric in metrics:
            metric['step_size'] = step_size
        
        all_metrics.extend(metrics)
    
    # Save combined metrics
    combined_df = pd.DataFrame(all_metrics)
    combined_df.to_csv(os.path.join(args.output_dir, 'combined_metrics.csv'), index=False)
    
    # Generate summary by step size
    summary_df = combined_df.groupby('step_size').mean()
    summary_df.to_csv(os.path.join(args.output_dir, 'summary_by_step_size.csv'))
    
    # Print summary
    print("\n===== Summary by Step Size =====")
    print(summary_df)
    
    # Plot key metrics vs step size if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        key_metrics = ['psnr', 'ssim', 'lpips']
        if 'vmaf' in summary_df.columns:
            key_metrics.append('vmaf')
        
        bit_acc_cols = [col for col in summary_df.columns if 'bit_acc' in col]
        key_metrics.extend(bit_acc_cols[:1])  # Add first bit accuracy metric
        
        plt.figure(figsize=(12, 8))
        for metric in key_metrics:
            if metric in summary_df.columns:
                plt.plot(summary_df.index, summary_df[metric], 'o-', label=metric)
        
        plt.xlabel('Step Size')
        plt.ylabel('Metric Value')
        plt.title('Metrics vs Step Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, 'metrics_vs_step_size.png'))
        print(f"Saved plot to {os.path.join(args.output_dir, 'metrics_vs_step_size.png')}")
    except ImportError:
        print("matplotlib not available, skipping plot generation")


if __name__ == '__main__':
    main()
