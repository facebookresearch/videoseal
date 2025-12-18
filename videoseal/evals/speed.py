# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test the speed of different methods on CPU or CUDA.

Example usage:
python -m videoseal.evals.speed \
    --checkpoint videoseal_1.0 \
        baselines/hidden \
        baselines/mbrs \
        baselines/cin \
        baselines/wam \
        baselines/trustmark \
    --dataset sa-v --is_video true --num_samples 10 \
    --device cpu \
    --videoseal_chunk_size 128 --num_frames 240 --lowres_attenuation true
"""
    
import argparse
import os
import time
from typing import List, Dict, Any

import pandas as pd
import tqdm
import torch
from torch.utils.data import Dataset, Subset

from ..models import Videoseal
from ..utils import Timer, bool_inst
from ..utils.cfg import setup_dataset, setup_model_from_checkpoint

class SpeedTester:
    def __init__(self, device='cuda'):
        self.device = device
        self.timer = Timer()
        
    @torch.no_grad()
    def test_speed(
        self,
        model: Videoseal,
        dataset: Dataset, 
        is_video: bool,
        num_frames: int = 24*3,
        video_aggregation: str = "avg",
        lowres_attenuation: bool = False,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
        num_runs: int = 3,
        warmup_runs: int = 1,
    ) -> Dict[str, Any]:
        """
        Test the speed of embedding and extraction for a model on a given dataset.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            is_video: Whether the data is video
            num_frames: Number of frames to process
            video_aggregation: Aggregation method for detection of video frames
            lowres_attenuation: Whether to do attenuation at low resolution
            interpolation: Interpolation parameters
            num_runs: Number of runs to average timing over
            warmup_runs: Number of warmup runs before timing
            
        Returns:
            Dictionary with timing metrics
        """
        metrics = {
            'device': self.device,
            'model_name': model.name if hasattr(model, 'name') else 'unknown',
            'checkpoint': model.checkpoint_path if hasattr(model, 'checkpoint_path') else 'unknown',
            'is_video': is_video,
            'embedding_time': [],
            'extraction_time': [],
            'image_shape': [],
        }

        for it, batch_items in enumerate(tqdm.tqdm(dataset)):
            if batch_items is None:
                continue
            
            imgs, masks = batch_items[0], batch_items[1]
            if not is_video:
                imgs = imgs.unsqueeze(0)  # c h w -> 1 c h w
                masks = masks.unsqueeze(0) if isinstance(masks, torch.Tensor) else masks
            
            # Record shape information
            metrics['image_shape'].append(f"{imgs.shape[-4]}x{imgs.shape[-2]}x{imgs.shape[-1]}")
            
            # Cut frames to the specified number
            imgs = imgs[:num_frames]
            masks = masks[:num_frames] if masks is not None else None
            
            # Move to device
            # imgs = imgs.to(self.device)
            # if masks is not None:
            #     masks = masks.to(self.device)
            
            # Warmup runs
            for _ in range(warmup_runs):
                outputs = model.embed(imgs, is_video=is_video, interpolation=interpolation, lowres_attenuation=lowres_attenuation)
                if is_video:
                    _ = model.extract_message(outputs["imgs_w"], video_aggregation, interpolation)
                else:
                    _ = model.detect(outputs["imgs_w"], is_video=False)
            
            # Timed runs - embedding
            embed_times = []
            for _ in range(num_runs):
                torch.cuda.synchronize() if self.device == 'cuda' else None
                start_time = time.time()
                outputs = model.embed(imgs, is_video=is_video, interpolation=interpolation, lowres_attenuation=lowres_attenuation)
                torch.cuda.synchronize() if self.device == 'cuda' else None
                embed_times.append(time.time() - start_time)
            
            imgs_w = outputs["imgs_w"]
            metrics['embedding_time'].append(sum(embed_times) / num_runs)
            
            # Timed runs - extraction
            extract_times = []
            for _ in range(num_runs):
                torch.cuda.synchronize() if self.device == 'cuda' else None
                start_time = time.time()
                
                if is_video:
                    _ = model.extract_message(imgs_w, video_aggregation, interpolation)
                else:
                    _ = model.detect(imgs_w, is_video=False)
                
                torch.cuda.synchronize() if self.device == 'cuda' else None
                extract_times.append(time.time() - start_time)
            
            metrics['extraction_time'].append(sum(extract_times) / num_runs)
        
        # Compute average metrics
        metrics['avg_embedding_time'] = sum(metrics['embedding_time']) / len(metrics['embedding_time'])
        metrics['avg_extraction_time'] = sum(metrics['extraction_time']) / len(metrics['extraction_time'])
        
        # For video data, compute per-frame metrics
        if is_video:
            avg_frames = sum([int(shape.split('x')[0]) for shape in metrics['image_shape']]) / len(metrics['image_shape'])
            metrics['avg_embedding_ms_per_frame'] = (metrics['avg_embedding_time'] / avg_frames) * 1000
            metrics['avg_extraction_ms_per_frame'] = (metrics['avg_extraction_time'] / avg_frames) * 1000
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Test speed of embedding and extraction')
    parser.add_argument('--checkpoint', type=str, nargs='+', required=True, help='Path(s) to the model checkpoint(s)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs to average timing over')
    parser.add_argument('--warmup_runs', type=int, default=1, help='Number of warmup runs before timing')

    group = parser.add_argument_group('Dataset')
    group.add_argument("--dataset", type=str, help="Name of the dataset.")
    group.add_argument('--is_video', type=bool_inst, default=False, 
                       help='Whether the data is video')
    group.add_argument('--short_edge_size', type=int, default=-1, 
                       help='Resizes the short edge of the image to this size at loading time, and keep the aspect ratio. If -1, no resizing.')
    group.add_argument('--num_frames', type=int, default=24*3, 
                       help='Number of frames to evaluate for video quality')
    group.add_argument('--num_samples', type=int, default=5, 
                          help='Number of samples to evaluate')
    group.add_argument('--video_aggregation', type=str, default="avg",
                            help='Aggregation method for detection of video frames')

    group = parser.add_argument_group('Model parameters to override. If not provided, the checkpoint values are used.')
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

    group = parser.add_argument_group('Interpolation')
    group.add_argument('--interpolation_mode', type=str, default='bilinear',
                      choices=['nearest', 'bilinear', 'bicubic', 'area'],
                      help='Interpolation mode for resizing')
    group.add_argument('--interpolation_align_corners', type=bool_inst, default=False,
                      help='Align corners for interpolation')
    group.add_argument('--interpolation_antialias', type=bool_inst, default=True,
                      help='Use antialiasing for interpolation')

    group = parser.add_argument_group('Output')
    group.add_argument("--output_dir", type=str, default="output/speed", help="Output directory for logs")
    group.add_argument("--output_file", type=str, default="speed_results.csv", help="Output file name for results")

    args = parser.parse_args()

    # change some of the params to accept None from json
    if args.scaling_w in ['None', 'none', -1]:
        args.scaling_w = None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize dataset 
    args.simple_video_dataset = True  # use simple video dataset for evaluation to speed up
    dataset = setup_dataset(args)
    dataset = Subset(dataset, range(args.num_samples))

    # Setup interpolation
    interpolation = {
        "mode": args.interpolation_mode,
        "align_corners": args.interpolation_align_corners,
        "antialias": args.interpolation_antialias
    }

    # Setup the speed tester
    tester = SpeedTester(device=args.device)
    
    # Test speed for each checkpoint
    all_results = []
    
    for checkpoint_path in args.checkpoint:
        print(f"\nTesting checkpoint: {checkpoint_path}")
        
        # Setup the model
        model = setup_model_from_checkpoint(checkpoint_path)
        model.eval()
        model.compile()
        
        # Override model parameters in args
        if hasattr(model, 'blender') and hasattr(model.blender, 'scaling_w'):
            model.blender.scaling_w = float(args.scaling_w or model.blender.scaling_w)
        if hasattr(model, 'chunk_size'):
            model.chunk_size = args.videoseal_chunk_size or model.chunk_size
        if hasattr(model, 'step_size'):
            model.step_size = args.videoseal_step_size or model.step_size
        if hasattr(model, 'video_mode'):
            model.video_mode = args.videoseal_mode or model.mode
        if hasattr(model, 'img_size'):
            model.img_size = args.img_size_proc or model.img_size
        
        # Record checkpoint path
        model.checkpoint_path = checkpoint_path
        model.name = os.path.basename(os.path.dirname(checkpoint_path)) if '/' in checkpoint_path else 'unknown'
        
        # Move model to device
        model.to(args.device)
        
        # Run speed tests
        results = tester.test_speed(
            model=model,
            dataset=dataset,
            is_video=args.is_video,
            num_frames=args.num_frames,
            video_aggregation=args.video_aggregation,
            lowres_attenuation=args.lowres_attenuation,
            interpolation=interpolation,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
        )
        
        all_results.append(results)
    
    # Create DataFrame from results
    df = pd.DataFrame([{
        'model_name': r['model_name'],
        'checkpoint': r['checkpoint'],
        'device': r['device'],
        'is_video': r['is_video'],
        'avg_embedding_time': r['avg_embedding_time'],
        'avg_extraction_time': r['avg_extraction_time'],
    } for r in all_results])
    
    # Add per-frame metrics for video data
    if args.is_video:
        df['Embed ms/frame'] = [r['avg_embedding_ms_per_frame'] for r in all_results]
        df['Extract ms/frame'] = [r['avg_extraction_ms_per_frame'] for r in all_results]
        
    # Save results
    output_path = os.path.join(args.output_dir, args.output_file)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("\nSpeed Test Results:")
    print(df)
    
    # Print average metrics
    print("\nAverage Metrics:")
    if args.is_video:
        print(f"Average Embed Time per frame: {df['Embed ms/frame'].mean():.2f} ms/frame")
        print(f"Average Extract Time per frame: {df['Extract ms/frame'].mean():.2f} ms/frame")
    else:
        print(f"Average Embed Time: {df['Avg Embed Time (s)'].mean():.4f} seconds")
        print(f"Average Extract Time: {df['Avg Extract Time (s)'].mean():.4f} seconds")
    
    
if __name__ == '__main__':
    main()
