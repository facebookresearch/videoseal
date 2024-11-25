"""
python -m videoseal.evals.full \
    --checkpoint /private/home/hadyelsahar/work/code/videoseal/2024_logs/1016-hybrid-vs-ours/_lambda_d=0.5_lambda_i=0.5_optimizer=AdamW,lr=1e-4_videowam_step_size=4_video_start=500_embedder_model=unet_small2/checkpoint.pth \
    --dataset coco --is_video false \
    --dataset sa-v --is_video true --num_samples 1 \


    /private/home/hadyelsahar/work/code/videoseal/2024_logs_large-exp/1109-videoseal0.2-discloss-fix-hing-sleepwake-4nodes/_scaling_w=0.5_lambda_i=0.5_disc_hinge_on_logits_fake=True_sleepwake=False_video_start=500/checkpoint.pth
    /private/home/hadyelsahar/work/code/videoseal/2024_logs/1016-hybrid-vs-ours/_lambda_d=0.5_lambda_i=0.5_optimizer=AdamW,lr=1e-4_videowam_step_size=4_video_start=500_embedder_model=unet_small2/checkpoint.pth
"""


from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict
import omegaconf
import argparse
import os
import numpy as np
import pandas as pd
from lpips import LPIPS

import torch
from torch.utils.data import Dataset, Subset
from torchvision.utils import save_image
import torchvision.transforms as transforms
import tqdm


from .metrics import vmaf_on_tensor, bit_accuracy, iou, accuracy, pvalue, capacity
from ..data.datasets import ImageFolder, VideoDataset, CocoImageIDWrapper
from ..modules.jnd import JND
from ..models import VideoWam, build_embedder, build_extractor, build_baseline
from ..augmentation import get_validation_augs
from ..augmentation.augmenter import get_dummy_augmenter
from ..evals.metrics import psnr, ssim, bd_rate
from ..utils import Timer, bool_inst
from ..utils.data import parse_dataset_params, Modalities
from ..utils.image import create_diff_img
from ..utils.display import save_vid


@dataclass
class SubModelConfig:
    model: str
    params: omegaconf.DictConfig

@dataclass
class VideoWamConfig:
    args: omegaconf.DictConfig
    embedder: SubModelConfig
    extractor: SubModelConfig

def get_config_from_checkpoint(ckpt_path: Path) -> VideoWamConfig:
    exp_dir, exp_name = os.path.dirname(ckpt_path).rsplit('/', 1)
    logfile_path = os.path.join(exp_dir, 'logs', exp_name + '.stdout')

    # Load parameters from log file
    with open(logfile_path, 'r') as file:
        for line in file:
            if '__log__:' in line:
                params = json.loads(line.split('__log__:')[1].strip())
                break

    # Create an omegaconf OmegaConf object from the parameters
    args = omegaconf.OmegaConf.create(params)
    if (not isinstance(args, omegaconf.DictConfig)):
        raise Exception("Expected logfile to contain params dictionary.")

    # embedder
    embedder_cfg = omegaconf.OmegaConf.load(args.embedder_config)
    args.embedder_model = args.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[args.embedder_model]
    # extractor
    extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)
    args.extractor_model = args.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[args.extractor_model]

    return VideoWamConfig(
        args=args,
        embedder=SubModelConfig(model=args.embedder_model, params=embedder_params),
        extractor=SubModelConfig(model=args.extractor_model, params=extractor_params),
    )

def setup_model(config: VideoWamConfig, ckpt_path: Path) -> VideoWam:
    args = config.args
    
    # Build models
    embedder = build_embedder(config.embedder.model, config.embedder.params, args.nbits)
    extractor = build_extractor(config.extractor.model, config.extractor.params, args.img_size_extractor, args.nbits)
    augmenter = get_dummy_augmenter()  # does nothing
    
    # build attenuation
    if args.attenuation.lower() != "none":
        attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)
        attenuation = JND(**attenuation_cfg[args.attenuation])
    else:
        attenuation = None

    # Build the complete model
    wam = VideoWam(embedder, extractor, augmenter, 
                attenuation=attenuation,
                scaling_w=args.scaling_w, scaling_i=args.scaling_i, 
                img_size=args.img_size,
                chunk_size=args.videowam_chunk_size,
                step_size=args.videowam_step_size
            )
    
    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        msg = wam.load_state_dict(checkpoint['model'], strict=False)
        print("Model loaded successfully from", ckpt_path, f"with message: {msg}")
    else:
        msg = f"Checkpoint path does not exist:{ckpt_path}"
        raise FileNotFoundError(msg)
    
    # compile
    wam.compile()

    return wam

def setup_model_from_checkpoint(ckpt_path: str) -> VideoWam:
    """
    # Example usage
    ckpt_path = '/checkpoint/pfz/2024_logs/0911_vseal_pw/extractor_model=sam_tiny/checkpoint.pth'
    wam = setup_model_from_checkpoint(ckpt_path)

    or 
    ckpt_path = 'baseline/wam'
    wam = setup_model_from_checkpoint(ckpt_path)
    """
    # load baselines. Should be in the format of "baseline/{method}"
    if "baseline" in ckpt_path:
        method = ckpt_path.split('/')[-1]
        return build_baseline(method)
    # load videoseal checkpoints
    else:
        config = get_config_from_checkpoint(ckpt_path)
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


@torch.no_grad()
def evaluate(
    wam: VideoWam,
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
        wam (VideoWam): The model to evaluate
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

    # create lpips
    lpips_loss = LPIPS(net="alex").eval()

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
            outputs = wam.embed(imgs, is_video=is_video)
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
                imgs[:num_frames],
                is_video).mean().item()
            metrics['ssim'] = ssim(
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
                    save_vid(imgs - imgs_w, diff_path, fps)
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
                            preds = wam.detect_and_aggregate(imgs_aug)  # 1 k     
                            preds = torch.cat([torch.ones(preds.size(0), 1).to(preds.device), preds], dim=1)  # 1 1+k
                            outputs = {"preds": preds}
                            msgs = msgs[:1]  # 1 k
                        else:    
                            outputs = wam.detect(imgs_aug, is_video=False)  # 1 k
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
    group.add_argument("--dataset", type=str, 
                    #    choices=["coco", "coco-stuff-blurred", "sa-v", "sa-v-3s", "sa-1b"], 
                       help="Name of the dataset.")
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
    group.add_argument("--attenuation", type=str, default=None,
                        help="Attenuation model to use")
    group.add_argument("--scaling_w", type=float, default=None,
                        help="Scaling factor for the watermark in the embedder model")
    group.add_argument('--videowam_chunk_size', type=int, default=None, 
                        help='Number of frames to chunk during forward pass')
    group.add_argument('--videowam_step_size', type=int, default=None,
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
    model.blender.scaling_w = args.scaling_w or model.blender.scaling_w
    model.chunk_size = args.videowam_chunk_size or model.chunk_size
    model.step_size = args.videowam_step_size or model.step_size

    # Setup the device
    avail_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device or avail_device
    model.to(device)

    # Override attenuation build
    if args.attenuation is not None:
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
        wam = model, 
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