"""
python -m videoseal.evals.full \
    --checkpoint /private/home/hadyelsahar/work/code/videoseal/2024_logs/1013-hybrid-large-sweep-allaugs/_lambda_d=0.5_lambda_i=0.5_optimizer=AdamW,lr=5e-5_prop_img_vid=0.9_videowam_step_size=4_video_start=500_embedder_model=unet_small2/checkpoint.pth \
    --dataset coco --is_video false \

    --dataset sa-v --is_video true \

    
    /private/home/hadyelsahar/work/code/videoseal/2024_logs/1013-hybrid-large-sweep-allaugs/_lambda_d=0.5_lambda_i=0.5_optimizer=AdamW,lr=5e-5_prop_img_vid=0.9_videowam_step_size=4_video_start=500_embedder_model=vae_small_bw/checkpoint.pth
    /private/home/hadyelsahar/work/code/videoseal/2024_logs/1013-hybrid-large-sweep-allaugs/_lambda_d=0.5_lambda_i=0.5_optimizer=AdamW,lr=5e-5_prop_img_vid=0.9_videowam_step_size=4_video_start=500_embedder_model=unet_small2/checkpoint.pth
"""


import json
import omegaconf
import argparse
import os
import time

import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image
import torchvision.transforms as transforms

from .metrics import vmaf_on_tensor, bit_accuracy, iou, accuracy
from ..data.datasets import ImageFolder, VideoDataset, CocoImageIDWrapper
from ..models import VideoWam, build_embedder, build_extractor
from ..augmentation import get_validation_augs
from ..augmentation.augmenter import get_dummy_augmenter
from ..evals.metrics import psnr, ssim
from ..utils import Timer
from ..utils.data import parse_dataset_params, Modalities
from ..utils.image import create_diff_img
from ..utils.display import save_vid

import videoseal.utils as utils


def setup_model_from_checkpoint(ckpt_path):
    """
    # Example usage
    ckpt_path = '/checkpoint/pfz/2024_logs/0911_vseal_pw/extractor_model=sam_tiny/checkpoint.pth'
    exp_dir = '/checkpoint/pfz/2024_logs/0911_vseal_pw'
    exp_name = '_extractor_model=sam_tiny'

    wam = load_model_from_checkpoint(exp_dir, exp_name)
    """
    exp_dir, exp_name = os.path.dirname(ckpt_path).rsplit('/', 1)
    logfile_path = os.path.join(exp_dir, 'logs', exp_name + '.stdout')

    # Load parameters from log file
    with open(logfile_path, 'r') as file:
        for line in file:
            if '__log__:' in line:
                params = json.loads(line.split('__log__:')[1].strip())
                break

    # Create an argparse Namespace object from the parameters
    args = argparse.Namespace(**params)
    
    # Load configurations
    for path in [args.embedder_config, args.extractor_config, args.augmentation_config]:
        path = os.path.join(exp_dir, "code", path)
    # embedder
    embedder_cfg = omegaconf.OmegaConf.load(args.embedder_config)
    args.embedder_model = args.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[args.embedder_model]
    # extractor
    extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)
    args.extractor_model = args.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[args.extractor_model]
    # augmenter
    augmenter_cfg = omegaconf.OmegaConf.load(args.augmentation_config)
    
    # Build models
    embedder = build_embedder(args.embedder_model, embedder_params, args.nbits)
    extractor = build_extractor(extractor_cfg.model, extractor_params, args.img_size_extractor, args.nbits)
    augmenter = get_dummy_augmenter()  # does nothing
    
    # Build the complete model
    wam = VideoWam(embedder, extractor, augmenter, 
                scaling_w=args.scaling_w, scaling_i=args.scaling_i, 
                img_size=args.img_size,
                chunk_size=args.videowam_chunk_size,
                step_size=args.videowam_step_size
            )
    
    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        wam.load_state_dict(checkpoint['model'])
        print("Model loaded successfully from", ckpt_path)
    else:
        msg = f"Checkpoint path does not exist:{ckpt_path}"
        raise FileNotFoundError(msg)
    
    return wam


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
            mode = 'val'
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
    decoding: bool = True,
    detection: bool = False,
):
    """
    QQs: eval only quality? eval only bit accuracy?
        augs?
        one by one or batched?
    Args:
        wam (VideoWam): The model to evaluate
        dataset (Dataset): The dataset to evaluate on
        is_video (bool): Whether the data is video
        output_dir (str): Directory to save the output images
        num_frames (int): Number of frames to evaluate for video quality
    """
    all_metrics = []
    validation_augs = get_validation_augs(is_video)
    timer = Timer()

    for it, batch_items in enumerate(dataset):
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

        # save images and videos
        if it < save_first:
            base_name = os.path.join(output_dir, f'val')
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

        # masks, for now, are all ones
        masks = torch.ones_like(imgs[:, :1])  # b 1 h w
        imgs_masked = imgs_w * masks + imgs * (1 - masks)

        # extraction for different augmentations
        for transform, strengths in validation_augs:
            # Create an instance of the transformation
            transform_instance = transform()

            for strength in strengths:
                imgs_aug, masks_aug = transform_instance(
                    imgs_masked, masks, strength)
                selected_aug = str(transform.__name__).lower()
                selected_aug += f"_{strength}"

                # extract watermark
                timer.start()
                outputs = wam.detect(imgs_aug, is_video=is_video)
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

        print(metrics)
        all_metrics.append(metrics)

    return all_metrics



def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation')

    group = parser.add_argument_group('Dataset')
    group.add_argument("--dataset", type=str, help="Name of the dataset.")
    group.add_argument('--is_video', type=utils.bool_inst, default=False, help='Whether the data is video')
    group.add_argument('--short_edge_size', type=int, default=-1, help='Short edge size for resizing, -1 for no resizing')
    group.add_argument('--videowam_chunk_size', type=int, default=64, help='Number of frames to chunk during forward pass')
    group.add_argument('--num_frames', type=int, default=24*3, help='Number of frames to evaluate for video quality')

    group = parser.add_argument_group('Experiment')
    group.add_argument("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    group.add_argument('--save_first', type=int, default=-1, help='Number of images/videos to save')

    args = parser.parse_args()

    # Setup the model
    model = setup_model_from_checkpoint(args.checkpoint)
    model.eval()

    # Setup the device
    avail_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device or avail_device
    model.to(device)
    model.chunk_size = args.videowam_chunk_size

    # Setup the dataset    
    dataset = setup_dataset(args)

    # Evaluate the model
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = evaluate(
        wam = model, 
        dataset = dataset, 
        is_video = args.is_video, 
        output_dir = args.output_dir,
        save_first = args.save_first,
    )

if __name__ == '__main__':
    main()