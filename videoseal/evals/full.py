"""
python -m videoseal.evals.full \
    --checkpoint /checkpoint/pfz/2024_logs/1011_vseal_video_yesno/_lambda_d=0.0_video_start=50/checkpoint.pth \
    --data_dir /datasets01/COCO/060817/val2014/

    --data_dir /large_experiments/meres/sa-v/sav_val_videos/

    /private/home/hadyelsahar/work/code/videoseal/2024_logs/1013-hybrid-large-sweep-allaugs/_lambda_d=0.5_lambda_i=0.5_optimizer=AdamW,lr=5e-5_prop_img_vid=0.9_videowam_step_size=4_video_start=500_embedder_model=vae_small_bw/checkpoint.pth
"""


import json
import omegaconf
import argparse
import os

import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image


from .metrics import vmaf_on_tensor
from ..data.loader import get_dataloader_segmentation, get_video_dataloader
from ..data.datasets import VideoDataset
from ..models import VideoWam, build_embedder, build_extractor
from ..augmentation.augmenter import get_dummy_augmenter
from ..evals.metrics import psnr, ssim

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
    # print(args)
    
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
                   scaling_w=args.scaling_w, scaling_i=args.scaling_i)
    
    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        wam.load_state_dict(checkpoint['model'])
        print("Model loaded successfully from", ckpt_path)
        # print(line)
    else:
        msg = f"Checkpoint path does not exist:{ckpt_path}"
        raise FileNotFoundError(msg)
    
    return wam


def setup_dataset(args):
    if args.is_video:
        dataset = VideoDataset(
            folder_paths = [args.data_dir],
            transform = None,
            frames_per_clip = args.frames_per_clip,
            frame_step = args.frame_step,
            num_clips = args.num_clips,
            output_resolution = args.short_edge_size,
            num_workers = 0,
        )
    else:
        dataset = CocoImageIDWrapper(
            root=data_dir, 
            annFile=ann_file, 
            transform=transform, 
            mask_transform=mask_transform,
            random_nb_object=random_nb_object, 
            multi_w=multi_w, 
            max_nb_masks=max_nb_masks
        )
    return dataset


@torch.no_grad()
def evaluate(
    model: VideoWam,
    dataset: Dataset, 
    output_dir: str,
):
    """
    
    eval only quality? eval only bit accuracy?
    augs?
    """
    metrics = []

    for it, batch_items in enumerate(dataset):

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

            # forward embedder
            embed_time = time.time()
            outputs = wam.embed(imgs, is_video=is_video)
            embed_time = (time.time() - embed_time) / imgs.shape[0]
            msgs = outputs["msgs"]  # b k
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

                if epoch_modality == Modalities.VIDEO:
                    fps = 24 // 1
                    ori_path = ori_path.replace(".png", ".mp4")
                    wm_path = wm_path.replace(".png", ".mp4")
                    diff_path = diff_path.replace(".png", ".mp4")
                    save_vid(imgs, ori_path, fps)
                    save_vid(imgs_w, wm_path, fps)
                    save_vid(imgs - imgs_w, diff_path, fps)
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

                for transform, strengths in validation_augs:
                    # Create an instance of the transformation
                    transform_instance = transform()

                    for strength in strengths:
                        do_resize = False  # hardcode for now, might need to change
                        if not do_resize:
                            imgs_aug, masks_aug = transform_instance(
                                imgs_masked, masks, strength)
                        else:
                            # h, w = imgs_w.shape[-2:]
                            h, w = params.img_size_extractor, params.img_size_extractor
                            imgs_aug, masks_aug = transform_instance(
                                imgs_masked, masks, strength)
                            if imgs_aug.shape[-2:] != (h, w):
                                imgs_aug = nn.functional.interpolate(imgs_aug, size=(h, w),
                                                                    mode='bilinear', align_corners=False, antialias=True)
                                masks_aug = nn.functional.interpolate(masks_aug, size=(h, w),
                                                                    mode='bilinear', align_corners=False, antialias=True)
                        selected_aug = str(transform.__name__).lower()
                        selected_aug += f"_{strength}"

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
                            iou0 = iou(mask_preds, masks, label=0).mean().item()
                            iou1 = iou(mask_preds, masks, label=1).mean().item()
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

    return metrics



def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation')
    
    group = parser.add_argument_group('Dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--short_edge_size', type=int, default=-1, help='Short edge size for resizing, -1 for no resizing')
    parser.add_argument('--is_video', type=utils.bool_inst, default=False, help='Whether the data is video')
    group.add_argument('--frames_per_clip', default=32, type=int, help='Number of frames per clip for video datasets')
    group.add_argument('--frame_step', default=1, type=int, help='Step between frames for video datasets')
    group.add_argument('--num_clips', default=2, type=int, help='Number of clips per video for video datasets')

    args = parser.parse_args()

    # Setup the model
    model = setup_model_from_checkpoint(args.checkpoint)
    model.eval()

    # Setup the device
    avail_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device or avail_device
    model.to(device)

    # Setup the dataset    
    dataset = setup_dataset(args)

    eval(model, dataset, device, args.output_dir)

if __name__ == '__main__':
    main()