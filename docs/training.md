# Training Models

This guide explains how to train Video Seal models from scratch, including data preparation, image pre-training, and video fine-tuning.

## Data preparation

You only need a folder of images to start training. Create a simple YAML configuration file in `configs/datasets/` to point to your image/video directory.

Example dataset config:
```yaml
# configs/datasets/myimages.yaml
train_dir: /path/to/images/train/
val_dir: /path/to/images/val/
train_annotation_file: null
val_annotation_file: null
```

The image data loader supports both simple image folders and COCO-format annotations (optional).


## Training commands

### Image pre-training

To train an image watermarking model (128 bits) from scratch:

```bash
OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 train.py --local_rank 0 \
    --video_dataset none --image_dataset myimages --workers 8 \
    --extractor_model convnext_tiny --embedder_model unet_small2_yuv_quant --hidden_size_multiplier 1 --nbits 128 \
    --scaling_w_schedule Cosine,scaling_min=0.2,start_epoch=200,epochs=200 --scaling_w 1.0 --scaling_i 1.0 --attenuation jnd_1_1 \
    --epochs 601 --iter_per_epoch 1000 --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=601,warmup_lr_init=1e-8,warmup_t=20 --optimizer AdamW,lr=5e-4 \
    --lambda_dec 1.0 --lambda_d 0.1 --lambda_i 0.1 --perceptual_loss yuv  --num_augs 2 --augmentation_config configs/all_augs_v3.yaml --disc_in_channels 1 --disc_start 50
```

For a 256-bit model, simply change `--nbits 128` to `--nbits 256`.

### Video fine-tuning

After pre-training on images, you can fine-tune on video data:

```bash
OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 train.py --local_rank 0 \
    --video_dataset myvideos --image_dataset none --workers 0 --frames_per_clip 16 \
    --resume_from /path/to/image/checkpoint.pth --resume_optimizer_state True --resume_disc True \
    --videoseal_step_size 4 --lowres_attenuation True --img_size_proc 256 --img_size_val 768 --img_size 768 \
    --extractor_model convnext_tiny --embedder_model unet_small2_yuv_quant --hidden_size_multiplier 1 --nbits 128 \
    --scaling_w_schedule None --scaling_w 0.2 --scaling_i 1.0 --attenuation jnd_1_1 \
    --epochs 601 --iter_per_epoch 100 --scheduler None --optimizer AdamW,lr=1e-5 \
    --lambda_dec 1.0 --lambda_d 0.5 --lambda_i 0.1 --perceptual_loss yuv  --num_augs 2 --augmentation_config configs/all_augs_v3.yaml --disc_in_channels 1 --disc_start 50
```

### Important parameters

- `--nbits`: Number of bits in the watermark (128, 256)
- `--scaling_w`: Watermark strength (higher values = more visible but more robust)


## Pre-trained models

### Full models

**Image models**

The image models are trained with these parameters: https://dl.fbaipublicfiles.com/videoseal/train_img_y.json.
Here are the final weights with discriminator and optimizer state at the end of training, and the saved logs:

| Model | Description | Training Checkpoint | Logs |
|-------|-------------|---------------------|------|
| 128-bit | Image-trained model with 128 bits | [y_128b_img.pth](https://dl.fbaipublicfiles.com/videoseal/y_128b_img_full.pth) | [logs](https://dl.fbaipublicfiles.com/videoseal/log_y_128b_img.txt) |
| 256-bit | Image-trained model with 256 bits | [y_256b_img.pth](https://dl.fbaipublicfiles.com/videoseal/y_256b_img_full.pth) | [logs](https://dl.fbaipublicfiles.com/videoseal/log_y_256b_img.txt) |

Note: Inference-only model files (linked in the main README) are smaller versions of these checkpoints with only the necessary weights for inference.


## Training Tips

1. Make sure that training kicks off (bit accuracy should increase fast). If not, try in this order: to remove perceptual loss (set `--lambda_i 0`), to increase `--scaling_w`, to remove the augmentations.
2. Adjust `--scaling_w` during training with `scaling_w_schedule` for better robustness (start high, then decrease)
