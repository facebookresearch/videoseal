import torch

import videoseal.utils.optim as uoptim
from videoseal.augmentation.augmenter import Augmenter
from videoseal.evals.metrics import psnr
from videoseal.utils.data import Modalities, parse_dataset_params
from videoseal.data.transforms import get_resize_transform
from videoseal.models import VideoWam, build_embedder, build_extractor
from videoseal.data.loader import get_dataloader_segmentation
import omegaconf


def run_one_epoch(params):
    parse_dataset_params(params)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Convert params to OmegaConf object
    params = omegaconf.OmegaConf.create(vars(params))
    train_transform, train_mask_transform = get_resize_transform(params.img_size, resize_only=params.resize_only)
    image_train_loader = get_dataloader_segmentation(params.image_dataset_config.train_dir,
                                                    params.image_dataset_config.train_annotation_file,
                                                    transform=train_transform,
                                                    mask_transform=train_mask_transform,
                                                    batch_size=params.batch_size,
                                                    num_workers=params.workers, shuffle=True)

    # build the augmenter
    augmenter_cfg = omegaconf.OmegaConf.load(params.augmentation_config)
    augmenter_cfg.num_augs = params.num_augs
    augmenter = Augmenter(
        **augmenter_cfg,
    ).to(device)

    # build extractor
    extractor_cfg = omegaconf.OmegaConf.load(params.extractor_config)
    params.extractor_model = params.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[params.extractor_model]
    extractor = build_extractor(params.extractor_model, extractor_params, params.img_size_proc, params.nbits)
    
    # build embedder
    embedder_cfg = omegaconf.OmegaConf.load(params.embedder_config)
    params.embedder_model = params.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[params.embedder_model]
    embedder = build_embedder(params.embedder_model, embedder_params, params.nbits, params.hidden_size_multiplier)
    
    # build the complete model
    wam = VideoWam(embedder, extractor, augmenter, None,
                   params.scaling_w, params.scaling_i,
                   img_size=params.img_size_proc,
                   chunk_size=params.videowam_chunk_size,
                   step_size=params.videowam_step_size,
                   blending_method=params.blending_method,
                   lowres_attenuation=params.lowres_attenuation)
    
    uoptim.restart_from_checkpoint(
        "/checkpoint/tuantran/2025_logs/0512-1549_fp16_100/expe/checkpoint079.pth",
        model=wam,
    )

    for it, batch_items in enumerate(image_train_loader):
        batch_imgs, batch_masks = batch_items[0], batch_items[1]
        imgs, masks = batch_imgs[0], batch_masks[0]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = wam(imgs, masks, is_video=False)
            psnr_score = psnr(outputs["imgs_w"], imgs).mean().item()
            print("psnr score: {psnr_score}")