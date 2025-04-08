import os
import sys
import piq
import glob
import tqdm
import torch
import omegaconf
import torchvision
from PIL import Image, ImageFont, ImageDraw

sys.path.append("../../")
from videoseal.models import build_extractor
from videoseal.quality_metric.eval_metrics import DummyPairedDataset


patchify_image = torchvision.transforms.Compose([
    lambda x: x.convert("RGB"),
    torchvision.transforms.Resize((768, 768)),
    torchvision.transforms.ToTensor(),
    lambda x: x.view(3, 3, 256, 3, 256).permute(3, 1, 0, 2, 4).reshape(9, 3, 256, 256),
])
transform_image = torchvision.transforms.Compose([
    lambda x: x.convert("RGB"),
    torchvision.transforms.Resize((768, 768)),
    torchvision.transforms.ToTensor(),
    lambda x: x.view(1, 3, 768, 768),
])


def get_artifact_discriminator(device="cuda:0", ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = "/checkpoint/soucek/2025_logs/quality_test6_btnll_test_videosealv2_largersize768_artificialfft_waves.gauss.lines/expe/checkpoint.pth"

    state_dict = torch.load(ckpt_path, weights_only=True, map_location="cpu")["model"]
    extractor_params = omegaconf.OmegaConf.load(os.path.join(os.path.dirname(ckpt_path), "configs/extractor.yaml"))["convnext_tiny"]

    model = build_extractor("convnext_tiny", extractor_params, img_size=256, nbits=0)
    model.load_state_dict(state_dict)
    model = model.eval().to(device)

    return model


def get_clip_model(device="cuda:0"):
    return piq.CLIPIQA(data_range=1.).to(device)


def optimize(img: Image, model, device="cuda:0", num_steps=100, lr=0.05):
    img = patchify_image(img).to(device)
    param = torch.nn.Parameter(torch.zeros_like(img)).to(device)

    optim = torch.optim.SGD([param], lr=lr)
    for _ in range(num_steps):
        optim.zero_grad()
        loss = -model((img + param).clip(0, 1)).mean()
        loss.backward()
        optim.step()
    
    return (img + param).clip(0, 1).detach().cpu()


def outputs_to_image(original_img, watermarked_img, optimized_img, watermark_multiplier=10, watermark_method=None, removal_method=None, **kwargs):
    font = ImageFont.truetype("Optimistic.ttf", 15)

    original_img = patchify_image(original_img)
    watermarked_img = patchify_image(watermarked_img)

    watermark = watermarked_img - original_img
    removed_watermark = watermarked_img - optimized_img
    residual = original_img - optimized_img
    
    # B, 3, H, W
    image = Image.fromarray(torch.cat(list(torch.cat([
        original_img,
        watermarked_img,
        optimized_img,
        torch.abs(watermark * watermark_multiplier),
        torch.abs(removed_watermark * watermark_multiplier),
        torch.abs(residual * watermark_multiplier)
    ], 2)), 2).permute(1, 2, 0).clip(0, 1).mul_(255.).to(torch.uint8).numpy())

    if watermark_method is not None:
        watermark_method = f" ({watermark_method})"
    else:
        watermark_method = ""

    if removal_method is not None:
        removal_method = f" - {removal_method} (" + ", ".join([f"{k}{v}" for k, v in kwargs.items()]) + ")"
    else:
        removal_method = ""

    draw = ImageDraw.Draw(image)
    for i, txt in enumerate(["Original", "Watermarked" + watermark_method, "Optimized" + removal_method, "Watermark" + watermark_method, "Removed Watermark" + removal_method, "Residual" + removal_method]):
        draw.text((10, 10 + i * 256), txt, font=font)

    return image


def devel_main():
    device = "cuda:1"
    model = get_artifact_discriminator(device=device)
    removal_method_name = "ArtifactDisc"
    opt_params = dict(num_steps=100, lr=0.05)
    # model = get_clip_model(device=device)
    # removal_method_name = "CLIP-IQA"
    # opt_params = dict(num_steps=100, lr=1)

    ROOT = "/private/home/soucek/videoseal/metrics"
    watermarking_methods = sorted([os.path.basename(x) for x in glob.glob(f"{ROOT}/*")])
    
    output_dir = os.path.join(".", f"visualizations/{removal_method_name}")
    os.makedirs(output_dir, exist_ok=True)

    for wm_method in watermarking_methods:
        imageds = DummyPairedDataset(wm_method, max_size=768)

        for i, (img_wm, img_ori) in enumerate(tqdm.tqdm(imageds, leave=False)):
            optimized_img = optimize(img_wm, model, device=device, **opt_params)
            visualization = outputs_to_image(img_ori, img_wm, optimized_img, watermark_method=wm_method, removal_method=removal_method_name, **opt_params)
            visualization.save(os.path.join(output_dir, f"{wm_method}_{i}.png"))


def optimize_multiple(img: Image, model, device="cuda:0", num_steps=[100], lr=0.05, patchify=True):
    if patchify:
        img = patchify_image(img).to(device)
    else:
        img = transform_image(img).to(device)
    param = torch.nn.Parameter(torch.zeros_like(img)).to(device)

    optim = torch.optim.SGD([param], lr=lr)

    out_imgs = []
    for i in range(1, max(num_steps) + 1):
        optim.zero_grad()
        loss = -model((img + param).clip(0, 1)).mean()
        loss.backward()
        optim.step()

        if i in num_steps:
            out_img = (img + param).detach().clip(0, 1).mul_(255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
            if patchify:
                out_img = out_img.view(3, 3, 256, 256, 3).permute(1, 2, 0, 3, 4)
            out_img = out_img.reshape(768, 768, 3).numpy()
            out_img = Image.fromarray(out_img)
            out_imgs.append(out_img)
    
    return out_imgs


if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Usage: python artifact_discriminator_watermark_removal.py <image_dir> [output_dir]"
    image_dir = sys.argv[1]
    output_dir = os.path.normpath(sys.argv[2] if len(sys.argv) > 2 else "output")

    device = "cuda"
    model = get_artifact_discriminator(device=device)

    strengths = [50, 100, 200, 300, 500]
    for i in strengths:
        os.makedirs(output_dir + f"_{i:03d}", exist_ok=True)
    
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    for image_file in tqdm.tqdm(image_files):
        image_ori = Image.open(image_file)
        
        images = optimize_multiple(image_ori, model, device=device, num_steps=strengths, patchify=True)
        for i, image in zip(strengths, images):
            image.save(os.path.join(output_dir + f"_{i:03d}", os.path.basename(image_file)))
