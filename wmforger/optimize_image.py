# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tqdm
import torch
import omegaconf
import torchvision
import numpy as np
from PIL import Image

from wmforger.models import build_extractor


transform_image = torchvision.transforms.Compose([
    lambda x: x.convert("RGB"),
    torchvision.transforms.Resize((768, 768)),
    torchvision.transforms.ToTensor(),
    lambda x: x.view(1, 3, 768, 768),
])


def get_artifact_discriminator(ckpt_path, device="cuda:0"):
    model_type = "convnext_tiny"
    state_dict = torch.load(ckpt_path, weights_only=True, map_location="cpu")["model"]
    extractor_params = omegaconf.OmegaConf.load("configs/extractor.yaml")[model_type]

    model = build_extractor(model_type, extractor_params, img_size=256, nbits=0)
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    return model


def optimize(img: Image, model, device="cuda:0", num_steps=64, lr=0.2):
    """
    Optimizes the image at 768x768 to maximize the preference model output.
    The number of steps sets the strength of the attack (more steps, larger perturbation).
    """
    img = transform_image(img).to(device)
    param = torch.nn.Parameter(torch.zeros_like(img)).to(device)

    optim = torch.optim.SGD([param], lr=lr)
    for _ in tqdm.tqdm(range(num_steps)):
        optim.zero_grad()
        loss = -model((img + param).clip(0, 1)).mean()
        loss.backward()
        optim.step()
    
    return (img + param).clip(0, 1).detach().cpu()


def get_watermark(img: Image, optimized_img: torch.Tensor):
    """
    Returns the cleaned image and the stolen watermark (in [0, 255] units, at the size of img).

    Only the perturbation found by the optimization is resized back to the original size, i.e. the
    watermark is resize(transform_image(img) - optimized_img). Taking the original image minus the
    optimized image resized back instead would add the error of the resize round trip
    (original -> 768x768 -> original), which on typical images is as large as the watermark itself.
    """
    img_small = transform_image(img).to(optimized_img.device)
    watermark = torch.nn.functional.interpolate(
        img_small - optimized_img, size=img.size[::-1], mode="bilinear", align_corners=True, antialias=False
    )
    img_torch = torchvision.transforms.functional.to_tensor(img.convert("RGB")).unsqueeze(0)
    cleaned = (img_torch - watermark.cpu()).clip(0, 1)
    cleaned_img = Image.fromarray(cleaned.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).squeeze(0).numpy())
    return cleaned_img, watermark.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to the image file", required=True)
    parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint file", required=True)
    parser.add_argument("--num_steps", type=int, default=64,
                        help="Optimization steps; more steps remove more of the watermark but change the image more")
    parser.add_argument("--lr", type=float, default=0.2, help="SGD learning rate")
    args = parser.parse_args()

    device = "cuda"
    model = get_artifact_discriminator(ckpt_path=args.ckpt_path, device=device)

    image = Image.open(args.image).convert("RGB")
    optimized_img = optimize(image, model, device=device, num_steps=args.num_steps, lr=args.lr)
    
    cleaned_image, watermark = get_watermark(image, optimized_img)

    os.makedirs("output", exist_ok=True)
    Image.fromarray(np.round(np.abs(watermark * 16)).astype(np.uint8)).save("output/watermark.png")
    cleaned_image.save("output/cleaned_image.png")
    print("Extracted watermark saved to output/watermark.png")
