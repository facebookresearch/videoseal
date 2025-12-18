# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
from PIL import Image

import torch
import torchvision.transforms as transforms


def jpeg_compress(image: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Compress a PyTorch image using JPEG compression and return as a PyTorch tensor.

    Parameters:
        image (torch.Tensor): The input image tensor of shape 3xhxw.
        quality (int): The JPEG quality factor.

    Returns:
        torch.Tensor: The compressed image as a PyTorch tensor.
    """
    assert image.min() >= 0 and image.max(
    ) <= 1, f'Image pixel values must be in the range [0, 1], got [{image.min()}, {image.max()}]'
    pil_image = transforms.ToPILImage()(image)  # convert to PIL image
    # Create a BytesIO object and save the PIL image as JPEG to this object
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    # Load the JPEG image from the BytesIO object and convert back to a PyTorch tensor
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    tensor_image = transforms.ToTensor()(compressed_image)
    return tensor_image


def webp_compress(image: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Compress a PyTorch image using WebP compression and return as a PyTorch tensor.

    Parameters:
        image (torch.Tensor): The input image tensor of shape 3xhxw.
        quality (int): The WebP quality factor.

    Returns:
        torch.Tensor: The compressed image as a PyTorch tensor.
    """
    image = torch.clamp(image, 0, 1)  # clamp the pixel values to [0, 1]
    pil_image = transforms.ToPILImage()(image)  # convert to PIL image
    # Create a BytesIO object and save the PIL image as WebP to this object
    buffer = io.BytesIO()
    pil_image.save(buffer, format='WebP', quality=quality)
    # Load the WebP image from the BytesIO object and convert back to a PyTorch tensor
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    tensor_image = transforms.ToTensor()(compressed_image)
    return tensor_image


def median_filter(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply a median filter to a batch of images.

    Parameters:
        images (torch.Tensor): The input images tensor of shape BxCxHxW.
        kernel_size (int): The size of the median filter kernel.

    Returns:
        torch.Tensor: The filtered images.
    """
    # Ensure the kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    # Compute the padding size
    padding = kernel_size // 2
    # Pad the images
    images_padded = torch.nn.functional.pad(
        images, (padding, padding, padding, padding))
    # Extract local blocks from the images
    blocks = images_padded.unfold(2, kernel_size, 1).unfold(
        3, kernel_size, 1)  # BxCxHxWxKxK
    # Compute the median of each block
    medians = blocks.median(dim=-1).values.median(dim=-1).values  # BxCxHxW
    return medians


def create_diff_img(img1, img2):
    """
    Create a difference image between two images.

    Parameters:
        img1 (torch.Tensor): The first image tensor of shape 3xHxW.
        img2 (torch.Tensor): The second image tensor of shape 3xHxW.

    Returns:
        torch.Tensor: The difference image tensor of shape 3xHxW.
    """
    diff = img1 - img2
    # diff = 0.5 + 10*(img1 - img2)
    # normalize the difference image
    diff = (diff - diff.min()) / ((diff.max() - diff.min()) + 1e-6)
    diff = 2*torch.abs(diff - 0.5)
    # diff = 20*torch.abs(diff)
    return diff.clamp(0, 1)


if __name__ == '__main__':
    # Example usage: python src/utils/image.py
    x = torch.rand(3, 256, 256)  # random image
    x_jpeg = jpeg_compress(x, 80)  # compress
    x_webp = webp_compress(x, 80)  # compress

    print(x[0, 0:5, 0:5])  # print the first 5x5 pixels of the first channel
    # print the first 5x5 pixels of the first channel
    print(x_jpeg[0, 0:5, 0:5])
    # print the first 5x5 pixels of the first channel
    print(x_webp[0, 0:5, 0:5])
