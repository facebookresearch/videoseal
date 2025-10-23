# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


def jpeg_compress(image: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Compress a PyTorch image using JPEG compression and return as a PyTorch tensor.

    Parameters:
        image (torch.Tensor): The input image tensor of shape 3xhxw.
        quality (int): The JPEG quality factor.

    Returns:
        torch.Tensor: The compressed image as a PyTorch tensor.
    """
    image = torch.clamp(image, 0, 1)  # clamp the pixel values to [0, 1]
    image = (image * 255).round() / 255 
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
    image = (image * 255).round() / 255 
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


class JPEG(nn.Module):
    def __init__(self, min_quality=None, max_quality=None, passthrough=True):
        super(JPEG, self).__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.passthrough = passthrough

    def get_random_quality(self):
        if self.min_quality is None or self.max_quality is None:
            raise ValueError("Quality range must be specified")
        return torch.randint(self.min_quality, self.max_quality + 1, size=(1,)).item()

    def jpeg_single(self, image, quality):
        if self.passthrough:
            return (jpeg_compress(image, quality).to(image.device) - image).detach() + image
        else:
            return jpeg_compress(image, quality).to(image.device)

    def forward(self, image: torch.tensor, mask, quality=None):
        quality = quality or self.get_random_quality()
        image = torch.clamp(image, 0, 1)
        if len(image.shape) == 4:  # b c h w
            for ii in range(image.shape[0]):
                image[ii] = self.jpeg_single(image[ii], quality)
        else:
            image = self.jpeg_single(image, quality)
        return image, mask
    
    def __repr__(self):
        return f"JPEG"


class GaussianBlur(nn.Module):
    def __init__(self, min_kernel_size=None, max_kernel_size=None):
        super(GaussianBlur, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size

    def get_random_kernel_size(self):
        if self.min_kernel_size is None or self.max_kernel_size is None:
            raise ValueError("Kernel size range must be specified")
        kernel_size = torch.randint(self.min_kernel_size, self.max_kernel_size + 1, size=(1,)).item()
        return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    def forward(self, image, mask, kernel_size=None):
        kernel_size = kernel_size or self.get_random_kernel_size()
        image = F.gaussian_blur(image, kernel_size)
        return image, mask

    def __repr__(self):
        return f"GaussianBlur"


class MedianFilter(nn.Module):
    def __init__(self, min_kernel_size=None, max_kernel_size=None, passthrough=True):
        super(MedianFilter, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.passthrough = passthrough

    def get_random_kernel_size(self):
        if self.min_kernel_size is None or self.max_kernel_size is None:
            raise ValueError("Kernel size range must be specified")
        kernel_size = torch.randint(self.min_kernel_size, self.max_kernel_size + 1, size=(1,)).item()
        return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    def forward(self, image, mask, kernel_size=None):
        kernel_size = kernel_size or self.get_random_kernel_size()
        if self.passthrough:
            image = (median_filter(image, kernel_size) - image).detach() + image
        else:
            image = median_filter(image, kernel_size)
        return image, mask
    
    def __repr__(self):
        return f"MedianFilter"


class Brightness(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Brightness, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("min_factor and max_factor must be provided")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image, mask, factor=None):
        factor = self.get_random_factor() if factor is None else factor
        image = F.adjust_brightness(image, factor)
        return image, mask

    def __repr__(self):
        return f"Brightness"


class Contrast(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Contrast, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("min_factor and max_factor must be provided")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image, mask, factor=None):
        factor = self.get_random_factor() if factor is None else factor
        image = F.adjust_contrast(image, factor)
        return image, mask

    def __repr__(self):
        return f"Contrast"

class Saturation(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Saturation, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("Factor range must be specified")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image, mask, factor=None):
        factor = self.get_random_factor() if factor is None else factor
        image = F.adjust_saturation(image, factor)
        return image, mask

    def __repr__(self):
        return f"Saturation"

class Hue(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Hue, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("Factor range must be specified")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image, mask, factor=None):
        factor = self.get_random_factor() if factor is None else factor
        image = F.adjust_hue(image, factor)
        return image, mask

    def __repr__(self):
        return f"Hue"

class GaussianNoise(nn.Module):
    def __init__(self, min_std=None, max_std=None):
        super(GaussianNoise, self).__init__()
        self.min_std = min_std
        self.max_std = max_std

    def get_random_std(self):
        if self.min_std is None or self.max_std is None:
            raise ValueError("Standard deviation range must be specified")
        return torch.rand(1).item() * (self.max_std - self.min_std) + self.min_std

    def forward(self, image, mask, std=None):
        std = self.get_random_std() if std is None else std
        noise = torch.randn_like(image) * std
        image = image + noise
        return image, mask

    def __repr__(self):
        return f"GaussianNoise"


class Grayscale(nn.Module):
    def __init__(self):
        super(Grayscale, self).__init__()
        
    def forward(self, image, mask, *args, **kwargs):
        """
        Convert image to grayscale. The strength parameter is ignored.
        """
        # Convert to grayscale using the ITU-R BT.601 standard (luma component)
        # Y = 0.299 R + 0.587 G + 0.114 B
        grayscale = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        grayscale = grayscale.expand_as(image)
        return grayscale, mask

    def __repr__(self):
        return f"Grayscale"
