"""
Test with:
    python -m videoseal.augmentation.valuemetric
"""

import io

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

from ..utils.image import jpeg_compress, median_filter


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

