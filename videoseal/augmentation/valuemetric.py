# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


if __name__ == "__main__":
    import os

    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor
    from torchvision.utils import save_image

    from ..data.transforms import default_transform

    # Define the transformations and their parameter ranges
    transformations = [
        (Brightness, [0.5, 1.5]),
        (Contrast, [0.5, 1.5]),
        (Saturation, [0.5, 1.5]),
        (Hue, [-0.5, -0.25, 0.25, 0.5]),
        (JPEG, [40, 60, 80]),
        (GaussianBlur, [3, 5, 9, 17]),
        (MedianFilter, [3, 5, 9, 17]),
        (GaussianNoise, [0.05, 0.1, 0.15, 0.2]),
        (Grayscale, [-1]),  # Grayscale doesn't need a strength parameter
        # (bmshj2018, [2, 4, 6, 8])
    ]

    # Load images
    imgs = [
        Image.open("/private/home/pfz/_images/gauguin_256.png"),
        Image.open("/private/home/pfz/_images/tahiti_256.png")
    ]
    imgs = torch.stack([default_transform(img) for img in imgs])

    # Create the output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Sweep over the strengths for each augmentation
    for transform, strengths in transformations:
        for strength in strengths:
            # Create an instance of the transformation
            transform_instance = transform()

            # Apply the transformation to the images
            imgs_transformed, _ = transform_instance(imgs, None, strength)

            # Save the transformed images
            filename = f"{transform.__name__}_strength_{strength}.png"
            save_image(imgs_transformed.clamp(0, 1), os.path.join(output_dir, filename))

            # Print the path to the saved image
            print(f"Saved transformed images ({transform.__name__}, strength={strength}) to:", os.path.join(
                output_dir, filename))
