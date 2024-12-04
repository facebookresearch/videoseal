"""
Test with:
    python -m videoseal.augmentation.geometric
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, image, mask, *args, **kwargs):
        return image, mask
    
    def __repr__(self):
        return f"Identity"


class Rotate(nn.Module):
    def __init__(self, min_angle=None, max_angle=None):
        super(Rotate, self).__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle

    def get_random_angle(self):
        if self.min_angle is None or self.max_angle is None:
            raise ValueError("min_angle and max_angle must be provided")
        return torch.randint(self.min_angle, self.max_angle + 1, size=(1,)).item()

    def forward(self, image, mask, angle=None):
        angle = angle or self.get_random_angle()
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)
        return image, mask
    
    def __repr__(self):
        return f"Rotate"


class Resize(nn.Module):
    def __init__(self, min_size=None, max_size=None):
        super(Resize, self).__init__()
        # float between 0 and 1, representing the total area of the output image compared to the input image
        self.min_size = min_size
        self.max_size = max_size

    def get_random_size(self, h, w):
        if self.min_size is None or self.max_size is None:
            raise ValueError("min_size and max_size must be provided")
        output_size = (
            torch.randint(int(self.min_size * h),
                          int(self.max_size * h) + 1, size=(1, )).item(),
            torch.randint(int(self.min_size * w),
                          int(self.max_size * w) + 1, size=(1, )).item()
        )
        return output_size

    def forward(self, image, mask, size=None):
        h, w = image.shape[-2:]
        if size is None:
            output_size = self.get_random_size(h, w)
        else:
            output_size = (int(size * h), int(size * w))
        image = F.resize(image, output_size, antialias=True)
        mask = F.resize(mask, output_size, antialias=True)
        return image, mask
    
    def __repr__(self):
        return f"Resize"


class Crop(nn.Module):
    def __init__(self, min_size=None, max_size=None):
        super(Crop, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

    def get_random_size(self, h, w):
        if self.min_size is None or self.max_size is None:
            raise ValueError("min_size and max_size must be provided")
        output_size = (
            torch.randint(int(self.min_size * h),
                          int(self.max_size * h) + 1, size=(1, )).item(),
            torch.randint(int(self.min_size * w),
                          int(self.max_size * w) + 1, size=(1, )).item()
        )
        return output_size

    def forward(self, image, mask, size=None):
        h, w = image.shape[-2:]
        if size is None:
            output_size = self.get_random_size(h, w)
        else:
            output_size = (int(size * h), int(size * w))
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=output_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return image, mask

    def __repr__(self):
        return f"Crop"


class Perspective(nn.Module):
    def __init__(self, min_distortion_scale=None, max_distortion_scale=None):
        super(Perspective, self).__init__()
        self.min_distortion_scale = min_distortion_scale
        self.max_distortion_scale = max_distortion_scale

    def get_random_distortion_scale(self):
        if self.min_distortion_scale is None or self.max_distortion_scale is None:
            raise ValueError(
                "min_distortion_scale and max_distortion_scale must be provided")
        return self.min_distortion_scale + torch.rand(1).item() * \
            (self.max_distortion_scale - self.min_distortion_scale)

    def forward(self, image, mask, distortion_scale=None):
        distortion_scale = distortion_scale or self.get_random_distortion_scale()
        width, height = image.shape[-1], image.shape[-2]
        startpoints, endpoints = self.get_perspective_params(
            width, height, distortion_scale)
        image = F.perspective(image, startpoints, endpoints)
        mask = F.perspective(mask, startpoints, endpoints)
        return image, mask

    @staticmethod
    def get_perspective_params(width, height, distortion_scale):
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale *
                half_width) + 1, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale *
                half_height) + 1, size=(1, )).item())
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale *
                half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale *
                half_height) + 1, size=(1, )).item())
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale *
                half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale *
                half_height) - 1, height, size=(1, )).item())
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale *
                half_width) + 1, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale *
                half_height) - 1, height, size=(1, )).item())
        ]
        startpoints = [[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return f"Perspective"


class HorizontalFlip(nn.Module):
    def __init__(self):
        super(HorizontalFlip, self).__init__()

    def forward(self, image, mask, *args, **kwargs):
        image = F.hflip(image)
        mask = F.hflip(mask)
        return image, mask

    def __repr__(self):
        return f"HorizontalFlip"
