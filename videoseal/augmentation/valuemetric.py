"""
Test with:
    python -m videoseal.augmentation.valuemetric
"""

import io
import sys
import random
import typing as tp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import v2
from PIL import Image, ImageFont, ImageDraw

from ..data.transforms import default_transform


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
        self.relative_strength = 1.0

    def get_random_quality(self):
        if self.min_quality is None or self.max_quality is None:
            raise ValueError("Quality range must be specified")
        quality = torch.randint(self.min_quality, self.max_quality + 1, size=(1,)).item()
        return round(quality + (100 - quality) * (1 - self.relative_strength))

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
        return "JPEG"


class GaussianBlur(nn.Module):
    def __init__(self, min_kernel_size=None, max_kernel_size=None):
        super(GaussianBlur, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.relative_strength = 1.0

    def get_random_kernel_size(self):
        if self.min_kernel_size is None or self.max_kernel_size is None:
            raise ValueError("Kernel size range must be specified")
        kernel_size = torch.randint(self.min_kernel_size, self.max_kernel_size + 1, size=(1,)).item()
        kernel_size = round(kernel_size + (1 - kernel_size) * (1 - self.relative_strength))
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
        self.relative_strength = 1.0

    def get_random_kernel_size(self):
        if self.min_kernel_size is None or self.max_kernel_size is None:
            raise ValueError("Kernel size range must be specified")
        kernel_size = torch.randint(self.min_kernel_size, self.max_kernel_size + 1, size=(1,)).item()
        kernel_size = round(kernel_size + (1 - kernel_size) * (1 - self.relative_strength))
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


class Pad(nn.Module):
    def __init__(self, img_size: int = 256, pad_value: tp.Optional[int] = None, min_pad: int = None, max_pad: int = None):
        super(Pad, self).__init__()
        if pad_value is not None:
            txt_ = f"Warning: Using fixed padding value {pad_value} is deprecated. Please set `min_pad` and `max_pad` instead."
            print(txt_, file=sys.stderr, flush=True)

        self.min_pad = min_pad
        self.max_pad = max_pad
        self.pad = None

        # If padding is not specified, we get random border
        if pad_value is None:
            pad_value = random.randint(0, img_size // 2)
            self.pad = v2.Pad(padding=pad_value)

        self.relative_strength = 1.0

    def forward(self, image):
        if self.pad is not None:
            pad_image = self.pad(image)
        else:
            pad_value = self.min_pad + random.randint(self.min_pad, self.max_pad)
            pad_value = round(pad_value * self.relative_strength)
            if pad_value == 0:
                return image
            pad_image = v2.Pad(padding=pad_value)(image)

        adapt_dim = False
        if len(pad_image.shape) < 4:  # c h w  -> b c h w
            adapt_dim = True
            pad_image = pad_image.unsqueeze(0)

        # Resize to original image size
        h, w = image.size(-2), image.size(-1)
        pad_image = nn.functional.interpolate(pad_image, size=(h, w), mode="bilinear", align_corners=True)
        if adapt_dim:
            pad_image = pad_image.squeeze(0)
        return pad_image


class InsertMemeText(nn.Module):

    def __init__(self, text: str, color: str = "white", image_size: int = 256, font_size: int = 24, rel_pos: int = 10):
        super(InsertMemeText, self).__init__()

        # Get font
        self.font = self._prepare_font(font_size)

        # Split long text in multiple lines, with space between them
        self.lines = self._prepare_lines(text, image_size)

        self.rel_pos = rel_pos
        self.color = color
        self.relative_strength = 1.0

    def _prepare_font(self, font_size: int = 24):
        # Get a true type font safely via matplotlib's font_manager.
        # We could get rid of this and use a separate .ttf file instead, but fonts like Arial
        # can be tricky due to licensing
        from matplotlib import font_manager
        dejavu_font_path = font_manager.findfont(font_manager.FontProperties())
        return ImageFont.truetype(dejavu_font_path, size=font_size)

    def _prepare_lines(self, text: str, image_size: int = 256):

        # Since all images are of equal size, we can probe one fake image to calculate the wrap text
        fake_image = transforms.ToPILImage()(torch.rand(3, image_size, image_size))
        draw = ImageDraw.Draw(fake_image)
        words = text.split()
        lines = []
        line = ""
        max_width = image_size - 5

        for word in words:
            _line = f"{line} {word}"
            if draw.textlength(_line, font=self.font) <= max_width:
                line = _line
            else:
                lines.append(line)
                line = word

        # flush the last line
        if line:
            lines.append(line)

        ascent, descent = self.font.getmetrics()
        line_height = ascent + descent + 5  # Add spacing

        return lines, line_height

    def meme_single(self, img):
        pil_image = transforms.ToPILImage()(img)
        draw = ImageDraw.Draw(pil_image)
        lines, line_height = self.lines
        width = pil_image.size[0]
        x, y = width // 2, self.rel_pos
        for line in lines:
            draw.text((x, y), line, font=self.font, fill=self.color, anchor="mm")
            y += line_height

        return default_transform(pil_image)

    def forward(self, image):
        if self.relative_strength < 0.5:
            return image

        """Add the meme text to the image tensor at the relative position"""
        if len(image.size()) == 4:  # batch mode
            img_memes = [self.meme_single(i) for i in image]
        else:
            img_memes = [self.meme_single(image)]

        return torch.stack(img_memes)


class InsertLogo(nn.Module):
    def __init__(self, logo_path: str, image_size: int = 256, logo_scale: float = 0.2):
        super(InsertLogo, self).__init__()
        logo_image = Image.open(logo_path).convert("RGBA")
        logo_max_size = int(min(image_size, image_size) * logo_scale)
        logo_image.thumbnail((logo_max_size, logo_max_size), Image.Resampling.LANCZOS)
        logo_width, logo_height = logo_image.size
        self.logo_pos = (image_size - logo_width, image_size - logo_height)
        self.logo_img = logo_image
        self.relative_strength = 1.0

    def embed_logo_single(self, img):
        pil_image = transforms.ToPILImage()(img)
        pil_image.paste(self.logo_img, self.logo_pos, self.logo_img)
        return default_transform(pil_image)

    def forward(self, image):
        if self.relative_strength < 0.5:
            return image

        if len(image.size()) == 4:  # batch mode
            img_logos = [self.embed_logo_single(i) for i in image]
        else:
            img_logos = [self.embed_logo_single(image)]

        return torch.stack(img_logos)


class Brightness(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Brightness, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.relative_strength = 1.0

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("min_factor and max_factor must be provided")
        factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        return factor + (1 - factor) * (1 - self.relative_strength)

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
        self.relative_strength = 1.0

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("min_factor and max_factor must be provided")
        factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        return factor + (1 - factor) * (1 - self.relative_strength)

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
        self.relative_strength = 1.0

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("Factor range must be specified")
        factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        return factor + (1 - factor) * (1 - self.relative_strength)

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
        self.relative_strength = 1.0

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("Factor range must be specified")
        factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        return factor * self.relative_strength

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
        self.relative_strength = 1.0

    def get_random_std(self):
        if self.min_std is None or self.max_std is None:
            raise ValueError("Standard deviation range must be specified")
        std = torch.rand(1).item() * (self.max_std - self.min_std) + self.min_std
        return std * self.relative_strength

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
        self.relative_strength = 1.0
        
    def forward(self, image, mask, *args, **kwargs):
        """
        Convert image to grayscale. The strength parameter is ignored.
        """
        # Convert to grayscale using the ITU-R BT.601 standard (luma component)
        # Y = 0.299 R + 0.587 G + 0.114 B
        grayscale = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        grayscale = grayscale.expand_as(image)
        grayscale = self.relative_strength * grayscale + (1 - self.relative_strength) * image
        return grayscale, mask

    def __repr__(self):
        return f"Grayscale"


if __name__ == "__main__":
    import os

    import torch
    from PIL import Image
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
