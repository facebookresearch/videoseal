"""
Test with:
    python -m videoseal.augmentation.overlay
"""

import glob
import os
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import draw_bounding_boxes


def _get_default_font_path() -> str:
    """Find a default DejaVu font path for text rendering."""
    try:
        return font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    except Exception:
        print("Warning: DejaVu Sans not found. Falling back to default font.")
        return font_manager.findfont(font_manager.FontProperties())


def _load_emojis_from_dir(emoji_dir: str | None, min_side: int = 32) -> List[Image.Image]:
    """
    Load all image files in ``emoji_dir`` as RGBA PIL images.
    If none are found, fall back to a small procedural set.
    """
    imgs: List[Image.Image] = []
    local_dir = os.path.expanduser(emoji_dir)
    if os.path.isdir(local_dir):
        patterns = ["*.png", "*.webp", "*.jpg", "*.jpeg"]
        files: List[str] = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(local_dir, "**", pattern), recursive=True))
        for file_path in sorted(files):
            im = Image.open(file_path).convert("RGBA")
            if min(im.size) < min_side:
                im = im.resize((max(min_side, im.size[0]), max(min_side, im.size[1])), Image.LANCZOS)
            imgs.append(im)
    else:
        print(f"No emoji directory found at {local_dir!r}. To use emoji overlays, run `unzip assets/twemojis.zip`.")
        imgs = [
            Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        ]  # red square
    return imgs


DEFAULT_EMOJI_DIR = str(Path(__file__).resolve().parents[2] / "assets" / "twemojis")


class InsertMemeText(nn.Module):
    """
    Adds meme-style text padding (top, bottom, or both) and resizes back.
    Applies the same padding and text to all images in a batch.
    Uses ``torchvision.utils.draw_bounding_boxes`` for text rendering (no PIL text drawing).
    """

    def __init__(self, min_pad_factor: float = 0.05, max_pad_factor: float = 0.2):
        super().__init__()
        self.min_pad_factor = min_pad_factor
        self.max_pad_factor = max_pad_factor
        self.relative_strength = 1.0
        self.font_path = _get_default_font_path()
        self.meme_texts = [
            "THIS IS FINE",
            "EPIC FAIL",
            "MUCH WOW",
            "VERY NICE",
            "SO COOL",
            "TOP QUALITY",
            "BEST EVER",
            "STONKS",
            "NOT STONKS",
            "BIG BRAIN",
            "SMOL BRAIN",
            "POV:",
            "NOBODY:",
            "ME:",
            "MOOD",
            "SAME ENERGY",
            "BASED",
            "CRINGE",
            "RATIO",
            "L + RATIO",
            "WHEN YOU...",
            "I CAN HAZ",
            "Y U NO",
            "ALL THE",
            "FOMO",
            "YOLO",
            "FLEX",
            "VIBE CHECK",
        ]

    def get_random_pad_factor(self) -> float:
        factor = torch.rand(1).item() * (self.max_pad_factor - self.min_pad_factor) + self.min_pad_factor
        return factor * self.relative_strength

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None, pad_factor: float | None = None):
        if self.relative_strength < 0.5 or len(image.shape) != 4:
            return image, mask

        pad_factor = self.get_random_pad_factor() if pad_factor is None else pad_factor

        b, c, h, w = image.shape
        device = image.device

        pad_location = random.randint(0, 2)

        if pad_location == 0:  # top only
            pad_h = int(h * pad_factor)
            new_h = h + pad_h
            top_pad_h, bottom_pad_h = pad_h, 0
        elif pad_location == 1:  # bottom only
            pad_h = int(h * pad_factor)
            new_h = h + pad_h
            top_pad_h, bottom_pad_h = 0, pad_h
        else:  # both
            pad_h = int(h * pad_factor / 2)
            new_h = h + 2 * pad_h
            top_pad_h, bottom_pad_h = pad_h, pad_h

        fill_color = torch.rand(1, c, 1, 1, device=device)

        canvas = torch.zeros((b, c, new_h, w), dtype=image.dtype, device=device)
        canvas = canvas + fill_color
        canvas[:, :, top_pad_h : top_pad_h + h, :] = image

        text = random.choice(self.meme_texts)
        bottom_text = random.choice(self.meme_texts)
        font_size = max(6, min(int(max(top_pad_h, bottom_pad_h) * 0.6), 96))

        boxes = []
        labels = []
        if top_pad_h > 0:
            boxes.append(torch.tensor([0, 0, w - 1, max(0, top_pad_h - 1)], device=device))
            labels.append(text)
        if bottom_pad_h > 0:
            boxes.append(torch.tensor([0, top_pad_h + h, w - 1, new_h - 1], device=device))
            labels.append(bottom_text)
        boxes_tensor = torch.stack(boxes) if boxes else torch.empty((0, 4), device=device)

        if boxes_tensor.numel() == 0:
            drawn_tensor = canvas
        else:
            colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(b)]
            drawn = [
                draw_bounding_boxes(
                    canvas[ii],
                    colors=colors[ii] if len(colors) > ii else (255, 255, 255),
                    boxes=boxes_tensor,
                    labels=labels,
                    fill=False,
                    width=1,
                    font=self.font_path,
                    font_size=font_size,
                )
                for ii in range(b)
            ]
            drawn_tensor = torch.stack(drawn)

        if mask is not None:
            new_mask = torch.zeros((b, mask.shape[1], new_h, w), dtype=mask.dtype, device=device)
            new_mask[:, :, top_pad_h : top_pad_h + h, :] = mask
            mask = new_mask

        return drawn_tensor, mask

    def __repr__(self) -> str:
        return f"InsertMemeText(min_pad={self.min_pad_factor}, max_pad={self.max_pad_factor})"


class InsertText(nn.Module):
    """
    Inserts random text over a batch of images.
    Applies the *same* text and position to all images in the batch.
    """

    def __init__(self, min_font_size: int = 12, max_font_size: int = 48, num_words: int = 3):
        super().__init__()
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.num_words = num_words
        self.relative_strength = 1.0
        self.font_path = _get_default_font_path()
        self.words = [
            "SALE",
            "NEW",
            "HOT",
            "COOL",
            "BEST",
            "TOP",
            "WOW",
            "LOL",
            "OMG",
            "DEAL",
            "FREE",
            "WIN",
            "BONUS",
            "GIFT",
            "SPECIAL",
            "LIMITED",
        ]

    def _get_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self.font_path, size=font_size)

    def get_random_font_size(self) -> int:
        sampled = torch.randint(self.min_font_size, self.max_font_size + 1, size=(1,)).item()
        return round(self.min_font_size + (sampled - self.min_font_size) * self.relative_strength)

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None, font_size: int | None = None):
        if self.relative_strength < 0.5 or len(image.shape) != 4:
            return image, mask

        font_size = self.get_random_font_size() if font_size is None else font_size

        b, c, h, w = image.shape
        device = image.device

        text = " ".join(random.choices(self.words, k=self.num_words))
        color_tensor = torch.rand(1, c, 1, 1, device=device)
        font = self._get_font(font_size)

        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        offset_x = bbox[0]
        offset_y = bbox[1]

        if text_w == 0 or text_h == 0:
            return image, mask

        pil_mask = Image.new("L", (text_w, text_h), 0)
        draw_mask = ImageDraw.Draw(pil_mask)
        draw_mask.text((-offset_x, -offset_y), text, font=font, fill=255)

        mask_tensor = TF.to_tensor(pil_mask).to(device).unsqueeze(0)
        stamp = color_tensor.expand(-1, -1, text_h, text_w)

        x = random.randint(0, max(0, w - text_w))
        y = random.randint(0, max(0, h - text_h))

        y_start_img = max(0, y)
        y_end_img = min(h, y + text_h)
        x_start_img = max(0, x)
        x_end_img = min(w, x + text_w)

        y_start_stamp = max(0, -y)
        y_end_stamp = min(text_h, h - y)
        x_start_stamp = max(0, -x)
        x_end_stamp = min(text_w, w - x)

        if y_start_img >= y_end_img or x_start_img >= x_end_img:
            return image, mask

        img_slice = image[:, :, y_start_img:y_end_img, x_start_img:x_end_img]
        stamp_slice = stamp[:, :, y_start_stamp:y_end_stamp, x_start_stamp:x_end_stamp]
        mask_slice = mask_tensor[:, :, y_start_stamp:y_end_stamp, x_start_stamp:x_end_stamp]

        blended_slice = img_slice * (1 - mask_slice) + stamp_slice * mask_slice

        image_out = image.clone()
        image_out[:, :, y_start_img:y_end_img, x_start_img:x_end_img] = blended_slice

        if mask is not None:
            mask_out = mask.clone()
            added_pixels = (mask_slice > 0).to(mask_out.dtype)
            mask_out[:, :, y_start_img:y_end_img, x_start_img:x_end_img] *= (1 - added_pixels)
            mask = mask_out

        return image_out, mask

    def __repr__(self) -> str:
        return f"InsertText(min_font={self.min_font_size}, max_font={self.max_font_size})"


class InsertEmoji(nn.Module):
    """
    Overlay a random emoji (same for the batch) onto tensor images.
    ``min_size``/``max_size`` are treated as proportions of the image's smaller edge.
    Mostly copy-pasted from AugLy (github.com/facebookresearch/AugLy).
    """

    def __init__(self, min_size: float = 0.06, max_size: float = 0.2, emoji_dir: str | None = None):
        super().__init__()
        assert 0.0 < min_size <= max_size <= 1.0, "min_size/max_size must be proportions in (0,1]"
        self.min_size = min_size
        self.max_size = max_size
        self.relative_strength = 1.0
        self.emoji_dir = emoji_dir or DEFAULT_EMOJI_DIR
        self._emoji_pils = _load_emojis_from_dir(self.emoji_dir)

    def _choose_and_prepare_stamp(
        self, target_device: torch.device, target_dtype: torch.dtype, target_edge: int, size_px: int
    ):
        pil = random.choice(self._emoji_pils)
        w0, h0 = pil.size
        scale = size_px / max(w0, h0)
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        pil_resized = pil.resize((new_w, new_h), Image.LANCZOS)

        stamp_rgba = TF.to_tensor(pil_resized).to(device=target_device, dtype=target_dtype)
        if stamp_rgba.size(0) == 3:
            alpha = torch.ones(1, stamp_rgba.size(1), stamp_rgba.size(2), device=target_device, dtype=target_dtype)
            stamp_rgba = torch.cat([stamp_rgba, alpha], dim=0)
        stamp_rgb = stamp_rgba[:3].unsqueeze(0)
        stamp_alpha = stamp_rgba[3:4].unsqueeze(0)
        return stamp_rgb, stamp_alpha

    def get_random_size_px(self, h: int, w: int, emoji_size_override: int | None = None) -> int:
        if emoji_size_override is not None:
            return int(emoji_size_override)
        edge = min(h, w)
        prop = (self.min_size + (self.max_size - self.min_size) * random.random()) * self.relative_strength
        prop = max(self.min_size, min(prop, self.max_size))
        return max(4, int(round(edge * prop)))

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None, emoji_size: int | None = None):
        if self.relative_strength < 0.5 or len(image.shape) != 4:
            return image, mask

        b, c, h, w = image.shape
        device = image.device
        dtype = image.dtype

        size_px = self.get_random_size_px(h, w, emoji_size)
        stamp_rgb, stamp_alpha = self._choose_and_prepare_stamp(device, dtype, min(h, w), size_px)
        _, _, hs, ws = stamp_rgb.shape

        if c != 3:
            stamp_rgb = stamp_rgb[:, :1, :, :].expand(-1, c, -1, -1)

        max_x = max(0, w - ws)
        max_y = max(0, h - hs)
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0

        y0, y1 = y, min(h, y + hs)
        x0, x1 = x, min(w, x + ws)
        sy0, sy1 = 0, y1 - y0
        sx0, sx1 = 0, x1 - x0

        if y0 >= y1 or x0 >= x1:
            return image, mask

        img_slice = image[:, :, y0:y1, x0:x1]
        stamp_rgb_slice = stamp_rgb[:, :, sy0:sy1, sx0:sx1].to(device=device, dtype=dtype)
        stamp_alpha_slice = stamp_alpha[:, :, sy0:sy1, sx0:sx1].to(device=device, dtype=dtype)

        stamp_rgb_b = stamp_rgb_slice.expand(b, -1, -1, -1)
        stamp_alpha_b = stamp_alpha_slice.expand(b, -1, -1, -1)

        out_slice = img_slice * (1.0 - stamp_alpha_b) + stamp_rgb_b * stamp_alpha_b

        out = image.clone()
        out[:, :, y0:y1, x0:x1] = out_slice

        if mask is not None:
            mask_out = mask.clone()
            added_pixels = (stamp_alpha_b > 0).to(mask_out.dtype)
            mask_out[:, :, y0:y1, x0:x1] *= (1 - added_pixels)
            mask = mask_out

        return out, mask

    def __repr__(self) -> str:
        return f"InsertEmoji(min_prop={self.min_size}, max_prop={self.max_size}, dir={self.emoji_dir})"


if __name__ == "__main__":
    import torch
    from PIL import Image
    from torchvision.utils import save_image

    from ..data.transforms import default_transform

    transformations = [
        (InsertText, [12, 24, 36, 48]),
        (InsertEmoji, [24, 48, 72, 96]),
        (InsertMemeText, [0.1, 0.2, 0.3]),
    ]

    imgs = [
        Image.open("/private/home/pfz/_images/gauguin_256.png"),
        Image.open("/private/home/pfz/_images/tahiti_256.png"),
    ]
    imgs = torch.stack([default_transform(img) for img in imgs]).to(device="cuda")

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    for transform, strengths in transformations:
        for strength in strengths:
            transform_instance = transform()
            imgs_transformed, mask = transform_instance(imgs, torch.ones_like(imgs[:, :1]), strength)
            save_image(imgs_transformed, os.path.join(output_dir, f"{transform.__name__}_strength_{strength}.png"))
            # if mask is not None:
            #     save_image(mask, os.path.join(output_dir, f"{transform.__name__}_strength_{strength}_mask.png"))
