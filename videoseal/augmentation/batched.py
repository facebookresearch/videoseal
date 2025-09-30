# parts taken from https://github.com/pytorch/vision/blob/a095de183d3811d79ed0db2715e7a1c3162fa19d/torchvision/transforms/_functional_tensor.py
import math
import ffmpeg
import io
import os
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Optional
from torchvision.transforms.functional import gaussian_blur


def _rand_rotate(min_angle, max_angle, prob_90deg):
    theta = random.random() * (max_angle - min_angle) + min_angle
    if random.random() < prob_90deg:
        theta += random.choice([-90, 90])
    theta = math.radians(theta)
    sin = math.sin(theta)
    cos = math.cos(theta)

    from_ = torch.eye(3, dtype=torch.float32)
    to_ = torch.eye(3, dtype=torch.float32)
    from_[:2, 2] = -0.5
    to_[:2, 2] = 0.5
    return to_ @ torch.tensor([
        [cos, -sin, 0],
        [sin,  cos, 0],
        [  0,    0, 1]
    ], dtype=torch.float32) @ from_


def _rand_resize(min_size, max_size, max_center_deviation):
    size = random.random() * (max_size - min_size) + min_size

    deviation = torch.tensor([
        random.random(), random.random()
    ], dtype=torch.float32).sub_(0.5) * (max_center_deviation / 2)

    center_ = torch.eye(3, dtype=torch.float32)
    center_[:2, 2] = (1 - 1/size) / 2 + deviation

    return center_ @ torch.tensor([
        [1/size, 0, 0],
        [0, 1/size, 0],
        [0, 0, 1]
    ], dtype=torch.float32)


def _rand_rotate_and_resize(min_angle, max_angle, prob_90deg, min_size, max_size, max_center_deviation):
    return _rand_resize(min_size, max_size, max_center_deviation) @ _rand_rotate(min_angle, max_angle, prob_90deg)


def _rand_hflip(probability):
    if random.random() < probability:
        return torch.tensor([
            [-1, 0, 1],
            [ 0, 1, 0],
            [ 0, 0, 1]
        ], dtype=torch.float32)
    return torch.eye(3, dtype=torch.float32)


def _rand_crop(min_size, max_size):
    size_x = random.random() * (max_size - min_size) + min_size
    size_y = random.random() * (max_size - min_size) + min_size
    center_x = random.random() * (1 - size_x)
    center_y = random.random() * (1 - size_y)

    return torch.tensor([
        [size_y, 0, center_y],
        [0, size_x, center_x],
        [0, 0, 1]
    ], dtype=torch.float32)


def _rand_identity():
    return torch.eye(3, dtype=torch.float32)


def _rand_perspective(min_distortion_scale, max_distortion_scale):
    d = random.random() * (max_distortion_scale - min_distortion_scale) + min_distortion_scale
    d *= 0.5  # to match the original code, distortion of 1 == up to 100% of the image

    startpoints = [[0, 0], [1, 0], [1, 1], [0, 1]]
    endpoints = [[    random.random() * d,     random.random() * d],
                 [1 - random.random() * d,     random.random() * d],
                 [1 - random.random() * d, 1 - random.random() * d],
                 [    random.random() * d, 1 - random.random() * d]]

    a_matrix = torch.zeros(8, 8, dtype=torch.float64)
    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    b_matrix = torch.tensor(startpoints, dtype=torch.float64).view(8)
    # do least squares in double precision to prevent numerical issues
    res = torch.linalg.lstsq(a_matrix, b_matrix, driver="gels").solution.to(torch.float32)
    return torch.nn.functional.pad(res, (0, 1), mode='constant', value=1.0).view(3, 3)
    

def _rand_color(pastel_factor = 0.5):
    return torch.tensor([(random.uniform(0, 1.0) + pastel_factor) / (1.0 + pastel_factor) for _ in range(3)], dtype=torch.float32)


class BatchedGeometric(nn.Module):

    def __init__(self, params: Dict[str, Dict], probs: Dict[str, float], mode: str = "bilinear", fill: str = "zeros", random_hflip: bool = True, grid_type: str = "random"):
        super(BatchedGeometric, self).__init__()
        
        if mode not in ["bilinear", "nearest"]:
            raise ValueError(f"mode must be 'bilinear' or 'nearest' but is {mode}")
        self.mode = mode
        if fill not in ["zeros", "random_color", "random_image", "random"]:
            raise ValueError(f"fill must be 'zeros', 'random_color', 'random_image' or 'random' but is {fill}")
        self.fill = fill
        self.random_hflip = random_hflip
        if grid_type not in ["random", "affine", "perspective"]:
            raise ValueError(f"mode must be 'random', 'affine or 'perspective' but is {grid_type}")
        self.grid_type = grid_type

        self.aug_params = params
        self.aug_names = sorted(probs.keys())
        self.aug_probs = torch.tensor([probs[n] for n in self.aug_names], dtype=torch.float32)

        self.funcs = {}
        for k in self.aug_names:
            if f"_rand_{k}" not in globals():
                raise ValueError(f"augmentation {k} has no corresponding function _rand_{k}")
            self.funcs[k] = globals()[f"_rand_{k}"]

        # manually add/overwrite identity function
        self.funcs["identity"] = _rand_identity
        self.aug_params["identity"] = {}

    def _affine_grid(self, mats: torch.Tensor, ow: int, oh: int, dtype: torch.dtype, device: torch.device):
        d = 0.5
        base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
        x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=device)
        base_grid[..., 0].copy_(x_grid)
        y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=device).unsqueeze_(-1)
        base_grid[..., 1].copy_(y_grid)
        base_grid[..., 2].fill_(1)

        base_grid = base_grid.repeat(len(mats), 1, 1, 1)

        theta = mats[:, :2]
        rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * ow, 0.5 * oh], dtype=theta.dtype, device=theta.device)
        output_grid = base_grid.view(-1, oh * ow, 3).bmm(rescaled_theta)
        return output_grid.view(-1, oh, ow, 2)

    def _perspective_grid(self, mats: torch.Tensor, ow: int, oh: int, dtype: torch.dtype, device: torch.device):
        assert tuple(mats.shape[1:]) == (3, 3), f"mats.shape must be (N, 3, 3) but is {tuple(mats.shape)}"

        theta1 = mats[:, :2]
        theta2 = mats[:, 2:].repeat(1, 2, 1)

        d = 0.5
        base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
        x_grid = torch.linspace(d, ow * 1.0 + d - 1.0, steps=ow, device=device).div_(ow)
        base_grid[..., 0].copy_(x_grid)
        y_grid = torch.linspace(d, oh * 1.0 + d - 1.0, steps=oh, device=device).unsqueeze_(-1).div_(oh)
        base_grid[..., 1].copy_(y_grid)
        base_grid[..., 2].fill_(1)

        base_grid = base_grid.repeat(len(mats), 1, 1, 1)

        rescaled_theta1 = theta1.transpose(1, 2) #/ torch.tensor([0.5 * ow, 0.5 * oh], dtype=dtype, device=device)
        output_grid1 = base_grid.view(-1, oh * ow, 3).bmm(rescaled_theta1)
        output_grid2 = base_grid.view(-1, oh * ow, 3).bmm(theta2.transpose(1, 2))

        output_grid = (output_grid1 / output_grid2) * 2 - 1.0
        output_grid = output_grid.view(-1, oh, ow, 2)
        return output_grid

    def _sample_transformations(self, n: int, repeat: int = 1, application_mask: Optional[torch.Tensor] = None):
        mats = torch.empty((n, 3, 3), dtype=torch.float32)

        selected_augs = torch.multinomial(self.aug_probs, n, replacement=True).tolist()
        for i, aug_idx in enumerate(selected_augs):
            k = self.aug_names[aug_idx]
            if application_mask is not None and not application_mask[i].item():
                k = "identity"  # force identity if application_mask is False
            mats[i] = self.funcs[k](**self.aug_params[k])
            if self.random_hflip:
                mats[i] = _rand_hflip(probability=0.5) @ mats[i]

        if repeat > 1:
            mats = mats.repeat_interleave(repeat, dim=0)
        return mats

    def _get_fill(self, shape: tuple, dtype: torch.dtype, device: torch.device, fill_image: Optional[torch.Tensor] = None):
        if self.fill == "zeros":
            return None

        fill = torch.zeros(shape, dtype=dtype, device=device)

        if self.fill == "random_color":
            for i in range(shape[0]):
                if random.random() < 0.5:
                    fill[i] = _rand_color(pastel_factor=0.5).view(3, 1, 1).to(dtype=dtype, device=device)
            return fill

        assert fill_image is not None, "fill_image must be provided if fill is 'random_image' or 'random'"
        assert 4 <= len(fill_image.shape) <=5 and fill_image.shape[1] == 3, \
            f"fill_image.shape must be (N, 3, H, W) or (N, 3, T, H, W) but is {tuple(fill_image.shape)}"
        assert torch.is_floating_point(fill_image), f"fill_image must be a floating point tensor but is {fill_image.dtype}"
        fill_image = fill_image.detach()

        for i in range(shape[0]):
            v = random.random()
            if (self.fill == "random_image" and v < 0.5) or (self.fill == "random" and v < 0.33):
                fill_ = fill_image[random.randint(0, len(fill_image) - 1)]
                if len(fill_image.shape) == 5:
                    # select a random frame from the video
                    fill_ = fill_[:, random.randint(0, fill_.shape[1] - 1)]
                fill[i] = fill_.to(dtype=dtype, device=device)
            elif (self.fill == "random" and v < 0.66):
                fill[i] = _rand_color(pastel_factor=0.5).view(3, 1, 1).to(dtype=dtype, device=device)
            # else keep fill[i] as zero
        return fill

    def forward(self, images: torch.Tensor, application_mask: Optional[torch.Tensor] = None, fill_image: Optional[torch.Tensor] = None, return_mats: bool = False):
        assert 4 <= len(images.shape) <=5 and images.shape[1] == 3, \
            f"images.shape must be (N, 3, H, W) or (N, 3, T, H, W) but is {tuple(images.shape)}"
        assert torch.is_floating_point(images), f"images must be a floating point tensor but is {images.dtype}"

        N, T, H, W = images.shape[0], 1, images.shape[-2], images.shape[-1]
        input_is_video = len(images.shape) == 5
        
        if input_is_video:
            T = images.shape[2]
            images = images.permute(0, 2, 1, 3, 4).reshape(N * T, 3, H, W)

        dtype = images.dtype
        device = images.device

        mats = self._sample_transformations(N, repeat=T, application_mask=application_mask).to(device)
        if self.grid_type == "affine" or (self.grid_type == "random" and random.random() < 0.5):
            output_grid = self._affine_grid(mats, W, H, dtype=dtype, device=device)
        else:
            output_grid = self._perspective_grid(mats, W, H, dtype=dtype, device=device)

        fill = self._get_fill((N, 3, H, W), dtype=dtype, device=device, fill_image=fill_image)
        if fill is not None:
            if T > 1:
                fill = fill.repeat_interleave(T, dim=0)
            mask = torch.ones((N * T, 1, H, W), dtype=dtype, device=device)
            images = torch.cat((images, mask), dim=1)

        images_transformed = torch.nn.functional.grid_sample(
            images, output_grid, mode=self.mode, padding_mode="zeros", align_corners=False)

        if fill is not None:
            mask = images_transformed[:, -1:]
            images_transformed = images_transformed[:, :-1]
            mask = mask.expand_as(images_transformed)

            if self.mode == "nearest":
                mask = mask < 0.5
                images_transformed[mask] = fill[mask]
            else:  # 'bilinear'
                images_transformed = images_transformed * mask + (1.0 - mask) * fill
        
        if input_is_video:
            images_transformed = images_transformed.view(N, T, 3, H, W).permute(0, 2, 1, 3, 4)
            mats = mats.view(N, T, 3, 3)

        if return_mats:
            return images_transformed, mats
        return images_transformed

    def revert_transformation(self, images: torch.Tensor, mats: torch.Tensor):
        assert 4 <= len(images.shape) <=5 and images.shape[1] == 3, \
            f"images.shape must be (N, 3, H, W) or (N, 3, T, H, W) but is {tuple(images.shape)}"
        assert torch.is_floating_point(images), f"images must be a floating point tensor but is {images.dtype}"
        assert 3 <= len(mats.shape) <= 4 and mats.shape[-2:] == (3, 3), \
            f"mats.shape must be (N, 3, 3) or (N, T, 3, 3) but is {tuple(mats.shape)}"
        assert images.shape[0] == mats.shape[0] and (images.shape[2] == mats.shape[1] or len(images.shape) == 4), \
            f"images.shape[0] must match mats.shape[0]"

        N, T, H, W = images.shape[0], 1, images.shape[-2], images.shape[-1]
        input_is_video = len(images.shape) == 5
        
        if input_is_video:
            T = images.shape[2]
            images = images.permute(0, 2, 1, 3, 4).reshape(N * T, 3, H, W)
            mats = mats.reshape(N * T, 3, 3)

        dtype = images.dtype
        device = images.device

        mats_inv = torch.linalg.inv(mats).to(device)
        output_grid = self._perspective_grid(mats_inv, W, H, dtype=dtype, device=device)

        images_transformed = torch.nn.functional.grid_sample(
            images, output_grid, mode=self.mode, padding_mode="zeros", align_corners=False)

        if input_is_video:
            images_transformed = images_transformed.view(N, T, 3, H, W).permute(0, 2, 1, 3, 4)
        return images_transformed


def _rgb_to_grayscale(imgs: torch.Tensor) -> torch.Tensor:
    r, g, b = imgs.unbind(dim=-3)
    output = (0.2989 * r + 0.587 * g + 0.114 * b).to(imgs.dtype)
    output = output.unsqueeze(dim=-3)
    return output


def _rgb2hsv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occurring, so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv2rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


def _rand_hue(rgb, gray, min_factor, max_factor):
    factor = random.random() * (max_factor - min_factor) + min_factor

    img = _rgb2hsv(rgb)
    h, s, v = img.unbind(dim=-3)
    h = (h + factor) % 1.0
    img = torch.stack((h, s, v), dim=-3)
    return _hsv2rgb(img)


def _rand_gaussian_blur(rgb, gray, min_kernel_size, max_kernel_size):
    kernel_size = round(random.random() * (max_kernel_size - min_kernel_size) + min_kernel_size)
    if kernel_size % 2 == 0: kernel_size += 1

    return gaussian_blur(rgb, kernel_size=kernel_size)


class BatchedValuemetric(nn.Module):

    def __init__(self, params: Dict[str, Dict], probs: Dict[str, float]):
        super(BatchedValuemetric, self).__init__()

        self.aug_params = params
        self.aug_names = sorted(probs.keys())
        self.aug_probs = torch.tensor([probs[n] for n in self.aug_names], dtype=torch.float32)
        self.aug_params["identity"] = {}

        self.blend_map = {
            "brightness": lambda rgb, gray, **kwargs: torch.zeros_like(rgb),
            "contrast": lambda rgb, gray, **kwargs: torch.mean(gray, dim=(-3, -2, -1), keepdim=True).expand_as(rgb),
            "gaussian_blur": _rand_gaussian_blur,
            "grayscale": lambda rgb, gray, **kwargs: gray.expand_as(rgb),
            "hue": _rand_hue,
            "identity": lambda rgb, gray, **kwargs: rgb,
            "saturation": lambda rgb, gray, **kwargs: gray.expand_as(rgb),
        }

        for k in self.aug_names:
            if k not in self.blend_map:
                raise ValueError(f"augmentation {k} is not supported, available augmentations are: {', '.join(sorted(self.blend_map.keys()))}")

    def _rand_factor(self, min_factor: float, max_factor: float) -> float:
        return random.random() * (max_factor - min_factor) + min_factor

    def _sample_blends(self, images: torch.Tensor, application_mask: Optional[torch.Tensor] = None):
        assert len(images.shape) == 5 and images.shape[2] == 3, \
            f"images.shape must be (N, T, 3, H, W) but is {tuple(images.shape)}"
        
        N = len(images)
        grays = _rgb_to_grayscale(images)
        target_blends = torch.empty_like(images)
        target_factors = torch.empty(N, dtype=torch.float32)

        selected_augs = torch.multinomial(self.aug_probs, N, replacement=True).tolist()
        for i, aug_idx in enumerate(selected_augs):
            k = self.aug_names[aug_idx]
            if application_mask is not None and not application_mask[i].item():
                k = "identity"  # force identity if application_mask is False
            target_blends[i] = self.blend_map[k](images[i], grays[i], **self.aug_params[k])
            if k in ["gaussian_blur", "grayscale", "hue", "identity"]:
                target_factors[i] = 0.0
            else:
                target_factors[i] = self._rand_factor(
                    self.aug_params[k]["min_factor"], self.aug_params[k]["max_factor"])

        return target_blends, target_factors.to(images.device)

    def _blend(self, images: torch.Tensor, target_blends: torch.Tensor, target_factors: torch.Tensor):
        target_factors = target_factors.view(-1, 1, 1, 1, 1)
        return (target_factors * images + (1.0 - target_factors) * target_blends).clamp(0, 1).to(images.dtype)

    def forward(self, images: torch.Tensor, application_mask: Optional[torch.Tensor] = None, **kwargs):
        assert 4 <= len(images.shape) <=5 and images.shape[1] == 3, \
            f"images.shape must be (N, 3, H, W) or (N, 3, T, H, W) but is {tuple(images.shape)}"
        assert torch.is_floating_point(images), f"images must be a floating point tensor but is {images.dtype}"

        input_is_video = len(images.shape) == 5
        if input_is_video:
            images = images.permute(0, 2, 1, 3, 4)
        else:
            images = images.unsqueeze(dim=1)
        
        target_blends, target_factors = self._sample_blends(images, application_mask=application_mask)
        blended_images = self._blend(images, target_blends, target_factors)

        if input_is_video:
            blended_images = blended_images.permute(0, 2, 1, 3, 4)
        else:
            blended_images = blended_images.squeeze(dim=1)
        return blended_images


class BatchedCompression(nn.Module):

    def __init__(self, params: Dict[str, Dict], probs: Dict[str, float], deterministic: bool = True):
        super(BatchedCompression, self).__init__()
        self.deterministic = deterministic
        
        self.aug_params = params
        self.aug_names = sorted(probs.keys())
        self.aug_probs = torch.tensor([probs[n] for n in self.aug_names], dtype=torch.float32)

        self.codec_map = {
            "h264": {"codec": "h264", "pixel_format": "yuv420p", "key": "crf"},
            "h264rgb": {"codec": "h264", "pixel_format": "rgb24", "key": "crf"},
            "h265": {"codec": "hevc", "pixel_format": "yuv420p", "key": "crf"},
            # "vp9": {"codec": "vp9", "pixel_format": "yuv420p", "key": "crf"},
            "mjpeg": {"codec": "mjpeg", "pixel_format": "rgb24", "key": "quality"},
            "jpeg": {"codec": "jpeg", "pixel_format": None, "key": "quality"},
            "identity": {},
        }

        for k in self.aug_names:
            if k not in self.codec_map:
                raise ValueError(f"compression {k} is not supported, available options are: {', '.join(sorted(self.codec_map.keys()))}")

    def _compress_frames(self, frames: torch.Tensor, codec: str, pixel_format: str, crf: int = None, quality: int = None):
        assert len(frames.shape) == 4 and frames.shape[3] == 3, \
            f"frames.shape must be (N, H, W, 3) but is {tuple(frames.shape)}"
        assert frames.dtype == torch.uint8, f"frames must be a uint8 tensor but is {frames.dtype}"

        assert codec != "mjpeg" or quality is not None, f"mjpeg requires quality argument"
        assert codec == "mjpeg" or crf is not None, f"codec {codec} requires crf argument"
        
        kwargs = {}
        if quality is not None:
            kwargs["q"] = 100 - quality
        if crf is not None:
            kwargs["crf"] = crf

        N, H, W, _ = frames.shape

        ffmpeg_path = "/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg"
        if not os.path.exists(ffmpeg_path):
            ffmpeg_path = "ffmpeg"

        encode_proc = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(W, H))
            .output('pipe:', format='matroska', vcodec=codec, pix_fmt=pixel_format, **kwargs)
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, cmd=ffmpeg_path)
        )
        decode_proc = (
            ffmpeg
            .input('pipe:', format='matroska', vcodec=codec)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

        encoded_frames, err = encode_proc.communicate(input=frames.numpy().tobytes())
        if encode_proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed: {err.decode('utf-8')}")

        decoded_frames, err = decode_proc.communicate(input=encoded_frames)
        if decode_proc.returncode != 0:
            raise RuntimeError(f"ffmpeg decoding failed: {err.decode('utf-8')}")

        frames = torch.frombuffer(decoded_frames, dtype=torch.uint8).view(N, H, W, 3).clone()
        return frames
    
    def _compress_images(self, images: torch.Tensor, codec: str, quality: int):
        assert len(images.shape) == 4 and images.shape[3] == 3, \
            f"frames.shape must be (N, H, W, 3) but is {tuple(images.shape)}"
        assert images.dtype == torch.uint8, f"frames must be a uint8 tensor but is {images.dtype}"
        assert codec == "jpeg", f"codec must be 'jpeg' for image compression but is {codec}"

        compressed_images = torch.empty_like(images)
        for i in range(len(images)):
            pil_image = Image.fromarray(images[i].numpy())
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed_images[i] = torch.from_numpy(np.array(Image.open(buffer)))
        return compressed_images

    def _augment(self, images: torch.Tensor, augmentation: str):
        if augmentation == "identity":
            return images
        
        c = self.codec_map[augmentation]
        codec, pixel_format, key = c["codec"], c["pixel_format"], c["key"]

        min_factor, max_factor = self.aug_params[augmentation][f"min_{key}"], self.aug_params[augmentation][f"max_{key}"]
        factor = random.randint(min_factor, max_factor)
        
        if codec == "jpeg":
            return self._compress_images(images, codec=codec, **{key: factor})
        return self._compress_frames(
            images, codec=codec, pixel_format=pixel_format, **{key: factor})

    def forward(self, images: torch.Tensor, application_mask: Optional[torch.Tensor] = None, **kwargs):
        assert 4 <= len(images.shape) <=5 and images.shape[1] == 3, \
            f"images.shape must be (N, 3, H, W) or (N, 3, T, H, W) but is {tuple(images.shape)}"
        assert torch.is_floating_point(images), f"images must be a floating point tensor but is {images.dtype}"

        input_is_video = len(images.shape) == 5
        if input_is_video:
            images_nograd = images.permute(0, 2, 3, 4, 1).detach()
        else:
            images_nograd = images.permute(0, 2, 3, 1).unsqueeze(dim=1).detach()
        images_nograd = images_nograd.mul(255).round_().to(dtype=torch.uint8, device="cpu")

        selected_indices = torch.arange(len(images_nograd))
        if application_mask is not None:
            selected_indices = selected_indices[application_mask]
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]

        if self.deterministic:
            min_idx = 0
            cum_sum = torch.cumsum(self.aug_probs / self.aug_probs.sum() * len(selected_indices), dim=0)

            for i, aug_name in enumerate(self.aug_names):
                max_idx = round(cum_sum[i].item())
                if max_idx == min_idx:
                    continue

                for i in range(min_idx, max_idx):
                    j = selected_indices[i]
                    images_nograd[j] = self._augment(images_nograd[j], augmentation=aug_name)
                min_idx = max_idx
        else:
            raise NotImplementedError("Non-deterministic compression pass is not implemented yet")
        
        images_nograd = images_nograd.permute(0, 4, 1, 2, 3)
        if not input_is_video:
            images_nograd = images_nograd.squeeze(dim=2)
        
        images_nograd = images_nograd.to(dtype=images.dtype, device=images.device).div_(255.0)
        return images + (images_nograd - images).detach()


class BatchedAugmenter(nn.Module):

    def __init__(self, **kwargs):
        super(BatchedAugmenter, self).__init__()
        self.augmentations = nn.ModuleDict()

        self.force_apply = kwargs.get("force_apply", [])
        assert all(a in ["geometric", "valuemetric", "compression"] for a in self.force_apply), \
            f"force_apply must contain only 'geometric', 'valuemetric' or 'compression' but is {self.force_apply}"
        
        self.num_augs = kwargs.get("num_augs", 1)
        assert len(self.force_apply) <= self.num_augs, \
            f"there is more augmentations in force_apply ({len(self.force_apply)}) than num_augs ({self.num_augs})"

        geometric_params = kwargs.get("geometric_params", None)
        if geometric_params is not None:
            self.augmentations.add_module("geometric", BatchedGeometric(**geometric_params))

        valuemetric_params = kwargs.get("valuemetric_params", None)
        if valuemetric_params is not None:
            self.augmentations.add_module("valuemetric", BatchedValuemetric(**valuemetric_params))

        compression_params = kwargs.get("compression_params", None)
        if compression_params is not None:
            self.augmentations.add_module("compression", BatchedCompression(**compression_params))

        self.probabilities = kwargs.get("probabilities", {})
        assert all(k in self.probabilities for k in self.augmentations.keys()), \
            f"all augmentations must have a probability defined in probabilities but got {self.probabilities.keys()}"
        self.probabilities = {k: v / sum(self.probabilities.values()) for k, v in self.probabilities.items()}
        self.leftover_counts = {k: 0.0 for k in self.probabilities.keys()}

    def forward(self, images: torch.Tensor, **kwargs):
        selected_augs = list(self.augmentations.keys())
        random.shuffle(selected_augs)

        n_force_apply_remaining = len(self.force_apply)
        aug_count = torch.zeros(len(images), dtype=torch.int32)

        for aug_name in selected_augs:
            if aug_name in self.force_apply:
                mask_ = torch.ones(len(images), dtype=torch.int32)
                n_force_apply_remaining -= 1
            else:
                n = self.probabilities[aug_name] * len(images) * self.num_augs + self.leftover_counts[aug_name]
                indices = torch.where(aug_count < (self.num_augs - n_force_apply_remaining))[0]
                indices = indices[torch.randperm(len(indices))][:round(n)]
                self.leftover_counts[aug_name] = n - round(n)

                mask_ = torch.zeros(len(images), dtype=torch.int32)
                mask_[indices] = 1

            images = self.augmentations[aug_name](images, application_mask=mask_.bool(), **kwargs)
            aug_count += mask_
            # print(f"{aug_name:13}{''.join([str(x) for x in mask_.numpy().tolist()])} {self.leftover_counts[aug_name]:6.3f}", flush=True)
        
        # n_errors = (aug_count != self.num_augs).int().sum().item()
        # if n_errors > 0:
        #     print(f"Some images were not augmented enough ({n_errors}/{len(images)})", flush=True)

        # match the API of the original augmenter
        return images
