# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import random
import numpy as np


def apply_jnd(imgs: torch.Tensor, imgs_w: torch.Tensor, hmaps: torch.Tensor, mode: str, alpha: float = 1.0) -> torch.Tensor:
    """ 
    Apply the JND model to the images.
    Args:
        imgs (torch.Tensor): The original images.
        imgs_w (torch.Tensor): The watermarked images.
        hmaps (torch.Tensor): The JND heatmaps.
        mode (str): The mode of applying the JND model.
            If 'multiply', the JND model is applied by multiplying the heatmaps with the difference between the watermarked and original images.
            If 'clamp', the JND model is applied by clamping the difference between the -jnd and +jnd values.
        alpha (float): The alpha value.
    """
    deltas = alpha * (imgs_w - imgs)
    if mode == 'multiply':
        deltas = hmaps * deltas
    elif mode == 'clamp':
        deltas = torch.clamp(deltas, -hmaps, hmaps)
    return imgs + deltas


class JND(nn.Module):
    """ https://ieeexplore.ieee.org/document/7885108 """
    
    def __init__(self,
            in_channels: int = 1,
            out_channels: int = 3,
            blue: bool = False,
            apply_mode: str = "multiply"
    ) -> None:
        super(JND, self).__init__()

        # setup input and output methods
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blue = blue
        groups = self.in_channels

        # create kernels
        kernel_x = torch.tensor(
            [[-1., 0., 1.], 
            [-2., 0., 2.], 
            [-1., 0., 1.]]
        ).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor(
            [[1., 2., 1.], 
            [0., 0., 0.], 
            [-1., -2., -1.]]
        ).unsqueeze(0).unsqueeze(0)
        kernel_lum = torch.tensor(
            [[1., 1., 1., 1., 1.], 
             [1., 2., 2., 2., 1.], 
             [1., 2., 0., 2., 1.], 
             [1., 2., 2., 2., 1.], 
             [1., 1., 1., 1., 1.]]
        ).unsqueeze(0).unsqueeze(0)

        # Expand kernels for 3 input channels and 3 output channels, apply the same filter to each channel
        kernel_x = kernel_x.repeat(groups, 1, 1, 1)
        kernel_y = kernel_y.repeat(groups, 1, 1, 1)
        kernel_lum = kernel_lum.repeat(groups, 1, 1, 1)

        self.conv_x = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups)
        self.conv_lum = nn.Conv2d(3, 3, kernel_size=(5, 5), padding=2, bias=False, groups=groups)

        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)
        self.conv_lum.weight = nn.Parameter(kernel_lum, requires_grad=False)

        # setup apply mode
        self.apply_mode = apply_mode

    def jnd_la(self, x: torch.Tensor, alpha: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        """ Luminance masking: x must be in [0,255] """
        la = self.conv_lum(x) / 32
        mask_lum = la <= 127
        la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum]/127 + eps))
        la[~mask_lum] = 3/128 * (la[~mask_lum] - 127) + 3
        return alpha * la

    def jnd_cm(self, x: torch.Tensor, beta: float = 0.117, eps: float = 1e-5) -> torch.Tensor:
        """ Contrast masking: x must be in [0,255] """
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        cm = torch.sqrt(grad_x**2 + grad_y**2)
        cm = 16 * cm**2.4 / (cm**2 + 26**2)
        return beta * cm

    # @torch.no_grad()
    def heatmaps(
        self, 
        imgs: torch.Tensor, 
        clc: float = 0.3
    ) -> torch.Tensor:
        """ imgs must be in [0,1] """
        imgs = 255 * imgs
        rgbs = torch.tensor([0.299, 0.587, 0.114])
        if self.in_channels == 1:
            imgs = rgbs[0] * imgs[...,0:1,:,:] + rgbs[1] * imgs[...,1:2,:,:] + rgbs[2] * imgs[...,2:3,:,:]  # luminance: b 1 h w
        la = self.jnd_la(imgs)
        cm = self.jnd_cm(imgs)
        hmaps = torch.clamp_min(la + cm - clc * torch.minimum(la, cm), 0)   # b 1or3 h w
        if self.out_channels == 3 and self.in_channels == 1:
            # rgbs = (1-rgbs).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            hmaps = hmaps.repeat(1, 3, 1, 1)  # b 3 h w
            if self.blue:
                hmaps[:, 0] = hmaps[:, 0] * 0.5
                hmaps[:, 1] = hmaps[:, 1] * 0.5
                hmaps[:, 2] = hmaps[:, 2] * 1.0
            # return  hmaps * rgbs.to(hmaps.device)  # b 3 h w
        elif self.out_channels == 1 and self.in_channels == 3:
            # rgbs = (1-rgbs).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # return torch.sum(
            #     hmaps * rgbs.to(hmaps.device), 
            #     dim=1, keepdim=True
            # )  # b c h w * 1 c -> b 1 h w
            hmaps = torch.sum(hmaps / 3, dim=1, keepdim=True)  # b 1 h w
        return hmaps / 255

    def forward(self, imgs: torch.Tensor, imgs_w: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """ imgs and deltas must be in [0,1] """
        hmaps = self.heatmaps(imgs, clc=0.3)
        imgs_w = apply_jnd(imgs, imgs_w, hmaps, self.apply_mode, alpha)
        return imgs_w


class FFTWatermarkBase(torch.nn.Module):

    def __init__(self, alpha_base, alpha_rand, jnd_alpha_base, jnd_alpha_rand):
        super().__init__()
        self.jnd = JND(in_channels=1, out_channels=3)
        self.alpha_base = alpha_base
        self.alpha_rand = alpha_rand
        self.jnd_alpha_base = jnd_alpha_base
        self.jnd_alpha_rand = jnd_alpha_rand

    def embed(self, imgs: torch.Tensor, **kwargs):
        # imgs.shape == [N, 3, H, W], in range [0, 1]
        imgs_w = torch.cat([self.blend_watermark(img.unsqueeze(0)) for img in imgs], 0)
        return {"imgs_w": imgs_w.mul_(255).round_().div_(255)}

    def blend_watermark(self, torch_img: torch.Tensor):
        # torch_img.shape == [1, 3, H, W], in range [0, 1]
        if random.random() < 0.5:
            # 'white' watermark
            wm = self.generate_random_watermark_fft()
            torch_wm = torch.from_numpy(wm).unsqueeze(0).unsqueeze(0).to(torch_img.device)
        else:
            # RGB watermark
            wm = np.stack([self.generate_random_watermark_fft(), self.generate_random_watermark_fft(), self.generate_random_watermark_fft()], 0)
            torch_wm = torch.from_numpy(wm).unsqueeze(0).to(torch_img.device)

        torch_wm = torch.nn.functional.interpolate(torch_wm, size=torch_img.shape[2:])

        if random.random() < 0.5:
            # attenuated watermark
            torch_img_w = torch.clip(torch_img + (random.random() * self.jnd_alpha_rand + self.jnd_alpha_base) * torch_wm, 0, 1)
            torch_img_w = self.jnd(torch_img, torch_img_w)
        else:
            # watermark everywhere
            torch_img_w = torch.clip(torch_img + (random.random() * self.alpha_rand + self.alpha_base) * torch_wm, 0, 1)

        return torch_img_w


class FFTWatermarkWaves(FFTWatermarkBase):

    def __init__(self):
        super().__init__(alpha_base=0.05, alpha_rand=0, jnd_alpha_base=0.5, jnd_alpha_rand=1.5)

    @staticmethod
    def generate_random_watermark_fft():
        H, W = 512, 512
        fourier_wm = np.zeros((H, W), dtype=np.complex128)
        val1_min, val1_max = 1000000, 10000000
        p1_min, p1_max, p2_max = 0, 60, 200

        getv = lambda: random.randint(val1_min, val1_max)
        # getp = lambda: round(math.pow(random.randint(p1_min, p1_max), 0.8))
        # getq = lambda max_: round(math.pow(random.randint(p1_min, max_), 0.8))
        def getr(max_):
            radius = math.pow(random.randint(p1_min, max_), 0.8)
            angle = random.random() * math.pi / 2
            return round(math.sin(angle) * radius), round(math.cos(angle) * radius)

        max_ = random.randint(p1_max, p2_max)
        # for _ in range(random.randint(2, 50)):
        #     fourier_wm[H//2 - getq(max_), W//2 - getq(max_)] = getv() + getv() * 1j
        for _ in range(random.randint(2, 50)):
            a, b = getr(max_)
            fourier_wm[H//2 - a, W//2 - b] = getv() + getv() * 1j

        wm = np.real(np.fft.ifft2(np.fft.ifftshift(fourier_wm))) / 5
        wm = np.float32(wm.clip(-255, 255) / 255)
        return wm


class FFTWatermarkGaussian(FFTWatermarkBase):

    def __init__(self):
        super().__init__(alpha_base=0.05, alpha_rand=0, jnd_alpha_base=1, jnd_alpha_rand=2)

    @staticmethod
    def generate_random_watermark_fft():
        H, W = 512, 512
        fourier_wm = np.zeros((H, W), dtype=np.complex128)

        X_coords, Y_coords = np.meshgrid(np.arange(H), np.arange(W))
        coords = np.stack([X_coords - W / 2, Y_coords - H / 2], 2).reshape(-1, 2)

        power = 4 - math.sqrt(random.random()) * 3
        sigma = random.random() * 30 + 20
        quad_form = (np.power(np.abs(coords / sigma), power)).sum(1) ** (1 / power)
        pd = np.exp(-quad_form / 2)

        fourier_wm[Y_coords.reshape(-1), X_coords.reshape(-1)] = np.random.random(size=(H, W)).reshape(-1) * pd / pd.max() * 1000000j

        wm = np.real(np.fft.ifft2(np.fft.ifftshift(fourier_wm))) / 5
        wm = np.float32(wm.clip(-255, 255) / 255)
        return wm


class FFTWatermarkLines(FFTWatermarkBase):

    def __init__(self):
        super().__init__(alpha_base=0.1, alpha_rand=0.15, jnd_alpha_base=4, jnd_alpha_rand=4)

    @staticmethod
    def generate_random_watermark_fft():
        def gaussian_pdf(x, mu, sigma):
            return np.exp(-((x - mu) / sigma)**2 / 2) / (sigma * np.sqrt(2 * np.pi))

        H, W = 512, 512
        fourier_wm = np.zeros((H, W), dtype=np.complex128)

        sigma = random.random() * 35 + 5
        sigma1 = random.random() * 30 + 20
        sigma2 = random.random() * 30 + 20
        n_lines1 = random.randint(3, 10)
        n_lines2 = random.randint(3, 10)

        for c in np.round(np.abs(np.random.randn(n_lines1)) * sigma).astype(np.int32):
            fourier_wm[H//2 - c] = fourier_wm[H//2 + c] = (1.5 + np.random.random(size=W)) * gaussian_pdf(c, 0, sigma1)

        for c in np.round(np.abs(np.random.randn(n_lines2)) * sigma).astype(np.int32):
            fourier_wm[:, W//2 - c] = fourier_wm[:, W//2 + c] = (1.5 + np.random.random(size=H)) * gaussian_pdf(c, 0, sigma2)

        fourier_wm = fourier_wm / fourier_wm.max() * 1000000j
        wm = np.real(np.fft.ifft2(np.fft.ifftshift(fourier_wm))) / 5
        wm = np.float32(wm.clip(-255, 255) / 255)
        return wm
