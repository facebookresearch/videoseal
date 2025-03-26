import math
import torch
import random
import numpy as np

from videoseal.modules.jnd import JND


class FFTWatermark(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.jnd = JND(in_channels=1, out_channels=3)

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
            torch_img_w = torch.clip(torch_img + (random.random() * 1.5 + 0.5) * torch_wm, 0, 1)
            torch_img_w = self.jnd(torch_img, torch_img_w)
        else:
            # watermark everywhere
            torch_img_w = torch.clip(torch_img + 0.05 * torch_wm, 0, 1)

        return torch_img_w

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
