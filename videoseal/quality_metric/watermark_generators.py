import math
import torch
import random
import numpy as np

from videoseal.modules.jnd import JND


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
