
import torch
import torch.nn as nn
import torch.nn.functional as F

class JND(nn.Module):
    """ https://ieeexplore.ieee.org/document/7885108 """
    
    def __init__(self, 
            preprocess = lambda x: x,
            postprocess = lambda x: x,
            in_channels = 1,
            out_channels = 3,
            blue = False
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

        # setup pre and post processing
        self.preprocess = preprocess
        self.postprocess = postprocess

    def jnd_la(self, x, alpha=1.0, eps=1e-5):
        """ Luminance masking: x must be in [0,255] """
        la = self.conv_lum(x) / 32
        mask_lum = la <= 127
        la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum]/127 + eps))
        la[~mask_lum] = 3/128 * (la[~mask_lum] - 127) + 3
        return alpha * la

    def jnd_cm(self, x, beta=0.117, eps=1e-5):
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
        """ imgs must be in [0,1] after preprocess """
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
        """ imgs and deltas must be in [0,1] after preprocess """
        imgs = self.preprocess(imgs)
        imgs_w = self.preprocess(imgs_w)
        hmaps = self.heatmaps(imgs, clc=0.3)
        imgs_w = imgs + alpha * hmaps * (imgs_w - imgs)
        return self.postprocess(imgs_w)


class JNDSimplified(nn.Module):
    """ https://ieeexplore.ieee.org/document/7885108 """
    
    def __init__(self, 
            preprocess = lambda x: x,
            postprocess = lambda x: x,
            in_channels = 1,
            out_channels = 3,
            blue = False
    ) -> None:
        super(JNDSimplified, self).__init__()

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

        # Expand kernels for 3 input channels and 3 output channels, apply the same filter to each channel
        kernel_x = kernel_x.repeat(groups, 1, 1, 1)
        kernel_y = kernel_y.repeat(groups, 1, 1, 1)

        self.conv_x = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups)

        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)

        # setup pre and post processing
        self.preprocess = preprocess
        self.postprocess = postprocess

    # @torch.no_grad()
    def heatmaps(
        self, 
        imgs: torch.Tensor, 
        min_hmap_value: float = 0.5,
        max_hmap_value: float = 1.0,
        max_squared_gradient_value_for_clipping = 64000,
    ) -> torch.Tensor:
        """ imgs must be in [0,1] after preprocess """
        imgs = 255 * imgs
        rgbs = torch.tensor([0.299, 0.587, 0.114])
        if self.in_channels == 1:
            imgs = rgbs[0] * imgs[...,0:1,:,:] + rgbs[1] * imgs[...,1:2,:,:] + rgbs[2] * imgs[...,2:3,:,:]  # luminance: b 1 h w
        grad_x = self.conv_x(imgs)
        grad_y = self.conv_y(imgs)
        cm = grad_x**2 + grad_y**2
        cm = torch.clamp(cm, max=max_squared_gradient_value_for_clipping)
        hmaps = (max_hmap_value - min_hmap_value) * cm / max_squared_gradient_value_for_clipping + min_hmap_value
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
        return hmaps

    def forward(self, imgs: torch.Tensor, imgs_w: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """ imgs and deltas must be in [0,1] after preprocess """
        imgs = self.preprocess(imgs)
        imgs_w = self.preprocess(imgs_w)
        hmaps = self.heatmaps(imgs)
        imgs_w = imgs + alpha * hmaps * (imgs_w - imgs)
        return self.postprocess(imgs_w)


class VarianceBasedJND(nn.Module):
    """ https://fb.workplace.com/groups/333602299016183/permalink/596243969418680/
        https://fb.workplace.com/groups/333602299016183/permalink/628744052835338/ """
    
    def __init__(self, 
            preprocess = lambda x: x,
            postprocess = lambda x: x,
            in_channels = 3,
            out_channels = 1,
            min_heatmap_value = 0.2,
            max_heatmap_value = 1.0,
            max_squared_gradient_value_for_clipping=2500,  # mode == contrast*
            avg_pool_kernel_size=5,  # mode == variance
            max_variance_value_for_clipping=2000,  # mode == variance 
            mode = "contrast"
    ) -> None:
        super(VarianceBasedJND, self).__init__()
        assert mode in ["contrast", "contrast_sqrt", "variance"], "SimplifiedJND.mode must be either 'contrast', 'contrast_sqrt' or 'variance'"

        self.mode = mode
        self.max_squared_gradient_value_for_clipping = max_squared_gradient_value_for_clipping
        self.min_heatmap_value = min_heatmap_value
        self.max_heatmap_value = max_heatmap_value

        self.avg_pool_kernel_size = avg_pool_kernel_size
        self.max_variance_value_for_clipping = max_variance_value_for_clipping

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
        kernel_to_grayscale = torch.tensor(
            [0.299, 0.587, 0.114]
        ).unsqueeze(1).unsqueeze(1).unsqueeze(0)

        self.conv_rgb = nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        self.conv_rgb.weight = nn.Parameter(kernel_to_grayscale, requires_grad=False)
        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)

        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels == 1
        assert in_channels in [1, 3]

        # setup pre and post processing
        self.preprocess = preprocess
        self.postprocess = postprocess

    def heatmaps(self, x):
        assert len(x.shape) == 4 and x.shape[1] == self.in_channels, "shape must be [B, in_channels, H, W]"
        x = x.to(self.conv_x.weight.device)

        if self.in_channels == 3:
            x = self.conv_rgb(x)
        
        if self.mode.startswith("contrast"):
            return self.compute_contrast(x, do_sqrt=self.mode=="contrast_sqrt")
        if self.mode == "variance":
            return self.compute_variance(x)
        
        raise NotImplemented(f"{self.mode} not implemented")

    def compute_contrast(self, x, do_sqrt=False):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        cm = grad_x.mul_(grad_x).add_(grad_y.mul_(grad_y))

        max_ = self.max_squared_gradient_value_for_clipping
        if max_ is None:
            max_ = cm.max()
        if do_sqrt:
            cm.sqrt_()
            max_ = max_ ** 0.5

        cm = cm.clamp(0, max_)
        # Map the raw gradient values between 0 and (max_heatmap_value - min_heatmap_value)
        # Make sure that every pixel obtains at least the minimum heatmap value
        cm.mul_((self.max_heatmap_value - self.min_heatmap_value) / max_).add_(self.min_heatmap_value)
        return cm

    def compute_variance(self, x):
        k = self.avg_pool_kernel_size
        mean = F.avg_pool2d(F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode='reflect'), kernel_size=k, stride=1)

        # Calculate the squared difference from the mean
        squared_diff = mean.sub_(x).square_()
        vm = F.avg_pool2d(F.pad(squared_diff, (k // 2, k // 2, k // 2, k // 2), mode='reflect'), kernel_size=k, stride=1)

        max_ = self.max_variance_value_for_clipping
        if max_ is None:
            max_ = vm.max()

        vm.clamp_(0, max_)
        # Map the raw gradient values between 0 and (max_heatmap_value - min_heatmap_value)
        # Make sure that every pixel obtains at least the minimum heatmap value
        vm.mul_((self.max_heatmap_value - self.min_heatmap_value) / max_).add_(self.min_heatmap_value)
        return vm

    def forward(self, imgs: torch.Tensor, imgs_w: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """ imgs and deltas must be in [0,1] after preprocess """
        imgs = self.preprocess(imgs)
        imgs_w = self.preprocess(imgs_w)
        hmaps = self.heatmaps(imgs * 255)  # the c++ code uses the original [0, 255] range, therefore we do the same

        imgs_w = imgs + alpha * hmaps * (imgs_w - imgs)
        return self.postprocess(imgs_w)
