
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatternComplexity(nn.Module):
    """ https://ieeexplore.ieee.org/document/7885108 """
    
    def __init__(self, preprocess=lambda x: x):
        super(PatternComplexity, self).__init__()
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
        kernel_x = kernel_x.repeat(3, 1, 1, 1)
        kernel_y = kernel_y.repeat(3, 1, 1, 1)

        self.conv_x = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=3)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=3)

        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)

        self.preprocess = preprocess

    def jnd_cm(self, x, beta=0.117, eps=1e-8):
        """ Contrast masking: x must be in [0,255] """
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        cm = torch.sqrt(grad_x**2 + grad_y**2)
        return beta * cm
    
    def theta(self, x, eps=1e-8):
        """ x must be in [0,255] """
        grad_x = self.conv_x(x) + eps
        grad_y = self.conv_y(x)
        if torch.isnan(grad_x).any():
            raise ValueError("grad_x has nans")
        if torch.isnan(grad_y).any():
            raise ValueError("grad_y has nans")
        thetas = torch.atan2(grad_y, grad_x)
        if torch.isnan(thetas).any():
            raise ValueError("thetas has nans")
        return thetas

    # @torch.no_grad()
    def heatmaps(
        self, 
        imgs: torch.Tensor, 
        clc: float = 0.3, 
        input_method = "multi_channels",
        output_method = "multi_channels"
    ) -> torch.Tensor:
        """ imgs must be in [0,1] after preprocess """
        imgs = self.preprocess(imgs)
        # imgs = 255 * self.preprocess(imgs)
        # rgbs = torch.tensor([0.299, 0.587, 0.114])
        if input_method == 'single_channels':
            # imgs = imgs[...,0:1,:,:] + imgs[...,1:2,:,:] + imgs[...,2:3,:,:]
            imgs = imgs[...,0:1,:,:] + imgs[...,1:2,:,:] + imgs[...,2:3,:,:]
            imgs = imgs.repeat(1, 3, 1, 1)  # hack to make it work with the multi_channels method
        hmaps = self.theta(imgs)
        # hmaps = self.jnd_cm(imgs)
        if output_method == "multi_channels":
            # rgbs = (1-rgbs).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # return  hmaps * rgbs.to(hmaps.device)  # b 3 h w
            return  hmaps
        elif output_method == "single_channels":
            # rgbs = (1-rgbs).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            return torch.sum(
                hmaps, 
                # hmaps * rgbs.to(hmaps.device), 
                dim=1, keepdim=True
            )  # b c h w * 1 c -> b 1 h w

    def forward(self, imgs: torch.Tensor, deltas: torch.Tensor, alpha: float = 1.0, input_method = "multi_channels", output_method = "multi_channels") -> torch.Tensor:
        """ imgs and deltas must be in [0,1] after preprocess """
        hmaps = self.heatmaps(imgs, clc=0.3, input_method=input_method, output_method=output_method)
        return imgs + alpha * hmaps * deltas


class PCLoss(nn.Module):
    def __init__(self, 
        preprocess = lambda x: x,
        loss_type: int = 0
    ):
        super(PCLoss, self).__init__()
        self.loss_type = loss_type
        self.pc = PatternComplexity(preprocess=preprocess)
        # self.msa = lambda x, y: 1 - torch.cos(x - y)
        # self.mse = nn.MSELoss()
        angular_diff = lambda x, y: torch.min((x - y) ** 2, (360 - torch.abs(x - y)) ** 2)
        self.msa = lambda x, y: angular_diff(x, y)
    
    def forward(
        self, 
        imgs: torch.Tensor,
        imgs_w: torch.Tensor,
    ):
        # thetas_o = self.pc.heatmaps(imgs, input_method="single_channels", output_method="single_channels")
        # thetas_delta = self.pc.heatmaps(deltas, input_method="single_channels", output_method="single_channels")
        thetas_o = self.pc.heatmaps(imgs, input_method="single_channels", output_method="single_channels")  # b 1 h w
        deltas = imgs_w - imgs  
        thetas_delta = self.pc.heatmaps(deltas, input_method="single_channels", output_method="single_channels")  # b 1 h w
        if self.loss_type == 0:
            loss = self.msa(thetas_o, thetas_delta).mean(dim=(1,2,3))
        elif self.loss_type == 1:
            loss = (self.msa(thetas_o, thetas_delta) * torch.mean(deltas**2, dim=1, keepdim=True)).mean(dim=(1,2,3))
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")
        if torch.isnan(loss).any():
            raise ValueError("Loss has nans")
        return loss
