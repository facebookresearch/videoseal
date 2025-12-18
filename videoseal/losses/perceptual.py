import torch
import torch.nn as nn
from lpips import LPIPS


from .watson_fft import ColorWrapper, WatsonDistanceFft
from .watson_vgg import WatsonDistanceVgg
from .dists import DISTS
from .jndloss import JNDLoss
from .focal import FocalFrequencyLoss
from .ssim import SSIM, MSSSIM
from .yuvloss import YUVLoss

loss_weights_paths = {
    "dists": "/path/to/loss_weights/dists_ckpt.pth",
    "watson_vgg": "/path/to/loss_weights/rgb_watson_vgg_trial0.pth",
    "watson_dft": "/path/to/loss_weights/rgb_watson_fft_trial0.pth",
}

def build_loss(loss_name):
    if loss_name == "none":
        return NoneLoss()
    elif loss_name == "lpips":
        return LPIPS(net="vgg").eval()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "yuv":
        return YUVLoss()
    elif loss_name == "focal":
        return FocalFrequencyLoss()
    elif loss_name == "ssim":
        return SSIM()
    elif loss_name == "msssim":
        return MSSSIM()
    elif loss_name == "jnd":
        return JNDLoss(loss_type=0)
    elif loss_name == "jnd2":
        return JNDLoss(loss_type=2)
    elif loss_name == "dists":
        # See https://github.com/dingkeyan93/DISTS/blob/master/DISTS_pytorch for the weights
        return DISTS(loss_weights_paths["dists"]).eval()
    elif loss_name == "watson_vgg":
        # See https://github.com/SteffenCzolbe/PerceptualSimilarity for the weights
        model = WatsonDistanceVgg(reduction="none")
        ckpt_loss = loss_weights_paths["watson_vgg"]
        model.load_state_dict(torch.load(ckpt_loss))
        return model.eval()
    elif loss_name == "watson_dft":
        # See https://github.com/SteffenCzolbe/PerceptualSimilarity for the weights
        model = ColorWrapper(WatsonDistanceFft, (), {"reduction": "none"})
        ckpt_loss = loss_weights_paths["watson_dft"]
        model.load_state_dict(torch.load(ckpt_loss))
        return model.eval()
    else:
        raise ValueError(f"Loss type {loss_name} not supported.")
    

class NoneLoss(nn.Module):
    def forward(self, x, y):
        return torch.zeros(1, requires_grad=True)


class PerceptualLoss(nn.Module):
    def __init__(
        self, 
        percep_loss: str
    ):
        super(PerceptualLoss, self).__init__()

        self.percep_loss = percep_loss
        self.perceptual_loss = self.create_perceptual_loss(percep_loss)

    def create_perceptual_loss(
        self, 
        percep_loss: str
    ):
        """
        Create a perceptual loss function from a string.
        Args:
            percep_loss: (str) The perceptual loss string.
                Example: "lpips", "lpips+mse", "lpips+0.1_mse", ...
        """
        # split the string into the different losses
        parts = percep_loss.split('+')

        # only one loss
        if len(parts) == 1:
            loss = parts[0]
            self.losses = {loss: build_loss(loss)}
            return self.losses[loss]
        
        # several losses
        self.losses = {}
        for part in parts:
            if '_' in part:  # Check if the format is 'weight_loss'
                weight, loss_key = part.split('_')
            else:
                weight, loss_key = 1, part
            self.losses[loss_key] = build_loss(loss_key)
        
        # create the combined loss function
        def combined_loss(x, y):
            total_loss = 0
            for part in parts:
                if '_' in part:  # Check if the format is 'weight_loss'
                    weight, loss_key = part.split('_')
                else:
                    weight, loss_key = 1, part
                weight = float(weight)
                total_loss += weight * self.losses[loss_key](x, y).mean()
            return total_loss
        
        return combined_loss

    def forward(
        self, 
        imgs: torch.Tensor,
        imgs_w: torch.Tensor,
    ) -> torch.Tensor:
        return self.perceptual_loss(imgs, imgs_w)

    def to(self, device, *args, **kwargs):
        """
        Override the to method to move only some of the perceptual loss functions to the device.
        + the losses are not moved by default since they are in a dict.
        """
        super().to(device)
        activated = []
        for loss in self.losses.keys():
            if loss in self.percep_loss:
                activated.append(loss)
        for loss in activated:
            self.losses[loss] = self.losses[loss].to(device)
        return self

    def __repr__(self):
        return f"PerceptualLoss(percep_loss={self.percep_loss})"
