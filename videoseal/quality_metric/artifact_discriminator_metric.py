import os
import torch
import omegaconf
import torchvision
from PIL import Image

from videoseal.models import build_extractor
from videoseal.quality_metric.thirdparty_metrics import MetricResult, MetricObjective, MetricType

class MetricArtifactDiscriminator:

    def __init__(self, ckpt_path=None, device="cpu"):
        if ckpt_path is None:
            # ckpt_path = "/checkpoint/soucek/2025_logs/quality_test6_btnll_test_videosealv2_largersize768/_seed=75427/checkpoint.pth"
            ckpt_path = "/checkpoint/soucek/2025_logs/quality_test6_btnll_test_videosealv2_largersize768_artificialfft_waves.gauss.lines/expe/checkpoint024.pth"
        
        state_dict = torch.load(ckpt_path, weights_only=True, map_location="cpu")["model"]
        extractor_params = omegaconf.OmegaConf.load(os.path.join(os.path.dirname(ckpt_path), "configs/extractor.yaml"))["convnext_tiny"]

        self.model = build_extractor("convnext_tiny", extractor_params, img_size=256, nbits=0)
        self.model.load_state_dict(state_dict)
        self.model = self.model.eval().to(device)
        self.device = device

        self.patchify_image = torchvision.transforms.Compose([
            lambda x: x.convert("RGB"),
            torchvision.transforms.Resize((768, 768)),
            torchvision.transforms.ToTensor(),
            lambda x: x.view(3, 3, 256, 3, 256).permute(3, 1, 0, 2, 4).reshape(9, 3, 256, 256),
        ])

    @torch.no_grad
    def __call__(self, img: Image):
        imgs = self.patchify_image(img).to(self.device)
        score = self.model(imgs)
        score = score.mean()
        return MetricResult(score.item(), MetricObjective.MAXIMIZE, MetricType.NO_REFERENCE, "ArtifactDisc")


class ArtifactDiscriminatorLoss(torch.nn.Module):

    def __init__(self, ckpt_path=None, frozen=True):
        if ckpt_path is None or ckpt_path.lower() == "none":
            ckpt_path = "/checkpoint/soucek/2025_logs/quality_test6_btnll_test_videosealv2_largersize768_artificialfft_waves.gauss.lines/expe/checkpoint.pth"
        
        state_dict = torch.load(ckpt_path, weights_only=True, map_location="cpu")["model"]
        extractor_params = omegaconf.OmegaConf.load(os.path.join(os.path.dirname(ckpt_path), "configs/extractor.yaml"))["convnext_tiny"]

        self.model = build_extractor("convnext_tiny", extractor_params, img_size=256, nbits=0)
        self.model.load_state_dict(state_dict)
        if frozen:
            self.model = self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, imgs_w: torch.Tensor, imgs_ori=None) -> torch.Tensor:
        # if `imgs_ori`` is None, the loss is computed only with respect to the watermarked images, maximizing the discriminator score.
        # if `imgs_ori` is provided, it computes the ranking loss (the loss used to train this model).
        assert imgs_w.dim() == 4 and imgs_w.shape[1] == 3 and imgs_w.dtype == torch.float32, "Input tensor must be 4D and RGB (B, 3, H, W) with values in [0, 1]." 

        if imgs_ori is not None:
            wm_logits = self.model(imgs_w)
            real_logits = self.model(imgs_ori)
            loss = F.binary_cross_entropy_with_logits(real_logits - wm_logits, torch.ones_like(real_logits)).mean()
        else:
            loss = -self.model(imgs_w).mean()
        return loss
