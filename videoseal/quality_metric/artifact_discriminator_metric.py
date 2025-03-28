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
