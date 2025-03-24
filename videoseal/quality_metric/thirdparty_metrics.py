import sys
import torch
from PIL import Image
import torchvision.transforms
import numpy as np
import argparse
import pickle

sys.path = ["./MANIQA"] + sys.path
from MANIQA.models.maniqa import MANIQA as MANIQA_model
sys.path = ["./TReS"] + sys.path[1:]
from TReS.models import Net as TReS_model
sys.path = ["./CONTRIQUE"] + sys.path[1:]
from CONTRIQUE.modules.network import get_network
from CONTRIQUE.modules.CONTRIQUE_model import CONTRIQUE_model
sys.path = sys.path[1:]


class MetricARNIQA:
    # ARNIQA (WACV 2024 Oral) --- Learning Distortion Manifold for Image Quality Assessment
    # https://github.com/miccunifi/ARNIQA

    def __init__(self, device="cpu"):
        # Load the model
        self.model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA", regressor_dataset="kadid10k")
        self.model.eval().to(device)

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = device

    @torch.no_grad
    def __call__(self, img: Image):
        img_ds = torchvision.transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

        # Get the center and corners crops
        img = MetricARNIQA.center_corners_crop(img, crop_size=224)
        img_ds = MetricARNIQA.center_corners_crop(img_ds, crop_size=224)

        img = [torchvision.transforms.ToTensor()(crop) for crop in img]
        img = torch.stack(img, dim=0)
        img = self.normalize(img).to(self.device)
        img_ds = [torchvision.transforms.ToTensor()(crop) for crop in img_ds]
        img_ds = torch.stack(img_ds, dim=0)
        img_ds = self.normalize(img_ds).to(self.device)

        score = self.model(img, img_ds, return_embedding=False, scale_score=True)
        score = score.mean()
        return score.item()

    @staticmethod
    def center_corners_crop(img: Image, crop_size: int = 224):
        """
        Return the center crop and the four corners of the image.

        Args:
            img (PIL.Image): image to crop
            crop_size (int): size of each crop

        Returns:
            crops (List[PIL.Image]): list of the five crops
        """
        width, height = img.size

        # Calculate the coordinates for the center crop and the four corners
        cx = width // 2
        cy = height // 2
        crops = [
            torchvision.transforms.functional.crop(img, cy - crop_size // 2, cx - crop_size // 2, crop_size, crop_size),  # Center
            torchvision.transforms.functional.crop(img, 0, 0, crop_size, crop_size),  # Top-left corner
            torchvision.transforms.functional.crop(img, height - crop_size, 0, crop_size, crop_size),  # Bottom-left corner
            torchvision.transforms.functional.crop(img, 0, width - crop_size, crop_size, crop_size),  # Top-right corner
            torchvision.transforms.functional.crop(img, height - crop_size, width - crop_size, crop_size, crop_size)  # Bottom-right corner
        ]

        return crops


class MANIQA_Normalize:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img_org']
        d_name = sample['d_name']

        d_img = (d_img - self.mean) / self.var

        sample = {'d_img_org': d_img, 'd_name': d_name}
        return sample


class MANIQA_ToTensor:

    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        d_name = sample['d_name']
        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        sample = {
            'd_img_org': d_img,
            'd_name': d_name
        }
        return sample


class MANIQA_Image:

    def __init__(self, img: Image, transform, num_crops=20):
        self.img_name = "N/A"
        self.img = np.array(img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))
        self.transform = transform

        c, h, w = self.img.shape
        new_h = 224
        new_w = 224

        self.img_patches = []
        for i in range(num_crops):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)
        
        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample


class MetricMANIQA:
    # [CVPRW 2022] MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment
    # 1st place -- NTIRE2022 Perceptual Image Quality Assessment Challenge Track 2 No-Reference competition
    # https://github.com/IIGROUP/MANIQA

    def __init__(self, device="cpu"):
        self.model = MANIQA_model(embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8, img_size=224,
                            window_size=4, depths=[2, 2], num_heads=[4, 4], num_tab=2, scale=0.8)
        self.model.load_state_dict(torch.load("./MANIQA/ckpt_koniq10k.pt", weights_only=True, map_location="cpu"), strict=True)
        self.model = self.model.eval().to(device)
        self.device = device

    @torch.no_grad
    def __call__(self, img: Image):
        img = MANIQA_Image(img, transform=torchvision.transforms.Compose([MANIQA_Normalize(0.5, 0.5), MANIQA_ToTensor()]), num_crops=20)
        
        avg_score = 0
        for i in range(20):
            patch_sample = img.get_patch(i)
            patch = patch_sample['d_img_org'].to(self.device)
            patch = patch.unsqueeze(0)
            score = self.model(patch)
            avg_score += score
        score = avg_score / 20

        return score.item()


class MetricTReS:
    # No-Reference Image Quality Assessment via Transformers, Relative Ranking, and Self-Consistency (WACV 2022)
    # https://github.com/isalirezag/TReS

    def __init__(self, device="cpu", model="live"):
        config = argparse.Namespace()
        config.network = 'resnet50'
        config.nheadt = 16
        config.num_encoder_layerst = 2
        config.dim_feedforwardt = 64

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        assert model in ["live", "kadid10k", "tid2013"], "unknown model type"

        self.model = TReS_model(config, device).to(device)
        self.model.load_state_dict(torch.load(f'TReS/bestmodel_1_2021-{model}.zip', weights_only=True, map_location="cpu"))
        self.model.eval()
        self.device = device

    @torch.no_grad
    def __call__(self, img: Image):
        img = self.transforms(img).to(self.device).unsqueeze(0)
        img = torch.as_tensor(img)
        pred, _ = self.model(img)

        return pred.item()


class MetricCONTRIQUE:

    def __init__(self, device="cpu"):
        encoder = get_network('resnet50', pretrained=False)
        self.model = CONTRIQUE_model(None, encoder, 2048).eval()
        self.model.load_state_dict(torch.load("CONTRIQUE/models/CONTRIQUE_checkpoint25.tar", weights_only=True, map_location="cpu"))
        self.model = self.model.to(device)
        self.regressor = pickle.load(open("CONTRIQUE/models/CLIVE.save", 'rb'))
        self.device = device

    @torch.no_grad
    def __call__(self, img: Image):
        sz = img.size
        img2 = img.resize((sz[0] // 2, sz[1] // 2))

        img = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        img2 = torchvision.transforms.ToTensor()(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _,_, _, _, model_feat, model_feat_2, _, _ = self.model(img, img2)
        feat = np.hstack((model_feat.detach().cpu().numpy(), model_feat_2.detach().cpu().numpy()))

        score = self.regressor.predict(feat)[0]
        return float(score)


if __name__ == "__main__":
    device = "cuda:0"
    metrics = {
        "ARNIQA": MetricARNIQA(device),
        "MANIQA": MetricMANIQA(device),
        "TReS": MetricTReS(device, model="live"),
        "CONTRIQUE": MetricCONTRIQUE(device)
    }

    img1 = Image.open("./test_image.png")
    img2 = Image.open("./test_image_watermarked.png")

    print(f"{'Method':20}{'score (ori)':>12}{'score (wm)':>12}")
    print("-" * (20 + 12 * 2))
    for name in sorted(metrics.keys()):
        metric = metrics[name]
        score1 = metric(img1)
        score2 = metric(img2)

        print(type(score1))

        print(f"{name:20}{score1:12.3f}{score2:12.3f}")
