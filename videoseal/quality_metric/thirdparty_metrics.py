import sys
import torch
from PIL import Image
import torchvision.transforms
import numpy as np
import argparse
import pickle
import timm
import piq

sys.path = ["./MANIQA"] + sys.path
from MANIQA.models.maniqa import MANIQA as MANIQA_model
sys.path = ["./TReS"] + sys.path[1:]
from TReS.models import Net as TReS_model
sys.path = ["./CONTRIQUE"] + sys.path[1:]
from CONTRIQUE.modules.network import get_network as CONTRIQUE_get_network
from CONTRIQUE.modules.CONTRIQUE_model import CONTRIQUE_model
sys.path = ["./AHIQ"] + sys.path[1:]
from AHIQ.options.test_options import TestOptions as AHIQ_TestOptions
from AHIQ.model.deform_regressor import deform_fusion as AHIQ_deform_fusion, Pixel_Prediction as AHIQ_Pixel_Prediction
from AHIQ.utils.util import SaveOutput as AHIQ_SaveOutput
from AHIQ.script.extract_feature import get_resnet_feature as AHIQ_get_resnet_feature, get_vit_feature as AHIQ_get_vit_feature
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
        encoder = CONTRIQUE_get_network('resnet50', pretrained=False)
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


class MetricCLIPIQA:
    # Exploring CLIP for Assessing the Look and Feel of Images (AAAI 2023)
    # https://github.com/IceClear/CLIP-IQA -- here implemented using PyTorch Image Quality (PIQ) package

    def __init__(self, device="cpu"):
        self.metric = piq.CLIPIQA(data_range=1.).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, img: Image):
        x = torch.tensor(np.array(img)).permute(2, 0, 1)[None, ...] / 255.
        score = self.metric(x.to(self.device))
        return score.item()


class MetricPieAPP:
    # PieAPP: Perceptual Image-Error Assessment through Pairwise Preference
    # https://github.com/prashnani/PerceptualImageError -- here implemented using PyTorch Image Quality (PIQ) package

    def __init__(self, device="cpu"):
        self.metric = piq.PieAPP(reduction='none', stride=32).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, img: Image, reference: Image):
        x = torch.tensor(np.array(img)).permute(2, 0, 1)[None, ...] / 255.
        y = torch.tensor(np.array(reference)).permute(2, 0, 1)[None, ...] / 255.
        score = self.metric(x.to(self.device), y.to(self.device))
        return score.item()


class MetricLPIPS:
    # The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (CVPR 2018)
    # https://github.com/richzhang/PerceptualSimilarity -- here implemented using PyTorch Image Quality (PIQ) package

    def __init__(self, device="cpu"):
        self.metric = piq.LPIPS(reduction='none').to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, img: Image, reference: Image):
        x = torch.tensor(np.array(img)).permute(2, 0, 1)[None, ...] / 255.
        y = torch.tensor(np.array(reference)).permute(2, 0, 1)[None, ...] / 255.
        score = self.metric(x.to(self.device), y.to(self.device))
        return score.item()


class MetricDISTS:
    # Image Quality Assessment: Unifying Structure and Texture Similarity.
    # https://github.com/dingkeyan93/DISTS -- here implemented using PyTorch Image Quality (PIQ) package

    def __init__(self, device="cpu"):
        self.metric = piq.DISTS(reduction='none').to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, img: Image, reference: Image):
        x = torch.tensor(np.array(img)).permute(2, 0, 1)[None, ...] / 255.
        y = torch.tensor(np.array(reference)).permute(2, 0, 1)[None, ...] / 255.
        score = self.metric(x.to(self.device), y.to(self.device))
        return score.item()


class MetricPSNR:

    def __init__(self, device="cpu"):
        self.device = device

    @torch.no_grad()
    def __call__(self, img: Image, reference: Image):
        x = torch.tensor(np.array(img)).permute(2, 0, 1)[None, ...] / 255.
        y = torch.tensor(np.array(reference)).permute(2, 0, 1)[None, ...] / 255.
        score = piq.psnr(x.to(self.device), y.to(self.device), data_range=1., reduction='none')
        return score.item()


class MetricSSIM:

    def __init__(self, device="cpu"):
        self.device = device

    @torch.no_grad()
    def __call__(self, img: Image, reference: Image):
        x = torch.tensor(np.array(img)).permute(2, 0, 1)[None, ...] / 255.
        y = torch.tensor(np.array(reference)).permute(2, 0, 1)[None, ...] / 255.
        score = piq.ssim(x.to(self.device), y.to(self.device), data_range=1.)
        return score.item()


class MetricAHIQ:
    # Attention Helps CNN See Better: Hybrid Image Quality Assessment Network (CVPRW 2022)
    # 1st place -- NTIRE2022 Perceptual Image Quality Assessment Challenge Track 1 Full-Reference competition
    # https://github.com/IIGROUP/AHIQ

    def __init__(self, device="cpu"):
        self.opt = AHIQ_TestOptions().parse()
        self.device = device

        self.resnet50 =  timm.create_model('resnet50',pretrained=True).to(self.device)
        if self.opt.patch_size == 8:
            self.vit = timm.create_model('vit_base_patch8_224',pretrained=True).to(self.device)
        else:
            self.vit = timm.create_model('vit_base_patch16_224',pretrained=True).to(self.device)
        self.deform_net = AHIQ_deform_fusion(self.opt).to(self.device)
        self.regressor = AHIQ_Pixel_Prediction().to(self.device)

        # init saveoutput
        self.save_output = AHIQ_SaveOutput()
        hook_handles = []
        for layer in self.resnet50.modules():
            if isinstance(layer, timm.models.resnet.Bottleneck):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        for layer in self.vit.modules():
            if isinstance(layer, timm.models.vision_transformer.Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        checkpoint = torch.load("AHIQ/checkpoints/ahiq_pipal/AHIQ_vit_p8_epoch33.pth", weights_only=False, map_location="cpu")
        self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
        self.deform_net.load_state_dict(checkpoint['deform_net_model_state_dict'])

    @torch.no_grad
    def __call__(self, img: Image, reference: Image):
        img = torch.from_numpy(np.transpose((np.array(img).astype('float32') / 255 - 0.5) * 2, (2, 0, 1))).unsqueeze(0).to(self.device)
        reference = torch.from_numpy(np.transpose((np.array(reference).astype('float32') / 255 - 0.5) * 2, (2, 0, 1))).unsqueeze(0).to(self.device)

        pred = 0
        for _ in range(self.opt.n_ensemble):
            _ , _, h, w = reference.size()
            assert self.opt.n_ensemble > 9
            new_h = new_w = self.opt.crop_size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            r_img = reference[:, :, top:top + new_h, left:left + new_w]
            d_img = img[:, :, top:top + new_h, left:left + new_w]

            self.vit(d_img)
            vit_dis = AHIQ_get_vit_feature(self.save_output)
            self.save_output.outputs.clear()

            self.vit(r_img)
            vit_ref = AHIQ_get_vit_feature(self.save_output)
            self.save_output.outputs.clear()

            B, N, C = vit_ref.shape
            H, W = 28, 28
            assert H * W == N 
            vit_ref = vit_ref.transpose(1, 2).view(B, C, H, W)
            vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)

            self.resnet50(d_img)
            cnn_dis = AHIQ_get_resnet_feature(self.save_output) 
            self.save_output.outputs.clear()
            cnn_dis = self.deform_net(cnn_dis, vit_ref)

            self.resnet50(r_img)
            cnn_ref = AHIQ_get_resnet_feature(self.save_output)
            self.save_output.outputs.clear()
            cnn_ref = self.deform_net(cnn_ref,vit_ref)

            pred += self.regressor(vit_dis, vit_ref, cnn_dis, cnn_ref)

        pred /= self.opt.n_ensemble
        return pred.squeeze().item()


if __name__ == "__main__":
    device = "cuda:0"
    noreference_metrics = {
        "ARNIQA": MetricARNIQA(device),
        "MANIQA": MetricMANIQA(device),
        "TReS": MetricTReS(device, model="live"),
        "CONTRIQUE": MetricCONTRIQUE(device),
        "CLIP-IQA": MetricCLIPIQA(device),
    }
    reference_metrics = {
        "AHIQ": MetricAHIQ(device),
        "PieAPP": MetricPieAPP(device),
        "LPIPS": MetricLPIPS(device),
        "DISTS": MetricDISTS(device),
        "PSNR": MetricPSNR(device),
        "SSIM": MetricSSIM(device),
    }

    img1 = Image.open("./test_image.png")
    img2 = Image.open("./test_image_watermarked.png")
    img3 = Image.open("./test_image_watermarked_videoseal.png")

    print(f"{'Method(NO REFERENCE)':23}{'score(CIN)':>12}{'score(ori)':>12}")
    print("-" * (23 + 12 * 2))
    for name in sorted(noreference_metrics.keys()):
        metric = noreference_metrics[name]
        score1 = metric(img1)
        score2 = metric(img2)
        print(f"{name:23}{score2:12.3f}{score1:12.3f}")
    print("")

    print(f"{'Method(REFERENCE)':23}{'score(CIN)':>12}{'score(VS)':>12}")
    print("-" * (23 + 12 * 2))
    for name in sorted(reference_metrics.keys()):
        metric = reference_metrics[name]
        score1 = metric(img2, img1)
        score2 = metric(img3, img1)
        print(f"{name:23}{score1:12.3f}{score2:12.3f}")
