"""
Run with:
    python -m videoseal.augmentation.neuralcompression
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import compressai
    from compressai.zoo import models as compressai_models
    COMPRESSAI_AVAILABLE = True
except ImportError:
    COMPRESSAI_AVAILABLE = False
    print("CompressAI package not found. Install with pip install compressai")

try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Diffusers package not found. Install with pip install diffusers")

try:
    from taming.models.vqgan import VQModel, GumbelVQ
    from omegaconf import OmegaConf
    TAMING_AVAILABLE = True
except ImportError:
    TAMING_AVAILABLE = False
    print("Taming Transformers not found. Install for VQGAN support")


compression_model_paths = {
    'vqgan-1024': {
        'config': '/checkpoint/pfz/projects/autoencoders/ldm/vqgan_imagenet_f16_1024/configs/model_noloss.yaml',
        'ckpt': '/checkpoint/pfz/projects/autoencoders/ldm/vqgan_imagenet_f16_1024/checkpoints/last.ckpt'
    },
    'vqgan-16384': {
        'config': '/checkpoint/pfz/projects/autoencoders/ldm/vqgan_imagenet_f16_16384/configs/model_noloss.yaml',
        'ckpt': '/checkpoint/pfz/projects/autoencoders/ldm/vqgan_imagenet_f16_16384/checkpoints/last.ckpt'
    }
}


def get_model(model_name, quality):
    if model_name in compressai_models:
        return compressai_models[model_name](quality=quality, pretrained=True)
    else:
        avail_models = list(compressai_models.keys())
        raise ValueError(f"Model {model_name} not found. Available models: {avail_models}")


def get_diffusers_model(model_id):
    """Load a model from the Diffusers library"""
    model = AutoencoderKL.from_pretrained(model_id)
    return model


def load_vqgan_from_config(config_path, ckpt_path, is_gumbel=False):
    """Load a VQGAN model from config and checkpoint paths"""
    config = OmegaConf.load(config_path)
    if is_gumbel:
        # We don't have any GumbelVQ for now.
        model = GumbelVQ(**config.model.params)
    else:
        # Default to VQModel.
        model = VQModel(**config.model.params)
    
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    
    return model.eval()


class NeuralCompression(nn.Module):
    def __init__(self, model_name, quality):
        super(NeuralCompression, self).__init__()
        self.model_name = model_name
        self.quality = quality
        self.model = get_model(model_name, quality)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        if self.model_name not in ['bmshj2018-factorized']:
            # resize to closest multiple of 64
            h, w = image.shape[-2:]
            h = max((h // 64) * 64, 64)
            w = max((w // 64) * 64, 64)
            if image.shape[-2:] != (h, w):
                image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
                if mask is not None:
                    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        x_hat = self.model(image.to('cpu'))['x_hat'].to(image.device)
        return x_hat, mask
    
    def __repr__(self):
        return f"{self.model_name} (q={self.quality})"


class DiffusersCompression(nn.Module):
    """Base class for models from the Diffusers library"""
    def __init__(self, model_id):
        super(DiffusersCompression, self).__init__()
        self.model_id = model_id
        self.model = get_diffusers_model(model_id)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        # Handle input size requirements if any
        h, w = image.shape[-2:]
        original_size = (h, w)
        
        # Some diffusers models require dimensions to be multiples of 16
        if h % 16 != 0 or w % 16 != 0:
            h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
            w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
            image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        
        # VQModel and AutoencoderKL have different API
        if isinstance(self.model, VQModel):
            # For VQModel
            encoded = self.model.encode(image)
            if isinstance(encoded, tuple):
                # Some models return a tuple of (z, indices)
                z = encoded[0]
            else:
                z = encoded
            x_hat = self.model.decode(z)
        else:
            # For AutoencoderKL
            x_hat = self.model.decode(self.model.encode(image).latent_dist.sample()).sample
        
        # Resize back to original if needed
        if original_size != (h, w):
            x_hat = F.interpolate(x_hat, size=original_size, mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=False)
                
        return x_hat, mask
    
    def __repr__(self):
        return f"Diffusers-{self.model_id.split('/')[-1]}"


class TamingVQGANCompression(nn.Module):
    """Base class for VQGAN models from Taming Transformers"""
    def __init__(self, config_path, ckpt_path, is_gumbel=False):
        super(TamingVQGANCompression, self).__init__()
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.is_gumbel = is_gumbel
        self.model = load_vqgan_from_config(config_path, ckpt_path, is_gumbel)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Split checkpoint path by "/" and get part that contains "vqgan".
        self.model_name = next((part for part in self.ckpt_path.split('/') if 'vqgan' in part.lower()), 'vqgan')

    def preprocess(self, x):
        """Preprocess image to VQGAN input format [-1, 1]"""
        return 2.0 * x - 1.0

    def postprocess(self, x):
        """Convert VQGAN output back to [0, 1]"""
        return (x + 1.0) / 2.0

    def forward(self, image: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        # Handle input size requirements for VQGAN (multiple of 16)
        h, w = image.shape[-2:]
        original_size = (h, w)
        
        if h % 16 != 0 or w % 16 != 0:
            h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
            w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
            image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        
        # VQGAN expects input in range [-1, 1]
        image = self.preprocess(image)
        
        # Encode and decode with VQGAN
        z, _, _ = self.model.encode(image)
        x_hat = self.model.decode(z)
        
        # Convert back to [0, 1] range
        x_hat = self.postprocess(x_hat)
        
        # Resize back to original if needed
        if original_size != (h, w):
            x_hat = F.interpolate(x_hat, size=original_size, mode='bilinear', align_corners=False)
            if mask is not None:
                mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=False)
                
        return x_hat, mask
    
    def __repr__(self):
        return f"VQGAN-{self.model_name}"


class StableDiffusionVAE(DiffusersCompression):
    def __init__(self):
        super(StableDiffusionVAE, self).__init__("stabilityai/sd-vae-ft-ema")


class StableDiffusionXLVAE(DiffusersCompression):
    def __init__(self):
        super(StableDiffusionXLVAE, self).__init__("madebyollin/sdxl-vae-fp16-fix")


class BMSHJ2018Hyperprior(NeuralCompression):
    def __init__(self, quality):
        super(BMSHJ2018Hyperprior, self).__init__("bmshj2018-hyperprior", quality)


class BMSHJ2018Factorized(NeuralCompression):
    def __init__(self, quality):
        super(BMSHJ2018Factorized, self).__init__("bmshj2018-factorized", quality)


class MBT2018Mean(NeuralCompression):
    def __init__(self, quality):
        super(MBT2018Mean, self).__init__("mbt2018-mean", quality)


class MBT2018(NeuralCompression):
    def __init__(self, quality):
        super(MBT2018, self).__init__("mbt2018", quality)


class Cheng2020Anchor(NeuralCompression):
    def __init__(self, quality):
        super(Cheng2020Anchor, self).__init__("cheng2020-anchor", quality)


class Cheng2020Attn(NeuralCompression):
    def __init__(self, quality):
        super(Cheng2020Attn, self).__init__("cheng2020-attn", quality)


class VQGAN1024(TamingVQGANCompression):
    """VQGAN model with 1024 codes"""
    def __init__(self):
        config_path = compression_model_paths['vqgan-1024']['config']
        ckpt_path = compression_model_paths['vqgan-1024']['ckpt']
        super(VQGAN1024, self).__init__(config_path, ckpt_path)


class VQGAN16384(TamingVQGANCompression):
    """VQGAN model with 16384 codes"""
    def __init__(self):
        config_path = compression_model_paths['vqgan-16384']['config']
        ckpt_path = compression_model_paths['vqgan-16384']['ckpt']
        super(VQGAN16384, self).__init__(config_path, ckpt_path)


if __name__ == "__main__":
    import os
    import torch
    from PIL import Image
    from torchvision.utils import save_image
    import torchvision.transforms.functional as TF
    from datetime import datetime

    from ..data.transforms import default_transform

    # Define the compression models to test
    compression_models = []
    
    # CompressAI models
    if COMPRESSAI_AVAILABLE:
        compression_models.extend([
            (BMSHJ2018Factorized, [1, 6]),  # Factorized Prior (Ballé et al., 2018)
            (BMSHJ2018Hyperprior, [1, 6]),  # Scale Hyperprior (Ballé et al., 2018)
            (MBT2018Mean, [1, 6]),         # Mean-Scale Hyperprior (Minnen et al., 2018)
            (MBT2018, [1, 6]),             # Joint Autoregressive Hierarchical Priors
            (Cheng2020Anchor, [1, 6]),     # Cheng2020 with anchor points
            (Cheng2020Attn, [1, 6]),       # Cheng2020 with attention (SOTA)
        ])
    
    # Diffusers models
    if DIFFUSERS_AVAILABLE:
        compression_models.extend([
            (StableDiffusionVAE, [None]),    # SD VAE EMA
            (StableDiffusionXLVAE, [None]),  # SDXL VAE
        ])

    # VQGAN models
    if TAMING_AVAILABLE:
        compression_models.extend([
            (VQGAN1024, [None]),   # VQGAN with 1024 codes
            (VQGAN16384, [None]),  # VQGAN with 16384 codes
        ])

    # Define image sizes to test (width, height)
    # Testing a variety of sizes including powers of 2, odd sizes, and non-standard dimensions
    image_sizes = [
        (32, 32),    # Small square, power of 2
        (64, 64),    # Medium square, power of 2
        (96, 96),    # Non-power of 2
        (128, 128),  # Larger square, power of 2
        (224, 224),  # Common CNN input size
        (256, 256),  # Large square, power of 2
        (320, 320),  # Common video size
        (384, 384),  # Common video size
        (512, 512),  # Very large, might cause memory issues
        (256, 512),  # Rectangle, with dimensions as powers of 2
    ]

    # Load test images
    original_imgs = [
        Image.open("/private/home/pfz/_images/gauguin_256.png"),
        Image.open("/private/home/pfz/_images/tahiti_256.png")
    ]

    # Create the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/neural_compression_size_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Testing {len(compression_models)} compression models on {len(image_sizes)} different image sizes...")
    print(f"Output directory: {output_dir}")

    # Model test results
    model_results = {}

    # Test each model
    for model_class, quality_levels in compression_models:
        model_name = model_class.__name__
        
        for quality in quality_levels:
            if quality is None:
                quality_str = ""
                print(f"\nTesting {model_name} on various image sizes:")
            else:
                quality_str = f"_q{quality}"
                print(f"\nTesting {model_name} (quality={quality}) on various image sizes:")
            
            # Load model once per quality level
            try:
                if quality is None:
                    model = model_class()
                else:
                    model = model_class(quality=quality)
                    
                # Initialize result dictionary for this model
                model_key = f"{model_name}{quality_str}"
                model_results[model_key] = {"success": [], "fail": []}
                
            except Exception as e:
                print(f"  Error initializing {model_name}{quality_str}: {e}")
                continue
            
            # Test on each image size
            for width, height in image_sizes:
                size_key = f"{width}x{height}"
                print(f"  Testing size: {size_key}...", end=" ")
                
                try:
                    # Resize the images to the target size
                    resized_imgs = []
                    for img in original_imgs:
                        resized = img.resize((width, height), Image.LANCZOS)
                        resized_imgs.append(default_transform(resized))
                    imgs = torch.stack(resized_imgs)
                    
                    # Process the images
                    with torch.no_grad():
                        reconstructed_imgs, _ = model(imgs, None)
                    
                    # Save the reconstructed images
                    filename = f"{model_name}{quality_str}_{width}x{height}.png"
                    save_image(reconstructed_imgs.clamp(0, 1), os.path.join(output_dir, filename))
                    
                    print(f"Success")
                    model_results[model_key]["success"].append(size_key)
                        
                except Exception as e:
                    print(f"Failed: {e}")
                    model_results[model_key]["fail"].append(size_key)

    # Print summary report
    print("\n\nCOMPRESSION MODEL COMPATIBILITY SUMMARY")
    print("====================================")
    
    print("\nModel performance across image sizes:")
    for model_key, results in model_results.items():
        success_count = len(results["success"])
        fail_count = len(results["fail"])
        total = success_count + fail_count
        success_rate = (success_count / total * 100) if total > 0 else 0
        
        print(f"  {model_key}: {success_count} successful, {fail_count} failed ({success_rate:.1f}% success rate)")
        
        if fail_count > 0:
            print(f"    Failed sizes: {', '.join(results['fail'])}")
    
    # Analyze size compatibility
    size_results = {}
    for size in image_sizes:
        size_key = f"{size[0]}x{size[1]}"
        size_results[size_key] = {"success": 0, "fail": 0}
    
    for model_results in model_results.values():
        for size in model_results["success"]:
            size_results[size]["success"] += 1
        for size in model_results["fail"]:
            size_results[size]["fail"] += 1
    
    print("\nImage size compatibility across models:")
    for size_key, counts in size_results.items():
        total = counts["success"] + counts["fail"]
        success_rate = (counts["success"] / total * 100) if total > 0 else 0
        print(f"  {size_key}: {counts['success']} successful, {counts['fail']} failed ({success_rate:.1f}% success rate)")
    
    print(f"\nAll test outputs saved to {output_dir}")
