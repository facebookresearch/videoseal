"""
Run with:
    python -m videoseal.augmentation.vqvae
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

try:
    from neuralcompression.models import (
        FactorizedPrior,
        JointAutoregressiveHierarchicalPriors,
        ScaleHyperprior,
        MeanScaleHyperprior
    )
    NEURAL_COMPRESSION_AVAILABLE = True
except ImportError:
    NEURAL_COMPRESSION_AVAILABLE = False
    print("NeuralCompression package not found. Install with pip install neuralcompression")

try:
    from taming.models.vqgan import VQModel
    VQGAN_AVAILABLE = True
except ImportError:
    VQGAN_AVAILABLE = False
    print("VQGAN package not found. Clone from https://github.com/CompVis/taming-transformers")


class VQVAEAugmentation:
    """
    VQVAE-based augmentation that compresses and reconstructs images using various models.
    
    Supported models:
    - NeuralCompression models (FactorizedPrior, ScaleHyperprior, etc.)
    - VQGAN models
    """
    def __init__(
        self, 
        model_type: str = 'mshp',
        pretrained_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        vqgan_config: Optional[Dict] = None,
        quality: int = 5,
    ):
        """
        Initialize the VQVAE augmentation.
        
        Args:
            model_type (str): Type of model to use. Options:
                - 'fp': FactorizedPrior
                - 'shp': ScaleHyperprior
                - 'mshp': MeanScaleHyperprior
                - 'jahp': JointAutoregressiveHierarchicalPriors
                - 'vqgan': VQGAN model
            pretrained_path (str, optional): Path to pretrained weights.
            device (str): Device to use.
            vqgan_config (dict, optional): Config for VQGAN model.
            quality (int): Quality factor (1-10). Lower means more compression.
        """
        self.model_type = model_type
        self.device = device
        self.quality = max(1, min(10, quality))  # Clamp between 1 and 10
        self.compression_factor = 11 - self.quality  # Transform to 1-10 scale
        
        # Load the appropriate model
        self.model = self._load_model(model_type, pretrained_path, vqgan_config)
        if self.model is not None:
            self.model.to(device)
            self.model.eval()
    
    def _load_model(self, model_type: str, pretrained_path: Optional[str], vqgan_config: Optional[Dict]) -> nn.Module:
        """Load the specified compression model."""
        model = None
        
        # Neural Compression models
        if model_type in ['fp', 'shp', 'mshp', 'jahp'] and NEURAL_COMPRESSION_AVAILABLE:
            if model_type == 'fp':
                model = FactorizedPrior(3)
            elif model_type == 'shp':
                model = ScaleHyperprior(3)
            elif model_type == 'mshp':
                model = MeanScaleHyperprior(3)
            elif model_type == 'jahp':
                model = JointAutoregressiveHierarchicalPriors(3)
                
            # Load pretrained weights if provided
            if pretrained_path and os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location=self.device)
                model.load_state_dict(state_dict)
                
        # VQGAN models
        elif model_type == 'vqgan' and VQGAN_AVAILABLE:
            if vqgan_config is None:
                raise ValueError("VQGAN requires a config dictionary")
                
            model = VQModel(**vqgan_config)
            
            # Load pretrained weights if provided
            if pretrained_path and os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location=self.device)
                model.load_state_dict(state_dict['state_dict'] if 'state_dict' in state_dict else state_dict)
        
        else:
            available_types = []
            if NEURAL_COMPRESSION_AVAILABLE:
                available_types.extend(['fp', 'shp', 'mshp', 'jahp'])
            if VQGAN_AVAILABLE:
                available_types.append('vqgan')
                
            if not available_types:
                raise ImportError("Neither NeuralCompression nor VQGAN are available. Please install at least one.")
            else:
                raise ValueError(f"Unknown model type '{model_type}'. Available types: {available_types}")
                
        return model
    
    def __call__(
        self, 
        imgs: torch.Tensor, 
        masks: torch.Tensor, 
        strength: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply VQVAE augmentation to images.
        
        Args:
            imgs (torch.Tensor): Images tensor [B, C, H, W] or video tensor [B, T, C, H, W]
            masks (torch.Tensor): Mask tensor
            strength (float): Strength of the augmentation (0.0 to 1.0)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Augmented images and masks
        """
        if self.model is None:
            return imgs, masks
        
        if strength <= 0:
            return imgs, masks
            
        # Adjust compression based on strength
        effective_compression = self.compression_factor * strength

        # Handle different tensor shapes (image vs video)
        is_video = len(imgs.shape) == 5
        
        with torch.no_grad():
            if is_video:
                B, T, C, H, W = imgs.shape
                # Process each frame
                processed_frames = []
                
                for t in range(T):
                    frames = imgs[:, t].to(self.device)  # [B, C, H, W]
                    processed = self._process_batch(frames, effective_compression)
                    processed_frames.append(processed.cpu())
                    
                augmented_imgs = torch.stack(processed_frames, dim=1)  # [B, T, C, H, W]
            else:
                # Process batch of images
                imgs = imgs.to(self.device)
                augmented_imgs = self._process_batch(imgs, effective_compression).cpu()
                
        return augmented_imgs, masks
    
    def _process_batch(self, imgs: torch.Tensor, compression_factor: float) -> torch.Tensor:
        """Process a batch of images through the model."""
        if self.model_type in ['fp', 'shp', 'mshp', 'jahp'] and NEURAL_COMPRESSION_AVAILABLE:
            # Neural Compression models use different encoding/decoding methods
            bottleneck = self.model.compress(imgs)
            
            # Apply additional quantization based on compression_factor if needed
            if compression_factor > 1:
                for key in bottleneck.keys():
                    if isinstance(bottleneck[key], torch.Tensor):
                        # Quantize more aggressively based on compression_factor
                        scale = 2 ** int(compression_factor - 1)
                        bottleneck[key] = torch.round(bottleneck[key] / scale) * scale
            
            reconstructed = self.model.decompress(bottleneck)
            
        elif self.model_type == 'vqgan' and VQGAN_AVAILABLE:
            # VQGAN encoding/decoding
            encoded, codebook_indices = self.model.encode(imgs)
            
            # Apply additional dropout to encoded features based on compression_factor
            if compression_factor > 1:
                dropout_prob = min(0.9, (compression_factor - 1) / 10)
                mask = torch.bernoulli(torch.ones_like(encoded) * (1 - dropout_prob))
                encoded = encoded * mask
                
            reconstructed = self.model.decode(encoded)
            
        else:
            return imgs
            
        return reconstructed
        
    def __str__(self) -> str:
        """String representation of the augmentation."""
        return f"VQVAEAugmentation(model={self.model_type}, quality={self.quality})"


# Register with augmentation factory
def get_vqvae_augmentation(
    model_type: str = 'mshp',
    pretrained_path: Optional[str] = None,
    quality: int = 5,
    vqgan_config: Optional[Dict] = None,
    **kwargs
) -> VQVAEAugmentation:
    """Factory function to create VQVAE augmentation."""
    return VQVAEAugmentation(
        model_type=model_type,
        pretrained_path=pretrained_path,
        vqgan_config=vqgan_config,
        quality=quality,
        **kwargs
    )


if __name__ == "__main__":
    import os
    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor
    from torchvision.utils import save_image

    from ..data.transforms import default_transform

    # Define the models and their parameter ranges
    compression_models = [
        # Format: (model_type, quality_values)
        ('mshp', [3, 5, 8]),  # Different quality levels for Mean-Scale Hyperprior
    ]
    
    # If VQGAN is available, add it to the test
    if VQGAN_AVAILABLE:
        # Simple VQGAN config for testing
        vqgan_config = {
            "embed_dim": 256,
            "n_embed": 1024,
            "ddconfig": {
                "double_z": False,
                "z_channels": 256,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 1, 2, 2, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [16],
                "dropout": 0.0
            }
        }
        
        # Add VQGAN to test models if a pretrained path is available
        # For testing without a model, we'll just test the API
        compression_models.append(('vqgan', [5, 8], vqgan_config))

    # Load images
    imgs = [
        Image.open("/private/home/pfz/_images/gauguin_256.png"),
        Image.open("/private/home/pfz/_images/tahiti_256.png")
    ]
    imgs = torch.stack([default_transform(img) for img in imgs])

    # Create the output directory
    output_dir = "outputs/vqvae_test"
    os.makedirs(output_dir, exist_ok=True)

    # Test with different models and quality levels
    for model_item in compression_models:
        model_type = model_item[0]
        quality_values = model_item[1]
        
        # Get VQGAN config if available
        vqgan_config = model_item[2] if len(model_item) > 2 else None
        
        # Initialize augmentation for this model type
        print(f"Testing {model_type} model...")
        try:
            aug = VQVAEAugmentation(
                model_type=model_type,
                vqgan_config=vqgan_config,
                quality=5  # Default quality, will be overridden for testing
            )
            
            # Test different quality levels and strengths
            for quality in quality_values:
                aug.quality = quality
                aug.compression_factor = 11 - quality
                
                for strength in [0.3, 0.6, 1.0]:
                    # Apply augmentation
                    try:
                        imgs_transformed, _ = aug(imgs.clone(), None, strength)
                    
                        # Save the transformed images
                        filename = f"{model_type}_quality_{quality}_strength_{strength}.png"
                        save_image(imgs_transformed.clamp(0, 1), os.path.join(output_dir, filename))
                        
                        # Print the path to the saved image
                        print(f"Saved transformed images ({model_type}, quality={quality}, strength={strength}) to:",
                              os.path.join(output_dir, filename))
                    except Exception as e:
                        print(f"Error processing {model_type} with quality {quality}, strength {strength}: {e}")
        
        except Exception as e:
            print(f"Could not initialize {model_type} model: {e}")
            
    print(f"\nAll test outputs saved to {output_dir}")
