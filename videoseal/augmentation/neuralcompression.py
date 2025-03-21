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


def get_model(model_name, quality):
    if model_name in compressai_models:
        return compressai_models[model_name](quality=quality, pretrained=True)
    else:
        avail_models = list(compressai_models.keys())
        raise ValueError(f"Model {model_name} not found. Available models: {avail_models}")


class NeuralCompression(nn.Module):
    def __init__(self, model_name, quality):
        super(NeuralCompression, self).__init__()
        self.model_name = model_name
        self.quality = quality
        self.model = get_model(model_name, quality)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, mask, *args, **kwargs):
        x_hat = self.model(image.to('cpu'))['x_hat'].to(image.device)
        return x_hat, mask
    
    def __repr__(self):
        return f"{self.model_name} (q={self.quality})"

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


if __name__ == "__main__":
    import os
    import torch
    from PIL import Image
    from torchvision.utils import save_image
    import torchvision.transforms.functional as TF
    import csv
    from datetime import datetime

    from ..data.transforms import default_transform

    # Define the compression models and their quality levels to test.
    compression_models = [
        (BMSHJ2018Factorized, [1, 6]),  # Factorized Prior (Ballé et al., 2018)
        (BMSHJ2018Hyperprior, [1, 6]),  # Scale Hyperprior (Ballé et al., 2018)
        (MBT2018Mean, [1, 6]),         # Mean-Scale Hyperprior (Minnen et al., 2018)
        (MBT2018, [1, 6]),             # Joint Autoregressive Hierarchical Priors
        (Cheng2020Anchor, [1, 6]),     # Cheng2020 with anchor points
        (Cheng2020Attn, [1, 6]),       # Cheng2020 with attention (SOTA)
    ]

    # Define image sizes to test (width, height)
    # Testing a variety of sizes including powers of 2, odd sizes, and non-standard dimensions
    image_sizes = [
        (32, 32),    # Small square, power of 2
        (64, 64),    # Medium square, power of 2
        (128, 128),  # Larger square, power of 2
        (256, 256),  # Large square, power of 2
        (96, 96),    # Non-power of 2
        (224, 224),  # Common CNN input size
        (48, 96),    # 1:2 aspect ratio
        (96, 48),    # 2:1 aspect ratio
        (100, 100),  # Non-power of 2
        (101, 101),  # Odd size
        (33, 65),    # Odd size, rectangular
        (16, 16),    # Very small, might cause issues
        (512, 512),  # Very large, might cause memory issues
        (1, 32),     # Extreme aspect ratio
        (32, 1),     # Extreme aspect ratio
        (3, 3),      # Extremely small, likely to cause issues
    ]

    # Load test images.
    original_imgs = [
        Image.open("/private/home/pfz/_images/gauguin_256.png"),
        Image.open("/private/home/pfz/_images/tahiti_256.png")
    ]

    # Create the output directory.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/neural_compression_size_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Create CSV file for results
    csv_file = os.path.join(output_dir, "compression_size_results.csv")
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Quality", "Width", "Height", "Status", "Error"])

    if COMPRESSAI_AVAILABLE:
        print(f"Testing {len(compression_models)} compression models on {len(image_sizes)} different image sizes...")

        # Test each model with different quality levels and image sizes
        for model_class, quality_levels in compression_models:
            model_name = model_class.__name__
            
            for quality in quality_levels:
                print(f"\nTesting {model_name} (quality={quality}) on various image sizes:")
                
                # Load model once per quality level
                try:
                    model = model_class(quality=quality)
                except Exception as e:
                    print(f"  Error initializing {model_name} (quality={quality}): {e}")
                    continue
                
                # Test on each image size
                for width, height in image_sizes:
                    print(f"  Testing size: {width}x{height}...", end=" ")
                    
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
                        filename = f"{model_name}_q{quality}_{width}x{height}.png"
                        save_image(reconstructed_imgs.clamp(0, 1), os.path.join(output_dir, filename))
                        
                        print(f"Success")
                        
                        # Write success to CSV
                        with open(csv_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([model_name, quality, width, height, "Success", ""])
                            
                    except Exception as e:
                        print(f"Failed: {e}")
                        
                        # Write failure to CSV
                        with open(csv_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([model_name, quality, width, height, "Failed", str(e)])
        
        # Generate a summary report
        print("\nGenerating summary report...")
        model_success_counts = {}
        size_success_counts = {}
        
        with open(csv_file, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            
            for row in reader:
                model, quality, width, height, status, _ = row
                model_key = f"{model} (q={quality})"
                size_key = f"{width}x{height}"
                
                if model_key not in model_success_counts:
                    model_success_counts[model_key] = {"success": 0, "fail": 0}
                if size_key not in size_success_counts:
                    size_success_counts[size_key] = {"success": 0, "fail": 0}
                
                if status == "Success":
                    model_success_counts[model_key]["success"] += 1
                    size_success_counts[size_key]["success"] += 1
                else:
                    model_success_counts[model_key]["fail"] += 1
                    size_success_counts[size_key]["fail"] += 1
        
        # Write summary report
        summary_file = os.path.join(output_dir, "summary_report.txt")
        with open(summary_file, 'w') as file:
            file.write("COMPRESSION MODEL COMPATIBILITY SUMMARY\n")
            file.write("====================================\n\n")
            
            file.write("Model performance across image sizes:\n")
            for model_key, counts in model_success_counts.items():
                success_rate = counts["success"] / (counts["success"] + counts["fail"]) * 100
                file.write(f"  {model_key}: {counts['success']} successful, {counts['fail']} failed ")
                file.write(f"({success_rate:.1f}% success rate)\n")
            
            file.write("\nImage size compatibility across models:\n")
            for size_key, counts in size_success_counts.items():
                success_rate = counts["success"] / (counts["success"] + counts["fail"]) * 100
                file.write(f"  {size_key}: {counts['success']} successful, {counts['fail']} failed ")
                file.write(f"({success_rate:.1f}% success rate)\n")
        
        print(f"Summary report generated at {summary_file}")
        
    else:
        print("CompressAI is not available. Please install with: pip install compressai")
    
    print(f"\nAll test outputs saved to {output_dir}")
    print(f"Detailed results saved to {csv_file}")