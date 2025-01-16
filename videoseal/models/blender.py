import torch
import torch.nn as nn
import torch.nn.functional as F


class Blender(nn.Module):

    AVAILABLE_BLENDING_METHODS = [
        "additive", "multiplicative", "spatial_smoothed", "variance_based"
    ]

    def __init__(self,  scaling_i, scaling_w, method="additive"):
        """
        Initializes the Blender class with a specific blending method and optional post-processing.

        Parameters:
            method (str): The blending method to use. 
            scaling_i (float): Scaling factor for the original image.
            scaling_w (float): Scaling factor for the watermark.
        """
        super(Blender, self).__init__()
        self.method = method
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

        # Map method names to functions
        self.blend_methods = {
            "additive": self.additive_blend,
            "multiplicative": self.multiplicative_blend,
            "spatial_smoothed": self.spatial_smoothed_blend,
            "variance_based": self.variance_based_blend,
        }

        if self.method not in self.blend_methods:
            raise ValueError(f"Unknown blending method: {self.method}")

    def forward(self, imgs, preds_w):
        """
        Blends the original images with the predicted watermarks.
        E.g., if method is additive
            If scaling_i = 0.0 and scaling_w = 1.0, the watermarked image is predicted directly.
            If scaling_i = 1.0 and scaling_w = 0.2, the watermark is additive.
        Parameters:
            imgs (torch.Tensor): The original image batch tensor.
            preds_w (torch.Tensor): The watermark batch tensor.

        Returns:
            torch.Tensor: Blended and attenuated image batch.
        """
        # In case of Y channel only, repeat it to 3 channels (same as embedding only in Y channel)
        if preds_w.shape[1] == 1:
            preds_w = preds_w.repeat(1, 3, 1, 1)

        # Perform blending
        blend_function = self.blend_methods[self.method]
        blended_output = blend_function(imgs, preds_w)

        return blended_output

    def additive_blend(self, imgs, preds_w):
        """
        Adds the watermark to the original images.

        - When preds_w = 0, returns imgs unaltered.
        - Can allow the network to learn watermark strength by adjusting the scaling factors.
        """
        return self.scaling_i * imgs + self.scaling_w * preds_w

    def multiplicative_blend(self, imgs, preds_w):
        """
        Multiplies the watermark with the original images.

        - When preds_w = 0, returns imgs unaltered.
        - Higher scaling_w increases the watermark's visibility proportionally to the image intensity.
        """
        return self.scaling_i * imgs * (1 + self.scaling_w * preds_w)

    def spatial_smoothed_blend(self, imgs, preds_w):
        """
        Spatial attention blend that smooths the watermark.

        - When preds_w = 0, returns imgs unaltered.
        - The attention mask is smoothed to reduce abrupt changes, creating a more uniform blending effect.
        """
        # Create and smooth the attention mask
        preds_w = torch.sigmoid(preds_w)
        attention_mask = F.avg_pool2d(preds_w, kernel_size=5,
                                      stride=1, padding=(5-1)//2)  # Smooth
        return self.scaling_i * imgs * (1 - attention_mask) + self.scaling_w * attention_mask * preds_w

    def variance_based_blend(self, imgs, preds_w):
        """
        Variance-based blend that adjusts blending using global variance of the watermark.

        - When preds_w = 0, returns imgs unaltered.
        - The network might learn to reduce the global variance for low contrast watermarks, 
          leading to a softer blend, or increase it for high contrast watermarks, 
          raising blending strength uniformly across the image to avoid patchiness.
        """
        # Compute global variance of the watermark for consistent blending
        global_var = torch.var(preds_w, dim=(1, 2, 3), keepdim=True)
        # Scale blending strength by the global variance
        blend_strength = torch.sigmoid(global_var * self.scaling_w)
        return self.scaling_i * imgs * (1 - blend_strength) + blend_strength * preds_w
