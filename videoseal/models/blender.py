import torch
import torch.nn as nn
import torch.nn.functional as F


class Blender(nn.Module):

    AVAILABLE_BLENDING_METHODS = [
        "additive", "multiplicative", "sigmoid_additive", "normalized_additive",
        "sigmoid_multiplicative", "adaptive", "norm_ratio", "spatial_attention",
        "variance_based", "exponential_decay"
    ]

    def __init__(self,  scaling_i, scaling_w, method="additive", clamp=True, attenuation="none"):
        """
        Initializes the Blender class with a specific blending method and optional post-processing.

        Parameters:
            method (str): The blending method to use. Options include:
                - "additive", "multiplicative", "sigmoid_additive", "normalized_additive",
                  "sigmoid_multiplicative", "adaptive", "norm_ratio", "spatial_attention",
                  "variance_based", "exponential_decay"
            clamp (bool): If True, clamps the output values to the range [0, 1].
            attenuation (str): Post-blending attenuation method. Options include:
                - "none": No attenuation
                - "mean": Attenuate based on the mean of the blended image and watermark
        """
        super(Blender, self).__init__()
        self.method = method
        self.clamp = clamp
        self.attenuation = attenuation
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        self.attentuation = attenuation

        # Map method names to functions
        self.blend_methods = {
            "additive": self.additive_blend,
            "multiplicative": self.multiplicative_blend,
            "sigmoid_additive": self.sigmoid_additive_blend,
            "normalized_additive": self.normalized_additive_blend,
            "sigmoid_multiplicative": self.sigmoid_multiplicative_blend,
            "adaptive": self.adaptive_blend,
            "norm_ratio": self.norm_ratio_blend,
            "spatial_attention": self.spatial_attention_blend,
            "variance_based": self.variance_based_blend,
            "exponential_decay": self.exponential_decay_blend
        }

        if self.method not in self.blend_methods:
            raise ValueError(f"Unknown blending method: {self.method}")

    def forward(self, imgs, preds_w):
        """
        Applies the specified blending method to the input tensors and attenuates the result if specified.

        Parameters:
            imgs (torch.Tensor): The original image batch tensor.
            preds_w (torch.Tensor): The watermark batch tensor.

        Returns:
            torch.Tensor: Blended and attenuated image batch.
        """
        # Perform blending
        blend_function = self.blend_methods[self.method]
        blended_output = blend_function(imgs, preds_w)

        # Apply attenuation if specified
        if self.attentuation is not None:
            blended_output = self.attentuation(imgs, preds_w)

        # Clamp output if specified
        if self.clamp:
            blended_output = torch.clamp(blended_output, 0, 1)

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

    def sigmoid_additive_blend(self, imgs, preds_w):
        """
        Adds a scaled sigmoid of the watermark to the original images.

        - When preds_w = 0, returns imgs unaltered.
        - The network may learn to adjust the blending based on watermark visibility through scaling.
        """
        return self.scaling_i * imgs + self.scaling_w * torch.sigmoid(preds_w)

    def normalized_additive_blend(self, imgs, preds_w):
        """
        Adds a normalized watermark to the original images.

        - When preds_w = 0, returns imgs unaltered.
        - Ensures that the watermark does not dominate the original image.
        """
        norm_preds_w = preds_w / \
            (torch.norm(preds_w, p=1, dim=(1, 2, 3), keepdim=True) + 1e-6)
        return self.scaling_i * imgs + self.scaling_w * norm_preds_w

    def sigmoid_multiplicative_blend(self, imgs, preds_w):
        """
        Multiplies the original images by a sigmoid of the watermark.
        """
        return self.scaling_i * imgs * torch.sigmoid(preds_w * self.scaling_w)

    def adaptive_blend(self, imgs, preds_w):
        """
        Blends images adaptively based on watermark intensity.

        - When preds_w = 0, returns imgs unaltered.
        - The network can learn to adaptively blend by adjusting the strength based on watermark content.
        """
        return self.scaling_i * imgs * (1 + torch.abs(preds_w)) + self.scaling_w * preds_w

    def norm_ratio_blend(self, imgs, preds_w):
        """
        Blends using the ratio of image to watermark norms.

        - When preds_w = 0, returns imgs unaltered.
        - This encourages the watermark to fit within the scale of the original image.
        """
        norm_i = torch.norm(imgs, p=2, dim=(1, 2, 3), keepdim=True)
        norm_w = torch.norm(preds_w, p=2, dim=(1, 2, 3), keepdim=True)
        return self.scaling_i * imgs * (norm_w / (norm_i + 1e-6)) + self.scaling_w * preds_w

    def spatial_attention_blend(self, imgs, preds_w):
        """
        Spatial attention blend that emphasizes blending in high watermark regions.

        - When preds_w = 0, returns imgs unaltered.
        - The attention mask is smoothed to reduce abrupt changes, creating a more uniform blending effect.
        """
        # Create and smooth the attention mask
        attention_mask = torch.sigmoid(preds_w * self.scaling_w)
        attention_mask = F.avg_pool2d(
            attention_mask, kernel_size=5, stride=1, padding=(5-1)//2)  # Smooth
        return self.scaling_i * imgs * (1 - attention_mask) + attention_mask * preds_w

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

    def exponential_decay_blend(self, imgs, preds_w):
        """
        Exponential decay blend that reduces watermark influence in high-intensity areas.

        - When preds_w = 0, returns imgs unaltered.
        - This method allows the network to learn how to maintain watermark visibility 
          while reducing its influence in brighter image areas, creating a smoother transition.
        """
        # Exponential decay with a lower threshold for smoothness
        # Threshold at 0.5
        decay_factor = torch.exp(-self.scaling_w * (imgs - 0.5).clamp(min=0))
        return self.scaling_i * imgs * decay_factor + (1 - decay_factor) * preds_w
