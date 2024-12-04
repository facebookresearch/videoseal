
import torch
import torch.nn as nn
# from compressai import zoo

class CompressionLoss(nn.Module):
    def __init__(self, model_name):
        # ex: msillm_quality_6  noganms_quality_6
        super(CompressionLoss, self).__init__()
        assert model_name in torch.hub.list("facebookresearch/NeuralCompression")
        model = torch.hub.load("facebookresearch/NeuralCompression", model_name)
        model = model.eval()
        self.model = model
        self.mse = nn.MSELoss()

    def forward(
        self, 
        imgs: torch.Tensor,
        imgs_w: torch.Tensor,
    ) -> torch.Tensor:
        compressed = self.model.decoder(
            self.model.encoder(imgs)
        )
        compressed_w = self.model.decoder(
            self.model.encoder(imgs_w)
        )
        loss = self.mse(compressed, compressed_w)
        return loss

    # override to(device)
    def to(self, device):
        self = super().to(device)
        self.model.update()
        self.model.update_tensor_devices("compress")
        
        