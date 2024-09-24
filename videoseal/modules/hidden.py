
import torch
from torch import nn


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, normalization):

        super(ConvBNRelu, self).__init__()
        
        if normalization == "batch":
            norm_layer = nn.BatchNorm2d(channels_out)
        elif normalization == "group":
            norm_layer = nn.GroupNorm(4, channels_out)
        else:
            raise NotImplementedError

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            norm_layer,
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)


class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, z_channels, last_tanh=True, normalization="batch"):
        super(HiddenEncoder, self).__init__()
        self.num_bits = num_bits
        layers = [ConvBNRelu(3, z_channels, normalization)]

        for _ in range(num_blocks-1):
            layer = ConvBNRelu(z_channels, z_channels, normalization)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(z_channels + 3 + num_bits, z_channels, normalization)

        self.final_layer = nn.Conv2d(z_channels, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1
        msgs = msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)) # b l h w

        encoded_image = self.conv_bns(imgs)

        concat = torch.cat([msgs, encoded_image, imgs], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w


class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, num_blocks, num_bits, z_channels, normalization="batch", pixelwise=False):
        super(HiddenDecoder, self).__init__()
        self.num_bits = num_bits

        layers = [ConvBNRelu(3, z_channels, normalization)]
        for _ in range(num_blocks):
            layers.append(ConvBNRelu(z_channels, z_channels, normalization))
        self.layers = nn.Sequential(*layers)

        self.pixelwise = pixelwise
        if self.pixelwise:
            self.linear = nn.Conv2d(z_channels, num_bits+1, stride=1, kernel_size=1)
        else:
            self.linear = nn.Linear(z_channels, num_bits+1)

    def forward(self, img_w):
        x = self.layers(img_w) # b d h w
        if not self.pixelwise:  # not pixelwise
            x = x.mean(dim=[-2, -1])  # b d      
        x = self.linear(x) # b l+1 ...
        return x

