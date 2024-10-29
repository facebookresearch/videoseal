
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ChanRMSNorm, Upsample, Downsample, get_activation, get_normalization


class ResidualMLP(nn.Module):
    def __init__(self, channels):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity
        return F.relu(out)


class UnshuffleDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnshuffleDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ShuffleUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShuffleUp, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.conv(x)
        return self.upsample(x)


class ConvNext7x7Seq(nn.Module):
    def __init__(self, channels):
        super(ConvNext7x7Seq, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.conv(x))


class Encoder(nn.Module):
    def __init__(self, N, M):
        super(Encoder, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = nn.Conv2d(48, N, kernel_size=3, stride=1, padding=1)
        self.residual_mlp1 = ResidualMLP(N)
        self.unshuffle_down1 = UnshuffleDown(N, N)
        self.residual_mlp2 = ResidualMLP(N)
        self.unshuffle_down2 = UnshuffleDown(N, N)
        self.convnext_seq = ConvNext7x7Seq(M)
        self.conv2 = nn.Conv2d(N, M, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        x = self.residual_mlp1(x)
        x = self.unshuffle_down1(x)
        x = self.residual_mlp2(x)
        x = self.unshuffle_down2(x)
        x = self.convnext_seq(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, N, M):
        super(Decoder, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.conv1 = nn.Conv2d(N, 48, kernel_size=3, stride=1, padding=1)
        self.residual_mlp1 = ResidualMLP(N)
        self.shuffle_up1 = ShuffleUp(N, N)
        self.residual_mlp2 = ResidualMLP(N)
        self.shuffle_up2 = ShuffleUp(M, N)
        self.convnext_seq = ConvNext7x7Seq(M)
        self.conv2 = nn.Conv2d(M, M, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv1(x)
        x = self.residual_mlp1(x)
        x = self.shuffle_up1(x)
        x = self.residual_mlp2(x)
        x = self.shuffle_up2(x)
        x = self.convnext_seq(x)
        x = self.conv2(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, N, M):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(N, M)
        self.decoder = Decoder(N, M)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':

    N = 128 
    M = 192 
    model = Autoencoder(N, M)

    # Example Input
    input_image = torch.randn(1, 3, 256, 256)  
    output_image = model(input_image)
    print(output_image.shape)