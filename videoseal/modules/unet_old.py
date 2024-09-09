
import torch
from torch import nn


# See https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/003efc4c8819de47ff11b5a0af7ba09aee7f5fc1/models/networks.py

class BottleNeck(nn.Module):
    def __init__(
        self, 
        *, 
        num_blocks: int, 
        dim: int, 
        norm_layer: nn.Module, 
        use_dropout: bool, 
        use_bias: bool
    ) -> None:
        super(BottleNeck, self).__init__()
        model = [self.build_conv_block(
                    dim=dim, 
                    norm_layer=norm_layer, 
                    use_dropout=use_dropout, 
                    use_bias=use_bias
                ) for _ in range(num_blocks)]
        self.model = nn.Sequential(*model)

    def build_conv_block(
        self, 
        *, 
        dim: int, 
        norm_layer: nn.Module, 
        use_dropout: bool, 
        use_bias: bool
    ) -> nn.Sequential:
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.model(x)], 1)  # b 2c h' w'


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self, 
        *, 
        out_channels: int, 
        z_channels: int, 
        in_channels: int = None, 
        submodule: nn.Module = None, 
        outermost: bool = False, 
        norm_layer: nn.Module = nn.BatchNorm2d, 
        use_dropout: bool = False
    ) -> None:
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if in_channels is None:
            in_channels = out_channels
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        downnorm = norm_layer(z_channels)
        upnorm = norm_layer(out_channels)

        if outermost:
            downconv = nn.Conv2d(out_channels, z_channels, kernel_size=7, padding=0)
            down = [nn.ReflectionPad2d(3), downconv, downnorm, downrelu]
            upconv = nn.Conv2d(z_channels * 2, out_channels, kernel_size=7, padding=0)
            up = [uprelu, nn.ReflectionPad2d(3), upconv, nn.Tanh()]
            model = down + [submodule] + up
        else:
            upconv = nn.ConvTranspose2d(z_channels * 2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
            downconv = nn.Conv2d(in_channels, z_channels, kernel_size=3, stride=2, padding=1, bias=use_bias)
            down = [downconv, downnorm, downrelu]
            up = [upconv, upnorm, uprelu]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
    def __init__(
        self, 
        *, 
        in_channels: int, 
        out_channels: int, 
        num_blocks: int, 
        z_channels: int = 64, 
        norm_layer: nn.Module = nn.BatchNorm2d, 
        use_dropout: bool = False
    ) -> None:
        super(Unet, self).__init__()
        unet_block = BottleNeck(
            num_blocks=num_blocks, 
            dim=z_channels * 4, 
            norm_layer=norm_layer, 
            use_dropout=use_dropout, 
            use_bias=False
        )    
        unet_block = UnetSkipConnectionBlock(
            out_channels=z_channels * 2,
            z_channels=z_channels * 4,
            submodule=unet_block,
            norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            out_channels=z_channels,
            z_channels=z_channels * 2,
            submodule=unet_block,
            norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            out_channels=out_channels,
            z_channels=z_channels,
            in_channels=in_channels,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer
        ) # add the outermost layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


class BottleNeckMsg(nn.Module):
    def __init__(
        self, 
        msg_processor: nn.Module,
        num_blocks: int, 
        z_channels: int, 
        norm_layer: nn.Module, 
        use_dropout: bool, 
        use_bias: bool,
        *args, **kwargs
    ) -> None:
        super(BottleNeckMsg, self).__init__()
        self.msg_processor = msg_processor
        in_chans = z_channels + self.msg_processor.hidden_size
        model = []
        for _ in range(num_blocks):
            model += self.build_conv_block(
                        in_chans=in_chans, 
                        out_chans=z_channels,
                        norm_layer=norm_layer, 
                        use_dropout=use_dropout, 
                        use_bias=use_bias
                    )
            in_chans = z_channels
        self.model = nn.Sequential(*model)

    def build_conv_block(
        self, 
        in_chans: int, 
        out_chans: int,
        norm_layer: nn.Module, 
        use_dropout: bool, 
        use_bias: bool
    ) -> nn.Sequential:
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=0, bias=use_bias), norm_layer(in_chans), nn.ReLU(True)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=0, bias=use_bias), norm_layer(out_chans)]
        return nn.Sequential(*conv_block)
        
    def forward(
            self, 
            latents: torch.Tensor,
            msgs: torch.Tensor
        ) -> torch.Tensor:
        latents_w = self.msg_processor(latents, msgs)  # b c+c' h w
        return torch.cat([
            latents,  # b c h w
            self.model(latents_w)  # b c+c' h w -> b c h w
        ], 1)  # b 2c h' w'


class UnetSkipConnectionBlockMsg(nn.Module):
    def __init__(
        self, 
        out_channels: int, 
        z_channels: int, 
        in_channels: int = None, 
        submodule: nn.Module = None, 
        outermost: bool = False, 
        norm_layer: nn.Module = nn.BatchNorm2d, 
        use_dropout: bool = False,
        *args, **kwargs
    ) -> None:
        super(UnetSkipConnectionBlockMsg, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if in_channels is None:
            in_channels = out_channels
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        downnorm = norm_layer(z_channels)
        upnorm = norm_layer(out_channels)

        if outermost:
            downconv = nn.Conv2d(out_channels, z_channels, kernel_size=7, padding=0)
            down = [nn.ReflectionPad2d(3), downconv, downnorm, downrelu]
            upconv = nn.Conv2d(z_channels * 2, out_channels, kernel_size=7, padding=0)
            up = [uprelu, nn.ReflectionPad2d(3), upconv, nn.Tanh()]
        else:
            # upconv = nn.ConvTranspose2d(z_channels * 2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
            upconv = self.build_up_conv_block(z_channels * 2, out_channels)
            downconv = nn.Conv2d(in_channels, z_channels, kernel_size=3, stride=2, padding=1, bias=use_bias)
            down = [downconv, downnorm, downrelu]
            up = [upconv, upnorm, uprelu]
        
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.submodule = submodule

    def forward(
        self, 
        latents: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.up(
            self.submodule(
                self.down(latents), msgs
            )
        )
        if self.outermost:
            return outputs
        else:   # add skip connections
            return torch.cat([latents, outputs], 1)

    def build_up_conv_block(self, chans_in: int, chans_out: int):
        return nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(chans_in, chans_out, kernel_size=3, stride=1, padding=0)
        )



class UnetMsg(nn.Module):
    def __init__(
        self, 
        msg_processor: nn.Module,
        in_channels: int, 
        out_channels: int, 
        num_blocks: int, 
        z_channels: int = 64, 
        norm_layer: nn.Module = nn.BatchNorm2d, 
        use_dropout: bool = False,
        last_tanh: bool = False,
        *args, **kwargs
    ) -> None:
        super(UnetMsg, self).__init__()
        unet_block = BottleNeckMsg(
            msg_processor=msg_processor,
            num_blocks=num_blocks, 
            z_channels=z_channels * 4, 
            norm_layer=norm_layer, 
            use_dropout=use_dropout, 
            use_bias=False
        )    
        unet_block = UnetSkipConnectionBlockMsg(
            out_channels=z_channels * 2,
            z_channels=z_channels * 4,
            submodule=unet_block,
            norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlockMsg(
            out_channels=z_channels,
            z_channels=z_channels * 2,
            submodule=unet_block,
            norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlockMsg(
            out_channels=out_channels,
            z_channels=z_channels,
            in_channels=in_channels,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer
        ) # add the outermost layer
        self.last_tanh = last_tanh

    def forward(
        self, 
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(imgs, msgs)
        if self.last_tanh:
            return torch.tanh(outputs)
        return outputs