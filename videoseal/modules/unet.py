"""
Test with:
    python -m videoseal.modules.unet
"""

import einops
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .common import AvgPool3dWrapper, ChanRMSNorm, Upsample, Downsample, get_activation, get_normalization, get_conv_layer

# https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
# https://github.com/milesial/Pytorch-UNet/blob/master/train.py


class ResnetBlock(nn.Module):
    """Conv Norm Act * 2"""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        act_layer: nn.Module, 
        norm_layer: nn.Module, 
        mid_channels: int = None, 
        id_init: bool = False, 
        conv_layer: nn.Module = nn.Conv2d
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_layer(in_channels, mid_channels,
                       kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            act_layer(),
            conv_layer(mid_channels, out_channels,
                       kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer()
        )
        self.res_conv = conv_layer(in_channels, out_channels, kernel_size=1)
        if id_init:
            self._id_init(self.res_conv)

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x) + self.res_conv(x)

    def _id_init(self, m: nn.Module) -> nn.Module:
        """
        Initialize the weights of the residual convolution to be the identity
        """
        if isinstance(m, nn.Conv2d):
            with torch.no_grad():
                in_channels, out_channels, h, w = m.weight.size()
                if in_channels == out_channels:
                    identity_kernel = torch.eye(in_channels).view(in_channels, in_channels, 1, 1)
                    m.weight.copy_(identity_kernel)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            raise NotImplemented("identity-initialized residual convolutions not implemented for conv3d")
        return m


class UBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        act_layer: nn.Module, 
        norm_layer: nn.Module, 
        upsampling_type: str = 'bilinear', 
        id_init: bool = False, 
        conv_layer: nn.Module = nn.Conv2d
    ) -> None:
        super().__init__()
        self.up = Upsample(upsampling_type, in_channels,
                           out_channels, 2, act_layer)
        self.conv = ResnetBlock(
            out_channels, out_channels, act_layer, norm_layer, id_init=id_init, conv_layer=conv_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        return self.conv(x)


class DBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        act_layer: nn.Module, 
        norm_layer: nn.Module, 
        upsampling_type: str = 'bilinear', 
        id_init: bool = False, 
        conv_layer: nn.Module = nn.Conv2d
    ) -> None:
        super().__init__()
        if upsampling_type == 'bilinear':
            self.down = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=3, stride=2, padding=1)
        else:
            self.down = Downsample(in_channels, out_channels, act_layer)
        self.conv = ResnetBlock(
            out_channels, out_channels, act_layer, norm_layer, id_init=id_init, conv_layer=conv_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        return self.conv(x)


class BottleNeck(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        channels_in: int,
        channels_out: int,
        act_layer: nn.Module,
        norm_layer: nn.Module,
        id_init: bool = False,
        conv_layer: nn.Module = nn.Conv2d,
        *args, **kwargs
    ) -> None:
        super(BottleNeck, self).__init__()
        model = []
        for _ in range(num_blocks):
            model += [ResnetBlock(channels_in, channels_out,
                                  act_layer, norm_layer, id_init=id_init, conv_layer=conv_layer)]
            channels_in = channels_out
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)  # b c+c' h w -> b c h w


class UNetMsg(nn.Module):
    def __init__(
        self, 
        msg_processor: nn.Module,
        in_channels: int,
        out_channels: int,
        z_channels: int,
        num_blocks: int,
        activation: str,
        normalization: str,
        z_channels_mults: tuple[int, ...],
        upsampling_type: str = 'bilinear',
        downsampling_type: str = 'bilinear',
        last_tanh: bool = True,
        zero_init: bool = False,
        id_init: bool = False,
        conv_layer: str = "conv2d",
        time_pooling: bool = False,
        time_pooling_kernel_size: int = 1,
        time_pooling_depth: int = 1,
        time_pooling_stride: int = None,
        *args, **kwargs
    ) -> None:
        super(UNetMsg, self).__init__()
        self.msg_processor = msg_processor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.num_blocks = num_blocks
        self.z_channels_mults = z_channels_mults
        self.last_tanh = last_tanh
        self.connect_scale = 2 ** -0.5

        # select layers and activations
        norm_layer: nn.Module = get_normalization(normalization)
        act_layer: nn.Module = get_activation(activation)
        conv_layer_type: nn.Module = get_conv_layer(conv_layer)

        # Calculate the z_channels for each layer based on z_channels_mults
        z_channels_list = [self.z_channels * m for m in self.z_channels_mults]

        # Initial convolution
        self.inc = ResnetBlock(
            in_channels, z_channels_list[0], act_layer, norm_layer, id_init=id_init, conv_layer=conv_layer_type)

        # Downward path
        self.downs = nn.ModuleList()
        for ii in range(len(z_channels_list) - 1):
            self.downs.append(DBlock(
                z_channels_list[ii], z_channels_list[ii + 1], act_layer, norm_layer, downsampling_type, id_init, conv_layer=conv_layer_type))

        # Message mixing and middle blocks
        z_channels_list[-1] = z_channels_list[-1] + self.msg_processor.hidden_size
        self.bottleneck = BottleNeck(
            num_blocks, z_channels_list[-1], z_channels_list[-1], act_layer, norm_layer, id_init=id_init, conv_layer=conv_layer_type)

        # Upward path
        self.ups = nn.ModuleList()
        for ii in reversed(range(len(z_channels_list) - 1)):
            self.ups.append(UBlock(
                2 * z_channels_list[ii + 1], z_channels_list[ii], act_layer, norm_layer, upsampling_type, id_init, conv_layer=conv_layer_type))

        # Final output convolution
        self.outc = nn.Conv2d(z_channels_list[0], out_channels, 1)
        if zero_init:
            self.zero_init_(self.outc)
        
        # time_pooling is a dictionary with keys "depth", "kernel_size", and optional "stride"
        self.time_pooling = time_pooling
        self.time_pooling_depth = time_pooling_depth
        if self.time_pooling:
            time_pooling_stride = time_pooling_kernel_size if time_pooling_stride is None else time_pooling_stride
            self.temporal_pool = AvgPool3dWrapper(
                time_pooling_kernel_size, 
                time_pooling_stride, 
                padding=0, ceil_mode=True, count_include_pad=False,
            )
        else:
            self.temporal_pool = nn.Identity()
            self.temporal_pool.kernel_size = 1

    def forward(
        self,
        imgs: Tensor,
        msgs: Tensor
    ) -> Tensor:
        nb_imgs: int = len(imgs)
        # Initial convolution
        x1 = self.inc(imgs)
        hiddens = [x1]

        # Downward path
        for ii, dblock in enumerate(self.downs):
            if self.time_pooling and ii == self.time_pooling_depth:
                temp_downscale = self.temporal_pool(hiddens[-1])  # b d h w -> b/k d h w
                hiddens.append(dblock(temp_downscale))
            else:
                hiddens.append(dblock(hiddens[-1]))  # b d h w -> b d' h/2 w/2

        # Middle path
        last_hidden = hiddens.pop()
        processed_hidden = self.msg_processor(last_hidden, msgs)  # b c+c' h w
        hiddens.append(processed_hidden)
        x = self.bottleneck(hiddens[-1])

        # Upward path
        for ublock in self.ups:
            # Recover the original number of frames for temporal pooling.
            if len(x) != len(hiddens[-1]):
                x = torch.repeat_interleave(x, repeats=self.temporal_pool.kernel_size, dim=0)  # b/k d h w -> b d h w
                x = x[:nb_imgs]
            
            skip_connection = hiddens.pop()
            x = torch.cat(
                (x, skip_connection * self.connect_scale), 
                dim=1
            )  # b d h w -> b 2d h w
            x = ublock(x)  # b d h w

        # Recover the original number of frames for temporal pooling.
        if len(x) != len(hiddens[-1]):
            x = torch.repeat_interleave(x, repeats=self.temporal_pool.kernel_size, dim=0)  # b/k d h w -> b d h w
            x = x[:nb_imgs]

        # Output layer
        logits = self.outc(x)
        if self.last_tanh:
            logits = torch.tanh(logits)
        return logits

    def use_checkpointing(self) -> None:
        # Apply checkpointing to save memory during training. Not used.
        self.inc = torch.utils.checkpoint(self.inc)
        for ii in range(len(self.downs)):
            self.downs[ii] = torch.utils.checkpoint(self.downs[ii])
        for ii in range(len(self.ups)):
            self.ups[ii] = torch.utils.checkpoint(self.ups[ii])
        self.outc = torch.utils.checkpoint(self.outc)

    def zero_init_(self, m: nn.Module) -> nn.Module:
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        return m

def test_trace_unetmsg() -> bool:
    """Test if UNetMsg can be converted to TorchScript using tracing."""
    import torch
    from videoseal.modules.msg_processor import MsgProcessor
    
    # Create a small UNetMsg model for testing
    nbits: int = 8
    hidden_size: int = 16
    in_channels: int = 3
    out_channels: int = 3
    z_channels: int = 32
    
    # Create message processor and model
    msg_processor: MsgProcessor = MsgProcessor(
        nbits=nbits,
        hidden_size=hidden_size,
        msg_processor_type="binary+concat"
    )
    
    model: UNetMsg = UNetMsg(
        msg_processor=msg_processor,
        in_channels=in_channels,
        out_channels=out_channels,
        z_channels=z_channels,
        num_blocks=2,
        activation="relu",
        normalization="batch",
        z_channels_mults=(1, 2, 4),
    )
    
    # Generate sample inputs
    batch_size: int = 2
    height: int = 64
    width: int = 64
    imgs = torch.randn(batch_size, in_channels, height, width)
    msgs = msg_processor.get_random_msg(batch_size)
    
    # Test tracing
    try:
        model.eval()
        with torch.no_grad():
            original_output = model(imgs, msgs)
            traced_model: torch.jit.ScriptModule = torch.jit.trace(model, (imgs, msgs))
            traced_output = traced_model(imgs, msgs)
        
        max_diff: float = (original_output - traced_output).abs().max().item()
        print(f"Successfully traced UNetMsg! Max output diff: {max_diff:.6e}")
        traced_model.save("unetmsg_traced.pt")
        return True
    except Exception as e:
        print(f"Failed to trace UNetMsg: {e}")
        return False

def test_script_unetmsg() -> bool:
    """Test if UNetMsg can be converted to TorchScript using scripting."""
    import torch
    from videoseal.modules.msg_processor import MsgProcessor
    
    # Create a small UNetMsg model for testing
    nbits: int = 8
    hidden_size: int = 16
    in_channels: int = 3
    out_channels: int = 3
    z_channels: int = 32
    
    # Create message processor and model
    msg_processor = MsgProcessor(
        nbits=nbits,
        hidden_size=hidden_size,
        msg_processor_type="binary+concat"
    )
    
    model = UNetMsg(
        msg_processor=msg_processor,
        in_channels=in_channels,
        out_channels=out_channels,
        z_channels=z_channels,
        num_blocks=2,
        activation="relu",
        normalization="batch",
        z_channels_mults=(1, 2, 4)
    )
    
    try:
        scripted_model = torch.jit.script(model)
        print("Successfully scripted UNetMsg!")
        scripted_model.save("unetmsg_scripted.pt")
        return True
    except Exception as e:
        print(f"Failed to script UNetMsg: {e}")
        return False

if __name__ == "__main__":
    # print("Testing tracing...")
    # test_trace_unetmsg()
    print("\nTesting scripting...")
    test_script_unetmsg()
