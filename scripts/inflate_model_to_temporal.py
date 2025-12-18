# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Run with:
    python inflate_model_to_temporal.py --ckpt <path_to_checkpoint> --output <path_to_output_checkpoint>

"""

import argparse
import torch
import sys
sys.path.append("../")

from videoseal.modules.vit import ImageEncoderViT
from videoseal.modules.convnext import ConvNeXtV2


conv3d_missing_keys=[
    'embedder.unet.inc.double_conv.0.conv.weight', 'embedder.unet.inc.double_conv.3.conv.weight', 
    'embedder.unet.inc.res_conv.conv.weight', 'embedder.unet.inc.res_conv.conv.bias', 
    'embedder.unet.downs.0.conv.double_conv.0.conv.weight', 'embedder.unet.downs.0.conv.double_conv.3.conv.weight', 
    'embedder.unet.downs.0.conv.res_conv.conv.weight', 'embedder.unet.downs.0.conv.res_conv.conv.bias', 
    'embedder.unet.downs.1.conv.double_conv.0.conv.weight', 'embedder.unet.downs.1.conv.double_conv.3.conv.weight', 
    'embedder.unet.downs.1.conv.res_conv.conv.weight', 'embedder.unet.downs.1.conv.res_conv.conv.bias', 
    'embedder.unet.downs.2.conv.double_conv.0.conv.weight', 'embedder.unet.downs.2.conv.double_conv.3.conv.weight', 
    'embedder.unet.downs.2.conv.res_conv.conv.weight', 'embedder.unet.downs.2.conv.res_conv.conv.bias', 
    'embedder.unet.bottleneck.model.0.double_conv.0.conv.weight', 'embedder.unet.bottleneck.model.0.double_conv.3.conv.weight', 
    'embedder.unet.bottleneck.model.0.res_conv.conv.weight', 'embedder.unet.bottleneck.model.0.res_conv.conv.bias', 
    'embedder.unet.bottleneck.model.1.double_conv.0.conv.weight', 'embedder.unet.bottleneck.model.1.double_conv.3.conv.weight', 
    'embedder.unet.bottleneck.model.1.res_conv.conv.weight', 'embedder.unet.bottleneck.model.1.res_conv.conv.bias', 
    'embedder.unet.bottleneck.model.2.double_conv.0.conv.weight', 'embedder.unet.bottleneck.model.2.double_conv.3.conv.weight', 
    'embedder.unet.bottleneck.model.2.res_conv.conv.weight', 'embedder.unet.bottleneck.model.2.res_conv.conv.bias', 
    'embedder.unet.bottleneck.model.3.double_conv.0.conv.weight', 'embedder.unet.bottleneck.model.3.double_conv.3.conv.weight', 
    'embedder.unet.bottleneck.model.3.res_conv.conv.weight', 'embedder.unet.bottleneck.model.3.res_conv.conv.bias', 
    'embedder.unet.bottleneck.model.4.double_conv.0.conv.weight', 'embedder.unet.bottleneck.model.4.double_conv.3.conv.weight', 
    'embedder.unet.bottleneck.model.4.res_conv.conv.weight', 'embedder.unet.bottleneck.model.4.res_conv.conv.bias', 
    'embedder.unet.bottleneck.model.5.double_conv.0.conv.weight', 'embedder.unet.bottleneck.model.5.double_conv.3.conv.weight', 
    'embedder.unet.bottleneck.model.5.res_conv.conv.weight', 'embedder.unet.bottleneck.model.5.res_conv.conv.bias', 
    'embedder.unet.bottleneck.model.6.double_conv.0.conv.weight', 'embedder.unet.bottleneck.model.6.double_conv.3.conv.weight', 
    'embedder.unet.bottleneck.model.6.res_conv.conv.weight', 'embedder.unet.bottleneck.model.6.res_conv.conv.bias', 
    'embedder.unet.bottleneck.model.7.double_conv.0.conv.weight', 'embedder.unet.bottleneck.model.7.double_conv.3.conv.weight', 
    'embedder.unet.bottleneck.model.7.res_conv.conv.weight', 'embedder.unet.bottleneck.model.7.res_conv.conv.bias', 
    'embedder.unet.ups.0.conv.double_conv.0.conv.weight', 'embedder.unet.ups.0.conv.double_conv.3.conv.weight', 
    'embedder.unet.ups.0.conv.res_conv.conv.weight', 'embedder.unet.ups.0.conv.res_conv.conv.bias', 
    'embedder.unet.ups.1.conv.double_conv.0.conv.weight', 'embedder.unet.ups.1.conv.double_conv.3.conv.weight', 
    'embedder.unet.ups.1.conv.res_conv.conv.weight', 'embedder.unet.ups.1.conv.res_conv.conv.bias', 
    'embedder.unet.ups.2.conv.double_conv.0.conv.weight', 'embedder.unet.ups.2.conv.double_conv.3.conv.weight', 
    'embedder.unet.ups.2.conv.res_conv.conv.weight', 'embedder.unet.ups.2.conv.res_conv.conv.bias'
]


def inflate_convs_unet_embedder_inplace(state_dict):
    for k in conv3d_missing_keys:
        k2 = k.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")
        # bias is only renamed
        if ".bias" in k:
            state_dict[k] = state_dict[k2]
            del state_dict[k2]
        else:
            s = list(state_dict[k2].shape)
            assert len(s) == 4
            # 1x1 convs are inflated only to 1x1x1 therefore no change to the weight except for "unsqueeze"
            if s[2:] == [1,1]:
                state_dict[k] = state_dict[k2].unsqueeze(2)
                del state_dict[k2]
                continue
            assert s[2:] == [3,3]

            # 3x3 convs are inflated to 3x3x3 convs, initialized by zeros
            nw = state_dict[k2].unsqueeze(2)
            nw = torch.cat([torch.zeros_like(nw), nw, torch.zeros_like(nw)], 2)

            state_dict[k] = nw
            del state_dict[k2]


def add_temporal_attention_into_sam_inplace(state_dict):
    # THIS WORKS ONLY for SAM SMALL!!!
    # instantiate SAM
    sam = ImageEncoderViT(
        img_size=256,
        embed_dim=384,
        out_chans=384,
        depth=12,
        num_heads=6,
        patch_size=16,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=8,
        mlp_ratio=4,
        qkv_bias=True,
        use_rel_pos=True,
        temporal_attention=True,
        max_temporal_length=32,
    )
    # set all outputs of temporal blocs to zero by default
    for b in sam.temp_blocks:
        b.mlp.lin2.bias.data.fill_(0.)
        b.mlp.lin2.weight.data.fill_(0.)

    # get names of parameters that are not in the vanilla SAM
    vars_checkpoint = [k[len("detector.image_encoder."):] for k in state_dict.keys() if k.startswith("detector.image_encoder")]
    new_vars = [k for k, _ in sam.named_parameters() if k not in vars_checkpoint]

    sam_state_dict = sam.state_dict()
    for k in new_vars:
        state_dict[f"detector.image_encoder.{k}"] = sam_state_dict[k]


def add_temporal_attention_into_convnext_inplace(state_dict):
    convnext_ = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], temporal_convs=True, temporal_attention=True)

    # set all outputs of temp blocks to zero
    for l in convnext_.stages.children():
        for ll in l.children():
            ll.temp_block.mlp.lin2.bias.data.fill_(0.)
            ll.temp_block.mlp.lin2.weight.data.fill_(0.)

    for k in list(state_dict.keys()):
        if k.startswith("detector.convnext.downsample_layers"):
            # first conv layer is not inflated
            if k.startswith("detector.convnext.downsample_layers.0"):
                continue

            # rename wrapped conv parameters
            new_k = k.replace(".1.weight", ".1.conv.weight").replace(".1.bias", ".1.conv.bias")
            if k == new_k:
                continue
            state_dict[new_k] = state_dict[k]
            del state_dict[k]

            # add temporal conv parameters initialized as identity
            if "weight" in k:
                t = state_dict[new_k]
                out_ch = t.shape[0]
                ks = 3
                nw = torch.cat([
                    torch.zeros((out_ch, out_ch, ks // 2, 1, 1), dtype=t.dtype, device=t.device),
                    torch.eye(out_ch, dtype=t.dtype, device=t.device).view(out_ch, out_ch, 1, 1, 1),
                    torch.zeros((out_ch, out_ch, ks // 2, 1, 1), dtype=t.dtype, device=t.device)
                ], 2)
                new_var_k = new_k.replace(".conv.", ".temp_conv.")
                state_dict[new_var_k] = nw

    # add all temp blocks into the state dict
    sd = convnext_.state_dict()
    for k in sd.keys():
        if k.startswith("stages") and ".temp_block." in k:
            new_k = f"detector.convnext.{k}"
            state_dict[new_k] = sd[k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder_type", type=str, choices=["unet"], required=True)
    parser.add_argument("--extractor_type", type=str, choices=["sam_small", "convnext_tiny"], required=True)

    parser.add_argument("--ckpt", type=str, help="The model checkpoint.", required=True)
    parser.add_argument("--output", type=str, help="The name of the inflated model checkpoint.", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=True)
    assert "model" in ckpt, "The model checkpoint has an unexpected format"
    
    inflate_convs_unet_embedder_inplace(ckpt["model"])
    if args.extractor_type == "sam_small":
        add_temporal_attention_into_sam_inplace(ckpt["model"])
    elif args.extractor_type == "convnext_tiny":
        add_temporal_attention_into_convnext_inplace(ckpt["model"])

    torch.save(ckpt, args.output)
