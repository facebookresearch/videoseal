# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# adapted from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/losses/contperceptual.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.discriminator import NLayerDiscriminator
from ..utils.optim import freeze_grads
from .perceptual import PerceptualLoss

def hinge_d_loss(logits_real, logits_fake):
    """
    https://paperswithcode.com/method/gan-hinge-loss
    """
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    """
    Adopt weight if global step is less than threshold
    """
    if global_step < threshold:
        weight = value
    return weight

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class VideosealLoss(nn.Module):
    def __init__(self,
                 balanced=True, total_norm=0.0,
                 disc_weight=1.0, percep_weight=1.0, detect_weight=1.0, decode_weight=0.0,
                 disc_start=0, disc_num_layers=3, disc_in_channels=3, disc_loss="hinge", use_actnorm=False,
                 percep_loss="lpips",
                 ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]

        self.balanced = balanced
        self.total_norm = total_norm

        self.percep_weight = percep_weight
        self.detect_weight = detect_weight
        self.disc_weight = disc_weight
        self.decode_weight = decode_weight

        # self.perceptual_loss = PerceptualLoss(percep_loss=percep_loss).to(torch.device("cuda"))
        self.perceptual_loss = PerceptualLoss(percep_loss=percep_loss)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else nn.BCEWithLogitsLoss()

        self.detection_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.decoding_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    @torch.no_grad()
    def calculate_adaptive_weights(
        self,
        losses,
        weights,
        last_layer,
        total_norm=0,
        choose_norm_idx=-1,
        eps=1e-12
    ) -> list:
        # calculate gradients for each loss
        grads = []

        for loss in losses:
            # allows for the computation of gradients w.r.t. intermediate layers if possible
            try:
                grads.append(torch.autograd.grad(
                    loss, last_layer, retain_graph=True)[0])
            except Exception as e:
                print(f"Error computing gradient: {str(e)}")
                grads.append(torch.zeros_like(last_layer))
        grad_norms = [torch.norm(grad) for grad in grads]

        # calculate base weights
        total_weight = sum(weights)
        ratios = [w / total_weight for w in weights]

        # choose total_norm to be the norm of the biggest weight
        assert choose_norm_idx or total_norm > 0, "Either choose_norm_idx or total_norm should be provided"
        if total_norm <= 0:  # if not provided, use the norm of the chosen weight
            # choose_norm_idx = ratios.index(max(ratios))
            total_norm = grad_norms[choose_norm_idx]

        # calculate adaptive weights
        scales = [r * total_norm / (eps + norm)
                  for r, norm in zip(ratios, grad_norms)]
        return scales

    def forward(self,
        inputs: torch.Tensor, reconstructions: torch.Tensor,
        masks: torch.Tensor, msgs: torch.Tensor, preds: torch.Tensor,
        optimizer_idx: int, global_step: int,
        last_layer=None, cond=None,
    ) -> tuple:
        
        if optimizer_idx == 0:  # embedder update
            weights = {}
            losses = {}

            # perceptual loss
            if self.percep_weight > 0:
                losses["percep"] = self.perceptual_loss(
                    imgs=inputs.contiguous(),
                    imgs_w=reconstructions.contiguous(),
                ).mean()
                weights["percep"] = self.percep_weight

            # discriminator loss
            if self.disc_weight > 0:

                with freeze_grads(self.discriminator):
                    disc_factor = adopt_weight(1.0, global_step, threshold=self.discriminator_iter_start)
                    logits_fake = self.discriminator(reconstructions.contiguous())
                    losses["disc"] = - logits_fake.mean()
                    weights["disc"] = disc_factor * self.disc_weight

            # detection loss
            if self.detect_weight > 0:
                detection_loss = self.detection_loss(
                    preds[:, 0:1].contiguous(),
                    masks.contiguous(),
                ).mean()
                losses["detect"] = detection_loss
                weights["detect"] = self.detect_weight

            # decoding loss
            if self.decode_weight > 0:
                msg_preds = preds[:, 1:]  # b nbits ...
                if msg_preds.dim() == 2:  # extract predicts msg
                    decoding_loss = self.decoding_loss(
                        msg_preds.contiguous(),  # b nbits
                        msgs.contiguous().float()
                    ).mean()
                else:  # extract predicts msg per pixel
                    masks = masks.expand_as(msg_preds).bool()  # b nbits h w
                    bsz, nbits, h, w = msg_preds.size()
                    # b nbits h w
                    msg_targs = msgs.unsqueeze(
                        -1).unsqueeze(-1).expand_as(msg_preds)
                    msg_preds_ = msg_preds.masked_select(masks).view(
                        bsz, nbits, -1)  # b 1 h w -> b nbits n
                    msg_targs_ = msg_targs.masked_select(masks).view(
                        bsz, nbits, -1)  # b 1 h w -> b nbits n
                    decoding_loss = self.decoding_loss(
                        msg_preds_.contiguous(),
                        msg_targs_.contiguous().float()
                    ).mean()
                losses["decode"] = decoding_loss
                weights["decode"] = self.decode_weight

            # calculate adaptive weights
            # turn off adaptive weights if any of the detector or embedder losses are turned off
            if last_layer is not None and self.balanced:
                scales = self.calculate_adaptive_weights(
                    losses=losses.values(),
                    weights=weights.values(),
                    last_layer=last_layer,
                    total_norm=self.total_norm,
                )
                scales = {k: v for k, v in zip(weights.keys(), scales)}
            else:
                scales = weights
            total_loss = sum(scales[key] * losses[key] for key in losses)
            # log
            log = {
                "total_loss": total_loss.clone().detach().mean(),
                **{f"loss_{k}": v.clone().detach().mean() for k, v in losses.items()},
                **{f"scale_{k}": v for k, v in scales.items()}
            }
            return total_loss, log

        if optimizer_idx == 1:  # discriminator update
            if cond is None:
                # detach here prevents gradient leakage to any module other than the discriminator
                logits_real = self.discriminator(
                    inputs.contiguous().detach())
                logits_fake = self.discriminator(
                    reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(
                1.0, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "disc_factor": disc_factor,
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log

    def to(self, device, *args, **kwargs):
        """
        Override for custom perceptual loss to device.
        """
        super().to(device)
        self.perceptual_loss = self.perceptual_loss.to(device)
        return self
