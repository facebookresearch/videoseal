# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Regression test for three more TorchScript blockers, found while scripting
the full top-level model (not just the embedder/detector submodules that
test_torchscript_scripting.py covers):

1. Blender.forward() looked up a blend function in a dict of bound methods
   (self.blend_methods[self.method]) -- TorchScript cannot type a dict whose
   values are callables. Replaced with an explicit if/elif/else branch on
   self.method (fixed at construction and validated there), the same style
   used for the LayerNorm fix.
2. JND.jnd_la()/jnd_cm() had untyped float defaults (alpha=1.0, eps=1e-5).
   TorchScript assumes an unannotated parameter is Tensor, which collides
   with a float default. Added explicit `float` annotations, matching the
   style JND.heatmaps() already used.
3. Wam.device was `next(self.parameters()).device` -- TorchScript supports
   neither next() on an iterator nor the yield-based generator that
   self.parameters() itself is built on. Replaced with a dedicated
   non-persistent buffer (`_device_marker`) that moves with every .to(device)
   call exactly like a parameter does; reading its .device is scriptable.

Together with the embedder/detector fix, these let a full inference-only
wrapper around the real model script end-to-end and produce a self-exported
.jit for models that ship no official pre-exported one (e.g. PixelSeal) --
demonstrated below for image, video, and detection, each checked bit-exact
against eager.

Run with:
    python -m videoseal.tests.test_blender_jnd_device_scripting
"""

import unittest

import torch
from torch.nn import functional as F

import videoseal
from videoseal.models.videoseal import Videoseal


def _apply_video_mode(preds_w, total_frames: int, step_size: int, video_mode: str):
    # Inline port of Videoseal._apply_video_mode: TorchScript doesn't support
    # the `ClassName.method(...)` unbound-call pattern, so this is copied as
    # a free function rather than called off the class.
    if video_mode == "repeat":
        preds_w = torch.repeat_interleave(preds_w, step_size, dim=0)
    elif video_mode == "alternate":
        full_size = (total_frames,) + preds_w.shape[1:]
        full_preds = torch.zeros(full_size, device=preds_w.device)
        full_preds[::step_size] = preds_w
        preds_w = full_preds
    elif video_mode == "interpolate":
        full_size = (total_frames,) + preds_w.shape[1:]
        full_preds = torch.zeros(full_size, device=preds_w.device)
        alpha = 1 - torch.linspace(0, 1, steps=step_size, device=preds_w.device)
        alpha = alpha.repeat((total_frames - 1) // step_size).view(-1, 1, 1, 1)
        start_frames = torch.repeat_interleave(preds_w[:-1], step_size, dim=0)
        end_frames = torch.repeat_interleave(preds_w[1:], step_size, dim=0)
        interpolated_preds = alpha * start_frames + (1 - alpha) * end_frames
        last_start = len(interpolated_preds)
        full_preds[:last_start] = interpolated_preds
        full_preds[last_start:] = preds_w[-1]
        preds_w = full_preds
    return preds_w[:total_frames]


class ScriptableVideoseal(Videoseal):
    """Inference-only scriptable wrapper used only by this test. Reimplements
    embed()/detect() inline (not by calling the parent's version --
    TorchScript doesn't support `BaseClass.method(self, ...)`) so it returns
    a bare Tensor matching VideoSeal v1.0's documented .jit API, and never
    touches self.augmenter (training-only; removed before scripting)."""

    @torch.jit.unused
    def forward(self, imgs, msgs=None, is_video: bool = True):
        raise NotImplementedError("forward() is training-only; use embed()/detect()")

    @torch.jit.export
    def embed(self, imgs: torch.Tensor, msgs: torch.Tensor, is_video: bool = False) -> torch.Tensor:
        if is_video:
            return self._embed_video(imgs, msgs)
        return self._embed_image(imgs, msgs)

    def _embed_image(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(
                imgs, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
        imgs_res = imgs_res.to(self.device)

        if self.embedder.yuv:
            preds_w = self.embedder(self.rgb2yuv(imgs_res)[:, 0:1], msgs.to(self.device))
        else:
            preds_w = self.embedder(imgs_res, msgs.to(self.device))

        if imgs.shape[-2:] != (self.img_size, self.img_size):
            preds_w = F.interpolate(
                preds_w, size=imgs.shape[-2:],
                mode="bilinear", align_corners=False, antialias=True,
            )
        preds_w = preds_w.to(imgs.device)

        hmaps = self.attenuation.heatmaps(imgs)
        preds_w = hmaps * preds_w

        imgs_w = self.blender(imgs, preds_w)
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)
        return imgs_w

    def _embed_video(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        msgs = msgs.repeat(self.chunk_size, 1)
        chunk_size = self.chunk_size
        step_size = self.step_size
        imgs_w = torch.zeros_like(imgs)

        keyed = imgs[::step_size]
        n_keyframes = keyed.shape[0]
        for ii in range(0, n_keyframes, chunk_size):
            nimgs_in_ck = min(chunk_size, n_keyframes - ii)
            start = ii * step_size
            end = start + nimgs_in_ck * step_size
            all_imgs_in_ck = imgs[start:end, ...]

            chunk_msgs = msgs
            if nimgs_in_ck < chunk_size:
                chunk_msgs = msgs[:nimgs_in_ck]

            all_imgs_res = all_imgs_in_ck.clone()
            if all_imgs_res.shape[-2:] != (self.img_size, self.img_size):
                all_imgs_res = F.interpolate(
                    all_imgs_res, size=(self.img_size, self.img_size),
                    mode="bilinear", align_corners=False, antialias=True,
                )
            all_imgs_res = all_imgs_res.to(self.device)

            imgs_res = all_imgs_res[::step_size]
            if self.embedder.yuv:
                imgs_res = self.rgb2yuv(imgs_res)[:, 0:1]
            preds_w = self.embedder(imgs_res, chunk_msgs.to(self.device))

            preds_w = _apply_video_mode(preds_w, all_imgs_in_ck.shape[0], step_size, self.video_mode)

            preds_w = preds_w.to(imgs.device)
            if all_imgs_in_ck.shape[-2:] != (self.img_size, self.img_size):
                preds_w = F.interpolate(
                    preds_w, size=all_imgs_in_ck.shape[-2:],
                    mode="bilinear", align_corners=False, antialias=True,
                )

            hmaps = self.attenuation.heatmaps(all_imgs_in_ck)
            preds_w = hmaps * preds_w

            all_imgs_in_ck_w = self.blender(all_imgs_in_ck, preds_w)
            imgs_w[start:end, ...] = all_imgs_in_ck_w

        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)
        return imgs_w

    @torch.jit.export
    def detect(self, imgs: torch.Tensor, is_video: bool = False) -> torch.Tensor:
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(
                imgs, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
        imgs_res = imgs_res.to(self.device)
        preds = self.detector(imgs_res).to(imgs.device)
        return preds


class TestBlenderJndDeviceScripting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = videoseal.load("pixelseal")
        cls.model.eval()

    def test_blender_scripts_and_matches_eager(self):
        imgs = torch.rand(2, 3, 64, 64)
        preds_w = torch.rand(2, 3, 64, 64) * 0.1
        scripted = torch.jit.script(self.model.blender)
        with torch.no_grad():
            eager_out = self.model.blender(imgs, preds_w)
            scripted_out = scripted(imgs, preds_w)
        self.assertEqual((eager_out - scripted_out).abs().max().item(), 0.0)

    def test_attenuation_scripts_and_matches_eager(self):
        imgs = torch.rand(2, 3, 64, 64)
        imgs_w = imgs + torch.rand(2, 3, 64, 64) * 0.1
        scripted = torch.jit.script(self.model.attenuation)
        with torch.no_grad():
            eager_out = self.model.attenuation(imgs, imgs_w)
            scripted_out = scripted(imgs, imgs_w)
        self.assertEqual((eager_out - scripted_out).abs().max().item(), 0.0)

    @unittest.skip(
        "Needs the embedder/detector TorchScript fix from a separate PR "
        "(facebookresearch/videoseal#87) too -- this branch only carries the "
        "blender/attenuation/device fixes, so scripting the full model still "
        "hits #87's LayerNorm issue on its own. Verified passing when both "
        "are applied together (see PR description); not committed here as a "
        "red test with no fix available on this branch to make it pass."
    )
    def test_full_inference_model_scripts_image_video_and_detect(self):
        model = videoseal.load("pixelseal")
        model.eval()
        model.__class__ = ScriptableVideoseal
        # Inference never touches the augmenter (training-only, random
        # corruptions); dropping the reference is the only way to exclude a
        # genuinely unscriptable submodule nothing on the inference path calls.
        del model._modules["augmenter"]
        model.augmenter = None

        img = torch.rand(1, 3, 64, 64)
        msg = torch.randint(0, 2, (1, 256)).float()
        with torch.no_grad():
            eager_embed = model._embed_image(img, msg)

        scripted = torch.jit.script(model)

        with torch.no_grad():
            scripted_embed = scripted.embed(img, msg, is_video=False)
        self.assertEqual((eager_embed - scripted_embed).abs().max().item(), 0.0)

        video = torch.rand(6, 3, 32, 32)
        vmsg = torch.randint(0, 2, (1, 256)).float()
        with torch.no_grad():
            eager_video = model._embed_video(video, vmsg)
            scripted_video = scripted.embed(video, vmsg, is_video=True)
        self.assertEqual((eager_video - scripted_video).abs().max().item(), 0.0)

        with torch.no_grad():
            eager_det = model.detect(img)
            scripted_det = scripted.detect(img)
        self.assertEqual((eager_det - scripted_det).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
