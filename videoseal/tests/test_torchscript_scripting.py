# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Regression test for torch.jit.script compatibility of the embedder and
detector submodules (e.g. for PixelSeal, which — unlike VideoSeal v1.0 —
ships no pre-exported .jit checkpoint, so users who need one must script
it themselves).

Before this fix, torch.jit.script(model.embedder) and
torch.jit.script(model.detector) both failed with the same underlying
cause: videoseal.modules.common.LayerNorm.forward() had an if/elif with
no else, so TorchScript inferred its return type as Optional[Tensor]
instead of Tensor — which then broke every nn.Sequential using it as a
norm layer (both the embedder's Upsample blocks and the detector's
ConvNeXt stages). Three further issues blocked scripting the rest of the
model: Embedder/Extractor.preprocess/postprocess were lambdas assigned as
instance attributes (unscriptable); ConvNeXtV2.forward_features() indexed
a ModuleList with a loop variable instead of iterating it; and
PixelDecoder.forward() had a stale Tuple[Tensor, Tensor] return
annotation for a method that returns a plain Tensor.

Run with:
    python -m videoseal.tests.test_torchscript_scripting
"""

import unittest

import torch

import videoseal


class TestTorchScriptScripting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = videoseal.load("pixelseal")
        cls.model.eval()
        torch.manual_seed(0)
        cls.img = torch.rand(1, 3, 64, 64)
        cls.msg = torch.randint(0, 2, (1, cls.model.embedder.msg_processor.nbits)).float()

    def test_embedder_scripts_and_matches_eager(self):
        scripted_embedder = torch.jit.script(self.model.embedder)
        embedder_input = (
            self.model.rgb2yuv(self.img)[:, 0:1] if self.model.embedder.yuv else self.img
        )
        with torch.no_grad():
            eager_out = self.model.embedder(embedder_input, self.msg)
            scripted_out = scripted_embedder(embedder_input, self.msg)
        self.assertEqual(
            (eager_out - scripted_out).abs().max().item(), 0.0
        )

    def test_detector_scripts_and_matches_eager(self):
        scripted_detector = torch.jit.script(self.model.detector)
        with torch.no_grad():
            eager_out = self.model.detector(self.img)
            scripted_out = scripted_detector(self.img)
        eager_preds = eager_out["preds"] if isinstance(eager_out, dict) else eager_out
        scripted_preds = scripted_out["preds"] if isinstance(scripted_out, dict) else scripted_out
        self.assertEqual(
            (eager_preds - scripted_preds).abs().max().item(), 0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
