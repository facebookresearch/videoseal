# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test Augmenter.forward() in eval mode.

Regression test for a bug where the eval (self.training=False) branch
referenced `imgs_aug` before it was assigned, and returned an undefined
`selected_aug` instead of joining `selected_augs`. This made any call to
Videoseal.forward()/__call__ in eval mode crash with UnboundLocalError.

Run with:
    python -m videoseal.tests.test_augmenter
"""

import unittest

import torch

from videoseal.augmentation.augmenter import get_dummy_augmenter


class TestAugmenterEvalMode(unittest.TestCase):

    def setUp(self):
        self.augmenter = get_dummy_augmenter()
        self.augmenter.eval()

    def test_forward_eval_mode_does_not_crash(self):
        imgs_w = torch.rand(2, 3, 32, 32)
        imgs = torch.rand(2, 3, 32, 32)
        masks = torch.ones(2, 1, 32, 32)

        imgs_aug, mask_targets, selected_aug = self.augmenter(
            imgs_w, imgs, masks, is_video=False, do_resize=True
        )

        self.assertEqual(imgs_aug.shape, imgs_w.shape)
        self.assertEqual(mask_targets.shape[0], imgs_w.shape[0])
        self.assertIsInstance(selected_aug, str)
        self.assertGreater(len(selected_aug), 0)

    def test_forward_eval_mode_multiple_augs(self):
        augmenter = get_dummy_augmenter()
        augmenter.num_augs = 3
        augmenter.eval()

        imgs_w = torch.rand(1, 3, 16, 16)
        imgs = torch.rand(1, 3, 16, 16)
        masks = torch.ones(1, 1, 16, 16)

        imgs_aug, mask_targets, selected_aug = augmenter(
            imgs_w, imgs, masks, is_video=False, do_resize=True
        )

        # one "+"-joined entry per augmentation round
        self.assertEqual(selected_aug.count("+"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
