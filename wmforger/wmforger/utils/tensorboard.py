# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.tensorboard import SummaryWriter

from .dist import is_main_process


class CustomTensorboardWriter:
    """
    A custom TensorBoard writer that handles writing to TensorBoard only when running in the main process.
    Attributes:
        writer (SummaryWriter): The underlying TensorBoard writer. This is None if not running in the main process.
    """

    def __init__(self, log_dir=None, log_scalars_every=10, *args, **kwargs):

        self.writer = SummaryWriter(
            log_dir=log_dir, *args, **kwargs) if is_main_process() else None

    def add_scalars(self, prefix, metrics, step):
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def add_scalar(self, *args, **kwargs):
        if self.writer is not None:
            self.writer.add_scalar(*args, **kwargs)

    def add_hparams(self, params: dict, metrics: dict):
        if self.writer is not None:
            self.writer.add_hparams(params, metrics)

    def add_graph(self, *args, **kwargs):
        if self.writer is not None:
            self.writer.add_graph(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.writer is not None:
            self.writer.add_image(*args, **kwargs)

    def add_images(self, *args, **kwargs):
        if self.writer is not None:
            self.writer.add_images(*args, **kwargs)

    def add_video(self, *args, **kwargs):
        if self.writer is not None:
            self.writer.add_video(*args, **kwargs)

    def close(self):
        if self.writer is not None:
            self.writer.close()
