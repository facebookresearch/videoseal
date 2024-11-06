import math

import numpy as np
from decord import VideoReader
from numpy.typing import NDArray

class DemoVideoBatchReader:
    def __init__(
        self,
        video_reader: VideoReader,
        batch_size: int,
    ):
        self.video_reader = video_reader
        self.num_frames = len(video_reader)
        self.num_batches = math.ceil(self.num_frames / batch_size)
        self.batch_size = batch_size
        self.batch_i = 0

    def __iter__(self):
        return self

    def __next__(self) -> NDArray[np.uint8]:
        if self.batch_i >= self.num_batches:
            raise StopIteration
        frame_i_low = self.batch_i * self.batch_size
        frame_i_high = min((self.batch_i + 1) * self.batch_size, self.num_frames)
        self.batch_i += 1
        return self.video_reader.get_batch(
            range(frame_i_low, frame_i_high),
        ).asnumpy()