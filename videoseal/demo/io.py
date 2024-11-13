import math
from typing import IO

import av
import numpy as np
from av.video.frame import PictureType
from av.video.stream import VideoStream
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


class DemoVideoWriter:
    def __init__(
        self,
        output_video_file: IO,
        fps: int,
        width: int,
        height: int,
        video_codec: str = "libx264",
        options: dict = {},
    ):
        self.container = av.open(output_video_file, mode="w", format="mp4")
        stream = self.container.add_stream(video_codec, rate=fps, options=options)
        if not isinstance(stream, VideoStream):
            return
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.width = width
        stream.height = height
        self.stream = stream

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()

    def write(self, video: NDArray[np.uint8]):
        for frame in video:
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            frame.pict_type = PictureType.NONE
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
