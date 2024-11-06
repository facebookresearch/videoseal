import io
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
import numpy as np

from videoseal.demo.io import DemoVideoBatchReader, DemoVideoWriter


def test_batch_reader():
    dir = Path(__file__).parent.resolve()
    with open(dir / "dog.mp4", "rb") as f:
        vr = VideoReader(f, ctx=cpu())
        num_frames = len(vr)
        batch_size = 4
        reader = DemoVideoBatchReader(vr, batch_size=batch_size)
        num_frames_processed = 0
        for frames in reader:
            num_frames_processed += len(frames)
            if reader.batch_i < reader.num_batches - 1:
                assert len(frames) == batch_size
            else:
                assert len(frames) > 0 and len(frames) <= batch_size
        assert num_frames_processed == num_frames


def test_writer():
    width = 10
    height = 20
    num_frames = 1
    fps = 30
    frame_data = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

    with io.BytesIO() as video_io:
        with DemoVideoWriter(video_io, fps, width, height) as writer:
            writer.write(frame_data)

        video_io.seek(0)
        vr = VideoReader(video_io)
        output_fps = vr.get_avg_fps()
        output_height, output_width = vr[0].shape[:2]

        assert output_fps == fps
        assert output_width == width
        assert output_height == height


def test_writer_crf():
    width = 10
    height = 20
    num_frames = 60
    fps = 30
    frame_data = (np.random.random((num_frames, height, width, 3)) * 255).astype(
        np.uint8
    )

    with io.BytesIO() as low_crf_io:
        with DemoVideoWriter(
            low_crf_io, fps, width, height, options={"crf": "10"}
        ) as writer:
            writer.write(frame_data)
        low_crf_len = len(low_crf_io.getvalue())

    with io.BytesIO() as high_crf_io:
        with DemoVideoWriter(
            high_crf_io, fps, width, height, options={"crf": "40"}
        ) as writer:
            writer.write(frame_data)
        high_crf_len = len(high_crf_io.getvalue())

    # low crf encoding should produce a larger file size
    assert low_crf_len > high_crf_len
