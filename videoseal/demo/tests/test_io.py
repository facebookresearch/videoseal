from pathlib import Path

from decord import VideoReader, cpu

from videoseal.demo.io import DemoVideoBatchReader


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
