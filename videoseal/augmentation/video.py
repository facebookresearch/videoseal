"""
Test with:
    python -m videoseal.augmentation.video
"""

import io
import av
import numpy as np
import torch
import torch.nn as nn

class VideoCompression(nn.Module):
    """
    Simulate video artifacts by compressing and decompressing a video using PyAV.
    Attributes:
        codec (str): Codec to use for compression.
        crf (int): Constant Rate Factor for compression quality.
        fps (int): Frames per second of the video.
    """

    def __init__(self, codec='libx264', crf=28, fps=24):
        super(VideoCompression, self).__init__()
        self.codec = codec  # values [28, 34, 40, 46]
        self.pix_fmt = 'yuv420p' if codec != 'libx264rgb' else 'rgb24'
        self.threads = 1  # limit the number of threads to avoid memory issues
        self.crf = crf
        self.fps = fps

    def _preprocess_frames(self, frames) -> torch.Tensor:
        frames = frames.clamp(0, 1).permute(0, 2, 3, 1)
        frames = (frames * 255).to(torch.uint8).detach().cpu().numpy()
        return frames
    
    def _postprocess_frames(self, frames) -> torch.Tensor:
        frames = np.stack(frames) / 255
        frames = torch.tensor(frames, dtype=torch.float32)
        frames = frames.permute(0, 3, 1, 2)
        return frames

    def _compress_frames(self, buffer, frames) -> io.BytesIO:
        """
        Compress the input video frames.
        Uses the PyAV library to compress the frames, then writes them to the buffer.
        Finally, returns the buffer with the compressed video.
        """
        with av.open(buffer, mode='w', format='mp4') as container:
            stream = container.add_stream(self.codec, rate=self.fps)
            stream.width, stream.height = frames.shape[2], frames.shape[1]
            stream.pix_fmt = self.pix_fmt
            stream.options = {
                'crf': str(self.crf), 
                'threads': str(self.threads), 
                'x265-params': 'log_level=none'  # Disable x265 logging
            }
            for frame_arr in frames:
                frame = av.VideoFrame.from_ndarray(frame_arr, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        buffer.seek(0)
        # file_size = buffer.getbuffer().nbytes
        return buffer

    def _decompress_frames(self, buffer) -> list:
        """
        Decompress the input video frames.
        Uses the PyAV library to decompress the frames, then returns them as a list of frames.
        """
        with av.open(buffer, mode='r') as container:
            output_frames = []
            frame = ""
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format='rgb24')
                output_frames.append(img)
        return output_frames

    def forward(self, frames, mask=None, crf=None) -> torch.Tensor:
        """
        Compress and decompress the input video frames.
        Parameters:
            frames (torch.Tensor): Video frames as a tensor with shape (T, C, H, W).
            mask (torch.Tensor): Optional mask for the video frames.
            crf (int): Constant Rate Factor for compression quality, if not provided, uses the self.crf value.
        Returns:
            torch.Tensor: Decompressed video frames as a tensor with shape (T, C, H, W).
        """
        self.crf = crf or self.crf

        input_frames = self._preprocess_frames(frames)
        with io.BytesIO() as buffer:
            buffer = self._compress_frames(buffer, input_frames)
            output_frames = self._decompress_frames(buffer)
        output_frames = self._postprocess_frames(output_frames)
        output_frames = output_frames.to(frames.device)

        compressed_frames = frames + (frames - output_frames).detach()
        del frames  # Free memory

        return compressed_frames, mask

    def __repr__(self) -> str:
        return f"Compressor(codec={self.codec}, crf={self.crf}, fps={self.fps})"


class VideoCompressorAugmenter(VideoCompression):
    """
    A compressor augmenter that randomly selects a CRF value from a list of values.

    Attributes:
        codec (str): Codec to use for compression.
        fps (int): Frames per second of the video.
        crf_values (list): List of CRF values to select from.
    """

    def __init__(self, codec='libx264', fps=24, crf_values=[28, 34, 40, 46]):
        super(VideoCompressorAugmenter, self).__init__(
            codec=codec, crf=None, fps=fps)
        self.crf_values = crf_values

    def get_random_crf(self):
        """Randomly select a CRF value from the list of values."""
        return np.random.choice(self.crf_values)

    def forward(self, frames, mask=None, *args, **kwargs) -> torch.Tensor:
        """Compress and decompress the input video frames with a randomly selected CRF value."""
        crf = self.get_random_crf()
        output, mask = super().forward(frames, mask, crf)
        return output, mask


class H264(VideoCompression):
    def __init__(self, crf_min=None, crf_max=None, fps=24):
        super(VideoCompressorAugmenter, self).__init__(
            codec='libx264', fps=fps)
        self.crf_min = crf_min
        self.crf_max = crf_max

    def get_random_crf(self):
        if self.min_crf is None or self.max_crf is None:
            raise ValueError("min_crf and max_crf must be provided")
        return torch.randint(self.min_crf, self.max_crf + 1, size=(1,)).item()

    def forward(self, frames, mask=None, crf=None) -> torch.Tensor:
        crf = crf or self.get_random_crf()
        output, mask = super().forward(frames, mask, crf)
        return output, mask


def compress_decompress(frames, codec='libx264', crf=28, fps=24) -> torch.Tensor:
    """
    Simulate video artifacts by compressing and decompressing a video using PyAV.

    Parameters:
        frames (torch.Tensor): Video frames as a tensor with shape (T, C, H, W).
        codec (str): Codec to use for compression.
        crf (int): Constant Rate Factor for compression quality.
        fps (int): Frames per second of the video.

    Returns:
        torch.Tensor: Decompressed video frames as a tensor with shape (T, C, H, W).
    """
    compressor = VideoCompression(codec=codec, crf=crf, fps=fps)
    return compressor(frames)


if __name__ == "__main__":    
    from videoseal.data.loader import load_video
    vid_o = 'assets/videos/sav_013754.mp4'
    print("> test compression")
    
    vid_o = load_video(vid_o) / 255
    vid_o = vid_o[:60]  # Use only the first 60 frames
    # for codec in ['libx264', 'libx264rgb', 'libx265', 'libvpx-vp9']:
    for codec in ['libvpx-vp9']:
        crfs = [28, 34, 40, 46] if codec != 'libvpx-vp9' else [-1]
        for crf in crfs:
            try:
                compressor = VideoCompression(codec=codec, crf=crf)
                compressed_frames, _ = compressor(vid_o)
                mse = torch.nn.functional.mse_loss(vid_o, compressed_frames)
                print(f"Codec: {codec}, CRF: {crf} - MSE: {mse:.4f}")
            except Exception as e:
                print(f":warning: An error occurred with {codec}: {str(e)}")

    # should print
    # Codec: libx264, CRF: 28 - MSE: 0.0005
    # Codec: libx264, CRF: 34 - MSE: 0.0011
    # Codec: libx264, CRF: 40 - MSE: 0.0025
    # Codec: libx264, CRF: 46 - MSE: 0.0048
    # Codec: libx264rgb, CRF: 28 - MSE: 0.0004
    # Codec: libx264rgb, CRF: 34 - MSE: 0.0008
    # Codec: libx264rgb, CRF: 40 - MSE: 0.0014
    # Codec: libx264rgb, CRF: 46 - MSE: 0.0025
    # Codec: libx265, CRF: 28 - MSE: 0.0004
    # Codec: libx265, CRF: 34 - MSE: 0.0010
    # Codec: libx265, CRF: 40 - MSE: 0.0021
    # Codec: libx265, CRF: 46 - MSE: 0.0041
    # Codec: libvpx-vp9, CRF: -1 - MSE: 0.0011