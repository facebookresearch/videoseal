
import io
import random

import av
import numpy as np
import torch
import torch.nn as nn

from videoseal.data.transforms import normalize_img, unnormalize_img


class VideoCompression(nn.Module):
    """
    Simulate video artifacts by compressing and decompressing a video using PyAV.
    Attributes:
        codec (str): Codec to use for compression.
        crf (int): Constant Rate Factor for compression quality.
        fps (int): Frames per second of the video.
    """

    def __init__(self, codec='libx264', crf=28, fps=24, return_aux=False):
        super(VideoCompression, self).__init__()
        self.codec = codec  # values [28, 34, 40, 46]
        self.crf = crf
        self.fps = fps
        self.return_aux = return_aux

    def forward(self, frames, mask=None) -> torch.Tensor:
        """
        Compress and decompress the input video frames.
        Parameters:
            frames (torch.Tensor): Video frames as a tensor with shape (T, C, H, W).
        Returns:
            torch.Tensor: Decompressed video frames as a tensor with shape (T, C, H, W).
        """
        device = frames.device  # Get the device of the input frames
        # Save the original frames for skip gradients
        orig_frames = frames.clone()
        # Preprocess the frames for compression
        frames = unnormalize_img(frames)
        frames = frames.clamp(0, 1).permute(0, 2, 3, 1)  # t c h w -> t w h c
        frames = (frames * 255).to(torch.uint8).detach().cpu().numpy()
        # Create an in-memory bytes buffer
        with io.BytesIO() as buffer:

            # Create a PyAV container for output in memory
            container = av.open(buffer, mode='w', format='mp4')
            # Add a video stream to the container
            stream = container.add_stream(self.codec, rate=self.fps)
            stream.width = frames.shape[2]
            stream.height = frames.shape[1]
            stream.pix_fmt = 'yuv420p' if self.codec != 'libx264rgb' else 'rgb24'
            stream.options = {'crf': str(self.crf)}  # Set the CRF value
            # Write frames to the stream
            for frame_arr in frames:
                frame = av.VideoFrame.from_ndarray(frame_arr, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Finalize the file
            for packet in stream.encode():
                container.mux(packet)
            container.close()

            if self.return_aux:
                # Get the size of the buffer
                file_size = buffer.getbuffer().nbytes
            # Read from the in-memory buffer
            buffer.seek(0)

            with av.open(buffer, mode='r') as container:
                output_frames = []
                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format='rgb24')
                    output_frames.append(img)

        print("<<<<succesfull buffer at least once>>")
        return orig_frames, mask

        del frames  # Free memory
        # Postprocess the output frames
        output_frames = np.stack(output_frames) / 255
        output_frames = torch.tensor(output_frames, dtype=torch.float32)
        output_frames = output_frames.permute(0, 3, 1, 2)  # t w h c -> t c h w
        output_frames = normalize_img(output_frames)

        # Apply skip gradients
        compressed_frames = orig_frames + \
            (orig_frames - output_frames).detach()
        # del orig_frames  # Free memory
        if self.return_aux:
            return compressed_frames, mask, file_size
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
            codec=codec, crf=None, fps=fps, return_aux=False)
        self.crf_values = crf_values

    def get_random_crf(self):
        """Randomly select a CRF value from the list of values."""
        return random.choice(self.crf_values)

    def forward(self, frames, mask=None) -> torch.Tensor:
        """Compress and decompress the input video frames with a randomly selected CRF value."""
        crf = self.get_random_crf()
        self.crf = crf
        output, mask = super().forward(frames, mask)
        return output, mask


def compress_decompress(frames, codec='libx264', crf=28, fps=24, return_aux=False) -> torch.Tensor:
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
    compressor = VideoCompression(
        codec=codec, crf=crf, fps=fps, return_aux=return_aux)
    return compressor(frames)
