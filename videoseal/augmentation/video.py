
import io

import av
import numpy as np
import torch

from videoseal.data.transforms import normalize_img, unnormalize_img


def compress_decompress(frames, codec='libx264', crf=23, fps=24, return_aux=False) -> torch.Tensor:
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
    device = frames.device
    frames = unnormalize_img(frames)
    frames = frames.clamp(0, 1).permute(0, 2, 3, 1)  # t c h w -> t w h c
    frames = (frames * 255).to(torch.uint8).cpu().numpy()

    # Create an in-memory bytes buffer
    buffer = io.BytesIO()

    # Create a PyAV container for output in memory
    container = av.open(buffer, mode='w', format='mp4')

    # Add a video stream to the container
    stream = container.add_stream(codec, rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = 'yuv420p' if codec != 'libx264rgb' else 'rgb24'
    stream.options = {'crf': str(crf)}

    # Write frames to the stream
    for frame_arr in frames:
        frame = av.VideoFrame.from_ndarray(frame_arr, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    # Finalize the file
    for packet in stream.encode():
        container.mux(packet)

    container.close()

    if return_aux:
        # Get the size of the buffer
        file_size = buffer.getbuffer().nbytes
        print(f'Compressed video size: {file_size / 1e6:.2f} MB')

    # Read from the in-memory buffer
    buffer.seek(0)
    container = av.open(buffer, mode='r')
    output_frames = []

    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        output_frames.append(img)

    container.close()

    output_frames = np.stack(output_frames) / 255
    output_frames = torch.tensor(output_frames, dtype=torch.float32)
    output_frames = output_frames.permute(0, 3, 1, 2)  # t w h c -> t c h w
    output_frames = normalize_img(output_frames)

    if return_aux:
        return output_frames, file_size

    # move back to device for interface consistency
    return output_frames.to(device)
