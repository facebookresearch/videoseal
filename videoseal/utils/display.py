import os
from typing import Union

import cv2
import torch
import torchvision
from torch import Tensor


def save_vid(vid: Tensor, out_path: str, fps: int) -> None:
    """
    Saves a video tensor to a file.

    Args:
    vid (Tensor): The video tensor with shape (T, C, H, W) where
                  T is the number of frames,
                  C is the number of channels (should be 3),
                  H is the height,
                  W is the width.
    out_path (str): The output path for the saved video file.
    fps (int): Frames per second of the output video.
    normalize (bool): Flag to determine whether to normalize the video tensor.

    Raises:
    AssertionError: If the input tensor does not have the correct dimensions or channel size.
    """
    # Assert the video tensor has the correct dimensions
    assert vid.dim() == 4, "Input video tensor must have 4 dimensions (T, C, H, W)"
    assert vid.size(1) == 3, "Video tensor's channel size must be 3"

    # Clamp the values and convert to numpy
    vid = vid.clamp(0, 1)

    # Convert from (T, C, H, W) to (T, H, W, C)
    vid = vid.permute(0, 2, 3, 1) * 255
    vid = vid.to(torch.uint8).cpu()

    # Write the video file
    torchvision.io.write_video(
        out_path, vid, fps=fps, video_codec='libx264', options={'crf': '21'})


def get_fps(video_path: Union[str, os.PathLike]) -> tuple:
    """
    Retrieves the FPS and frame count of a video.

    Args:
    video_path (Union[str, os.PathLike]): Path to the video file.

    Returns:
    tuple: Contains the FPS (float) and frame count (int).

    Raises:
    AssertionError: If the video file does not exist.
    """
    assert os.path.exists(
        video_path), f"Video file does not exist: {video_path}"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    return fps, frame_count
