import os
from typing import Union

import cv2
import torch
import torchvision
from torch import Tensor


def save_img(img: Tensor, out_path: str) -> None:
    """
    Saves an image tensor to a file.

    Args:
    img (Tensor): The image tensor with shape (C, H, W) where
                  C is the number of channels (should be 3),
                  H is the height,
                  W is the width.
    out_path (str): The output path for the saved image file.

    Raises:
    AssertionError: If the input tensor does not have the correct dimensions or channel size.
    """
    # Assert the image tensor has the correct dimensions
    assert img.dim() == 3, "Input image tensor must have 3 dimensions (C, H, W)"
    assert img.size(0) == 3, "Image tensor's channel size must be 3"

    # Clamp the values and convert to numpy
    img = img.clamp(0, 1) * 255
    img = img.to(torch.uint8).cpu()

    # Write the image file
    img_pil = torchvision.transforms.ToPILImage()(img)
    img_pil.save(out_path)

def save_vid(vid: Tensor, out_path: str, fps: int, crf: int = 23) -> None:
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
    crf (int): Constant Rate Factor for the output video (default is 23).

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
    options = {"crf": f"{crf}"}
    torchvision.io.write_video(out_path, vid, fps=fps, video_codec="libx264", options=options)


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
