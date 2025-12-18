# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
import tempfile
from scipy.io.wavfile import write

import cv2

import torch
import torchvision


def save_img(img: torch.Tensor, out_path: str) -> None:
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

def save_vid(vid: torch.Tensor, out_path: str, fps: int, crf: int=11) -> None:
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
    torchvision.io.write_video(out_path, vid, fps=fps, video_codec="h264", options={"crf": f"{crf}"})

def save_video_audio_to_mp4(video_tensor: torch.Tensor, audio_tensor: torch.Tensor, 
                            fps: int, audio_sample_rate: int, output_filename: str) -> None:
    """
    Saves the given video and audio tensors into an MP4 file using FFmpeg.

    Args:
        video_tensor (torch.Tensor): Video tensor of shape [T, C, H, W].
        audio_tensor (torch.Tensor): Audio tensor of shape [channels, timesteps].
        fps (int): Frame rate of the video (frames per second).
        audio_sample_rate (int): Sample rate of the audio (samples per second).
        output_filename (str): Output filename for the MP4 video.

    Returns:
        None
    """
    # Create a temporary directory to store video frames
    temp_dir = tempfile.mkdtemp()

    # Ensure video_tensor is in the shape [T, C, H, W] and permute it to [T, H, W, C]
    T, C, H, W = video_tensor.shape
    video_tensor = video_tensor.permute(0, 2, 3, 1)  # [T, H, W, C]

    # Save each video frame as a PNG image
    for i in range(T):
        # Clamp the values and convert to numpy
        frame_filename = os.path.join(temp_dir, f"frame_{i:04d}.png")
        frame = (video_tensor[i].clamp(0,1) * 255).byte().cpu().numpy()  # [H, W, C] in RGB
        frame = frame[:, :, [2, 1, 0]]  # Convert RGB to BGR
        cv2.imwrite(frame_filename, frame)  # Save to PNG

    # Save the audio tensor to a temporary WAV file
    temp_audio_file = tempfile.mktemp(suffix=".wav")

    # Normalize audio to the range of [-32768, 32767] for 16-bit PCM
    audio_tensor = torch.clamp(audio_tensor, -1, 1)  # Clip to ensure it is in [-1, 1]
    audio_tensor = (audio_tensor * 32767).to(torch.int16)  # Convert to int16 for WAV format

    # Convert to NumPy array and write to WAV file
    audio_numpy = audio_tensor.cpu().numpy()

    # Use scipy.io.wavfile.write to handle multichannel audio
    write(temp_audio_file, audio_sample_rate, audio_numpy.T)  # Transpose to [timesteps, channels]

    # Use FFmpeg to create a video from images
    video_command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-r', str(fps),
        '-pix_fmt', 'rgb24',
        '-y', 'video.mp4'
    ]
    subprocess.run(video_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Use FFmpeg to add audio to the video
    final_command = [
        'ffmpeg',
        '-i', 'video.mp4',
        '-i', temp_audio_file,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y', output_filename
    ]
    subprocess.run(final_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clean up temporary files
    shutil.rmtree(temp_dir)
    os.remove(temp_audio_file)
    os.remove('video.mp4')

    print(f"Video saved to {output_filename}")

def get_fps(video_path: str|os.PathLike) -> tuple:
    """
    Retrieves the FPS and frame count of a video.

    Args:
    video_path (str|os.PathLike): Path to the video file.

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
