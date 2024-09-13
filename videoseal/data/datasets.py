# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# adapted from https://github.com/facebookresearch/jepa/blob/main/src/datasets/video_dataset.py


import glob
import logging
import os
import pathlib
import warnings
from logging import getLogger
from typing import Callable, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from tqdm import tqdm

from videoseal.data.transforms import get_transforms_segmentation

from .utils import LRUDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """Video classification dataset that loads video files directly from specified folders."""

    def __init__(
        self,
        # List of folder paths containing .mp4 video files
        folder_paths: List[str],
        datasets_weights: Optional[List[float]] = None,
        frames_per_clip: int = 16,  # Number of frames in each video clip
        frame_step: int = 4,  # Step size between frames within a clip
        num_clips: int = 1,  # Number of clips to sample from each video
        # Optional transformation function to be applied to each clip
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        # Optional transformation function applied on the video before clipping
        shared_transform: Optional[Callable] = None,
        # If True, sample clips randomly inside the video
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False,  # If True, allow clips to overlap
        # If True, exclude videos that are shorter than the required clip length
        filter_short_videos: bool = False,
        # Maximum allowed video file size in bytes
        filter_long_videos: Union[int, float] = int(10**9),
        # Optional, specific duration in seconds for each clip
        duration: Optional[float] = None,
        output_resolution: tuple = (224, 224),  # Desired output resolution
        num_workers: int = 1,  # numbers of cpu to run the preprocessing of each batch
        # If True, flatten clips into individual frames
        flatten_clips_to_frames: bool = True,
    ):
        self.folder_paths = folder_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.mask_transform = mask_transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.output_resolution = output_resolution
        self.num_workers = num_workers
        self.flatten_clips_to_frames = flatten_clips_to_frames

        if VideoReader is None:
            raise ImportError(
                'Unable to import "decord" which is required to read videos.')
        # Load video paths from folders
        self.videofiles = []

        self.num_video_files_per_dataset = []
        for folder_path in self.folder_paths:
            logger.info("Loading videos from %s", folder_path)
            video_files = glob.glob(os.path.join(folder_path, '*.mp4'))
            logger.info("Found %d videos in %s", len(video_files), folder_path)

            for video_file in tqdm(video_files,
                                   desc=f"Processing videos in {folder_path}"):
                self.videofiles.append(video_file)

            self.num_video_files_per_dataset.append(len(video_files))
            logger.info("Total videos loaded from %s: %d",
                        folder_path, len(video_files))

        # [Optional] Weights for each sample to be used by downstream weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_video_files_per_dataset):
                self.sample_weights += [dw / ns] * ns
            logger.info("Sample weights have been calculated and applied.")

        # Initialize video buffer
        # Set the maximum size of the buffer
        self.video_buffer = LRUDict(maxsize=10)

    def __getitem__(self, index):
        if self.flatten_clips_to_frames:
            # Calculate the index of the videofile, clip, and frame
            videofile_index = index // (self.num_clips * self.frames_per_clip)
            clip_index = (index % (self.num_clips *
                          self.frames_per_clip)) // self.frames_per_clip
            frame_index = index % self.frames_per_clip
        else:
            # Calculate the index of the sample and clip
            videofile_index = index // self.num_clips
            clip_index = index % self.num_clips

        video_file = self.videofiles[videofile_index]

        # if the video_file was not processed before, process it and safe to buffer
        if video_file not in self.video_buffer:
            # Keep trying to load videos until you find a valid sample
            loaded_video = False
            while not loaded_video:
                buffer, frames_indices = self.loadvideo_decord(
                    video_file)  # [T H W 3]
                loaded_video = len(buffer) > 0
                if not loaded_video:
                    videofile_index = np.random.randint(self.__len__())
                    video_file = self.videofiles[videofile_index]

            def split_into_clips(video):
                """ Split video into a list of clips """
                fpc = self.frames_per_clip
                nc = self.num_clips
                return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

            # Parse video into frames & apply data augmentations
            if self.shared_transform is not None:
                buffer = self.shared_transform(buffer)
            buffer = split_into_clips(buffer)

            # Convert buffer to PyTorch tensor and permute dimensions
            # Permute is used to rearrange the dimensions of the tensor.
            # In this case, we're rearranging the dimensions from (frames, height, width, channels)
            # to (frames, channels, height, width), which is the expected input format for
            # torch.nn.functional.interpolate.
            buffer = torch.from_numpy(np.concatenate(
                buffer, axis=0)).permute(0, 3, 1, 2).float()
            # Apply torch.nn.functional.interpolate transformation
            buffer = torch.nn.functional.interpolate(
                buffer, size=self.output_resolution, mode='bilinear')
            # Reshape buffer back to (num_clips, frames_per_clip, channels, height, width)
            buffer = buffer.view(
                self.num_clips, self.frames_per_clip, *buffer.shape[1:])

            # Store the loaded video in the buffer
            self.video_buffer[video_file] = (buffer, frames_indices)

        # load directly from buffer here should be processed already
        buffer, frames_positions_in_clips = self.video_buffer[video_file]

        if self.flatten_clips_to_frames:
            # Return a single frame and its index
            frame = buffer[clip_index, frame_index]
            frame_index_in_video = frames_positions_in_clips[clip_index][frame_index]

            if self.transform is not None:
                frame = self.transform(frame)

            # Get MASKS
            # TODO: Dummy mask of 1s
            # TODO: implement mask transforms
            mask = torch.ones_like(frame[0:1, ...])
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)

            return frame, mask, frame_index_in_video
        else:
            # Return a clip and its frame indices
            clip = buffer[clip_index]
            clip_frame_indices = frames_positions_in_clips[clip_index]

            if self.transform is not None:
                clip = torch.stack([self.transform(frame) for frame in clip])

            # Get MASKS
            # TODO: Dummy mask of 1s
            # TODO: implement mask transforms
            mask = torch.ones_like(clip[:, 0:1, ...])
            if self.mask_transform is not None:
                mask = torch.stack([self.mask_transform(one_mask)
                                   for one_mask in mask])

            return clip, mask, clip_frame_indices

    def loadvideo_decord(self, sample):
        """ Load video content using Decord """

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        try:
            vr = VideoReader(
                fname, num_threads=self.num_workers, ctx=cpu(0))
        except Exception:
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fps = vr.get_avg_fps()
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f'skipping video of length {len(vr)}')
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx,
                                  end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(
                        0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate(
                        (indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(
                        indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(
                        0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate(
                        (indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(
                        indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (
                            len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        if self.flatten_clips_to_frames:
            return len(self.videofiles) * self.num_clips * self.frames_per_clip
        else:
            return len(self.videofiles) * self.num_clips


if __name__ == "__main__":
    import time

    # Specify the path to the folder containing the MP4 files
    video_folder_path = "./assets/videos"

    train_transform, train_mask_transform, val_transform, val_mask_transform = get_transforms_segmentation(
        img_size=256)

   # Create an instance of the VideoDataset
    dataset = VideoDataset(
        folder_paths=[video_folder_path],
        frames_per_clip=16,
        frame_step=4,
        num_clips=10,
        output_resolution=(250, 250),
        num_workers=50,
        flatten_clips_to_frames=False,
        transform=train_transform
    )

    # Load and print stats for 3 videos for demonstration
    num_videos_to_print_stats = 10
    for i in range(min(num_videos_to_print_stats, len(dataset))):
        start_time = time.time()
        video_data, masks, frames_positions = dataset[i]
        end_time = time.time()
        print(f"Stats for video {i+1}/{num_videos_to_print_stats}:")
        print(
            f"  Time taken to load video: {end_time - start_time:.2f} seconds")
        print(f"  frames positions in returned clip: {frames_positions}")
        print(f"  Shape of video data: {video_data.shape}")
        print(f"  Data type of video data: {video_data.dtype}")
        print(f"Finished processing video {i+1}/{num_videos_to_print_stats}")

    print("Completed video stats test.")
