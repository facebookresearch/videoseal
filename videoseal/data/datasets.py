# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Video dataset adapted from https://github.com/facebookresearch/jepa/blob/main/src/datasets/video_dataset.py

import functools
import glob
import json
import logging
import os
import random
import warnings

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm

from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools import mask as mask_utils

from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision.datasets.folder import default_loader, is_image_file
from torchvision.transforms import ToTensor

try:
    from decord import VideoReader, cpu
    decord_available = True
except ImportError:
    VideoReader = None
    decord_available = False

from ..utils import suppress_output
from ..utils.data import LRUDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@functools.lru_cache()
def get_image_paths(path):
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])


class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path,  transform=None, mask_transform=None):
        # assuming 'path' is a folder of image files path and
        # 'annotation_path' is the base path for corresponding annotation json files
        self.samples = get_image_paths(path)
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = ToTensor()(img)

        if self.transform:
            img = self.transform(img)

        # Get MASKS
        mask = torch.ones_like(img[0:1, ...])

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        
        return img, mask

    def __len__(self):
        return len(self.samples) 


class CocoImageIDWrapper(CocoDetection):
    def __init__(
        self, root, annFile, transform=None, mask_transform=None,
        random_nb_object=True, max_nb_masks=4, multi_w=False
    ) -> None:
        """
        Args:
            root (str): Root directory where images are saved.
            annFile (str): Path to json annotation file.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            mask_transform (callable, optional): The same as transform but for the mask.
            random_nb_object (bool, optional): If True, randomly sample the number of objects in the image. Defaults to True.
            max_nb_masks (int, optional): Maximum number of masks to return. Defaults to 4.
            multi_w (bool, optional): If True, return multiple masks as a single tensor. Defaults to False.
        """
        with suppress_output():
            super().__init__(root, annFile, transform=transform, target_transform=mask_transform)
        self.random_nb_object = random_nb_object
        self.max_nb_masks = max_nb_masks
        self.multi_w = multi_w

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, int):
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        img = self._load_image(id)
        mask = self._load_mask(id)
        if mask is None:
            return None  # Skip this image if no valid mask is available

        # convert PIL to tensor
        img = ToTensor()(img)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def _load_mask(self, id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        if not anns:
            return None  # Return None if there are no annotations

        img_info = self.coco.loadImgs(id)[0]
        original_height = img_info['height']
        original_width = img_info['width']

        # Initialize a list to hold all masks
        masks = []
        if self.random_nb_object and np.random.rand() < 0.5:
            random.shuffle(anns)
            anns = anns[:np.random.randint(1, len(anns)+1)]
        if not (self.multi_w):
            mask = np.zeros((original_height, original_width),
                            dtype=np.float32)
            # one mask for all objects
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                mask = np.maximum(mask, m)
            mask = torch.tensor(mask, dtype=torch.float32)
            return mask[None, ...]  # Add channel dimension
        else:
            anns = anns[:self.max_nb_masks]
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                masks.append(m)
            # Stack all masks along a new dimension to create a multi-channel mask tensor
            if masks:
                masks = np.stack(masks, axis=0)
                masks = torch.tensor(masks, dtype=torch.bool)
                # Check if the number of masks is less than max_nb_masks
                if masks.shape[0] < self.max_nb_masks:
                    # Calculate the number of additional zero masks needed
                    additional_masks_count = self.max_nb_masks - masks.shape[0]
                    # Create additional zero masks
                    additional_masks = torch.zeros(
                        (additional_masks_count, original_height, original_width), dtype=torch.bool)
                    # Concatenate the original masks with the additional zero masks
                    masks = torch.cat([masks, additional_masks], dim=0)
            else:
                # Return a tensor of shape (max_nb_masks, height, width) filled with zeros if there are no masks
                masks = torch.zeros(
                    (self.max_nb_masks, original_height, original_width), dtype=torch.bool)
            return masks


class VideoDataset(Dataset):
    """Video classification dataset that loads video files directly from specified folders."""

    def __init__(
        self,
        # List of folder paths containing .mp4 video files
        folder_paths: list[str],
        datasets_weights: list[float] = None,
        frames_per_clip: int = 16,  # Number of frames in each video clip
        frame_step: int = 4,  # Step size between frames within a clip
        num_clips: int = 1,  # Number of clips to sample from each video
        # Optional transformation function to be applied to each clip
        transform: callable = None,
        mask_transform: callable = None,
        # Optional transformation function applied on the video before clipping
        shared_transform: callable = None,
        # If True, sample clips randomly inside the video
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False,  # If True, allow clips to overlap
        # If True, exclude videos that are shorter than the required clip length
        filter_short_videos: bool = False,
        # Maximum allowed video file size in bytes
        filter_long_videos: int | float = int(10**9),
        # Optional, specific duration in seconds for each clip
        duration: float = None,
        output_resolution: tuple | int = (
            256, 256),  # Desired output resolution
        num_workers: int = 1,  # numbers of cpu to run the preprocessing of each batch
        subsample_frames: bool = True  # if set to false return full video
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
        self.subsample_frames = subsample_frames

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

            for video_file in tqdm.tqdm(video_files,
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
        self.video_buffer = LRUDict(maxsize=150)

    def __getitem__(self, index):
        if self.subsample_frames:
            return self.get_clip(index)
        else:
            return self.get_vid(index)

    def get_vid(self, index):
        video_file = self.videofiles[index]
        video, mask = self.load_full_video_decord(
            video_file,
            num_workers=self.num_workers
        )
        if self.transform is not None:
            video = torch.stack([self.transform(frame) for frame in video])
        if self.mask_transform is not None:
            mask = torch.stack([self.mask_transform(one_mask)
                               for one_mask in mask])
        return video, mask

    def get_clip(self, index):
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
            # Reshape buffer back to (num_clips, frames_per_clip, channels, height, width)
            buffer = buffer.view(
                self.num_clips, self.frames_per_clip, *buffer.shape[1:])

            # Store the loaded video in the buffer
            self.video_buffer[video_file] = (buffer, frames_indices)

        # load directly from buffer here should be processed already
        buffer, frames_positions_in_clips = self.video_buffer[video_file]

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

    @staticmethod
    def load_full_video_decord(fname, num_workers=8):
        """
        Load full video content using Decord.
        Args:
            fname (str): The path to the video file.
            num_workers (int): The number of worker threads to use for video loading. Defaults to 8.
        Returns:
            tuple: A tuple containing the loaded video frames as a PyTorch tensor (Frames, H, W , C) and a mask tensor.
        Raises:
            warnings.warn: If the video file is not found or is too short.
        """
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None
        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')
            return [], None
        try:
            vr = VideoReader(
                fname, num_threads=num_workers, ctx=cpu(0))
        except Exception:
            return [], None
        vid_np = vr.get_batch(range(len(vr))).asnumpy()
        vid_np = vid_np.transpose(0, 3, 1, 2) / 255.0  # normalize to 0 - 1
        vid_pt = torch.from_numpy(vid_np).float()
        mask = torch.ones_like(vid_pt[:, 0:1, ...])
        return vid_pt, mask

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
            vr = VideoReader(fname, num_threads=self.num_workers, ctx=cpu(0))
            ori_height, ori_width = vr[0].shape[:2]
            if isinstance(self.output_resolution, int):
                if self.output_resolution == -1:  # keep original resolution
                    height = -1
                    width = -1
                else:  # keep aspect ratio
                    scale = self.output_resolution / min(ori_height, ori_width)
                    height = int(ori_height * scale)
                    width = int(ori_width * scale)
            else:
                width = self.output_resolution[1]
                height = self.output_resolution[0]
            vr = VideoReader(fname, width=width, height=height,
                             num_threads=self.num_workers, ctx=cpu(0))
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

        # returned video is between 0 and 255 now normalized to 0 - 1
        buffer = buffer / 255.0

        return buffer, clip_indices

    def __len__(self):
        return len(self.videofiles) * self.num_clips


if __name__ == "__main__":
    import time

    dataset = ImageFolder(path="/large_experiments/meres/sa-1b/anonymized_resized/valid/", annotations_folder="/datasets01/segment_anything/annotations/release_040523/")
    print(dataset[0][1])


    # Specify the path to the folder containing the MP4 files
    video_folder_path = "./assets/videos"

    from .transforms import get_resize_transform

    train_transform, train_mask_transform = get_resize_transform(img_size=256)
    val_transform, val_mask_transform = get_resize_transform(img_size=256)

   # Create an instance of the VideoDataset
    dataset = VideoDataset(
        folder_paths=[video_folder_path],
        frames_per_clip=16,
        frame_step=4,
        num_clips=10,
        output_resolution=(1250, 1250),
        num_workers=50,
        transform=train_transform
    )

    # Load and print stats for 3 videos for demonstration
    num_videos_to_print_stats = 3
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
