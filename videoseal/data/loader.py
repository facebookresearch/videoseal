# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings

import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, DistributedSampler

from ..utils.dist import is_dist_avail_and_initialized
from .datasets import CocoImageIDWrapper, ImageFolder, VideoDataset
from .transforms import default_transform


def load_video(fname, num_workers=8):
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
    return vid_pt


def get_dataloader(
    data_dir: str,
    transform: callable = default_transform,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8
) -> DataLoader:
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageFolder(data_dir, transform=transform)
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, num_workers=num_workers,
                                pin_memory=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True, drop_last=True)
    return dataloader


def custom_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])

    images, masks = zip(*batch)
    images = torch.stack(images)

    # Find the maximum number of masks in any single image
    max_masks = max(mask.shape[0] for mask in masks)
    if max_masks == 1:
        masks = torch.stack(masks)
        return images, masks

    # Pad each mask tensor to have 'max_masks' masks and add the inverse mask
    padded_masks = []
    for mask in masks:
        # Calculate the union of all masks in this image
        # Assuming mask is of shape [num_masks, H, W]
        union_mask = torch.max(mask, dim=0).values

        # Calculate the inverse of the union mask
        inverse_mask = ~union_mask

        # Pad the mask tensor to have 'max_masks' masks
        pad_size = max_masks - mask.shape[0]
        if pad_size > 0:
            padded_mask = F.pad(mask, pad=(
                0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
        else:
            padded_mask = mask

        # Append the inverse mask to the padded mask tensor
        # padded_mask = torch.cat([padded_mask, inverse_mask.unsqueeze(0)], dim=0)

        padded_masks.append(padded_mask)

    # Stack the padded masks
    masks = torch.stack(padded_masks)

    return images, masks


def get_dataloader_segmentation(
    data_dir: str,
    ann_file: str,
    transform: callable,
    mask_transform: callable,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8,
    random_nb_object=True,
    multi_w=False,
    max_nb_masks=4
) -> DataLoader:
    """ Get dataloader for COCO dataset. """
    # Initialize the CocoDetection dataset
    if "coco" in data_dir:
        dataset = CocoImageIDWrapper(root=data_dir, annFile=ann_file, transform=transform, mask_transform=mask_transform,
                                     random_nb_object=random_nb_object, multi_w=multi_w, max_nb_masks=max_nb_masks)
    else:
        dataset = ImageFolder(path=data_dir, transform=transform, mask_transform=mask_transform)

    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)

    return dataloader


def get_video_dataloader(
    data_dir: str,
    transform: callable = None,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 8,
    drop_last: bool = False,
    **dataset_kwargs
) -> DataLoader:
    """
    Get dataloader for the videos in the data_dir. The data_dir must contain .mp4 video files.

    Args:
        data_dir (str): Directory containing video files.
        transform (callable, optional): Transformation function to be applied to each video clip.
        batch_size (int): Number of videos per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses to use for data loading.
        **dataset_kwargs: Additional keyword arguments to pass to the VideoDataset constructor.

    Returns:
        DataLoader: Configured DataLoader for the video dataset.
    """
    # Update dataset_kwargs with any specific parameters
    dataset_kwargs.update({
        'folder_paths': [data_dir],
        'transform': transform
    })

    # Create an instance of the VideoDataset
    dataset = VideoDataset(num_workers=num_workers, **dataset_kwargs)
    # Check if distributed training is available and initialized
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, num_workers=num_workers,
                                pin_memory=True, drop_last=drop_last)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True, drop_last=drop_last)
    return dataloader


# Test the VideoLoader class
if __name__ == "__main__":
    # run
    # python -m videoseal.data.loader

    # Path to the directory containing the video files
    video_folder_path = "./assets/videos/"

    # Create the video dataloader to load flat frames
    video_dataloader = get_video_dataloader(
        data_dir=video_folder_path,
        frames_per_clip=16,
        frame_step=4,
        num_clips=4,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        output_resolution=(250, 250),
        flatten_clips_to_frames=True,
    )
    # Iterate through the dataloader and print stats for each batch
    for video_batch, masks_batch, frames_positions in video_dataloader:
        print(
            f"loaded a batch of {video_batch.shape} size , each consists of a frame")
        print(video_batch.shape)
        print(frames_positions)
        break

    # Create the video dataloader to load flat frames
    video_dataloader = get_video_dataloader(
        data_dir=video_folder_path,
        frames_per_clip=16,
        frame_step=4,
        num_clips=4,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        output_resolution=(250, 250),
        flatten_clips_to_frames=False,
    )
    # Iterate through the dataloader and print stats for each batch
    for video_batch, masks_batch, frames_positions in video_dataloader:
        print(
            f"loaded a batch of {video_batch.shape[0]} size , each consists of a {video_batch.shape[1]} clips")
        print(video_batch.shape)
        print(frames_positions)
        break

    print("Video dataloader test completed.")
