# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test with:
    python -m videoseal.augmentation.video
"""

import io
import av
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F


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
        self.threads = 10  # limit the number of threads to avoid memory issues
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
                'x265-params': 'log_level=none',  # Disable x265 logging
            }
            for frame_arr in frames:
                frame = av.VideoFrame.from_ndarray(frame_arr, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        buffer.seek(0)
        # file_size = buffer.getbuffer().nbytes
        # print(f"Compressed video size - crf:{self.crf} - {file_size / 1e6:.2f} MB")
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

        # if width or height is not divisible by 2, pad the frames, as some codecs require even dimensions
        if frames.shape[2] % 2 != 0 or frames.shape[3] % 2 != 0:
            frames = nn.functional.pad(frames, (0, frames.shape[3] % 2, 0, frames.shape[2] % 2), mode='constant', value=0)
            if mask is not None:
                mask = nn.functional.pad(mask, (0, mask.shape[3] % 2, 0, mask.shape[2] % 2), mode='constant', value=0)

        input_frames = self._preprocess_frames(frames)  # convert to np.uint8
        with io.BytesIO() as buffer:
            buffer = self._compress_frames(buffer, input_frames)
            output_frames = self._decompress_frames(buffer)
        output_frames = self._postprocess_frames(output_frames) 
        output_frames = output_frames.to(frames.device)

        compressed_frames = frames + (output_frames - frames).detach()
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
    def __init__(self, min_crf=None, max_crf=None, fps=24):
        super(H264, self).__init__(
            codec='libx264', fps=fps)
        self.min_crf = min_crf
        self.max_crf = max_crf

    def get_random_crf(self):
        if self.min_crf is None or self.max_crf is None:
            raise ValueError("min_crf and max_crf must be provided")
        return torch.randint(self.min_crf, self.max_crf + 1, size=(1,)).item()

    def forward(self, frames, mask=None, crf=None) -> torch.Tensor:
        crf = crf or self.get_random_crf()
        output, mask = super().forward(frames, mask, crf)
        return output, mask

    def __repr__(self) -> str:
        return f"H264"

class H264rgb(VideoCompression):
    def __init__(self, min_crf=None, max_crf=None, fps=24):
        super(H264rgb, self).__init__(
            codec='libx264rgb', fps=fps)
        self.min_crf = min_crf
        self.max_crf = max_crf

    def get_random_crf(self):
        if self.min_crf is None or self.max_crf is None:
            raise ValueError("min_crf and max_crf must be provided")
        return torch.randint(self.min_crf, self.max_crf + 1, size=(1,)).item()

    def forward(self, frames, mask=None, crf=None) -> torch.Tensor:
        crf = crf or self.get_random_crf()
        output, mask = super().forward(frames, mask, crf)
        return output, mask

    def __repr__(self) -> str:
        return f"H264rgb"


class H265(VideoCompression):
    def __init__(self, min_crf=None, max_crf=None, fps=24):
        super(H265, self).__init__(
            codec='libx265', fps=fps)
        self.min_crf = min_crf
        self.max_crf = max_crf

    def get_random_crf(self):
        if self.min_crf is None or self.max_crf is None:
            raise ValueError("min_crf and max_crf must be provided")
        return torch.randint(self.min_crf, self.max_crf + 1, size=(1,)).item()

    def forward(self, frames, mask=None, crf=None) -> torch.Tensor:
        crf = crf or self.get_random_crf()
        output, mask = super().forward(frames, mask, crf)
        return output, mask

    def __repr__(self) -> str:
        return f"H265"


class VP9(VideoCompression):
    def __init__(self, fps=24):
        super(VP9, self).__init__(
            codec='libvpx-vp9', fps=fps)
        self.crf = -1

    def forward(self, frames, mask=None, *args, **kwargs) -> torch.Tensor:
        output, mask = super().forward(frames, mask)
        return output, mask

    def __repr__(self) -> str:
        return f"VP9"


class AV1(VideoCompression):
    def __init__(self, min_crf=None, max_crf=None, fps=24):
        super(AV1, self).__init__(
            codec='libsvtav1', fps=fps)
        self.min_crf = min_crf
        self.max_crf = max_crf

    def get_random_crf(self):
        if self.min_crf is None or self.max_crf is None:
            raise ValueError("min_crf and max_crf must be provided")
        return torch.randint(self.min_crf, self.max_crf + 1, size=(1,)).item()

    def forward(self, frames, mask=None, crf=None) -> torch.Tensor:
        crf = crf or self.get_random_crf()
        output, mask = super().forward(frames, mask, crf)
        return output, mask
    
    def __repr__(self) -> str:
        return f"AV1"


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


class SpeedChange(nn.Module):
    """
    Changes the speed of the video by duplicating or dropping frames.
    
    Attributes:
        min_speed (float): Minimum speed factor (values < 1 slow down the video).
        max_speed (float): Maximum speed factor (values > 1 speed up the video).
    """
    
    def __init__(self, min_speed=0.5, max_speed=1.5):
        super(SpeedChange, self).__init__()
        self.min_speed = min_speed
        self.max_speed = max_speed
    
    def get_random_speed(self):
        """Randomly select a speed factor within the specified range."""
        if self.min_speed is None or self.max_speed is None:
            raise ValueError("min_speed and max_speed must be provided")
        return random.uniform(self.min_speed, self.max_speed)
    
    def forward(self, frames, mask=None, speed_factor=None, *args, **kwargs):
        """
        Parameters:
            frames (torch.Tensor): Video frames as a tensor with shape (T, C, H, W).
            mask (torch.Tensor): Optional mask for the video frames.
            speed_factor (float): Specific speed factor to use. If None, a random one is selected.
        
        Returns:
            torch.Tensor: Video frames with adjusted speed, with shape (T, C, H, W).
        """
        num_frames = frames.shape[0]
        
        # Use provided speed factor or get a random one
        speed_factor = speed_factor if speed_factor is not None else self.get_random_speed()
        
        if speed_factor == 1.0:
            # No change
            return frames, mask
        
        # Calculate new indices
        if speed_factor < 1.0:  # Slow down (duplicate frames)
            indices = torch.linspace(0, num_frames - 1, int(num_frames / speed_factor))
            indices = indices.round().long().clamp(0, num_frames - 1)
        else:  # Speed up (skip frames)
            indices = torch.linspace(0, num_frames - 1, int(num_frames * speed_factor))
            indices = indices[:num_frames].round().long().clamp(0, num_frames - 1)
            
        # Use the indices to create the new frames
        new_frames = frames[indices]
        new_mask = mask[indices] if mask is not None else None
        
        return new_frames, new_mask

    def __repr__(self) -> str:
        return f"SpeedChange(min_speed={self.min_speed}, max_speed={self.max_speed})"


class TemporalReorder(nn.Module):
    """
    Randomly reorders small chunks of frames to create temporal discontinuities.
    
    Attributes:
        min_chunk_size (int): Minimum size of frame chunks to reorder.
        max_chunk_size (int): Maximum size of frame chunks to reorder.
        reorder_prob (float): Probability of reordering chunks.
    """
    
    def __init__(self, min_chunk_size=2, max_chunk_size=5, reorder_prob=0.5):
        super(TemporalReorder, self).__init__()
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.reorder_prob = reorder_prob
    
    def get_random_chunk_size(self):
        """Randomly select a chunk size within the specified range."""
        if self.min_chunk_size is None or self.max_chunk_size is None:
            raise ValueError("min_chunk_size and max_chunk_size must be provided")
        return random.randint(self.min_chunk_size, self.max_chunk_size)
    
    def forward(self, frames, mask=None, chunk_size=None, swap_probability=None, *args, **kwargs):
        """
        Parameters:
            frames (torch.Tensor): Video frames as a tensor with shape (T, C, H, W).
            mask (torch.Tensor): Optional mask for the video frames.
            chunk_size (int): Specific chunk size to use. If None, a random one is selected.
            swap_probability (float): Specific swap probability. If None, the preset reorder_prob is used.
        
        Returns:
            torch.Tensor: Video frames with reordered chunks, shape (T, C, H, W).
        """
        num_frames = frames.shape[0]
        
        # Use provided chunk size or get a random one
        chunk_size = chunk_size if chunk_size is not None else self.get_random_chunk_size()
        swap_probability = swap_probability if swap_probability is not None else self.reorder_prob
        
        # If video is too short, return original
        if num_frames < chunk_size * 2:
            return frames, mask
        
        # Calculate number of chunks
        num_chunks = num_frames // chunk_size
        
        # Only use complete chunks
        usable_frames = num_chunks * chunk_size
        
        # Reshape into chunks
        frame_chunks = frames[:usable_frames].view(num_chunks, chunk_size, *frames.shape[1:])
        
        # Create reordering indices for chunks
        chunk_indices = list(range(num_chunks))
        
        # Decide which chunks to reorder
        chunks_to_reorder = []
        for i in range(0, num_chunks-1, 2):
            if random.random() < swap_probability and i+1 < num_chunks:
                chunks_to_reorder.append(i)
        
        # Reorder selected chunk pairs
        for i in chunks_to_reorder:
            chunk_indices[i], chunk_indices[i+1] = chunk_indices[i+1], chunk_indices[i]
        
        # Apply reordering
        reordered_chunks = frame_chunks[chunk_indices]
        
        # Combine chunks back into a sequence
        reordered_frames = reordered_chunks.reshape(-1, *frames.shape[1:])
        
        # Add back any remaining frames
        if usable_frames < num_frames:
            reordered_frames = torch.cat([reordered_frames, frames[usable_frames:]], dim=0)
        
        # Handle mask if provided
        if mask is not None:
            if usable_frames < num_frames:
                mask_chunks = mask[:usable_frames].view(num_chunks, chunk_size, *mask.shape[1:])
                reordered_mask = mask_chunks[chunk_indices].reshape(-1, *mask.shape[1:])
                reordered_mask = torch.cat([reordered_mask, mask[usable_frames:]], dim=0)
            else:
                mask_chunks = mask.view(num_chunks, chunk_size, *mask.shape[1:])
                reordered_mask = mask_chunks[chunk_indices].reshape(-1, *mask.shape[1:])
            return reordered_frames, reordered_mask
        
        return reordered_frames, mask

    def __repr__(self) -> str:
        return f"TemporalReorder(min_chunk_size={self.min_chunk_size}, max_chunk_size={self.max_chunk_size}, reorder_prob={self.reorder_prob})"


class WindowAveraging(nn.Module):
    """
    Applies temporal averaging over a sliding window of frames.
    
    This simulates motion blur or interpolated frames often seen in 
    lower quality video processing.
    
    Attributes:
        min_window_size (int): Minimum size of the averaging window.
        max_window_size (int): Maximum size of the averaging window.
        min_alpha (float): Minimum blend strength.
        max_alpha (float): Maximum blend strength.
    """
    
    def __init__(self, min_window_size=2, max_window_size=5, min_alpha=0.3, max_alpha=0.7):
        super(WindowAveraging, self).__init__()
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
    
    def get_random_window_size(self):
        """Randomly select a window size within the specified range."""
        if self.min_window_size is None or self.max_window_size is None:
            raise ValueError("min_window_size and max_window_size must be provided")
        return random.randint(self.min_window_size, self.max_window_size)
    
    def get_random_alpha(self):
        """Randomly select an alpha blend value within the specified range."""
        if self.min_alpha is None or self.max_alpha is None:
            raise ValueError("min_alpha and max_alpha must be provided")
        return random.uniform(self.min_alpha, self.max_alpha)
    
    def forward(self, frames, mask=None, window_size=None, alpha=None, *args, **kwargs):
        """
        Parameters:
            frames (torch.Tensor): Video frames as a tensor with shape (T, C, H, W).
            mask (torch.Tensor): Optional mask for the video frames.
            window_size (int): Specific window size to use. If None, a random one is selected.
            alpha (float): Specific alpha blend value to use. If None, a random one is selected.
        
        Returns:
            torch.Tensor: Averaged video frames as a tensor with shape (T, C, H, W).
        """
        num_frames = frames.shape[0]
        
        # If the video is too short, return the original
        if num_frames <= self.min_window_size:
            return frames, mask
        
        # Use provided parameters or get random ones
        window_size = window_size if window_size is not None else self.get_random_window_size()
        window_size = min(window_size, num_frames)  # Ensure window size is not larger than video
        alpha = alpha if alpha is not None else self.get_random_alpha()
        
        # Create output frames with the same size as input
        output_frames = frames.clone()
        
        # Apply sliding window averaging
        for i in range(num_frames):
            # Calculate window start and end, handling boundary conditions
            half_window = window_size // 2
            window_start = max(0, i - half_window)
            window_end = min(num_frames, i + half_window + 1)
            
            # Extract window frames
            window_frames = frames[window_start:window_end]
            
            # Compute average of window frames
            window_avg = torch.mean(window_frames, dim=0, keepdim=True).squeeze(0)
            
            # Blend original frame with window average
            output_frames[i] = (1 - alpha) * frames[i] + alpha * window_avg
        
        return output_frames, mask

    def __repr__(self) -> str:
        return f"WindowAveraging(min_window={self.min_window_size}, max_window={self.max_window_size})"


class DropFrame(nn.Module):
    """
    It randomly replaces random frames with the adjacent ones.

    Attributes:
        drop_frame_prob (float): Probability that a frame is replaced by its neighbour.
    """

    def __init__(self, drop_frame_prob=0.125):
        super(DropFrame, self).__init__()
        self.drop_frame_prob = drop_frame_prob

    def get_random_drop_prob(self):
        """Return the drop frame probability."""
        return self.drop_frame_prob

    def forward(self, frames, mask=None, drop_prob=None, *args, **kwargs) -> torch.Tensor:
        """
        Parameters:
            frames (torch.Tensor): Video frames as a tensor with shape (T, C, H, W).
            mask (torch.Tensor): Optional mask for the video frames.
            drop_prob (float): Specific drop probability. If None, the preset drop_frame_prob is used.
        Returns:
            torch.Tensor: Video frames as a tensor with shape (T, C, H, W).
        """
        drop_prob = drop_prob if drop_prob is not None else self.drop_frame_prob
        
        output = frames.clone()
        for i in range(len(frames)):
            if random.random() >= drop_prob:
                continue

            diff_ = -1 if random.random() < 0.5 else 1
            new_i = (i + diff_) % len(frames)
            output[i] = frames[new_i]
        return output, mask

    def __repr__(self) -> str:
        return f"DropFrame(prob={self.drop_frame_prob})"


if __name__ == "__main__":    
    import os
    import time
    from videoseal.data.loader import load_video
    from torchvision.utils import save_image
    import torchvision

    vid_o = 'assets/videos/sa-v/sav_013754.mp4'
    print("> test compression")

    vid_o = load_video(vid_o)
    vid_o = vid_o[:60]  # Use only the first 60 frames

    # Create the output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # # h264, h264rgb, h265, vp9, av1 (slow)
    # # for codec in ['libx264', 'libx264rgb', 'libx265', 'libvpx-vp9', 'libaom-av1']:
    # for codec in ['libx264', 'libx264rgb', 'libx265', 'libvpx-vp9']:
    #     crfs = [28, 34, 40, 46] if codec not in ['libvpx-vp9'] else [-1]
    #     for crf in crfs:
    #         try:
    #             compressor = VideoCompression(codec=codec, crf=crf)
    #             start = time.time()
    #             compressed_frames, _ = compressor(vid_o)
    #             end = time.time()
    #             mse = torch.nn.functional.mse_loss(vid_o, compressed_frames)
    #             print(f"Codec: {codec}, CRF: {crf} - MSE: {mse:.2e} - Time: {end - start:.2f}s")

    #             # Save first, middle and last frame of both original and compressed video
    #             indices = [0, len(vid_o)//2, -1]
    #             for idx in indices:
    #                 # Create filename
    #                 filename = f"{codec.replace('lib', '')}_crf_{crf}_frame_{idx}.png"
    #                 # Stack original and compressed frame side by side
    #                 comparison = torch.cat([vid_o[idx], compressed_frames[idx]], dim=2)
    #                 # Save the comparison image
    #                 save_image(comparison.clamp(0, 1), os.path.join(output_dir, filename))
    #                 print(f"Saved comparison frame {idx} to:", os.path.join(output_dir, filename))

    #         except Exception as e:
    #             print(f":warning: An error occurred with {codec}: {str(e)}")

    # Function to save video tensor as MP4
    def save_video(frames, path):
        # Convert tensor to numpy array of uint8 (0-255)
        frames_np = (frames.clamp(0, 1) * 255).to(torch.uint8)
        frames_np = frames_np.permute(0, 2, 3, 1)  # T,C,H,W -> T,H,W,C
        
        # Save using torchvision's write_video
        torchvision.io.write_video(path, frames_np.cpu(), fps=24, video_codec='h264')
        print(f"Saved video to: {path}")

    # Save the original video
    save_video(vid_o, os.path.join(output_dir, "original.mp4"))

    # Test the new augmentations
    print("\n> Testing new augmentations")
    
    # Create instances of the augmenters
    augmenters = {
        "SpeedChange": SpeedChange(min_speed=0.7, max_speed=1.3),
        "TemporalReorder": TemporalReorder(min_chunk_size=3, max_chunk_size=6, reorder_prob=0.7),
        "DropFrame": DropFrame(drop_frame_prob=0.2),
        "WindowAveraging": WindowAveraging(min_window_size=2, max_window_size=5, min_alpha=0.3, max_alpha=0.7)
    }
    
    # Test each augmenter with both random and specific parameters
    for name, augmenter in augmenters.items():
        try:
            # Test with random parameters
            start = time.time()
            augmented_frames_random, _ = augmenter(vid_o.clone())
            end = time.time()
            
            print(f"Augmenter: {name} (random params) - Time: {end - start:.2f}s")
            
            # Save the randomly augmented video
            save_video(augmented_frames_random, os.path.join(output_dir, f"{name}_random.mp4"))
            
            # Test with specific parameters
            specific_params = {}
            if name == "SpeedChange":
                specific_params = {"speed_factor": 2.0}
            elif name == "TemporalReorder":
                specific_params = {"chunk_size": 4, "swap_probability": 0.8}
            elif name == "DropFrame":
                specific_params = {"drop_prob": 0.3}
            elif name == "WindowAveraging":
                specific_params = {"window_size": 3, "alpha": 1.0}
                
            start = time.time()
            augmented_frames_specific, _ = augmenter(vid_o.clone(), **specific_params)
            end = time.time()
            
            print(f"Augmenter: {name} (specific params: {specific_params}) - Time: {end - start:.2f}s")
            
            # Save the specifically augmented video
            save_video(augmented_frames_specific, os.path.join(output_dir, f"{name}_specific.mp4"))
            
            # Save comparison frames for both random and specific parameters
            indices = [0, len(vid_o)//2, -1]
            for idx in indices:
                if idx < len(augmented_frames_random):
                    # Random parameters
                    filename = f"{name}_random_frame_{idx}.png"
                    comparison = torch.cat([vid_o[idx], augmented_frames_random[idx]], dim=2)
                    save_image(comparison.clamp(0, 1), os.path.join(output_dir, filename))
                    
                    # Specific parameters
                    filename = f"{name}_specific_frame_{idx}.png"
                    comparison = torch.cat([vid_o[idx], augmented_frames_specific[idx]], dim=2)
                    save_image(comparison.clamp(0, 1), os.path.join(output_dir, filename))
                    
                    print(f"Saved comparison frames {idx} for {name}")
        
        except Exception as e:
            print(f":warning: An error occurred with {name}: {str(e)}")
    
    # Test combinations of augmentations with specific parameters
    print("\n> Testing combinations of augmentations with specific parameters")
    combined_frames = vid_o.clone()
    
    # Define specific parameters for each augmentation
    specific_params_dict = {
        "SpeedChange": {"speed_factor": [2.0]},
        "TemporalReorder": {"chunk_size": 5},
        "DropFrame": {"drop_prob": 0.4},
        "WindowAveraging": {"window_size": 4, "alpha": 1.0}
    }
    
    # Apply augmentations sequentially with specific parameters
    for name, augmenter in augmenters.items():
        try:
            params = specific_params_dict.get(name, {})
            combined_frames, _ = augmenter(combined_frames, **params)
            print(f"Applied {name} with parameters: {params}")
        except Exception as e:
            print(f":warning: Failed to apply {name}: {str(e)}")
    
    # Save the combined augmented video
    save_video(combined_frames, os.path.join(output_dir, "combined_augmentations.mp4"))
    
    # Save combined result frames
    indices = [0, len(vid_o)//2, -1]
    for idx in indices:
        if idx < len(combined_frames):
            filename = f"combined_specific_augmentations_frame_{idx}.png"
            comparison = torch.cat([vid_o[idx], combined_frames[idx]], dim=2)
            save_image(comparison.clamp(0, 1), os.path.join(output_dir, filename))
            print(f"Saved combined specific augmentation frame {idx} to:", os.path.join(output_dir, filename))
