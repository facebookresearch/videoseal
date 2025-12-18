# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
To run
    python -m videoseal.evals.metrics
"""

import os
import math
import subprocess
import tempfile
import re
import numpy as np
from scipy import stats, interpolate

import torch
import pytorch_msssim

def psnr(x, y, is_video=False):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
        is_video: If True, the PSNR is computed over the entire batch, not on each image separately
    """
    delta = 255 * (x - y)
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    peak = 20 * math.log10(255.0)
    avg_on_dims = (0,1,2,3) if is_video else (1,2,3)
    noise = torch.mean(delta**2, dim=avg_on_dims)
    psnr = peak - 10*torch.log10(noise)
    return psnr

def ssim(x, y, data_range=1.0):
    """
    Return SSIM
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    return pytorch_msssim.ssim(x, y, data_range=data_range, size_average=False)

def msssim(x, y, data_range=1.0):
    """
    Return MSSSIM
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    return pytorch_msssim.ms_ssim(x, y, data_range=data_range, size_average=False)

def linf(x, y, data_range=1.0):
    """
    Return L_inf in pixel space (integer between 0 and 255)
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    multiplier = 255.0 / data_range
    return torch.max(torch.abs(x - y)) * multiplier
    
def iou(preds, targets, threshold=0.0, label=1):
    """
    Return IoU for a specific label (0 or 1).
    Args:
        preds (torch.Tensor): Predicted masks with shape Bx1xHxW
        targets (torch.Tensor): Target masks with shape Bx1xHxW
        label (int): The label to calculate IoU for (0 for background, 1 for foreground)
        threshold (float): Threshold to convert predictions to binary masks
    """
    preds = preds > threshold  # Bx1xHxW
    targets = targets > 0.5
    if label == 0:
        preds = ~preds
        targets = ~targets
    intersection = (preds & targets).float().sum((1,2,3))  # B
    union = (preds | targets).float().sum((1,2,3))  # B
    # avoid division by zero
    union[union == 0.0] = intersection[union == 0.0] = 1
    iou = intersection / union
    return iou

def accuracy(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    threshold: float = 0.0
) -> torch.Tensor:
    """
    Return accuracy
    Args:
        preds (torch.Tensor): Predicted masks with shape Bx1xHxW
        targets (torch.Tensor): Target masks with shape Bx1xHxW
    """
    preds = preds > threshold  # b 1 h w
    targets = targets > 0.5
    correct = (preds == targets).float()  # b 1 h w
    accuracy = torch.mean(correct, dim=(1,2,3))  # b
    return accuracy

def pvalue(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Return p values
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        mask (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
    """
    nbits = targets.shape[-1]
    bit_accs = bit_accuracy(preds, targets, mask, threshold)  # b
    pvalues = [stats.binomtest(int(p*nbits), nbits, 0.5, alternative='greater').pvalue for p in bit_accs]
    return torch.tensor(pvalues)  # b

def plogp(p: torch.Tensor) -> torch.Tensor:
    """
    Return p log p
    Args:
        p (torch.Tensor): Probability tensor with shape BxK
    """
    plogp = p * torch.log2(p)
    plogp[p == 0] = 0
    return plogp

def capacity(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Return normalized bit accuracy, defined as the capacity of the nbits channels,
    in the case of a binary symmetric channel of error probability being the bit. acc.
    """
    nbits = targets.shape[-1]
    bit_accs = bit_accuracy(preds, targets, mask, threshold)  # b
    entropy = - plogp(bit_accs) - plogp(1-bit_accs)
    capacity = 1 - entropy
    capacity = nbits * capacity
    return capacity

def bit_accuracy(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Return bit accuracy
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        mask (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    preds = preds > threshold  # b k ...
    if preds.dim() == 4:  # bit preds are pixelwise
        bsz, nbits, h, w = preds.size()
        if mask is not None:
            mask = mask.expand_as(preds).bool()
            preds = preds.masked_select(mask).view(bsz, nbits, -1)  # b k n
            preds = preds.mean(dim=-1, dtype=float)  # b k
        else:
            preds = preds.mean(dim=(-2, -1), dtype=float) # b k
    preds = preds > 0.5  # b k
    targets = targets > 0.5  # b k
    correct = (preds == targets).float()  # b k
    bit_acc = torch.mean(correct, dim=-1)  # b
    return bit_acc

def bit_accuracy_1msg(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    masks: torch.Tensor = None,
    threshold: float = 0.0
) -> torch.Tensor:
    """
    Computes the bit accuracy for each pixel, then averages over all pixels.
    Better for "k-bit" evaluation during training since it's independent of detection performance.
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    preds = preds > threshold  # b k h w
    targets = targets > 0.5  # b k
    correct = (preds == targets.unsqueeze(-1).unsqueeze(-1)).float()  # b k h w
    if masks is not None:  
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(correct).bool()
        correct_list = [correct[i].masked_select(masks[i]) for i in range(len(masks))]
        bit_acc = torch.tensor([torch.mean(correct_list[i]).item() for i in range(len(correct_list))])
    else:
        bit_acc = torch.mean(correct, dim=(1,2,3))  # b
    return bit_acc

def bit_accuracy_inference(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    masks: torch.Tensor,
    method: str = 'hard',
    threshold: float = 0.0
) -> torch.Tensor:
    """
    Computes the message by averaging over all pixels, then computes the bit accuracy.
    Closer to how the model is evaluated during inference.
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
        method (str): Method to compute bit accuracy. Options: 'hard', 'soft'
    """
    if method == 'hard':
        # convert every pixel prediction to binary, select based on masks, and average
        preds = preds > threshold  # b k h w
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == 'semihard':
        # select every pixel prediction based on masks, and average
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == 'soft':
        # average every pixel prediction, use masks "softly" as weights for averaging
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(preds)  # b k h w
        preds = torch.sum(preds * masks, dim=(2,3)) / torch.sum(masks, dim=(2,3))  # b k
    preds = preds > 0.5  # b k
    targets = targets > 0.5  # b k
    correct = (preds == targets).float()  # b k
    bit_acc = torch.mean(correct, dim=(1))  # b
    return bit_acc

def bit_accuracy_mv(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    masks: torch.Tensor = None,
    threshold: float = 0.0
) -> torch.Tensor:
    """
    (Majority vote)
    Return bit accuracy
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    preds = preds > threshold  # b k h w
    targets = targets > 0.5  # b k
    correct = (preds == targets.unsqueeze(-1).unsqueeze(-1)).float()  # b k h w
    if masks is not None:  
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(correct).bool()
        preds = preds.masked_select(masks).view(bsz, nbits, -1)  # b k n
        # correct = correct.masked_select(masks).view(bsz, nbits, -1)  # b k n
        # correct = correct.unsqueeze(-1)  # b k n 1
    # Perform majority vote for each bit
    preds_majority, _ = torch.mode(preds, dim=-1)  # b k
    # Compute bit accuracy
    correct = (preds_majority == targets).float()  # b k
    # bit_acc = torch.mean(correct, dim=(1,2,3))  # b
    bit_acc = torch.mean(correct, dim=-1)  # b
    return bit_acc

def tensor_to_video(
        tensor, filename, fps, 
        codec=None, crf=23,
        ffmpeg_bin: str = '/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg',
    ):
    """ Saves a video tensor into a video file."""
    T, C, H, W = tensor.shape
    assert C == 3, "Video must have 3 channels (RGB)."
    video_data = (tensor * 255).to(torch.uint8).numpy()
    
    # write video using ffmpeg
    if codec is None:
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
            video_data.tofile(temp_file.name)
            temp_filename = temp_file.name
        command = [
            f'{ffmpeg_bin}', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
            '-r', f'{fps}', '-i', temp_filename, filename
        ]
    elif codec == 'libx264':
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
            video_data.tofile(temp_file.name)
            temp_filename = temp_file.name
        command = [
            f'{ffmpeg_bin}', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
            '-r', f'{fps}', '-i', temp_filename,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', f'{crf}',
            filename
        ]
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(temp_filename)

def vmaf_on_file(
    vid_o: str,
    vid_w: str,
    ffmpeg_bin: str = '/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg',
) -> float:
    """
    Runs `ffmpeg -i vid_o.mp4 -i vid_w.mp4 -filter_complex libvmaf` and returns the score.
    """
    # Execute the command and capture the output to get the VMAF score
    command = [
        ffmpeg_bin,
        '-i', vid_o,
        '-i', vid_w,
        '-lavfi', 'libvmaf=\'n_threads=8\'',
        '-f', 'null', '-'
    ]
    result = subprocess.run(command, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    vmaf_score = None
    for line in result.stderr.split('\n'):
        if "VMAF score:" in line:
            # numerical part of the VMAF score with regex
            match = re.search(r"VMAF score: ([0-9.]+)", line)
            if match:
                vmaf_score = float(match.group(1))
                break
    return vmaf_score

def vmaf_on_tensor(
    tensor1, 
    tensor2=None, 
    fps=24, codec='libx264', crf=23, 
    return_aux=False
):
    """
    Compute VMAF between original and compressed/watermarked video tensors.
    Args:
        tensor1: Original video tensor
        tensor2: Compressed/Watermarked video tensor. \
            If None, the original tensor is used as the second tensor, but with the codec and crf specified.
        fps: Frames per second
        codec: Codec to use when saving the video to file
        crf: Constant Rate Factor for libx264 codec
        return_aux: Return additional information
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4') as file1, \
         tempfile.NamedTemporaryFile(suffix='.mp4') as file2:
        
        # save tensors to video files
        if tensor2 is None:
            tensor_to_video(tensor1, file1.name, fps=fps, codec=None)
            tensor2 = tensor1
        else:
            tensor_to_video(tensor1, file1.name, fps=fps, codec=codec, crf=crf)
        tensor_to_video(tensor2, file2.name, fps=fps, codec=codec, crf=crf)
        
        # compute VMAF
        vmaf_score = vmaf_on_file(file1.name, file2.name)

        # aux info
        MB = 1024 * 1024
        filesize1 = os.path.getsize(file1.name) / MB
        filesize2 = os.path.getsize(file2.name) / MB
        duration1 = len(tensor1) / fps
        duration2 = len(tensor2) / fps
        bps1 = filesize1 / duration1
        bps2 = filesize2 / duration2
        aux = {
            'filesize1': filesize1,
            'filesize2': filesize2,
            'duration1': duration1,
            'duration2': duration2,
            'bps1': bps1,
            'bps2': bps2
        }

        # return
        if return_aux:
            return vmaf_score, aux
        return vmaf_score

def bd_rate(R1, PSNR1, R2, PSNR2, piecewise=0):
    """
    Almost copy paste from https://github.com/Anserw/Bjontegaard_metric/blob/master/bjontegaard_metric.py
    """
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapezoid(v1, dx=interval)
        int2 = np.trapezoid(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff


if __name__ == '__main__':
    # Test the PSNR function
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)
    print("> test psnr")
    try:
        print("OK!", psnr(x, y))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Test the IoU function
    preds = torch.rand(1, 1, 256, 256)
    targets = torch.rand(1, 1, 256, 256)
    print("> test iou")
    try:
        print("OK!", iou(preds, targets))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Test the accuracy function
    preds = torch.rand(1, 1, 256, 256)
    targets = torch.rand(1, 1, 256, 256)
    print("> test accuracy")
    try:
        print("OK!", accuracy(preds, targets))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Load
    from videoseal.data.loader import load_video
    filename_o = 'assets/videos/sav_013754.mp4'
    filename_w = 'assets/videos/sav_013754.mp4'
    vid_o = load_video(filename_o)
    vid_w = load_video(filename_w)

    # Test the vmaf function
    print("> test vmaf")
    try:
        result = vmaf_on_file(filename_o, filename_w)
        if result is not None:
            print("OK!", result)
        else:
            raise Exception("VMAF score not found in the output.")
    except Exception as e:
        print(f"!!! An error occurred: {str(e)}")
        print(f"Try checking that ffmpeg is installed and that vmaf is available.")

    # Test the vmaf function on tensors
    print("> test vmaf on tensor")
    try:
        result = vmaf_on_tensor(vid_o, vid_w, return_aux=True, codec='libx264')
        if result is not None:
            print("OK!", result)
        else:
            raise Exception("VMAF score not found in the output.")
    except Exception as e:
        print(f"!!! An error occurred: {str(e)}")
        print(f"Try checking that ffmpeg is installed and that vmaf is available.")
    
    # Test the vmaf function on single tensor
    print("> test vmaf on tensor")
    vid_o = vid_o[:24*2]
    try:
        for crf in [23, 28, 33]:
            result = vmaf_on_tensor(vid_o, return_aux=True, codec='libx264', crf=crf)
            if result is not None:
                print(f"crf={crf}", result)
            else:
                raise Exception("VMAF score not found in the output.")
    except Exception as e:
        print(f"!!! An error occurred: {str(e)}")
        print(f"Try checking that ffmpeg is installed and that vmaf is available.")
