"""
To run
    python -m videoseal.evals.metrics
"""

import io
import math
import subprocess
import tempfile
import av
import re
import numpy as np

import torch
from pytorch_msssim import ssim as pytorch_ssim

def psnr(x, y):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    delta = 255 * (x - y)
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    peak = 20 * math.log10(255.0)
    noise = torch.mean(delta**2, dim=(1,2,3))  # B
    psnr = peak - 10*torch.log10(noise)
    return psnr

def ssim(x, y, data_range=1.0):
    """
    Return SSIM
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    return pytorch_ssim(x, y, data_range=data_range, size_average=False)


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

def bit_accuracy(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor = None,
    threshold: float = 0.0
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
            '-filter_complex', 'libvmaf',
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
    vid_o: torch.Tensor,
    vid_w: torch.Tensor,
    codec='libx264', crf=23, fps=24, 
    ffmpeg_bin: str = '/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg',
    return_aux: bool = False
) -> float:
    """
    Computes the VMAF score between two videos represented as tensors, 
    and encoded using the specified codec.

    Args:
        vid_o (torch.Tensor): Original video tensor with shape TxCxHxW with normalized values (≈ [0,1])
        vid_w (torch.Tensor): Watermarked video tensor with shape TxCxHxW with normalized values (≈ [0,1])
        codec (str): Codec to use for encoding the video
        crf (int): Constant Rate Factor for the codec
        fps (int): Frames per second of the video
    """
    vid_o = vid_o.clamp(0, 1).permute(0, 2, 3, 1)  # t c h w -> t w h c
    vid_o = (vid_o * 255).to(torch.uint8).numpy()
    # Create an in-memory bytes buffer
    buffer = io.BytesIO()
    # Create a PyAV container for output in memory
    container = av.open(buffer, mode='w', format='mp4')
    # Add a video stream to the container
    stream = container.add_stream(codec, rate=fps)
    stream.width = vid_o.shape[2]
    stream.height = vid_o.shape[1]
    stream.pix_fmt = 'yuv420p' if codec != 'libx264rgb' else 'rgb24'
    stream.options = {'crf': str(crf)}
    # Write frames to the stream
    for frame_arr in vid_o:
        frame = av.VideoFrame.from_ndarray(frame_arr, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    # Finalize the file
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    
    vid_w = vid_w.clamp(0, 1).permute(0, 2, 3, 1)  # t c h w -> t w h c
    vid_w = (vid_w * 255).to(torch.uint8).numpy()
    # Create an in-memory bytes buffer
    buffer1 = io.BytesIO()
    # Create a PyAV container for output in memory
    container1 = av.open(buffer1, mode='w', format='mp4')
    # Add a video stream to the container
    stream = container1.add_stream(codec, rate=fps)
    stream.width = vid_w.shape[2]
    stream.height = vid_w.shape[1]
    stream.pix_fmt = 'yuv420p' if codec != 'libx264rgb' else 'rgb24'
    stream.options = {'crf': str(crf)}
    # Write frames to the stream
    for frame_arr in vid_w:
        frame = av.VideoFrame.from_ndarray(frame_arr, format='rgb24')
        for packet in stream.encode(frame):
            container1.mux(packet)
    # Finalize the file
    for packet in stream.encode():
        container1.mux(packet)
    container1.close()

    # Seek back to the beginning of the buffers
    buffer.seek(0)
    buffer1.seek(0)
    # Save the buffers to temporary files
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_file_o, \
         tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_file_w:
        tmp_file_o.write(buffer.read())
        tmp_file_w.write(buffer1.read())
        tmp_file_o.flush()
        tmp_file_w.flush()
        # Compute VMAF score using the temporary files
        vmaf_score = vmaf_on_file(tmp_file_o.name, tmp_file_w.name, ffmpeg_bin)
        if return_aux:
            MB = 1024 * 1024
            filesize_o = tmp_file_o.tell() / MB
            duration_o = len(vid_o) / fps
            bps_o = filesize_o / duration_o
            filesize_w = tmp_file_w.tell() / MB
            duration_w = len(vid_w) / fps
            bps_w = filesize_w / duration_w
            aux = {
                'filesize_o': filesize_o,
                'filesize_w': filesize_w,
                'duration_o': duration_o,
                'duration_w': duration_w,
                'bps_o': bps_o,
                'bps_w': bps_w
            }
            return vmaf_score, aux
    return vmaf_score

    

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

    # # # Test the vmaf function
    vid_o = 'assets/videos/sav_013754.mp4'
    vid_w = 'assets/videos/sav_013754.mp4'
    print("> test vmaf")
    try:
        result = vmaf_on_file(vid_o, vid_w)
        if result is not None:
            print("OK!", result)
        else:
            raise Exception("VMAF score not found in the output.")
    except Exception as e:
        print(f"!!! An error occurred: {str(e)}")
        print(f"Try checking that ffmpeg is installed and that vmaf is available.")

    # Test the vmaf function on tensors
    print("> test vmaf on tensor")
    from videoseal.data.loader import load_video
    vid_o = load_video(vid_o)
    vid_w = load_video(vid_w)
    vid_o = vid_o / 255
    vid_w = vid_w / 255
    try:
        result = vmaf_on_tensor(vid_o, vid_w, return_aux=True)
        if result is not None:
            print("OK!", result)
        else:
            raise Exception("VMAF score not found in the output.")
    except Exception as e:
        print(f"!!! An error occurred: {str(e)}")
        print(f"Try checking that ffmpeg is installed and that vmaf is available.")
