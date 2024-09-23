"""
python -m videoseal.evals.metrics
"""

import torch
import math
import subprocess
import re

from videoseal.data.transforms import image_std

def psnr(x, y):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    delta = x - y
    delta = 255 * (delta * image_std.view(1, 3, 1, 1).to(x.device))
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    peak = 20 * math.log10(255.0)
    noise = torch.mean(delta**2, dim=(1,2,3))  # B
    psnr = peak - 10*torch.log10(noise)
    return psnr

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
    if preds.dim() == 4:
        bsz, nbits, h, w = preds.size()
        mask = mask.expand_as(preds).bool()
        preds = preds.masked_select(mask).view(bsz, nbits, -1)  # b k n
        preds = preds.mean(dim=-1, dtype=float)  # b k
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
    vid_w: str
) -> float:
    """
    Runs `ffmpeg -i vid_o.mp4 -i vid_w.mp4 -filter_complex libvmaf` and returns the score.
    """
    command = [
            'ffmpeg',
            '-i', vid_o,
            '-i', vid_w,
            '-filter_complex', 'libvmaf',
            '-f', 'null', '-'
        ]
    # Execute the command and capture the output
    try:
        result = subprocess.run(command, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        for line in result.stderr.split('\n'):
            if "VMAF score:" in line:
                # numerical part of the VMAF score with regex
                match = re.search(r"VMAF score: ([0-9.]+)", line)
                if match:
                    return float(match.group(1))
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def vmaf_on_tensor(
    vid_o: torch.Tensor,
    vid_w: torch.Tensor
) -> float:
    """
    ...
    """
    raise NotImplementedError


if __name__ == '__main__':
    # Test the PSNR function
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)
    print("test psnr")
    try:
        print("OK!", psnr(x, y))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Test the IoU function
    preds = torch.rand(1, 1, 256, 256)
    targets = torch.rand(1, 1, 256, 256)
    print("test iou")
    try:
        print("OK!", iou(preds, targets))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Test the accuracy function
    preds = torch.rand(1, 1, 256, 256)
    targets = torch.rand(1, 1, 256, 256)
    print("test accuracy")
    try:
        print("OK!", accuracy(preds, targets))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Test the vmaf function
    vid_o = 'assets/videos/sav_013754.mp4'
    vid_w = 'assets/videos/sav_013754.mp4'
    print("test vmaf")
    try:
        print("OK!", vmaf_on_file(vid_o, vid_w))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Try checking that ffmpeg is installed and that vmaf is available.")