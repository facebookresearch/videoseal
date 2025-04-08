from PIL import Image
import numpy as np
import glob, os
import torch
import tqdm
import sys
import math
from scipy import stats
import pycvvdp

sys.path.append("../../")
from videoseal.utils.cfg import setup_model_from_checkpoint


cvvdp_display = pycvvdp.cvvdp(display_name='standard_fhd')


def get_model(method_name, wm_image_dir):
    model = setup_model_from_checkpoint(method_name).eval().cuda()

    with open(os.path.join(wm_image_dir, "message.txt"), "r") as f:
        message = f.read().strip()

    message = [int(ch.strip()) for ch in message.split(",") if ch.strip() != ""]
    msg_gt = torch.tensor(message)

    @torch.no_grad()
    def detect_fc(img: Image):
        input_ = torch.tensor(np.array(img)).float().div_(255).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        pred_msg = (model.detect(input_, is_video=False)["preds"] > 0).int().squeeze()[1:].cpu()

        nbits = len(msg_gt)
        acc = (pred_msg == msg_gt).float().mean().item()
        return {
            "bit_accuracy": acc,
            "log_pvalue": -np.log10(stats.binomtest(int(acc * nbits), nbits, 0.5, alternative='greater').pvalue)
        }

    return detect_fc


@torch.no_grad()
def cvvdp(img: Image, ref: Image):
    x = torch.tensor(np.array(img)).permute(2, 0, 1) / 255.
    y = torch.tensor(np.array(ref)).permute(2, 0, 1) / 255.

    score = cvvdp_display.predict(x.cuda(), y.cuda(), dim_order="CHW")[0]
    return score.item()


def psnr(img: Image, ref: Image):
    delta = np.array(img).astype(np.float32) - np.array(ref).astype(np.float32)
    peak = 20 * math.log10(255.0)
    noise = np.mean(delta ** 2)
    psnr = peak - 10 * np.log10(noise)
    return psnr


def compute_metrics(wm_image: Image, ref_image: Image, detect_fc):
    detection = detect_fc(wm_image)
    psnr_score = psnr(wm_image, ref_image)
    cvvdp_score = cvvdp(wm_image, ref_image)
    return {
        "bit_acc": detection["bit_accuracy"],
        "log_pvalue": detection["log_pvalue"],
        "psnr": psnr_score,
        "cvvdp": cvvdp_score
    }


def eval(source_image_dir, altered_image_dir, method_name, original_image_dir=None):
    detect_fc = get_model(method_name, source_image_dir)

    source_image_files = sorted(glob.glob(os.path.join(source_image_dir, "*.png")))
    results = {}

    for image_file in tqdm.tqdm(source_image_files):
        altered_image_file = os.path.join(altered_image_dir, os.path.basename(image_file))
        if not os.path.exists(altered_image_file):
            continue

        src_img = Image.open(image_file)
        img = Image.open(altered_image_file)

        residual = np.array(img).astype(np.float32) - np.array(src_img.resize(img.size)).astype(np.float32)
        interpolated_residual = torch.nn.functional.interpolate(
            torch.tensor(residual).permute(2, 0, 1).unsqueeze(0),
            size=src_img.size[::-1]
        ).squeeze().permute(1, 2, 0).numpy()

        tgt_img = np.array(src_img).astype(np.float32) + interpolated_residual
        tgt_img = Image.fromarray(np.uint8(np.round(np.clip(tgt_img, 0, 255))))

        ori_img = src_img
        if original_image_dir is not None:
            original_image_file = os.path.join(original_image_dir, os.path.basename(image_file)).replace("_1_wm.png", "_0_ori.png")
            ori_img = Image.open(original_image_file)
        
        metrics = compute_metrics(tgt_img, ori_img, detect_fc)
        results[os.path.basename(image_file)] = metrics

    return results


if __name__ == "__main__":
    method = "baseline/cin"
    source_image_dir = "input/CIN_100_wm"
    original_image_dir = "input/CIN_100_ori"
    altered_image_dirs = sorted(glob.glob("outputs/CIN_100*"))

    joint_results = []
    for altered_image_dir in altered_image_dirs:
        results = eval(source_image_dir, altered_image_dir, method, original_image_dir=original_image_dir)
        joint_results.append((os.path.basename(altered_image_dir), results))
        break

    # print results
    metrics = ["bit_acc", "log_pvalue", "psnr", "cvvdp"]
    HEADER = f"{'TYPE':30}" + "".join([f"{x:>12}" for x in metrics])
    print(HEADER)
    print("-" * len(HEADER))

    for wm_removal_name, results in joint_results:
        print(f"{wm_removal_name:30}", end="")
        for metric in metrics:
            r = np.mean([v[metric] for v in results.values()])
            print(f"{r:12.3f}", end="")
        print()
