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


def eval(altered_images, watermark_image_dir, original_image_dir, detect_fc):
    results = {}
    for altered_image_file in tqdm.tqdm(altered_images):
        image_file = os.path.join(watermark_image_dir, os.path.basename(altered_image_file))

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
            original_image_file = os.path.join(original_image_dir, os.path.basename(image_file).replace("_1_wm.png", "_0_ori.png"))
            ori_img = Image.open(original_image_file)
        
        metrics = compute_metrics(tgt_img, ori_img, detect_fc)
        results[os.path.basename(image_file)] = metrics

    return results


if __name__ == "__main__":
    base_dir = "data/watermarked_images"
    watermarking_methods = [os.path.basename(m) for m in sorted(glob.glob(base_dir + "/*"))]

    method_name_mapping = {
        "CIN": "baseline/cin",
        "MBRS": "baseline/mbrs",
        "TrustMark": "baseline/trustmark",
        "VideoSealv1": "videoseal_0.0",
        "VideoSealv2pp256bits": "videoseal_1.0"
    }
    removal_methods = sorted(glob.glob("data/watermarks_removed_*/*"))

    print("Running the following evaluations...")
    run_params = []
    for method_dir in removal_methods:
        removal_method = os.path.basename(os.path.dirname(method_dir)).replace("watermarks_removed_", "")
        wm_method = os.path.basename(method_dir).split("_")[0]
        additional_params = ",".join(os.path.basename(method_dir).split("_")[1:])

        if wm_method not in method_name_mapping:
            print(f"  ! Method {wm_method} does not have associated any watermark detection model.")
            continue
        if wm_method not in watermarking_methods:
            print(f"  ! Method {wm_method} does not have any watermarked images.")
            continue

        print(f"  '{removal_method}' for '{wm_method}' with params: {additional_params}")
        run_params.append((removal_method, method_dir, wm_method, additional_params))
    print("-" * 36)

    joint_results = []
    for _, method_dir, wm_method, _ in run_params:
        watermark_image_dir = os.path.join(base_dir, wm_method)
        detect_fc = get_model(method_name_mapping[wm_method], watermark_image_dir)

        altered_images = sorted(glob.glob(os.path.join(method_dir, "*.png")))
        joint_results.append(
            eval(altered_images, watermark_image_dir, watermark_image_dir, detect_fc)
        )

    # print results
    metrics = ["bit_acc", "log_pvalue", "psnr", "cvvdp"]
    HEADER = f"{'REMOVAL METHOD':25}{'WATERMARK':15}" + "".join([f"{x:>12}" for x in metrics])
    print(HEADER)
    print("-" * len(HEADER))

    for params, results in zip(run_params, joint_results):
        print(f"{params[0] + ' (' + params[3] + ')':25}{params[2]:15}", end="")
        for metric in metrics:
            r = np.mean([v[metric] for v in results.values()])
            print(f"{r:12.3f}", end="")
        print()
