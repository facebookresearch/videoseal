from PIL import Image
import numpy as np
import glob, os
import torch
import tqdm
import sys
import random
from scipy import stats

sys.path.append("../../")
from videoseal.utils.cfg import setup_model_from_checkpoint
from videoseal.quality_metric.eval_watermark_removal import compute_metrics, get_model, cvvdp, psnr


# https://arxiv.org/pdf/2310.00076
# ROBUSTNESS OF AI-IMAGE DETECTORS: FUNDAMENTAL LIMITS AND PRACTICAL ATTACKS
def eval_watermark_noise_pasting(paste_image_dir, method_name, alpha=0.1):
    model = setup_model_from_checkpoint(method_name).eval().cuda()

    paste_image_files = sorted(glob.glob(os.path.join(paste_image_dir, "*.png")))

    results = {}
    for random_image_file in tqdm.tqdm(paste_image_files):
        img = Image.open(random_image_file)

        img_torch = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div_(255).unsqueeze(0).cuda()
        img_rand = torch.rand(img_torch.shape).cuda()
        img_rand_wm = model.embed(img_rand, is_video=False)

        z = alpha * img_rand_wm["imgs_w"]
        gamma = 1.0 - z.max()
        x = gamma * img_torch / img_torch.max()
        result = torch.round((x + z).clip(0, 1).mul_(255)).div_(255)

        spoofed_message = model.detect(result, is_video=False)

        nbits = len(img_rand_wm["msgs"].squeeze())
        acc = ((spoofed_message["preds"].cpu() > 0)[:, 1:].int() == img_rand_wm["msgs"]).float().mean().item()

        img_result = Image.fromarray(result.squeeze(0).permute(1, 2, 0).mul(255).to(torch.uint8).cpu().numpy())
        psnr_score = psnr(img_result, img)
        cvvdp_score = cvvdp(img_result, img)

        results[os.path.basename(random_image_file)] = {
            "bit_acc": acc,
            "log_pvalue": -np.log10(stats.binomtest(int(acc * nbits), nbits, 0.5, alternative='greater').pvalue),
            "psnr": psnr_score,
            "cvvdp": cvvdp_score
        }

    return results


# https://neurips.cc/virtual/2024/poster/94798
# Can Simple Averaging Defeat Modern Watermarks?
def eval_copy_paste_mean(watermark_image_dir, nonwatermarked_image_dir, paste_image_dir, method_name):
    detect_fc = get_model(method_name, watermark_image_dir)

    source_image_files = sorted(glob.glob(os.path.join(watermark_image_dir, "*.png")))
    target_image_files = sorted(glob.glob(os.path.join(nonwatermarked_image_dir, "*.png")) + glob.glob(os.path.join(nonwatermarked_image_dir, "*.jpg")))

    assert len(source_image_files) == len(target_image_files), "Mismatch in number of watermarked and non-watermarked images"

    paste_image_files = sorted(glob.glob(os.path.join(paste_image_dir, "*.png")))
    random.seed(42)
    random.shuffle(paste_image_files)

    residual = 0
    for im_fn1, im_fn2 in zip(tqdm.tqdm(source_image_files), target_image_files):
        im1 = Image.open(im_fn1).resize((768, 768))
        im2 = Image.open(im_fn2).resize((768, 768))

        residual += np.array(im1).astype(np.float32) - np.array(im2).astype(np.float32)
    residual = residual / len(source_image_files)

    results = {}
    for random_image_file in tqdm.tqdm(paste_image_files):
        img = Image.open(random_image_file)

        interpolated_residual = torch.nn.functional.interpolate(
            torch.tensor(residual).permute(2, 0, 1).unsqueeze(0),
            size=img.size[::-1]
        ).squeeze().permute(1, 2, 0).numpy()

        tgt_img = np.array(img).astype(np.float32) + interpolated_residual
        tgt_img = Image.fromarray(np.uint8(np.round(np.clip(tgt_img, 0, 255))))
        
        metrics = compute_metrics(tgt_img, img, detect_fc)
        results[os.path.basename(random_image_file)] = metrics

    return results


def eval_copy_paste_from_residual(watermark_image_dir, altered_image_dir, paste_image_dir, method_name):
    detect_fc = get_model(method_name, watermark_image_dir)

    source_image_files = sorted(glob.glob(os.path.join(watermark_image_dir, "*.png")))
    paste_image_files = sorted(glob.glob(os.path.join(paste_image_dir, "*.png")))
    random.seed(42)
    random.shuffle(paste_image_files)

    results = {}
    for image_file, random_image_file in zip(tqdm.tqdm(source_image_files), paste_image_files):
        altered_image_file = os.path.join(altered_image_dir, os.path.basename(image_file))

        wm_img = Image.open(image_file)
        img = Image.open(altered_image_file)
        rnd_img = Image.open(random_image_file)

        residual = np.array(img).astype(np.float32) - np.array(wm_img.resize(img.size)).astype(np.float32)
        interpolated_residual = torch.nn.functional.interpolate(
            torch.tensor(residual).permute(2, 0, 1).unsqueeze(0),
            size=rnd_img.size[::-1]
        ).squeeze().permute(1, 2, 0).numpy()

        tgt_img = np.array(rnd_img).astype(np.float32) - interpolated_residual
        tgt_img = Image.fromarray(np.uint8(np.round(np.clip(tgt_img, 0, 255))))
        
        metrics = compute_metrics(tgt_img, rnd_img, detect_fc)
        results[os.path.basename(random_image_file)] = metrics

    return results



if __name__ == "__main__":
    method = "baseline/cin"
    watermark_image_dir = "input/CIN_100_wm"
    altered_image_dirs = sorted(glob.glob("outputs/CIN_100_wm_sgdremoved_100*"))
    paste_image_dir = "input/CIN_9xx_ori"

    joint_results = [
        ("wm_averaging", eval_copy_paste_mean("input/CIN_900_wm", "input/test_900", paste_image_dir, method)),
        # ("wm_noise_pasting(a=0.1)", eval_watermark_noise_pasting(paste_image_dir, method, alpha=0.1)),
    ]

    # for altered_image_dir in altered_image_dirs:
    #     results = eval_copy_paste_from_residual(watermark_image_dir, altered_image_dir, paste_image_dir, method)
    #     joint_results.append((os.path.basename(altered_image_dir), results))
    #     break

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
