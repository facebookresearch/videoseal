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
# Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks
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
def eval_copy_paste_mean(watermark_image_dir, nonwatermarked_image_dir, paste_image_dir, detect_fc, max_images=None):
    source_image_files = [x for x in sorted(glob.glob(os.path.join(watermark_image_dir, "*.png"))) if not x.endswith("ori.png") and not x.endswith("diff.png")]
    target_image_files = sorted(glob.glob(os.path.join(nonwatermarked_image_dir, "*.png")) + glob.glob(os.path.join(nonwatermarked_image_dir, "*.jpg")))

    if max_images is not None:
        source_image_files = source_image_files[:max_images]
        target_image_files = target_image_files[:max_images]
    
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


def eval_copy_paste_from_residual(altered_images, watermark_image_dir, paste_to_image_dir, detect_fc):
    paste_image_files = sorted(glob.glob(os.path.join(paste_to_image_dir, "*.png")))
    random.seed(42)
    random.shuffle(paste_image_files)

    assert len(altered_images) <= len(paste_image_files), f"There are not enough images in the `paste_to_image_dir` ({paste_to_image_dir})."

    results = {}
    for altered_image_file, random_image_file in zip(tqdm.tqdm(altered_images), paste_image_files):
        image_file = os.path.join(watermark_image_dir, os.path.basename(altered_image_file))

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
    assert len(sys.argv) == 3, "Usage: python eval_watermark_forging.py <nonwm_image_dir1> <nonwm_image_dir2>"
    paste_to_image_dir = sys.argv[1]  # "/private/home/soucek/videoseal/data/paste_to_images"
    random_nonwm_image_dir = sys.argv[2]  # "/large_experiments/omniseal/sa-1b-full/test/test"

    base_dir = "data/watermarked_images"
    watermarking_methods = [os.path.basename(m) for m in sorted(glob.glob(base_dir + "/*"))]

    method_name_mapping = {
        "CIN": "baseline/cin",
        "MBRS": "baseline/mbrs",
        "TrustMark": "baseline/trustmark",
        "VideoSealv1": "videoseal_0.0",
        "VideoSealv2pp256bits": "videoseal_1.0"
    }
    removal_methods = sorted(glob.glob("data/watermarks_removed_ours/*"))

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

    #############################
    
    joint_results = []
    for _, method_dir, wm_method, _ in run_params:
        watermark_image_dir = os.path.join(base_dir, wm_method)
        detect_fc = get_model(method_name_mapping[wm_method], watermark_image_dir)

        altered_images = sorted(glob.glob(os.path.join(method_dir, "*.png")))
        joint_results.append(
            eval_copy_paste_from_residual(altered_images, watermark_image_dir, paste_to_image_dir, detect_fc)
        )

    #############################

    for method_name in watermarking_methods:
        for alpha in [0.1, 0.2]:
            joint_results.append(
                eval_watermark_noise_pasting(paste_to_image_dir, method_name_mapping[method_name], alpha=alpha)
            )
            run_params.append(("wm_noise_pasting", paste_to_image_dir, method_name, f"a={alpha}"))

    #############################
    
    for method_name in watermarking_methods:
        watermark_image_dir = os.path.join(base_dir, method_name)
        detect_fc = get_model(method_name_mapping[method_name], watermark_image_dir)

        for max_images in [900, 100, 10]:
            joint_results.append(
                eval_copy_paste_mean(watermark_image_dir, random_nonwm_image_dir, paste_to_image_dir, detect_fc, max_images=max_images)
            )
            run_params.append(("wm_averaging", paste_to_image_dir, method_name, f"n={max_images}"))

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
