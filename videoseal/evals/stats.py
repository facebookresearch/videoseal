"""
Run with:
python -m videoseal.evals.stats \
    --checkpoint baseline/wam \
    --dataset sa-1b-full-resized --is_video false --num_samples 10
"""


import argparse
import os
import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt  
from matplotlib import cm  # Add this import for colormap

from ..augmentation import get_validation_augs
from ..utils.cfg import setup_dataset, setup_model_from_checkpoint
from ..utils import Timer, bool_inst


@torch.no_grad()
def extract_stats(
    model,
    dataset: Dataset,
    is_video: bool,
    output_dir: str,
    video_aggregation: str = "avg",
    interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
):
    """
    Extracts watermark messages for all images/videos in the dataset and saves them in a CSV file.

    Args:
        model: The watermark extractor model.
        dataset (Dataset): The dataset to evaluate on.
        is_video (bool): Whether the data is video.
        output_dir (str): Directory to save the output CSV files.
        video_aggregation (str): Aggregation method for detection of video frames.
        interpolation (dict): Interpolation settings for resizing.
    """
    os.makedirs(output_dir, exist_ok=True)
    validation_augs = get_validation_augs(is_video)
    timer = Timer()

    for validation_aug, strengths in validation_augs:
        for strength in strengths:
            aug_name = f"{validation_aug}_{strength}".replace(", ", "_")
            csv_path = os.path.join(output_dir, f"stats_{aug_name}.csv")
            print(f"Saving stats to {csv_path}")

            with open(csv_path, "w") as f:
                f.write("index,message\n")

                for idx, batch_items in enumerate(tqdm.tqdm(dataset)):
                    if batch_items is None:
                        continue

                    imgs, masks = batch_items[0], batch_items[1]
                    if not is_video:
                        imgs = imgs.unsqueeze(0)  # c h w -> 1 c h w

                    # Apply augmentation
                    imgs_aug, _ = validation_aug(imgs, masks, strength)

                    # Extract watermark
                    if is_video:
                        preds = model.detect_and_aggregate(
                            imgs_aug, video_aggregation, interpolation
                        )  # 1 k
                    else:
                        preds = model.detect(imgs_aug, is_video=False)["preds"]  # 1 k

                    mask_preds = preds[:, 0:1]  # b 1 ..
                    bit_preds = preds[:, 1:]  # b k ...
                    if len(bit_preds.shape) > 2:
                        dims_to_avg = list(range(2, len(bit_preds.shape)))
                        bit_preds = bit_preds.mean(dim=dims_to_avg)  # b k
                    bit_preds = bit_preds.squeeze(0)  # k

                    # Save extracted message
                    message = ",".join(map(str, bit_preds.tolist()))
                    f.write(f'{idx},"{message}"\n')

            # Load the CSV file and generate a combined histogram
            data = pd.read_csv(csv_path)
            messages = data["message"].astype(str).str.split(",", expand=True).astype(float)

            plt.figure(figsize=(30, 12))  # Make the plot wider
            colormap = cm.plasma  # Use the plasma colormap
            means = messages.mean(axis=0)  # Calculate the mean for each bit
            abs_means = means.abs()  # Absolute values of means
            top_bits = abs_means.nlargest(10).index  # Indices of the 10 bits furthest from 0
            colors = colormap((abs_means[top_bits] / abs_means.max()).values)  # Normalize and map to colormap

            for i, bit_idx in enumerate(top_bits):
                plt.hist(
                    messages[bit_idx],
                    bins=100,
                    alpha=0.5,
                    color=colors[i],
                    label=f"Bit {bit_idx} (mean={means[bit_idx]:.2f})",
                )

            plt.title(f"Combined Histogram for Top 10 Bits Furthest from 0 ({aug_name})")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.legend(loc="upper right", fontsize="small")
            plot_path = os.path.join(output_dir, f"top_10_combined_histogram_{aug_name}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved combined histogram for top 10 bits to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract watermark stats from a dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")

    group = parser.add_argument_group("Dataset")
    group.add_argument("--dataset", type=str, help="Name of the dataset.")
    group.add_argument("--is_video", type=bool_inst, default=False, help="Whether the data is video")
    group.add_argument('--short_edge_size', type=int, default=-1, 
                       help='Resizes the short edge of the image to this size at loading time, and keep the aspect ratio. If -1, no resizing.')
    group.add_argument('--num_frames', type=int, default=24*3, 
                       help='Number of frames to evaluate for video quality')
    group.add_argument('--num_samples', type=int, default=100, 
                          help='Number of samples to evaluate')

    group = parser.add_argument_group("Experiment")
    group.add_argument("--output_dir", type=str, default="output/", help="Output directory for CSV files")

    group = parser.add_argument_group("Interpolation")
    group.add_argument("--interpolation_mode", type=str, default="bilinear",
                       choices=["nearest", "bilinear", "bicubic", "area"],
                       help="Interpolation mode for resizing")
    group.add_argument("--interpolation_align_corners", type=bool_inst, default=False,
                       help="Align corners for interpolation")
    group.add_argument("--interpolation_antialias", type=bool_inst, default=True,
                       help="Use antialiasing for interpolation")

    args = parser.parse_args()

    # Setup the model
    model = setup_model_from_checkpoint(args.checkpoint)
    model.eval()

    # Setup the device
    avail_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device or avail_device
    model.to(device)

    # Setup the dataset
    dataset = setup_dataset(args)
    dataset = Subset(dataset, range(args.num_samples))

    # Extract stats
    interpolation = {
        "mode": args.interpolation_mode,
        "align_corners": args.interpolation_align_corners,
        "antialias": args.interpolation_antialias,
    }
    extract_stats(
        model=model,
        dataset=dataset,
        is_video=args.is_video,
        output_dir=args.output_dir,
        video_aggregation="avg",
        interpolation=interpolation,
    )


if __name__ == "__main__":
    main()