"""
Feature extraction script for image datasets.
Run with:
    python -m examples.extract
"""

import argparse
import json
import os
import tqdm

import torch
from torchvision import transforms

from helper import build_backbone, get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction from images using various models")
    parser.add_argument("--output_dir", type=str, default='output', help="Directory to save extracted features")
    parser.add_argument("--data_dir", type=str, default="/datasets01/imagenet_full_size/061417", help="Directory containing images")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14", help="Model architecture to use")
    parser.add_argument("--model_path", type=str, default="", help="Path to model weights")
    parser.add_argument("--resize_size", type=int, default=288, help="Resize images to this size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--chunk", type=int, default=1, help="Number of chunks to split the dataset")
    parser.add_argument("--chunk_id", type=int, default=0, help="ID of the chunk to process")
    return parser


def extract_features(params):
    """Extract features from images using the specified model."""
    os.makedirs(params.output_dir, exist_ok=True)
    
    print('>>> Building backbone...')
    model = build_backbone(path=params.model_path, name=params.model_name)
    model.eval()
    model.to(device)

    print('>>> Creating dataloader...')
    NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    default_transform = transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE_IMAGENET,
        transforms.Resize((params.resize_size, params.resize_size), antialias=True),
    ])
    
    img_loader = get_dataloader(
        params.data_dir, 
        default_transform, 
        batch_size=params.batch_size, 
        shuffle=False, 
        collate_fn=None, 
        chunk_id=params.chunk_id, 
        chunk=params.chunk
    )

    print('>>> Extracting features...')
    features = []
    with open(os.path.join(params.output_dir, "log.txt"), 'w') as f:
        with torch.no_grad():
            for ii, imgs in enumerate(tqdm.tqdm(img_loader)):
                imgs = imgs.to(device)
                fts = model(imgs)
                features.append(fts.cpu())
                for jj in range(fts.shape[0]):
                    sample_fname = img_loader.dataset.dataset.samples[img_loader.dataset.indices[ii*params.batch_size + jj]]
                    f.write(f"{sample_fname}\n")

    print('>>> Saving features...')
    features = torch.concat(features, dim=0)
    torch.save(features, os.path.join(params.output_dir, 'fts.pth'))
    print(f"Features saved to {os.path.join(params.output_dir, 'fts.pth')}")


if __name__ == '__main__':
    params = get_parser().parse_args()
    print("__log__:{}".format(json.dumps(vars(params))))
    extract_features(params)
