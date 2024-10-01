"""
from root directory:
    python videoseal/evals/speed.py --device cuda
"""

import os
import torch
import torch.nn.functional as F
import time
import argparse
import pandas as pd
import omegaconf

from videoseal.models import build_embedder, build_extractor, Embedder, Extractor
from videoseal.data.transforms import normalize_img, unnormalize_img

def sync(device):
    """ wait for the GPU to finish processing, before measuring time """
    if device.startswith('cuda'):
        torch.cuda.synchronize()

def benchmark_model(model, img_size, data_loader, device):
    model.to(device)
    model.eval()
    times = []
    times_interp = []
    times_norm = []
    bsz = data_loader.batch_size
    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            h_orig, w_orig = imgs.size(-2), imgs.size(-1)
            # interpolate
            start_time = time.time()
            imgs = F.interpolate(imgs, size=(img_size, img_size), mode='bilinear', align_corners=False)
            sync(device)
            end_time = time.time()
            times_interp.append(end_time - start_time)
            # normalize
            start_time = time.time()
            imgs = normalize_img(imgs)
            sync(device)
            end_time = time.time()
            times_norm.append(end_time - start_time)
            # forward pass
            if isinstance(model, Embedder):
                msgs = model.get_random_msg(bsz=imgs.size(0))
                msgs = msgs.to(device)
                start_time = time.time()
                _ = model(imgs, msgs)
            elif isinstance(model, Extractor):
                start_time = time.time()
                _ = model(imgs)
            sync(device)
            end_time = time.time()
            times.append(end_time - start_time)
            # unnormalize
            start_time = time.time()
            imgs = unnormalize_img(imgs)
            sync(device)
            end_time = time.time()
            times_norm[-1] += end_time - start_time
            # interpolate
            start_time = time.time()
            imgs = F.interpolate(imgs, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            sync(device)
            end_time = time.time()
            times_interp[-1] += end_time - start_time
            
    results = {}
    for label, tt in [('forward', times), ('interp', times_interp), ('norm', times_norm)]:
        tt.pop(0)  # Remove the first batch
        time_total = sum(tt)
        time_per_batch = time_total / len(tt)
        time_per_img = time_per_batch / bsz
        curr_result = {
            f'{label}_time_per_img': time_per_img,
            f'{label}_time_per_batch': time_per_batch,
            f'{label}_time_total': time_total
        }
        results.update(curr_result)
    results.update({'nsamples': len(tt)})
    return results

def get_data_loader(batch_size, img_size, num_workers, nsamples):
    from torchvision.datasets import FakeData
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    transform = ToTensor()
    total_size = nsamples * batch_size
    dataset = FakeData(size=total_size, image_size=(3, img_size, img_size), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return loader

def main(args):
    device = args.device.lower()
    data_loader = get_data_loader(args.batch_size, args.img_size, args.workers, args.nsamples)

    embedder_cfg = omegaconf.OmegaConf.load(args.embedder_config)
    extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)
    if args.embedder_models is None:
        all_models = list(embedder_cfg.keys())
        all_models.remove('model')
        print("Available embedder models:", list(all_models))
        args.embedder_models = ','.join(all_models)
    if args.extractor_models is None:
        all_models = list(extractor_cfg.keys())
        all_models.remove('model')
        print("Available extractor models:", list(all_models))
        args.extractor_models = ','.join(all_models)

    results = []

    for embedder_name in args.embedder_models.split(','):
        embedder_args = embedder_cfg[embedder_name]
        embedder = build_embedder(embedder_name, embedder_args, args.nbits)
        embedder = embedder.to(device)
        embedder_stats = benchmark_model(embedder, args.img_size, data_loader, device)
        results.append({
            'model': embedder_name,
            'params': sum(p.numel() for p in embedder.parameters() if p.requires_grad) / 1e6,
            **embedder_stats,
        })
        print(results[-1])
    
    for extractor_name in args.extractor_models.split(','):
        extractor_args = extractor_cfg[extractor_name]
        extractor = build_extractor(extractor_name, extractor_args, args.img_size_work, args.nbits)
        extractor = extractor.to(device)
        extractor_stats = benchmark_model(extractor, args.img_size_work, data_loader, device)
        results.append({
            'model': extractor_name,
            'params': sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6,
            **extractor_stats
        })
        print(results[-1])

    # Save results to CSV
    df = pd.DataFrame(results)
    print(df)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df.to_csv(os.path.join(args.output_dir, 'speed_results.csv'), index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--nsamples', type=int, default=11)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--img_size_work', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--embedder_config', type=str, default='configs/embedder.yaml')
    parser.add_argument('--extractor_config', type=str, default='configs/extractor.yaml')
    parser.add_argument('--embedder_models', type=str, default=None)
    parser.add_argument('--extractor_models', type=str, default=None)
    parser.add_argument('--nbits', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()
    main(args)