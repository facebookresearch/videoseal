"""
python evals/speed.py 

--embedder_config path/to/embedder_config.yaml --extractor_config path/to/extractor_config.yaml --embedder_model model_name --extractor_model model_name
"""

import torch
import time
import argparse
from videoseal.models import build_embedder, build_extractor, Embedder, Extractor
import omegaconf

def benchmark_embedder(
    embedder: Embedder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    embedder.to(device)
    embedder.eval()
    times = []
    bsz = data_loader.batch_size
    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            msgs = embedder.get_random_msg(bsz=imgs.size(0))
            msgs = msgs.to(device)
            start_time = time.time()
            _ = embedder(imgs, msgs)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    time_total = sum(times)
    time_per_batch = time_total / len(times)
    time_per_img = time_per_batch / bsz
    return {
        'time_per_img': time_per_img,
        'time_per_batch': time_per_batch,
        'time_total': time_total,
        'nsamples': len(times)
    }

def benchmark_extractor(
    extractor: Extractor,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    extractor.to(device)
    extractor.eval()
    times = []
    bsz = data_loader.batch_size
    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            start_time = time.time()
            _ = extractor(imgs)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    time_total = sum(times)
    time_per_batch = time_total / len(times)
    time_per_img = time_per_batch / bsz
    return {
        'time_per_img': time_per_img,
        'time_per_batch': time_per_batch,
        'time_total': time_total,
        'nsamples': len(times)
    }

def get_data_loader(batch_size, img_size, num_workers, nsamples):
    from torchvision.datasets import FakeData
    from torchvision.transforms import Compose, Resize, ToTensor
    from torch.utils.data import DataLoader

    transform = Compose([Resize((img_size, img_size)), ToTensor()])
    total_size = nsamples * batch_size
    dataset = FakeData(size=total_size, image_size=(3, img_size, img_size), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return loader

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = get_data_loader(args.batch_size, args.img_size, args.workers, args.nsamples)

    # Load configurations and build models
    embedder_cfg = omegaconf.OmegaConf.load(args.embedder_config)
    args.embedder_model = args.embedder_model or embedder_cfg.model
    embedder_args = embedder_cfg[args.embedder_model]
    embedder = build_embedder(args.embedder_model, embedder_args, args.nbits)
    embedder = embedder.to(device)
    print(embedder)
    print(f'embedder: {sum(p.numel() for p in embedder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # Build the extractor model
    extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)
    args.extractor_model = args.extractor_model or extractor_cfg.model
    extractor_args = extractor_cfg[args.extractor_model]
    extractor = build_extractor(args.extractor_model, extractor_args, args.img_size_extractor, args.nbits)
    extractor = extractor.to(device)
    print(extractor)
    print(f'extractor: {sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # Benchmark embedder
    embedder_stats = benchmark_embedder(embedder, data_loader, device)
    print(f"Embedder stats: {embedder_stats}")

    # Benchmark extractor
    extractor_stats = benchmark_extractor(extractor, data_loader, device)
    print(f"Extractor stats: {extractor_stats}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--nsamples', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--img_size_extractor', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--embedder_config', type=str, default='configs/embedder.yaml')
    parser.add_argument('--extractor_config', type=str, default='configs/extractor.yaml')
    parser.add_argument('--embedder_model', type=str, default=None)
    parser.add_argument('--extractor_model', type=str, default=None)
    parser.add_argument('--nbits', type=int, default=32)
    args = parser.parse_args()
    main(args)