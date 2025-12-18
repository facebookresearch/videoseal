# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
from root directory:
    python -m videoseal.evals.flops --device cuda --nbits 128 --hidden_size_multiplier 1
"""

import argparse
import os
import time

import omegaconf
import pandas as pd
import torch
import torch.nn.functional as F
from calflops import calculate_flops
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from ..utils import Timer, bool_inst
from ..models import (Embedder, Extractor, build_embedder,
                              build_extractor)


def sync(device):
    """ wait for the GPU to finish processing, before measuring time """
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    return


def get_flops(model, channels, img_size, device):
    if isinstance(model, Embedder):
        msgs = model.get_random_msg(bsz=1)
        msgs = msgs.to(device)
        img_size = (1, channels, img_size, img_size)
        return calculate_flops(
            model,
            args=[torch.randn(img_size), msgs],
            output_as_string=False,
            output_precision=4,
            print_results=False
        )
    elif isinstance(model, Extractor):
        img_size = (1, channels, img_size, img_size)
        return calculate_flops(
            model,
            args=[torch.randn(img_size)],
            output_as_string=False,
            output_precision=4,
            print_results=False
        )


@torch.no_grad()
def benchmark_model(model, img_size, data_loader, device):
    model.to(device)
    model.eval()

    times = []
    times_interp = []
    times_norm = []
    timer = Timer()
    bsz = data_loader.batch_size
    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            h_orig, w_orig = imgs.size(-2), imgs.size(-1)
            # interpolate
            timer.start()
            imgs = F.interpolate(imgs, size=(
                img_size, img_size), mode='bilinear', align_corners=False)
            sync(device)
            times_interp.append(timer.end())
            # normalize
            timer.start()
            sync(device)
            times_norm.append(timer.end())
            # forward pass
            if isinstance(model, Embedder):
                msgs = model.get_random_msg(bsz=imgs.size(0))
                msgs = msgs.to(device)
                timer.start()
                _ = model(imgs, msgs)
            elif isinstance(model, Extractor):
                timer.start()
                _ = model(imgs)
            sync(device)
            times.append(timer.end())
            # unnormalize
            timer.start()
            sync(device)
            times_norm[-1] += timer.end()
            # interpolate
            timer.start()
            imgs = F.interpolate(imgs, size=(h_orig, w_orig),
                                 mode='bilinear', align_corners=False)
            sync(device)
            times_interp[-1] += timer.end()

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


def get_data_loader(batch_size, img_size, channels, num_workers, nsamples):

    transform = ToTensor()
    total_size = nsamples * batch_size
    dataset = FakeData(
        size=total_size, 
        image_size=(channels, img_size, img_size), 
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)
    return loader


def main(args):
    device = args.device.lower()

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
        result = {'model': embedder_name}
        # build
        if embedder_name not in embedder_cfg:
            continue
        embedder_args = embedder_cfg[embedder_name]
        embedder = build_embedder(embedder_name, embedder_args, args.nbits, args.hidden_size_multiplier)
        embedder = embedder.to(device)
        # flops
        if args.do_flops:
            channels = 1 if 'yuv' in embedder_name else 3
            flops, macs, params = get_flops(
                embedder, channels, args.img_size_proc, device)
            result.update({
                'gflops': flops / 1e9,
                'gmacs': macs / 1e9,
                'params(m)': params / 1e6
            })
        # benchmark
        if args.do_speed:
            channels = 1 if 'yuv' in embedder_name else 3
            data_loader = get_data_loader(args.batch_size, args.img_size, channels, args.workers, args.nsamples)
            embedder_stats = benchmark_model(
                embedder, args.img_size_proc, data_loader, device)
            result.update({
                # 'model': embedder_name,
                # 'params': sum(p.numel() for p in embedder.parameters() if p.requires_grad) / 1e6,
                **embedder_stats,
            })
        results.append(result)
        print(results[-1])

    for extractor_name in args.extractor_models.split(','):
        result = {'model': extractor_name}
        # build
        if extractor_name not in extractor_cfg:
            continue
        extractor_args = extractor_cfg[extractor_name]
        extractor = build_extractor(
            extractor_name, extractor_args, args.img_size_proc, args.nbits)
        extractor = extractor.to(device)
        # flops
        if args.do_flops:
            channels = 1 if 'yuv' in extractor_name else 3
            flops, macs, params = get_flops(
                extractor, channels, args.img_size_proc, device)
            result.update({
                'gflops': flops / 1e9,
                'gmacs': macs / 1e9,
                'params(m)': params / 1e6
            })
        # benchmark
        if args.do_speed:
            channels = 1 if 'yuv' in extractor_name else 3
            data_loader = get_data_loader(args.batch_size, args.img_size, channels, args.workers, args.nsamples)
            extractor_stats = benchmark_model(
                extractor, args.img_size_proc, data_loader, device)
            result.update({
                # 'model': extractor_name,
                # 'params': sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6,
                **extractor_stats,
            })
        results.append(result)
        print(results[-1])

    # Save results to CSV
    df = pd.DataFrame(results)
    print(df)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df.to_csv(os.path.join(args.output_dir, 'speed_results.csv'),
              index=False, float_format='%.5f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--nsamples', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--img_size_proc', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--embedder_config', type=str,
                        default='configs/embedder.yaml')
    parser.add_argument('--extractor_config', type=str,
                        default='configs/extractor.yaml')
    parser.add_argument('--embedder_models', type=str, default=None)
    parser.add_argument('--extractor_models', type=str, default=None)
    parser.add_argument('--nbits', type=int, default=32)
    parser.add_argument('--hidden_size_multiplier', type=int
                        , default=1)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--do_flops', type=bool_inst, default=True, 
                        help='Calculate FLOPS for each model')
    parser.add_argument('--do_speed', type=bool_inst, default=False,
                        help='Run speed benchmark for each model')

    args = parser.parse_args()
    main(args)
