import os
import sys
import copy
import glob
import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

sys.path.append("../../")
from videoseal.quality_metric import thirdparty_metrics
from videoseal.quality_metric import artifact_discriminator_metric


class DummyPairedDataset(Dataset):

    def __init__(self, method1, method2=None, max_size=None):
        ROOT = "/private/home/soucek/videoseal/metrics"
        assert os.path.exists(os.path.join(ROOT, method1))
        if method2 is not None:
            assert os.path.exists(os.path.join(ROOT, method2))

        self.files = sorted(glob.glob(os.path.join(ROOT, method1, "*_val_1_wm.png")))
        self.method1, self.method2 = method1, method2
        self.max_size = max_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.method2 is None:
            fn1, fn2 = self.files[idx], self.files[idx].replace(f"_val_1_wm.png", f"_val_0_ori.png")
        else:
            fn1, fn2 = self.files[idx], self.files[idx].replace(f"/{self.method1}/", f"/{self.method2}/")
        
        img1, img2 = Image.open(fn1), Image.open(fn2)
        assert img1.size == img2.size

        if self.max_size is not None:
            coef = max(img1.size) / self.max_size
            new_size = round(img1.size[0] / coef), round(img1.size[1] / coef)
            img1, img2 = img1.resize(new_size), img2.resize(new_size)

        return img1, img2


class Evaluator:

    def __init__(self, device="cpu"):
        self.noreference_metrics = thirdparty_metrics.get_metrics_noref(device)
        self.reference_metrics = thirdparty_metrics.get_metrics_ref(device)
        self.noreference_metrics = self.noreference_metrics | {
            "ArtifactDisc": artifact_discriminator_metric.MetricArtifactDiscriminator(device=device)
        }

    def eval_both(self, method1_name: str, method2_name: str, max_size=None):
        method1 = DummyPairedDataset(method1_name, max_size=max_size)
        method2 = DummyPairedDataset(method2_name, max_size=max_size)

        results1 = {k: None for k in self.noreference_metrics.keys()} | {k: None for k in self.reference_metrics.keys()}
        results2 = copy.deepcopy(results1)
        ranking_results = {k: [] for k in self.noreference_metrics.keys()} | {k: [] for k in self.reference_metrics.keys()}

        for (img1, img_ori), (img2, _) in zip(tqdm.tqdm(method1, leave=False), method2):
            for name, metric in self.noreference_metrics.items():
                r1, r2 = metric(img1), metric(img2)
                results1[name] = r1 if results1[name] is None else results1[name] + r1
                results2[name] = r2 if results2[name] is None else results2[name] + r2
                ranking_results[name].append(r1 < r2)
            for name, metric in self.reference_metrics.items():
                r1, r2 = metric(img1, img_ori), metric(img2, img_ori)
                results1[name] = r1 if results1[name] is None else results1[name] + r1
                results2[name] = r2 if results2[name] is None else results2[name] + r2
                ranking_results[name].append(r1 < r2)

        ranking_results = [(k, float(np.mean(v))) for k, v in ranking_results.items()]
        print(f"M1={method1_name}, M2={method2_name}")
        print(f"{'Method':23}{'winrate':>8}{'score(M1)':>12}{'score(M2)':>12}")
        print("-" * (23 + 8 + 12 * 2 + 8))
        for k, v in sorted(ranking_results, key=lambda x: -x[1]):
            print(f"{k:23}{v:8.2f}{results1[k]:12.3f}{results2[k]:12.3f}{' (noref)' if k in self.noreference_metrics else ''}")
        print("")


    def eval_noreference(self, method_name: str, max_size=None):
        method = DummyPairedDataset(method_name, max_size=max_size)

        results1 = {k: None for k in self.noreference_metrics.keys()}
        results2 = copy.deepcopy(results1)
        ranking_results = {k: [] for k in self.noreference_metrics.keys()}

        for img1, img_ori in tqdm.tqdm(method, leave=False):
            for name, metric in self.noreference_metrics.items():
                r1, r2 = metric(img1), metric(img_ori)
                results1[name] = r1 if results1[name] is None else results1[name] + r1
                results2[name] = r2 if results2[name] is None else results2[name] + r2
                ranking_results[name].append(r1 < r2)

        ranking_results = [(k, float(np.mean(v))) for k, v in ranking_results.items()]
        print(f"M1={method_name}")
        print(f"{'Method':23}{'winrate':>8}{'score(M1)':>12}{'score(ori)':>12}")
        print("-" * (23 + 8 + 12 * 2 + 8))
        for k, v in sorted(ranking_results, key=lambda x: -x[1]):
            print(f"{k:23}{v:8.2f}{results1[k]:12.3f}{results2[k]:12.3f}{' (noref)' if k in self.noreference_metrics else ''}")
        print("")


if __name__ == "__main__":
    evaluator = Evaluator(device="cuda:1")

    evaluator.eval_noreference("HIDDEN")
    evaluator.eval_noreference("TrustMark")
    evaluator.eval_noreference("MBRS")
    evaluator.eval_noreference("WAM")
    evaluator.eval_noreference("CIN")
    evaluator.eval_noreference("VideoSealv1")
    evaluator.eval_noreference("VideoSealv2image")
    evaluator.eval_noreference("VideoSealCVVDP")
    evaluator.eval_noreference("VideoSealv2")
    evaluator.eval_noreference("VideoSealv2pp")
    evaluator.eval_noreference("VideoSealv2pp256bit")
    
    evaluator.eval_both("VideoSealv1", "VideoSealv2")
    evaluator.eval_both("VideoSealv2image", "VideoSealv2")
    evaluator.eval_both("TrustMark", "VideoSealv2")
    evaluator.eval_both("VideoSealv1", "VideoSealv2pp")
    evaluator.eval_both("VideoSealv2", "VideoSealv2pp")
    evaluator.eval_both("TrustMark", "VideoSealv2pp")
    evaluator.eval_both("MBRS", "VideoSealv2pp")
    evaluator.eval_both("WAM", "VideoSealv2pp")
    evaluator.eval_both("CIN", "VideoSealv2pp")
    evaluator.eval_both("VideoSealv2pp", "VideoSealv2pp256bit")
