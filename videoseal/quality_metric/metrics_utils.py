import os
import enum
import glob
from PIL import Image
from torch.utils.data import Dataset


class MetricType(enum.Enum):
    NO_REFERENCE = 0
    REFERENCE = 1

class MetricObjective(enum.Enum):
    MINIMIZE = 0
    MAXIMIZE = 1


class MetricResult:

    def __init__(self, value: float, metric_objective: MetricObjective, metric_type: MetricType, metric_name: str):
        self.value = value
        self.objective = metric_objective
        self.type = metric_type
        self.name = metric_name
        self._count = 1

    def __eq__(self, other):
        assert self.name == other.name
        return self.value == other.value
    
    def __lt__(self, other):
        assert self.name == other.name
        return self.value < other.value if self.objective == MetricObjective.MAXIMIZE else self.value > other.value

    def __add__(self, other):
        assert self.name == other.name
        count = self._count + other._count
        value = (self._count * self.value + other._count * other.value) / count

        result = MetricResult(value, self.objective, self.type, self.name)
        result._count = count
        return result

    def __repr__(self):
        return str(self.value)

    def __float__(self):
        return self.value

    def __format__(self, format_spec):
        format_spec = f"{{value:{format_spec}}}"
        return format_spec.format(value=self.value)


class DummyPairedDataset(Dataset):

    def __init__(self, method1, method2=None, max_size=None, data_root=None):
        ROOT = "/private/home/soucek/videoseal/metrics"
        if data_root is not None:
            ROOT = data_root
        
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
