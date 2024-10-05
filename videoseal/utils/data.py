
import threading
from collections import OrderedDict

import omegaconf


class Modalities:
    IMAGE = 'image'
    VIDEO = 'video'
    HYBRID = 'hybrid'

import psutil

class LRUDict(OrderedDict):
    def __init__(self, maxsize=10):
        super().__init__()
        self.maxsize = maxsize
        self.lock = threading.RLock()  # Use a reentrant lock to avoid deadlocks

    def __setitem__(self, key, value):
        with self.lock:
            # Insert the item in the dictionary
            super().__setitem__(key, value)

            # If the dictionary exceeds max size, remove the least recently used items
            if len(self) > self.maxsize:
                self._cleanup()

    def __getitem__(self, key, cleaningup=False):
        with self.lock:
            value = super().__getitem__(key)
            # Move the accessed item to the end to mark it as recently used
            return value

    def __delitem__(self, key):
        with self.lock:
            super().__delitem__(key)

    def _cleanup(self):
        # Remove the least recently used items until we're back under the limit
        num_to_clear = max(1, int(0.1 * self.maxsize))  # Clear 10% or at least 1
        for _ in range(num_to_clear):
            self.popitem(last=False)  # Remove from the start (LRU)


def parse_dataset_params(params):
    """
    Parses the dataset parameters and loads the dataset configuration if needed.

    Logic:
    1. If a dataset name is provided (--image_dataset or --video_dataset), load the corresponding configuration from configs/datasets/<dataset_name>.yaml.
    2. If neither dataset name is provided, raise an error.

    Args:
        params (argparse.Namespace): The parsed command-line arguments.

    Returns:
        params (argparse.Namespace): The parsed command-line arguments.
    """
    # Load dataset configurations
    image_dataset_cfg = None
    video_dataset_cfg = None

    if params.image_dataset is not None:
        image_dataset_cfg = omegaconf.OmegaConf.load(
            f"configs/datasets/{params.image_dataset}.yaml")
    if params.video_dataset is not None:
        video_dataset_cfg = omegaconf.OmegaConf.load(
            f"configs/datasets/{params.video_dataset}.yaml")

    # Check if at least one dataset is provided
    if image_dataset_cfg is None and video_dataset_cfg is None:
        raise ValueError("Provide at least one dataset name")

    # Set modality
    if image_dataset_cfg is not None and video_dataset_cfg is not None:
        params.modality = Modalities.HYBRID
    elif image_dataset_cfg is not None:
        params.modality = Modalities.IMAGE
    else:
        params.modality = Modalities.VIDEO

    # Merge the dataset configurations with the args
    for cfg in [image_dataset_cfg, video_dataset_cfg]:
        if cfg is not None:
            dataset_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
            for key, value in dataset_dict.items():
                setattr(params, key, value)

    # Store dataset configurations
    params.image_dataset_config = omegaconf.OmegaConf.to_container(
        image_dataset_cfg, resolve=True) if image_dataset_cfg is not None else None
    params.video_dataset_config = omegaconf.OmegaConf.to_container(
        video_dataset_cfg, resolve=True) if video_dataset_cfg is not None else None

    return params
