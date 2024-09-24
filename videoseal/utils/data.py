
import threading
from collections import OrderedDict

import omegaconf

modality_to_datasets = {
    "image": ["coco"],
    "video": ["sa-v"]
}

class LRUDict(OrderedDict):
    def __init__(self, maxsize=10):
        super().__init__()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def __setitem__(self, key, value):
        with self.lock:
            super().__setitem__(key, value)

        if len(self) >= self.maxsize:
            # Clear 10% of the max size
            num_to_clear = int(self.maxsize * 0.1)
            keys_to_remove = list(self.keys())[:num_to_clear]
            for key in keys_to_remove:
                del self[key]

    def __getitem__(self, key):
        with self.lock:
            return super().__getitem__(key)

    def __delitem__(self, key):
        with self.lock:
            return super().__delitem__(key)


def parse_dataset_params(params):
    """
    Parses the dataset parameters and loads the dataset configuration if needed.

    Logic:
    1. If explicit directory paths are provided (--train_dir, --val_dir, etc.), use those.
    2. If a dataset name is provided (--dataset), load the corresponding configuration from configs/datasets/<dataset_name>.yaml.
    3. If neither explicit directory paths nor a dataset name is provided, raise an error.

    Args:
        params (argparse.Namespace): The parsed command-line arguments.

    Returns:
        params (argparse.Namespace): The parsed command-line arguments.
    """
    assert params.dataset in modality_to_datasets[
        params.modality], f"Invalid dataset '{params.dataset}' for modality '{params.modality}'"

    if params.train_dir is not None and params.val_dir is not None:
        pass
    elif params.dataset is not None:
        # Load dataset configuration
        dataset_cfg = omegaconf.OmegaConf.load(
            f"configs/datasets/{params.dataset}.yaml")
        # Merge the dataset configuration with the args
        dataset_dict = omegaconf.OmegaConf.to_container(dataset_cfg, resolve=True)
        # Update params with the fields from the dataset configuration
        for key, value in dataset_dict.items():
            setattr(params, key, value)
    else:
        # Raise an error if neither explicit directory paths nor a dataset name is provided
        raise ValueError(
            "Either provide dataset name or explicit train and val directories")

    if params.modality == "image":
        # Check that annotation files are provided for image modality
        assert params.train_annotation_file is not None and params.val_annotation_file is not None, \
            "Annotation files are required for image modality"

    return params