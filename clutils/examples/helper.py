import os
import numpy as np
import logging
import socket
import subprocess
from PIL import ImageFile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from torchvision import models
from torchvision.datasets.folder import default_loader, is_image_file


ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_backbone(path, name):
    """ Build a pretrained torchvision backbone from its name.

    Args:
        path: path to the checkpoint, can be an URL
        name: name of the architecture from torchvision (see https://pytorch.org/vision/stable/models.html) 
        Using other architectures (such as non-convolutional ones) might need changes in the implementation.
    """
    if name == 'torchscript':
        model = torch.jit.load(path)
        return model
    if 'dinov2' in name:
        model = torch.hub.load('facebookresearch/dinov2', name)
        return model
    if 'dino' in name:
        model = torch.hub.load('facebookresearch/dino', name)
        return model
    if hasattr(models, name):
        # Use weights parameter instead of pretrained to avoid deprecation warning
        model_fn = getattr(models, name)
        if 'resnet' in name:
            # For ResNet models, use the specific weights enum
            weights_enum = getattr(models, f"{name.capitalize()}_Weights").DEFAULT
            model = model_fn(weights=weights_enum)
        else:
            # For other models, just use the function without pretrained
            model = model_fn(weights="DEFAULT")
        
        # Remove classification head
        model.head = nn.Identity() if hasattr(model, 'head') else model.head
        model.fc = nn.Identity() if hasattr(model, 'fc') else model.fc
    else:
        raise ValueError(f"Model {name} not found in torchvision models")
    
    if path is not None and path != "":
        try:
            if path.startswith("http"):
                checkpoint = torch.hub.load_state_dict_from_url(path, progress=False)
            else:
                # Use weights_only=True to avoid security warnings
                checkpoint = torch.load(path, weights_only=True)
            
            state_dict = checkpoint
            for ckpt_key in ['state_dict', 'model_state_dict', 'teacher']:
                if ckpt_key in checkpoint:
                    state_dict = checkpoint[ckpt_key]
            
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    return model.to(device, non_blocking=True)


def get_image_paths(path):
    print(f"Resolving files in: {path}")
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])


def collate_fn(batch):
    """ Collate function for data loader. Allows to have img of different size"""
    return batch


def get_dataloader(data_dir, transform, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_fn, chunk_id=None, chunk=None):
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageFolder(data_dir, transform=transform)
    if chunk_id is not None:
        selected_imgs = np.array_split(np.arange(len(dataset)), chunk)[chunk_id]
        dataset = Subset(dataset, selected_imgs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)


class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path, transform=None, loader=default_loader):
        self.samples = get_image_paths(path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.samples[idx])
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def is_distributed():
    return get_world_size() > 1


def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_logging_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import sys
    import builtins as __builtin__

    # Deactivate printing when not in master process
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    class RankFormatter(logging.Formatter):
        def format(self, record):
            record.rank = dist.get_rank()
            return super().format(record)

    # Set up logging with rank
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = RankFormatter('[rank%(rank)s]:%(asctime)s:%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.propagate = False


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    The params need to contain the following attributes:
        - local_rank
        - master_port
        - debug_slurm
    """
    params.is_slurm_job = 'SLURM_JOB_ID' in os.environ and not params.debug_slurm
    print("SLURM job: %s" % str(params.is_slurm_job))

    # SLURM job
    if params.is_slurm_job:

        assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            print(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        params.node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        params.world_size = int(os.environ['SLURM_NTASKS'])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.master_addr = hostnames.split()[0].decode('utf-8')
        if params.master_port==-1:
            params.master_port = '19500'
        assert 10001 <= int(params.master_port) <= 20000 or params.world_size == 1
        print(PREFIX + "Master address: %s" % params.master_addr)
        print(PREFIX + "Master port   : %i" % int(params.master_port))

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.master_addr
        os.environ['MASTER_PORT'] = str(params.master_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch or torchrun
    elif params.local_rank != -1:

        assert params.master_port == -1

        # read environment variables
        params.global_rank = int(os.environ["RANK"])
        params.world_size = int(os.environ['WORLD_SIZE'])
        params.local_rank = int(os.environ['LOCAL_RANK'])

        # # number of nodes / node ID
        params.n_gpu_per_node = 2
        params.n_nodes = 1
        params.node_id = 0

    # local job (single GPU)
    else:
        assert params.local_rank == -1
        assert params.master_port == -1
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.distributed = params.world_size > 1

    # summary
    PREFIX = "%i - " % params.global_rank
    print(PREFIX + "Number of nodes: %i" % params.n_nodes)
    print(PREFIX + "Node ID        : %i" % params.node_id)
    print(PREFIX + "Local rank     : %i" % params.local_rank)
    print(PREFIX + "Global rank    : %i" % params.global_rank)
    print(PREFIX + "World size     : %i" % params.world_size)
    print(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(params.is_master))
    print(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(params.distributed))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())

    # initialize multi-GPU
    if params.distributed:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        print("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )

        # set GPU device
        torch.cuda.set_device(params.local_rank)
        dist.barrier()
        setup_logging_for_distributed(params.is_master)


def average_metrics(metrics: dict[str, float], count=1.):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unnormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))
