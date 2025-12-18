# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math

import torch
import timm.optim as optim
import timm.scheduler as scheduler

from contextlib import contextmanager


class ScalingScheduler:
    """
    Set the scaling parameter depending on the epoch.
    Ex:
        "Linear,scaling_min=0.05,epochs=100"
        ScalingScheduler(model, "scaling", "linear", 0.3, 0.05, 100)
    """
    def __init__(
        self, 
        obj: object,
        attribute: str,
        name: str,
        scaling_o: float,
        scaling_min: float,
        epochs: int,
        start_epoch: int = 0,
        end_epoch: int = None
    ):
        self.obj = obj
        self.attribute = attribute
        self.name = name.lower()
        self.scaling = scaling_o
        self.scaling_min = scaling_min
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch if end_epoch is not None else start_epoch + epochs
    
    @staticmethod
    def linear_scaling(value_o, value_f, epoch, epochs):
        return value_o + (value_f - value_o) * epoch / epochs

    @staticmethod
    def cosine_scaling(value_o, value_f, epoch, epochs):
        return value_f + 0.5 * (value_o - value_f) * (1 + math.cos(epoch / epochs * math.pi))

    def step(self, epoch):
        if epoch < self.start_epoch:
            new_scaling = self.scaling
        elif epoch > self.end_epoch:
            new_scaling = self.scaling_min
        else:
            epoch_in_schedule = epoch - self.start_epoch
            if self.name == "none" or self.name == "constant":
                new_scaling = self.scaling
            elif self.name == "linear":
                new_scaling = self.linear_scaling(self.scaling, self.scaling_min, epoch_in_schedule, self.epochs)
            elif self.name == "cosine":
                new_scaling = self.cosine_scaling(self.scaling, self.scaling_min, epoch_in_schedule, self.epochs)
            else:
                raise ValueError(f"Unknown scaling schedule '{self.name}'")
        setattr(self.obj, self.attribute, new_scaling)
        return new_scaling

@contextmanager
def freeze_grads(model):
    """
    Temporarily freezes the parameters of a PyTorch model.
    Args:
        model (torch.nn.Module): The model whose parameters will be frozen.
    """
    original_requires_grad = {}
    for param in model.parameters():
        original_requires_grad[param] = param.requires_grad
        param.requires_grad = False
    try:
        yield
    finally:
        for param, requires_grad in original_requires_grad.items():
            param.requires_grad = requires_grad

def parse_params(s):
    """
    Parse parameters into a dictionary, used for optimizer and scheduler parsing.
    Example: 
        "SGD,lr=0.01" -> {"name": "SGD", "lr": 0.01}
    """
    s = s.replace(' ', '').split(',')
    params = {}
    params['name'] = s[0]
    for x in s[1:]:
        x = x.split('=')
        params[x[0]]=float(x[1])
    return params

def build_optimizer(
    model_params, 
    name, 
    **optim_params
) -> torch.optim.Optimizer:
    """ Build optimizer from a dictionary of parameters """
    tim_optimizers = sorted(name for name in optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(optim.__dict__[name]))
    torch_optimizers = sorted(name for name in torch.optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.__dict__[name]))
    if hasattr(optim, name):
        return getattr(optim, name)(model_params, **optim_params)
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(model_params, **optim_params)
    raise ValueError(f'Unknown optimizer "{name}", choose among {str(tim_optimizers+torch_optimizers)}')

def build_lr_scheduler(
    optimizer, 
    name, 
    **lr_scheduler_params
) -> torch.optim.lr_scheduler._LRScheduler:
    """ 
    Build scheduler from a dictionary of parameters 
    Args:
        name: name of the scheduler
        optimizer: optimizer to be used with the scheduler
        params: dictionary of scheduler parameters
    Ex:
        CosineLRScheduler, optimizer {t_initial=50, cycle_mul=2, cycle_limit=3, cycle_decay=0.5, warmup_lr_init=1e-6, warmup_t=5}
    """
    if name == "None" or name == "none":
        return None
    tim_schedulers = sorted(name for name in scheduler.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(scheduler.__dict__[name]))
    torch_schedulers = sorted(name for name in torch.optim.lr_scheduler.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.lr_scheduler.__dict__[name]))
    if hasattr(scheduler, name):
        return getattr(scheduler, name)(optimizer, **lr_scheduler_params)
    elif hasattr(torch.optim.lr_scheduler, name):
        return getattr(torch.optim.lr_scheduler, name)(optimizer, **lr_scheduler_params)
    raise ValueError(f'Unknown scheduler "{name}", choose among {str(tim_schedulers+torch_schedulers)}')

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    Args:
        ckp_path: path to the checkpoint
        run_variables: dictionary of variables to re-load
        kwargs: dictionary of objects to re-load. The key is the name of the object in the checkpoint file, the value is the object to load.
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=True)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                try:
                    msg = value.load_state_dict(checkpoint[key], strict=True)
                except:
                    checkpoint[key] = {k.replace("module.", ""): v for k, v in checkpoint[key].items()}
                    msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))
    print(flush=True)

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # test the optimizer
    params = parse_params("SGD,lr=0.01")
    print(params)
    model_params = torch.nn.Linear(10, 10).parameters()
    optimizer = build_optimizer(**params, model_params=model_params)
    print(optimizer)

    # test the scheduler
    params = parse_params("CosineLRScheduler,t_initial=50,cycle_mul=2,cycle_limit=3,cycle_decay=0.5,warmup_lr_init=1e-6,warmup_t=5")
    print(params)
    lr_scheduler = build_lr_scheduler(optimizer=optimizer, **params)
    print(lr_scheduler)

    # test the schedules
    class Test:
        def __init__(self, scaling):
            self.scaling = scaling

    scaling_o = 0.3
    scaling_min = 0.1
    # create and save plots
    scaling_sched = ScalingScheduler(Test, "scaling", "linear", scaling_o, scaling_min, 100)
    print("Linear: ", [scaling_sched.step(ii) for ii in range(100)])
    plt.plot([scaling_sched.step(ii) for ii in range(100)], label="linear")
    scaling_sched = ScalingScheduler(Test, "scaling", "cosine", scaling_o, scaling_min, 100)
    print("Cosine: ", [scaling_sched.step(ii) for ii in range(100)])
    plt.plot([scaling_sched.step(ii) for ii in range(100)], label="cosine")
    plt.savefig("schedules.png")
