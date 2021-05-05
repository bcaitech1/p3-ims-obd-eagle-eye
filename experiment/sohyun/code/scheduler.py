import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

_scheduler_dict = {
    "step_lr": StepLR,
    "Reduce_lr": ReduceLROnPlateau,
    "cosine_lr": CosineAnnealingLR,
}


def scheduler_entrypoint(scheduler_name):
    return _scheduler_dict[scheduler_name]


def create_scheduler(scheduler_name, optimizer):

    create_fn = scheduler_entrypoint(scheduler_name)
    if scheduler_name == "step_lr":
        scheduler = create_fn(optimizer, step_size=5, gamma=0.5)
    elif scheduler_name == "Reduce_lr":
        scheduler = create_fn(optimizer, factor=0.1, patience=1)
    elif scheduler_name == "cosine_lr":
        scheduler = create_fn(optimizer, T_max=2)
    return scheduler

