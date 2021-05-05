import os
import torch


def save_model(model, model_path):
    check_point = {"net": model.state_dict()}
    torch.save(model.state_dict(), model_path)
