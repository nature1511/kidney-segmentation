import os
import random
import numpy as np
import torch
from pathlib import Path
from segmentation.config import Configs as CFG


def set_seed(seed: int = CFG.random_seed) -> None:
    """Makes results reproducible

    Args:
        seed (int, optional): random seed. Defaults to CFG.random_seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_model(
    model,
    optimizer=None,
    model_name="model_name",
    path="weight",
    lr_scheduler=None,
):
    """Saving the model and training parameters

    Args:
        model : model
        optimizer : optimizer. Defaults to None.
        model_name (str): model_name_. Defaults to "model_name".
        path (str): Path to save. Defaults to "weight".
        lr_scheduler (_type_, optional): lr_scheduler. Defaults to None.
    """
    if isinstance(path, str):
        path = Path(path)
    path_save_state = path / ("state " + model_name + ".pth")
    state = {
        "model_name": model_name,
        "model": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "lr_scheduler_state": (
            None if lr_scheduler is None else lr_scheduler.state_dict()
        ),
    }
    torch.save(state, path_save_state)
