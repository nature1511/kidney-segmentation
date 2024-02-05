import os
import random

import numpy as np
import torch

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
