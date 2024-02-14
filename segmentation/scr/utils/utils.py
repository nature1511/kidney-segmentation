import os
import random
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

from segmentation.config import CFG


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
) -> None:
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
    path.mkdir(parents=True, exist_ok=True)
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


def min_max_normalization(x: torch.Tensor) -> torch.Tensor:
    """Prepare min / max normalization (x-min)/(max-min)
    Args:
        x (torch.Tensor): input.shape=(batch,f1,...)

    Returns:
        torch.Tensor: normalized tensor
    """
    shape = x.shape
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    min_ = x.min(dim=-1, keepdim=True)[0]
    max_ = x.max(dim=-1, keepdim=True)[0]
    if min_.mean() == 0 and max_.mean() == 1:
        return x.reshape(shape)

    x = (x - min_) / (max_ - min_ + 1e-9)
    return x.reshape(shape)


def resize_to_size(img: torch.Tensor, image_size: int = CFG.image_size) -> torch.Tensor:
    """Resizes the image if its side is smaller than image_size
        padding is the average between the maximum and minimum pixels

    Args:
        img (torch.Tensor): image
        image_size (int): image size. Defaults to CFG.image_size.

    Returns:
        torch.Tensor: _description_
    """
    if image_size > img.shape[0]:

        start = (image_size - img.shape[0]) // 2
        val1 = (min(img[0, :]).item() + max(img[0, :]).item()) // 2
        val2 = (
            min(img[img.shape[0] - 1, :]).item() + max(img[img.shape[0] - 1, :]).item()
        ) // 2

        top = torch.full((start, img.shape[1]), (val1 + val2) // 2).to(torch.uint8)
        botton = torch.full((start, img.shape[1]), (val1 + val2) // 2).to(torch.uint8)

        img = torch.cat((top, img, botton), axis=0).to(torch.uint8)
    if image_size > img.shape[1]:

        start = (image_size - img.shape[1]) // 2
        val1 = (min(img[:, 0]).item() + max(img[:, 0]).item()) // 2
        val2 = (
            min(img[:, img.shape[1] - 1]).item() + max(img[:, img.shape[1] - 1]).item()
        ) // 2
        left = torch.full((img.shape[0], start), (val1 + val2) // 2).to(torch.uint8)
        right = torch.full((img.shape[0], start), (val1 + val2) // 2).to(torch.uint8)
        img = torch.cat((left, img, right), axis=1).to(torch.uint8)

    return img


def norm_with_clip(x: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Percentile normalization

    Args:
        x (torch.Tensor):
        smooth (float): smooth. Defaults to 1e-5.

    Returns:
        torch.Tensor: Percentile normalized input
    """
    dim = list(range(1, x.ndim))
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    x = (x - mean) / (std + smooth)
    x[x > 5] = (x[x > 5] - 5) * 1e-3 + 5
    x[x < -3] = (x[x < -3] + 3) * 1e-3 - 3
    return x


def add_noise(
    x: torch.Tensor, max_randn_rate=0.1, randn_rate=None, x_already_normed=False
):
    """input.shape=(batch,f1,f2,...) output's var will be normalizate"""
    ndim = x.ndim - 1
    if x_already_normed:
        x_std = torch.ones([x.shape[0]] + [1] * ndim, device=x.device, dtype=x.dtype)
        x_mean = torch.zeros([x.shape[0]] + [1] * ndim, device=x.device, dtype=x.dtype)
    else:
        dim = list(range(1, x.ndim))
        x_std = x.std(dim=dim, keepdim=True)
        x_mean = x.mean(dim=dim, keepdim=True)
    if randn_rate is None:
        randn_rate = (
            max_randn_rate
            * np.random.rand()
            * torch.rand(x_mean.shape, device=x.device, dtype=x.dtype)
        )
    cache = (x_std**2 + (x_std * randn_rate) ** 2) ** 0.5
    # https://blog.csdn.net/chaosir1991/article/details/106960408

    return (
        x
        - x_mean
        + torch.randn(size=x.shape, device=x.device, dtype=x.dtype) * randn_rate * x_std
    ) / (cache + 1e-7)


def filter_noise(x: torch.Tensor) -> torch.Tensor:
    TH = x.reshape(-1)
    index = -int(len(TH) * CFG.chopping_percentile)
    TH: int = np.partition(TH, index)[index]
    x[x > TH] = int(TH)
    ########################################################################
    TH = x.reshape(-1)
    index = -int(len(TH) * CFG.chopping_percentile)
    TH: int = np.partition(TH, -index)[-index]
    x[x < TH] = int(TH)
    return x


def create_img_lb_paths(path_img_dir, path_lb_dir):
    path_img_dir = Path(path_img_dir)
    path_lb_dir = Path(path_lb_dir)

    path_img_dir = sorted(list(path_img_dir.rglob("*.tif")))
    path_lb_dir = sorted(list(path_lb_dir.rglob("*.tif")))

    images_labels = defaultdict(list)
    for img in path_img_dir:
        images_labels[img.name].append(img)
    for lb in path_lb_dir:
        images_labels[lb.name].append(lb)
    new_dict = dict(filter(lambda item: len(item[1]) > 1, images_labels.items()))
    print(f"Общее число изображений с масками сегментации : {len(new_dict)}")
    img_path = list(map(lambda key: str(new_dict[key][0]), new_dict))
    lb_path = list(map(lambda key: str(new_dict[key][1]), new_dict))
    return img_path, lb_path
