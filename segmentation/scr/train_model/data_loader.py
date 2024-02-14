import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from tqdm import tqdm

from ..utils.utils import *

from segmentation.config import CFG

from segmentation.scr.utils import transforms


class Data_loader(Dataset):
    def __init__(self, paths, is_label):
        self.paths = paths
        self.is_label = is_label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index], cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img)
        if self.is_label:
            img = (img != 0).to(torch.uint8) * 255
        else:
            img = img.to(torch.uint8)
        return img


def load_data(paths, is_label=False):
    data_loader = Data_loader(paths, is_label)
    data_loader = DataLoader(data_loader, batch_size=16)
    data = []
    for x in tqdm(data_loader):
        data.append(x)
    x = torch.cat(data, dim=0)
    del data
    if not is_label:
        #  ########################################################################
        x = filter_noise(x)
        x = (min_max_normalization(x.to(torch.float16)[None])[0] * 255).to(torch.uint8)
    return x


class Kaggld_Dataset(Dataset):
    def __init__(self, x: list, y: list, arg=False):
        super(Dataset, self).__init__()
        self.x = x  # list[(C,H,W),...]
        self.y = y  # list[(C,H,W),...]
        self.image_size = CFG.image_size
        self.in_chans = CFG.in_chans
        self.arg = arg
        if arg:
            self.transform = transforms.get_transform(transform_type="train")
        else:
            self.transform = transforms.get_transform(transform_type="val")

    def __len__(self) -> int:
        return sum([y.shape[0] - self.in_chans for y in self.y])

    def __getitem__(self, index):
        i = 0
        for x in self.x:
            if index > x.shape[0] - self.in_chans:
                index -= x.shape[0] - self.in_chans
                i += 1
            else:
                break
        x = self.x[i]
        y = self.y[i]

        x_index = np.random.randint(0, x.shape[1] - self.image_size)
        y_index = np.random.randint(0, x.shape[2] - self.image_size)

        x = x[
            index : index + self.in_chans,
            x_index : x_index + self.image_size,
            y_index : y_index + self.image_size,
        ]
        y = y[
            index + self.in_chans // 2,
            x_index : x_index + self.image_size,
            y_index : y_index + self.image_size,
        ]

        data = self.transform(image=x.numpy().transpose(1, 2, 0), mask=y.numpy())
        x = data["image"]
        y = data["mask"] >= 127

        if self.arg:
            if np.random.randint(2):
                x = x.flip(dims=(0,))

        return x, y
