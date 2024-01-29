import pandas as pd
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
import rasterio
from tqdm import tqdm
import numpy as np
import cv2
import warnings
from segmentation.config import Configs as CFG
from segmentation.scr.rle_coding import *


from functools import lru_cache

warnings.filterwarnings(
    "ignore", category=rasterio.errors.NotGeoreferencedWarning)


@lru_cache(maxsize=128)
def ropen(img_fpath):
    """
        Image reading and caching for speedup

    Args:
        img_fpath : path to image

    """
    return rasterio.open(img_fpath)


class Generate_Tiled_Dataset(Dataset):
    """
    Сlass for generating tilling images and storing them in folders

    https://www.kaggle.com/code/squidinator/sennet-hoa-in-memory-tiled-dataset-pytorch
    """

    def __init__(
        self,
        name_data: str,
        path_img_dir: str,
        path_lb_dir: str,
        tile_size: List = CFG.tile_size,
        overlap_pct: int = CFG.overlap_pct,
        strong_empty: bool = True,
        sample_limit: int = 30000 * 10,
        cache_dir: str = CFG.cache_dir,
    ):
        """
            Сlass initialization
        Args:
            name_data str: dataset_name
            path_img_dir str: path to images
            path_lb_dir str: path to labels
            tile_size List : tilling size. Defaults to CFG.tile_size.
            overlap_pct int: minimum overlap 
            of tillings as a percentage. Defaults to CFG.overlap_pct.

            strong_empty bool: stronger empty image filter. Defaults to True.
            sample_limit int: limit the size of new images. Defaults to 30000*10.
            cache_dir str: path to new images. Defaults to CFG.cache_dir.
        """

        self.name_data = name_data
        self.path_img_dir = Path(path_img_dir)
        self.path_lb_dir = Path(path_lb_dir)
        self.tile_size = np.array(tile_size)
        self.overlap_pct = overlap_pct
        self.strong_empty = strong_empty
        self.sample_limit = sample_limit
        self.cache_dir = cache_dir

        self.path_img_dir = sorted(list(self.path_img_dir.rglob("*.tif")))
        self.samples = []

        for p_img in tqdm(
            self.path_img_dir, total=len(self.path_img_dir), desc="Reading images"
        ):
            p_lb = self.path_lb_dir / p_img.name
            with rasterio.open(p_img) as reader:
                width, height = reader.width, reader.height
                img_r = reader.read()
                px_max, px_min = img_r.max(), img_r.min()
            # print(p_lb)
            self.samples.append(
                (p_img, p_lb, [px_min, px_max], (height, width)))

        min_overlap = float(overlap_pct) * 0.01
        max_stride = self.tile_size * (1.0 - min_overlap)

        list_tiles = []
        empty = 0
        nonempty = 0

        for file_path, label_path, px_stats, img_dims in tqdm(
            self.samples, total=len(self.samples), desc="Generating tiles"
        ):
            # [(C,H,W),...]
            height, width = img_dims
            num_patches = np.ceil(np.array([height, width]) / max_stride).astype(
                np.int64
            )
            starts = [
                np.int32(np.linspace(
                    0, height - self.tile_size[0], num_patches[0])),
                np.int32(np.linspace(
                    0, width - self.tile_size[1], num_patches[1])),
            ]
            stops = [starts[0] + self.tile_size[0],
                     starts[1] + self.tile_size[1]]
            mask = cv2.imread(
                str(label_path), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            rle = rle_encode(mask)

            for y1, y2 in zip(starts[0], stops[0]):
                for x1, x2 in zip(starts[1], stops[1]):
                    mask_tile = mask[y1:y2, x1:x2]
                    is_empty = np.all(mask_tile == 0)

                    if self.strong_empty:
                        is_empty = is_empty or (
                            mask_tile.sum() < (0.05 * self.tile_size[0])
                        )
                    if is_empty:
                        empty += 1
                    else:
                        nonempty += 1

                    list_tiles.append(
                        (
                            file_path,
                            rle,
                            is_empty,
                            (x1, y1, x2 - x1, y2 - y1),
                            px_stats,
                            (height, width),
                        )
                    )
        print(
            f"Dataset contains {empty} empty and {nonempty} non-empty tiles.")

        self.tiles = []
        if self.cache_dir:
            if isinstance(self.cache_dir, str):
                self.cache_dir = Path(self.cache_dir)
            self.cache_dir_img = self.cache_dir / self.name_data / "images"
            self.cache_dir_lb = self.cache_dir / self.name_data / "labels"

            self.cache_dir_img.mkdir(parents=True, exist_ok=True)
            self.cache_dir_lb.mkdir(parents=True, exist_ok=True)

        self.df = pd.DataFrame(
            columns=["path_img", "path_lb",
                     "is_empty", "bbx", "px_stats", "size"]
        )
        self.path_df = self.cache_dir / (self.name_data + ".csv")

        if self.sample_limit < len(list_tiles):
            pos_idxs_to_sample = np.random.choice(
                len(list_tiles), min(sample_limit, len(list_tiles)), replace=False
            )
            self.tiles = list(map(list_tiles.__getitem__, pos_idxs_to_sample))
        else:
            self.tiles = list_tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple:
        """
            Writing an image to a folder by index and writing to a dataframe

        Args:
            idx int: index

        Returns:
            tuple: str(img_fpath), rle, is_empty, bbox
        """

        img_fpath, rle, is_empty, bbox, px_stats, size = self.tiles[idx]

        base_name = img_fpath.name.split(".")[0]
        base_name += f"_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png"

        img_path = self.cache_dir_img / base_name
        lb_path = self.cache_dir_lb / base_name

        img = ropen(img_fpath).read(
            1, window=rasterio.windows.Window(*bbox) if len(bbox) > 0 else None
        )
        if img.ndim == 3:
            img = np.mean(img, axis=2)

        # If we read the window from the full scene, compress the dynamic range to UINT8
        # TODO: Save full scene statistics and compress relative to those, rather than tile level statistics
        img = (img - px_stats[0]) / (px_stats[1] - px_stats[0])
        img *= 255.0
        img = img.astype(np.uint8)
        cv2.imwrite(str(img_path), img)

        mask = rle_decode(rle, size).astype(np.uint8)

        if len(bbox) > 0:
            x, y, w, h = bbox
            mask = mask[y: y + h, x: x + w]

        cv2.imwrite(str(lb_path), mask)
        self.df.loc[idx, :] = [img_path, lb_path,
                               is_empty, bbox, px_stats, size]
        return (
            str(img_fpath),
            rle,
            is_empty,
            bbox,
        )
