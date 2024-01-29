import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, List, TypeVar
from pathlib import Path
import rasterio
from tqdm import tqdm
import numpy as np
import cv2
import warnings
from segmentation.config import Configs as CFG
from segmentation.scr.rle_coding import *
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
def random_sub_df(df: PandasDataFrame, sample_limit: Optional[int] = None, empty_tile_pct: int = 10):
    """_summary_

    Args:
        df PandasDataFrame: dataframe
        sample_limit int: sample limit. Defaults to None.
        empty_tile_pct int: percentage of empty masks. Defaults to 10.

    Returns:
        PandasDataFrame: dataframe with a certain percentage of empty masks
    """
    if sample_limit is None:
        sample_limit = df.shape[0]
    print(sample_limit)
    # pct_no_empty = df['is_empty'].value_counts(normalize=True)[False]
    # pct_empty = df['is_empty'].value_counts(normalize=True)[True]
    count_no_empty = df['is_empty'].value_counts()[False]
    count_empty = df['is_empty'].value_counts()[True]
    print(
        f"Dataset contains {count_empty} empty and {count_no_empty} non-empty tiles.")

    num_empty_tiles_to_sample = int(sample_limit * empty_tile_pct / 100)
    num_pos_tiles_to_sample = int(sample_limit * (1 - empty_tile_pct / 100))

    if num_empty_tiles_to_sample > count_empty:
        num_empty_tiles_to_sample = count_empty
        sample_limit = int(count_empty / empty_tile_pct * 100)
        num_pos_tiles_to_sample = int(
            sample_limit * (1 - empty_tile_pct / 100))

    if num_pos_tiles_to_sample > count_no_empty:
        num_pos_tiles_to_sample = count_no_empty
        sample_limit = int(count_no_empty / (1 - empty_tile_pct / 100))
        num_empty_tiles_to_sample = int(sample_limit * empty_tile_pct / 100)

    print(
        f"Sample {num_empty_tiles_to_sample} empty and 
        {num_pos_tiles_to_sample} non-empty tiles.")

    df_empty = df[df['is_empty'] == True].sample(num_empty_tiles_to_sample)
    df_no_empty = df[df['is_empty'] == False].sample(num_pos_tiles_to_sample)
    frames = [df_empty, df_no_empty]
    return pd.concat(frames).sort_index()