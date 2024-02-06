from typing import Optional, TypeVar
from pathlib import Path
import pandas as pd
import cv2
from torch.utils.data import Dataset


PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")


def random_sub_df(
    df: PandasDataFrame, sample_limit: Optional[int] = None, empty_tile_pct: int = 10
):
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
    count_no_empty = df["is_empty"].value_counts()[False]
    count_empty = df["is_empty"].value_counts()[True]
    print(f"Dataset contains {count_empty} empty and {count_no_empty} non-empty tiles.")

    num_empty_tiles_to_sample = int(sample_limit * empty_tile_pct / 100)
    num_pos_tiles_to_sample = int(sample_limit * (1 - empty_tile_pct / 100))

    if num_empty_tiles_to_sample > count_empty:
        num_empty_tiles_to_sample = count_empty
        sample_limit = int(count_empty / empty_tile_pct * 100)
        num_pos_tiles_to_sample = int(sample_limit * (1 - empty_tile_pct / 100))

    if num_pos_tiles_to_sample > count_no_empty:
        num_pos_tiles_to_sample = count_no_empty
        sample_limit = int(count_no_empty / (1 - empty_tile_pct / 100))
        num_empty_tiles_to_sample = int(sample_limit * empty_tile_pct / 100)

    print(
        f"Sample {num_empty_tiles_to_sample} empty and {num_pos_tiles_to_sample} non-empty tiles."
    )

    df_empty = df[df["is_empty"] == True].sample(num_empty_tiles_to_sample)
    df_no_empty = df[df["is_empty"] == False].sample(num_pos_tiles_to_sample)
    frames = [df_empty, df_no_empty]
    return pd.concat(frames).sort_index()


class Tilling_Dataset(Dataset):
    """Creating a dataloader for image tiling"""

    def __init__(
        self,
        name_data: str,
        path_to_df: str,
        use_random_sub: bool = False,
        empty_tile_pct: int = 0,
        sample_limit: Optional[int] = None,
        transform=None,
    ):
        super().__init__()
        self.name_data = name_data
        self.path_to_df = Path(path_to_df)
        self.use_random_sub = use_random_sub
        self.empty_tile_pct = empty_tile_pct
        self.sample_limit = sample_limit
        self.transform = transform

        df = pd.read_csv(self.path_to_df)
        if self.use_random_sub:
            self.df = random_sub_df(
                df=df,
                sample_limit=self.sample_limit,
                empty_tile_pct=self.empty_tile_pct,
            )
        else:
            self.df = df

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, idx) -> tuple:
        img_path, lb_path, _, bbx, _, size = self.df.iloc[idx, :].values
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # (H,W ,C)
        mask = cv2.imread(lb_path, cv2.IMREAD_GRAYSCALE)  # .astype('float32')
        if self.transform:
            augmented = self.transform(image=img, mask=mask)  # c h w
            img, mask = augmented["image"], augmented["mask"]

        else:
            img = torch.from_numpy(img)
            img = torch.permute(img, (2, 0, 1))
            mask = torch.from_numpy(mask)

        img = img.type(torch.float32)
        img = img / 255
        mask = mask.type(torch.float32)
        mask = mask / 255

        return img, mask, bbx, size
