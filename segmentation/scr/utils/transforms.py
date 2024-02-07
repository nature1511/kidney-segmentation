from typing import Optional

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from segmentation.config import Configs as CFG


def get_transform(transform_type: Optional[str] = "weak"):
    """Generate transform for data
    Args:
        transform_type (str): transform for train or test. Defaults to 'train'.
        list transform_type ["weak", "strong" , "val"]
    """

    if transform_type == "weak":
        return A.Compose(
            [
                A.VerticalFlip(p=CFG.p_rot),
                A.HorizontalFlip(p=CFG.p_rot),
                A.RandomRotate90(p=CFG.p_rot),
                A.ElasticTransform(
                    alpha=60, sigma=190 * 0.05, alpha_affine=2, p=CFG.p_aug
                ),
                A.RandomBrightnessContrast(),
                A.CLAHE(clip_limit=5),
                ToTensorV2(p=1),
                A.Normalize(mean=CFG.mean_till1, std=CFG.std_till1, p=1),
            ],
            p=1,
        )

    if transform_type == "strong":
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.Rotate(limit=30, p=CFG.p_rot),
                        A.Compose(
                            [
                                A.Transpose(p=CFG.p_rot),
                                A.VerticalFlip(p=CFG.p_rot),
                                A.HorizontalFlip(p=CFG.p_rot),  # ])
                            ]
                        ),
                    ]
                ),
                # A.HueSaturationValue(15, 15, 10, p=1)
                # A.CLAHE(clip_limit=6)
                A.OneOf(
                    [
                        A.GridDistortion(p=CFG.p_aug),
                        A.ElasticTransform(
                            alpha=80, sigma=160 * 0.05, alpha_affine=5, p=CFG.p_aug
                        ),
                    ]
                ),
                A.OneOf(
                    [
                        A.Rotate(limit=30, p=CFG.p_rot),
                        A.Compose(
                            [
                                A.Transpose(p=CFG.p_rot),
                                A.VerticalFlip(p=CFG.p_rot),
                                A.HorizontalFlip(p=CFG.p_rot),
                            ]
                        ),
                    ]
                ),
                A.OneOf(
                    [
                        A.GaussNoise(p=CFG.p_aug, var_limit=[10, 100]),
                        A.RandomGamma(p=CFG.p_aug),
                    ]
                ),
                A.OneOf(
                    [
                        A.HueSaturationValue(10, 15, 10),
                        A.CLAHE(clip_limit=4),
                        A.RandomBrightnessContrast(),
                    ]
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=25,
                    border_mode=0,
                    p=CFG.p_rot,
                ),
                ToTensorV2(p=1),
                A.Normalize(mean=CFG.mean_till1, std=CFG.std_till1, p=1),
            ],
            p=1,
        )

    if transform_type == "val":
        return A.Compose(
            [
                # A.Resize(img_size, img_size, p=1.0),
                ToTensorV2(p=1),
                A.Normalize(mean=CFG.mean_till1, std=CFG.std_till1, p=1),
            ],
            p=1.0,
        )

    return None
