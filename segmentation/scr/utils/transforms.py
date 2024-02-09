
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from segmentation.config import CFG


def get_transform(transform_type: str = "train"):
    """Generate transform for data
    Args:
        transform_type (str): transform for train or test. Defaults to 'train'.
        list transform_type ["train" , "val"]
    """

    if transform_type == "train":
        return A.Compose(
            [
                A.VerticalFlip(p=CFG.p_rot),
                A.HorizontalFlip(p=CFG.p_rot),
                A.RandomRotate90(p=CFG.p_rot),
                A.Affine(scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
                         translate_percent={"x": (0, 0.1), "y": (0, 0.1)},
                         rotate=(-30, 30), shear=(-5, 5), p=CFG.p_aug),
                A.RandomBrightnessContrast(
                    brightness_limit=0.4, contrast_limit=0.4, p=CFG.p_aug),
                A.OneOf([
                    A.ElasticTransform(
                        alpha=1, sigma=50, alpha_affine=10, border_mode=1, p=CFG.p_aug),
                    A.GridDistortion(
                        num_steps=5, distort_limit=0.1, border_mode=1, p=CFG.p_aug)
                ], p=CFG.p_aug),


                A.GaussNoise(var_limit=0.05, p=0.2),
                # A.CLAHE(clip_limit=5),
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
