import torch
from segmentation.config import CFG


def dice_coef(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    thr=0.5,
    dim=(-1, -2),
    smooth=CFG.smooth,
    # epsilon=1e-7,
):
    """Calculate Dice coef

    Args:
        y_pred (torch.Tensor): predicitions from logits
        y_true (torch.Tensor): labels
        thr (float, optional): cutoff threshold. Defaults to 0.5.
        dim (tuple, optional): dimensions. Defaults to (-1, -2).
        smooth (int, optional): smooth. Defaults to 0.

    Returns:
        _type_: _description_
    """
    y_pred = y_pred.sigmoid()
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + smooth) / (den + smooth)
            ).mean()  # .clamp_min(epsilon)
    return dice
