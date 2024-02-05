import torch


def dice_coef(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    thr=0.5,
    dim=(-1, -2),
    smooth=0,
    epsilon=1e-7,
):
    y_pred = y_pred.sigmoid()
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + smooth) / (den + smooth)).mean()  # .clamp_min(epsilon)
    return dice
