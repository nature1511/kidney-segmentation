import torch
from segmentation.config import CFG


def dice_coef(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    thr: float = 0.5,
    dim=(-1, -2),
    smooth: float = CFG.smooth,
    from_logits: bool = True,
    # epsilon=1e-7,
):
    """Calculate Dice coef

    Args:
        y_pred (torch.Tensor): predicitions
        y_true (torch.Tensor): labels
        thr (float, optional): cutoff threshold. Defaults to 0.5.
        dim (tuple, optional): dimensions. Defaults to (-1, -2).
        smooth (int, optional): smooth. Defaults to 0.
        from_logits (bool): predicitions from logits or not

    Returns:
        _type_: _description_
    """
    if from_logits:
        y_pred = y_pred.sigmoid()
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + smooth) / (den + smooth)).mean()  # .clamp_min(epsilon)
    return dice
