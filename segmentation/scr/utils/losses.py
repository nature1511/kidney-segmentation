import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """Dice loss is based on
    segmentation-models-pytorch
    https://stackoverflow.com/questions/67230305/i-want-to-confirm-which-of-these-methods-to-calculate-dice-loss-is-correct
    """

    def __init__(
        self, from_logits: bool = True, smooth: float = 0.0, eps: float = 1e-7
    ):
        super().__init__()
        self.from_logits = from_logits
        self.eps = eps
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert (
            y_pred.shape[0] == y_true.shape[0]
        ), f"y_pred {y_pred.shape} != y_true {y_true.shape}"
        if self.from_logits:
            y_pred = F.logsigmoid(y_pred).exp()
        batch_size = y_pred.shape[0]
        y_pred = y_pred.view(batch_size, 1, -1)
        y_true = y_true.view(batch_size, 1, -1)

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        intersection = torch.sum(y_pred * y_true, dim=(0, 2))
        union = torch.sum(y_pred + y_true, dim=(0, 2))
        dice_score = (2.0 * intersection + self.smooth) / (
            union + self.smooth + 1e-8
        ).clamp_min(self.eps)
        loss = 1.0 - dice_score

        mask = y_true.sum((0, 2)) > 0

        loss *= mask.to(loss.dtype)
        return loss.mean()


class BCE_DICE(nn.Module):
    """Combination of Cross-Entropy and Soft Dice Losses.
    Based on https://arxiv.org/pdf/2209.06078.pdf
    """

    def __init__(
        self,
        mode: str = "BCE",
        num_epoch: int = 1,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.mode = mode
        self.smooth = smooth
        self.eps = eps
        self.n = 0
        self.N = num_epoch - 1
        self.loss = {
            "BCE": self.bce,
            "DICE": self.dice,
            "SUM": self.sum,
            "SOFT_TUNNING": self.soft_tunning,
            "HARD_TUNNING": self.hard_tunning,
        }
        try:
            self.loss[self.mode]
        except KeyError:
            print(f"Incorrect mode:  {self.mode}")
            print("Mod list [BCE, DICE, SUM, SOFT_TUNNING, HARD_TUNNING]")

    def update_n(self):
        """Update counter for Soft Fine-tuning or Hard Fine-tuning."""
        self.n += 1

    def bce(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Binary cross entropy with logits

        Args:
            y_pred (torch.Tensor): predictions from_logits
            y_true (torch.Tensor): label

        """
        batch_size = y_pred.shape[0]
        y_pred = y_pred.view(batch_size, 1, -1)
        y_true = y_true.view(batch_size, 1, -1)
        return F.binary_cross_entropy_with_logits(y_pred, y_true)

    def dice(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Dice loss

        Args:
            y_pred (torch.Tensor): predictions from_logits
            y_true (torch.Tensor): label

        """
        dice_loss = DiceLoss(from_logits=True, smooth=self.smooth, eps=self.eps)
        return dice_loss(y_pred, y_true)

    def sum(self, y_pred, y_true):
        """BCE + Dice loss

        Args:
            y_pred (torch.Tensor): predictions from_logits
            y_true (torch.Tensor): label

        """
        return self.bce(y_pred, y_true) + self.dice(y_pred, y_true)

    def soft_tunning(self, y_pred, y_true):
        """Soft Fine-tuning. We minimize a linear combination that
        starts giving full
            weight to BCE and ends up giving only weight to Dice,
        with intermediate
            weights linearly interpolated.

        Args:
            y_pred (torch.Tensor): predictions from_logits
            y_true (torch.Tensor): label

        """
        if self.n == 0:
            return self.bce(y_pred, y_true)
        beta = self.n / self.N
        return (1.0 - beta) * self.bce(y_pred, y_true) + beta * self.dice(
            y_pred, y_true
        )

    def hard_tunning(self, y_pred, y_true):
        """Hard Fine-tuning.
        We minimize the BCE loss in the first part of the
        training and switch to the Dice loss only
        in the last 10% of the training.

        Args:
            y_pred (torch.Tensor): predictions from_logits
            y_true (torch.Tensor): label

        """
        if self.n < 0.9 * self.N:

            return self.bce(y_pred, y_true)

        return self.dice(y_pred, y_true)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert y_pred.shape[0] == y_true.shape[0]

        return self.loss[self.mode](y_pred, y_true)
