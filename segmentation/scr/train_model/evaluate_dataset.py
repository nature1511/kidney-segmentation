import torch
from tqdm import tqdm
from ..utils.utils import *
from segmentation.scr.utils.metrics import dice_coef


def evaluate_dataset(model, val_loader):
    model.eval()
    timer = tqdm(range(len(val_loader)))
    val_scores = 0
    scores = []
    for i, (x, y) in enumerate(val_loader):
        x = x.cuda().to(torch.float32)
        y = y.cuda().to(torch.float32)
        x = norm_with_clip(x.reshape(-1, *x.shape[2:])).reshape(x.shape)

        with torch.no_grad():
            pred = model(x)

        score = dice_coef(pred.detach(), y)
        scores.append(score.detach().cpu().item())
        val_scores = (val_scores * i + score) / (i + 1)
        timer.set_description(f"eval--> score:{val_scores:.4f}")
        timer.update()
    timer.close()
    return scores, val_scores
