import torch
from tqdm import tqdm
from ..utils.utils import *
from segmentation.scr.utils.metrics import dice_coef

from segmentation.config import CFG


def train_one_loop(model, train_loader, loss_fn, optimizer, epoch, scheduler=None):
    model.train()
    timer = tqdm(range(len(train_loader)))
    losss = 0
    scores = 0

    for i, (x, y) in enumerate(train_loader):
        x = x.cuda().to(torch.float32)
        y = y.cuda().to(torch.float32)
        x = norm_with_clip(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
        x = add_noise(x, max_randn_rate=0.5, x_already_normed=True)

        pred = model(x)
        loss = loss_fn(pred, y)
        loss = loss / CFG.n_accumulate

        loss.backward()  # loss.backward()  # backward-pass

        if (i + 1) % CFG.n_accumulate == 0 or (i + 1 == len(train_loader)):

            optimizer.step()  # update weights
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        score = dice_coef(pred.detach(), y)
        losss = (losss * i + loss.item()) / (i + 1)
        scores = (scores * i + score) / (i + 1)
        timer.set_description(
            f"epoch:{epoch},loss:{losss:.4f},score:{scores:.4f},lr{optimizer.param_groups[0]['lr']:.4e}"
        )
        timer.update()
        del loss, pred

    timer.close()
    return losss, scores
