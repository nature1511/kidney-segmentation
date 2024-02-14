import torch
from tqdm import tqdm
import numpy as np
import time
from segmentation.config import CFG

from ..utils.utils import *
from segmentation.scr.utils.metrics import dice_coef
from segmentation.scr.train_model.train_one_loop import train_one_loop


def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler=None):

    train_metrics, val_metrics, train_losses, val_losses = [], [], [], []
    best_metric = -np.inf
    total_time = 0
    for epoch in range(CFG.epochs):

        start = time.time()
        losss, scores = train_one_loop(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            scheduler=scheduler,
        )

        train_losses.append(losss)
        train_metrics.append(scores)

        model.eval()
        timer = tqdm(range(len(val_loader)))
        val_losss = 0
        val_scores = 0
        for i, (x, y) in enumerate(val_loader):
            x = x.cuda().to(torch.float32)
            y = y.cuda().to(torch.float32)
            x = norm_with_clip(x.reshape(-1, *x.shape[2:])).reshape(x.shape)

            with torch.no_grad():
                pred = model(x)
                loss = loss_fn(pred, y)
            score = dice_coef(pred.detach(), y)
            val_losss = (val_losss * i + loss.item()) / (i + 1)
            val_scores = (val_scores * i + score) / (i + 1)
            timer.set_description(f"val-->loss:{val_losss:.4f},score:{val_scores:.4f}")
            timer.update()

        val_metrics.append(val_scores)
        val_losses.append(val_losss)
        if val_scores > best_metric:
            best_metric = val_scores
            save_model(
                model=model,
                optimizer=optimizer,
                model_name=model.__class__.__name__
                + "_best_model_at_"
                + str(epoch + 1),
                path=CFG.path_to_save_state_model,
                lr_scheduler=scheduler,
            )
        with open("train_results.txt", "w") as file_handler:
            file_handler.write("train_loss\n")
            for item in train_losses:
                file_handler.write("{}\t".format(item))

            file_handler.write("\nval_loss\n")
            for item in val_losses:
                file_handler.write("{}\t".format(item))

            file_handler.write("\nval_metrics\n")
            for item in val_metrics:
                file_handler.write("{}\t".format(item))

            file_handler.write("\ntrain_metrics\n")
            for item in train_metrics:
                file_handler.write("{}\t".format(item))

        end = time.time()
        total_time += end - start
        timer.set_description(
            f"val-->loss:{val_losss:.4f},score:{val_scores:.4f}.___Took {((end - start) / 60):.3f} minutes for epoch {epoch + 1}"
        )
        timer.update()

        timer.close()

    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            total_time // 3600, (total_time % 3600) // 60, (total_time % 3600) % 60
        )
    )

    timer.close()
