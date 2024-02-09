import torch
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import gc
import time
from colorama import Fore, Style
from segmentation.scr.utils.metrics import dice_coef
from segmentation.config import CFG
from segmentation.scr.utils.utils import save_model


c_ = Fore.GREEN
sr_ = Style.RESET_ALL

# from segmentation.scr
pd.options.mode.chained_assignment = None


def train_one_loop(
    model, optimizer, loss_func, train_loader, device=CFG.device, grad_clip=None
):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0

    dataset_size = 0
    optimizer.zero_grad()
    pbar = tqdm(enumerate(train_loader), total=len(
        train_loader), desc="Train ")
    for step, batch in pbar:
        images, masks, _, _ = batch
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.shape[0]
        dataset_size += batch_size

        y_pred = model(images)

        loss = loss_func(y_pred, masks)
        loss = loss / CFG.n_accumulate
        loss.backward()  # loss.backward()  # backward-pass

        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        del images
        del masks
        del y_pred

        if (step + 1) % CFG.n_accumulate == 0 or (step + 1 == len(train_loader)):
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=CFG.clip_norm
                )
            optimizer.step()  # update weights
            optimizer.zero_grad()

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            epoch=f"{step + 1}",
            train_loss=f"{running_loss / dataset_size:0.4f}",
            # train_dice = f'{train_dice:0.4f}',
            # train_jaccard = f'{train_jaccard:0.4f}',
            lr=f"{current_lr:0.5f}",
            gpu_mem=f"{mem:0.2f} GB",
        )
    epoch_loss = running_loss / dataset_size

    gc.collect()
    torch.cuda.empty_cache()

    return epoch_loss


def valid_one_epoch(model, dataloader, loss_func, device=CFG.device):
    # (model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    val_scores = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid ")
    for _, batch in pbar:
        images, masks, _, _ = batch
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)
        with torch.no_grad():

            y_pred = model(images)
            loss = loss_func(y_pred, masks)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        val_dice = dice_coef(
            y_pred=y_pred, y_true=masks).cpu().detach().numpy()
        val_scores.append([val_dice])
        del images
        del masks
        del y_pred
        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            valid_loss=f"{running_loss / dataset_size:0.4f}",
            gpu_memory=f"{mem:0.2f} GB",
        )
    val_score = np.mean(val_scores, axis=0)
    epoch_loss = running_loss / dataset_size

    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss, val_score[0]


def train_model(
    model,
    optimizer,
    loss_func,
    train_loader,
    val_loader,
    num_epochs=1,
    grad_clip=None,
    scheduler=None,
    device=CFG.device,
    path_to_save=CFG.path_to_save_state_model,
):

    best_metric = -np.inf
    metrics_mas, train_losses, val_losses = [], [], []
    total_time = 0
    for epoch in range(num_epochs):
        start = time.time()
        print()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)
        train_loss = train_one_loop(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            train_loader=train_loader,
            grad_clip=grad_clip,
            device=device,
        )
        val_loss, dice_metric = valid_one_epoch(
            model=model, loss_func=loss_func, dataloader=val_loader, device=device
        )
        train_losses.append(train_loss)
        metrics_mas.append(val_loss)
        metrics_mas.append(dice_metric)
        if scheduler:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(dice_metric)
            else:
                scheduler.step()
        if loss_func.__class__.__name__ == 'BCE_DICE':
            loss_func.update_n()

        # deep copy the model
        print(f"Epoch #{epoch+1} train loss: {train_loss:.3f}")
        print(f"Epoch #{epoch+1} val loss: {val_loss:.3f}")
        print(f"Epoch #{epoch+1} dice_metric: {dice_metric}")

        if dice_metric > best_metric:
            print(f"{c_}Valid metrics Improved ({best_metric} ---> {dice_metric})")
            best_metric = dice_metric
            save_model(
                model=model,
                optimizer=optimizer,
                model_name=model.__class__.__name__
                + "_best_model_at_"
                + str(epoch + 1),
                path=path_to_save,
                lr_scheduler=scheduler,
            )

        with open("train_results.txt", "w") as file_handler:
            file_handler.write("train_loss\n")
            for item in train_losses:
                file_handler.write("{}\t".format(item))

            file_handler.write("\nval_loss\n")
            for item in val_losses:
                file_handler.write("{}\t".format(item))

            file_handler.write("\ndice_metric\n")
            for item in metrics_mas:
                file_handler.write("{}\t".format(item))

        end = time.time()
        total_time += end - start
        print(f"{sr_}Took {((end - start) / 60):.3f} minutes for epoch {epoch + 1}")

    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            total_time // 3600, (total_time %
                                 3600) // 60, (total_time % 3600) % 60
        )
    )
    print(train_losses)
    print()
    print(val_losses)
    print()
    print(metrics_mas)
