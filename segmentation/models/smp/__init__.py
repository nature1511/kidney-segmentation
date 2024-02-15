import torch
import torch.nn as nn
from ...config import CFG
from .decoder import Unet


class CustomModel(nn.Module):
    def __init__(self, CFG=CFG, weight=None):
        super().__init__()
        self.model = Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=weight,
            in_channels=CFG.in_chans,
            classes=CFG.target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.model(image)

        return output[:, 0]


class WrappedModel(nn.Module):
    def __init__(self, CFG=CFG):
        super(WrappedModel, self).__init__()
        self.module = CustomModel(CFG=CFG)

    def forward(self, x):
        return self.module(x)


def get_pretrained_model(
    path_to_model: str = CFG.path_to_state_model, train_parallel: bool = False, CFG=CFG
):
    """Returns the pretrained model

    Args:
        path_to_model (str): path to weigth. Defaults to CFG.path_to_state_model.
        train_parallel (bool, optional): the model was trained on one or more GPUs. Defaults to False.
    """
    if train_parallel:
        model = WrappedModel(CFG=CFG)
    else:
        model = CustomModel(CFG=CFG)
    model = model.to(CFG.device)

    if not torch.cuda.is_available() or CFG.device == "cpu":
        ckpt = torch.load(path_to_model, map_location=torch.device("cpu"))
    elif CFG.device == "cuda" and torch.cuda.is_available():
        ckpt = torch.load(path_to_model)
    model.load_state_dict(ckpt)

    return model
