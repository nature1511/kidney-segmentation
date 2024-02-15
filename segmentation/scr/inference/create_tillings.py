import numpy as np
from ...config import CFG
from ..utils.utils import resize_to_size


def create_tillings(images, overlap_pct=CFG.tilling_overlap_pct):
    if len(images.shape) == 2:
        # print('ss')
        images = images.unsqueeze(0).repeat(CFG.in_chans, 1, 1)
    min_overlap = float(overlap_pct) * 0.01
    max_stride = CFG.image_size * (1.0 - min_overlap)

    height, width = images.shape[-2], images.shape[-1]
    num_patches = np.ceil(np.array([height, width]) / max_stride).astype(np.int32)

    starts = [
        np.int32(np.linspace(0, height - CFG.input_size, num_patches[0])),
        np.int32(np.linspace(0, width - CFG.input_size, num_patches[1])),
    ]
    stops = [starts[0] + CFG.input_size, starts[1] + CFG.input_size]

    indexs = []
    tills = []
    for y1, y2 in zip(starts[0], stops[0]):
        for x1, x2 in zip(starts[1], stops[1]):
            tills.append(images[..., y1:y2, x1:x2])
            indexs.append((y1, y2, x1, x2))

    return tills, indexs
