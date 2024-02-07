import torch


class Configs:

    # configs for tilling dataset kidney 1
    path_img_kidney1 = "data\\train\\kidney_1_dense\\images"
    path_lb_kidney1 = "data\\train\\kidney_1_dense\\labels"
    path_df_kidney_1_till = "data\\kidney_1_tilling.csv"
    tile_size = (256, 256)
    overlap_pct = 0
    cache_dir = "data"

    # configs for tilling dataset kidney 1
    path_img_kidney3 = "data\\train\\kidney_3_sparse\\images"
    path_lb_kidney3 = "data\\train\\kidney_3_dense\\labels"
    path_df_kidney_3_till = "data\\kidney_3_tilling.csv"
    tile_size = (256, 256)
    overlap_pct = 0
    cache_dir = "data"

    # configs for transforms
    p_rot = 0.3
    p_aug = 0.3
    # cofigs for train / eval model
    random_seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_batch_size = 8
    n_accumulate = max(1, 8 // train_batch_size)
    valid_batch_size = train_batch_size * 2

    epochs = 28
    lr = 3e-4
