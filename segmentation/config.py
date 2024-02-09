import torch


class CFG:
    # metrics
    smooth = 1e-7

    # configs for tilling dataset kidney 1
    path_img_kidney1 = "data\\train\\kidney_1_dense\\images"
    path_lb_kidney1 = "data\\train\\kidney_1_dense\\labels"
    path_df_kidney_1_till = "data\\kidney_1_tilling.csv"
    tile_size = (256, 256)
    overlap_pct = 0
    cache_dir = "data"
    mean_till1 = (0.2425, 0.2425, 0.2425)
    std_till1 = (0.1766, 0.1766, 0.1766)

    # configs for tilling dataset kidney 3
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
    random_seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_batch_size = 8
    n_accumulate = max(1, 32 // train_batch_size)
    valid_batch_size = train_batch_size * 2
    clip_norm = 5
    epochs = 28
    lr = 3e-4
    dice_th = 0.5

    path_to_save_state_model = "weight"
    path_weight_model = "weight"
