class Configs:
    # configs for tilling dataset kidney 1
    path_img_kidney1 = "data\\train\\kidney_1_dense\\images"
    path_lb_kidney1 = "data\\train\\kidney_1_dense\\labels"
    path_df_kidney_1_til = "data\\kidney_1_tilling.csv"
    tile_size = (512, 512)
    overlap_pct = 10
    cache_dir = "data"

    # configs for tilling dataset kidney 1
    path_img_kidney3 = "data\\train\\kidney_3_sparse\\images"
    path_lb_kidney3 = "data\\train\\kidney_3_dense\\labels"
    path_df_kidney_3_til = "data\\kidney_3_tilling.csv"
    tile_size = (512, 512)
    overlap_pct = 10
    cache_dir = "data"

    # configs for transforms
    p_rot = 0.3
    p_aug = 0.3
    # cofigs for train / eval model
    random_seed = 42
