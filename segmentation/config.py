import torch


class CFG:
    # ============== dice metrics ============
    smooth = 1e-7
    # ============== pred target =============
    target_size = 1

    # ============== model CFG =============
    model_name = "Unet"
    backbone = "mobilenet_v2"

    in_chans = 5  # 65
    # ============== training CFG =============
    image_size = 512
    input_size = 512

    train_batch_size = 4
    n_accumulate = max(1, 16 // train_batch_size)
    valid_batch_size = train_batch_size

    epochs = 20
    lr = 6e-5
    chopping_percentile = 1e-3
    # ============== fold =============
    valid_id = 1

    # ============== augmentation =============
    p_augm = 0.5
    p_rot = 0.8
    # ============== settings ===================
    random_seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path_to_save_state_model = "weight"
