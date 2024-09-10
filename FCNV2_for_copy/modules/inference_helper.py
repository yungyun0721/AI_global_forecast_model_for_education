import numpy as np
import torch, os
import fourcastnetv2 as nvs

def load_statistics(weight_path_global, channels=73):
    path = os.path.join(weight_path_global, "global_means.npy")
    global_means = np.load(path)
    # global_means = global_means[:, :channels, ...]
    global_means = global_means.astype(np.float32)

    path = os.path.join(weight_path_global, "global_stds.npy")
    global_stds = np.load(path)
    # global_stds = global_stds[:, :channels, ...]
    global_stds = global_stds.astype(np.float32)
    
    # # change the order
    # modify_global_stds = np.full([1,73,1,1],0.0)
    # modify_global_means = np.full([1,73,1,1],0.0)
    # # modify surface
    # modify_global_stds[0,  :2, 0, 0] = global_stds[0,  :2, 0, 0]
    # modify_global_stds[0, 2:6, 0, 0] = global_stds[0, 4:8, 0, 0]
    # modify_global_stds[0, 6:8, 0, 0] = global_stds[0, 2:4, 0, 0]
    # modify_global_means[0,  :2, 0, 0] = global_means[0,  :2, 0, 0]
    # modify_global_means[0, 2:6, 0, 0] = global_means[0, 4:8, 0, 0]
    # modify_global_means[0, 6:8, 0, 0] = global_means[0, 2:4, 0, 0]
    # # modify upper
    # modify_global_stds[0, 8:21, 0, 0] = global_stds[0, 34:47, 0, 0]
    # modify_global_stds[0,21:34, 0, 0] = global_stds[0, 47:60, 0, 0]
    # modify_global_stds[0,34:47, 0, 0] = global_stds[0,  8:21, 0, 0]
    # modify_global_stds[0,47:60, 0, 0] = global_stds[0, 21:34, 0, 0]
    # modify_global_stds[0,60:  , 0, 0] = global_stds[0, 60:  , 0, 0]
    # modify_global_means[0, 8:21, 0, 0] = global_means[0, 34:47, 0, 0]
    # modify_global_means[0,21:34, 0, 0] = global_means[0, 47:60, 0, 0]
    # modify_global_means[0,34:47, 0, 0] = global_means[0,  8:21, 0, 0]
    # modify_global_means[0,47:60, 0, 0] = global_means[0, 21:34, 0, 0]
    # modify_global_means[0,60:  , 0, 0] = global_means[0, 60:  , 0, 0]
    # modify_global_means = modify_global_means.astype(np.float32)
    # modify_global_stds = modify_global_stds.astype(np.float32)
    
    
    # return modify_global_means, modify_global_stds
    return global_means, global_stds


def load_model(checkpoint_file, device="cpu"):
    model = nvs.FourierNeuralOperatorNet()

    model.zero_grad()
    # Load weights

    checkpoint = torch.load(checkpoint_file, map_location=device)

    weights = checkpoint["model_state"]
    drop_vars = ["module.norm.weight", "module.norm.bias"]
    weights = {k: v for k, v in weights.items() if k not in drop_vars}

    # Make sure the parameter names are the same as the checkpoint
    # need to use strict = False to avoid this error message when
    # using sfno_76ch::
    # RuntimeError: Error(s) in loading state_dict for Wrapper:
    # Missing key(s) in state_dict: "module.trans_down.weights",
    # "module.itrans_up.pct",
    try:
        # Try adding model weights as dictionary
        new_state_dict = dict()
        for k, v in checkpoint["model_state"].items():
            name = k[7:]
            if name != "ged":
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except Exception:
        model.load_state_dict(checkpoint["model_state"])

    # Set model to eval mode and return
    model.eval()
    model.to(device)

    return model

def normalise(data, means, stds, reverse=False):
    """Normalise data using pre-saved global statistics"""
    if reverse:
        new_data = data * stds + means
    else:
        new_data = (data - means) / stds
    return new_data

def nan_extend(data):
    return np.concatenate(
        (data, np.full_like(data[:, :, [-1], :], np.nan, dtype=data.dtype)), axis=2
    )