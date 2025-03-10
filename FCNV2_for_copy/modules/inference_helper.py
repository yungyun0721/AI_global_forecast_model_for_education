import numpy as np
import torch, os
import fourcastnetv2 as nvs
from modules.afnonet import AFNONet, PrecipNet

def load_statistics(weight_path_global, channels=73):
    path = os.path.join(weight_path_global, "global_means.npy")
    global_means = np.load(path)
    # global_means = global_means[:, :channels, ...]
    global_means = global_means.astype(np.float32)

    path = os.path.join(weight_path_global, "global_stds.npy")
    global_stds = np.load(path)
    # global_stds = global_stds[:, :channels, ...]
    global_stds = global_stds.astype(np.float32)

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
    dims = data.shape[1]
    if reverse:
        new_data = data[:,:dims,...] * stds[:,:dims,...] + means[:,:dims,...]
    else:
        new_data = (data[:,:dims,...] - means[:,:dims,...]) / stds[:,:dims,...]
    return new_data

def nan_extend(data):
    return np.concatenate(
        (data, np.full_like(data[:, :, [-1], :], np.nan, dtype=data.dtype)), axis=2
    )
    

def load_precip_model(checkpoint_file, device="cpu"):
    out_channels = 1 
    in_channels = 20
    model = AFNONet(in_chans=in_channels, out_chans=out_channels)
    model = PrecipNet(backbone=model)

    model.zero_grad()
    # Load weights

    checkpoint = torch.load(checkpoint_file, map_location=device)

    asset_dim = checkpoint["model_state"][
        tuple(checkpoint["model_state"])[1]
    ].shape[1]
    model_dim = 20

    if asset_dim != model_dim:
        raise ValueError(
            f"Asset version ({asset_dim} variables) does not match model version"
            f"({model_dim} variables), please redownload correct weights."
        )

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

