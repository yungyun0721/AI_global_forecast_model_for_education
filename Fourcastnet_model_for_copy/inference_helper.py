import numpy as np
import torch
from afnonet import AFNONet, PrecipNet



def load_model(checkpoint_file, device="cpu", precip=False, backbone_channels=20, precip_channels = 20):
    out_channels = 1 if precip else backbone_channels
    in_channels = 20 if precip else backbone_channels

    model = AFNONet(in_chans=in_channels, out_chans=out_channels)

    if precip:
        model = PrecipNet(backbone=model)

    model.zero_grad()
    # Load weights

    checkpoint = torch.load(checkpoint_file, map_location=device)

    asset_dim = checkpoint["model_state"][
        tuple(checkpoint["model_state"])[1]
    ].shape[1]
    model_dim = precip_channels if precip else backbone_channels

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