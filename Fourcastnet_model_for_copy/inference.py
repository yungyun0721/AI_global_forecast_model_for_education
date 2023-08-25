# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#%%


#%%

import logging
import os

import numpy as np
import torch
# from ai_models.model import Model

from afnonet import  unlog_tp_torch  # noqa
from inference_helper import nan_extend, normalise, load_model

LOG = logging.getLogger(__name__)

# weight_path = '/wk171/yungyun/FCN_test_from_ECMWF/ai-models/'
weight_path = 'model_weight/'
input_data_dir = 'input_data'# add inital time
output_data_dir = 'output_data'

# Input
area = [90, 0, -90, 360 - 0.25]
grid = [0.25, 0.25]

# setting
precip_flag = True
n_lat = 720
n_lon = 1440
precip_channels = 20
backbone_channels = 26
device = "cpu"

# load global_means and global_stds to do data preprocessing
def load_statistics(backbone_channels=26):
    path = os.path.join(weight_path, "global_means.npy")
    LOG.info("Loading %s", path)
    global_means = np.load(path)
    global_means = global_means[:, :backbone_channels, ...]
    global_means = global_means.astype(np.float32)

    path = os.path.join(weight_path, "global_stds.npy")
    LOG.info("Loading %s", path)
    global_stds = np.load(path)
    global_stds = global_stds[:, :backbone_channels, ...]
    global_stds = global_stds.astype(np.float32)
    return global_means, global_stds

# load initial data and do data preprocessing
all_fields = np.load(os.path.join(input_data_dir, 'inital_condition.npy')).astype(np.float32)
global_means, global_stds = load_statistics()
# all_fields = np.float32(all_fields)

# all_fields_numpy = all_fields.to_numpy(dtype=np.float32)[np.newaxis, :, :-1, :]
all_fields_numpy = all_fields[np.newaxis, :, :-1, :]

all_fields_numpy = normalise(all_fields_numpy,global_means,global_stds)

# load model wight (weather and precipitation)
backbone_ckpt = os.path.join(weight_path, "backbone.ckpt")
backbone_model = load_model(backbone_ckpt, precip=False, backbone_channels=backbone_channels)

if precip_flag:
    precip_ckpt = os.path.join(weight_path, "precip.ckpt")
    precip_model = load_model(precip_ckpt, precip=True)

# Run the inference session
input_iter = torch.from_numpy(all_fields_numpy).to(device)
torch.set_grad_enabled(False)

# run model and save output
# 40*6h =240h
for i in range(40):
    
    output = backbone_model(input_iter)
    if precip_flag:
        precip_output = precip_model(output[:, : precip_channels, ...])
    print('finish '+str(i*6)+'h')
    input_iter = output

    output = nan_extend(normalise(output.cpu().numpy(),global_means, global_stds, reverse=True))
    # output = nan_extend(output.cpu().numpy())
    precip_output = nan_extend(unlog_tp_torch(precip_output.cpu()).numpy())
    
    # Save the results
    np.save(os.path.join(output_data_dir, f'output_weather_{(i+1)*6}h'), output.squeeze())
    np.save(os.path.join(output_data_dir, f'output_precipitation_{(i+1)*6}h'), precip_output.squeeze())
    


# which machine to run CPU or GPU
# def device():
#     device = "cpu"

#     if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#         device = "mps"

#     if torch.cuda.is_available() and torch.backends.cuda.is_built():
#         device = "cuda"

#     LOG.info(
#         "Using device '%s'. The speed of inference depends greatly on the device.",
#         device.upper(),
#     )

#     return device

# class FourCastNet0(FourCastNet):
#     download_url = (
#         "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnet/0.0/{file}"
#     )

#     assets_extra_dir = "0.0"

#     param_sfc = ["10u", "10v", "2t", "sp", "msl", "tcwv"]

#     param_level_pl = (["t", "u", "v", "z", "r"], [1000, 850, 500, 50])

#     ordering = [
#         "10u",
#         "10v",
#         "2t",
#         "sp",
#         "msl",
#         "t850",
#         "u1000",
#         "v1000",
#         "z1000",
#         "u850",
#         "v850",
#         "z850",
#         "u500",
#         "v500",
#         "z500",
#         "t500",
#         "z50",
#         "r500",
#         "r850",
#         "tcwv",
#     ]


# class FourCastNet1(FourCastNet):
#     download_url = (
#         "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnet/0.1/{file}"
#     )

#     param_sfc = ["10u", "10v", "2t", "sp", "msl", "tcwv", "100u", "100v"]

#     param_level_pl = (["t", "u", "v", "z", "r"], [1000, 850, 500, 250, 50])

#     assets_extra_dir = "0.1"

#     ordering = [
#         "10u",
#         "10v",
#         "2t",
#         "sp",
#         "msl",
#         "t850",
#         "u1000",
#         "v1000",
#         "z1000",
#         "u850",
#         "v850",
#         "z850",
#         "u500",
#         "v500",
#         "z500",
#         "t500",
#         "z50",
#         "r500",
#         "r850",
#         "tcwv",
#         "100u",
#         "100v",
#         "u250",
#         "v250",
#         "z250",
#         "t250",
#     ]


# def model(model_version, **kwargs):
#     models = {
#         "0": FourCastNet0,
#         "1": FourCastNet1,
#         "release": FourCastNet0,
#         "latest": FourCastNet1,
#     }
#     return models[model_version](**kwargs)
