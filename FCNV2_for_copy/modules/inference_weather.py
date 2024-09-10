# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#%%

import os
import numpy as np
import torch
# from ai_models.model import Model

from modules.inference_helper import nan_extend, normalise, load_model, load_statistics

# setting
weight_path_global = './weight'

# load_weight
# Input
area = [90, 0, -90, 360 - 0.25]
grid = [0.25, 0.25]

# setting
n_lat = 720
n_lon = 1440
device = "cpu"
cpu_num = 10

ordering = [ "10u",   "10v", "100u", "100v",   "2t",   "sp",  "msl", "tcwv",\
             "u50",  "u100", "u150", "u200", "u250", "u300", "u400", "u500",\
             "u600", "u700", "u850", "u925","u1000",  "v50", "v100", "v150",\
             "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850",\
             "v925","v1000",  "z50", "z100", "z150", "z200", "z250", "z300",\
             "z400", "z500", "z600", "z700", "z850", "z925","z1000",  "t50",\
             "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600",\
             "t700", "t850", "t925","t1000",  "r50", "r100", "r150", "r200",\
             "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000"]


# load model and parameters
global_means, global_stds = load_statistics(weight_path_global, channels = 73)
backbone_ckpt = os.path.join(weight_path_global, "weights.tar")
backbone_model = load_model(backbone_ckpt)

def FCN2_weather(input_file, output_dir, fore_hour=240):
    
    torch.set_num_threads(cpu_num)
    fore_hour = np.int_(fore_hour)
    # save output FCN_weather
    if not os.path.isdir(f'{output_dir}'):
        os.mkdir(f'{output_dir}')
    
    print(f'start predict')
    # load data 
    initial_data = np.load(f'{input_file}').astype(np.float32)
    np.save(os.path.join(output_dir, f'output_weather_0h'), initial_data.squeeze())
    # data preprocessing
    all_fields_numpy = initial_data[np.newaxis, :, :, :]
    all_fields_numpy = normalise(all_fields_numpy,global_means,global_stds)
    input_iter = torch.from_numpy(all_fields_numpy).to(device)
    torch.set_grad_enabled(False)
    


    for time_index in range(np.int_(fore_hour/6)):
        output = backbone_model(input_iter)
        input_iter = output
        # reverse normalise
        # output = nan_extend(normalise(output.cpu().numpy(),global_means, global_stds, reverse=True))
        output = normalise(output.cpu().numpy(),global_means, global_stds, reverse=True)
        
        np.save(os.path.join(output_dir, f'output_weather_{(time_index+1)*6}h'), output.squeeze())
        if np.mod(time_index,4)==3:
            print(f'finish {int(time_index/4)+1} days')
    print(f'Done')


   

