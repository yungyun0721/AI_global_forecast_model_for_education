import datetime
import numpy as np
import xarray,os
import dataclasses
import torch

from aurora import Aurora, AuroraSmall, Batch, Metadata

cpu_num = 6
def aurora_model(input_dir, output_folder, fore_hour=72):

    # model = AuroraSmall()
    model = Aurora(use_lora=False)
    # model.load_checkpoint("weight/aurora-0.25-pretrained.ckpt")
    model.load_checkpoint("weight/aurora-0.25-fintuned.ckpt", strict=False)

    input_iter = xarray.load_dataset(input_dir)
    # need to 2 time (docs/reference to example_era5.ipynb and aroura/rollout.py)
    batch = Batch(
        surf_vars={np.array(input_iter.coords['surf_vars'])[k]: torch.from_numpy(np.array(input_iter.variables['surface_variables'][k,...])) for k in range(4)},
        static_vars={np.array(input_iter.coords['static_vars'])[k]: torch.from_numpy(np.array(input_iter.variables['static_variables'][k,...])) for k in range(3)},
        atmos_vars={np.array(input_iter.coords['atmos_vars'])[k]: torch.from_numpy(np.array(input_iter.variables['upper_variables'][k,...])) for k in range(5)},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 720 + 1)[:-1],
            lon=torch.linspace(0, 360, 1440 + 1)[:-1],
            time=(datetime.datetime.strptime(np.array(input_iter.coords['datetime'])[0], "%Y%m%d%H"),),
            atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        ),
    )
    print('predicting....')
    model.eval()
    torch.set_num_threads(cpu_num)
    for time_index in range(np.int_(fore_hour/6)):

        with torch.inference_mode():
            prediction = model.forward(batch)
        # prediction = model.forward(batch)
        
        print('saving....'+str(time_index*6+6)+'hr')
        upper = np.stack([prediction.atmos_vars["z"], prediction.atmos_vars["u"], prediction.atmos_vars["v"], prediction.atmos_vars["t"], prediction.atmos_vars["q"]],axis=0)
        sfc = np.stack([prediction.surf_vars["2t"], prediction.surf_vars["10u"], prediction.surf_vars["10v"], prediction.surf_vars["msl"]],axis=0)
        static = np.stack([prediction.static_vars["lsm"],prediction.static_vars["z"], prediction.static_vars["slt"]],axis=0)
        
        output_ds = xarray.Dataset(
            data_vars={
                "upper_variables": (("atmos_vars","level","lat", "lon"), upper[:,0,0,:,:,:]),
                "surface_variables": (("surf_vars","lat", "lon"), sfc[:,0,0,:,:]),
                "static_variables": (("static_vars","lat", "lon"), static),
                    },         
            coords={
                "lon":("lon",prediction.metadata.lon),
                "lat":("lat",prediction.metadata.lat),
                "level":("level",np.array(prediction.metadata.atmos_levels).astype(np.int32)),
                "datetime":("datetime",[prediction.metadata.time[0]]),
                "atmos_vars":("atmos_vars",["z", "u", "v", "t", "q"]),
                "surf_vars":("surf_vars",["2t", "10u", "10v", "msl"]),
                "static_vars":("static_vars",["lsm", "z", "slt"]),
                    },
            # attrs={"attr": "example"}
        )
        
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
            
        save_path = f'{output_folder}/aurora_output_{time_index*6+6}hr.nc'
        output_ds.to_netcdf(save_path)
        

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            prediction,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in prediction.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in prediction.atmos_vars.items()
            },
        )
        if np.mod(time_index,4)==3:
            print(f'finish {int(time_index/4)+1} days')


    print("Done")