import argparse
import numpy as np
import xarray, os
import datetime
import pandas as pd
#%%
def main(timestr: str) -> None:
    
    timestr_front = (datetime.datetime.strptime(timestr,"%Y%m%d%H")-datetime.timedelta(hours=6)).strftime("%Y%m%d%H")
    path1 = (
        f'http://nomads.ncep.noaa.gov:80/dods/gfs_0p25/gfs{timestr_front[:-2]}/gfs_0p25_{timestr_front[-2:]}z'
    )
    print(path1)
    ncep1 = xarray.open_dataset(path1)

    path2 = (
        f'http://nomads.ncep.noaa.gov:80/dods/gfs_0p25/gfs{timestr[:-2]}/gfs_0p25_{timestr[-2:]}z'
    )
    print(path2)
    ncep2 = xarray.open_dataset(path2)


    base_inputs_path = 'weight/' 

    lev = ncep1.lev
    target_lev = np.array([ 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    indices = np.flip(np.where(np.isin(lev, target_lev))[0])
    lon = np.arange(0, 360, 0.25).astype(np.float32)
    lat = np.arange(-90, 90, 0.25).astype(np.float32)
    time_test = np.array([  0,  21600000000000], dtype='timedelta64')
    time_test = pd.array(['0 days 00:00:00', '0 days 06:00:00'], dtype='timedelta64[ns]')
    


    static_file = xarray.load_dataset(base_inputs_path+'static.nc').compute()
    z_static = np.array(static_file.variables['z'])
    lsm      = np.array(static_file.variables['lsm'])
    slt      = np.array(static_file.variables['slt'])
    static = np.concatenate([z_static, lsm, slt], axis=0)
    static = np.flip(static,axis=1)

    t2m = np.concatenate((ncep1.tmp2m[0, :, :].values.reshape(1, 1, 721, 1440),ncep2.tmp2m[0, :, :].values.reshape(1, 1, 721, 1440)),axis=1)
    u10 = np.concatenate((ncep1.ugrd10m[0, :, :].values.reshape(1, 1, 721, 1440),ncep2.ugrd10m[0, :, :].values.reshape(1, 1, 721, 1440)),axis=1)
    v10 = np.concatenate((ncep1.vgrd10m[0, :, :].values.reshape(1, 1, 721, 1440),ncep2.vgrd10m[0, :, :].values.reshape(1, 1, 721, 1440)),axis=1)
    mslp = np.concatenate((ncep1.msletmsl[0, :, :].values.reshape(1, 1, 721, 1440),ncep2.msletmsl[0, :, :].values.reshape(1, 1, 721, 1440)),axis=1)
    
    height = np.concatenate(((ncep1.hgtprs[0, indices, :, :].values * 9.80665).reshape(1, 1, -1, 721, 1440),(ncep2.hgtprs[0, indices, :, :].values* 9.80665).reshape(1, 1, -1, 721, 1440)),axis=1)
    u = np.concatenate((ncep1.ugrdprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440),ncep2.ugrdprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440)),axis=1)
    v = np.concatenate((ncep1.vgrdprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440),ncep2.vgrdprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440)),axis=1)
    temperatrue = np.concatenate((ncep1.tmpprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440),ncep2.tmpprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440)),axis=1)
    specific_humidity = np.concatenate((ncep1.spfhprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440),ncep2.spfhprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440)),axis=1)
    
    upper = np.stack([height, u, v, temperatrue, specific_humidity], axis=0)
    sfc = np.stack([t2m, u10, v10, mslp], axis=0)
    upper = np.flip(upper,axis=4)
    sfc = np.flip(sfc,axis=3)
    
    inputs_ds = xarray.Dataset(
        data_vars={
            "upper_variables": (("atmos_vars","batch","time","level","lat", "lon"), upper[:,:,:,:,:-1,:]),
            "surface_variables": (("surf_vars","batch","time","lat", "lon"), sfc[:,:,:,:-1,:]),
            "static_variables": (("static_vars","lat", "lon"), static[:,:-1,:]),
                },         
        coords={
            "lon":("lon",lon),
            "lat":("lat",np.flip(lat)),
            "level":("level",target_lev.astype(np.int32)),
            "time":("time",time_test),
            "datetime":("batch",[timestr]),
            "atmos_vars":("atmos_vars",["z", "u", "v", "t", "q"]),
            "surf_vars":("surf_vars",["2t", "10u", "10v", "msl"]),
            "static_vars":("static_vars",["lsm", "z", "slt"]),
                },
        # attrs={"attr": "example"}
    )


    if not os.path.isdir('input_data'):
        os.mkdir('input_data')
        
    save_path = 'input_data/aurora_input.nc'
    inputs_ds.to_netcdf(save_path)
    print('Done')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduled-time", required=True, help="format: 2024011500")
    args = parser.parse_args()
    main(args.scheduled_time)    
