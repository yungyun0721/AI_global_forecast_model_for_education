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


    base_inputs_path = 'graphcast_weight/stats/' 

    lev = ncep1.lev
    target_lev = np.array([ 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    indices = np.where(np.isin(lev, target_lev))[0]
    lon = np.arange(0, 360, 0.25).astype(np.float32)
    lat = np.arange(-90, 90.1, 0.25).astype(np.float32)
    time_test = np.array([  0,  21600000000000], dtype='timedelta64')
    time_test = pd.array(['0 days 00:00:00', '0 days 06:00:00'], dtype='timedelta64[ns]')



    land_sea_mask = xarray.load_dataset(base_inputs_path+'graphcast_land_sea_mask.nc').compute()
    geopotential_at_surface = xarray.load_dataset(base_inputs_path+'graphcast_geopotential_at_surface.nc').compute()

    t2m = np.concatenate((ncep1.tmp2m[0, :, :].values.reshape(1, 1, 721, 1440),ncep2.tmp2m[0, :, :].values.reshape(1, 1, 721, 1440)),axis=1)
    mslp = np.concatenate((ncep1.msletmsl[0, :, :].values.reshape(1, 1, 721, 1440),ncep2.msletmsl[0, :, :].values.reshape(1, 1, 721, 1440)),axis=1)
    v10 = np.concatenate((ncep1.vgrd10m[0, :, :].values.reshape(1, 1, 721, 1440),ncep2.vgrd10m[0, :, :].values.reshape(1, 1, 721, 1440)),axis=1)
    u10 = np.concatenate((ncep1.ugrd10m[0, :, :].values.reshape(1, 1, 721, 1440),ncep2.ugrd10m[0, :, :].values.reshape(1, 1, 721, 1440)),axis=1)
    
    temperatrue = np.concatenate((ncep1.tmpprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440),ncep2.tmpprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440)),axis=1)
    height = np.concatenate(((ncep1.hgtprs[0, indices, :, :].values * 9.80665).reshape(1, 1, -1, 721, 1440),(ncep2.hgtprs[0, indices, :, :].values* 9.80665).reshape(1, 1, -1, 721, 1440)),axis=1)
    u = np.concatenate((ncep1.ugrdprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440),ncep2.ugrdprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440)),axis=1)
    v = np.concatenate((ncep1.vgrdprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440),ncep2.vgrdprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440)),axis=1)
    w = np.concatenate((ncep1.dzdtprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440),ncep2.dzdtprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440)),axis=1)
    specific_humidity = np.concatenate((ncep1.spfhprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440),ncep2.spfhprs[0, indices, :, :].values.reshape(1, 1, -1, 721, 1440)),axis=1)
    
    # toa_solar made by other modular
    # toa_solar = np.concatenate((np.flip(ncep1.uswrftoa[1, :, :].values, axis=0).reshape(1, 1, 721, 1440),np.flip(ncep2.uswrftoa[1, :, :].values, axis=0).reshape(1, 1, 721, 1440)),axis=1)
    # tp = np.concatenate((np.flip(ncep1.apcpsfc[0, :, :].values, axis=0).reshape(1, 1, 721, 1440),np.flip(ncep2.apcpsfc[0, :, :].values, axis=0).reshape(1, 1, 721, 1440)), axis=1)

    tp_nan = u10.copy()*np.nan
    datetime_test =  pd.array([f'{timestr_front[:4]}-{timestr_front[4:6]}-{timestr_front[6:8]}T{timestr_front[8:]}:00:00.000000000', f'{timestr[:4]}-{timestr[4:6]}-{timestr[6:8]}T{timestr[8:]}:00:00.000000000'], dtype='datetime64').reshape([1,-1])
    
    inputs_ds = xarray.Dataset(
        data_vars={
            "geopotential_at_surface": (("lat", "lon"), np.array(geopotential_at_surface.variables['geopotential_at_surface'])),
            "land_sea_mask": (("lat", "lon"), np.array(land_sea_mask.variables['land_sea_mask'])),
            "2m_temperature": (("batch", "time","lat", "lon"), t2m),
            "mean_sea_level_pressure": (("batch", "time","lat", "lon"), mslp),
            "10m_v_component_of_wind": (("batch", "time","lat", "lon"), v10),
            "10m_u_component_of_wind": (("batch", "time","lat", "lon"), u10),
            "total_precipitation_6hr": (("batch", "time","lat", "lon"), tp_nan),
            # "toa_incident_solar_radiation": (("batch", "time","lat", "lon"), toa_solar),
            "temperature": (("batch", "time", "level","lat", "lon"), temperatrue),
            "geopotential": (("batch", "time", "level","lat", "lon"), height),
            "u_component_of_wind": (("batch", "time", "level","lat", "lon"), u),
            "v_component_of_wind": (("batch", "time", "level","lat", "lon"), v),
            "vertical_velocity": (("batch", "time", "level","lat", "lon"), w),
            "specific_humidity": (("batch", "time", "level","lat", "lon"), specific_humidity),
                },         
        coords={
            "lon":("lon",lon),
            "lat":("lat",lat),
            "level":("level",target_lev.astype(np.int32)),
            "time":("time",time_test),
            "datetime":(("batch","time"),datetime_test),
                },
        # attrs={"attr": "example"}
    )


    if not os.path.isdir('input_data'):
        os.mkdir('input_data')
        
    save_path = 'input_data/graphcast_input.nc'
    inputs_ds.to_netcdf(save_path)
    print('Done')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduled-time", required=True, help="format: 2024011500")
    args = parser.parse_args()
    main(args.scheduled_time)    
