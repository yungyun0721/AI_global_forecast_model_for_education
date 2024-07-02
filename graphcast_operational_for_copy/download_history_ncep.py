import os
import pygrib as pg
import numpy as np
import xarray
import pandas as pd
import datetime
##https://data.rda.ucar.edu/ds084.1/2022/20220525/gfs.0p25.2022052500.f000.grib2

# time_list = ['2022052500','2022052512','2022052600','2022052612','2022052700','2022052712']
time_list = ['2023072400','2022052600']
#     # Load Grib2 file and save the Pangu initial field to specific path (output_data_path)

base_inputs_path = 'graphcast_weight/stats/' 
land_sea_mask = xarray.load_dataset(base_inputs_path+'graphcast_land_sea_mask.nc').compute()
geopotential_at_surface = xarray.load_dataset(base_inputs_path+'graphcast_geopotential_at_surface.nc').compute()

for initial_i in range(len(time_list)):
    timestr_front = (datetime.datetime.strptime(time_list[initial_i],"%Y%m%d%H")-datetime.timedelta(hours=6)).strftime("%Y%m%d%H")
    output_data_path1 = f'/bk2/yungyun/ncepgfs/2022/gfs.0p25.{timestr_front}.f000.grib2'
    output_data_path2 = f'/bk2/yungyun/ncepgfs/2022/gfs.0p25.{time_list[initial_i]}.f000.grib2'

    grib_file1 = pg.open(output_data_path1)
    grib_file2 = pg.open(output_data_path2)

    lat = np.arange(-90, 90.1, 0.25).astype(np.float32)
    lon = np.arange(0, 360, 0.25).astype(np.float32)

    height1 = np.full([13,721,1440],np.nan)
    specific_humidity1 = np.full([13,721,1440],np.nan)
    temperatrue1 = np.full([13,721,1440],np.nan)
    u1 = np.full([13,721,1440],np.nan)
    v1 = np.full([13,721,1440],np.nan)
    w1 = np.full([13,721,1440],np.nan)

    height2 = np.full([13,721,1440],np.nan)
    specific_humidity2 = np.full([13,721,1440],np.nan)
    temperatrue2 = np.full([13,721,1440],np.nan)
    u2 = np.full([13,721,1440],np.nan)
    v2 = np.full([13,721,1440],np.nan)
    w2 = np.full([13,721,1440],np.nan)
    
    target_lev = np.array([ 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    
    height_indices = [195, 221, 237, 253, 269, 285, 317, 349, 381, 413, 461, 493, 558]
    specific_humidity_indices = [199, 225, 241, 257, 273, 289, 321, 353, 385, 417, 465, 497, 545]
    temperatrue_indices = [196, 222, 238, 254, 270, 286, 318, 350, 382, 414, 462, 494, 542]
    w_indices = [201, 227, 243, 259, 275, 291, 323, 355, 387, 419, 467, 499, 547]
    u_indices = [202, 228, 244, 260, 276, 292, 324, 356, 388, 420, 468, 500, 548]
    v_indices = [203, 229, 245, 261, 277, 293, 325, 357, 389, 421, 469, 501, 549]

    for i in range(13):
        height1[i,:,:] = grib_file1[height_indices[i]].values * 9.80665
        specific_humidity1[i,:,:] = grib_file1[specific_humidity_indices[i]].values
        temperatrue1[i,:,:] = grib_file1[temperatrue_indices[i]].values
        u1[i,:,:] = grib_file1[u_indices[i]].values
        v1[i,:,:] = grib_file1[v_indices[i]].values
        w1[i,:,:] = grib_file1[w_indices[i]].values

    for i in range(13):
        height2[i,:,:] = grib_file2[height_indices[i]].values * 9.80665
        specific_humidity2[i,:,:] = grib_file2[specific_humidity_indices[i]].values
        temperatrue2[i,:,:] = grib_file2[temperatrue_indices[i]].values
        u2[i,:,:] = grib_file2[u_indices[i]].values
        v2[i,:,:] = grib_file2[v_indices[i]].values
        w2[i,:,:] = grib_file2[w_indices[i]].values

    # make xarray file 
    time_test = pd.array(['0 days 00:00:00', '0 days 06:00:00'], dtype='timedelta64[ns]')
    t2m = np.concatenate((grib_file1.select(name='2 metre temperature')[0].values.reshape(1, 1, 721, 1440),grib_file2.select(name='2 metre temperature')[0].values.reshape(1, 1, 721, 1440)),axis=1)
    mslp = np.concatenate((grib_file1.select(name='Pressure reduced to MSL')[0].values.reshape(1, 1, 721, 1440),grib_file2.select(name='Pressure reduced to MSL')[0].values.reshape(1, 1, 721, 1440)),axis=1)
    v10 = np.concatenate((grib_file1.select(name='10 metre U wind component')[0].values.reshape(1, 1, 721, 1440),grib_file2.select(name='10 metre U wind component')[0].values.reshape(1, 1, 721, 1440)),axis=1)
    u10 = np.concatenate((grib_file1.select(name='10 metre V wind component')[0].values.reshape(1, 1, 721, 1440),grib_file2.select(name='10 metre V wind component')[0].values.reshape(1, 1, 721, 1440)),axis=1)
    
    temperatrue = np.concatenate((temperatrue1.reshape(1, 1, -1, 721, 1440),temperatrue2.reshape(1, 1, -1, 721, 1440)),axis=1)
    height = np.concatenate((height1.reshape(1, 1, -1, 721, 1440),height2.reshape(1, 1, -1, 721, 1440)),axis=1)
    u = np.concatenate((u1.reshape(1, 1, -1, 721, 1440),u2.reshape(1, 1, -1, 721, 1440)),axis=1)
    v = np.concatenate((v1.reshape(1, 1, -1, 721, 1440),v2.reshape(1, 1, -1, 721, 1440)),axis=1)
    w = np.concatenate((w1.reshape(1, 1, -1, 721, 1440),w2.reshape(1, 1, -1, 721, 1440)),axis=1)
    specific_humidity = np.concatenate((specific_humidity1.reshape(1, 1, -1, 721, 1440),specific_humidity2.reshape(1, 1, -1, 721, 1440)),axis=1)
    
    # toa_solar made by other modular
    # toa_solar = np.concatenate((np.flip(ncep1.uswrftoa[1, :, :].values, axis=0).reshape(1, 1, 721, 1440),np.flip(ncep2.uswrftoa[1, :, :].values, axis=0).reshape(1, 1, 721, 1440)),axis=1)
    # tp = np.concatenate((np.flip(ncep1.apcpsfc[0, :, :].values, axis=0).reshape(1, 1, 721, 1440),np.flip(ncep2.apcpsfc[0, :, :].values, axis=0).reshape(1, 1, 721, 1440)), axis=1)

    tp_nan = u10.copy()*np.nan
    datetime_test =  pd.array([f'{timestr_front[:4]}-{timestr_front[4:6]}-{timestr_front[6:8]}T{timestr_front[8:]}:00:00.000000000', f'{time_list[initial_i][:4]}-{time_list[initial_i][4:6]}-{time_list[initial_i][6:8]}T{time_list[initial_i][8:]}:00:00.000000000'], dtype='datetime64').reshape([1,-1])

    inputs_ds = xarray.Dataset(
        data_vars={
            "geopotential_at_surface": (("lat", "lon"), np.array(geopotential_at_surface.variables['geopotential_at_surface'])),
            "land_sea_mask": (("lat", "lon"), np.array(land_sea_mask.variables['land_sea_mask'])),
            "2m_temperature": (("batch", "time","lat", "lon"), np.flip(t2m, axis=2)),
            "mean_sea_level_pressure": (("batch", "time","lat", "lon"), np.flip(mslp, axis=2)),
            "10m_v_component_of_wind": (("batch", "time","lat", "lon"), np.flip(v10, axis=2)),
            "10m_u_component_of_wind": (("batch", "time","lat", "lon"), np.flip(u10, axis=2)),
            "total_precipitation_6hr": (("batch", "time","lat", "lon"), np.flip(tp_nan, axis=2)),
            # "toa_incident_solar_radiation": (("batch", "time","lat", "lon"), toa_solar),
            "temperature": (("batch", "time", "level","lat", "lon"), np.flip(temperatrue, axis=3)),
            "geopotential": (("batch", "time", "level","lat", "lon"), np.flip(height, axis=3)),
            "u_component_of_wind": (("batch", "time", "level","lat", "lon"), np.flip(u, axis=3)),
            "v_component_of_wind": (("batch", "time", "level","lat", "lon"), np.flip(v, axis=3)),
            "vertical_velocity": (("batch", "time", "level","lat", "lon"), np.flip(w, axis=3)),
            "specific_humidity": (("batch", "time", "level","lat", "lon"), np.flip(specific_humidity, axis=3)),
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
        
    save_path = f'input_data/graphcast_input_{time_list[initial_i]}.nc'
    inputs_ds.to_netcdf(save_path)
    print('Done')
    

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--scheduled-time", required=True, help="format: 2023072000")
#     parser.add_argument("--scheduled-TC_ID", required=True, help="format: 202305W")
#     args = parser.parse_args()
#     main(args.scheduled_time, args.scheduled_TC_ID) 