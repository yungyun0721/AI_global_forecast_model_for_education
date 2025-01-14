import argparse

import requests
import os
import pygrib as pg
import numpy as np
##https://data.rda.ucar.edu/d084001/2022/20220525/gfs.0p25.2022052500.f000.grib2

time_list = ['2020051100']
# Set the path for saving NCEP GFS file(s)
ncep_path = f'input_data/'
if not os.path.isdir(ncep_path):
    os.mkdir(ncep_path)
files = [f'{time_list[i][0:4]}/{time_list[i][0:8]}/gfs.0p25.{time_list[i]}.f000.grib2' for i in range(len(time_list))]
print(files)
#%%
# download the data file(s)
for file in files:
    idx = file.rfind("/")
    if (idx > 0):
        ofile = ncep_path + file[idx+1:]
    else:
        ofile = ncep_path + file
    print(ofile)

    response = requests.get("https://data.rda.ucar.edu/d084001/" + file)
    with open(ofile, "wb") as f:
        f.write(response.content)
        f.close()

    grib_file = pg.open(ofile)

    target_lev = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    wind_level_index =   [9,11,12,13,14,15,17,19,21,23,26,28,31]
    temperature_index = [11,13,14,15,16,17,19,21,23,25,28,30,33]
    RH_index =[8,10,11,12,13,14,16,18,20,22,25,27,30]
    
    # height_indices = [195,221,237,253,269,285,317,349,381,413,461,493,558]
    # relative_humidity_indices = [197,223,239,255,271,287,319,351,383,415,463,495,543]
    # temperatrue_indices = [196,222,238,254,270,286,318,350,382,414,462,494,542]
    # u_indices = [202,228,244,260,276,292,324,356,388,420,468,500,548]
    # v_indices = [203,229,245,261,277,293,325,357,389,421,469,501,549]
    
    u10 = grib_file.select(name='10 metre U wind component')[0].values.reshape(1,721,1440)
    v10 = grib_file.select(name='10 metre V wind component')[0].values.reshape(1,721,1440)
    t2m = grib_file.select(name='2 metre temperature')[0].values.reshape(1,721,1440)
    sp = grib_file.select(name='Surface pressure')[0].values.reshape(1,721,1440)
    mslp = grib_file.select(name='Pressure reduced to MSL')[0].values.reshape(1,721,1440)
    tcwv = grib_file.select(name='Precipitable water')[0].values.reshape(1,721,1440)
    wind_100u = grib_file.select(name='100 metre U wind component')[0].values.reshape(1,721,1440)
    wind_100v = grib_file.select(name='100 metre V wind component')[0].values.reshape(1,721,1440)
    
    z_upper = np.empty((13,721,1440))
    r_upper = np.empty((13,721,1440))
    t_upper = np.empty((13,721,1440))
    u_upper = np.empty((13,721,1440))
    v_upper = np.empty((13,721,1440))
    for i in range(13):
        z_upper[i,:,:] = grib_file.select(name='Geopotential Height')[temperature_index[i]].values * 9.80665
        r_upper[i,:,:] = grib_file.select(name='Relative humidity')[RH_index[i]].values
        t_upper[i,:,:] = grib_file.select(name='Temperature')[temperature_index[i]].values
        u_upper[i,:,:] = grib_file.select(name='U component of wind')[wind_level_index[i]].values
        v_upper[i,:,:] = grib_file.select(name='V component of wind')[wind_level_index[i]].values
        
    IC = np.concatenate([u10, v10, wind_100u, wind_100v, t2m, sp, mslp, tcwv,  \
                        u_upper, v_upper, z_upper, t_upper,  r_upper], axis=0)
    
    if not os.path.isdir('input_data'):
        os.mkdir('input_data')
    np.save(f'./input_data/inital_condition_{file[23:33]}.npy', IC.astype(np.float32))
    print('Done')
    