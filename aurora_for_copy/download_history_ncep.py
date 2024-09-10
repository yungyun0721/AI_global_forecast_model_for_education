import os
import requests
import pygrib as pg
import netCDF4 as nc
import numpy as np
import xarray
import pandas as pd
import datetime
##https://data.rda.ucar.edu/ds084.1/2022/20220525/gfs.0p25.2022052500.f000.grib2

timestr = '2023072600'

# download data
timestr_front = (datetime.datetime.strptime(timestr,"%Y%m%d%H")-datetime.timedelta(hours=6)).strftime("%Y%m%d%H")
time_list = [timestr_front] + [timestr]
# Set the path for saving NCEP GFS file(s)
ncep_path = f'input_data/'
if not os.path.isdir(ncep_path):
    os.mkdir(ncep_path)
files = [f'{time_list[i][0:4]}/{time_list[i][0:8]}/gfs.0p25.{time_list[i]}.f000.grib2' for i in range(len(time_list))]
print(files)

for file in files:
    idx = file.rfind("/")
    if (idx > 0):
        ofile = ncep_path + file[idx+1:]
    else:
        ofile = ncep_path + file
    print(ofile)

    response = requests.get("https://data.rda.ucar.edu/ds084.1/" + file)
    with open(ofile, "wb") as f:
        f.write(response.content)
        f.close()


# static load
base_inputs_path = 'weight/' 
target_lev = np.array([ 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
lon = np.arange(0, 360, 0.25).astype(np.float32)
lat = np.arange(-90, 90, 0.25).astype(np.float32)
time_test = pd.array(['0 days 00:00:00', '0 days 06:00:00'], dtype='timedelta64[ns]')

static_file = xarray.load_dataset(base_inputs_path+'static.nc').compute()
z_static = np.array(static_file.variables['z'])
lsm      = np.array(static_file.variables['lsm'])
slt      = np.array(static_file.variables['slt'])
static = np.concatenate([z_static, lsm, slt], axis=0)
static = np.flip(static,axis=1)

# perpare input data
idx = file.rfind("/")
output_data_path1 = ncep_path + files[0][idx+1:]
output_data_path2 = ncep_path + files[1][idx+1:]

grib_file1 = pg.open(output_data_path1)
grib_file2 = pg.open(output_data_path2)

lat = np.arange(-90, 90.1, 0.25).astype(np.float32)
lon = np.arange(0, 360, 0.25).astype(np.float32)

height1 = np.full([13,721,1440],np.nan)
specific_humidity1 = np.full([13,721,1440],np.nan)
temperatrue1 = np.full([13,721,1440],np.nan)
u1 = np.full([13,721,1440],np.nan)
v1 = np.full([13,721,1440],np.nan)

height2 = np.full([13,721,1440],np.nan)
specific_humidity2 = np.full([13,721,1440],np.nan)
temperatrue2 = np.full([13,721,1440],np.nan)
u2 = np.full([13,721,1440],np.nan)
v2 = np.full([13,721,1440],np.nan)

target_lev = np.array([ 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])

height_indices = [195, 221, 237, 253, 269, 285, 317, 349, 381, 413, 461, 493, 558]
specific_humidity_indices = [199, 225, 241, 257, 273, 289, 321, 353, 385, 417, 465, 497, 545]
temperatrue_indices = [196, 222, 238, 254, 270, 286, 318, 350, 382, 414, 462, 494, 542]
u_indices = [202, 228, 244, 260, 276, 292, 324, 356, 388, 420, 468, 500, 548]
v_indices = [203, 229, 245, 261, 277, 293, 325, 357, 389, 421, 469, 501, 549]

for i in range(13):
    height1[i,:,:] = grib_file1[height_indices[i]].values * 9.80665
    specific_humidity1[i,:,:] = grib_file1[specific_humidity_indices[i]].values
    temperatrue1[i,:,:] = grib_file1[temperatrue_indices[i]].values
    u1[i,:,:] = grib_file1[u_indices[i]].values
    v1[i,:,:] = grib_file1[v_indices[i]].values

for i in range(13):
    height2[i,:,:] = grib_file2[height_indices[i]].values * 9.80665
    specific_humidity2[i,:,:] = grib_file2[specific_humidity_indices[i]].values
    temperatrue2[i,:,:] = grib_file2[temperatrue_indices[i]].values
    u2[i,:,:] = grib_file2[u_indices[i]].values
    v2[i,:,:] = grib_file2[v_indices[i]].values

# make xarray file 
t2m = np.concatenate((grib_file1.select(name='2 metre temperature')[0].values.reshape(1, 1, 721, 1440),grib_file2.select(name='2 metre temperature')[0].values.reshape(1, 1, 721, 1440)),axis=1)
mslp = np.concatenate((grib_file1.select(name='Pressure reduced to MSL')[0].values.reshape(1, 1, 721, 1440),grib_file2.select(name='Pressure reduced to MSL')[0].values.reshape(1, 1, 721, 1440)),axis=1)
v10 = np.concatenate((grib_file1.select(name='10 metre U wind component')[0].values.reshape(1, 1, 721, 1440),grib_file2.select(name='10 metre U wind component')[0].values.reshape(1, 1, 721, 1440)),axis=1)
u10 = np.concatenate((grib_file1.select(name='10 metre V wind component')[0].values.reshape(1, 1, 721, 1440),grib_file2.select(name='10 metre V wind component')[0].values.reshape(1, 1, 721, 1440)),axis=1)

temperatrue = np.concatenate((temperatrue1.reshape(1, 1, -1, 721, 1440),temperatrue2.reshape(1, 1, -1, 721, 1440)),axis=1)
height = np.concatenate((height1.reshape(1, 1, -1, 721, 1440),height2.reshape(1, 1, -1, 721, 1440)),axis=1)
u = np.concatenate((u1.reshape(1, 1, -1, 721, 1440),u2.reshape(1, 1, -1, 721, 1440)),axis=1)
v = np.concatenate((v1.reshape(1, 1, -1, 721, 1440),v2.reshape(1, 1, -1, 721, 1440)),axis=1)
specific_humidity = np.concatenate((specific_humidity1.reshape(1, 1, -1, 721, 1440),specific_humidity2.reshape(1, 1, -1, 721, 1440)),axis=1)

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
    
save_path = f'input_data/aurora_input_{timestr}.nc'
inputs_ds.to_netcdf(save_path)
print('Done')
    

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--scheduled-time", required=True, help="format: 2023072000")
#     parser.add_argument("--scheduled-TC_ID", required=True, help="format: 202305W")
#     args = parser.parse_args()
#     main(args.scheduled_time, args.scheduled_TC_ID) 