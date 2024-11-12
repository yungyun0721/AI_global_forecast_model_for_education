import argparse
import requests
import numpy as np
import xarray, os
def main(timestr: str) -> None:

    if not os.path.isdir('input_data'):
        os.mkdir('input_data')
    save_folder = 'input_data/inital_condition_IFS.npy'
    grib_save = 'input_data/IFS_IC.grib2'
    download_name =f'https://storage.googleapis.com/ecmwf-open-data/{timestr[:8]}/{timestr[8:]}z/ifs/0p25/oper/{timestr}0000-0h-oper-fc.grib2'
    print(download_name)
    response = requests.get(download_name)
    with open(grib_save, "wb") as f:
        f.write(response.content)
        f.close()

    ds_pressure = xarray.open_dataset(grib_save,filter_by_keys={'typeOfLevel': 'isobaricInhPa'})
    ds_surface = xarray.open_dataset(grib_save,filter_by_keys={'typeOfLevel': 'surface'})
    ds_10m = xarray.open_dataset(grib_save,filter_by_keys={'level': 10})
    ds_2m = xarray.open_dataset(grib_save,filter_by_keys={'level': 2})
    ds_100m = xarray.open_dataset(grib_save,filter_by_keys={'level': 100})
    ds_msl = xarray.open_dataset(grib_save,filter_by_keys={'typeOfLevel': 'meanSea'})
    ds_entir = xarray.open_dataset(grib_save,filter_by_keys={'typeOfLevel':'entireAtmosphere'})

    target_lev = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    # target_lev = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    mslp = np.roll(np.array(ds_msl.variables['msl'][:]).reshape(1, 721, 1440),720,axis=-1)
    v10 = np.roll(np.array(ds_10m.variables['v10'][:]).reshape(1, 721, 1440),720,axis=-1)
    u10 = np.roll(np.array(ds_10m.variables['u10'][:]).reshape(1, 721, 1440),720,axis=-1)
    t2m = np.roll(np.array(ds_2m.variables['t2m'][:]).reshape(1, 721, 1440),720,axis=-1)
    sp = np.roll(np.array(ds_surface.variables['sp'][:]).reshape(1, 721, 1440),720,axis=-1)
    tcwv = np.roll(np.array(ds_entir.variables['tcwv'][:]).reshape(1, 721, 1440),720,axis=-1)
    wind_100u = np.roll(np.array(ds_100m.variables['u100'][:]).reshape(1, 721, 1440),720,axis=-1)
    wind_100v = np.roll(np.array(ds_100m.variables['v100'][:]).reshape(1, 721, 1440),720,axis=-1)
    
    u_upper = np.roll(np.flip(np.array(ds_pressure.variables['u'][:]), axis=0).reshape(13, 721, 1440),720,axis=-1)
    v_upper = np.roll(np.flip(np.array(ds_pressure.variables['v'][:]), axis=0).reshape(13, 721, 1440),720,axis=-1)
    z_upper = np.roll(np.flip(np.array(ds_pressure.variables['gh'][:]), axis=0).reshape(13, 721, 1440),720,axis=-1)*9.8
    t_upper = np.roll(np.flip(np.array(ds_pressure.variables['t'][:]), axis=0).reshape(13, 721, 1440),720,axis=-1)
    r_upper = np.roll(np.flip(np.array(ds_pressure.variables['r'][:]), axis=0).reshape(13, 721, 1440),720,axis=-1)


    IC = np.concatenate([u10, v10, wind_100u, wind_100v, t2m, sp, mslp, tcwv,  \
                         u_upper, v_upper, z_upper, t_upper,  r_upper], axis=0)

    
    np.save(f'./{save_folder}', IC.astype(np.float32))
    print('Done')


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduled-time", required=True, help="format: 2023072000")
    args = parser.parse_args()
    main(args.scheduled_time)    
    
    
#     # Input
#     area = [90, 0, -90, 360 - 0.25]
#     grid = [0.25, 0.25]

#     param_sfc = ["10u", "10v", "2t", "sp", "msl", "tcwv", "100u", "100v"]

#     param_level_pl = (
#         ["t", "u", "v", "z", "r"],
#         [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
#     )

#     ordering = [
#     0.    "10u",
#     1.    "10v",
#     2.    "100u",
#     3.    "100v",
#     4.    "2t",
#     5.    "sp",
#     6.    "msl",
#     7.    "tcwv",
#     8.    "u50",
#     9.    "u100",
#     10.   "u150",
#     11.   "u200",
#     12.   "u250",
#     13.   "u300",
#     14.   "u400",
#     15.   "u500",
#     16.   "u600",
#     17.   "u700",
#     18.   "u850",
#     19.   "u925",
#     20.   "u1000",
#     21.   "v50",
#     22.   "v100",
#     23.   "v150",
#     24.   "v200",
#     25.   "v250",
#     26.   "v300",
#     27.   "v400",
#     28.   "v500",
#     29.   "v600",
#     30.   "v700",
#     31.   "v850",
#     32.   "v925",
#     33.   "v1000",
#     34.   "z50",
#     35.   "z100",
#     36.   "z150",
#     37.   "z200",
#     38.   "z250",
#     39.   "z300",
#     40.   "z400",
#     41.   "z500",
#     42.   "z600",
#     43.   "z700",
#     44.   "z850",
#     45.   "z925",
#     46.   "z1000",
#     47.   "t50",
#     48.   "t100",
#     49.   "t150",
#     50.   "t200",
#     51.   "t250",
#     52.   "t300",
#     53.   "t400",
#     54.   "t500",
#     55.   "t600",
#     56.   "t700",
#     57.   "t850",
#     58.   "t925",
#     59.   "t1000",
#     60.   "r50",
#     61.   "r100",
#     62.   "r150",
#     63.   "r200",
#         "r250",
#         "r300",
#         "r400",
#         "r500",
#         "r600",
#         "r700",
#         "r850",
#         "r925",
#         "r1000",
#     ]