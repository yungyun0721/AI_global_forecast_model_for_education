import argparse

import numpy as np
import xarray, os
def main(timestr: str) -> None:

    path = (
        f'http://nomads.ncep.noaa.gov:80/dods/gfs_0p25/gfs{timestr[:-2]}/gfs_0p25_{timestr[-2:]}z'
    )
    print(path)
    ncep = xarray.open_dataset(path)

    lev = ncep.lev
    target_lev = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    # target_lev = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    indices = np.where(np.isin(lev, target_lev))[0]

    # height = np.flip(ncep.hgtprs[0, indices, :, :].values * 9.80665, axis=1).reshape(1, -1, 721, 1440)
    # specific_humidity = np.flip(ncep.spfhprs[0, indices, :, :].values, axis=1).reshape(1, -1, 721, 1440)
    # temperatrue = np.flip(ncep.tmpprs[0, indices, :, :].values, axis=1).reshape(1, -1, 721, 1440)
    # u = np.flip(ncep.ugrdprs[0, indices, :, :].values, axis=1).reshape(1, -1, 721, 1440)
    # v = np.flip(ncep.vgrdprs[0, indices, :, :].values, axis=1).reshape(1, -1, 721, 1440)


    mslp = np.flip(ncep.msletmsl[0, :, :].values, axis=0).reshape(1, 721, 1440)
    v10 = np.flip(ncep.vgrd10m[0, :, :].values, axis=0).reshape(1, 721, 1440)
    u10 = np.flip(ncep.ugrd10m[0, :, :].values, axis=0).reshape(1, 721, 1440)
    t2m = np.flip(ncep.tmp2m[0, :, :].values, axis=0).reshape(1, 721, 1440)
    sp = np.flip(ncep.pressfc[0, :, :].values, axis=0).reshape(1, 721, 1440)
    tcwv = np.flip(ncep.pwatclm[0, :, :].values, axis=0).reshape(1, 721, 1440)
    wind_100u = np.flip(ncep.ugrd100m[0, :, :].values, axis=0).reshape(1, 721, 1440) 
    wind_100v = np.flip(ncep.vgrd100m[0, :, :].values, axis=0).reshape(1, 721, 1440) 

    u_upper = np.flip(np.flip(ncep.ugrdprs[0, indices, :, :].values, axis=1), axis=0).reshape(13, 721, 1440)
    v_upper = np.flip(np.flip(ncep.vgrdprs[0, indices, :, :].values, axis=1), axis=0).reshape(13, 721, 1440)
    z_upper = np.flip(np.flip(ncep.hgtprs[0, indices, :, :].values, axis=1), axis=0).reshape(13, 721, 1440)*9.80665
    t_upper = np.flip(np.flip(ncep.tmpprs[0, indices, :, :].values, axis=1), axis=0).reshape(13, 721, 1440)
    r_upper = np.flip(np.flip(ncep.rhprs[0, indices, :, :].values, axis=1), axis=0).reshape(13, 721, 1440)
    


    IC = np.concatenate([u10, v10, wind_100u, wind_100v, t2m, sp, mslp, tcwv,  \
                         u_upper, v_upper, z_upper, t_upper,  r_upper], axis=0)

    
    if not os.path.isdir('input_data'):
        os.mkdir('input_data')
    np.save('./input_data/inital_condition.npy', IC.astype(np.float32))
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