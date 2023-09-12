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
    target_lev = [1000, 850, 500, 250, 50]
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

    t850 = np.flip(ncep.tmpprs[0, indices[1], :, :].values, axis=0).reshape(1, 721, 1440) 
    u1000 = np.flip(ncep.ugrdprs[0, indices[0], :, :].values, axis=0).reshape(1, 721, 1440) 
    v1000 = np.flip(ncep.vgrdprs[0, indices[0], :, :].values, axis=0).reshape(1, 721, 1440) 
    z1000 = np.flip(ncep.hgtprs[0, indices[0], :, :].values, axis=0).reshape(1, 721, 1440)*9.8
    u850 = np.flip(ncep.ugrdprs[0, indices[1], :, :].values, axis=0).reshape(1, 721, 1440) 
    v850 = np.flip(ncep.vgrdprs[0, indices[1], :, :].values, axis=0).reshape(1, 721, 1440) 
    z850 = np.flip(ncep.hgtprs[0, indices[1], :, :].values, axis=0).reshape(1, 721, 1440) *9.8
    u500 = np.flip(ncep.ugrdprs[0, indices[2], :, :].values, axis=0).reshape(1, 721, 1440) 
    v500 = np.flip(ncep.vgrdprs[0, indices[2], :, :].values, axis=0).reshape(1, 721, 1440) 
    z500 = np.flip(ncep.hgtprs[0, indices[2], :, :].values, axis=0).reshape(1, 721, 1440) *9.8
    t500 = np.flip(ncep.tmpprs[0, indices[2], :, :].values, axis=0).reshape(1, 721, 1440) 
    z50 = np.flip(ncep.hgtprs[0, indices[4], :, :].values, axis=0).reshape(1, 721, 1440) *9.8
    r500 = np.flip(ncep.rhprs[0, indices[2], :, :].values, axis=0).reshape(1, 721, 1440) 
    r850 = np.flip(ncep.rhprs[0, indices[1], :, :].values, axis=0).reshape(1, 721, 1440)
    u250 = np.flip(ncep.ugrdprs[0, indices[3], :, :].values, axis=0).reshape(1, 721, 1440) 
    v250 = np.flip(ncep.vgrdprs[0, indices[3], :, :].values, axis=0).reshape(1, 721, 1440)
    z250 = np.flip(ncep.hgtprs[0, indices[3], :, :].values, axis=0).reshape(1, 721, 1440) *9.8
    t250 = np.flip(ncep.tmpprs[0, indices[3], :, :].values, axis=0).reshape(1, 721, 1440)
    

    # IC = np.concatenate([u10, v10, t2m, sp, mslp, t850, u1000, v1000, z1000,\
    #                     u850, v850, z850, u500, v500, z500, t500, z50, r500,\
    #                     r850, tcwv], axis=0)
    IC = np.concatenate([u10, v10, t2m, sp, mslp, t850, u1000, v1000, z1000,\
                        u850, v850, z850, u500, v500, z500, t500, z50, r500,\
                        r850, tcwv, wind_100u, wind_100v, u250, v250, z250, t250], axis=0)
    
    if not os.path.isdir('input_data'):
        os.mkdir('input_data')
    np.save('./input_data/inital_condition.npy', IC.astype(np.float32))
    print('Done')


#         "100u",
#         "100v",
#         "u250",
#         "v250",
#         "z250",
#         "t250",
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduled-time", required=True, help="format: 2023072000")
    args = parser.parse_args()
    main(args.scheduled_time)    

# upper = np.concatenate([height, specific_humidity, temperatrue, u, v], axis=0)
# sfc = np.concatenate([mslp, v10, u10, t2m], axis=0)

# var_key_dict = {
#     0: "u10",
#     1: "v10",
#     2: "t2m",
#     3: "sp",
#     4: "msl",
#     5: "t850",
#     6: "u1000",
#     7: "v1000",
#     8: "z1000",
#     9: "u850",
#     10: "v850",
#     11: "z850",
#     12: "u500",
#     13: "v500",
#     14: "z500",
#     15: "t500",
#     16: "z50",
#     17: "r500",
#     18: "r850",
#     19: "tcwv",
# }


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

