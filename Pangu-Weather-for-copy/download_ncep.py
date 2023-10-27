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
    target_lev = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    indices = np.where(np.isin(lev, target_lev))[0]
    
    height = np.flip(ncep.hgtprs[0, indices, :, :].values * 9.80665, axis=1).reshape(1, -1, 721, 1440)
    specific_humidity = np.flip(ncep.spfhprs[0, indices, :, :].values, axis=1).reshape(1, -1, 721, 1440)
    temperatrue = np.flip(ncep.tmpprs[0, indices, :, :].values, axis=1).reshape(1, -1, 721, 1440)
    u = np.flip(ncep.ugrdprs[0, indices, :, :].values, axis=1).reshape(1, -1, 721, 1440)
    v = np.flip(ncep.vgrdprs[0, indices, :, :].values, axis=1).reshape(1, -1, 721, 1440)

    mslp = np.flip(ncep.msletmsl[0, :, :].values, axis=0).reshape(1, 721, 1440)
    v10 = np.flip(ncep.vgrd10m[0, :, :].values, axis=0).reshape(1, 721, 1440)
    u10 = np.flip(ncep.ugrd10m[0, :, :].values, axis=0).reshape(1, 721, 1440)
    t2m = np.flip(ncep.tmp2m[0, :, :].values, axis=0).reshape(1, 721, 1440)
    
    upper = np.concatenate([height, specific_humidity, temperatrue, u, v], axis=0)
    sfc = np.concatenate([mslp, u10, v10, t2m], axis=0)
    
    if not os.path.isdir('input_data'):
        os.mkdir('input_data')
    np.save('./input_data/input_upper.npy', upper.astype(np.float32))    
    np.save('./input_data/input_surface.npy', sfc.astype(np.float32))  
    print('Done')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduled-time", required=True, help="format: 2023072000")
    args = parser.parse_args()
    main(args.scheduled_time)    
