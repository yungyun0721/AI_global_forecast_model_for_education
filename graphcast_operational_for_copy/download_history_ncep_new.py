import xarray as xr
import pandas as pd
import numpy as np
import os, glob
import requests
import argparse
from datetime import datetime, timedelta

# prepare for FCNV2 data
# IC_time = '2024072100'

TARGET_LEV = np.array([50, 100, 150, 200, 250, 300, 400, 500,
                       600, 700, 850, 925, 1000], dtype=np.int32)

def download_gfs(IC_time, out_path):
    url = (
        f"https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d084001/"
        f"{IC_time[0:4]}/{IC_time[0:8]}/"
        f"gfs.0p25.{IC_time}.f000.grib2"
    )
    print(url)
    r = requests.get(url)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)
        f.close()
        

def open_merge_grib(path, param_ids, extra_filter=None):
    ds_all = None

    for pid in param_ids:
        keys = {"paramId": pid}
        if extra_filter is not None:
            keys.update(extra_filter)

        tmp = xr.open_dataset(
            path,
            engine="cfgrib",
            filter_by_keys=keys,
        )

        ds_all = xr.merge([ds_all, tmp], compat="override") if ds_all is not None else tmp

    ds_all = ds_all.assign_coords(longitude=(ds_all.longitude % 360))
    ds_all = ds_all.sortby("longitude")
    ds_all = ds_all.sortby("latitude")

    return ds_all


def to_numpy_2d(da):
    return np.asarray(da).squeeze()


def to_numpy_3d(da):
    return np.asarray(da).squeeze()


def read_one_time(IC_time):
    
    base_inputs_path = 'graphcast_weight/stats/' 

    os.makedirs("input_data", exist_ok=True)
    IC_path = f"input_data/GFS_IC_raw_data_{IC_time}.grib2"
    download_gfs(IC_time, IC_path)

    # surface/static variables
    sfc_paramID = [165, 166, 167, 260074, 134, 3054, 228246, 228247]
    sfc = open_merge_grib(IC_path, sfc_paramID)

    # upper-air variables
    upper_paramID = [131, 132, 156, 157, 130]

    upper = open_merge_grib(
        IC_path,
        upper_paramID,
        extra_filter={"typeOfLevel": "isobaricInhPa"},
    )

    upper = upper.rename({"isobaricInhPa": "level"})
    upper = upper.sel(level=TARGET_LEV)
    upper = upper.sortby("level")

    # gh -> geopotential
    upper["z"] = upper["gh"] * 9.8
    upper = upper.drop_vars("gh")

    lat = np.asarray(sfc.latitude)
    lon = np.asarray(sfc.longitude)
    land_sea_mask = xr.load_dataset(base_inputs_path+'graphcast_land_sea_mask.nc').compute()
    geopotential_at_surface = xr.load_dataset(base_inputs_path+'graphcast_geopotential_at_surface.nc').compute()

    data = {
        "lat": lat,
        "lon": lon,
        "geopotential_at_surface": np.array(geopotential_at_surface.variables['geopotential_at_surface']).astype(np.float32),
        "land_sea_mask": np.array(land_sea_mask.variables['land_sea_mask']).astype(np.float32),
        "t2m": to_numpy_2d(sfc["t2m"]).astype(np.float32),
        "mslp": to_numpy_2d(sfc["msl"]).astype(np.float32),
        "v10": to_numpy_2d(sfc["v10"]).astype(np.float32),
        "u10": to_numpy_2d(sfc["u10"]).astype(np.float32),
        "tp": to_numpy_2d(sfc["tp"]).astype(np.float32) if "tp" in sfc else None,
        "temperature": to_numpy_3d(upper["t"]).astype(np.float32),
        "geopotential": to_numpy_3d(upper["z"]).astype(np.float32),
        "u": to_numpy_3d(upper["u"]).astype(np.float32),
        "v": to_numpy_3d(upper["v"]).astype(np.float32),
        "w": to_numpy_3d(upper["w"]).astype(np.float32),
        "specific_humidity": to_numpy_3d(upper["q"]).astype(np.float32),
    }

    for file in glob.glob(f"{IC_path[:-6]}*"):
        os.remove(file)

    print(f"finish {IC_time}")
    return data



def main(IC_time: str)-> None:
    dt = datetime.strptime(IC_time, "%Y%m%d%H")
    IC_time_m6 = (dt - timedelta(hours=6)).strftime("%Y%m%d%H")
    time_list = [IC_time_m6, IC_time]
    
    os.makedirs("input_data", exist_ok=True)
    
    all_data = [read_one_time(t) for t in time_list]
    print(all_data)
    lat = all_data[0]["lat"]
    lon = all_data[0]["lon"]

    def stack_time(name):
        # output shape: (batch, time, lat, lon)
        return np.stack([d[name] for d in all_data], axis=0)[None, ...]

    def stack_time_level(name):
        # output shape: (batch, time, level, lat, lon)
        return np.stack([d[name] for d in all_data], axis=0)[None, ...]

    # time_test = np.arange(len(time_list), dtype=np.int32)
    time_test = pd.array(['0 days 00:00:00', '0 days 06:00:00'], dtype='timedelta64[ns]')

    # datetime_test = np.array(time_list, dtype="datetime64[h]")[None, :]
    datetime_test =  pd.array([datetime.strptime(IC_time_m6,"%Y%m%d%H"), datetime.strptime(IC_time,"%Y%m%d%H")]).reshape([1,2])

    tp_nan = np.full_like(stack_time("t2m"), np.nan, dtype=np.float32)

    inputs_ds = xr.Dataset(
        data_vars={
            "geopotential_at_surface": (
                ("lat", "lon"),
                all_data[0]["geopotential_at_surface"],
            ),
            "land_sea_mask": (
                ("lat", "lon"),
                all_data[0]["land_sea_mask"],
            ),
            "2m_temperature": (
                ("batch", "time", "lat", "lon"),
                stack_time("t2m"),
            ),
            "mean_sea_level_pressure": (
                ("batch", "time", "lat", "lon"),
                stack_time("mslp"),
            ),
            "10m_v_component_of_wind": (
                ("batch", "time", "lat", "lon"),
                stack_time("v10"),
            ),
            "10m_u_component_of_wind": (
                ("batch", "time", "lat", "lon"),
                stack_time("u10"),
            ),
            "total_precipitation_6hr": (
                ("batch", "time", "lat", "lon"),
                tp_nan,
            ),
            "temperature": (
                ("batch", "time", "level", "lat", "lon"),
                stack_time_level("temperature"),
            ),
            "geopotential": (
                ("batch", "time", "level", "lat", "lon"),
                stack_time_level("geopotential"),
            ),
            "u_component_of_wind": (
                ("batch", "time", "level", "lat", "lon"),
                stack_time_level("u"),
            ),
            "v_component_of_wind": (
                ("batch", "time", "level", "lat", "lon"),
                stack_time_level("v"),
            ),
            "vertical_velocity": (
                ("batch", "time", "level", "lat", "lon"),
                stack_time_level("w"),
            ),
            "specific_humidity": (
                ("batch", "time", "level", "lat", "lon"),
                stack_time_level("specific_humidity"),
            ),
        },
        coords={
            "lon": ("lon", lon),
            "lat": ("lat", lat),
            "level": ("level", TARGET_LEV),
            "time": ("time", time_test),
            "datetime": (("batch", "time"), datetime_test),
        },
    )

    output_path = "input_data/GFS_initial_condition.nc"
    inputs_ds.to_netcdf(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduled-time", required=True, help="format: 2024072100")
    args = parser.parse_args()

    main(args.scheduled_time)
    
    