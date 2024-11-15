import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from colorbar import make_cmap


def plot_figure(data_source_file, save_file):
    plt.ioff()

    coast = pd.read_csv("coast.csv")

    if not os.path.isdir(save_file):
        os.mkdir(save_file)

    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(0, 359.75, 1440)

    lat_min = np.argwhere(lat == 0)[0][0]
    lat_max = np.argwhere(lat == 50.25)[0][0]
    lon_min = np.argwhere(lon == 90)[0][0]
    lon_max = np.argwhere(lon == 155.25)[0][0]

    lat = lat[lat_min:lat_max]
    lon = lon[lon_min:lon_max]

    xx = np.arange(90, 155.1, 0.25)
    yy = np.arange(0, 50.1, 0.25)
    X, Y = np.meshgrid(np.arange(0, 261, 1), np.arange(0, 201, 1))

    files = 0
    file_list = os.listdir(data_source_file)
    for file_name in file_list:
        if file_name.startswith("output_weather"):
            files += 1

    print(f"files count: {files}")

    for i in range(files):
        print(i)

        data = np.load(f"{data_source_file}/output_weather_{(i)*6}h.npy")
        fig, ax = plt.subplots(2, 3, figsize=(16, 9), dpi=200)
        ax = ax.flatten()

        #########################
        # # u = np.flip(data[7+13*2+11, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        # # v =  np.flip(data[7+13*3+11, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        # u = np.flip(data[18, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        # v = np.flip(data[18 + 13, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        # ws = (u**2 + v**2) ** 0.5
        #########################

        ### Surface
        msl = np.flip(data[6, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        u10 = np.flip(data[0, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        v10 = np.flip(data[1, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        ws10 = (u10**2 + v10**2) ** 0.5
        a0 = ax[0].imshow(
            ws10,
            cmap=make_cmap("clist_WS"),
            origin="lower",
            vmin=0,
            vmax=40,
        )
        ax[0].plot(
            (coast.lon_map - 90) / 0.25,
            (coast.lat_map - 0) / 0.25,
            color="k",
            linewidth=0.7,
        )
        ax[0].contour(
            msl / 100,
            levels=np.arange(900, 1033, 4),
            colors="k",
            linewidths=0.6,
        )
        ax[0].set_title("Wind Speed (m s$^{-1}$) and slp at sfc")
        ax[0].set_xticks(np.arange(0, 261, 40), np.arange(90, 156, 10))
        ax[0].set_yticks(np.arange(0, 201, 40), np.arange(0, 51, 10))
        ax[0].set_xlim([0, 261])
        ax[0].set_ylim([0, 201])
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(a0, cax=cax)

        ### 925 hPa
        u = np.flip(data[19, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        v = np.flip(data[32, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        t = np.flip(data[58, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        ws = (u**2 + v**2) ** 0.5
        a1 = ax[1].imshow(
            ws,
            cmap=make_cmap("clist_WS"),
            origin="lower",
            vmin=0,
            vmax=40,
        )
        ax[1].plot(
            (coast.lon_map - 90) / 0.25,
            (coast.lat_map - 0) / 0.25,
            color="k",
            linewidth=0.7,
        )
        ax[1].streamplot(
            np.arange(0, 261, 1),
            np.arange(0, 201, 1),
            u,
            v,
            density=[1.5, 1.5],
            color="k",
            linewidth=0.4,
        )
        ax[1].set_title("Wind Speed (shaded, m s$^{-1}$) at 925hPa")
        ax[1].set_xticks(np.arange(0, 261, 40), np.arange(90, 156, 10))
        ax[1].set_yticks(np.arange(0, 201, 40), np.arange(0, 51, 10))
        ax[1].set_xlim([0, 261])
        ax[1].set_ylim([0, 201])
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(a1, cax=cax)

        ### 850 hPa
        u = np.flip(data[18, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        v = np.flip(data[31, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        t = np.flip(data[57, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        ws = (u**2 + v**2) ** 0.5
        a2 = ax[2].imshow(
            t - 273.15,
            cmap=make_cmap("clist_temp"),
            origin="lower",
            vmin=-15,
            vmax=26,
        )
        ax[2].plot(
            (coast.lon_map - 90) / 0.25,
            (coast.lat_map - 0) / 0.25,
            color="k",
            linewidth=0.7,
        )
        ax[2].quiver(
            X[::4, ::4],
            Y[::4, ::4],
            u[::4, ::4],
            v[::4, ::4],
            angles="xy",
            scale_units="xy",
            scale=3.2,
        )
        ax[2].set_title("Wind and Temperature (shaded, $^{o}$C) at 850hPa")
        ax[2].set_xticks(np.arange(0, 261, 40), np.arange(90, 156, 10))
        ax[2].set_yticks(np.arange(0, 201, 40), np.arange(0, 51, 10))
        ax[2].set_xlim([0, 261])
        ax[2].set_ylim([0, 201])
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(a2, cax=cax)

        ### 700 hPa
        u = np.flip(data[17, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        v = np.flip(data[30, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        rh = np.flip(data[69, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        ws = (u**2 + v**2) ** 0.5
        a3 = ax[3].imshow(
            rh,
            cmap="gist_earth_r",
            origin="lower",
            vmin=50,
            vmax=110,
        )
        ax[3].plot(
            (coast.lon_map - 90) / 0.25,
            (coast.lat_map - 0) / 0.25,
            color="k",
            linewidth=0.7,
        )
        ax[3].streamplot(
            np.arange(0, 261, 1),
            np.arange(0, 201, 1),
            u,
            v,
            density=[1.5, 1.5],
            color="k",
            linewidth=0.4,
        )
        ax[3].set_title("Wind and Relative Humidity (shaded, %) at 700hPa")
        ax[3].set_xticks(np.arange(0, 261, 40), np.arange(90, 156, 10))
        ax[3].set_yticks(np.arange(0, 201, 40), np.arange(0, 51, 10))
        ax[3].set_xlim([0, 261])
        ax[3].set_ylim([0, 201])
        divider = make_axes_locatable(ax[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(a3, cax=cax)

        ### 500 hPa Streamline
        u = np.flip(data[15, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        v = np.flip(data[28, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        z = np.flip(data[41, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        ws = (u**2 + v**2) ** 0.5
        vor_u = (u[1:, :] - u[0:-1, :]) / 25000
        vor_u = np.concatenate((np.reshape(vor_u[0, :], (1, 261)), vor_u), axis=0)
        vor_v = (v[:, 1:] - v[:, 0:-1]) / 25000
        vor_v = np.concatenate((np.reshape(vor_v[:, 0], (201, 1)), vor_v), axis=1)
        vor = vor_u * (-1) + vor_v
        a4 = ax[4].imshow(
            10**5 * vor,
            cmap="bwr",
            origin="lower",
            vmin=-60,
            vmax=60,
        )
        ax[4].plot(
            (coast.lon_map - 90) / 0.25,
            (coast.lat_map - 0) / 0.25,
            color="k",
            linewidth=0.7,
        )
        ax[4].streamplot(
            np.arange(0, 261, 1),
            np.arange(0, 201, 1),
            u,
            v,
            density=[1.5, 1.5],
            color="k",
            linewidth=0.4,
        )
        ax[4].set_title("Wind and Vorticity (10$^{-5}$ s$^{-1}$) at 500hPa")
        ax[4].set_xticks(np.arange(0, 261, 40), np.arange(90, 156, 10))
        ax[4].set_yticks(np.arange(0, 201, 40), np.arange(0, 51, 10))
        ax[4].set_xlim([0, 261])
        ax[4].set_ylim([0, 201])
        divider = make_axes_locatable(ax[4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(a4, cax=cax)

        ### 500 hPa vorticity
        u = np.flip(data[15, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        v = np.flip(data[28, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        z = np.flip(data[41, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
        ws = (u**2 + v**2) ** 0.5
        vor_u = (u[1:, :] - u[0:-1, :]) / 25000
        vor_u = np.concatenate((np.reshape(vor_u[0, :], (1, 261)), vor_u), axis=0)
        vor_v = (v[:, 1:] - v[:, 0:-1]) / 25000
        vor_v = np.concatenate((np.reshape(vor_v[:, 0], (201, 1)), vor_v), axis=1)
        vor = vor_u * (-1) + vor_v
        a5 = ax[5].imshow(
            10**5 * vor,
            cmap="bwr",
            origin="lower",
            vmin=-60,
            vmax=60,
        )
        ax[5].plot(
            (coast.lon_map - 90) / 0.25,
            (coast.lat_map - 0) / 0.25,
            color="k",
            linewidth=0.7,
        )
        ax[5].contour(
            z / 10,
            levels=np.arange(5400, 6001, 40),
            colors="k",
            linewidths=0.6,
        )
        ax[5].set_title("Geopotential and Vorticity (10$^{-5}$ s$^{-1}$) at 500hPa")
        ax[5].set_xticks(np.arange(0, 261, 40), np.arange(90, 156, 10))
        ax[5].set_yticks(np.arange(0, 201, 40), np.arange(0, 51, 10))
        ax[5].set_xlim([0, 261])
        ax[5].set_ylim([0, 201])
        divider = make_axes_locatable(ax[5])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(a5, cax=cax)

        fig.suptitle(f"[+{i*6:0>3}h]", fontsize=20)
        plt.savefig(f"{save_file}/predict_{i*6:0>3}.png", facecolor="white")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_source_file", help="output data", default="../output_data"
    )
    parser.add_argument("--save_folder", help="plot save folder", default="plot_figure")
    args = parser.parse_args()
    plot_figure(args.data_source_file, args.save_folder)
