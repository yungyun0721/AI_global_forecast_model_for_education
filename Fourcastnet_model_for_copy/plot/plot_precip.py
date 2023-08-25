import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.ioff()  

coast = pd.read_csv('coast.csv')

lat = np.linspace(-90,90,721)
lon = np.linspace(0,359.75,1440)

lat_min  = np.argwhere(lat==0)[0][0]
lat_max  = np.argwhere(lat==50.25)[0][0]
lon_min  = np.argwhere(lon==90)[0][0]
lon_max  = np.argwhere(lon==155.25)[0][0]

lat = lat[lat_min:lat_max]
lon = lon[lon_min:lon_max]

files = 0
file_list = os.listdir('../output_data/')
for file_name in file_list:
    if file_name.startswith('output_weather'):
        files += 1
        
print(f'files count: {files}')

precip_lev =  [0, 0.5, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130,150,200,300,400]
precip_color = [
    "#fdfdfd",  # 0.01 - 0.10 inches
    "#c9c9c9",  # 0.10 - 0.25 inches
    "#9dfeff",
    "#01d2fd",  # 0.25 - 0.50 inches
    "#00a5fe",  # 0.50 - 0.75 inches
    "#0177fd",  # 0.75 - 1.00 inches
    "#27a31b",  # 1.00 - 1.50 inches
    "#00fa2f",  # 1.50 - 2.00 inches
    "#fffe33",  # 2.00 - 2.50 inches
    "#ffd328",  # 2.50 - 3.00 inches
    "#ffa71f",  # 3.00 - 4.00 inches
    "#ff2b06",
    "#da2304",  # 4.00 - 5.00 inches
    "#aa1801",  # 5.00 - 6.00 inches
    "#ab1fa2",  # 6.00 - 8.00 inches
    "#db2dd2",  # 8.00 - 10.00 inches
    "#ff38fb",  # 10.00+
    "#ffd5fd"]


for i in range(files):
    print(i)
    data = np.load(f'../output_data/output_precipitation_{(i+1)*6}h.npy')

    precip = np.flip(data[:, :], axis=0)[lat_min:lat_max, lon_min:lon_max]


    contourf = plt.contourf(lon, lat, precip*1000, levels=precip_lev, colors=precip_color)
    # contourf = plt.contourf(lon, lat, precip)
    # plt.streamplot(lon, lat, u, v, color='k', linewidth=0.5, density=1.5)

    plt.plot(coast.lon_map, coast.lat_map, color='k', linewidth=0.7)
    plt.xlim([90, 155])
    plt.ylim([0,50])
    plt.colorbar(contourf)
    plt.title(f'+{i*6}~{i*6+6} hour, precipitation')
    plt.savefig(f'precipitation_{i*6}.png')
    plt.close()
