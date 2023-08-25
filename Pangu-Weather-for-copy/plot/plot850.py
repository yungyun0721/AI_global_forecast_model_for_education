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
    if file_name.startswith('output_upper'):
        files += 1
        
print(f'files count: {files}')

for i in range(files):
    print(i)
    data = np.load(f'../output_data/output_upper_{i}.npy')

    u = np.flip(data[3, 2, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
    v =  np.flip(data[4, 2, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
    ws = (u**2+v**2)**0.5

    contourf = plt.contourf(lon, lat, ws, levels=np.linspace(0,40,41), cmap='jet')
    plt.streamplot(lon, lat, u, v, color='k', linewidth=0.5, density=1.5)

    plt.plot(coast.lon_map, coast.lat_map, color='k', linewidth=0.7)
    plt.xlim([90, 155])
    plt.ylim([0,50])
    plt.colorbar(contourf)
    plt.title(f'+{i*24} hour, 850 mb Wind Speed')
    plt.savefig(f'predict_{i*24}.png')
    plt.close()
