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

wsp_lev = [0,4,6,8,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,36,38,40,43,46,49,52,55,58,61,64,67,70,73,76,79,82,85]
wsp_color = ['#ffffff','#80ffff','#6fedf1','#5fdde4','#50cdd5','#40bbc7','#2facba','#1f9bac','#108c9f','#007a92',\
             '#00b432','#33c341','#67d251','#99e060','#cbf06f','#ffff80','#ffdd52','#ffdc52','#ffa63e','#ff6d29','#ff3713','#ff0000','#d70000','#af0000','#870000','#5f0000',\
             '#aa00ff','#b722fe','#c446ff','#d46aff','#e38dff','#f1b1ff','#ffd3ff',\
             '#ffc6ea','#ffb6d5','#ffa6c1','#ff97ac','#ff8798','#fe7884','#ff696e','#ff595a','#e74954','#cc3a4c','#b22846','#9a1941']

for i in range(files):
    print(i)
    data = np.load(f'../output_data/output_weather_{(i)*6}h.npy')

    # u = np.flip(data[7+13*2+11, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
    # v =  np.flip(data[7+13*3+11, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
    u = np.flip(data[18, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
    v =  np.flip(data[18+13, :, :], axis=0)[lat_min:lat_max, lon_min:lon_max]
    ws = (u**2+v**2)**0.5

    contourf = plt.contourf(lon, lat, ws, levels=wsp_lev, colors=wsp_color)
    plt.streamplot(lon, lat, u, v, color='k', linewidth=0.5, density=1.5)

    plt.plot(coast.lon_map, coast.lat_map, color='k', linewidth=0.7)
    plt.xlim([90, 155])
    plt.ylim([0,50])
    plt.colorbar(contourf)
    plt.title(f'+{i*6} hour, 850 mb Wind Speed')
    plt.savefig(f'predict_{i*6}.png')
    plt.close()
