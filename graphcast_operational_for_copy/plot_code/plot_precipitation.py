import numpy as np
import os,xarray
import pandas as pd
import matplotlib.pyplot as plt



output_folder = '/bk2/yungyun/dwp/graphcast/output_data_2022052600/'
output_files = os.listdir(output_folder)
output_files.sort()
coast = pd.read_csv('coast.csv')

save_folder = 'plt_save'
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)


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


lat = np.linspace(-90,90,721)
lon = np.linspace(0,359.75,1440)

lat_min  = np.argwhere(lat==0)[0][0]
lat_max  = np.argwhere(lat==50.25)[0][0]
lon_min  = np.argwhere(lon==90)[0][0]
lon_max  = np.argwhere(lon==155.25)[0][0]

lat = lat[lat_min:lat_max]
lon = lon[lon_min:lon_max]



for case_i in range(len(output_files)):
    plot_file_path = output_folder+output_files[case_i]
    example_batch = xarray.load_dataset(plot_file_path).compute()
    
    plt.figure(dpi=300)

    precip = example_batch.variables['total_precipitation_6hr'][0,0,:,:][lat_min:lat_max, lon_min:lon_max]
    contourf = plt.contourf(lon, lat, precip*1000, levels=precip_lev, colors=precip_color)


    plt.plot(coast.lon_map, coast.lat_map, color='k', linewidth=0.7)
    plt.xlim([90, 155])
    plt.ylim([0,50])
    plt.title(f'GraphCast precipitation: {output_files[case_i][:3]}')
    plt.colorbar()
    plt.savefig(f'{save_folder}/precipitation_{output_files[case_i][:-3]}.png')
    plt.close()