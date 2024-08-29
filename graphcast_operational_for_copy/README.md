此專案主要為他人之貢獻，是簡化版的預報。
相關細節請見：https://github.com/google-deepmind/graphcast

使用方式：
請git clone下來

進入複製下來的專案
``` 
cd */graphcast/
```

為了確保每個專案的環境互不干擾，建議開啟虛擬環境(非必要，可以不執行，但可能會影響其他沒有版控的專案)
```
python3 -m venv .venv
```

進入虛擬環境(呈上，非必要)
```
source .venv/bin/activate
```

在虛擬環境安裝此專案之相依套件。
```
pip install -r graphcast_requirements.txt
```
如果無法安裝成功，可嘗試用conda虛擬環境安裝
如執行後，遇到系統提示建議更新，則建議更新pip

下載模式及其他需要的資料創立graphcast_weight的資料夾
可參考現有的graphcast_weight資料夾(stats與parms)
**現在僅可使用graphcast operational版本**
graphcast模式必須放在專案目錄，也就是
```
├── root
│   ├── graphcast
│   ├── graphcast_weight
│   │   ├── stats
│   │   │   ├── graphcast_land_sea_mask.nc
│   │   │   ├── graphcast_geopotential_at_surface.nc
│   │   │   ├── stats_diffs_stddev_by_level.nc
│   │   │   ├── stats_mean_by_level.nc
│   │   │   ├── stats_stddev_by_level.nc
│   │   ├── parms
│   │   │   ├── params_GraphCast_operational.npz
│   ├── input_data
│   │   ├── graphcast_input_data.nc
│   ├── plot_code
│   │   ├── coast.csv
│   │   ├── plot_precipitation.py
│   │   ├── plot.plot_v850.py
│   ├── src
│   ├── supporting_module
│   │   ├── inference_helper.py
│   ├── download_history_ncep.py
│   ├── download_ncep.py
│   ├── graphcast_requirements.txt
│   ├── main.py
│   ├── README.md
```

下載天氣資料，範例：
```
python download_ncep.py --scheduled-time 2023072006
```
其中，2023072006代表2023年7月20號06Z的NCEP初始場，
相關細節請見 https://nomads.ncep.noaa.gov/dods/gfs_0p25
這個下載過程約需要兩分鐘

此步驟會下載一個檔案至*/input_data/graphcast_input_data.nc。
內包含兩個時間。(graphcast需要兩個時間做預報)檔案約1.3G，請確保有足夠儲存空間。

開始跑graphcast模式。由於82有12個CPU，因我不會調整cpu核心數，目前設定全開。

模式跑10天積分。請改成一下指令。
```
積分時間預設10天預報(40步階)
python main.py --input_data input_data/graphcast_input_data.nc --output_folder output_data

更改積分時間(ex:3天預報時間)
python main.py --input_data input_data/graphcast_input_data.nc --output_folder output_data --fore_hr 72
```

在82及我設定的環境，每10分鐘可積分6小時。執行後，output_data/底下會出現許多檔案，
gc_operational_predict_data_0.nc
gc_operational_predict_data_1.nc
gc_operational_predict_data_2.nc .......


_後的數字代表積分日數，例如_4.npy代表積分4*6小時。
每個積分會產生336M的檔案，請確保有足夠的儲存空間。

結果為nc檔可自行使用程式直接查看相對應的各變數

output資料細節介紹：
在gc_operational_predict_data_X.nc
地表層場含有t2m, mslp, u10, v10, total_precipitation
等壓層有13層:[50,100,200,250,300,400,500,600,700,850,925,1000]百帕，皆有T, Z, U, V, W, Q。

關於output檔案的使用，以850mb繪圖為例，執行以下程式碼可看到850風速與雨量結果。
```
可查看plot_code/plot_precipitation.py 跟 plot_v850.py的檔案
```

此命令會在plot_save/目錄底下生成圖片。

離開專案，退出虛擬環境
```
deactivate
```


