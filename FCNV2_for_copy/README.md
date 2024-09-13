此專案主要為他人之貢獻。
相關細節請見： https://github.com/ecmwf-lab/ai-models-fourcastnetv2 以及 https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_fcnv2_sm
FourCastNetv2內部code


使用方式：

請將本目錄複製到自己想要的地方

進入複製下來的專案
```
cd */FCNV2_for_copy/ 
```

為了確保每個專案的環境互不干擾，建議開啟虛擬環境(非必要，可以不執行，但可能會影響其他沒有版控的專案)
```
conda create --name FCNV2 python==3.10
```

進入虛擬環境(呈上，非必要)
``` 
conda activate FCNV2
```

更新pip
```
pip install --upgrade pip 
```

在虛擬環境安裝此專案之相依套件。
``` 
pip install -r requirements.txt 
```
如執行後，遇到系統提示建議更新，則建議更新pip

weighting部分:
    FourCastNetv2有73個變數:
    相關下載可參考
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_fcnv2_sm
    
    wget 'https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_fcnv2_sm/versions/v0.2/files/fcnv2_sm.zip'
    unzip fcnv2_sm.zip
    mv fcnv2_sm weight

或是直接使用我的weight
在modules/inference_weather.py檔中，更改weight_path位置
```
weight_path_global = './weight'
```
注意，由於我有可能會更改或移動專案位置，所以行有餘力時請直接複製或下載weight，但請注意儲存空間。

FourCastNetv2完成安裝後，目錄如下
```
├── root
│   ├── fourcastnetv2
│   ├── from_official
│   ├── input_data
│   │   ├── initial_condition.npy
│   ├── modules
│   │   ├── inference_helper.py
│   │   ├── inference_helper.py
│   ├── output_data
│   ├── plot
│   │   ├── coast.csv
│   │   ├── plot850.py
│   ├── weight
│   │   ├── global_means.npy
│   │   ├── global_stds.npy
│   │   ├── weights.tar
│   ├── download_history_ncep.py
│   ├── download_ncep.py
│   ├── model.py
│   ├── README.md
│   ├── requirements.txt
```

下載天氣資料，範例：
```
python download_ncep.py --scheduled-time 2023072006 
```
其中，2023072006代表2023年7月20號06Z的NCEP初始場，
相關細節請見 https://nomads.ncep.noaa.gov/dods/gfs_0p25
這個下載過程約需要兩分鐘


此步驟會下載一個檔案至*/input_data/inital_condition.npy。
Fourcastnetv2將此檔作為初始場，此檔案須290M，請確保有足夠儲存空間。

開始跑Fourcastnetv2模式。
(或許有調整到cpu核心，supporting_module/inference_weather.py 第28行 cpu_num = 10)

模式跑10天積分。請改成一下指令。
```
積分時間預設10天預報(40步階)
python main.py --input_data input_data/inital_condition.npy --output_folder output_data

更改積分時間(ex:3天預報時間)
python main.py --input_data input_data/inital_condition.npy --output_folder output_data --fore_hr 72
```

在82及我設定的環境，每10分鐘可積分6小時。執行後，output_data/底下會出現許多檔案，
output_weather_0hr.npy
output_weather_6hr.npy
output_weather_12hr.npy
output_weather_18hr.npy.......


_後的數字代表積分時間。
每個積分會產生290M的檔案，請確保有足夠的儲存空間。

結果為nc檔可自行使用程式直接查看相對應的各變數

output資料細節介紹：
output_weather_Xh.npy的矩陣維度為(26,721,1440)，73依序代表
ordering = [ "10u",   "10v", "100u", "100v",   "2t",   "sp",  "msl", "tcwv",\
             "u50",  "u100", "u150", "u200", "u250", "u300", "u400", "u500",\
             "u600", "u700", "u850", "u925","u1000",  "v50", "v100", "v150",\
             "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850",\
             "v925","v1000",  "z50", "z100", "z150", "z200", "z250", "z300",\
             "z400", "z500", "z600", "z700", "z850", "z925","z1000",  "t50",\
             "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600",\
             "t700", "t850", "t925","t1000",  "r50", "r100", "r150", "r200",\
             "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000"]
721與1440分別代表緯度與經度。

關於output檔案的使用，以850mb繪圖為例，執行以下程式碼可看到850風速結果。
``` 
cd plot/ 
python plot850.py 
```

此命令會在plot/目錄底下生成圖片。

離開專案，退出虛擬環境
```
deactivate
```

