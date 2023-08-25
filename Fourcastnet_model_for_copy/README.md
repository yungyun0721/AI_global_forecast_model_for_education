此專案主要為他人之貢獻。
相關細節請見： https://github.com/ecmwf-lab/ai-models 以及 https://github.com/NVlabs/FourCastNet
FourCastNet內部code


使用方式：

請將本目錄複製到自己想要的地方

進入複製下來的專案
```
cd */Fourcastnet_model_for_copy/ 
```

為了確保每個專案的環境互不干擾，建議開啟虛擬環境(非必要，可以不執行，但可能會影響其他沒有版控的專案)
```
python3 -m venv FCN_env 
```

進入虛擬環境(呈上，非必要)
``` 
source FCN_env/bin/activate 
```

更新pip
```
pip install --upgrade pip 
```

在虛擬環境安裝此專案之相依套件。只寫cpu版本。
``` 
pip install -r requirements_cpu.txt 
```
如執行後，遇到系統提示建議更新，則建議更新pip

weighting部分:
    FourCastNet已更新成26個變數
    model_weather: https://portal.nersc.gov/project/m4134/FCN_weights_v0.1/ 中 backbone_v0.1.ckpt(需改名成backbone.ckpt)

    預報6hr雨量只需20個變數
    precip_model: https://portal.nersc.gov/project/m4134/FCN_weights_v0/ 中 precip.ckpt

    其他變數:
    https://portal.nersc.gov/project/m4134/FCN_weights_v0.1/stats_v0.1/
    需下載 global_means.npy 及 global_stds.npy (FourCastNet變數前期處理會用到)

或是直接使用我的weight
在inference.py檔中，更改weight_path位置
```
weight_path = "/wk171/yungyun/FCN_test_from_ECMWF/ai-models/" 
```
注意，由於我有可能會更改或移動專案位置，所以行有餘力時請直接複製或下載weight，但請注意儲存空間。

FourCastNet必須放在專案/model_weight目錄下，也就是
```
├── root
│   ├── input_data
│   │   ├── initial_condition.npy
│   ├── output_data
│   ├── plot
│   │   ├── coast.csv
│   │   ├── plot850.py
│   │   ├── plotprecip.py
│   ├── model_weight
│   │   ├── backbone.ckpt
│   │   ├── precip.ckpt
│   │   ├── global_means.npy
│   │   ├── global_stds.npy
│   ├── afnonet.py
│   ├── download_ncep.py
│   ├── inference_helper.py
│   ├── inference.py
│   ├── model.py
│   ├── requirements_cpu.txt
```

下載天氣資料，範例：
```
python download_ncep.py --scheduled-time 2023072006 
```
其中，2023072006代表2023年7月20號06Z的NCEP初始場，
相關細節請見 https://nomads.ncep.noaa.gov/dods/gfs_0p25
這個下載過程約需要兩分鐘

此步驟會下載一個檔案至*/input_data/initial_condition.npy。
FourCastNet將此檔作為初始場，此檔案須103M，請確保有足夠儲存空間。

開始跑FourCastNet，使用CPU模式跑10天積分。如有需求更改請從inference_cpu.py第81行修改。
```
 python inference_cpu.py 
```

預報完天氣成場後，會直接進入雨量預報，因此有兩個檔案，皆會出現在output_data/底下
output_precipitation_6h.npy
output_precipitation_12h.npy .......

output_weather_6h.npy
output_weather_12h.npy .......

_後的數字代表預報的第幾個小時，例如_6h.npy代表積分6小時。
每個積分會產生4M + 102M的檔案，請確保有足夠的儲存空間。

output_weather_Xh.npy的矩陣維度為(26,721,1440)，26依序代表u10, v10, t2m, sp, mslp, t850, u1000, v1000, z1000,
                                                        u850, v850, z850, u500, v500, z500, t500, z50, r500,
                                                        r850, tcwv, wind_100u, wind_100v, u250, v250, z250, t250。
output_precipitation_Xh.npy的矩陣維度為(721,1440)，代表往前六小時的累積雨量。
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

