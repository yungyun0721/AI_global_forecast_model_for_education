此專案主要為他人之貢獻，是簡化版的預報。
相關細節請見： https://github.com/198808xc/Pangu-Weather

使用方式：
請git clone下來

進入複製下來的專案
``` 
cd */Pangu-Weather-for-copy/
```

為了確保每個專案的環境互不干擾，建議開啟虛擬環境(非必要，可以不執行，但可能會影響其他沒有版控的專案)
```
python3 -m venv .venv
```

進入虛擬環境(呈上，非必要)
```
source .venv/bin/activate
```

在虛擬環境安裝此專案之相依套件。在171請安裝CPU版本，因171的GPU記憶體不足，跑不動盤古模式。
```
pip install -r requirements_cpu.txt
```
如執行後，遇到系統提示建議更新，則建議更新pip

下載盤古模式。每個模式約1.1G，可從上面github連結下載，或使用軟連結至我載的24小時預報。
盤古模式必須放在專案目錄，也就是
```
├── root
│   ├── input_data
│   │   ├── input_surface.npy
│   │   ├── input_upper.npy
│   ├── output_data
│   ├── pangu_weather_1.onnx
│   ├── pangu_weather_3.onnx
│   ├── pangu_weather_6.onnx
│   ├── pangu_weather_24.onnx
│   ├── inference_cpu.py
│   ├── inference_gpu.py
│   ├── inference_iterative.py
```

下載天氣資料，範例：
```
python download_ncep.py --scheduled-time 2023072006
```
其中，2023072006代表2023年7月20號06Z的NCEP初始場，
相關細節請見 https://nomads.ncep.noaa.gov/dods/gfs_0p25
這個下載過程約需要兩分鐘

此步驟會下載兩個檔案至*/input_data/，分別是input_surface.npy、input_upper.npy。
盤古模式會將這兩個檔案作為初始場。這兩個檔案分別為16M與270M，請確保有足夠儲存空間。

開始跑盤古模式。由於171有16個CPU，因此我預設讓盤古模式使用10個CPU，同時所有記憶體設定全開。
如遇到記憶體不足問題，請至inference_cpu.py將ort.SessionOptions()記憶體設定為False。
如多人同時執行程式造成CPU不夠，請在第18行減少CPU數量。
請注意，若將options.intra_op_num_threads 設定為-1，代表會使用機器所有的CPU，
將有可能影響他人的程式或影響定時排程。

模式跑10天積分。如有需求更改請從inference_cpu.py第34行修改。
```
python inference_cpu.py
```

在171及我設定的環境，每40秒可積分一天。執行後，output_data/底下會出現許多檔案，
output_surface_0.npy
output_surface_1.npy .......

output_upper_0.npy
output_upper_1.npy .......

_後的數字代表積分日數，例如_4.npy代表積分4天。
每個積分會產生16M + 270M的檔案，請確保有足夠的儲存空間。

output_surface_X.npy的矩陣維度為(4,721,1440)，4依序代表MSLP, U10, V10, T2M。721與1440分別代表緯度與經度
output_upper_X.npy的矩陣維度為(5,13,721,1440)，5依序代表Z, Q, T, U, V； 
13代表層場，分別為[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]百帕。
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

