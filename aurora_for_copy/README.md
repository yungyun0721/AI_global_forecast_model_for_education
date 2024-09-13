此專案主要為他人之貢獻。
相關細節請見： https://github.com/microsoft/aurora、https://microsoft.github.io/aurora/intro.html 以及 https://github.com/ecmwf-lab/ai-models-aurora
Aurora內部code


使用方式：

請將本目錄複製到自己想要的地方

進入複製下來的專案
```
cd */aurora_for_copy/ 
```

為了確保每個專案的環境互不干擾，建議開啟虛擬環境(非必要，可以不執行，但可能會影響其他沒有版控的專案)(建議可用conda虛擬環境安裝)
```
conda create --name aurora python=3.10
```

進入虛擬環境(呈上，非必要)
``` 
conda activate aurora
```

更新pip
```
pip install --upgrade pip 
```

在虛擬環境安裝此專案之相依套件。
``` 
pip install -r requirements.txt 
pip install --force-reinstall charset-normalizer==3.1.0
```
如執行後，遇到系統提示建議更新，則建議更新pip

weighting部分:
    可參考https://huggingface.co/microsoft/aurora/tree/main (Files中)
    目前使用的是aurora-0.25-finetuned.ckpt的版本 (下載後需改名成aurora-0.25-finetuned.ckpt)
    若下載不同版本後可至supporting_module中，inference_helper.py的第15行(目前僅能使用0.25的版本)

    另外，使用的aurora model，在supporting_module中，inference_helper.py的第13行
    model = Aurora(use_lora=False)
    use_lora，代表兩種不同的版本詳細可參考https://microsoft.github.io/aurora/intro.html 

    其他變數:(static.nc檔)可參考 aurora_for_copy/from_official/docs/example_era5.ipynb

或是直接使用我的weight
在supporting_module中，inference_helper.py檔中第15行，更改weight位置
```
model.load_checkpoint("weight/aurora-0.25-finetuned.ckpt", strict=False)
```
注意，由於我有可能會更改或移動專案位置，所以行有餘力時請直接複製或下載weight，但請注意儲存空間。

完成後目錄如下
```
├── root
│   ├── aurora
│   ├── from_official
│   ├── input_data
│   │   ├── aurora_input.nc
│   ├── output_data
│   ├── plot
│   │   ├── coast.csv
│   │   ├── plot850.py
│   ├── supporting_module
│   │   ├── inference_helper.py
│   ├── weight
│   │   ├── aurora-0.25-finetuned.ckpt
│   │   ├── static.nc
│   ├── download_history_ncep.py
│   ├── download_ncep.py
│   ├── main.py
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

此步驟會下載一個檔案至*/input_data/aurora_input.nc。
Aurora將此檔作為初始場，此檔案須570M，請確保有足夠儲存空間。

開始跑Aurora模式。
(或許有調整到cpu核心，supporting_module/inference_helper.py 第6行 cpu_num = 6)

模式跑10天積分。請改成一下指令。
```
積分時間預設10天預報(40步階)
python main.py --input_data input_data/aurora_input.nc --output_folder output_data

更改積分時間(ex:3天預報時間)
python main.py --input_data input_data/aurora_input.nc --output_folder output_data --fore_hr 72
```

在82及我設定的環境，每10分鐘可積分6小時。執行後，output_data/底下會出現許多檔案，
aurora_output_6hr.nc
aurora_output_12hr.nc
aurora_output_18hr.nc .......


_後的數字代表積分時間。
每個積分會產生290M的檔案，請確保有足夠的儲存空間。

結果為nc檔可自行使用程式直接查看相對應的各變數

output資料細節介紹：
在aurora_output_X.nc
地表層場含有t2m, mslp, u10, v10
等壓層有13層:[50,100,200,250,300,400,500,600,700,850,925,1000]百帕，皆有T, Z, U, V, Q。

關於output檔案的使用，以850mb繪圖為例，執行以下程式碼可看到850風速與雨量結果。
```
可查看plot_code/plot850.py的檔案
```

此命令會在plot_save/目錄底下生成圖片。

離開專案，退出虛擬環境
```
deactivate
```

