

# yolov7とMiDaSを使った、単眼カメラの相対距離推定

### 目次

- [概要](#概要)
- [環境構築](#環境構築)
- [引数説明](#引数説明)
- [実行例](#実行例)
- [検出対象リスト](#検出対象リスト)

<a id="概要"></a>

### 概要

- YOLOv7とMiDasを使用して単眼カメラの写真から対象の相対深度を出力するプログラムです

- **注意**

    - 利用するにはyolov7とMIDaSの重み（weight）ファイルをweightsフォルダにダウンロードしておく必要があります。
    
        - [yolov7](https://github.com/WongKinYiu/yolov7)（yolov7.pt ファイルをダウンロードしてください）
        - [MIDaS](https://github.com/isl-org/MiDaS)（dpt_beit_large_512.pt ファイルをダウンロードしてください）

    - 本プログラムはGPUの利用を想定していません

<a id="環境構築"></a>

### 環境構築

1, このリポジトリをクローンします。

```bash 
git clone https://github.com/Shimamotogit/Depth_Estimation.git
```

2, Anaconda Prompt で仮想環境を作成する。

```bash
conda create -n Depth_Estimation python=3.10.8
```

3, ライブラリをインストールします。
```bash 
pip install -r ./Depth_Estimation/requirements.txt
```

4, 実行方法は以下の通りです。
（サンプル画像の処理結果が`save_data`フォルダに保存されます。）

```bash 
python ./Depth_Estimation/detect.py
```

初期値以外のパラメータを設定する場合は以下を参考にしてください。

```bash 
python ./Depth_Estimation/detect.py --img_path "./input_photo/photo_1.jpg" --save_path "./save_data" --iou_thres 0.45 --conf_thres 0.3 --classes 0 2 （人と車を検出対象とする場合）
```

[引数説明](#引数説明)

### 引数説明

引数名|type|説明
|:-:|:-:|:-:|
|--img_path|str|入力画像のパス
|--save_path|str|画像を保存するフォルダのパス
|--iou_thres|float|物体検出の際の Intersection over Union (IoU) 閾値
|--conf_thres|float|物体検出の信頼度の閾値
|--classes|list|検出対象の設定<br>（対象と対になる番号は[検出対象リスト](#検出対象リスト)を参照してください）

[実行例](#実行例)

## 実行例
### 1, 画像の入力

以下の画像が入力されます。

![input_photo](https://tk-2025.oops.jp/git/yolov7_and_midas/resize/photo_2.jpg)

### 内部処理

YOLOとMiDasを実行します。<br>
この時、YOLOで検出された対象の座標情報とMiDasが出力した深度マップ情報から対象の深度値を推定します。<br>
この時のイメージ写真が以下です。

![detect](https://tk-2025.oops.jp/git/yolov7_and_midas/resize/photo_2midas.jpg)

### 出力画像

深度値の浅い順かつ検出対象に設定されているものから順に番号を割り当て、画像に印字します。<br>
この時のイメージ写真が以下です。

![output_photo](https://tk-2025.oops.jp/git/yolov7_and_midas/resize/photo_22.jpg)

<a id="検出対象リスト"></a>

<a id="検出対象リスト"></a>

## 検出対象リスト

| 検出対象名 | 番号 |
| :-: | :-: |
| 人物 | 0
| 自転車 | 1
| 車 | 2
| バイク | 3
| 飛行機 | 4
| バス | 5
| 電車 | 6
| トラック | 7
| ボート | 8
| 信号機 | 9
| 消火栓 | 10
| 停止標識 | 11
| パーキングメーター | 12
| ベンチ | 13
| 鳥 | 14
| 猫 | 15
| 犬 | 16
| 馬 | 17
| 羊 | 18
| 牛 | 19
| 象 | 20
| クマ | 21
| シマウマ | 22
| キリン | 23
| バックパック | 24
| 傘 | 25
| ハンドバッグ | 26
| ネクタイ | 27
| スーツケース | 28
| フリスビー | 29
| スキー | 30
| スノーボード | 31
| スポーツボール | 32
| 凧 | 33
| 野球のバット | 34
| 野球のグローブ | 35
| スケートボード | 36
| サーフボード | 37
| テニスラケット | 38
| ボトル | 39
| ワイングラス | 40
| カップ | 41
| フォーク | 42
| ナイフ | 43
| スプーン | 44
| ボウル | 45
| バナナ | 46
| リンゴ | 47
| サンドイッチ | 48
| オレンジ | 49
| ブロッコリー | 50
| にんじん | 51
| ホットドッグ | 52
| ピザ | 53
| ドーナツ | 54
| ケーキ | 55
| 椅子 | 56
| ソファ | 57
| 鉢植え | 58
| ベッド | 59
| ダイニングテーブル | 60
| トイレ | 61
| テレビモニター | 62
| ノートパソコン | 63
| マウス | 64
| リモコン | 65
| キーボード | 66
| 携帯電話 | 67
| 電子レンジ | 68
| オーブン | 69
| トースター | 70
| 流し台 | 71
| 冷蔵庫 | 72
| 本 | 73
| 時計 | 74
| 花瓶 | 75
| はさみ | 76
| テディベア | 77
| ヘアドライヤー | 78
| 歯ブラシ | 79