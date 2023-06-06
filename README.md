

# yolov7とMiDaSを用いた相対距離推定
利用するにはyolov7とMIDaSの重み（weight）ファイルをweightsフォルダにダウンロードしておく必要があります。<br>

重み（weight）ファイルのダウンロードには以下を参照してください。<br>
　[yolov7](https://github.com/WongKinYiu/yolov7)<br>
　[MIDaS](https://github.com/isl-org/MiDaS)<br>
### 実行方法
　yolov7とMIDaSを用いた相対距離推定には `detect.py` を直接編集後に実行します。<br>
　※コマンドラインから実行することはできますが、コマンドライン引数には対応していないことに注意してください。<br>
### 引数説明

引数名|type|説明
|:---:|:---:|:---:|
|source|str|入力画像のパス
|save_path|str|出力画像の保存先フォルダパス
|iou_thres|float|最大値 = 1.0
|conf_thres|float|最大値 = 1.0
|classes|list|相対距離検出対象の設定|検出できるすべてを対象とする場合は`classes = all`を設定


**開発途中のため随時更新します。**
## 実行例１
### 入力画像
![input_photo](https://tk-2025.oops.jp/git/yolov7_and_midas/resize/photo_2.jpg)
### 内部処理
![detect](https://tk-2025.oops.jp/git/yolov7_and_midas/resize/photo_2midas.jpg)
### 出力画像
![output_photo](https://tk-2025.oops.jp/git/yolov7_and_midas/resize/photo_22.jpg)

