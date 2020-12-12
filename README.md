# face2mask
画像上の人物の顔にマスクを付与するモデル

![face2mask](https://github.com/varus1234/face2mask-2/blob/master/imgs/face2mask.jpg)

## 目的
昨今のコロナ禍を鑑みて、航空業界や鉄道業界等、『安心・安全』をモットーにしている企業の広告写真を刷新する必要があります。その際、face2maskを利用して現在の広告写真にマスクを付与することにより、新たに撮影し直すコストを削減します。

## モデルの概要
![gaiyou](https://github.com/varus1234/face2mask-2/blob/master/imgs/gaiyou.jpg)

入力画像から出力画像までの処理の流れは以下の通りです。
1. 入力画像から顔検出と切り抜き
2. 顔画像にマスクを付与
3. 入力画像へ切り抜いた部分を元に戻す

顔検出にはOpenCVの[カスケード型分類器](https://note.nkmk.me/python-opencv-face-detection-haar-cascade/)、顔画像へのマスク付与には[pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)を使用しています。

## モデルの訓練とテスト
訓練には`notebooks/train.ipynb`、テストには`notebooks/test.ipynb`を参照してください。
  
※それぞれGoogle Colab環境での実行を想定しているため、このrepositoryをご自身のGoogle Driveにcloneして実行することを推奨します。
