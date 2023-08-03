# gaze_detection
timm_pretrained.py: 転移学習のコード（resnet18dをロードして学習させています）
newdataset_CNN_Ghist.py: 畳み込み層で学習させたコード

## そのほかで試したReNetの結果
### 1:非注視、0:注視
学習データ

注視：7734　
非注視：15035

**ResNet34, GrayScale, epoch5, batch_size = 256**

本当は適用的ヒストグラム平坦化したいけど、Pytorchのオーグメンテーションでこれも一気にする方法がわからないのでとりあえずグレースケールで回してみる
![image](https://github.com/gjdklgjajgj/gaze_detection/assets/102703898/d385f3f9-38af-42b6-b4cd-0ebbe866ee22)




**ResNet34, GrayScale, epoch15, batch_size = 256**
![image](https://github.com/gjdklgjajgj/gaze_detection/assets/102703898/db415205-20ee-475c-8ee3-a65200316a59)




**ResNet34, GrayScale, epoch30, batch_size = 256**
![image](https://github.com/gjdklgjajgj/gaze_detection/assets/102703898/48d467f2-9263-4088-b00d-1e3456d468f6)

→ 損失が途中で収束したため、epoch50でも結果は変わらず
0.8065




**ResNet34, RGB, epoch50, batch_size = 256**
![image](https://github.com/gjdklgjajgj/gaze_detection/assets/102703898/1944d677-144e-4351-a6c0-8b4306da7f05)
![image](https://github.com/gjdklgjajgj/gaze_detection/assets/102703898/5095dbbc-7abd-49f6-8d1e-7b7584e4c25f)

0.781



**ResNet18d, RGB, epoch50, batch_size = 256**
![image](https://github.com/gjdklgjajgj/gaze_detection/assets/102703898/a14c8d71-0bf3-478f-9183-8e2d8bc0d988)
0.822



**ResNext, GrayScale, epoch50, batch_size = 256**
![image](https://github.com/gjdklgjajgj/gaze_detection/assets/102703898/de08cd0b-8d04-472e-9a09-63d8af59104a)
0.845



**ResNet50, GrayScale, epoch50, batch_size = 256**
![image](https://github.com/gjdklgjajgj/gaze_detection/assets/102703898/db12729c-b04f-404b-b07e-96d6172475f9)



## CNNでの結果
### 1:注視、0:非注視
学習データ

注視：7734　
非注視：7734
**３層CNN, GrayScale + 適用的ヒストグラム(8, 8), epoch50, batch_size = 256**

<img width="376" alt="Pasted Graphic 18" src="https://github.com/gjdklgjajgj/gaze_detection/assets/102703898/46e3d9d4-dc44-4cd4-bb8d-e1ce29036ff0">


