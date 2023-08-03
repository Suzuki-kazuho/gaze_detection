

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from IPython import display
from PIL import Image
import cv2
import glob
import re
import os
import random
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
import seaborn as sns




img_mirror_list = []
img_nmirror_list = []
#教師データ
label_list = []
#ファイルの名前を入れる（シャッフルした時にちゃんと連動しているかを確認）
img_name = []




#学習データディレクトリの中から、注視画像を取得しimg_mirror_listに入れる
for dir in os.listdir('new_TrainDataset/'):
    dir1 = 'new_TrainDataset/' + dir
    if dir == 'mirror':
        for file in os.listdir(dir1):
            filepath = dir1 + '/' + file
            image = cv2.imread(filepath)
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.equalizeHist(image)
                clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
                image = clahe.apply(image)
            except:
                print(filepath)
            image = cv2.resize(image, dsize=(64,64))
            img_mirror_list.append(image / 255.)
            label_list.append(1.0)
            img_name.append(filepath)



#学習データディレクトリの中から、非注視画像を取得しimg_nmirror_listに入れる
#非注視画像の枚数は注視画像と同じ数にする
count = 0
for dir in os.listdir('new_TrainDataset/'):
    dir1 = 'new_TrainDataset/' + dir
    if dir == 'nmirror':
        for file in os.listdir(dir1):
            filepath = dir1 + '/' + file
            image = cv2.imread(filepath)
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.equalizeHist(image)
                clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
                image = clahe.apply(image)
            except:
                print(filepath)
                continue
            image = cv2.resize(image, dsize=(64,64))
            img_nmirror_list.append(image / 255.)
            label_list.append(0.0)
            img_name.append(filepath)
            count = count + 1
            if count == len(img_mirror_list):
                break




#リストに入れた注視画像と非注視画像を一つにする
img = img_mirror_list + img_nmirror_list



#グレースケール化する
#imgとlabel/listをNumpy化
img = np.array(img)
label_list = np.array(label_list)
len_all_img = len(img_mirror_list) + len(img_nmirror_list)
#グレースケールで画像を読み込んだ場合はreshapeしなければならない(カラーで解析したいときはいらない)
img = img.reshape(len_all_img, 64, 64, 1)


#学習データをシャッフルする
#label_listとimgを関連づける
zip_lists = list(zip(label_list, img, img_name))
random.seed(0)
random.shuffle(zip_lists)
label_list, img, img_name = zip(*zip_lists)



#画像データと教師データをtrain:test:validation = 6:6:2に分割
img = np.array(img)
label_list = np.array(label_list)

indices = [int(img.shape[0] * n) for n in [0.2, 0.2 + 0.6]]
x_val,x_train, x_test = np.split(img, indices)

#label_listはlist型なのでnp型にする
label_list = np.array(label_list)
indices = [int(label_list.size * n) for n in [0.2, 0.2 + 0.6]]
y_val, y_train, y_test = np.split(label_list, indices)



#モデル構築
model = Sequential()
model.add(Conv2D(32, 15, padding='same', activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 15, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, 15, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
# model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[17]:


model.compile(loss='binary_crossentropy',
#             optimizer=RMSprop(),
            optimizer=SGD(lr=0.001),
            metrics=['accuracy'])


history = model.fit(x_train, y_train,epochs=50,batch_size=256,validation_data=(x_val,y_val))
#モデルを保存する
model.save('newv2_Train_same_Gclahe8_epoch50_batch256.h5')
