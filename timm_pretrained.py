#!/usr/bin/env python
# coding: utf-8

# # timmを使用した転移学習モデル

# ### 参考URL→ https://zenn.dev/piment/articles/4ff3b6dfd73103



import timm
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt



#in_chans=1でグレイスケール
model = timm.create_model('resnet18d',pretrained=True,num_classes = 2, in_chans = 1)
# model = timm.create_model('resnet18d', pretrained=True, num_classes = 2)


from torchvision import datasets
from torch.utils.data import DataLoader


# 学習データのパス
data_path = 'new_TrainDataset'

# バッチサイズ
batch_size = 256

# オーグメンテーション
#グレイスケール化、resize, 左右反転、正規化
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
#     """Convert a color image to grayscale and normalize the color range to [0,1]."""
])

# データセットの作成
dataset = datasets.ImageFolder(data_path, transform)


# 学習データに使用する割合
n_train_ratio = 70

# 割合から個数を出す
n_train = int(len(dataset) * n_train_ratio / 100)
n_val   = int(len(dataset) - n_train)


# 学習データと検証データに分割
train, val = torch.utils.data.random_split(dataset, [n_train, n_val])

#
# Data Loader
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)



from timm.utils import AverageMeter
from tqdm import tqdm

# 最適化手法
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

# 損失関数
criterion = torch.nn.CrossEntropyLoss()

# ログ記録用の変数
history = {"train": [], "test": []}

# 学習回数
for epoch in range(10):
    print("\nEpoch:", epoch)

    # 学習
    model.train()
    train_loss = AverageMeter()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        image = batch[0] #(batch_size, channel, size, size)
        label = batch[1] #(batch_size)
#         preds = model(image) #(batch_size, num_class)
        preds = model(image)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        train_loss.update(val = loss.item(), n = len(image))

    # 検証
    model.eval()
    test_loss = AverageMeter()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            image = batch[0] #(batch_size, channel, size, size)
            label = batch[1] #(batch_size)
            preds = model(image) #(batch_size, num_class)
            loss = criterion(preds, label)
            test_loss.update(val = loss.item(), n = len(image))

    # 誤差出力
    print(train_loss.avg)
    print(test_loss.avg)
    history["train"].append(train_loss.avg)
    history["test"].append(test_loss.avg)

#モデルの保存
torch.save(model.state_dict(), 'model_resnet18d_NoFrop_3_new_TrainDataset.pth')
