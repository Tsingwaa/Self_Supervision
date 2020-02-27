"""
@Author: Tsingwaa Tsang
@Date: 2020-02-06 15:09:19
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-16 21:23:17
@Description: Null
"""
import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as t

from data_loader.my_dataset import MnistDataset

mean_std_path = r'../data/mean_std.json'
with open(mean_std_path, 'r') as file:
    mean_std_dict = json.load(file)
mean = mean_std_dict['mean']
std = mean_std_dict['std']
# print(mean, std)

transfms = t.Compose([
    t.ToTensor(),
    t.Normalize((mean,), (std,)),
])

root = r'../data/'
# 创建 dataset
train_dataset = MnistDataset(root, transforms=transfms, stage='train')
valid_dataset = MnistDataset(root, transforms=transfms, stage='valid')
test_dataset = MnistDataset(root, transforms=transfms, stage='test')

# print(train_dataset[0])


train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=5, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, num_workers=0, batch_size=5)
test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=5)

for index, (data, label, rot_labels) in enumerate(train_dataloader):
    print(data.shape, label, rot_labels)
    if index == 1:
        break
