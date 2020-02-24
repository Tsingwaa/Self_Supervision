"""
@Author: Tsingwaa Tsang
@Date: 2020-02-06 15:09:19
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-16 21:23:17
@Description: Null
"""

from torch.utils.data import DataLoader
from torchvision import transforms as t

from data_loader.my_dataset import MnistDataset

trfms = t.Compose([
    t.ToTensor(),
    t.Normalize((0.1307,), (0.3081,)),
])
root = './data/'
# 创建 dataset
train_dataset = MnistDataset(root, transforms=trfms, stage='train')
test_dataset = MnistDataset(root, transforms=trfms, stage='test')

train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=16)
test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)
