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

transfms = t.Compose([
    t.ToTensor(),
    t.Normalize((0.1307,), (0.3081,)),
])

root = r'..\data\\'
# 创建 dataset
train_dataset = MnistDataset(root, transforms=transfms, stage='train')
valid_dataset = MnistDataset(root, transforms=transfms, stage='valid')
test_dataset = MnistDataset(root, transforms=transfms, stage='test')

train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=16)
valid_dataloader = DataLoader(valid_dataset, num_workers=0, batch_size=16)
test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=16)

# for index, label, rot_labels in enumerate(train_dataloader):
#     print(label.shape, rot_labels.shape)
#     if index == 1:
#         break
