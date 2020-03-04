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


def get_dataloader(_mean_std_path, _data_root):
    with open(_mean_std_path, 'r') as file:
        mean_std_dict = json.load(file)
    mean = mean_std_dict['mean']
    std = mean_std_dict['std']
    # print(mean, std)

    transfms = t.Compose([
        t.ToTensor(),
        t.Normalize((mean,), (std,)),
    ])

    # 创建 dataset
    train_dataset = MnistDataset(_data_root, transforms=transfms, stage='train')
    valid_dataset = MnistDataset(_data_root, transforms=transfms, stage='valid')
    test_dataset = MnistDataset(_data_root, transforms=transfms, stage='test')

    # print(train_dataset[0])

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=50, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, num_workers=0, batch_size=1)
    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=1)

    return {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}


if __name__ == '__main__':
    mean_std_path = '../data/mean_std.json'
    data_root = '../data/'
    dataloader = get_dataloader(mean_std_path, data_root)

    # for index, (data, label, rot_labels) in enumerate(dataloader['train']):
    #     print(data.shape, label, rot_labels)
    #     if index == 2:
    #         break
    torch.set_printoptions(precision=2, threshold=100000, linewidth=100000)
    for index, (data, label) in enumerate(dataloader['valid']):
        data = data.squeeze(1)
        data_rot3 = torch.rot90(data, 3, [1, 2])
        # print(data.shape, data_rot3.shape)
        print(label)
        # if index < 1:
        #     break
