"""
@Author: Tsingwaa Tsang
@Date: 2020-02-06 15:09:19
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-16 21:23:17
@Description: Null
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from base import BaseDataLoader
from my_dataset import MnistDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


trfms = T.Compose([
    T.Resize(224, 224),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
root = './data'
# 创建 dataset
train_dataset = MnistDataset(root, transforms=trfms, stage='train')
test_dataset = MnistDataset(root, transforms=trfms, stage='test')

train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=16)
test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)
