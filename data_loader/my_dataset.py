"""
@Author: Tsingwaa Tsang
@Date: 2020-02-07 21:53:29
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-13 23:09:00
@Description: Null
"""

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T


class MyCustomDataset(Dataset):
    def __init__(self, path, transforms=None):
        # stuff
        ...
        self.transforms = transforms

    def __getitem__(self, index):
        # stuff
        ...
        # 一些读取的数据
        img = '?'
        if self.transforms is not None:
            img = self.transforms(img)
        label = 1
        # 如果 transform 不为 None，则进行 transform 操作
        return (img, label)

    def __len__(self):
        return len(self)


if __name__ == "__main__":
    # 定义我们的 transforms (1)
    transformations = T.Compose([
        T.CenterCrop(100),
        T.ToTensor()
    ])
    # 创建 dataset
    custom_dataset = MyCustomDataset(..., transformations)
