"""
@Author: Tsingwaa Tsang
@Date: 2020-02-07 21:53:29
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-16 20:33:38
@Description: Null
"""


from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from glob import glob
import numpy as np
import torch
from PIL import Image


class MnistDataset(Dataset):
    def __init__(self, root, transforms=None, stage="train"):
        # stuff
        self._root = root
        self._transforms = transforms
        self._stage = stage
        self._tr_images = np.load(glob(root + r"*_train_images.npy"))
        self._tr_labels = np.load(glob(root + r"*_train_labels.npy"))
        self._ts_images = np.load(glob(root + r"*_test_images.npy"))
        self._ts_labels = np.load(glob(root + r"*_test_labels.npy"))

    def __getitem__(self, index):
        out_image = []
        out_label = []

        if self._stage == 'train':
            out_image = self._tr_images[index]
            out_label = self._tr_labels[index]
            if self._transforms is not None:
                out_image = self._transforms(out_image)

        elif self._stage == 'test':
            img = self._ts_images[index]
            label = self._ts_labels[index]
            if self._transforms is not None:
                img = self._transforms(img)
            # 先将原图及其标签输入
            out_image.append(img)
            # 考虑最终测试时，是否以大类准确为标准，当前以旋转为具体要求
            out_label.append(label)
            # out_label.append(4*label)

            for rot_times in [1, 2, 3]:
                rot_img = np.rot90(img, k=rot_times)

                if self._transforms is not None:
                    rot_img = self._transforms(rot_img)

                out_image.append(rot_img)
                # 考虑最终测试时，是否以大类准确为标准
                out_label.append(self._tr_labels[index] * 4 + rot_times)
                # out_label.append(self._labels[index] * 4 + rot_times)

        # 如果 transform 不为 None，则进行 transform 操作
        return (out_image, out_label)

    def __len__(self):
        return len(self._tr_labels)


if __name__ == "__main__":
    # 定义我们的 transforms (1)
    Trsfms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ])
    root = './data'
    # 创建 dataset
    train_dataset = MnistDataset(root, transforms=Trsfms, stage='train')
    test_dataset = MnistDataset(root, transforms=Trsfms, stage='test')
