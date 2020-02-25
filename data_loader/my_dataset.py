"""
@Author: Tsingwaa Tsang
@Date: 2020-02-07 21:53:29
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-16 20:33:38
@Description: Null
"""

from glob import glob
from random import randint

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as t


class MnistDataset(Dataset):
    def __init__(self, data_root, transforms=None, stage="train"):
        # stuff
        self._root = data_root
        self._transforms = transforms
        self._stage = stage

        self._tr_images = np.load(glob(data_root + r"*_train_images.npy")[0])
        self._tr_labels = np.load(glob(data_root + r"*_train_labels.npy")[0])

        self._val_images = np.load(glob(data_root + r"*_valid_images.npy")[0])
        self._val_labels = np.load(glob(data_root + r"*_valid_labels.npy")[0])

        self._ts_images = np.load(glob(data_root + r"*_test_images.npy")[0])
        self._ts_labels = np.load(glob(data_root + r"*_test_labels.npy")[0])

        self._images = {
            "train": self._tr_images,
            "valid": self._val_images,
            "test": self._ts_images
        }
        self._labels = {
            "train": self._tr_labels,
            "valid": self._val_labels,
            "test": self._ts_labels
        }

    def __getitem__(self, index):
        # 所有数据集（训练、验证、测试）都是原图加上其增强的图片，再输出

        rot_images = []

        ori_image = self._images[self._stage][index]
        label = self._labels[self._stage][index]

        rot_label = randint(0, 3)
        rot_image = np.rot90(ori_image, k=rot_label)

        # if self._transforms is not None:
        #     ori_image = self._transforms(ori_image)
        #
        # rot_images.append(ori_image)
        #
        # for rot_times in [1, 2, 3]:
        #     rot_img = np.rot90(ori_image, k=rot_times)
        #
        #     if self._transforms is not None:
        #         rot_img = self._transforms(rot_img)
        #
        #     rot_images.append(rot_img)
        #
        # rot_labels = [0, 1, 2, 3]
        #
        # rot_images = torch.stack(rot_images)  # 将四个图片的张量拼接

        # if self._stage == 'train':
        #     out_image = self._tr_images[index]
        #     out_label = self._tr_labels[index]
        #     if self._transforms is not None:
        #         out_image = self._transforms(out_image)
        #
        # elif self._stage == 'test':
        #     img = self._ts_images[index]
        #     label = self._ts_labels[index]
        #     if self._transforms is not None:
        #         img = self._transforms(img)
        #     # 先将原图及其标签输入
        #     out_image.append(img)
        #     # 考虑最终测试时，是否以大类准确为标准，当前以旋转为具体要求
        #     out_label.append(label)
        #     # out_label.append(4*label)
        #
        #     for rot_times in [1, 2, 3]:
        #         rot_img = np.rot90(img, k=rot_times)
        #
        #         if self._transforms is not None:
        #             rot_img = self._transforms(rot_img)
        #
        #         out_image.append(rot_img)
        #         # 考虑最终测试时，是否以大类准确为标准
        #         out_label.append(self._tr_labels[index] * 4 + rot_times)
        #         # out_label.append(self._labels[index] * 4 + rot_times)

        # 如果 transform 不为 None，则进行 transform 操作

        return rot_image, label, rot_label

    def __len__(self):
        return len(self._labels)


if __name__ == "__main__":
    pass
    # # 定义我们的 transforms (1)
    # transfms = t.Compose([
    #     t.ToTensor(),
    #     t.Normalize((0.1307,), (0.3081,)),
    # ])
    # root = r'..\data\\'
    # # 创建 dataset
    # train_dataset = MnistDataset(root, transforms=transfms, stage='train')
    # valid_dataset = MnistDataset(root, transforms=transfms, stage='valid')
    # test_dataset = MnistDataset(root, transforms=transfms, stage='test')
