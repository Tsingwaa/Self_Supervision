"""
@Author: Tsingwaa Tsang
@Date: 2020-02-06 15:09:19
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-13 22:53:54
@Description: Null
"""

import torch
from torch import nn
from torchvision.models import resnet18
from copy import deepcopy


class Net1FC(nn.Module):

    def __init__(self, model, all_classes):
        """
        @Description: this is to modify the input model by replace the last layer with 8 parallel layers.

        @Param model: The input model
        @Param num_classes: the classes of original images
        @Param aug_classes: the extra classes of augmented images
        """

        super(Net1FC, self).__init__()

        # 先去除最后一层fc层
        self._conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self._res_layer = nn.Sequential(*list(model.children())[1:][:-1])

        # 再定义四类的fc层
        self._fc = nn.Linear(model.fc.in_features, all_classes)

        self._softmax = nn.Softmax(dim=0)

    def foward(self, x, label, stage='train'):
        """
        @Param x: the input image
        @Param label: the label of the origin input image
        @Param stage: train or test
        @Param concate: bool symbool. concate the 8 output or not.

        @Return: the output probability
        """
        output_x = self._res_layer(x)

        if stage == "train":
            output_x = self._fc(output_x)
            output_x = self._softmax(output_x)

        elif stage == "test":
            output_x = self._fc(output_x)
            output_x = self._softmax(output_x)  # 对每一行做softmax

        return output_x


class Net8FC(nn.Module):

    def __init__(self, model, num_classes, aug_classes):
        """
        @Description: this is to modify the input model by replace the last layer with 8 parallel layers.

        @Param model: The input model
        @Param num_classes: the classes of original images
        @Param aug_classes: the extra classes of augmented images
        """
        super(Net8FC, self).__init__()

        self._num_classes = num_classes
        self._aug_classes = aug_classes

        # 创建第一层卷积层，并去除第一层和最后一层
        self._conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self._res_layer = nn.Sequential(*list(model.children())[1:][:-1])

        # 再定义四类的fc层
        self._fc1 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc2 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc3 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc4 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc5 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc6 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc7 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc8 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._softmax = nn.Softmax(dim=0)

    def foward(self, x, label, stage, is_concate=False):
        """
        @Description:

        @Param x: the input image
        @Param label: the label of the origin input image
        @Param stage: train or test
        @Param concate: bool symbool. concate the 8 output or not.

        @Return: the output probability
        """

        fc_dict = {
            1: self._fc1, 2: self._fc2, 3: self._fc3, 4: self._fc4,
            5: self._fc5, 6: self._fc6, 7: self._fc7, 8: self._fc8
        }
        output_x = self._res_layer(x)
        if stage == "train":
            output_x = fc_dict[label](output_x)
            output_x = self._softmax(output_x)

        elif stage == "test":
            # 测试时，输出通过所有分类器
            output_x1 = self._fc1(output_x)
            output_x2 = self._fc2(output_x)
            output_x3 = self._fc3(output_x)
            output_x4 = self._fc4(output_x)
            output_x5 = self._fc5(output_x)
            output_x6 = self._fc6(output_x)
            output_x7 = self._fc7(output_x)
            output_x8 = self._fc8(output_x)
            if is_concate:
                output_x = torch.cat([
                    output_x1, output_x2, output_x3, output_x4,
                    output_x5, output_x6, output_x7, output_x8
                ], 0)  # 横向拼接为一行
                output_x = self._softmax(output_x)
                # 此时输出为一行概率
            else:
                output_x = torch.cat([
                    output_x1, output_x2, output_x3, output_x4,
                    output_x5, output_x6, output_x7, output_x8
                ], 1)  # 纵向拼接为一个矩阵，每一行为一个通道输出
                output_x = self._softmax(output_x)  # 对每一行做softmax
                # 此时输出为二维概率矩阵

        return output_x


if __name__ == "__main__":
    copy_resnet18 = deepcopy(resnet18(True))
    # my_resnet18 = Net8FC(model=copy_resnet18, num_classes=8, aug_classes=4)
    my_resnet18_1fc = Net1FC(model=copy_resnet18, all_classes=32)
    my_resnet18_8fc = Net8FC(model=copy_resnet18, num_classes=8, aug_classes=4)
    print(my_resnet18_1fc)
