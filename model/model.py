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

        # 再定义八类的fc层
        self._fc0 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc1 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc2 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc3 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc4 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc5 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc6 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._fc7 = nn.Linear(model.fc.in_features, self._aug_classes)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x, label, stage, is_concate=False):
        """
        @Description:

        @Param x: the input image
        @Param label: the label of the origin input image
        @Param stage: train or test
        @Param concate: bool symbool. concate the 8 output or not.

        @Return: the output probability
        """

        # fc_dict = {
        #     0: self._fc0, 1: self._fc1, 2: self._fc2, 3: self._fc3,
        #     4: self._fc4, 5: self._fc5, 6: self._fc6, 7: self._fc7
        # }

        x = self._conv1(x)
        # print(x.shape)
        if stage == 'train':
            x = self._res_layer(x).view(50, 512)
        else:
            x = self._res_layer(x).view(4, 512)
        # print(x.shape)

        x0 = self._fc0(x)
        x1 = self._fc1(x)
        x2 = self._fc2(x)
        x3 = self._fc3(x)
        x4 = self._fc4(x)
        x5 = self._fc5(x)
        x6 = self._fc6(x)
        x7 = self._fc7(x)

        if is_concate:
            output = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], 1)  # 横向拼接为一行
            output = self._softmax(output)
            # 此时输出为一行概率
        else:
            x0 = self._softmax(x0)
            x1 = self._softmax(x1)
            x2 = self._softmax(x2)
            x3 = self._softmax(x3)
            x4 = self._softmax(x4)
            x5 = self._softmax(x5)
            x6 = self._softmax(x6)
            x7 = self._softmax(x7)
            # Crossentropy前面无需加softmax层
            output = [x0, x1, x2, x3, x4, x5, x6, x7]

        # print(output)

        return output


if __name__ == "__main__":
    copy_resnet18 = deepcopy(resnet18(True))
    # my_resnet18 = Net8FC(model=copy_resnet18, num_classes=8, aug_classes=4)
    my_resnet18_1fc = Net1FC(model=copy_resnet18, all_classes=32)
    my_resnet18_8fc = Net8FC(model=copy_resnet18, num_classes=8, aug_classes=5)
    print(my_resnet18_8fc)
