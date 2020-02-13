"""
@Author: Tsingwaa Tsang
@Date: 2020-02-06 15:09:19
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-13 22:39:36
@Description: Null
"""

import torch
from torch import nn
from torchvision.models import resnet18
from copy import deepcopy


class Net(nn.Module):
    def __init__(self, model, num_classes, aug_classes):
        """
        @Description: this is to modify the input model by replace the last layer with 8 parallel layers.

        @Param model: The input model
        @Param num_classes: the classes of original images
        @Param aug_classes: the extra classes of augmented images
        """

        super(Net, self).__init__()
        # 先去除最后一层fc层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        # 再定义四类的fc层
        self.fc1 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc2 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc3 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc4 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc5 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc6 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc7 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc8 = nn.Linear(model.fc.in_features, aug_classes)
        self.softmax = nn.Softmax(dim=0)

    def foward(self, x, label, stage, concate):
        """
        @Description:

        @Param x: the input image
        @Param label: the label of the input image
        @Param stage: train or test
        @Param concate: bool symbool. concate the 8 output or not.

        @Return: the output probability
        """

        fc_dict = {
            1: self.fc1, 2: self.fc2, 3: self.fc3, 4: self.fc4,
            5: self.fc5, 6: self.fc6, 7: self.fc7, 8: self.fc8
        }
        output_x = self.resnet_layer(x)
        if stage == "train":
            output_x = fc_dict[label](output_x)
            output_x = self.softmax(output_x)
        elif stage == "test":
            output_x1 = self.fc1(output_x)
            output_x2 = self.fc2(output_x)
            output_x3 = self.fc3(output_x)
            output_x4 = self.fc4(output_x)
            output_x5 = self.fc5(output_x)
            output_x6 = self.fc6(output_x)
            output_x7 = self.fc7(output_x)
            output_x8 = self.fc8(output_x)
            if concate == True:
                output_x = torch.cat([
                    output_x1, output_x2, output_x3, output_x4,
                    output_x5, output_x6, output_x7, output_x8
                ], 0)  # 横向拼接为一行
                output_x = self.softmax(output_x)
                # 此时输出为一行概率
            else:
                output_x = torch.cat([
                    output_x1, output_x2, output_x3, output_x4,
                    output_x5, output_x6, output_x7, output_x8
                ], 1)  # 纵向拼接为一个矩阵，每一行为一个通道输出
                output_x = self.softmax(output_x)  # 对每一行做softmax
                # 此时输出为二维概率矩阵

        return output_x


copy_resnet18 = deepcopy(resnet18(True))
my_resnet18 = Net(model=copy_resnet18, num_classes=8, aug_classes=4)
print(my_resnet18)
