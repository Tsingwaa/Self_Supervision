"""
* @Author: Tsingwaa Tsang
* @Date: 2020-02-06 15:09:19
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-11 00:46:33
* @Description: Null
"""


import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet18
from copy import deepcopy


my_resnet = deepcopy(resnet18(True))


class Net(nn.Module):
    def __init__(self, model, num_classes, aug_classes):

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

    def foward(self, x, label):
        """
        @Description: 

        @Param x: the input image
        @Param label: the label of the input image

        @Return: 
        """

        fc_dict = {
            1: self.fc1, 2: self.fc2, 3: self.fc3, 4: self.fc4,
            5: self.fc5, 6: self.fc6, 7: self.fc7, 8: self.fc8
        }
        output_x = self.resnet_layer(x)
        output_x = fc_dict[label](output_x)

        return output_x
